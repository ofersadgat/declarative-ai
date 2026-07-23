/**
 * Bounded fan-out + provider rate limiting, extracted verbatim from findmyprompt
 * `src/engine/execution/concurrency.ts` (the interface types live in ./contract).
 *
 * Three cooperating pieces:
 *   - `ConcurrencyLimiter` — a counting semaphore with a MUTABLE limit (the AIMD knob).
 *   - `TokenBucket` — TPM pre-admission over `estInput + estOutput` (Anthropic-style;
 *     OpenRouter has no TPM, so it runs with concurrency only).
 *   - `AdaptiveRateController` — wires them together and runs **AIMD** on the concurrency
 *     knob: halve on a 429, additively increase on sustained success. A schedule holds its
 *     concurrency slot across the token-bucket wait, so the wait is itself the throttle.
 *
 * Pure timing seams (`now`, `wait`) are injectable so the bucket/AIMD are deterministically
 * testable.
 */
import type { CallEstimate, CallTokenEstimate, RateLimiter } from "./contract";

export type { CallEstimate, CallTokenEstimate, RateLimiter };

/** A model's per-minute rate limits — the shape providers actually publish. Any subset may be present. */
export interface ModelRateLimits {
  rpm?: number;
  inputTpm?: number;
  outputTpm?: number;
}

/** Resolve a model id to its rate-limit BUCKET GROUP: a stable `key` (models sharing one published
 *  limit share one set of buckets) + the limits. Undefined ⇒ the model has no per-model buckets. */
export type ModelLimitResolver = (modelId: string) => { key: string; limits: ModelRateLimits } | undefined;

/** LONGEST-PREFIX resolver over a `{ modelPrefix → limits }` map:
 *  `claude-sonnet-4` matches `claude-sonnet-4-6`; the matched prefix is the bucket-group key. */
export function prefixModelLimitResolver(map: Record<string, ModelRateLimits>): ModelLimitResolver {
  const prefixes = Object.keys(map).sort((a, b) => b.length - a.length); // longest first
  return (modelId) => {
    for (const p of prefixes) if (modelId.startsWith(p)) return { key: p, limits: map[p]! };
    return undefined;
  };
}

/** Disabled-provider limiter: no pool, no buckets — `schedule` runs the call immediately. Keeps the
 *  `RateLimiter` seam OCCUPIED so decorators and "is a limiter present" checks compose unchanged. */
export class PassthroughRateLimiter implements RateLimiter {
  async schedule<T>(_est: CallEstimate, run: () => Promise<T>): Promise<T> {
    return run();
  }
  reportOutcome(): void {}
}

/**
 * Provider+model-specific rate limiting: routes EACH CALL to its own serving provider's limiter
 * (by the call's model id), so a mixed-provider run throttles every call against the right quota.
 * `pick` maps a model id to the provider limiter; outcomes route the same way, so AIMD feedback
 * credits the limiter that scheduled the call.
 */
export class ProviderDispatchRateLimiter implements RateLimiter {
  constructor(private readonly pick: (modelId: string | undefined) => RateLimiter) {}
  schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T> {
    return this.pick(est.modelId).schedule(est, run);
  }
  reportOutcome(outcome: { rateLimited?: boolean; modelId?: string }): void {
    this.pick(outcome.modelId).reportOutcome(outcome);
  }
}

type Wait = (ms: number) => Promise<void>;
const realWait: Wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** A counting semaphore whose limit can change at runtime (raising it immediately admits waiters). */
export class ConcurrencyLimiter {
  private limit: number;
  private active = 0;
  private readonly queue: Array<() => void> = [];

  constructor(limit: number) {
    this.limit = Math.max(1, Math.floor(limit));
  }

  get currentLimit(): number {
    return this.limit;
  }
  get activeCount(): number {
    return this.active;
  }
  get queuedCount(): number {
    return this.queue.length;
  }

  setLimit(n: number): void {
    this.limit = Math.max(1, Math.floor(n));
    while (this.queue.length > 0 && this.active < this.limit) {
      this.active++;
      this.queue.shift()!();
    }
  }

  private acquire(): Promise<void> {
    if (this.active < this.limit) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => this.queue.push(resolve));
  }

  private release(): void {
    this.active--;
    if (this.queue.length > 0 && this.active < this.limit) {
      this.active++;
      this.queue.shift()!();
    }
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

/** Continuous-refill token bucket for TPM pre-admission. Capacity ≈ the per-minute limit. */
export class TokenBucket {
  private tokens: number;
  private last: number;

  constructor(
    private readonly ratePerMs: number,
    private readonly capacity: number,
    private readonly now: () => number = () => Date.now(),
    private readonly wait: Wait = realWait,
  ) {
    this.tokens = capacity;
    this.last = now();
  }

  get available(): number {
    this.refill();
    return this.tokens;
  }

  private refill(): void {
    const t = this.now();
    this.tokens = Math.min(this.capacity, this.tokens + (t - this.last) * this.ratePerMs);
    this.last = t;
  }

  /** Block until `n` tokens are available, then remove them. `n` is clamped to capacity. */
  async remove(n: number): Promise<void> {
    const need = Math.min(Math.max(0, n), this.capacity);
    for (;;) {
      this.refill();
      if (this.tokens >= need) {
        this.tokens -= need;
        return;
      }
      const deficit = need - this.tokens;
      await this.wait(Math.ceil(deficit / this.ratePerMs));
    }
  }
}

export interface AdaptiveRateControllerOptions {
  /** Starting / max / min concurrency for the AIMD knob. */
  initialConcurrency?: number;
  maxConcurrency?: number;
  minConcurrency?: number;
  /** Additively raise concurrency by 1 after this many consecutive successes. Default 8. */
  increaseEvery?: number;
  /** LEGACY combined TPM cap — one shared input+output bucket (test seam / explicit override).
   *  Prefer `modelLimits`, which carries the split per-model shape providers actually publish. */
  tokensPerMinute?: number;
  /** Per-model buckets: model id → its rate-limit family (RPM/ITPM/OTPM). Buckets are created lazily
   *  per family key, so models sharing one published limit share one set of buckets. Omit → pool only. */
  modelLimits?: ModelLimitResolver;
  now?: () => number;
  wait?: Wait;
}

/**
 * The v1 rate controller: a concurrency pool + optional TPM bucket, with **AIMD** on the
 * concurrency knob — multiplicative decrease (halve) on a 429 / rate-limit signal, additive
 * increase (+1) on sustained success. Header backoff (`retry-after`) is the caller's retry
 * loop's job; this is the orthogonal "how fast to go next time" control.
 */
export class AdaptiveRateController implements RateLimiter {
  private readonly limiter: ConcurrencyLimiter;
  private readonly bucket?: TokenBucket;
  private readonly modelLimits?: ModelLimitResolver;
  /** Lazily-created per-family bucket sets — one per distinct `ModelLimitResolver` key. */
  private readonly familyBuckets = new Map<string, { req?: TokenBucket; input?: TokenBucket; output?: TokenBucket }>();
  private readonly maxConcurrency: number;
  private readonly minConcurrency: number;
  private readonly increaseEvery: number;
  private readonly now?: () => number;
  private readonly waitFn?: Wait;
  private successStreak = 0;

  constructor(opts: AdaptiveRateControllerOptions = {}) {
    this.minConcurrency = Math.max(1, opts.minConcurrency ?? 1);
    this.maxConcurrency = Math.max(this.minConcurrency, opts.maxConcurrency ?? 16);
    const initial = Math.min(this.maxConcurrency, Math.max(this.minConcurrency, opts.initialConcurrency ?? 4));
    this.limiter = new ConcurrencyLimiter(initial);
    this.increaseEvery = Math.max(1, opts.increaseEvery ?? 8);
    this.now = opts.now;
    this.waitFn = opts.wait;
    this.modelLimits = opts.modelLimits;
    if (opts.tokensPerMinute && opts.tokensPerMinute > 0) {
      this.bucket = new TokenBucket(opts.tokensPerMinute / 60_000, opts.tokensPerMinute, opts.now, opts.wait);
    }
  }

  get concurrency(): number {
    return this.limiter.currentLimit;
  }
  get activeCount(): number {
    return this.limiter.activeCount;
  }

  private bucketsFor(modelId: string | undefined): { req?: TokenBucket; input?: TokenBucket; output?: TokenBucket } | undefined {
    if (!modelId || !this.modelLimits) return undefined;
    const fam = this.modelLimits(modelId);
    if (!fam) return undefined;
    let b = this.familyBuckets.get(fam.key);
    if (!b) {
      const mk = (perMinute: number | undefined): TokenBucket | undefined =>
        perMinute && perMinute > 0 ? new TokenBucket(perMinute / 60_000, perMinute, this.now, this.waitFn) : undefined;
      b = { req: mk(fam.limits.rpm), input: mk(fam.limits.inputTpm), output: mk(fam.limits.outputTpm) };
      this.familyBuckets.set(fam.key, b);
    }
    return b;
  }

  async schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T> {
    return this.limiter.run(async () => {
      // Slot held across every bucket wait — the wait is itself the throttle.
      if (this.bucket) await this.bucket.remove(est.inputTokens + est.outputTokens);
      const fam = this.bucketsFor(est.modelId);
      if (fam) {
        if (fam.req) await fam.req.remove(1);
        if (fam.input) await fam.input.remove(est.inputTokens);
        if (fam.output) await fam.output.remove(est.outputTokens);
      }
      return run();
    });
  }

  reportOutcome(outcome: { rateLimited?: boolean }): void {
    if (outcome.rateLimited) {
      this.successStreak = 0;
      this.limiter.setLimit(Math.max(this.minConcurrency, Math.floor(this.limiter.currentLimit / 2)));
      return;
    }
    this.successStreak++;
    if (this.successStreak >= this.increaseEvery) {
      this.successStreak = 0;
      this.limiter.setLimit(Math.min(this.maxConcurrency, this.limiter.currentLimit + 1));
    }
  }
}
