/**
 * MODEL RESIDENCY (§5) — deciding which locally-served models are loaded, and admitting calls to them.
 *
 * A remote fleet meters you by rate and by money; your own machine meters you by MEMORY, and the two
 * are nothing alike. Rate headroom regenerates on its own, so waiting is always enough. Memory does not:
 * something has to be UNLOADED before something else can load, and the thing being unloaded may be in
 * use. That is what this module arbitrates.
 *
 * Four properties are the design, and each is a thing that went wrong in a simpler version:
 *
 *  - **A lease is SHARED, not exclusive.** One resident model serves many concurrent calls; the count
 *    is what makes eviction safe, because a model is evictable exactly when nothing holds it.
 *  - **Concurrency is bought with memory.** A context fixes its sequence count at load, and each
 *    sequence carries its own KV cache — measured, a 0.5B model was 374 MB of weights and 491 MB for
 *    four 4096-token sequences. So per-model concurrency is a residency decision, not a rate one.
 *  - **Queues are PER MODEL, and a resident model drains before it is swapped.** Global FIFO with room
 *    for one model and an A,B,A,B arrival pattern reloads on every single call.
 *  - **Nothing is preempted.** A generation in flight keeps its model until it finishes. The cost is a
 *    long call delaying a swap; the alternative is killing work that has already been paid for.
 *
 * The scheduler itself never imports `node-llama-cpp`: prediction is a {@link PlacementProbe} seam, so
 * the arbitration logic is testable against invented hardware and the optional peer stays optional.
 */
import { createLogger } from "@declarative-ai/log";

const log = createLogger("engine.providers.residency");

/**
 * Where a model's working set actually lands. The three tiers differ by ORDERS OF MAGNITUDE in speed,
 * which is why they are named rather than reduced to a boolean "fits".
 */
export type PlacementTier =
  /** Entirely in VRAM. Full speed. */
  | "vram"
  /** Spilled into system RAM — partial GPU offload, or none. Works, slowly. */
  | "ram"
  /** Spilled past system RAM into swap/pagefile. Technically runs; effectively does not. */
  | "swap";

/** What a probe predicts for one model under the CURRENT machine state. */
export interface Placement {
  tier: PlacementTier;
  /** Layers that would sit on the GPU, of `totalLayers`. Below the total means partial offload. */
  gpuLayers: number;
  totalLayers: number;
  /** The context the prediction assumed — placement is a property of model AND context, not model
   *  alone: the same 32B lands in VRAM at 4k and spills at 32k. */
  contextSize: number;
  vramBytes: number;
  ramBytes: number;
}

/** Predicts where a model would land right now. `undefined` ⇒ nothing to say (a remote model, or a
 *  local one this probe cannot measure), which admits the call without a placement decision. */
export interface PlacementProbe {
  predict(modelId: string): Promise<Placement | undefined>;
}

/** What the caller decides to do about a predicted placement. */
export type PlacementDecision =
  | { action: "proceed" }
  /** Load with a reduced GPU split, accepting the slowdown knowingly. */
  | { action: "degrade"; gpuLayers: number }
  | { action: "refuse"; reason: string };

/**
 * The caller's policy on degraded placement.
 *
 * A seam rather than a setting because the answer is a product decision, not a library one: a host may
 * auto-degrade, refuse, or ask its user. It receives the model id alongside the placement so a
 * per-model answer needs no signature change later, and it may be async so a host can escalate to a
 * human instead of answering from config.
 */
export type PlacementPolicy = (request: { modelId: string; placement: Placement }) => PlacementDecision | Promise<PlacementDecision>;

/**
 * The default when no policy is supplied: full-speed placement proceeds, anything degraded is REFUSED
 * and names what it would have cost.
 *
 * Refusing rather than coping is deliberate. A model quietly running ten times slower is the kind of
 * silent degradation this codebase declines elsewhere — and unlike a wrong guess about money, this one
 * is invisible in every metric except wall-clock. A host that wants degradation says so in one line.
 */
export const REFUSE_DEGRADED: PlacementPolicy = ({ placement }) =>
  placement.tier === "vram"
    ? { action: "proceed" }
    : {
        action: "refuse",
        reason:
          placement.tier === "swap"
            ? `placement would spill to swap (${placement.gpuLayers}/${placement.totalLayers} layers on GPU at context ${placement.contextSize}) — supply a PlacementPolicy to allow it`
            : `placement would spill into system RAM (${placement.gpuLayers}/${placement.totalLayers} layers on GPU at context ${placement.contextSize}) — supply a PlacementPolicy to allow it`,
      };

/** A held claim on a resident model. Release it exactly once, on every path. */
export interface ResidencyLease {
  readonly modelId: string;
  /** What the probe predicted, when it had something to say. */
  readonly placement: Placement | undefined;
  release(): void;
}

export interface ResidencyOptions {
  probe?: PlacementProbe;
  /** Defaults to {@link REFUSE_DEGRADED}. */
  policy?: PlacementPolicy;
  /** How many models may be loaded at once. Default 1 — the honest default for one GPU. */
  maxResident?: number;
  /** Concurrent calls per resident model. Default 1; raising it costs KV cache at load time. */
  maxConcurrentPerModel?: number;
  /**
   * Consecutive grants one model may take while another model has callers waiting. Default 8.
   *
   * Without it, drain-before-swap starves: a continuously-fed model holds residency forever and every
   * other model waits indefinitely. With it, a busy model still amortizes its load across a batch.
   */
  drainLimit?: number;
  /** Unload a model's weights. Called when residency is revoked; absent ⇒ eviction is bookkeeping only. */
  unload?: (modelId: string) => Promise<void>;
}

interface Resident {
  /** Calls currently holding this model. Evictable at exactly zero. */
  leases: number;
  /** Consecutive grants taken while others waited — reset when someone else gets a turn. */
  streak: number;
  /** Monotonic counter of the last grant, for least-recently-used eviction. */
  lastUsed: number;
}

interface Waiter {
  modelId: string;
  resolve: (lease: ResidencyLease) => void;
  reject: (err: unknown) => void;
  signal?: AbortSignal | undefined;
  settled: boolean;
}

/** Raised when a placement policy declines to run a model here. */
export class PlacementRefused extends Error {
  readonly kind = "placement-refused";
  constructor(
    readonly modelId: string,
    reason: string,
    readonly placement: Placement | undefined,
  ) {
    super(`model "${modelId}" was not admitted: ${reason}`);
    this.name = "PlacementRefused";
  }
}

/**
 * The arbiter. One instance per machine's worth of memory — sharing it is the point, since two
 * independently-scheduled workflows on one GPU would otherwise each believe they had all of it.
 */
export class ResidencyManager {
  private readonly resident = new Map<string, Resident>();
  private readonly queues = new Map<string, Waiter[]>();
  private readonly placements = new Map<string, Placement | undefined>();
  private clock = 0;
  /** Serializes admission decisions: two concurrent `acquire`s for different models must not both
   *  conclude there is room. */
  private pumping = false;

  constructor(private readonly options: ResidencyOptions = {}) {}

  /** Models currently loaded, for diagnostics and tests. */
  residentModels(): string[] {
    return [...this.resident.keys()];
  }

  /**
   * Claim residency for a model, waiting until it is loaded and a slot is free.
   *
   * Rejects only for a REFUSED placement or a canceled wait — a full machine is backpressure, not an
   * error, and a caller that must wait simply waits.
   */
  acquire(modelId: string, signal?: AbortSignal): Promise<ResidencyLease> {
    if (signal?.aborted === true) return Promise.reject(new Error("canceled before residency was acquired"));
    return new Promise<ResidencyLease>((resolve, reject) => {
      const waiter: Waiter = { modelId, resolve, reject, signal, settled: false };
      const queue = this.queues.get(modelId);
      if (queue) queue.push(waiter);
      else this.queues.set(modelId, [waiter]);
      if (signal !== undefined) {
        const onAbort = (): void => {
          if (waiter.settled) return;
          waiter.settled = true;
          this.drop(waiter);
          reject(new Error("canceled while queued for model residency"));
        };
        signal.addEventListener("abort", onAbort, { once: true });
      }
      void this.pump();
    });
  }

  private drop(waiter: Waiter): void {
    const queue = this.queues.get(waiter.modelId);
    if (queue === undefined) return;
    const at = queue.indexOf(waiter);
    if (at >= 0) queue.splice(at, 1);
    if (queue.length === 0) this.queues.delete(waiter.modelId);
  }

  private grant(waiter: Waiter, entry: Resident): void {
    waiter.settled = true;
    entry.leases += 1;
    entry.lastUsed = ++this.clock;
    let released = false;
    waiter.resolve({
      modelId: waiter.modelId,
      placement: this.placements.get(waiter.modelId),
      release: () => {
        // Idempotent: a double release would let a model be evicted while a call still holds it, which
        // is the one bookkeeping error whose symptom is a crash inside someone else's generation.
        if (released) return;
        released = true;
        entry.leases -= 1;
        void this.pump();
      },
    });
  }

  /**
   * The scheduling loop. Runs to quiescence, one decision at a time.
   *
   * Serialized by `pumping` because admission reads shared state and then awaits (the probe, the
   * policy, an unload); two interleaved passes would each see room for one more model and both take it.
   */
  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      for (;;) {
        this.reap();
        if (!(await this.step())) return;
      }
    } finally {
      this.pumping = false;
    }
  }

  /** Drop waiters whose callers went away while they were queued. */
  private reap(): void {
    for (const [modelId, queue] of [...this.queues]) {
      const live = queue.filter((w) => !w.settled);
      if (live.length === 0) this.queues.delete(modelId);
      else this.queues.set(modelId, live);
    }
  }

  /** One scheduling decision. Returns false when nothing more can be done right now. */
  private async step(): Promise<boolean> {
    const perModel = this.options.maxConcurrentPerModel ?? 1;
    const drainLimit = this.options.drainLimit ?? 8;
    const others = (modelId: string): boolean => [...this.queues.keys()].some((id) => id !== modelId);

    // 1. Serve a RESIDENT model that has room — the drain-before-swap half. Bounded by `drainLimit`
    //    when someone else is waiting, or a continuously-fed model never yields residency at all.
    for (const [modelId, queue] of this.queues) {
      const entry = this.resident.get(modelId);
      if (entry === undefined || queue.length === 0 || entry.leases >= perModel) continue;
      if (others(modelId) && entry.streak >= drainLimit) continue;
      const waiter = queue.shift()!;
      if (queue.length === 0) this.queues.delete(modelId);
      entry.streak = others(modelId) ? entry.streak + 1 : 0;
      // Everyone else's streak resets: the limit is about taking TURNS, and a model that just yielded
      // must not be barred again on a stale count.
      for (const [id, other] of this.resident) if (id !== modelId) other.streak = 0;
      this.grant(waiter, entry);
      return true;
    }

    // 2. Nothing resident can be served. Find a queued model that is NOT resident and make room.
    const pending = [...this.queues.keys()].find((id) => !this.resident.has(id));
    if (pending === undefined) return false;

    const maxResident = this.options.maxResident ?? 1;
    if (this.resident.size >= maxResident) {
      // Evict the least-recently-used IDLE model. A model with leases is in use and is never touched —
      // no preemption — so a machine whose every resident model is busy simply waits.
      const idle = [...this.resident.entries()].filter(([, r]) => r.leases === 0).sort((a, b) => a[1].lastUsed - b[1].lastUsed)[0];
      if (idle === undefined) return false; // everything is busy; a release will pump again
      const [victim] = idle;
      this.resident.delete(victim);
      this.placements.delete(victim);
      log.debug("evicting an idle model to make room", { victim, for: pending });
      await this.options.unload?.(victim);
    }

    // 3. Admit the model we just made room FOR — in the same step, deliberately.
    //
    // Returning between the eviction and the admission looks harmless and is not: re-entering re-picks
    // `pending` from the queue map, whose iteration order is insertion order, so the model that had
    // just been evicted to make way was chosen again the instant it stopped being resident. The
    // observed effect was a drain limit that yielded and then immediately handed residency back to the
    // same model — starvation with extra unloads. The two decisions are one decision.
    await this.admit(pending);
    // Either way there is progress to re-check: admitted (its queue can now be served) or refused
    // (every caller waiting on it has been settled).
    return true;
  }

  /** Run the probe and the policy for a model, and make it resident if allowed. */
  private async admit(modelId: string): Promise<boolean> {
    let placement: Placement | undefined;
    try {
      placement = await this.options.probe?.predict(modelId);
    } catch (err) {
      log.warn("placement probe failed; admitting without a prediction", { modelId, error: String(err) });
    }
    if (placement !== undefined) {
      const policy = this.options.policy ?? REFUSE_DEGRADED;
      let decision: PlacementDecision;
      try {
        decision = await policy({ modelId, placement });
      } catch (err) {
        decision = { action: "refuse", reason: `the placement policy threw: ${String(err)}` };
      }
      if (decision.action === "refuse") {
        this.rejectAll(modelId, new PlacementRefused(modelId, decision.reason, placement));
        return false;
      }
      if (decision.action === "degrade") {
        placement = { ...placement, gpuLayers: decision.gpuLayers };
        log.debug("admitting with a reduced GPU split", { modelId, gpuLayers: decision.gpuLayers });
      }
    }
    this.placements.set(modelId, placement);
    this.resident.set(modelId, { leases: 0, streak: 0, lastUsed: ++this.clock });
    return true;
  }

  private rejectAll(modelId: string, err: unknown): void {
    const queue = this.queues.get(modelId) ?? [];
    this.queues.delete(modelId);
    for (const waiter of queue) {
      if (waiter.settled) continue;
      waiter.settled = true;
      waiter.reject(err);
    }
  }

  /** Unload everything. Used on shutdown; a lease still held is NOT waited for — the caller is closing. */
  async close(): Promise<void> {
    const ids = [...this.resident.keys()];
    this.resident.clear();
    this.placements.clear();
    // EVERY queue, not just the resident models'. A caller waiting on a model that was never loaded —
    // the common case, since waiting is exactly what happens when there is no room for it — has no
    // resident entry to be found by, and would otherwise wait forever on a manager that is gone. A hang
    // on shutdown is still a hang.
    for (const modelId of [...this.queues.keys()]) this.rejectAll(modelId, new Error("the residency manager was closed"));
    await Promise.allSettled(ids.map((id) => this.options.unload?.(id)));
  }
}
