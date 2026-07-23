import { describe, expect, it } from "vitest";
import {
  AdaptiveRateController,
  ConcurrencyLimiter,
  PassthroughRateLimiter,
  prefixModelLimitResolver,
  ProviderDispatchRateLimiter,
  TokenBucket,
} from "../src/index";

const flush = async (n = 20): Promise<void> => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

describe("ConcurrencyLimiter (§6.2.B)", () => {
  it("never runs more than `limit` tasks at once", async () => {
    const limiter = new ConcurrencyLimiter(2);
    let active = 0;
    let peak = 0;
    const releases: Array<() => void> = [];
    const make = () =>
      limiter.run(
        () =>
          new Promise<void>((res) => {
            active++;
            peak = Math.max(peak, active);
            releases.push(() => {
              active--;
              res();
            });
          }),
      );

    const all = Promise.all([make(), make(), make(), make(), make()]);
    await flush();
    expect(releases.length).toBe(2); // only 2 admitted at once

    while (releases.length > 0) {
      releases.shift()!();
      await flush();
    }
    await all;
    expect(peak).toBe(2);
  });

  it("raising the limit immediately admits queued waiters (AIMD increase)", async () => {
    const limiter = new ConcurrencyLimiter(1);
    let active = 0;
    const releases: Array<() => void> = [];
    const make = () =>
      limiter.run(
        () =>
          new Promise<void>((res) => {
            active++;
            releases.push(() => {
              active--;
              res();
            });
          }),
      );
    void make();
    void make();
    void make();
    await flush();
    expect(active).toBe(1);

    limiter.setLimit(3);
    await flush();
    expect(active).toBe(3); // the two queued tasks were admitted at once
    releases.forEach((r) => r());
  });
});

describe("TokenBucket (TPM pre-admission)", () => {
  it("blocks until enough tokens have refilled, then removes them", async () => {
    let t = 0;
    const waits: number[] = [];
    const bucket = new TokenBucket(
      1, // 1 token/ms
      100, // capacity
      () => t,
      async (ms) => {
        waits.push(ms);
        t += ms; // advance the fake clock by the waited time
      },
    );

    await bucket.remove(60); // 100 -> 40, no wait
    expect(waits).toEqual([]);

    await bucket.remove(60); // need 60, have 40, deficit 20 -> wait 20ms -> refill 20 -> remove
    expect(waits).toEqual([20]);
    expect(bucket.available).toBeCloseTo(0);
  });
});

describe("AdaptiveRateController (AIMD)", () => {
  it("halves concurrency on a rate-limit and additively increases on sustained success", () => {
    const c = new AdaptiveRateController({ initialConcurrency: 8, maxConcurrency: 16, minConcurrency: 1, increaseEvery: 3 });
    expect(c.concurrency).toBe(8);

    c.reportOutcome({ rateLimited: true });
    expect(c.concurrency).toBe(4); // halved
    c.reportOutcome({ rateLimited: true });
    expect(c.concurrency).toBe(2);

    c.reportOutcome({});
    c.reportOutcome({});
    expect(c.concurrency).toBe(2); // < increaseEvery successes -> no change
    c.reportOutcome({});
    expect(c.concurrency).toBe(3); // 3rd success -> +1
  });

  it("never drops below minConcurrency", () => {
    const c = new AdaptiveRateController({ initialConcurrency: 2, minConcurrency: 1 });
    c.reportOutcome({ rateLimited: true });
    c.reportOutcome({ rateLimited: true });
    c.reportOutcome({ rateLimited: true });
    expect(c.concurrency).toBe(1);
  });

  it("schedule runs the work and returns its value (concurrency-only, no bucket)", async () => {
    const c = new AdaptiveRateController({ initialConcurrency: 4 });
    const out = await c.schedule({ inputTokens: 100, outputTokens: 0 }, async () => 42);
    expect(out).toBe(42);
  });
});

describe("per-model rate buckets (§6.2.B — the shape providers publish)", () => {
  const LIMITS = {
    // 60 input tokens/min = 1/ms with the fake clock below; rpm 60 = 1 req/ms.
    "claude-fable-5": { rpm: 60, inputTpm: 60, outputTpm: 120 },
    "claude-sonnet-4": { inputTpm: 6000 },
  };

  it("prefixModelLimitResolver matches longest prefix; unmatched models have no buckets", () => {
    const r = prefixModelLimitResolver(LIMITS);
    expect(r("claude-fable-5")?.key).toBe("claude-fable-5");
    expect(r("claude-sonnet-4-6")?.key).toBe("claude-sonnet-4"); // prefix match on the dated id
    expect(r("openai/gpt-5.5")).toBeUndefined();
  });

  it("throttles a model against ITS buckets; other families are untouched", async () => {
    let t = 0;
    const waits: number[] = [];
    const c = new AdaptiveRateController({
      initialConcurrency: 8,
      modelLimits: prefixModelLimitResolver(LIMITS),
      now: () => t,
      wait: async (ms) => {
        waits.push(ms);
        t += ms;
      },
    });
    // Fable's input bucket: capacity 60. First call takes 40 — no wait. Second call of 40 must wait ~20ms.
    await c.schedule({ modelId: "claude-fable-5", inputTokens: 40, outputTokens: 0 }, async () => 0);
    expect(waits).toEqual([]);
    await c.schedule({ modelId: "claude-fable-5", inputTokens: 40, outputTokens: 0 }, async () => 0);
    expect(waits.length).toBeGreaterThan(0); // throttled by fable's own ITPM bucket
    const fableWaits = waits.length;
    // Sonnet has a fat ITPM bucket and shares NOTHING with fable — no additional wait.
    await c.schedule({ modelId: "claude-sonnet-4-6", inputTokens: 40, outputTokens: 0 }, async () => 0);
    expect(waits.length).toBe(fableWaits);
    // A model with no matching family bypasses per-model buckets entirely.
    await c.schedule({ modelId: "openai/gpt-5.5", inputTokens: 10_000, outputTokens: 10_000 }, async () => 0);
    expect(waits.length).toBe(fableWaits);
  });

  it("the OTPM bucket meters output tokens separately from input", async () => {
    let t = 0;
    const waits: number[] = [];
    const c = new AdaptiveRateController({
      initialConcurrency: 8,
      modelLimits: prefixModelLimitResolver(LIMITS),
      now: () => t,
      wait: async (ms) => {
        waits.push(ms);
        t += ms;
      },
    });
    // Output bucket capacity 120: two calls of 80 output tokens — the second waits, even with 0 input.
    await c.schedule({ modelId: "claude-fable-5", inputTokens: 0, outputTokens: 80 }, async () => 0);
    expect(waits).toEqual([]);
    await c.schedule({ modelId: "claude-fable-5", inputTokens: 0, outputTokens: 80 }, async () => 0);
    expect(waits.length).toBeGreaterThan(0);
  });
});

describe("ProviderDispatchRateLimiter — provider+model-specific routing", () => {
  it("routes each call AND its outcome to the limiter of the call's own provider", async () => {
    const log: string[] = [];
    const fake = (name: string) => ({
      schedule: async <T,>(_est: unknown, run: () => Promise<T>): Promise<T> => {
        log.push(`${name}:schedule`);
        return run();
      },
      reportOutcome: () => void log.push(`${name}:outcome`),
    });
    const anthropic = fake("anthropic");
    const openrouter = fake("openrouter");
    const d = new ProviderDispatchRateLimiter((id) => (id?.startsWith("claude-") ? anthropic : openrouter));

    await d.schedule({ modelId: "claude-fable-5", inputTokens: 1, outputTokens: 1 }, async () => 0);
    d.reportOutcome({ modelId: "claude-fable-5" });
    // The SAME underlying model served through OpenRouter is OpenRouter traffic — different limiter.
    await d.schedule({ modelId: "anthropic/claude-fable-5", inputTokens: 1, outputTokens: 1 }, async () => 0);
    d.reportOutcome({ modelId: "anthropic/claude-fable-5" });

    expect(log).toEqual(["anthropic:schedule", "anthropic:outcome", "openrouter:schedule", "openrouter:outcome"]);
  });
});

describe("PassthroughRateLimiter (disabled provider)", () => {
  it("runs immediately with no throttling and swallows outcomes", async () => {
    const p = new PassthroughRateLimiter();
    const out = await p.schedule({ modelId: "x", inputTokens: 1e9, outputTokens: 1e9 }, async () => 7);
    expect(out).toBe(7);
    p.reportOutcome(); // no-op
  });
});

