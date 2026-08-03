import { describe, expect, it, vi } from "vitest";
import { PlacementRefused, REFUSE_DEGRADED, ResidencyManager, type Placement, type PlacementProbe } from "../src/residency.js";

/**
 * The residency ARBITER. No `node-llama-cpp` here on purpose: prediction is a seam, so the scheduling
 * logic is tested against invented hardware and runs the same on a machine with no GPU at all.
 */

const placement = (over: Partial<Placement> = {}): Placement => ({
  tier: "vram",
  gpuLayers: 32,
  totalLayers: 32,
  contextSize: 4096,
  vramBytes: 8e9,
  ramBytes: 1e8,
  ...over,
});

const probeOf = (byModel: Record<string, Placement | undefined>): PlacementProbe => ({
  predict: (modelId) => Promise.resolve(byModel[modelId]),
});

/** Settle the microtask queue so the scheduler's async pump runs to quiescence. */
const settle = async (): Promise<void> => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

describe("leases", () => {
  it("admits a call, and holds the model resident until it is released", async () => {
    const m = new ResidencyManager();
    const lease = await m.acquire("a");
    expect(m.residentModels()).toEqual(["a"]);
    lease.release();
    expect(m.residentModels()).toEqual(["a"]); // released ≠ evicted; nothing else wants the slot
  });

  it("SHARES a resident model across concurrent calls up to the declared limit", async () => {
    const m = new ResidencyManager({ maxConcurrentPerModel: 2 });
    const first = await m.acquire("a");
    const second = await m.acquire("a");
    let third = false;
    void m.acquire("a").then(() => (third = true));
    await settle();
    expect(third).toBe(false); // the third waits — concurrency is bought with KV cache, not free
    first.release();
    await settle();
    expect(third).toBe(true);
    second.release();
  });

  it("release is IDEMPOTENT", async () => {
    // A double release would let a model be evicted while a call still holds it — the one bookkeeping
    // error whose symptom is a crash inside someone else's generation.
    const m = new ResidencyManager({ maxConcurrentPerModel: 1 });
    const lease = await m.acquire("a");
    lease.release();
    lease.release();
    let second = false;
    void m.acquire("a").then(() => (second = true));
    await settle();
    expect(second).toBe(true);
    const third = m.acquire("a");
    let granted = false;
    void third.then(() => (granted = true));
    await settle();
    expect(granted).toBe(false); // only ONE slot was returned, not two
  });
});

describe("eviction", () => {
  it("evicts an IDLE model to make room, and unloads it", async () => {
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 1, unload });
    (await m.acquire("a")).release();
    await m.acquire("b");
    expect(m.residentModels()).toEqual(["b"]);
    expect(unload).toHaveBeenCalledWith("a");
  });

  it("NEVER evicts a model that is in use — no preemption", async () => {
    // A generation in flight keeps its model until it finishes. The cost is a long call delaying a
    // swap; the alternative is killing work already paid for.
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 1, unload });
    const held = await m.acquire("a");
    let gotB = false;
    void m.acquire("b").then(() => (gotB = true));
    await settle();
    expect(gotB).toBe(false);
    expect(unload).not.toHaveBeenCalled();
    held.release();
    await settle();
    expect(gotB).toBe(true);
    expect(unload).toHaveBeenCalledWith("a");
  });

  it("evicts the LEAST RECENTLY USED idle model", async () => {
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 2, unload });
    (await m.acquire("a")).release();
    (await m.acquire("b")).release();
    (await m.acquire("a")).release(); // `a` is now the more recent of the two
    await m.acquire("c");
    expect(unload).toHaveBeenCalledWith("b");
    expect(m.residentModels().sort()).toEqual(["a", "c"]);
  });
});

describe("drain before swap", () => {
  it("serves a resident model's whole queue rather than swapping per call", async () => {
    // The pathological case this exists for: global FIFO with room for one model and an A,B,A,B
    // arrival pattern reloads on every single call.
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 1, maxConcurrentPerModel: 1, unload });
    const order: string[] = [];
    const call = async (id: string) => {
      const lease = await m.acquire(id);
      order.push(id);
      lease.release();
    };
    await Promise.all([call("a"), call("b"), call("a"), call("b")]);
    // Four calls, two models, ONE swap — the two `a`s were served together and the two `b`s together.
    expect(unload).toHaveBeenCalledTimes(1);
    expect(order.join("")).toBe("aabb");
  });

  it("yields after `drainLimit` so a busy model cannot starve another", async () => {
    // Drain-before-swap without a bound is indefinite starvation: a continuously-fed model would hold
    // residency forever and every other model would wait for a gap that never comes.
    const m = new ResidencyManager({ maxResident: 1, maxConcurrentPerModel: 1, drainLimit: 2 });
    const order: string[] = [];
    const call = async (id: string) => {
      const lease = await m.acquire(id);
      order.push(id);
      lease.release();
    };
    await Promise.all([call("a"), call("a"), call("a"), call("a"), call("b")]);
    // `b` is served before the last `a`, rather than after every one of them.
    expect(order.indexOf("b")).toBeLessThan(order.length - 1);
    expect(order.filter((x) => x === "a")).toHaveLength(4);
  });
});

describe("placement policy", () => {
  it("admits a full-VRAM placement under the default policy", async () => {
    const m = new ResidencyManager({ probe: probeOf({ a: placement() }) });
    const lease = await m.acquire("a");
    expect(lease.placement?.tier).toBe("vram");
  });

  it("REFUSES a degraded placement by default, naming what it would have cost", async () => {
    // Refusing rather than coping: a model quietly running ten times slower is invisible in every
    // metric except wall-clock. A host that wants degradation says so in one line.
    const m = new ResidencyManager({ probe: probeOf({ a: placement({ tier: "ram", gpuLayers: 21 }) }) });
    await expect(m.acquire("a")).rejects.toBeInstanceOf(PlacementRefused);
    await expect(m.acquire("a")).rejects.toThrow(/spill into system RAM \(21\/32 layers/);
  });

  it("distinguishes a SWAP spill from a RAM spill in the refusal", async () => {
    const m = new ResidencyManager({ probe: probeOf({ a: placement({ tier: "swap", gpuLayers: 0 }) }) });
    await expect(m.acquire("a")).rejects.toThrow(/spill to swap/);
  });

  it("lets a caller's policy DEGRADE instead, and reports the split it chose", async () => {
    const m = new ResidencyManager({
      probe: probeOf({ a: placement({ tier: "ram", gpuLayers: 21 }) }),
      policy: () => ({ action: "degrade", gpuLayers: 16 }),
    });
    const lease = await m.acquire("a");
    expect(lease.placement?.gpuLayers).toBe(16);
  });

  it("settles EVERY caller waiting on a refused model", async () => {
    // They all asked for the same impossible thing; leaving them queued behind a decision that will
    // not change is a hang, not backpressure.
    const m = new ResidencyManager({ probe: probeOf({ a: placement({ tier: "swap" }) }) });
    const results = await Promise.allSettled([m.acquire("a"), m.acquire("a"), m.acquire("a")]);
    expect(results.map((r) => r.status)).toEqual(["rejected", "rejected", "rejected"]);
  });

  it("admits UNJUDGED when the probe has nothing to say", async () => {
    // A probe that abstains is better than one that guesses; an unmeasurable model is not a refused one.
    const m = new ResidencyManager({ probe: probeOf({}) });
    const lease = await m.acquire("unknown-model");
    expect(lease.placement).toBeUndefined();
  });

  it("admits unjudged when the probe THROWS, rather than failing the call", async () => {
    const m = new ResidencyManager({ probe: { predict: () => Promise.reject(new Error("nvml exploded")) } });
    await expect(m.acquire("a")).resolves.toBeDefined();
  });

  it("refuses when the POLICY throws — a fault must not become a silent admission", async () => {
    const m = new ResidencyManager({
      probe: probeOf({ a: placement({ tier: "ram" }) }),
      policy: () => {
        throw new Error("policy exploded");
      },
    });
    await expect(m.acquire("a")).rejects.toThrow(/policy exploded/);
  });

  it("REFUSE_DEGRADED is exported so a host can compose with the default rather than restate it", () => {
    expect(REFUSE_DEGRADED({ modelId: "a", placement: placement() })).toEqual({ action: "proceed" });
    expect(REFUSE_DEGRADED({ modelId: "a", placement: placement({ tier: "ram" }) })).toMatchObject({ action: "refuse" });
  });
});

describe("cancellation", () => {
  it("a cancel while QUEUED never loads the model", async () => {
    // Waiting can take minutes — an eviction, then a multi-gigabyte load. A caller that gave up must
    // not cause the work.
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 1, maxConcurrentPerModel: 1, unload });
    const held = await m.acquire("a");
    const ac = new AbortController();
    const queued = m.acquire("b", ac.signal);
    await settle();
    ac.abort();
    await expect(queued).rejects.toThrow(/canceled while queued/);
    held.release();
    await settle();
    expect(m.residentModels()).toEqual(["a"]); // `b` was never made resident
    expect(unload).not.toHaveBeenCalled();
  });

  it("an already-aborted caller is refused without queueing", async () => {
    const m = new ResidencyManager();
    await expect(m.acquire("a", AbortSignal.abort())).rejects.toThrow(/canceled before/);
  });

  it("a canceled waiter does not block the ones behind it", async () => {
    const m = new ResidencyManager({ maxResident: 1, maxConcurrentPerModel: 1 });
    const held = await m.acquire("a");
    const ac = new AbortController();
    const doomed = m.acquire("a", ac.signal);
    let survivor = false;
    void m.acquire("a").then(() => (survivor = true));
    await settle();
    ac.abort();
    await expect(doomed).rejects.toThrow();
    held.release();
    await settle();
    expect(survivor).toBe(true);
  });
});

describe("close", () => {
  it("unloads everything and settles anyone still waiting", async () => {
    const unload = vi.fn(() => Promise.resolve());
    const m = new ResidencyManager({ maxResident: 1, maxConcurrentPerModel: 1, unload });
    const held = await m.acquire("a");
    const queued = m.acquire("b");
    await settle();
    await m.close();
    expect(unload).toHaveBeenCalledWith("a");
    await expect(queued).rejects.toThrow(/closed/); // a hang on shutdown is still a hang
    held.release();
  });
});
