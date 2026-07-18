import { describe, expect, it } from "vitest";
import type { ExecFailure } from "@ai-exec/core";
import { backoffDelayMs, retryLoop } from "../src/retry";

const noWait = () => Promise.resolve();
const fail = (classification: ExecFailure["classification"], reason = "boom", retryAfterMs?: number): { failure: ExecFailure } => ({
  failure: { classification, reason, retryAfterMs },
});

describe("backoffDelayMs", () => {
  const opts = { baseBackoffMs: 500, maxBackoffMs: 60_000 };

  it("honors retry-after with bounded jitter, clamped to the max", () => {
    expect(backoffDelayMs(0, 10_000, opts, () => 0)).toBe(10_000);
    expect(backoffDelayMs(0, 10_000, opts, () => 1)).toBe(10_500); // + min(base, retryAfter) jitter
    expect(backoffDelayMs(0, 120_000, opts, () => 1)).toBe(60_000); // clamp
  });

  it("full-jitter exponential backoff without retry-after", () => {
    expect(backoffDelayMs(3, undefined, opts, () => 1)).toBe(Math.floor(500 * 2 ** 3) - 0);
    expect(backoffDelayMs(0, undefined, opts, () => 0)).toBe(0);
    expect(backoffDelayMs(20, undefined, opts, () => 1)).toBe(60_000); // ceiling clamp
  });
});

describe("retryLoop", () => {
  it("returns first success without extra attempts", async () => {
    let calls = 0;
    const res = await retryLoop(async () => {
      calls++;
      return { value: 42 } as { value: number; failure?: ExecFailure };
    }, { waitMs: noWait });
    expect(res.value).toBe(42);
    expect(calls).toBe(1);
  });

  it("retries network-retriable up to the cap, passing the attempt index through", async () => {
    const seen: number[] = [];
    const res = await retryLoop(
      async (i) => {
        seen.push(i);
        return fail("network-retriable");
      },
      { retryCap: 3, waitMs: noWait, random: () => 0 },
    );
    expect(seen).toEqual([0, 1, 2, 3]);
    expect(res.failure?.classification).toBe("network-retriable");
  });

  it("never re-rolls api-retriable by default, but does on explicit opt-in", async () => {
    let calls = 0;
    await retryLoop(async () => (calls++, fail("api-retriable")), { retryCap: 3, waitMs: noWait });
    expect(calls).toBe(1);
    calls = 0;
    await retryLoop(async () => (calls++, fail("api-retriable")), {
      retryCap: 2,
      retryApiRetriable: true,
      waitMs: noWait,
      random: () => 0,
    });
    expect(calls).toBe(3);
  });

  it("returns permanent failures immediately", async () => {
    let calls = 0;
    await retryLoop(async () => (calls++, fail("permanent")), { retryCap: 5, waitMs: noWait });
    expect(calls).toBe(1);
  });

  it("short-circuits on budget-exhausted and deadline-floor reasons even when network-retriable", async () => {
    let calls = 0;
    await retryLoop(async () => (calls++, fail("network-retriable", "budget-exhausted: wallet")), { waitMs: noWait });
    expect(calls).toBe(1);
    calls = 0;
    await retryLoop(async () => (calls++, fail("network-retriable", "deadline-floor: 3s left")), { waitMs: noWait });
    expect(calls).toBe(1);
  });

  it("stops when the budget gate refuses more attempts", async () => {
    let calls = 0;
    await retryLoop(async () => (calls++, fail("network-retriable")), {
      retryCap: 10,
      budget: { allowMore: () => calls < 2 },
      waitMs: noWait,
      random: () => 0,
    });
    expect(calls).toBe(2);
  });
});
