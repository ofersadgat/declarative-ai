import { describe, expect, it } from "vitest";
import { backoffDelayMs } from "../src/retry";

// The retry LOOP is `withRetry` (see wrappers.test.ts, "withRetry"). `retryLoop` — a second, parallel
// implementation exported from the index and called by nothing but the tests that used to live here —
// is gone; its budget gate and its two short-circuits are now conditions in `withRetry` itself and are
// covered there against the executor that actually ships.

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
