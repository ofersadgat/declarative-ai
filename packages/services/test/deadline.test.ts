import { describe, expect, it } from "vitest";
import {
  computeRemainingMs,
  deadlineDecision,
  DEFAULT_FLOOR_MS,
  DEFAULT_SAFETY_MARGIN_MS,
} from "../src/index";

describe("deadline cutoff (§6.2)", () => {
  const cfg = { maxDurationMs: 300_000 };

  it("measures remaining from the fixed stepStart origin, net of the margin", () => {
    // started at t=1000, 50s elapsed -> 300s - 50s - 10s margin = 240s remaining.
    expect(computeRemainingMs(1000, cfg, 1000 + 50_000)).toBe(300_000 - 50_000 - DEFAULT_SAFETY_MARGIN_MS);
  });

  it("proceeds while remaining time is at/above the floor", () => {
    const d = deadlineDecision(1000, cfg, 1000); // full budget
    expect(d.proceed).toBe(true);
    expect(d.remainingMs).toBe(300_000 - DEFAULT_SAFETY_MARGIN_MS);
  });

  it("refuses to start a call below the floor (avoids sub-1s aborts that lose salvage)", () => {
    // only ~6s of wall-clock budget: 6s - 10s margin is already negative.
    const d = deadlineDecision(0, { maxDurationMs: 6_000 }, 0);
    expect(d.proceed).toBe(false);
    expect(d.remainingMs).toBeLessThan(DEFAULT_FLOOR_MS);
  });

  it("honors custom margin/floor", () => {
    const d = deadlineDecision(0, { maxDurationMs: 100_000, safetyMarginMs: 2_000, floorMs: 1_000 }, 0);
    expect(d.remainingMs).toBe(98_000);
    expect(d.proceed).toBe(true);
  });
});
