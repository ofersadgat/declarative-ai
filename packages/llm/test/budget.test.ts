import { describe, expect, it } from "vitest";
import type { BudgetMeter, BudgetReservation } from "@declarative-ai/core";
import { LlmCallExecutor, type CallRunner } from "../src/executor";
import { withBudget, type BudgetOptions } from "../src/wrappers";
import { DEF, fakeRunner, okOutcome, specOf } from "./fakes";

/** A deterministic price model (no dependence on the real catalog): $0.00001 per token, flat. */
const pricing: NonNullable<BudgetOptions["pricing"]> = {
  estimateCostUsd: (_m, i, o) => (i + o) * 0.00001,
  affordableOutputTokens: (_m, _i, avail) => Math.floor(avail * 100_000),
};

/** A meter whose `reserve` is scripted per call; records every reserve estimate and every settle amount. */
function scriptedMeter(script: (est: number) => BudgetReservation | null, available = 1) {
  const reserves: number[] = [];
  const settled: number[] = [];
  const meter: BudgetMeter = {
    async reserve(est) {
      reserves.push(est);
      return script(est);
    },
    async availableCostUsd() {
      return available;
    },
  };
  const reservation = (ledgerId?: string): BudgetReservation => ({
    ledgerId,
    settle: async (actual) => void settled.push(actual),
  });
  return { meter, reserves, settled, reservation };
}

describe("withBudget — per-call reserve → settle", () => {
  it("reserves the estimate before the call and settles the ACTUAL cost after, stamping the ledger id", async () => {
    const m = scriptedMeter(() => m.reservation("led-1"));
    const { runner, calls } = fakeRunner([okOutcome()]);
    const exec = withBudget({ meter: m.meter, pricing }, new LlmCallExecutor({ runner }));

    const out = await exec.start(specOf(), {}).outcome;

    expect(out.error).toBeUndefined();
    expect(m.reserves).toHaveLength(1); // reserved once (estimate fit the balance)
    expect(m.settled).toEqual([okOutcome().metrics.cost]); // settled the actual recorded cost, not the estimate
    expect(out.metrics.ledgerId).toBe("led-1");
    expect(calls[0]!.params.maxOutputTokens).toBe(DEF.maxOutputTokens); // unclamped
  });

  it("reads the meter from ctx.meter when none is given at construction", async () => {
    const m = scriptedMeter(() => m.reservation());
    const { runner } = fakeRunner([okOutcome()]);
    const exec = withBudget({ pricing }, new LlmCallExecutor({ runner }));

    const out = await exec.start(specOf(), { meter: m.meter }).outcome;

    expect(out.error).toBeUndefined();
    expect(m.reserves).toHaveLength(1);
    expect(m.settled).toEqual([okOutcome().metrics.cost]);
  });

  it("clamps maxOutputTokens to the affordable ceiling and retries the reserve once when the full estimate is refused", async () => {
    let n = 0;
    const m = scriptedMeter(() => (++n === 1 ? null : m.reservation("led-c")), 0.005); // afford = 500 tokens
    const { runner, calls } = fakeRunner([okOutcome()]);
    const exec = withBudget({ meter: m.meter, pricing }, new LlmCallExecutor({ runner }));

    // A generous configured cap so the affordable ceiling (500) is the binding one.
    const out = await exec.start(specOf({}, { ...DEF, maxOutputTokens: 100_000 }), {}).outcome;

    expect(m.reserves).toHaveLength(2); // refused once, then reserved against the clamped ceiling
    expect(calls[0]!.params.maxOutputTokens).toBe(500); // the call was sent the provider-enforced clamp
    expect(out.error).toBeUndefined();
    expect(m.settled).toEqual([okOutcome().metrics.cost]);
  });

  it("refuses with an out-of-credits outcome (no call, no settle) when even the clamped reserve won't fit", async () => {
    const m = scriptedMeter(() => null, 0); // nothing affordable → afford = 0 < MIN_USEFUL
    const { runner, calls } = fakeRunner([okOutcome()]);
    const exec = withBudget({ meter: m.meter, pricing }, new LlmCallExecutor({ runner }));

    const out = await exec.start(specOf(), {}).outcome;

    expect(out.error?.classification).toBe("out-of-credits");
    expect(calls).toHaveLength(0); // the call was never made
    expect(m.settled).toHaveLength(0); // nothing to settle
  });

  it("settles a FAILED call (real spend, usually $0) rather than leaving the hold dangling", async () => {
    const m = scriptedMeter(() => m.reservation("led-f"));
    const failed = okOutcome({ error: { classification: "permanent", reason: "boom" }, metrics: { durationMs: 5 } });
    const { runner } = fakeRunner([failed]);
    const exec = withBudget({ meter: m.meter, pricing }, new LlmCallExecutor({ runner }));

    const out = await exec.start(specOf(), {}).outcome;

    expect(out.error?.classification).toBe("permanent");
    expect(m.settled).toEqual([0]); // failed metrics carry no cost → settle 0
  });

  it("is a pure passthrough when no meter is available", async () => {
    const throwIfCalled: CallRunner = async () => {
      throw new Error("meter should not have been consulted");
    };
    void throwIfCalled;
    const { runner } = fakeRunner([okOutcome()]);
    const exec = withBudget({ pricing }, new LlmCallExecutor({ runner }));

    const out = await exec.start(specOf(), {}).outcome; // no ctx.meter, no config.meter

    expect(out.error).toBeUndefined();
    expect(out.value).toEqual({ answer: "4" });
  });
});
