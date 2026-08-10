/**
 * `MetricsAlgebra` is a MONOID, and the identity law is what the rest of it rests on.
 *
 * `merge` alone is a semigroup, and a semigroup cannot answer "no work happened". Every layer that
 * needed to say it invented its own answer: `withRetry` hand-rolls the identity as an
 * `accumulated === undefined ?` branch, and — worse — a wrapper that SYNTHESIZES a failure had no way
 * to build the caller's `M` at all, so the failure constructors hardcoded `{ durationMs: 0 }`. That is
 * why `withDeadline`, `withMemoize` and `createOperationExecutor` were pinned to `ExecMetrics` while
 * their siblings were generic in it: a wrapper cannot be polymorphic in a type it cannot construct.
 *
 * So `empty()` is not decoration, and the LAW is not decoration either. If folding in an identity
 * perturbs a real measurement, then a refusal — which is exactly an identity-valued result — silently
 * corrupts the roll-up it lands in. The case that actually bit: `mergeWorkflowMetrics` took
 * `costSource: b.costSource` unconditionally, so merging any zero-cost step after a real provider call
 * downgraded that call's provenance from `provider` to whatever the step happened to claim.
 */
import { describe, expect, it } from "vitest";
import { EXEC_METRICS_ALGEBRA, emptyExecMetrics, mergeExecMetrics, type ExecMetrics } from "../src/index.js";
import { emptyWorkflowMetrics, mergeWorkflowMetrics } from "@declarative-ai/hw";

/** The law, as a reusable assertion — it has to hold on BOTH sides. */
function obeysIdentity<M extends ExecMetrics>(
  algebra: { merge(a: M, b: M): M; empty(): M },
  sample: M,
): void {
  expect(algebra.merge(sample, algebra.empty())).toEqual(sample);
  expect(algebra.merge(algebra.empty(), sample)).toEqual(sample);
}

describe("MetricsAlgebra.empty is a two-sided identity", () => {
  it("holds for the exec floor", () => {
    obeysIdentity(EXEC_METRICS_ALGEBRA, { durationMs: 120, startMs: 1_700_000_000_000 });
  });

  it("does not date a measurement to the epoch", () => {
    // `startMs` is ABSENT from the identity rather than zero. `mergeExecMetrics` takes the FIRST
    // observation, so a zero would win and every merged measurement would claim to have started in 1970.
    expect(emptyExecMetrics().startMs).toBeUndefined();
    expect(mergeExecMetrics(emptyExecMetrics(), { durationMs: 5, startMs: 999 }).startMs).toBe(999);
  });

  it("leaves optional fields absent rather than giving them a spurious zero", () => {
    // "Not reported" and "zero" are different claims. Folding in an identity must not convert one to
    // the other, or a run that measured no queueing starts reporting that it measured zero queueing.
    const merged = mergeExecMetrics(emptyExecMetrics(), { durationMs: 5 });
    expect("queuedMs" in merged).toBe(false);
    expect("childLlmCalls" in merged).toBe(false);
  });

  it("holds for the workflow algebra — which required fixing how it merges costSource", () => {
    obeysIdentity({ merge: mergeWorkflowMetrics, empty: emptyWorkflowMetrics }, {
      durationMs: 200,
      startMs: 1_700_000_000_000,
      costUsd: 0.42,
      costSource: "provider",
      childLlmCalls: 2,
      childCostUsd: 0.4,
    });
  });

  it("keeps the BETTER-KNOWN cost source, not the right-hand one", () => {
    // The live bug the law exposed: `costSource: b.costSource` meant any zero-cost step merged after a
    // real call reported that call's price as a guess. Ranked (`provider` > `table` > `unknown`), the
    // authoritative figure survives — and the identity becomes neutral for free.
    const provider = { durationMs: 10, costUsd: 1, costSource: "provider" as const };
    const guess = { durationMs: 0, costUsd: 0, costSource: "unknown" as const };
    expect(mergeWorkflowMetrics(provider, guess).costSource).toBe("provider");
    expect(mergeWorkflowMetrics(guess, provider).costSource).toBe("provider");
    // …and a better observation still upgrades a worse one, which is the whole point of ranking.
    const table = { durationMs: 0, costUsd: 0.5, costSource: "table" as const };
    expect(mergeWorkflowMetrics(table, provider).costSource).toBe("provider");
  });
});
