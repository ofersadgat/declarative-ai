/**
 * `mergeLlmMetrics` — the algebra a prompt executor registers so `exec` can fold retry attempts without
 * knowing what a token is. The interesting field is `costSource`: it is the provenance signal a settling
 * budget reads to tell "this call was free" from "we never priced this call".
 */
import { describe, expect, it } from "vitest";
import { mergeLlmMetrics, type LlmMetrics } from "../src/output";

const m = (over: Partial<LlmMetrics> = {}): LlmMetrics => ({ durationMs: 10, costUsd: 0, costSource: "unknown", ...over });

describe("mergeLlmMetrics", () => {
  it("adds money and tokens and keeps the FIRST start time", () => {
    const out = mergeLlmMetrics(m({ startMs: 100, costUsd: 0.001, inputTokens: 10 }), m({ startMs: 900, costUsd: 0.002, inputTokens: 4 }));
    expect(out).toMatchObject({ durationMs: 20, startMs: 100, costUsd: 0.003, inputTokens: 14 });
  });

  it("keeps the MORE AUTHORITATIVE costSource — a retry must not relabel real spend as un-priced", () => {
    // The concrete regression: attempt 1 was billed and priced from the table, attempt 2 measured
    // nothing (a runner that lost its metrics reports `{0, "unknown"}`). Taking `b`'s source
    // unconditionally yielded `{costUsd: 0.004, costSource: "unknown"}` — a real charge wearing the
    // label that tells a budget "we do not know what this cost".
    expect(mergeLlmMetrics(m({ costUsd: 0.004, costSource: "table" }), m({ costUsd: 0, costSource: "unknown" }))).toMatchObject({
      costUsd: 0.004,
      costSource: "table",
    });
  });

  it("ranks provider > table > unknown, in either argument order", () => {
    expect(mergeLlmMetrics(m({ costSource: "table" }), m({ costSource: "provider" })).costSource).toBe("provider");
    expect(mergeLlmMetrics(m({ costSource: "provider" }), m({ costSource: "table" })).costSource).toBe("provider");
    expect(mergeLlmMetrics(m({ costSource: "unknown" }), m({ costSource: "table" })).costSource).toBe("table");
  });

  it("stays `unknown` when NEITHER side was priced — the honest answer, not an upgrade", () => {
    expect(mergeLlmMetrics(m(), m()).costSource).toBe("unknown");
  });
});
