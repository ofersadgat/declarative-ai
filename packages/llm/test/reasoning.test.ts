import { describe, expect, it } from "vitest";
import { adaptReasoning } from "../src/reasoning";

describe("adaptReasoning — provider-neutral ReasoningSpec → provider providerOptions", () => {
  it("returns undefined when nothing is requested (a no-reasoning call is byte-identical)", () => {
    expect(adaptReasoning(undefined, { anthropic: false })).toBeUndefined();
    expect(adaptReasoning({}, { anthropic: true })).toBeUndefined();
  });

  it("OpenRouter takes the effort LEVEL", () => {
    expect(adaptReasoning({ effort: "high" }, { anthropic: false })).toEqual({ openrouter: { reasoning: { effort: "high" } } });
  });

  it("OpenRouter sends a budget as max_tokens when only a budget is given", () => {
    expect(adaptReasoning({ budgetTokens: 5000 }, { anthropic: false })).toEqual({ openrouter: { reasoning: { max_tokens: 5000 } } });
  });

  it("Anthropic takes a thinking BUDGET (a budget passes through)", () => {
    expect(adaptReasoning({ budgetTokens: 5000 }, { anthropic: true })).toEqual({ anthropic: { thinking: { type: "enabled", budgetTokens: 5000 } } });
  });

  it("Anthropic maps an effort level to a representative budget", () => {
    expect(adaptReasoning({ effort: "low" }, { anthropic: true })).toEqual({ anthropic: { thinking: { type: "enabled", budgetTokens: 2048 } } });
    expect(adaptReasoning({ effort: "high" }, { anthropic: true })).toEqual({ anthropic: { thinking: { type: "enabled", budgetTokens: 16384 } } });
  });

  it("an explicit budget wins over effort on Anthropic", () => {
    expect(adaptReasoning({ effort: "low", budgetTokens: 12000 }, { anthropic: true })).toEqual({ anthropic: { thinking: { type: "enabled", budgetTokens: 12000 } } });
  });
});
