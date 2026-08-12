import { describe, expect, it } from "vitest";
import { adaptReasoning } from "../src/reasoning.js";

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

  describe("the native openai route", () => {
    it("files the effort under the PROVIDER'S OWN key, spelled the way its client parses it", () => {
      // The whole point. `@ai-sdk/openai-compatible` reads `providerOptions[name]` where the name is
      // the one the client was built with (`openai`), and lowers `reasoningEffort` to
      // `reasoning_effort`. Under `openrouter` this request reaches the provider and is dropped
      // without a warning — the call runs at the model's default while everything else claims the
      // level was honoured.
      expect(adaptReasoning({ effort: "high" }, { anthropic: false, openai: true })).toEqual({
        openai: { reasoningEffort: "high" },
      });
    });

    it("clamps xhigh rather than refusing it, as every level-taking provider does", () => {
      expect(adaptReasoning({ effort: "xhigh" }, { anthropic: false, openai: true })).toEqual({
        openai: { reasoningEffort: "high" },
      });
    });

    it("converts a budget-only spec to a level, because there is no budget field to send it to", () => {
      expect(adaptReasoning({ budgetTokens: 1000 }, { anthropic: false, openai: true })).toEqual({ openai: { reasoningEffort: "low" } });
      expect(adaptReasoning({ budgetTokens: 8192 }, { anthropic: false, openai: true })).toEqual({ openai: { reasoningEffort: "medium" } });
      expect(adaptReasoning({ budgetTokens: 100_000 }, { anthropic: false, openai: true })).toEqual({ openai: { reasoningEffort: "high" } });
    });

    it("still emits nothing when nothing was requested", () => {
      expect(adaptReasoning(undefined, { anthropic: false, openai: true })).toBeUndefined();
      expect(adaptReasoning({}, { anthropic: false, openai: true })).toBeUndefined();
    });

    it("does not steal the openrouter arm — the same model relayed by OpenRouter keeps its shape", () => {
      expect(adaptReasoning({ effort: "high" }, { anthropic: false, openai: false })).toEqual({
        openrouter: { reasoning: { effort: "high" } },
      });
    });
  });
});
