import { describe, expect, it } from "vitest";
import {
  createRouter,
  familyForModel,
  isAnthropicModel,
} from "../src/router";

describe("provider router (§5)", () => {
  it("routes claude-* to anthropic and everything else to openrouter", () => {
    expect(isAnthropicModel("claude-haiku-4-5")).toBe(true);
    expect(isAnthropicModel("openai/gpt-4.1-mini")).toBe(false);
    expect(familyForModel("claude-opus-4-8")).toBe("anthropic");
    expect(familyForModel("meta-llama/llama-3.1-8b-instruct")).toBe("openrouter");
  });

  it("resolves a model object without needing a live key (key is used at call time)", () => {
    const router = createRouter({
      anthropicApiKey: "test",
      openRouterApiKey: "test",
      skipDispatcher: true,
    });
    expect(router.resolveModel("claude-haiku-4-5")).toBeDefined();
    expect(router.resolveModel("openai/gpt-4.1-mini")).toBeDefined();
    expect(router.isAnthropic("claude-haiku-4-5")).toBe(true);
  });

  it("sets the OpenRouter strict flag + require_parameters routing per the enforce decision (§5.1)", () => {
    const router = createRouter({ openRouterApiKey: "test", skipDispatcher: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- reach into the provider's settings.
    const settingsOf = (m: unknown): any => (m as { settings?: unknown }).settings;

    const strict = settingsOf(router.resolveModel("openai/gpt-4.1-mini", { strictStructuredOutput: true }));
    expect(strict.structuredOutputs).toEqual({ strict: true });
    expect(strict.provider).toEqual({ require_parameters: true }); // route only to capable upstreams

    const advisory = settingsOf(router.resolveModel("openai/gpt-4.1-mini", { strictStructuredOutput: false }));
    expect(advisory.structuredOutputs).toEqual({ strict: false });
    expect(advisory.provider).toBeUndefined(); // unconstrained routing for an advisory hint
  });
});
