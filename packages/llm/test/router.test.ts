import { describe, expect, it } from "vitest";
import {
  createModelRouter,
  familyForModel,
  isAnthropicModel,
  parseModelRoute,
  providerNativeId,
} from "../src/router";

describe("provider router (§5)", () => {
  it("parses the explicit `{route}/{model}` prefix and strips it to the provider-native id", () => {
    expect(parseModelRoute("anthropic/claude-sonnet-5")).toEqual({ route: "anthropic", providerId: "claude-sonnet-5" });
    expect(parseModelRoute("openrouter/openai/gpt-5")).toEqual({ route: "openrouter", providerId: "openai/gpt-5" });
    // An Anthropic model served THROUGH OpenRouter is route "openrouter" (non-native), unambiguously.
    expect(parseModelRoute("openrouter/anthropic/claude-opus-4.8")).toEqual({
      route: "openrouter",
      providerId: "anthropic/claude-opus-4.8",
    });
    expect(providerNativeId("anthropic/claude-sonnet-5")).toBe("claude-sonnet-5");
    expect(familyForModel("anthropic/claude-opus-4-8")).toBe("anthropic");
    expect(familyForModel("openrouter/meta-llama/llama-3.1-8b-instruct")).toBe("openrouter");
  });

  it("rejects a bare/unprefixed model id (routing is explicit, never guessed)", () => {
    expect(() => parseModelRoute("claude-opus-4-8")).toThrow(/must be route-prefixed/);
    expect(() => parseModelRoute("openai/gpt-5")).toThrow(/must be route-prefixed/); // "openai" is not a route
    expect(() => familyForModel("claude-opus-4-8")).toThrow(/must be route-prefixed/);
  });

  it("isAnthropicModel is a native-id predicate (route already stripped)", () => {
    expect(isAnthropicModel("claude-haiku-4-5")).toBe(true); // bare native anthropic id
    expect(isAnthropicModel("anthropic/claude-opus-4.8")).toBe(false); // vendor-prefixed = OpenRouter-served
    expect(isAnthropicModel("openai/gpt-4.1-mini")).toBe(false);
  });

  it("resolves a model object without needing a live key (key is used at call time)", () => {
    const router = createModelRouter({
      anthropicApiKey: "test",
      openRouterApiKey: "test",
      skipDispatcher: true,
    });
    expect(router.resolveModel("anthropic/claude-haiku-4-5")).toBeDefined();
    expect(router.resolveModel("openrouter/openai/gpt-4.1-mini")).toBeDefined();
    expect(router.isAnthropic("anthropic/claude-haiku-4-5")).toBe(true);
    expect(router.isAnthropic("openrouter/anthropic/claude-opus-4.8")).toBe(false); // Anthropic via OpenRouter
  });

  it("sets the OpenRouter strict flag + require_parameters routing per the enforce decision (§5.1)", () => {
    const router = createModelRouter({ openRouterApiKey: "test", skipDispatcher: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- reach into the provider's settings.
    const settingsOf = (m: unknown): any => (m as { settings?: unknown }).settings;

    const strict = settingsOf(router.resolveModel("openrouter/openai/gpt-4.1-mini", { strictStructuredOutput: true }));
    expect(strict.structuredOutputs).toEqual({ strict: true });
    expect(strict.provider).toEqual({ require_parameters: true }); // route only to capable upstreams

    const advisory = settingsOf(router.resolveModel("openrouter/openai/gpt-4.1-mini", { strictStructuredOutput: false }));
    expect(advisory.structuredOutputs).toEqual({ strict: false });
    expect(advisory.provider).toBeUndefined(); // unconstrained routing for an advisory hint
  });
});
