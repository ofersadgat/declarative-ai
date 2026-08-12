import { describe, expect, it } from "vitest";
import {
  MODEL_ROUTES,
  createModelRouter,
  familyForModel,
  isAnthropicModel,
  isLocalModel,
  isRemoteModel,
  parseModelRoute,
  providerNativeId,
} from "../src/router.js";

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
    expect(() => familyForModel("claude-opus-4-8")).toThrow(/must be route-prefixed/);
    // The set is CLOSED — a prefix is not a route just because it is followed by a slash.
    expect(() => parseModelRoute("mistral/large")).toThrow(/must be route-prefixed/);
  });

  it("tells the NATIVE openai route from the same model relayed by openrouter", () => {
    // `openai/gpt-5` used to be refused, because there was no native route and the only way to that
    // model was through OpenRouter. Both are legal now and they are DIFFERENT calls: another
    // endpoint, another key, another price row.
    expect(parseModelRoute("openai/gpt-5")).toEqual({ route: "openai", providerId: "gpt-5" });
    expect(parseModelRoute("openrouter/openai/gpt-5")).toEqual({ route: "openrouter", providerId: "openai/gpt-5" });
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

  describe("locally-served routes", () => {
    it("parses `local` and `embedded` like any other route", () => {
      expect(parseModelRoute("local/qwen2.5-32b-instruct")).toEqual({ route: "local", providerId: "qwen2.5-32b-instruct" });
      expect(parseModelRoute("embedded/qwen2.5-7b-instruct-q4_k_m")).toEqual({
        route: "embedded",
        providerId: "qwen2.5-7b-instruct-q4_k_m",
      });
      expect(familyForModel("embedded/qwen2.5-7b-instruct-q4_k_m")).toBe("embedded");
    });

    it("MODEL_ROUTES is the parser's whole vocabulary — every entry round-trips", () => {
      for (const route of MODEL_ROUTES) {
        expect(parseModelRoute(`${route}/some-model`)).toEqual({ route, providerId: "some-model" });
      }
    });

    it("splits remote from local, which is what scopes a rate limiter", () => {
      expect(isRemoteModel("anthropic/claude-haiku-4-5")).toBe(true);
      expect(isRemoteModel("openrouter/openai/gpt-5")).toBe(true);
      expect(isRemoteModel("local/qwen2.5-32b-instruct")).toBe(false);
      expect(isRemoteModel("embedded/qwen2.5-7b-instruct-q4_k_m")).toBe(false);
      // Exact complements — no id is both, none is neither.
      for (const id of ["anthropic/x", "openrouter/y/z", "local/a", "embedded/b"]) {
        expect(isLocalModel(id)).toBe(!isRemoteModel(id));
      }
      expect(() => isRemoteModel("claude-haiku-4-5")).toThrow(/must be route-prefixed/);
    });

    it("REFUSES an UNCONFIGURED local route rather than falling through to OpenRouter", () => {
      // The failure mode this prevents: `local/…` reaching the OpenRouter branch, being sent to a
      // remote provider with an API key attached, and 404-ing as though the model were unknown.
      const router = createModelRouter({ anthropicApiKey: "test", openRouterApiKey: "test", skipDispatcher: true });
      expect(() => router.resolveModel("local/qwen2.5-32b-instruct")).toThrow(/no OpenAI-compatible server is configured/);
      expect(() => router.resolveModel("embedded/qwen2.5-7b-instruct-q4_k_m")).toThrow(/no in-process weights are configured/);
      // ...while remaining a non-Anthropic route for every other reader.
      expect(router.isAnthropic("embedded/qwen2.5-7b-instruct-q4_k_m")).toBe(false);
    });

    it("serves a `local` model from the configured OpenAI-compatible server", () => {
      const router = createModelRouter({ local: { baseURL: "http://localhost:11434/v1" }, skipDispatcher: true });
      const model = router.resolveModel("local/qwen2.5-32b-instruct");
      expect(model).toBeDefined();
      expect((model as { modelId: string }).modelId).toBe("qwen2.5-32b-instruct");
      expect((model as { provider: string }).provider).toContain("local");
    });

    it("a RESOLVER picks the server per model — two local servers at once", () => {
      // Ollama on :11434 and LM Studio on :1234 is an ordinary setup, and `local/` alone cannot say
      // which one a model means.
      const router = createModelRouter({
        skipDispatcher: true,
        local: (id) =>
          id.startsWith("lms-")
            ? { baseURL: "http://localhost:1234/v1", name: "lmstudio" }
            : { baseURL: "http://localhost:11434/v1", name: "ollama" },
      });
      expect((router.resolveModel("local/qwen2.5-32b") as { provider: string }).provider).toContain("ollama");
      expect((router.resolveModel("local/lms-phi-4") as { provider: string }).provider).toContain("lmstudio");
    });

    it("a resolver returning undefined refuses THAT model by name", () => {
      const router = createModelRouter({ skipDispatcher: true, local: (id) => (id === "known" ? { baseURL: "http://x/v1" } : undefined) });
      expect(router.resolveModel("local/known")).toBeDefined();
      expect(() => router.resolveModel("local/unknown")).toThrow(/no OpenAI-compatible server is configured/);
    });

    it("asks for streamed usage by default", () => {
      // Without `stream_options.include_usage` most OpenAI-compatible servers stream a final chunk with
      // no usage, and every local call reports zero tokens — which a residency planner would read as
      // "this call consumed no context".
      const router = createModelRouter({ local: { baseURL: "http://localhost:11434/v1" }, skipDispatcher: true });
      const model = router.resolveModel("local/qwen2.5-32b") as unknown as { config: { includeUsage?: boolean } };
      expect(model.config.includeUsage).toBe(true);
    });

    it("follows the per-call enforce decision for json_schema vs json_object", () => {
      // The provider reads `supportsStructuredOutputs` as WHICH response_format to send, not whether.
      // Pinning it true would send strict json_schema on every call — the shape Ollama and LM Studio
      // are most likely to reject — even when the profile concluded the schema is advisory.
      const router = createModelRouter({ local: { baseURL: "http://localhost:11434/v1" }, skipDispatcher: true });
      const strict = router.resolveModel("local/m", { strictStructuredOutput: true }) as { supportsStructuredOutputs?: boolean };
      const advisory = router.resolveModel("local/m", { strictStructuredOutput: false }) as { supportsStructuredOutputs?: boolean };
      expect(strict.supportsStructuredOutputs).toBe(true);
      expect(advisory.supportsStructuredOutputs).toBe(false);
      expect((router.resolveModel("local/m") as { supportsStructuredOutputs?: boolean }).supportsStructuredOutputs).toBe(false);
    });

    it("a server config CAPS the enforce decision it cannot honor", () => {
      const router = createModelRouter({ skipDispatcher: true, local: { baseURL: "http://localhost:8080/v1", supportsStructuredOutputs: false } });
      const capped = router.resolveModel("local/m", { strictStructuredOutput: true }) as { supportsStructuredOutputs?: boolean };
      expect(capped.supportsStructuredOutputs).toBe(false); // profile wanted strict; the server cannot
    });

    it("caches one client per SERVER without letting differing configs collide", () => {
      // The bug the cache key guards against: keying on `baseURL` alone would serve every model on one
      // host the first model's flags, so a server declared unable to do json_schema would be sent one.
      const seen: string[] = [];
      const router = createModelRouter({
        skipDispatcher: true,
        local: (id) => {
          seen.push(id);
          return { baseURL: "http://localhost:8080/v1", supportsStructuredOutputs: id !== "plain" };
        },
      });
      const opts = { strictStructuredOutput: true };
      const rich = router.resolveModel("local/rich", opts) as { supportsStructuredOutputs?: boolean };
      const plain = router.resolveModel("local/plain", opts) as { supportsStructuredOutputs?: boolean };
      const richAgain = router.resolveModel("local/rich", opts) as { supportsStructuredOutputs?: boolean };
      expect(seen).toEqual(["rich", "plain", "rich"]); // the resolver is consulted on every resolve
      expect(rich.supportsStructuredOutputs).toBe(true);
      expect(plain.supportsStructuredOutputs).toBe(false); // same host, own answer
      expect(richAgain.supportsStructuredOutputs).toBe(true);
    });
  });
});
