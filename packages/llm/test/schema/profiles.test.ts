import { describe, expect, it } from "vitest";
import {
  ADVISORY,
  ANTHROPIC_AI_SDK,
  JSON_OBJECT,
  LLAMACPP_GRAMMAR,
  LOCAL_JSON_OBJECT,
  OPENROUTER_STRICT,
  PROFILE_REGISTRY,
  PROVIDER_DEFAULT_PROFILE_ID,
  profileForCaps,
  profileForModelId,
} from "../../src/schema/profiles.js";

describe("profileForCaps — capability-derived structured-output profile", () => {
  it("native claude-* → the SDK-deferring Anthropic profile, regardless of caps", () => {
    expect(profileForCaps("claude-haiku-4-5-20251001", undefined).id).toBe(ANTHROPIC_AI_SDK.id);
    // Native caps use Anthropic param names (no structured_outputs/response_format) — still the SDK profile,
    // NOT the text floor (the anthropic check wins before the caps derivation).
    expect(profileForCaps("claude-opus-4-8", ["max_tokens", "tools", "reasoning"]).id).toBe(ANTHROPIC_AI_SDK.id);
  });

  it("structured_outputs in supported_parameters → strict json_schema", () => {
    // The regression fix: qwen/llama/z-ai advertise structured_outputs, so they get STRICT — not the
    // advisory json_object floor the old family whitelist mislabeled them with (→ Alibaba json_object 400).
    expect(profileForCaps("qwen/qwen3.7-max", ["temperature", "response_format", "structured_outputs"]).id).toBe(OPENROUTER_STRICT.id);
    expect(profileForCaps("meta-llama/llama-4-maverick", ["structured_outputs"]).id).toBe(OPENROUTER_STRICT.id);
    expect(profileForCaps("z-ai/glm-4.7-flash", ["response_format", "structured_outputs"]).id).toBe(OPENROUTER_STRICT.id);
  });

  it("response_format but no structured_outputs → json_object (object tier)", () => {
    expect(profileForCaps("some/json-object-model", ["temperature", "response_format"]).id).toBe(JSON_OBJECT.id);
  });

  it("KNOWN caps with neither signal → the text floor (plain-text completion)", () => {
    expect(profileForCaps("some/text-only-model", ["temperature", "max_tokens"]).id).toBe(ADVISORY.id);
  });

  it("UNKNOWN caps (row not yet refreshed) → the openrouter provider default (strict), not the text floor", () => {
    expect(profileForCaps("brand/new-model", undefined).id).toBe(OPENROUTER_STRICT.id);
  });
});

describe("profileForModelId — catalog-first, else capability fallback", () => {
  it("routes claude-* to the Anthropic SDK profile", () => {
    // profileForModelId takes the full `{route}/{model}` id; it strips the route for the family heuristic.
    expect(profileForModelId("anthropic/claude-haiku-4-5-20251001").id).toBe(ANTHROPIC_AI_SDK.id);
  });

  it("an unknown OpenRouter model with no catalog row falls back to the provider default (strict)", () => {
    expect(profileForModelId("openrouter/brand/unseen-model").id).toBe(OPENROUTER_STRICT.id);
  });

  describe("locally-served routes", () => {
    it("resolves by ROUTE, never through the OpenRouter capability derivation", () => {
      // The trap: a local model has no `supported_parameters` (no registry publishes them for a GGUF),
      // so falling into `profileForCaps` would hit its unknown-caps arm and hand a GGUF on your desk
      // the OpenAI strict dialect — 10-deep nesting cap, 5000-property cap, "json" prompt specifier.
      expect(profileForModelId("local/qwen2.5-32b-instruct").id).toBe(LOCAL_JSON_OBJECT.id);
      expect(profileForModelId("embedded/qwen2.5-7b-instruct-q4_k_m").id).toBe(LLAMACPP_GRAMMAR.id);
      expect(profileForModelId("local/qwen2.5-32b-instruct").id).not.toBe(OPENROUTER_STRICT.id);
    });

    it("a local model whose NAME looks like claude does not get the Anthropic profile", () => {
      // `profileForCaps` keys the Anthropic branch off a bare `claude-` prefix. Route wins over the
      // name, or a locally-served distill would be sent through the Anthropic SDK's dialect.
      expect(profileForModelId("embedded/claude-ish-finetune-q4_k_m").id).toBe(LLAMACPP_GRAMMAR.id);
    });

    it("the grammar profile claims STRICT enforcement and no decoder ceilings", () => {
      // A GBNF grammar makes an off-schema token unrepresentable, so unlike every advisory tier this
      // one genuinely constrains — and unlike the strict tiers it has no size/nesting limits to trip.
      expect(LLAMACPP_GRAMMAR.supportsStructuredOutput).toBe("schema");
      expect(LLAMACPP_GRAMMAR.rootArray).toBe(true); // no forced object root
      expect(LLAMACPP_GRAMMAR.rootUnion).toBe(true);
      expect(LLAMACPP_GRAMMAR.unions).toBe("anyOf"); // alternation IS a grammar rule
      expect(LLAMACPP_GRAMMAR.maxDepth).toBeUndefined();
      expect(LLAMACPP_GRAMMAR.limits).toBeUndefined();
    });

    it("the local-server profile is the object tier, and can emit only an object root", () => {
      expect(LOCAL_JSON_OBJECT.supportsStructuredOutput).toBe("object");
      expect(LOCAL_JSON_OBJECT.rootArray).toBe(false); // json_object mode has no array-root form
      // ...unlike the grammar profile, whose start rule can be anything.
      expect(LLAMACPP_GRAMMAR.rootArray).toBe(true);
    });

    it("both profiles are registered, so a catalog row can reference them by id", () => {
      expect(PROFILE_REGISTRY[LOCAL_JSON_OBJECT.id]).toBe(LOCAL_JSON_OBJECT);
      expect(PROFILE_REGISTRY[LLAMACPP_GRAMMAR.id]).toBe(LLAMACPP_GRAMMAR);
      expect(PROVIDER_DEFAULT_PROFILE_ID.local).toBe(LOCAL_JSON_OBJECT.id);
      expect(PROVIDER_DEFAULT_PROFILE_ID.embedded).toBe(LLAMACPP_GRAMMAR.id);
    });
  });
});
