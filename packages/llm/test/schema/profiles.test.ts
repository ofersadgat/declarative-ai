import { describe, expect, it } from "vitest";
import {
  ADVISORY,
  ANTHROPIC_AI_SDK,
  JSON_OBJECT,
  OPENROUTER_STRICT,
  profileForCaps,
  profileForModelId,
} from "../../src/schema/profiles";

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
    expect(profileForModelId("claude-haiku-4-5-20251001").id).toBe(ANTHROPIC_AI_SDK.id);
  });

  it("an unknown OpenRouter model with no catalog row falls back to the provider default (strict)", () => {
    expect(profileForModelId("brand/unseen-model").id).toBe(OPENROUTER_STRICT.id);
  });
});
