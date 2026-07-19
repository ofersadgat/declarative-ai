/**
 * Provider config-schema registry (§4): each model family has a config JSON Schema,
 * and the LLM-config search space is "pick a provider schema, then fill it".
 *
 * Extracted from findmyprompt `src/engine/registry/providerConfig.ts` MINUS the
 * artifact-minting helpers (`configSchemaArtifact`, `configSchemaId`, `makeConfig`,
 * `providerConfigSeed`), which depended on findmyprompt's content-addressed artifact
 * model (`engine/artifacts` / `engine/model`) and its DB seed pipeline. The pure
 * schema documents and the family lookup are kept; callers that need content ids can
 * hash the schema with `@declarative-ai/core` `hashCanonical`.
 */
import type { ModelFamily } from "./router";

export type { ModelFamily } from "./router";

/** A JSON-Schema document (an untyped JSON object — it IS a schema). */
export type JsonSchemaDoc = Record<string, unknown>;

/**
 * The PROVIDER-NEUTRAL reasoning request (effort level and/or token budget), shared by every family
 * schema — provider divergence (Anthropic thinking-budget vs OpenRouter effort) is resolved on the way
 * OUT by `adaptReasoning` (`./reasoning`), never in the stored config. Mirrors the runtime
 * `ReasoningSpec` (@declarative-ai/core llmConfig).
 */
const REASONING_PROPERTY: JsonSchemaDoc = {
  type: "object",
  properties: {
    effort: { type: "string", enum: ["low", "medium", "high"] },
    budgetTokens: { type: "integer", minimum: 0 },
  },
  additionalProperties: false,
};

/**
 * Anthropic config space. `model` is the routing id (part of candidate identity); `reasoning` is the
 * neutral thinking request. Ranges match the Anthropic API.
 */
export const ANTHROPIC_CONFIG_SCHEMA: JsonSchemaDoc = {
  type: "object",
  title: "anthropic-config",
  properties: {
    model: { type: "string", description: "Anthropic model id, e.g. claude-haiku-4-5" },
    temperature: { type: "number", minimum: 0, maximum: 1 },
    topP: { type: "number", minimum: 0, maximum: 1 },
    topK: { type: "integer", minimum: 0 },
    maxOutputTokens: { type: "integer", minimum: 1 },
    stopSequences: { type: "array", items: { type: "string" } },
    reasoning: REASONING_PROPERTY,
  },
  required: ["model"],
  additionalProperties: false,
};

/**
 * OpenRouter config space — global/credit-based, no TPM; broader sampling knobs and a
 * `seed` (which we deliberately do NOT use for reproducibility, but the provider exposes it).
 */
export const OPENROUTER_CONFIG_SCHEMA: JsonSchemaDoc = {
  type: "object",
  title: "openrouter-config",
  properties: {
    model: { type: "string", description: "OpenRouter model id, e.g. openai/gpt-4.1-mini" },
    temperature: { type: "number", minimum: 0, maximum: 2 },
    topP: { type: "number", minimum: 0, maximum: 1 },
    topK: { type: "integer", minimum: 0 },
    maxOutputTokens: { type: "integer", minimum: 1 },
    frequencyPenalty: { type: "number", minimum: -2, maximum: 2 },
    presencePenalty: { type: "number", minimum: -2, maximum: 2 },
    seed: { type: "integer" },
    stopSequences: { type: "array", items: { type: "string" } },
    reasoning: REASONING_PROPERTY,
  },
  required: ["model"],
  additionalProperties: false,
};

const SCHEMAS: Record<ModelFamily, JsonSchemaDoc> = {
  anthropic: ANTHROPIC_CONFIG_SCHEMA,
  openrouter: OPENROUTER_CONFIG_SCHEMA,
};

/** The JSON-Schema document for a family. */
export function configSchemaFor(family: ModelFamily): JsonSchemaDoc {
  return SCHEMAS[family];
}
