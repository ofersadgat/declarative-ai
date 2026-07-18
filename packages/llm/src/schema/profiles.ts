import { defaultModelCatalog } from "../model-catalog";
import { isAnthropicModel } from "../router";
import type { ProviderSchemaProfile } from "./profile";

/**
 * Concrete provider profiles (§5.1) + the model→profile selector. Profiles are keyed by TRANSPORT
 * (provider × SDK) and describe the EFFECTIVE transport's capabilities: where an SDK underneath cleans
 * the schema, the profile declares those capabilities as supported so the adapter leaves them alone.
 * See `./profile` for the field semantics.
 */

/** The string `format` values a strict constrained decoder accepts (OpenAI's list; Anthropic adds `uri`). */
const OPENAI_FORMATS = ["date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"] as const;
const ANTHROPIC_FORMATS = [...OPENAI_FORMATS, "uri"] as const;

/**
 * OpenAI Structured Outputs (`strict: true`). The reference strict dialect: every object closed
 * (`additionalProperties:false`), optional emulated by nullable unions, unions flattened (OpenAI bans
 * `oneOf`/`allOf` and root `anyOf`), `{}` encoded as a JSON string, and the July-2025 size ceilings.
 */
export const OPENAI_STRICT: ProviderSchemaProfile = {
  id: "openai:strict",
  supportsStructuredOutput: "schema",
  optionalSupport: "nullable",
  nullable: "type-array",
  additionalProperties: "force-false",
  unions: "flatten",
  rootUnion: true, // flattening removes root unions, so this never blocks
  rootArray: false, // OpenAI requires an object root (strict AND json mode) — the adapter wraps
  collapseTypeArrays: false, // nullable lives in type arrays — must not be collapsed
  anyType: "encode-json-string",
  refs: "native",
  recursion: true,
  // OpenAI strict allows 10 levels of OBJECT nesting — arrays and the terminal leaf are FREE (a 20-deep
  // array chain is "1 level"); the "5 levels" in older docs is stale. Verified live 2026-06-13 by reading
  // the 400's reported count across object-only / array-only / array-of-object schemas. We count the SAME
  // way (`maxDepthCountStrategy: "objects"`), so the gate is EXACT: nesting arrays inside objects no longer
  // inflates the count and forces a needless advisory fallback. Comfortably covers the meta self-
  // application (object-nesting ~5). NB OPENROUTER_STRICT inherits this; its other upstreams
  // (Google/Anthropic/x-ai/Mistral) have unprobed limits, but a too-deep strict request that 400s is no
  // worse than the advisory fallback (which already tends to fail), and the common upstream is OpenAI.
  maxDepth: 10,
  maxDepthCountStrategy: "objects",
  // A schema STILL past the cap is losslessly object-key-flattened to fit the strict decoder, only
  // falling to advisory if depth is irreducible (deep array-of-leaf-object nesting). Inherited by
  // OPENROUTER_STRICT.
  maxDepthStrategy: "flatten-or-adapt",
  limits: { maxProperties: 5000, maxNameEnumChars: 120000, maxEnumValues: 1000 },
  keywords: {
    not: false,
    default: false,
    uniqueItems: false,
    minProperties: false,
    maxProperties: false,
    patternProperties: false,
    propertyNames: false,
    prefixItems: false,
    contains: false,
    minContains: false,
    maxContains: false,
    unevaluatedItems: false,
    unevaluatedProperties: false,
    format: { support: "yes", allowedValues: OPENAI_FORMATS },
  },
};

/**
 * OpenRouter (the catch-all transport). OpenRouter cleans nothing and normalizes every upstream to an
 * OpenAI-compatible `response_format.json_schema` interface — so the dialect IS OpenAI-strict regardless
 * of the routed model. Conservative depth/size ceilings (OpenAI's) keep us from 400-ing an OpenAI
 * upstream; non-OpenAI upstreams just fall back to advisory a little sooner. Per-upstream `limits`/
 * `supportsStructuredOutput` overrides can refine this later.
 */
export const OPENROUTER_STRICT: ProviderSchemaProfile = {
  ...OPENAI_STRICT,
  id: "openrouter:strict",
};

/**
 * Anthropic via `@ai-sdk/anthropic` (≥3.0.73). The SDK sanitizes the schema on its native path
 * (`additionalProperties:false`, strip-constraints-to-description, `oneOf`→`anyOf`, format whitelist),
 * so we model that as the transport NATIVELY supporting those: `additionalProperties:"leave"` (the SDK
 * adds it), keep unions as `anyOf` (the SDK does, no reconstruction → identity post-process), and strip
 * NO keywords (the SDK strips/describes them). The adapter thus does a near-identity transform. What the
 * SDK + Anthropic genuinely CANNOT do — represent `{}`, recursion, a root union — still forces advisory.
 *
 * Enforcement on the wire: for Claude 4.5+ models the SDK (default `structuredOutputMode: "auto"`)
 * sends Anthropic's NATIVE structured outputs (`output_config.format` json_schema) — real constrained
 * decoding, so `enforce: "strict"` here is honest. Pre-4.5 models fall back to the jsonTool emulation
 * (unconstrained); the §4 Ajv boundary is the gate there.
 */
export const ANTHROPIC_AI_SDK: ProviderSchemaProfile = {
  id: "anthropic:ai-sdk",
  supportsStructuredOutput: "schema",
  optionalSupport: "omit",
  nullable: "type-array",
  additionalProperties: "leave", // the SDK adds additionalProperties:false itself
  unions: "anyOf", // the SDK keeps anyOf (rewrites oneOf→anyOf); we don't flatten
  rootUnion: false,
  // Conservative: the SDK's pre-4.5 fallback (jsonTool input_schema) is object-root-only, and the
  // structured-outputs docs show only object roots for `output_config.format` (array roots are
  // undocumented/unverified live). Flip to true if a live test confirms root arrays are accepted.
  rootArray: false,
  collapseTypeArrays: false,
  anyType: "native",
  refs: "native",
  recursion: false,
  keywords: {}, // strip nothing — the SDK owns keyword handling on this transport
};

/**
 * Anthropic against the RAW Messages API (no SDK sanitizer in front). The full manual mirror of what
 * `@ai-sdk/anthropic` does internally — kept for direct-API use and as an A/B against `anthropic:ai-sdk`.
 * Constraints are moved to `description` (Anthropic's strip-and-describe), `minItems` is whitelisted to
 * {0,1}, unions are flattened, and `format` is filtered to the Anthropic list.
 */
export const ANTHROPIC_RAW: ProviderSchemaProfile = {
  id: "anthropic:raw",
  supportsStructuredOutput: "schema",
  optionalSupport: "omit",
  nullable: "type-array",
  additionalProperties: "force-false",
  unions: "flatten",
  rootUnion: true,
  rootArray: false, // raw Anthropic tool input_schema — object root only
  collapseTypeArrays: false,
  anyType: "encode-json-string",
  refs: "native",
  recursion: false,
  keywords: {
    minimum: "describe",
    maximum: "describe",
    exclusiveMinimum: "describe",
    exclusiveMaximum: "describe",
    multipleOf: "describe",
    minLength: "describe",
    maxLength: "describe",
    pattern: "describe",
    maxItems: "describe",
    uniqueItems: "describe",
    minProperties: "describe",
    maxProperties: "describe",
    not: "describe",
    minItems: { support: "describe", allowedValues: [0, 1] },
    format: { support: "describe", allowedValues: ANTHROPIC_FORMATS },
  },
};

/**
 * The JSON-OBJECT tier (`supportsStructuredOutput:"object"`) — the transport can force SOME JSON out
 * (`response_format:{type:"json_object"}`) but does NOT bind the grammar to a schema, so the schema
 * rides as an advisory hint and the §4 Ajv boundary is the gate. Field policies mirror {@link ADVISORY}
 * EXCEPT `rootArray:false` (json_object can only emit an OBJECT root, so a root array is wrapped) and
 * `promptRequiresJSONSpecifier:"force"` (OpenAI-compatible json_object upstreams — Alibaba/DashScope,
 * OpenAI — 400 unless the messages contain the word "json"; we inject it when the prompt lacks one).
 */
export const JSON_OBJECT: ProviderSchemaProfile = {
  id: "openrouter:json-object",
  supportsStructuredOutput: "object",
  promptRequiresJSONSpecifier: "force",
  optionalSupport: "omit",
  nullable: "type-array",
  additionalProperties: "leave",
  unions: "flatten",
  rootUnion: true,
  rootArray: false, // json_object emits an object root — a root array is wrapped ({ items: [...] })
  collapseTypeArrays: false,
  anyType: "native",
  refs: "native",
  recursion: true,
  keywords: {},
};

/**
 * The TEXT floor (`supportsStructuredOutput:false`) — the model supports NEITHER schema-constrained
 * decoding NOR json_object mode. The call is a PLAIN text completion (no `response_format`); the schema
 * is described in the prompt and the JSON is parsed out of the returned text (Ajv is the only gate). The
 * conservative choice for a model whose `supported_parameters` list neither `structured_outputs` nor
 * `response_format`. `rootArray:true` — nothing constrains the shape, so the hint keeps the true shape.
 */
export const ADVISORY: ProviderSchemaProfile = {
  id: "advisory",
  supportsStructuredOutput: false,
  optionalSupport: "omit",
  nullable: "type-array",
  additionalProperties: "leave",
  unions: "flatten",
  rootUnion: true,
  rootArray: true, // no decoder in play — the hint keeps the true (root-array) shape
  collapseTypeArrays: false,
  anyType: "native",
  refs: "native",
  recursion: true,
  keywords: {},
};

/**
 * Code-resident base profiles, keyed by their stable `id`. The single source of truth the migration
 * seed (`db/genSchemaProfilesSeedSql.ts`) materializes into the `schema_profiles` TABLE, and the id a
 * model's `schema_profiles`-FK references. The runtime reads the RESOLVED profile back off the catalog
 * (hydrated from the table); this registry is the code→row bridge + the pure-engine fallback.
 */
export const PROFILE_REGISTRY: Record<string, ProviderSchemaProfile> = {
  [OPENAI_STRICT.id]: OPENAI_STRICT,
  [OPENROUTER_STRICT.id]: OPENROUTER_STRICT,
  [JSON_OBJECT.id]: JSON_OBJECT,
  [ANTHROPIC_AI_SDK.id]: ANTHROPIC_AI_SDK,
  [ANTHROPIC_RAW.id]: ANTHROPIC_RAW,
  [ADVISORY.id]: ADVISORY,
};

/**
 * The provider→default-profile fallback, for a model whose `supported_parameters` carry no
 * structured-output signal (native Anthropic — the docs scrape lists Anthropic-native param names, not
 * OpenRouter's `structured_outputs`/`response_format` — and any not-yet-refreshed OpenRouter row). The
 * DB source of truth is the migration-seeded `runtime_config` (`schema_profile:default:<provider>`);
 * this constant is the identical code-resident mirror for the pure engine + the seed generator.
 * OpenRouter defaults to STRICT: it normalizes to json_schema and `require_parameters` routes to a
 * capable upstream (the overwhelming majority advertise `structured_outputs`); a rare incapable route
 * 400s no worse than an advisory hint would have.
 */
export const PROVIDER_DEFAULT_PROFILE_ID: Record<string, string> = {
  anthropic: ANTHROPIC_AI_SDK.id,
  openrouter: OPENROUTER_STRICT.id,
};

/**
 * Map a model's OpenRouter `supported_parameters` to its structured-output profile (§5.1) — the single
 * derivation shared by import (`persistModels`), the seed (`genModelsSeedSql`), and the pure-engine
 * fallback (`profileForModelId`). Native Anthropic wins first (its SDK carries structured output the
 * OpenRouter param names don't describe). Then: `structured_outputs`→strict json_schema,
 * `response_format`→json_object, a KNOWN list with neither→the text floor, and an UNKNOWN/absent list→
 * the provider default (so a not-yet-refreshed row degrades gracefully, not to the text floor).
 */
export function profileForCaps(modelId: string, supportedParameters: string[] | undefined): ProviderSchemaProfile {
  if (isAnthropicModel(modelId)) return ANTHROPIC_AI_SDK;
  if (supportedParameters === undefined) {
    const defaultId = PROVIDER_DEFAULT_PROFILE_ID.openrouter;
    return (defaultId ? PROFILE_REGISTRY[defaultId] : undefined) ?? OPENROUTER_STRICT;
  }
  if (supportedParameters.includes("structured_outputs")) return OPENROUTER_STRICT;
  if (supportedParameters.includes("response_format")) return JSON_OBJECT;
  return ADVISORY; // known caps, neither signal → text floor
}

/** The profile id {@link profileForCaps} resolves to — what import/seed write into
 *  `models.schema_profile_id`. */
export function profileIdForCaps(modelId: string, supportedParameters: string[] | undefined): string {
  return profileForCaps(modelId, supportedParameters).id;
}

/**
 * Resolve the profile for a model id at CALL time. The CATALOG wins when it has a recorded
 * (already-resolved, table-hydrated) profile for the model (§5.1) — the data-driven path. Absent that,
 * fall back to the capability derivation over the catalog's `supported_parameters` (which itself
 * handles native Anthropic + the provider default). Keyed by id alone because the executor (the single
 * adaptation site, §6.1) must adapt the schema and derive the wire mode BEFORE it resolves the model
 * object. An Anthropic model reached via OpenRouter has a non-`claude-` id + `structured_outputs` in
 * its caps, so it correctly resolves to the OpenRouter strict dialect, not the native SDK profile.
 */
export function profileForModelId(modelId: string): ProviderSchemaProfile {
  const recorded = defaultModelCatalog.schemaProfileFor(modelId);
  if (recorded) return recorded;
  return profileForCaps(modelId, defaultModelCatalog.supportedParametersFor(modelId));
}
