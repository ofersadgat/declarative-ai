/**
 * The per-model CATALOG (§5). Originally just a price table; now the single source of everything
 * the runtime knows about a model — cost, capabilities (which parameters it accepts / requires),
 * limits, and its structured-output schema profile. Prices change, providers get added, and models
 * get re-priced or re-capability'd, so this is an **updatable table**, not a frozen constant:
 * {@link ModelInfo} supports `upsert`/`remove`/`load`, and is designed to be hydrated/overridden at
 * startup (e.g. from a `models` store) without a code change. The in-code `DEFAULT_MODELS` is the seed.
 *
 * Identity follows the project's `{route}/{model}` structure (see {@link ParsedModel} in router.ts):
 * every row carries an explicit `route` ("anthropic" | "openrouter") and provider-native `model`, and
 * the catalog KEY is the combined `${route}/${model}` — the exact full id a caller already holds as
 * `def.model`. Matching is EXACT (a plain `Map` lookup), so `computeCost` et al. resolve a model iff its
 * `${route}/${model}` id was loaded. There is no longest-prefix fallback: a dated/variant id must be
 * present as its own row.
 *
 * {@link ModelInfo} is GENERIC over its seed rows: construct it with a literal array of rows and its
 * methods are typed to the exact set of `${route}/${model}` keys (an unknown model fails to COMPILE);
 * construct it with a plain `ModelInfoInterface[]` (or read the runtime-hydrated `ModelInfo.instance`)
 * and the methods accept any `string`. Strong seed → strong methods; weak seed → weak methods.
 *
 * PACKAGING: this module is dependency-FREE (its runtime imports are the generated seed data and the
 * effort ORDER from `llmConfig`, which itself imports nothing at runtime; the router/schema/json
 * imports are type-only) and is published as its own `@declarative-ai/llm/model-catalog`
 * subpath for exactly that reason. Model *identity*, *pricing* and *capabilities* are the parts a UI
 * legitimately needs, and reaching them through the package barrel would drag `call`→`router`→
 * `dispatcher` — hence `undici`/`node:net` and the AI SDK providers — into a browser bundle. Same
 * un-barrelling rationale as the ajv note in index.ts: keep the module graph honest, not just the API.
 */
import type { JsonValue, SchemaDocument } from "@declarative-ai/json";
import { GENERATED_MODELS } from "./model-catalog-data.generated.js";
import { REASONING_EFFORTS, type ReasoningEffort, type ReasoningSpec } from "./llmConfig.js";
import type { ModelRoute } from "./router.js";
import type { ProviderSchemaProfile } from "./schema/profile.js";

/**
 * The four rate dimensions that fully determine an Anthropic call's cost. The cache rates
 * default to multiples of the base input rate (Anthropic keeps the same ratios across tiers:
 * read 0.1x, write-5m 1.25x, write-1h 2x), so a row usually sets only input/output.
 */
export interface RateSet {
  inputPerMillion: number;
  outputPerMillion: number;
  /** Cached-read rate; defaults to `inputPerMillion * cacheReadMultiplier` (≈0.1x). */
  cacheReadPerMillion?: number;
  /** 5-minute cache-write rate; defaults to `inputPerMillion * cacheWriteMultiplier` (≈1.25x). */
  cacheWritePerMillion?: number;
  /** 1-hour cache-write rate; defaults to `inputPerMillion * cacheWrite1hMultiplier` (≈2x). */
  cacheWrite1hPerMillion?: number;
}

/** Input/output modalities a model accepts/produces (OpenRouter `architecture.*_modalities`). Arrays are
 *  `readonly` so a generated `as const` snapshot (see `DEFAULT_MODELS`) is assignable to this shape. */
export interface Modalities {
  input?: readonly string[];
  output?: readonly string[];
}

/**
 * Everything the runtime knows about ONE model. The rate fields ({@link RateSet}) drive cost; the
 * capability fields drive routing and the structured-output decision. Identity is `route` + `model`
 * (the project's `{route}/{model}` structure), and the catalog keys on the combined `${route}/${model}`.
 * Every non-rate, non-identity field is OPTIONAL — a row that only knows a price still works (cost
 * computes; capabilities fall back to heuristics), and the §5 refresh fills the rest in over time.
 *
 * (Was `ModelInfo`; renamed so the `ModelInfo` name is free for the catalog CLASS below.)
 */
export interface ModelInfoInterface extends RateSet {
  /**
   * Serving ROUTE — the route the call takes: "anthropic" (native Anthropic API) or "openrouter". This
   * is what distinguishes an OpenRouter-served Opus row from a Claude-API-served one (they carry
   * different prices). Same set as {@link ModelRoute} / the router's `{route}/…` prefix.
   */
  route: ModelRoute;
  /**
   * Provider-native model id (the part after the `{route}/` prefix) — e.g. `claude-opus-4-8` on the
   * anthropic route, or `openai/gpt-5` / `anthropic/claude-opus-4.8` on the openrouter route. Together
   * with {@link route} it forms the catalog key `${route}/${model}`.
   */
  model: string;
  /**
   * Provider-reported model creation/release time, UNIX SECONDS, when the source exposes one
   * (OpenRouter's `created`; the Anthropic docs scrape has none). Used to sort models newest-first;
   * never affects cost. Absent ⇒ unknown (the UI falls back to a version-number heuristic).
   */
  releasedAt?: number;
  /**
   * Premium pricing for requests whose total input exceeds `thresholdTokens` (Anthropic's
   * long-context tier). When the request crosses the threshold, ALL of its tokens are priced at
   * this set instead. Omit for models without a long-context tier (then base rates always apply).
   */
  longContext?: RateSet & { thresholdTokens: number };

  // --- Identity / display ---------------------------------------------------
  /** Display vendor ("OpenAI", "Anthropic", …) — the row label in the picker grid. Derivable from `model`. */
  provider?: string;
  /** Short chip label (the `model` with any `vendor/` prefix dropped). */
  label?: string;
  /**
   * Provider-NEUTRAL model id, the SAME across every serving route for one underlying model — so the
   * native `claude-opus-4-8` row and the OpenRouter `anthropic/claude-opus-4.8` row share
   * `canonicalId: "claude-opus-4-8"`. Derived by {@link canonicalIdFor} (drop any `vendor/` prefix,
   * normalize dots→hyphens). Lets the picker collapse the same model's routes and lets capability data
   * be reconciled across them. Absent ⇒ derive from `model`.
   */
  canonicalId?: string;

  // --- Capabilities / limits ------------------------------------------------
  /** Max context window in tokens (OpenRouter `context_length`). */
  contextLength?: number;
  /** Max output/completion tokens the provider allows (OpenRouter `top_provider.max_completion_tokens`). */
  maxOutputTokens?: number;
  /** Input/output modalities (OpenRouter `architecture.input/output_modalities`). */
  modalities?: Modalities;
  /**
   * What the model may be SENT, and what each value may be — a JSON Schema over the call's model
   * parameters ({@link MODEL_PARAMETER_KEYS}: `LlmCallConfig`'s own field names, never a wire name).
   *
   *  - A listed property is accepted, and its schema is the values it takes — `reasoning.effort`'s
   *    `enum` is the levels the model offers.
   *  - An unlisted one is not (`additionalProperties: false`). The executor FILTERS what it sends down
   *    to this, so a param no endpoint accepts (`temperature` on a reasoning model) is never sent —
   *    which keeps OpenRouter's `require_parameters` routing from rejecting the call (§5.1).
   *  - A value schema carrying no constraint (`{}`, or suggestions only in `examples`) is accepted
   *    with its values unknown — all a source that lists NAMES can say.
   *  - `required` names what the model must be sent.
   *
   * Absent ⇒ unknown: the model accepts everything (see the reasoning-family fallback in
   * {@link ModelInfo.parameters}). Built on the way in from each source's own vocabulary
   * ({@link parametersFromNames}). JaiRA decision 0009.
   */
  parameters?: SchemaDocument;
  /**
   * The structured-output tier the model's endpoint offers: constrained decoding against a schema,
   * JSON-object mode only, or neither. What `profileForCaps` derives a schema profile from when the
   * row records none. Absent ⇒ unknown (the route's default profile).
   */
  structuredOutput?: "schema" | "object" | false;
  /**
   * The RESOLVED structured-output schema profile (§5.1) for this model, resolved to a whole object at
   * hydrate time so the engine reads it synchronously. Absent ⇒ the static `profileForModelId` family
   * fallback applies.
   */
  schemaProfile?: ProviderSchemaProfile;
  /** False ⇒ retired/unavailable; hidden from the model picker. Absent ⇒ assumed available. */
  available?: boolean;
  /** Provenance — which source last wrote this row ("openrouter-models" | "anthropic-docs" | "seed"). */
  source?: string;

  // --- Weights (locally-served models) --------------------------------------
  /**
   * DESCRIPTOR: the model's weights are published under an open license. A property of the model, and
   * true of plenty of rows served only remotely (a Llama on OpenRouter is still open-weight).
   *
   * Deliberately NOT a claim that we can run it: openness is about the license, {@link downloads} is
   * about whether a servable artifact exists. A model can be open-weight with no GGUF anyone has
   * published, and a row can carry weights under a license that forbids redistribution.
   */
  openWeights?: boolean;
  /**
   * Quantization of THIS row's weights (`Q4_K_M`, `Q8_0`, `F16`, …).
   *
   * It is row-level rather than a variant list because quantization changes everything the runtime
   * keys on — footprint, quality, and how much context fits beside it — so each quant is its own
   * `${route}/${model}` id, collapsed back together for a picker by {@link canonicalId}.
   */
  quantization?: string;
  /**
   * Resident size of the weights in MEBIBYTES, for the residency planner.
   *
   * The weights alone: the KV cache is NOT included, because it is a function of the context size and
   * concurrency a CALL asks for rather than of the model. A planner adds the two.
   */
  weightsMb?: number;
  /** Where the weights can be fetched from. Several entries are MIRRORS of this row's one quant, not
   *  alternative quants — those are separate rows. Empty/absent ⇒ nothing to download (bring your own
   *  file). */
  downloads?: readonly WeightsLocation[];
}

/** One place a row's weights can be obtained from. */
export interface WeightsLocation {
  /** How to reach it: a HuggingFace repo reference, a plain URL, or a path already on this machine. */
  source: "hf" | "url" | "file";
  /**
   * The reference itself — `<org>/<repo>/<file.gguf>` for `hf`, an absolute URL for `url`, a
   * filesystem path for `file`. For a SPLIT model this names the FIRST part (see {@link parts}).
   */
  uri: string;
  /** Total bytes across every part, when the source publishes it. */
  sizeBytes?: number;
  /** Integrity check for {@link uri}, when the source publishes one. */
  sha256?: string;
  /**
   * The remaining parts of a SPLIT model, in order.
   *
   * Large GGUFs ship as `…-00001-of-00009.gguf` sets and every part is required — a downloader that
   * fetches only `uri` gets a file that looks complete and loads to an error. Absent ⇒ single file.
   */
  parts?: readonly string[];
  /** The source requires accepted terms plus a credential (most Llama/Gemma repos). A downloader with
   *  no token should fail NAMING this rather than reporting the 401 as a missing file. */
  gated?: boolean;
}

/**
 * A token breakdown for cost. `inputTokens`/`outputTokens` are the cache-INCLUSIVE totals
 * the AI SDK reports; the optional split makes cost billing-accurate under prompt caching —
 * each bucket is priced at its own rate. `cacheWrite1hTokens` (the 1-hour-TTL subset of
 * `cacheWriteTokens`, read from the provider `raw` usage) is priced at the 2x tier; the rest
 * of the writes are 5-minute. When no split is present, the flat input total is priced flat.
 */
export interface UsageForCost {
  inputTokens?: number | null;
  outputTokens?: number | null;
  noCacheTokens?: number | null;
  cacheReadTokens?: number | null;
  cacheWriteTokens?: number | null;
  /** 1-hour-TTL subset of `cacheWriteTokens` (Anthropic `cache_creation.ephemeral_1h_input_tokens`). */
  cacheWrite1hTokens?: number | null;
}

export interface PricingOptions {
  /** Default cache-read rate as a fraction of base input (Anthropic ≈ 0.1). */
  cacheReadMultiplier?: number;
  /** Default 5-minute cache-write rate as a fraction of base input (Anthropic ≈ 1.25). */
  cacheWriteMultiplier?: number;
  /** Default 1-hour cache-write rate as a fraction of base input (Anthropic ≈ 2.0). */
  cacheWrite1hMultiplier?: number;
}

/** The catalog key for a row — the project's `{route}/{model}` id. */
export function keyForModel(row: Pick<ModelInfoInterface, "route" | "model">): string {
  return `${row.route}/${row.model}`;
}

/** Drop any `vendor/` prefix from a provider-native `model` id (e.g. `openai/gpt-5` → `gpt-5`). */
function bareModel(model: string): string {
  return model.includes("/") ? model.slice(model.lastIndexOf("/") + 1) : model;
}

/**
 * Heuristic: OpenAI reasoning families (GPT-5*, o1/o3/o4*) reject sampling params (`temperature`,
 * `top_p`, `top_k`) — no OpenRouter endpoint lists them as supported. Used ONLY as a cold-start
 * fallback for {@link ModelInfo.parameters} when the §5 refresh hasn't yet recorded the model's real
 * `parameters`; the recorded data always wins when present. Accepts either a
 * bare `model` id or a full `${route}/${model}` key (the bare model segment is what's matched).
 */
export function isReasoningModel(modelId: string): boolean {
  const bare = bareModel(modelId).toLowerCase();
  return /^gpt-5/.test(bare) || /^o[1-9]/.test(bare);
}

/** Provider-NEUTRAL canonical id for a `model` id (drop any `vendor/` prefix, dots→hyphens) — the key
 *  that collapses the same model's native + OpenRouter routes (see {@link ModelInfoInterface.canonicalId}). */
export function canonicalIdFor(model: string): string {
  return bareModel(model).toLowerCase().replace(/\./g, "-");
}

/**
 * Fill the DERIVED IDENTITY fields onto a row — {@link ModelInfoInterface.canonicalId}, and (so a bare
 * seed row is self-describing) `provider` / `label`. Fill-only: values a parser set explicitly are kept.
 * This is what an OpenRouter row needs (its capabilities come from the feed).
 */
export function deriveIdentity(row: ModelInfoInterface): ModelInfoInterface {
  return {
    ...row,
    canonicalId: row.canonicalId ?? canonicalIdFor(row.model),
    provider: row.provider ?? displayProviderFor(row.model),
    label: row.label ?? bareModel(row.model),
  };
}

/** OpenRouter/native vendor slug → display name ("meta-llama" → "Meta", "x-ai" → "xAI"). */
const VENDOR_NAMES: Record<string, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  google: "Google",
  "meta-llama": "Meta",
  meta: "Meta",
  deepseek: "DeepSeek",
  amazon: "Amazon",
  mistralai: "Mistral",
  mistral: "Mistral",
  "x-ai": "xAI",
  qwen: "Qwen",
  cohere: "Cohere",
};

/**
 * Display vendor for a provider-native `model` id — the row label in the picker grid. A `vendor/model`
 * id resolves by its vendor slug; a bare id is matched by a known family prefix. Unknown ⇒ "Other", so a
 * newly-priced family still shows up before this table learns about it.
 */
export function displayProviderFor(model: string): string {
  if (model.includes("/")) {
    const vendor = model.slice(0, model.indexOf("/")).toLowerCase();
    return VENDOR_NAMES[vendor] ?? vendor.charAt(0).toUpperCase() + vendor.slice(1);
  }
  if (model.startsWith("claude")) return "Anthropic";
  if (model.startsWith("gpt") || model.startsWith("o3") || model.startsWith("o4")) return "OpenAI";
  if (model.startsWith("gemini") || model.startsWith("palm")) return "Google";
  if (model.startsWith("llama")) return "Meta";
  if (model.startsWith("mistral") || model.startsWith("mixtral") || model.startsWith("magistral")) return "Mistral";
  if (model.startsWith("deepseek")) return "DeepSeek";
  if (model.startsWith("grok")) return "xAI";
  if (model.startsWith("nova")) return "Amazon";
  return "Other";
}

// --- Parameters: what a model may be sent ------------------------------------

/** The call fields a row's `parameters` schema speaks about — `LlmCallConfig`'s model parameters. */
export const MODEL_PARAMETER_KEYS = [
  "maxOutputTokens",
  "temperature",
  "topP",
  "topK",
  "stopSequences",
  "presencePenalty",
  "frequencyPenalty",
  "seed",
  "tools",
  "toolChoice",
  "reasoning",
] as const;

/** One of {@link MODEL_PARAMETER_KEYS}. */
export type ModelParameterKey = (typeof MODEL_PARAMETER_KEYS)[number];

/** The decoding knobs among them — the ones the executor filters a call's settings down to. */
export const SAMPLING_PARAMETER_KEYS = ["temperature", "topP", "topK", "stopSequences", "presencePenalty", "frequencyPenalty", "seed"] as const;

/** One of {@link SAMPLING_PARAMETER_KEYS}. */
export type SamplingParameterKey = (typeof SAMPLING_PARAMETER_KEYS)[number];

/**
 * A source's parameter NAME → the call field it is. OpenRouter's snake_case vocabulary, which is also
 * OpenAI's; a name not here is not a call parameter (`response_format`, `structured_outputs` and
 * `verbosity` are transport facts or knobs the config does not carry) and is left out of the schema.
 */
export const WIRE_PARAMETER_NAMES: Readonly<Record<string, ModelParameterKey>> = {
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK",
  stop: "stopSequences",
  presence_penalty: "presencePenalty",
  frequency_penalty: "frequencyPenalty",
  seed: "seed",
  tools: "tools",
  tool_choice: "toolChoice",
  reasoning: "reasoning",
  reasoning_effort: "reasoning",
  include_reasoning: "reasoning",
};

/** The value schema a parameter gets when a source names it and says nothing more about its values. */
const VALUE_SCHEMAS: Readonly<Record<Exclude<ModelParameterKey, "reasoning">, JsonValue>> = {
  maxOutputTokens: { type: "integer", minimum: 1 },
  temperature: { type: "number", minimum: 0 },
  topP: { type: "number", minimum: 0, maximum: 1 },
  topK: { type: "integer", minimum: 1 },
  stopSequences: { type: "array", items: { type: "string" } },
  presencePenalty: { type: "number" },
  frequencyPenalty: { type: "number" },
  seed: { type: "integer" },
  tools: {},
  toolChoice: {},
};

/** What a source says about a model's reasoning — the input to {@link parametersFromNames}. */
export interface ReasoningCapability {
  /** The levels it takes, in any order. Absent ⇒ unknown: any level may be sent. */
  efforts?: readonly ReasoningEffort[];
  /** The level it thinks at when none is asked for. */
  defaultEffort?: ReasoningEffort;
  /** Whether it takes a thinking budget in tokens, and in what range. Absent ⇒ it does (the
   *  OpenRouter reading: a model that lists `reasoning` takes `reasoning.max_tokens`). */
  budget?: boolean | { minimum?: number; maximum?: number };
  /** It must be sent a reasoning request. */
  mandatory?: boolean;
}

/**
 * Build a row's `parameters` schema from a source's parameter NAMES plus whatever it says about values.
 *
 * The ONE builder every ingestion path goes through, so a row from OpenRouter, from Anthropic's model
 * list or from an agent's own report reads the same way. `reasoning` is described when the names list it
 * OR a capability is given (an agent reports levels without a names list).
 */
export function parametersFromNames(
  names: readonly string[],
  opts: { reasoning?: ReasoningCapability; maxOutputTokens?: number } = {},
): SchemaDocument {
  const properties: Record<string, JsonValue> = {};
  let reasons = opts.reasoning !== undefined;
  for (const name of names) {
    const key = WIRE_PARAMETER_NAMES[name];
    if (key === undefined) continue;
    if (key === "reasoning") reasons = true;
    else properties[key] ??= VALUE_SCHEMAS[key];
  }
  if (opts.maxOutputTokens !== undefined && opts.maxOutputTokens > 0) {
    properties["maxOutputTokens"] = { type: "integer", minimum: 1, maximum: opts.maxOutputTokens };
  }
  if (reasons) properties["reasoning"] = reasoningSchema(opts.reasoning);
  return {
    type: "object",
    additionalProperties: false,
    properties,
    ...(opts.reasoning?.mandatory === true ? { required: ["reasoning"] } : {}),
  };
}

/** The `reasoning` property's schema — see {@link ReasoningCapability}. */
function reasoningSchema(cap: ReasoningCapability | undefined): JsonValue {
  const properties: Record<string, JsonValue> = {};
  if (cap?.efforts === undefined) {
    // Accepted, levels unknown: a SUGGESTION list rather than an `enum`, so a form offers the usual
    // levels without the schema claiming the model takes every one of them.
    properties["effort"] = { type: "string", examples: [...REASONING_EFFORTS] };
  } else if (cap.efforts.length > 0) {
    properties["effort"] = {
      enum: orderedEfforts(cap.efforts),
      ...(cap.defaultEffort !== undefined ? { default: cap.defaultEffort } : {}),
    };
  }
  if (cap?.budget !== false) {
    const range = typeof cap?.budget === "object" ? cap.budget : {};
    properties["budgetTokens"] = {
      type: "integer",
      minimum: range.minimum ?? 1,
      ...(range.maximum !== undefined ? { maximum: range.maximum } : {}),
    };
  }
  return { type: "object", additionalProperties: false, properties };
}

/** The structured-output tier a source's parameter names imply (OpenRouter's convention). */
export function structuredOutputFromNames(names: readonly string[]): "schema" | "object" | false {
  if (names.includes("structured_outputs")) return "schema";
  if (names.includes("response_format")) return "object";
  return false;
}

/** Levels in {@link REASONING_EFFORTS} order, unknown spellings dropped, each once. */
export function orderedEfforts(levels: readonly string[]): ReasoningEffort[] {
  return REASONING_EFFORTS.filter((e) => levels.includes(e));
}

/** Fallback `parameters` for a reasoning-family model with no recorded capabilities: everything a
 *  reasoning endpoint typically takes EXCEPT the sampling params it rejects (so they get filtered). */
const REASONING_FALLBACK_PARAMETERS = parametersFromNames(["max_tokens", "reasoning", "seed", "tools", "tool_choice"]);

/**
 * The capability gate for one model — the SINGLE reading of its `parameters`, shared by `plan` (fit
 * reporting) and `executeStructuredCall` (param filtering), so the dry-run can never drift from what
 * execution actually sends. An unknown model (no schema) accepts everything — the prior behavior.
 */
export interface ParamAcceptance {
  /** Whether the model takes this parameter. */
  accepts(key: ModelParameterKey): boolean;
  acceptsReasoning: boolean;
  /** The effort levels it takes, ascending. `undefined` ⇒ unknown (any level may be sent); empty ⇒ it
   *  takes reasoning but no level. */
  efforts: readonly ReasoningEffort[] | undefined;
  /** Whether it takes a thinking budget. `undefined` ⇒ unknown. */
  acceptsBudget: boolean | undefined;
  /** The largest budget it takes, when its schema says. */
  maxBudget?: number;
  /** The schema read, or `undefined` for an unknown model. */
  schema: SchemaDocument | undefined;
}

/** Read a `parameters` schema into a {@link ParamAcceptance}. */
export function acceptanceOf(schema: SchemaDocument | undefined): ParamAcceptance {
  if (schema === undefined) {
    return { accepts: () => true, acceptsReasoning: true, efforts: undefined, acceptsBudget: undefined, schema };
  }
  const properties = objectOf(schema["properties"]) ?? {};
  const reasoning = objectOf(properties["reasoning"]);
  const inner = reasoning !== undefined ? objectOf(reasoning["properties"]) : undefined;
  // `{}` for reasoning: accepted, and nothing known about how.
  const effortSchema = inner !== undefined ? objectOf(inner["effort"]) : undefined;
  const levels = effortSchema !== undefined && Array.isArray(effortSchema["enum"]) ? effortSchema["enum"] : undefined;
  const budgetSchema = inner !== undefined ? objectOf(inner["budgetTokens"]) : undefined;
  return {
    accepts: (key) => properties[key] !== undefined,
    acceptsReasoning: reasoning !== undefined,
    efforts:
      reasoning === undefined ? [] : inner === undefined ? undefined : effortSchema === undefined ? [] : levels === undefined ? undefined : orderedEfforts(levels as string[]),
    acceptsBudget: reasoning === undefined ? false : inner === undefined ? undefined : budgetSchema !== undefined,
    ...(budgetSchema !== undefined && typeof budgetSchema["maximum"] === "number" ? { maxBudget: budgetSchema["maximum"] } : {}),
    schema,
  };
}

function objectOf(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

// --- Reasoning: fitting a request to what the model takes ---------------------

/** Representative thinking budgets for an effort level, for a model that takes only a budget. */
export const EFFORT_BUDGET: Readonly<Record<ReasoningEffort, number>> = {
  none: 0,
  minimal: 1024,
  low: 2048,
  medium: 8192,
  high: 16384,
  xhigh: 32768,
  max: 65536,
  ultra: 65536,
};

/** The effort level a budget asks for, on a model with no budget knob: the lowest level whose
 *  representative budget covers it, and `max` above them all. The inverse of {@link EFFORT_BUDGET}. */
export function effortForBudget(budgetTokens: number): ReasoningEffort {
  for (const level of ["low", "medium", "high", "xhigh"] as const) if (budgetTokens <= EFFORT_BUDGET[level]) return level;
  return "max";
}

/**
 * Clamp a level to the ones a model offers: the highest at or below the request — asking for more
 * thought than a model has is satisfied by all of it — and the lowest it offers when the request is
 * below them all.
 */
export function clampEffort(effort: ReasoningEffort, levels: readonly ReasoningEffort[]): ReasoningEffort {
  const rank = REASONING_EFFORTS.indexOf(effort);
  const ordered = orderedEfforts(levels);
  const below = ordered.filter((l) => REASONING_EFFORTS.indexOf(l) <= rank);
  return below.at(-1) ?? ordered[0] ?? effort;
}

/** A reasoning request fitted to one model, and what fitting it changed, in words. */
export interface FittedReasoning {
  spec: ReasoningSpec | undefined;
  notes: string[];
}

/**
 * Fit a reasoning request to what a model takes — the ONE place a level is clamped, a budget turned
 * into a level (or back), or the request dropped, shared by the plan that reports it and the call that
 * sends it (JaiRA decision 0009 §5). An unknown model gets the request as asked.
 */
export function fitReasoning(spec: ReasoningSpec | undefined, gate: ParamAcceptance): FittedReasoning {
  const notes: string[] = [];
  if (spec === undefined || (spec.effort === undefined && spec.budgetTokens === undefined)) return { spec: undefined, notes };
  if (!gate.acceptsReasoning) return { spec: undefined, notes: ["takes no reasoning request — dropped"] };
  let effort = spec.effort;
  let budget = spec.budgetTokens;
  if (effort !== undefined && gate.efforts !== undefined) {
    if (effort === "none" && !gate.efforts.includes("none")) {
      // "Don't think" to a model with no such level is honoured by asking for nothing.
      notes.push('has no "none" level — sent no reasoning request');
      effort = undefined;
    } else if (gate.efforts.length === 0) {
      if (budget === undefined && gate.acceptsBudget !== false) {
        budget = EFFORT_BUDGET[effort];
        notes.push(`takes no effort level — "${effort}" sent as a ${budget}-token budget`);
      } else {
        notes.push(`takes no effort level — "${effort}" dropped`);
      }
      effort = undefined;
    } else if (!gate.efforts.includes(effort)) {
      const clamped = clampEffort(effort, gate.efforts);
      notes.push(`"${effort}" is not one of its levels (${gate.efforts.join(", ")}) — sent as "${clamped}"`);
      effort = clamped;
    }
  }
  if (budget !== undefined) {
    if (gate.acceptsBudget === false) {
      if (effort === undefined && gate.efforts?.length !== 0) {
        const asked = effortForBudget(budget);
        effort = gate.efforts === undefined ? asked : clampEffort(asked, gate.efforts);
        notes.push(`takes no thinking budget — ${budget} tokens sent as effort "${effort}"`);
      } else {
        notes.push("takes no thinking budget — dropped");
      }
      budget = undefined;
    } else if (gate.maxBudget !== undefined && budget > gate.maxBudget) {
      notes.push(`a ${budget}-token budget is above its ${gate.maxBudget} — sent as ${gate.maxBudget}`);
      budget = gate.maxBudget;
    }
  }
  if (effort === undefined && budget === undefined) return { spec: undefined, notes };
  return { spec: { ...(effort !== undefined ? { effort } : {}), ...(budget !== undefined ? { budgetTokens: budget } : {}) }, notes };
}

/**
 * The set of `${route}/${model}` keys a catalog's methods are typed to. When `Rows` is a literal tuple
 * (a caller who passed an inline array of rows, thanks to the `const` type param), this is the EXACT
 * union of that seed's keys — so `computeCost` on any other model fails to compile. When `Rows` is a
 * plain `ModelInfoInterface[]` (the runtime-hydrated singleton, or a weakly-typed construction), it
 * widens to `string` — so the runtime consumers that pass a `string` model id compile unchanged.
 */
export type ModelKeyOf<Rows extends readonly ModelInfoInterface[]> =
  Rows extends readonly [ModelInfoInterface, ...ModelInfoInterface[]]
    ? { [I in keyof Rows]: `${Rows[I]["route"]}/${Rows[I]["model"]}` }[number]
    : string;

/**
 * The model catalog. Owns every catalog operation as a member function and, via the static
 * {@link ModelInfo.instance} accessor, backs the process-wide default. Generic over its seed rows so a
 * literal construction gets compile-time-checked model keys (see {@link ModelKeyOf}).
 */
export class ModelInfo<const Rows extends readonly ModelInfoInterface[] = readonly ModelInfoInterface[]> {
  private readonly rows = new Map<string, ModelInfoInterface>();
  private readonly cacheReadMultiplier: number;
  private readonly cacheWriteMultiplier: number;
  private readonly cacheWrite1hMultiplier: number;

  constructor(seed: Rows, opts: PricingOptions = {}) {
    this.cacheReadMultiplier = opts.cacheReadMultiplier ?? 0.1;
    this.cacheWriteMultiplier = opts.cacheWriteMultiplier ?? 1.25;
    this.cacheWrite1hMultiplier = opts.cacheWrite1hMultiplier ?? 2.0;
    this.load(seed);
  }

  // --- The process-wide default instance ------------------------------------

  static #instance: ModelInfo | undefined;

  /** The process-wide catalog. Lazily built from {@link DEFAULT_MODELS} on first read; replace via the
   *  setter to hydrate/override from a store at startup. Weakly typed (its methods accept any `string`)
   *  because it is the runtime-hydrated case. */
  static get instance(): ModelInfo {
    return (ModelInfo.#instance ??= new ModelInfo(DEFAULT_MODELS));
  }

  static set instance(inst: ModelInfo) {
    ModelInfo.#instance = inst;
  }

  // --- Mutation (hydrate / refresh) -----------------------------------------

  /** Insert or replace a model row (keyed by `${route}/${model}`). Runtime adds are NOT compile-checked
   *  against the seed keys — the compile-time guarantee is about the CONSTRUCTOR data (§generic). */
  upsert(row: ModelInfoInterface): void {
    this.rows.set(keyForModel(row), row);
  }

  /** Bulk upsert — e.g. hydrate from a store at startup. */
  load(rows: readonly ModelInfoInterface[]): void {
    for (const row of rows) this.upsert(row);
  }

  remove(model: ModelKeyOf<Rows>): void {
    this.rows.delete(model);
  }

  list(): ModelInfoInterface[] {
    return [...this.rows.values()];
  }

  // --- Lookups (exact `${route}/${model}` key) ------------------------------

  /** The full catalog row for a `${route}/${model}` id (exact match), or undefined if unknown. */
  lookup(model: ModelKeyOf<Rows>): ModelInfoInterface | undefined {
    return this.rows.get(model);
  }

  hasPricing(model: ModelKeyOf<Rows>): boolean {
    return this.rows.has(model);
  }

  /**
   * What the model may be sent — the recorded `parameters` schema, else (for an OpenAI reasoning model)
   * its sampling-less fallback, else `undefined` ⇒ the caller sends everything.
   */
  parameters(model: ModelKeyOf<Rows>): SchemaDocument | undefined {
    const row = this.rows.get(model);
    if (row?.parameters !== undefined) return row.parameters;
    if (isReasoningModel(model)) return REASONING_FALLBACK_PARAMETERS;
    return undefined;
  }

  /** The structured-output tier the model's endpoint offers, when recorded. */
  structuredOutput(model: ModelKeyOf<Rows>): "schema" | "object" | false | undefined {
    return this.rows.get(model)?.structuredOutput;
  }

  /** The capability gate for one model (see {@link ParamAcceptance}). Unknown model ⇒ accepts everything. */
  paramAcceptance(model: ModelKeyOf<Rows>): ParamAcceptance {
    return acceptanceOf(this.parameters(model));
  }

  /** The recorded (already-resolved) structured-output schema profile for a model, if any. */
  schemaProfile(model: ModelKeyOf<Rows>): ProviderSchemaProfile | undefined {
    return this.rows.get(model)?.schemaProfile;
  }

  /** Input/output modalities a model accepts/produces (synced from `architecture.*_modalities`), if known.
   *  Used to gate media inputs and validate output-modality requests. */
  modalities(model: ModelKeyOf<Rows>): Modalities | undefined {
    return this.rows.get(model)?.modalities;
  }

  // --- Cost -----------------------------------------------------------------

  /**
   * Billing-accurate USD cost from a token breakdown, or `null` for an unknown model.
   *
   * Models every Anthropic pricing dimension exactly (the API reports all of them):
   *  - uncached input at the base rate, output (incl. reasoning) at the output rate;
   *  - cache READS at ≈0.1x; cache WRITES split by TTL — the 1-hour subset at ≈2x, the rest
   *    (5-minute) at ≈1.25x; and
   *  - the LONG-CONTEXT tier: if total input exceeds the row's `longContext.thresholdTokens`,
   *    every token is repriced at the premium rate set.
   * When no cache split is present (a provider that doesn't report one), the flat `inputTokens`
   * total is priced at the (tier's) base input rate. Cache rates default to multiples of the
   * tier's base input rate, so a row normally only specifies input/output.
   */
  computeCost(model: ModelKeyOf<Rows>, usage: UsageForCost): number | null {
    const p = this.rows.get(model);
    if (!p) return null;
    const rates = this.effectiveRates(p, usage.inputTokens ?? 0);

    const { noCacheTokens, cacheReadTokens, cacheWriteTokens, cacheWrite1hTokens } = usage;
    const hasSplit = noCacheTokens != null || cacheReadTokens != null || cacheWriteTokens != null;
    let inputCost: number;
    if (hasSplit) {
      const writes = cacheWriteTokens ?? 0;
      const writes1h = Math.min(Math.max(cacheWrite1hTokens ?? 0, 0), writes); // clamp into [0, writes]
      const writes5m = writes - writes1h;
      inputCost =
        (noCacheTokens ?? 0) * rates.input +
        (cacheReadTokens ?? 0) * rates.cacheRead +
        writes5m * rates.cacheWrite +
        writes1h * rates.cacheWrite1h;
    } else {
      inputCost = (usage.inputTokens ?? 0) * rates.input;
    }
    const outputCost = (usage.outputTokens ?? 0) * rates.output;
    return (inputCost + outputCost) / 1_000_000;
  }

  /** Resolve concrete per-million rates, applying the long-context tier when the input crosses it. */
  private effectiveRates(
    row: ModelInfoInterface,
    totalInputTokens: number,
  ): { input: number; output: number; cacheRead: number; cacheWrite: number; cacheWrite1h: number } {
    const lc = row.longContext;
    const set: RateSet = lc && totalInputTokens > lc.thresholdTokens ? lc : row;
    const input = set.inputPerMillion;
    return {
      input,
      output: set.outputPerMillion,
      cacheRead: set.cacheReadPerMillion ?? input * this.cacheReadMultiplier,
      cacheWrite: set.cacheWritePerMillion ?? input * this.cacheWriteMultiplier,
      cacheWrite1h: set.cacheWrite1hPerMillion ?? input * this.cacheWrite1hMultiplier,
    };
  }

  /** USD cost from flat input/output totals (no cache split). Delegates to {@link computeCost}. */
  computeCostUsd(model: ModelKeyOf<Rows>, inputTokens: number | null, outputTokens: number | null): number | null {
    return this.computeCost(model, { inputTokens, outputTokens });
  }

  /**
   * The AFFORDABLE output-token ceiling for a tight budget: how many output tokens `availableUsd` buys
   * after the input's cost, at the model's output rate. `Infinity` for an un-priced model (no clamp
   * basis) and 0 when even the input doesn't fit. Used by the `withBudget` reserve/clamp lifecycle.
   */
  affordableOutputTokens(model: ModelKeyOf<Rows>, inputTokens: number, availableUsd: number): number {
    const inputUsd = this.computeCostUsd(model, inputTokens, 0);
    const perOutputToken = (this.computeCostUsd(model, 0, 1_000_000) ?? 0) / 1_000_000;
    if (inputUsd == null || perOutputToken <= 0) return Number.POSITIVE_INFINITY; // un-priced — no clamp basis
    const headroom = availableUsd - inputUsd;
    return headroom <= 0 ? 0 : Math.floor(headroom / perOutputToken);
  }
}

/**
 * Seed catalog — a STARTING point, kept current by the §5 refresh (scrapes the Anthropic docs
 * pricing table + OpenRouter models API, which hydrates `ModelInfo.instance` at startup). Values
 * verified against https://platform.claude.com/docs/en/about-claude/pricing (2026-06-02). The seed
 * carries PRICE only; capabilities (`parameters` / `structuredOutput` / `schemaProfile`) are filled by
 * the refresh, with heuristic fallbacks until then.
 *
 * Identity is `{route}/{model}`: native Claude → route "anthropic"; everything else routes via
 * OpenRouter, so OpenAI/embeddings use route "openrouter" and the OpenRouter-native `openai/…` model id
 * (there is no native OpenAI route in this project — routes are only "anthropic" | "openrouter").
 *
 * NB current Anthropic models (Opus 4.6+/Sonnet 4.6) include the full 1M context window at
 * STANDARD pricing — there is NO long-context premium today, so no row seeds `longContext`
 * (the tier mechanism stays for when a model reintroduces one). Opt-in modifiers NOT seeded
 * (applied by the caller when used): data-residency ×1.1, Fast mode, Batch ×0.5.
 *
 * This is the hand-maintained CORE seed. It is the fallback base for the snapshot generator
 * (`scripts/updateModelInfo.ts`, run by `npm run update:model-info`): the generator seeds a catalog with
 * these (completed) rows and then overlays the live refresh, so the core models survive even if a source
 * is down. The generator's output — {@link DEFAULT_MODELS} — is what the runtime actually uses.
 */
export const CORE_SEED_MODELS = [
  // OpenAI — native route (input / output $/MTok). Hand-seeded like the Anthropic rows below, and it
  // has to be: the generator refreshes from Anthropic's docs and OpenRouter's API, and neither of
  // those publishes OpenAI's own catalog. A hand row still prices correctly and still lets the picker
  // offer the model.
  //
  // Every cache rate is EXPLICIT here, unlike the Anthropic rows, because the class defaults are
  // Anthropic's ratios and OpenAI does not share them. Two divergences, both silent:
  //  - the cached-read discount is 0.1x on the GPT-5 family but 0.25x on 4.1/o3/o4-mini and 0.5x on
  //    4o, so the inherited 0.1x would under-report a cached call by up to fivefold;
  //  - there is NO cache-WRITE charge at all — caching is automatic and free — while the inherited
  //    multipliers would bill a write at 1.25x/2x of input for tokens nobody was charged for.
  { route: "openai", model: "gpt-5", inputPerMillion: 1.25, outputPerMillion: 10,
    cacheReadPerMillion: 0.125, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-5-mini", inputPerMillion: 0.25, outputPerMillion: 2,
    cacheReadPerMillion: 0.025, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-5-nano", inputPerMillion: 0.05, outputPerMillion: 0.4,
    cacheReadPerMillion: 0.005, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-4.1", inputPerMillion: 2, outputPerMillion: 8,
    cacheReadPerMillion: 0.5, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-4.1-mini", inputPerMillion: 0.4, outputPerMillion: 1.6,
    cacheReadPerMillion: 0.1, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-4o", inputPerMillion: 2.5, outputPerMillion: 10,
    cacheReadPerMillion: 1.25, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "gpt-4o-mini", inputPerMillion: 0.15, outputPerMillion: 0.6,
    cacheReadPerMillion: 0.075, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "o3", inputPerMillion: 2, outputPerMillion: 8,
    cacheReadPerMillion: 0.5, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  { route: "openai", model: "o4-mini", inputPerMillion: 1.1, outputPerMillion: 4.4,
    cacheReadPerMillion: 0.275, cacheWritePerMillion: 0, cacheWrite1hPerMillion: 0 },
  // Anthropic — native route (input / output $/MTok; cache rates derive 0.1x/1.25x/2x).
  { route: "anthropic", model: "claude-opus-4-8", inputPerMillion: 5, outputPerMillion: 25 },
  { route: "anthropic", model: "claude-opus-4-7", inputPerMillion: 5, outputPerMillion: 25 },
  { route: "anthropic", model: "claude-opus-4-6", inputPerMillion: 5, outputPerMillion: 25 },
  { route: "anthropic", model: "claude-opus-4-5", inputPerMillion: 5, outputPerMillion: 25 },
  { route: "anthropic", model: "claude-opus-4-1", inputPerMillion: 15, outputPerMillion: 75 },
  { route: "anthropic", model: "claude-opus-4-0", inputPerMillion: 15, outputPerMillion: 75 },
  { route: "anthropic", model: "claude-sonnet-4-8", inputPerMillion: 3, outputPerMillion: 15 },
  { route: "anthropic", model: "claude-sonnet-4-6", inputPerMillion: 3, outputPerMillion: 15 },
  { route: "anthropic", model: "claude-sonnet-4-5", inputPerMillion: 3, outputPerMillion: 15 },
  { route: "anthropic", model: "claude-sonnet-4-2", inputPerMillion: 3, outputPerMillion: 15 },
  { route: "anthropic", model: "claude-sonnet-4-0", inputPerMillion: 3, outputPerMillion: 15 },
  { route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 },
  { route: "anthropic", model: "claude-haiku-3-5", inputPerMillion: 0.8, outputPerMillion: 4 },
  { route: "anthropic", model: "claude-3-haiku", inputPerMillion: 0.25, outputPerMillion: 1.25 },

  // OpenAI — via OpenRouter (route "openrouter", OpenRouter-native `openai/…` ids).
  { route: "openrouter", model: "openai/gpt-4o-mini", inputPerMillion: 0.15, outputPerMillion: 0.6 },
  { route: "openrouter", model: "openai/gpt-4o", inputPerMillion: 2.5, outputPerMillion: 10 },
  { route: "openrouter", model: "openai/gpt-4.1-nano", inputPerMillion: 0.1, outputPerMillion: 0.4 },
  { route: "openrouter", model: "openai/gpt-4.1-mini", inputPerMillion: 0.4, outputPerMillion: 1.6 },
  { route: "openrouter", model: "openai/gpt-4.1", inputPerMillion: 2, outputPerMillion: 8 },
  { route: "openrouter", model: "openai/gpt-5-nano", inputPerMillion: 0.25, outputPerMillion: 1 },
  { route: "openrouter", model: "openai/gpt-5-mini", inputPerMillion: 1, outputPerMillion: 4 },
  { route: "openrouter", model: "openai/gpt-5", inputPerMillion: 3.75, outputPerMillion: 15 },
  { route: "openrouter", model: "openai/o3-mini", inputPerMillion: 1.1, outputPerMillion: 4.4 },
  { route: "openrouter", model: "openai/o3", inputPerMillion: 10, outputPerMillion: 40 },
  { route: "openrouter", model: "openai/o4-mini", inputPerMillion: 1.1, outputPerMillion: 4.4 },

  // NB other providers (Google, Meta, DeepSeek, Amazon, Mistral, xAI, …) are intentionally NOT
  // seeded here: their authoritative prices + ids + `releasedAt` + capabilities come from the §5
  // OpenRouter refresh (model-catalog-source.ts). Hand-seeding them would risk a fabricated price the
  // refresh never corrects (when its real `vendor/model` id differs from the guess).

  // Embeddings (§3.6 kernel) — via OpenRouter.
  { route: "openrouter", model: "openai/text-embedding-3-small", inputPerMillion: 0.02, outputPerMillion: 0 },
  { route: "openrouter", model: "openai/text-embedding-3-large", inputPerMillion: 0.13, outputPerMillion: 0 },
] as const satisfies readonly ModelInfoInterface[];

/**
 * What the runtime uses: the committed snapshot produced by `npm run update:model-info`
 * ({@link GENERATED_MODELS}), plus any {@link CORE_SEED_MODELS} row it does not already carry.
 *
 * The union matters for a route the GENERATOR cannot see. It refreshes from Anthropic's docs and
 * OpenRouter's API, so a native OpenAI row exists only in the seed — and a plain
 * `= GENERATED_MODELS` would drop it on every regeneration, silently emptying the `openai` route
 * while the code that added it still looked right.
 *
 * The seed half is passed through {@link deriveIdentity}, which the GENERATED half already went
 * through on the way in (`model-catalog-source`). Without it a folded-in row reaches the catalog
 * with no `provider`, `label` or `canonicalId` — so it prices correctly and then shows up in a picker
 * that groups by vendor as an unnamed row under no heading. That was invisible while the only seed
 * rows were Anthropic ones, because a generated row of the same key masked every one of them.
 *
 * This is a WEAKLY-typed array by annotation, unlike the two tuples it is built from: a value spliced
 * at runtime has no literal type to offer, and {@link KnownModelKey} is where the compile-time enum
 * lives instead. `ModelInfo.instance` is the weak (string-keyed) case anyway, so nothing regresses —
 * a caller who wants checked keys constructs `new ModelInfo(GENERATED_MODELS)` from a tuple directly.
 */
export const DEFAULT_MODELS: readonly ModelInfoInterface[] = (() => {
  const have = new Set(GENERATED_MODELS.map((row) => keyForModel(row)));
  const seeded = CORE_SEED_MODELS.filter((row) => !have.has(keyForModel(row))).map((row) => deriveIdentity(row));
  return [...GENERATED_MODELS, ...seeded];
})();

/**
 * The union of every `${route}/${model}` key the runtime knows out of the box — a compile-time enum.
 *
 * BOTH halves of {@link DEFAULT_MODELS}, which is why `CORE_SEED_MODELS` is `as const`: a key that
 * exists only in the seed — every native `openai/…` id — is as known as one the refresh wrote, and a
 * union naming only the snapshot would call it unknown.
 */
export type KnownModelKey = ModelKeyOf<typeof GENERATED_MODELS> | ModelKeyOf<typeof CORE_SEED_MODELS>;

/** Seed rows (a fresh copy per call). */
export function modelsSeed(): ModelInfoInterface[] {
  return DEFAULT_MODELS.map((r) => ({ ...r }));
}
