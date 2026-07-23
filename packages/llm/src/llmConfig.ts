/**
 * `LlmConfiguration` — the ONE canonical "how to call an LLM" type. Before this, the model + decoding
 * params were re-listed by hand in at least four places (`ConfigPoint` + `ConfigConstraints` in
 * `search/config/space.ts`, `SearchSpace` in `standardInterface.ts`, and the loose fields on
 * `StartSearchInput`). They all collapse onto this hierarchy.
 *
 * The shape is a small TYPE HIERARCHY keyed on what a model family accepts (the user's "extending
 * interfaces depending on provider / model type"): the universal base, a SAMPLING variant (temperature /
 * top-p / top-k), and a REASONING variant (reasoning effort, rejects the sampling knobs). The runtime
 * witness of "which variant is this model" already exists as the catalog capability lookup
 * (`supportedParametersFor` + `acceptsParam` in `search/config/space.ts`); these types just name it.
 *
 * This is the canonical SERIALIZABLE descriptor of an LLM call: it carries EVERY input the underlying
 * call mechanism accepts, so that a stored config transforms losslessly into a real call (`@declarative-ai/llm`
 * performs that transform). It therefore MIRRORS the AI SDK's high-level call-capability surface: the
 * `CallSettings` decoding knobs (maxOutputTokens/temperature/topP/topK/presence+frequency penalty/seed/
 * stopSequences) plus the `providerOptions` escape hatch, with two deliberate divergences: (a) runtime-
 * only fields (`abortSignal`, `maxRetries`, `timeout`, `headers`) are NOT stored — the transform supplies
 * them; (b) reasoning is kept PROVIDER-NEUTRAL (`ReasoningSpec`, adapted at the boundary) rather than
 * folded into provider-shaped `providerOptions`. When a new decoding knob is worth exposing, add it HERE
 * (and to `parseLlmConfig`) so no functionality is silently dropped. TOOLS are modeled too (`tools` +
 * `toolChoice` + `maxSteps`): the serializable DECLARATIONS live here (name/description/input schema, or a
 * provider-tool id/args); the runtime `execute` implementations are supplied by the executor at call time
 * (keyed by name) — the same split the model handle and validator use.
 *
 * The PROMPT is NOT here — it is layered on by {@link CallSignature} to make an
 * {@link LlmCallDefinition} for a concrete call, or by findmyprompt's `LlmParameters` for a search
 * point — so this stays purely "how to decode."
 *
 * Parse, don't validate: `parseLlmConfig` turns a stored JSON blob into one of these variants and THROWS
 * on malformed input (a present-but-wrong-typed field is an error, never silently coerced/dropped). Pure
 * data — no provider/DB coupling, no AI-SDK import — importable by the engine, server, and config UI alike.
 */
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type { CallSignature } from "./prompt";

/** A serializable bag of PROVIDER-SPECIFIC options, keyed by provider id (the SDK's `providerOptions` /
 *  `SharedV3ProviderOptions = Record<string, JSONObject>`). The escape hatch for functionality not
 *  modeled as a first-class field; passed through to the provider verbatim (merged with adapted reasoning
 *  at the call boundary). Kept as plain JSON so core stays AI-SDK-free. */
export type ProviderOptions = Record<string, Record<string, JsonValue>>;

/** How the model may select tools (the SDK's `ToolChoice`): let it decide (`auto`), forbid tools
 *  (`none`), force SOME tool (`required`), or force ONE named tool. */
export type LlmToolChoice = "auto" | "none" | "required" | { type: "tool"; toolName: string };

/**
 * A SERIALIZABLE tool declaration — everything the MODEL needs to know about a tool, minus the runtime
 * `execute` implementation (the executor supplies that at call time, keyed by `name`; a declared tool with
 * no supplied executor is a single-turn tool whose call is returned to the caller). Two flavors mirror the
 * SDK: a FUNCTION tool the host runs, or a PROVIDER tool the provider runs server-side (web search, etc.).
 */
export type ToolDefinition =
  | {
      type?: "function";
      /** Unique tool name the model calls (and the key the executor's implementation is looked up under). */
      name: string;
      description?: string;
      /** JSON Schema for the input the model must produce for a call. */
      inputSchema: JsonSchema;
      /** Request the provider's strict-mode input generation where supported. */
      strict?: boolean;
    }
  | {
      type: "provider";
      /** Local name the model calls it by. */
      name: string;
      /** Provider tool id, formatted `"<provider>.<unique-name>"` (e.g. `"anthropic.web_search"`). */
      id: `${string}.${string}`;
      /** Provider-defined configuration args for the tool. */
      args: Record<string, JsonValue>;
    };

/** Universal LLM-call params EVERY model accepts (both sampling and reasoning families). */
export interface LlmConfiguration {
  model: string;
  maxOutputTokens?: number;
  stopSequences?: string[];
  /** Deterministic-sampling seed. Accepted by both families (a reasoning endpoint takes it too), so it
   *  lives on the universal base rather than the sampling variant. */
  seed?: number;
  /** Raw provider-specific options passthrough (SDK `providerOptions`) — the full escape hatch. */
  providerOptions?: ProviderOptions;
  /** Tool DECLARATIONS the model may call (implementations injected at call time, keyed by name). */
  tools?: ToolDefinition[];
  /** How the model may select among `tools`. */
  toolChoice?: LlmToolChoice;
  /** Max tool-use steps (model→tool→model round-trips) when tools are executed in-loop. Ignored without
   *  executable tools; the executor bounds the loop at this many steps. */
  maxSteps?: number;
  /** LOGICAL session id — portable, caller/workflow-chosen. Resolved via the injected session store (its
   *  provider handle / transcript lives THERE, never in this portable declaration). Coordination key:
   *  different states/calls sharing this id continue the same conversation. */
  sessionId?: string;
  /** EXPLICIT provider session handle — the exact server session to resume, used when no logical id is
   *  declared (usually threaded as runtime data from a prior call's `Outcome.session.id`). Resolution
   *  prefers `sessionId`, falling back to this. */
  providerSessionId?: string;
  /** Requested OUTPUT modalities (e.g. `["text","image"]`) for models that must be ASKED to emit non-text
   *  output. Gated against the model's `modalities.output` at plan time; forwarded per-provider. */
  outputModalities?: string[];
}

/** Sampling models — the decoding knobs a temperature-sampling endpoint accepts (a reasoning model
 *  rejects these). Mirrors the SDK `CallSettings` sampling knobs. */
export interface SamplingConfiguration extends LlmConfiguration {
  temperature?: number;
  topP?: number;
  topK?: number;
  /** Penalizes tokens already present in the prompt (SDK `CallSettings.presencePenalty`). */
  presencePenalty?: number;
  /** Penalizes tokens by prior frequency (SDK `CallSettings.frequencyPenalty`). */
  frequencyPenalty?: number;
}

/** A model's "how hard to think" request, in PROVIDER-NEUTRAL terms (the standard interface): an effort
 *  LEVEL and/or a token BUDGET. Models differ in which they accept — some reasoning models take an effort
 *  level, others a thinking budget — so BOTH are carried here and ADAPTED to the provider's shape at the
 *  call boundary (`adaptReasoning`, `providers/reasoning.ts`); provider specifics never leak into the search
 *  config or the stored config JSON. */
export interface ReasoningSpec {
  effort?: "low" | "medium" | "high";
  budgetTokens?: number;
}

/** Reasoning models — accept a `reasoning` request (effort/budget) and REJECT the sampling knobs (so a
 *  reasoning model simply has no temperature/top-p branch, §config-as-dimensions). `reasoning` is REQUIRED:
 *  its presence is what makes a config a reasoning config, and it is the DISCRIMINANT of {@link LlmCallConfig}
 *  (`"reasoning" in cfg` / {@link isReasoningConfig} narrows the union cleanly). */
export interface ReasoningConfiguration extends LlmConfiguration {
  reasoning: ReasoningSpec;
}

/** A single RESOLVED LLM call config — a model is sampling XOR reasoning, so the union is the honest point
 *  shape (illegal "both at once" states are unrepresentable once parsed). Discriminated by `reasoning`
 *  (present ⇒ reasoning, absent ⇒ sampling). `enumerateConfigPoints` produces these (capability-pruned). */
export type LlmCallConfig = SamplingConfiguration | ReasoningConfiguration;

/** Narrow an {@link LlmCallConfig} to the reasoning variant (its `reasoning` field is the discriminant). */
export function isReasoningConfig(cfg: LlmCallConfig): cfg is ReasoningConfiguration {
  return "reasoning" in cfg;
}

// --- Parsing (parse, don't validate) ----------------------------------------

/** A stored value was present but not the type this field requires. Thrown by the `parse*` functions so a
 *  malformed config fails loudly at the boundary rather than being silently coerced or dropped. */
export class LlmConfigParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LlmConfigParseError";
  }
}

/** The raw, still-unparsed property bag a config layer arrives as. `unknown` VALUES are the sanctioned
 *  boundary position (§2.2): every one is narrowed by the `parse*` functions below before it escapes as a
 *  typed field, and this type is never the shape of a parsed result. */
type RawBag = Record<string, unknown>;

/** Coerce a field that must be a finite number IF PRESENT: absent ⇒ undefined; present-but-not-a-number ⇒
 *  throw (never a best-effort drop). */
function numField(v: unknown, field: string): number | undefined {
  if (v === undefined) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) throw new LlmConfigParseError(`${field} must be a finite number`);
  return v;
}

/** Coerce a value that must be a PLAIN OBJECT (not null / array): throw with `<what> must be an object`
 *  (+ optional hint) otherwise. The object-shaped sibling of {@link numField}. */
function requireObject(v: unknown, what: string, hint = ""): RawBag {
  if (v === null || typeof v !== "object" || Array.isArray(v)) throw new LlmConfigParseError(`${what} must be an object${hint}`);
  return v as RawBag;
}

/** Parse a `reasoning` blob into a typed `ReasoningSpec`, THROWING on any malformed field. Requires at
 *  least one of `effort`/`budgetTokens` (an empty reasoning request is meaningless). Keys are emitted in a
 *  FIXED order so re-serialization is content-stable. */
export function parseReasoningSpec(v: unknown): ReasoningSpec {
  const o = requireObject(v, "reasoning") as { effort?: unknown; budgetTokens?: unknown };
  if (o.effort !== undefined && o.effort !== "low" && o.effort !== "medium" && o.effort !== "high") {
    throw new LlmConfigParseError('reasoning.effort must be one of "low" | "medium" | "high"');
  }
  const budgetTokens = numField(o.budgetTokens, "reasoning.budgetTokens");
  if (o.effort === undefined && budgetTokens === undefined) {
    throw new LlmConfigParseError("reasoning must specify effort and/or budgetTokens");
  }
  return { ...(o.effort !== undefined ? { effort: o.effort } : {}), ...(budgetTokens !== undefined ? { budgetTokens } : {}) };
}

/** Parse a `toolChoice` value, THROWING on anything outside the allowed shapes. */
function parseToolChoice(v: unknown): LlmToolChoice | undefined {
  if (v === undefined) return undefined;
  if (v === "auto" || v === "none" || v === "required") return v;
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    const o = v as { type?: unknown; toolName?: unknown };
    if (o.type === "tool" && typeof o.toolName === "string" && o.toolName.length > 0) return { type: "tool", toolName: o.toolName };
  }
  throw new LlmConfigParseError('toolChoice must be "auto" | "none" | "required" | { type: "tool", toolName }');
}

/** Parse ONE tool declaration, THROWING on malformed shape. */
function parseTool(t: unknown, i: number): ToolDefinition {
  const o = requireObject(t, `tools[${i}]`);
  if (typeof o.name !== "string" || o.name.length === 0) throw new LlmConfigParseError(`tools[${i}].name must be a non-empty string`);
  if (o.type === "provider") {
    if (typeof o.id !== "string" || !/^[^.]+\.[^.]+/.test(o.id)) throw new LlmConfigParseError(`tools[${i}].id must be formatted "<provider>.<name>"`);
    return { type: "provider", name: o.name, id: o.id as `${string}.${string}`, args: requireObject(o.args, `tools[${i}].args`) as Record<string, JsonValue> };
  }
  if (o.type !== undefined && o.type !== "function") throw new LlmConfigParseError(`tools[${i}].type must be "function" | "provider"`);
  requireObject(o.inputSchema, `tools[${i}].inputSchema`, " (JSON Schema)");
  if (o.description !== undefined && typeof o.description !== "string") throw new LlmConfigParseError(`tools[${i}].description must be a string`);
  if (o.strict !== undefined && typeof o.strict !== "boolean") throw new LlmConfigParseError(`tools[${i}].strict must be a boolean`);
  return {
    type: "function",
    name: o.name,
    ...(o.description !== undefined ? { description: o.description } : {}),
    inputSchema: o.inputSchema as JsonSchema,
    ...(o.strict !== undefined ? { strict: o.strict } : {}),
  };
}

/** Parse the `tools` array, THROWING on a non-array or any malformed declaration. */
function parseTools(v: unknown): ToolDefinition[] | undefined {
  if (v === undefined) return undefined;
  if (!Array.isArray(v)) throw new LlmConfigParseError("tools must be an array");
  return v.map((t, i) => parseTool(t, i));
}

/** Parse a stored config JSON into a concrete {@link LlmCallConfig} — a `SamplingConfiguration` XOR a
 *  `ReasoningConfiguration` — THROWING on malformed input (parse, don't validate). The rules:
 *   - `model` MUST be a non-empty string;
 *   - numeric/array fields present-but-wrong-typed ⇒ throw (never coerced/dropped);
 *   - `reasoning` present ⇒ a ReasoningConfiguration, and the sampling knobs (temperature/topP/topK/
 *     penalties) MUST be absent (a model is sampling XOR reasoning); absent ⇒ a SamplingConfiguration.
 *  Replaces the old best-effort `readLlmConfig`/`FlatLlmConfig` flat bag. */
export function parseLlmConfig(json: JsonValue): LlmCallConfig {
  const j = requireObject(json, "config");
  if (typeof j.model !== "string" || j.model.length === 0) throw new LlmConfigParseError("model must be a non-empty string");

  // Strict surface: an unknown key is an ERROR, never silently dropped — a caller passing `temprature`
  // or a definition-layer field here must find out loudly, not run with the field ignored.
  const unknown = Object.keys(j).filter((k) => !CONFIG_KEYS.has(k));
  if (unknown.length > 0) {
    const defLayer = unknown.filter((k) => (SIGNATURE_KEYS as readonly string[]).includes(k));
    throw new LlmConfigParseError(
      `unknown config key(s): ${unknown.join(", ")}` +
        (defLayer.length > 0
          ? ` — ${defLayer.join(", ")} are SIGNATURE fields (prompt/schema/budget), split out by resolveConfig, not part of LlmConfiguration`
          : ""),
    );
  }

  const strArr = (v: unknown, field: string): string[] | undefined => {
    if (v === undefined) return undefined;
    if (!Array.isArray(v) || !v.every((x) => typeof x === "string")) throw new LlmConfigParseError(`${field} must be a string[]`);
    return v as string[];
  };
  const strField = (v: unknown, field: string): string | undefined => {
    if (v === undefined) return undefined;
    if (typeof v !== "string" || v.length === 0) throw new LlmConfigParseError(`${field} must be a non-empty string`);
    return v;
  };
  const providerOptions = ((): ProviderOptions | undefined => {
    if (j.providerOptions === undefined) return undefined;
    const po = requireObject(j.providerOptions, "providerOptions", " keyed by provider");
    for (const [k, v] of Object.entries(po)) requireObject(v, `providerOptions.${k}`);
    return po as ProviderOptions;
  })();

  const base: LlmConfiguration = {
    model: j.model,
    maxOutputTokens: numField(j.maxOutputTokens, "maxOutputTokens"),
    stopSequences: strArr(j.stopSequences, "stopSequences"),
    seed: numField(j.seed, "seed"),
    providerOptions,
    tools: parseTools(j.tools),
    toolChoice: parseToolChoice(j.toolChoice),
    maxSteps: numField(j.maxSteps, "maxSteps"),
    sessionId: strField(j.sessionId, "sessionId"),
    providerSessionId: strField(j.providerSessionId, "providerSessionId"),
    outputModalities: strArr(j.outputModalities, "outputModalities"),
  };

  if (j.reasoning !== undefined) {
    const present = SAMPLING_KEYS.filter((k) => j[k] !== undefined);
    if (present.length > 0) {
      throw new LlmConfigParseError(`a reasoning config cannot also set sampling knobs (${present.join(", ")})`);
    }
    return { ...base, reasoning: parseReasoningSpec(j.reasoning) };
  }
  return {
    ...base,
    temperature: numField(j.temperature, "temperature"),
    topP: numField(j.topP, "topP"),
    topK: numField(j.topK, "topK"),
    presencePenalty: numField(j.presencePenalty, "presencePenalty"),
    frequencyPenalty: numField(j.frequencyPenalty, "frequencyPenalty"),
  };
}

/** The sampling-only knobs (the ones a reasoning config must NOT carry) — shared by parse + resolve. */
export const SAMPLING_KEYS = ["temperature", "topP", "topK", "presencePenalty", "frequencyPenalty"] as const;

/** The SIGNATURE keys that may ride alongside the config knobs in a layer — the prompt inputs plus the
 *  per-call budget and the output schema. `resolveConfig` splits these out of the merged layers so the
 *  config bag itself parses strictly, then layers them back on to return an {@link LlmCallDefinition}.
 *  The old `DefinitionLayer` + split/re-merge dance existed only because this module lived BELOW `llm`
 *  and could not name the AI-SDK prompt types; from here it can (DESIGN §4.1). */
export const SIGNATURE_KEYS = ["system", "prompt", "messages", "attachments", "timeoutMs", "schema"] as const;

/** Every key `parseLlmConfig` accepts — the full strict config surface. */
const CONFIG_KEYS = new Set<string>([
  "model",
  "maxOutputTokens",
  "stopSequences",
  "seed",
  "providerOptions",
  "tools",
  "toolChoice",
  "maxSteps",
  "sessionId",
  "providerSessionId",
  "outputModalities",
  "reasoning",
  ...SAMPLING_KEYS,
]);

/** Structural checks for the SIGNATURE fields split out by {@link resolveConfig}: never silent — a
 *  present-but-malformed field throws. */
function checkSignature(d: RawBag): void {
  if (d.timeoutMs !== undefined) numField(d.timeoutMs, "timeoutMs");
  if (d.prompt !== undefined && typeof d.prompt !== "string" && !Array.isArray(d.prompt)) {
    throw new LlmConfigParseError("prompt must be a string or a message array");
  }
  if (d.messages !== undefined && !Array.isArray(d.messages)) throw new LlmConfigParseError("messages must be an array");
  if (d.system !== undefined && typeof d.system !== "string" && (d.system === null || typeof d.system !== "object")) {
    throw new LlmConfigParseError("system must be a string or a system message (array)");
  }
  if (d.attachments !== undefined) {
    if (!Array.isArray(d.attachments)) throw new LlmConfigParseError("attachments must be an array");
    d.attachments.forEach((a, i) => {
      const o = requireObject(a, `attachments[${i}]`);
      if (typeof o.mediaType !== "string" || o.mediaType.length === 0) {
        throw new LlmConfigParseError(`attachments[${i}].mediaType must be a non-empty string`);
      }
      if (!(o.data instanceof Uint8Array)) requireObject(o.data, `attachments[${i}].data`, " (bytes, { base64 }, or { url })");
    });
  }
  // `schema` joined SIGNATURE_KEYS when the output schema moved onto the definition (§5.2), and it is
  // cast to `JsonSchema<T>` on the way out with no parse behind it. Unchecked, a non-document value
  // reaches `adaptSchemaCached`, whose `WeakMap.get` throws a raw TypeError straight out of
  // `executeLlmCall` — which is documented to never throw.
  if (d.schema !== undefined) requireObject(d.schema, "schema", " (a JSON Schema document)");
}

// --- Resolution (compose fragments → one config) ----------------------------

/**
 * ONE layer handed to {@link resolveConfig}: a partial config (any knob of EITHER family — the merge is
 * family-aware and the strict parse rejects illegal combinations) optionally carrying signature fields.
 * `FlattenUnion` is what lets a single layer name sampling and reasoning knobs alike; the XOR is
 * enforced when the merged bag is parsed, not per layer.
 */
export type ConfigLayer<T = JsonValue> = Partial<FlattenUnion<LlmCallConfig>> & Partial<CallSignature<T>>;

/**
 * The serializable definition of ONE LLM call: the canonical {@link LlmCallConfig} (the
 * sampling-XOR-reasoning union) plus its {@link CallSignature} — the prompt inputs, the output schema,
 * and the time budget. It stays a UNION (not a flattened bag) so "sampling + reasoning at once" is
 * unrepresentable. Everything here is plain JSON — no live handle, no closures — which is what lets the
 * call become a durable step (hence `model` is a string id, resolved to a handle inside the executor).
 *
 * `StructuredCallParams` is GONE (DESIGN §4.1): it differed from this by two optional fields
 * and a required-ness change. `schema` belongs IN the definition (it is declarative and serializable —
 * its absence is why the old lowering had to smuggle it through `spec.outputSchema` and cast the
 * phantom away); `schemaId` was documented as informational only and is deleted; `timeoutMs: required`
 * is a call-site concern, so it became an argument.
 */
export type LlmCallDefinition<T = JsonValue> = LlmCallConfig & CallSignature<T>;

/** A named-config registry: `get` resolves an id to its (possibly-partial) config layer; `idOf` is a
 *  best-effort reverse lookup for PROVENANCE only (identity is the resolved content hash, never the id). */
export interface ConfigurationRegistry {
  get(id: string): ConfigLayer | undefined;
  idOf?(config: ConfigLayer): string | undefined;
}

/** A plain-map configuration registry. */
export class MapConfigurationRegistry implements ConfigurationRegistry {
  private readonly map = new Map<string, ConfigLayer>();
  set(id: string, config: ConfigLayer): this {
    this.map.set(id, config);
    return this;
  }
  get(id: string): ConfigLayer | undefined {
    return this.map.get(id);
  }
}

export interface ResolveResult<T = JsonValue> {
  /** The composed, strictly-parsed call definition — config knobs AND signature, ready to execute. */
  definition: LlmCallDefinition<T>;
  /** Non-fatal notes from the merge (e.g. an overridden family clearing the opposite family's knobs). */
  warnings: string[];
}

/**
 * Compose config fragments into ONE valid {@link LlmCallDefinition}: merge the raw property bags
 * LOW→HIGH (e.g. `[engineDefault, workflowDefault, registry.get(ref), inlineOverrides]`) — later layers
 * win per key — then {@link parseLlmConfig} the config half (strict: throws on a malformed merged bag)
 * and layer the signature half back on. The merge is FAMILY-AWARE: a higher layer that introduces
 * `reasoning` CLEARS accumulated sampling knobs (and a higher sampling knob clears accumulated
 * `reasoning`), each with a warning — "replace, don't explode". This is "parse, don't validate" applied
 * to composition: merge loosely, parse strictly. Absent layers are skipped.
 *
 * An explicit `undefined` on a layer means the layer SAYS NOTHING about that key — it never erases a
 * lower layer's value. (Only a whole-family override clears inherited knobs, and it warns when it does.)
 *
 * It returns the DEFINITION directly (§5.3). The old `{ config, definition }` pair existed only because
 * `parseLlmConfig` is strict and would throw on the prompt-shaped keys — so `resolveConfig` split them
 * out and the caller immediately merged them back.
 */
export function resolveConfig<T = JsonValue>(layers: Array<ConfigLayer<T> | undefined>): ResolveResult<T> {
  const warnings: string[] = [];
  const merged: RawBag = {};
  for (const rawLayer of layers) {
    if (!rawLayer) continue;
    const layer = rawLayer as RawBag;
    const introducesReasoning = layer.reasoning !== undefined;
    const introducesSampling = SAMPLING_KEYS.some((k) => layer[k] !== undefined);
    if (introducesReasoning) {
      const cleared = SAMPLING_KEYS.filter((k) => merged[k] !== undefined);
      if (cleared.length > 0) {
        for (const k of cleared) delete merged[k];
        warnings.push(`reasoning config replaces and clears inherited sampling knobs (${cleared.join(", ")})`);
      }
    }
    if (introducesSampling && merged.reasoning !== undefined) {
      delete merged.reasoning;
      warnings.push("sampling config replaces and clears inherited reasoning");
    }
    // Copy only DEFINED keys. An explicit `undefined` means "this layer says NOTHING about that knob",
    // never "clear it" — absent and present-but-undefined are the SAME thing throughout this function,
    // which is what the two family checks above already assume (`layer[k] !== undefined`). `Object.assign`
    // copied explicit `undefined`s over a lower layer's real value and erased it, and because
    // `introducesSampling` also tests `!== undefined` no warning fired either. This is not exotic: a
    // preset built by spreading a partially-filled record carries an `undefined` for every field it did
    // not fill. Clearing an inherited knob IS expressible — a layer that names the OPPOSITE family
    // clears it, loudly, with a warning.
    for (const [k, v] of Object.entries(layer)) {
      if (v !== undefined) merged[k] = v;
    }
  }
  // Split the SIGNATURE fields OUT of the bag so the config parses strictly — anything left that isn't
  // a config key then throws in `parseLlmConfig`.
  const signature: RawBag = {};
  for (const k of SIGNATURE_KEYS) {
    if (merged[k] !== undefined) {
      signature[k] = merged[k];
      delete merged[k];
    }
  }
  // `checkSignature` is the narrowing step that earns this view: it throws on any present-but-malformed
  // field, so what survives conforms to the shape the callers declared.
  checkSignature(signature);
  const config = parseLlmConfig(merged as JsonValue);
  return { definition: { ...config, ...(signature as CallSignature<T>) }, warnings };
}

/** The full SOLUTION POINT a prompt candidate stands for: the LLM call config PLUS the prompt texts.
 *  This is the `T` the search space is the menu-of-each-field of (`BaseSearchConfiguration<LlmParameters>`,
 *  `standardInterface.ts`) — i.e. the target PromptOp's complete parameter set. */
export type LlmParameters = LlmCallConfig & {
  systemPrompt: string;
  userPrompt: string;
};

// --- Type-level helpers for the search space --------------------------------

/** Turn every member of `T` into a MENU of options for that member — the search-space transform. A
 *  one-element menu = a bound (fixed) value; a multi-element menu = a search dimension ("pick one of N").
 *  This is what generalizes how `models` already worked to every config dimension at once. */
export type MakeMembersArrays<T> = { [K in keyof T]: T[K][] };

/** All keys across a UNION's members (`keyof (A | B)` alone yields only the SHARED keys, which would drop
 *  the per-variant params — temperature/top-p/top-k on sampling, reasoning on reasoning). */
export type AllKeys<T> = T extends unknown ? keyof T : never;

/** A union member's value at `K` (or `never` for members lacking `K`), distributed over the union. */
type ValueAt<T, K extends PropertyKey> = T extends unknown ? (K extends keyof T ? T[K] : never) : never;

/** Flatten a union of object types into ONE object with every member's keys (each made required +
 *  non-nullable so the downstream menu type is clean, e.g. `temperature: number` not `number | undefined`).
 *  Needed because the solution point type (`LlmParameters`) is a sampling-XOR-reasoning union, but a single
 *  search space must be able to carry menus for BOTH at once (capability-pruned per model at enumeration). */
export type FlattenUnion<T> = { [K in AllKeys<T>]-?: NonNullable<ValueAt<T, K>> };
