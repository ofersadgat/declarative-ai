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
 * The PROMPT is NOT here — it is layered on by the consumer (`@declarative-ai/llm`'s `LlmCallDefinition` for a
 * concrete call, findmyprompt's `LlmParameters` for a search point), so this stays purely "how to decode."
 *
 * Parse, don't validate: `parseLlmConfig` turns a stored JSON blob into one of these variants and THROWS
 * on malformed input (a present-but-wrong-typed field is an error, never silently coerced/dropped). Pure
 * data — no provider/DB coupling, no AI-SDK import — importable by the engine, server, and config UI alike.
 */

/** A serializable bag of PROVIDER-SPECIFIC options, keyed by provider id (the SDK's `providerOptions` /
 *  `SharedV3ProviderOptions = Record<string, JSONObject>`). The escape hatch for functionality not
 *  modeled as a first-class field; passed through to the provider verbatim (merged with adapted reasoning
 *  at the call boundary). Kept as plain JSON so core stays AI-SDK-free. */
export type ProviderOptions = Record<string, Record<string, unknown>>;

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
      inputSchema: Record<string, unknown>;
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
      args: Record<string, unknown>;
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

/** Coerce a field that must be a finite number IF PRESENT: absent ⇒ undefined; present-but-not-a-number ⇒
 *  throw (never a best-effort drop). */
function numField(v: unknown, field: string): number | undefined {
  if (v === undefined) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) throw new LlmConfigParseError(`${field} must be a finite number`);
  return v;
}

/** Coerce a value that must be a PLAIN OBJECT (not null / array): throw with `<what> must be an object`
 *  (+ optional hint) otherwise. The object-shaped sibling of {@link numField}. */
function requireObject(v: unknown, what: string, hint = ""): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) throw new LlmConfigParseError(`${what} must be an object${hint}`);
  return v as Record<string, unknown>;
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
    return { type: "provider", name: o.name, id: o.id as `${string}.${string}`, args: requireObject(o.args, `tools[${i}].args`) };
  }
  if (o.type !== undefined && o.type !== "function") throw new LlmConfigParseError(`tools[${i}].type must be "function" | "provider"`);
  requireObject(o.inputSchema, `tools[${i}].inputSchema`, " (JSON Schema)");
  if (o.description !== undefined && typeof o.description !== "string") throw new LlmConfigParseError(`tools[${i}].description must be a string`);
  if (o.strict !== undefined && typeof o.strict !== "boolean") throw new LlmConfigParseError(`tools[${i}].strict must be a boolean`);
  return {
    type: "function",
    name: o.name,
    ...(o.description !== undefined ? { description: o.description } : {}),
    inputSchema: o.inputSchema as Record<string, unknown>,
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
export function parseLlmConfig(json: unknown): LlmCallConfig {
  const j = requireObject(json, "config");
  if (typeof j.model !== "string" || j.model.length === 0) throw new LlmConfigParseError("model must be a non-empty string");

  // Strict surface: an unknown key is an ERROR, never silently dropped — a caller passing `temprature`
  // or a definition-layer field here must find out loudly, not run with the field ignored.
  const unknown = Object.keys(j).filter((k) => !CONFIG_KEYS.has(k));
  if (unknown.length > 0) {
    const defLayer = unknown.filter((k) => (LLM_DEFINITION_KEYS as readonly string[]).includes(k));
    throw new LlmConfigParseError(
      `unknown config key(s): ${unknown.join(", ")}` +
        (defLayer.length > 0
          ? ` — ${defLayer.join(", ")} are definition-layer fields (prompt/budget), split out by resolveConfig, not part of LlmConfiguration`
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

/** The DEFINITION-LAYER keys of an llm-call declaration: the prompt inputs + per-call budget that sit
 *  ALONGSIDE the config (`@declarative-ai/llm`'s `LlmCallDefinition = LlmCallConfig & CallPrompt`).
 *  `resolveConfig` splits these out of the merged layers (returned as `ResolveResult.definition`) so the
 *  config bag itself parses strictly. Typed loosely here — the precise (AI-SDK) types live in the llm
 *  package; core only names the seam. */
export const LLM_DEFINITION_KEYS = ["system", "prompt", "messages", "attachments", "timeoutMs"] as const;

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

/** Structural checks for the definition-layer fields split out by {@link resolveConfig} — loose (core is
 *  AI-SDK-free) but never silent: a present-but-malformed field throws. */
function checkDefinitionLayer(d: Record<string, unknown>): void {
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
      requireObject(o.data, `attachments[${i}].data`);
    });
  }
}

// --- Resolution (compose fragments → one config) ----------------------------

/** A named-config registry: `get` resolves an id to its (possibly-partial) raw property bag; `idOf` is a
 *  best-effort reverse lookup for PROVENANCE only (identity is the resolved content hash, never the id). */
export interface ConfigurationRegistry {
  get(id: string): Record<string, unknown> | undefined;
  idOf?(config: Record<string, unknown>): string | undefined;
}

/** A plain-map configuration registry. */
export class MapConfigurationRegistry implements ConfigurationRegistry {
  private readonly map = new Map<string, Record<string, unknown>>();
  set(id: string, config: Record<string, unknown>): this {
    this.map.set(id, config);
    return this;
  }
  get(id: string): Record<string, unknown> | undefined {
    return this.map.get(id);
  }
}

export interface ResolveResult {
  config: LlmCallConfig;
  /** DEFINITION-LAYER fields ({@link LLM_DEFINITION_KEYS}: system/prompt/messages/attachments/timeoutMs)
   *  found in the layers, merged low→high like the config keys and structurally checked. The caller layers
   *  these onto the parsed config to build the full call definition. */
  definition: Record<string, unknown>;
  /** Non-fatal notes from the merge (e.g. an overridden family clearing the opposite family's knobs). */
  warnings: string[];
}

/**
 * Compose config fragments into ONE valid {@link LlmCallConfig}: merge the raw property bags LOW→HIGH
 * (e.g. `[engineDefault, workflowDefault, registry.get(ref), inlineOverrides]`) — later layers win per key —
 * then {@link parseLlmConfig} the result (strict: throws on a malformed merged bag). The merge is
 * FAMILY-AWARE: a higher layer that introduces `reasoning` CLEARS accumulated sampling knobs (and a higher
 * sampling knob clears accumulated `reasoning`), each with a warning — "replace, don't explode". This is
 * "parse, don't validate" applied to composition: merge loosely, parse strictly. Absent layers are skipped.
 */
export function resolveConfig(layers: Array<Record<string, unknown> | undefined>): ResolveResult {
  const warnings: string[] = [];
  const merged: Record<string, unknown> = {};
  for (const layer of layers) {
    if (!layer) continue;
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
    Object.assign(merged, layer);
  }
  // Split the definition-layer fields (prompt inputs + budget) OUT of the bag so the config parses
  // strictly — anything left that isn't a config key then throws in `parseLlmConfig`.
  const definition: Record<string, unknown> = {};
  for (const k of LLM_DEFINITION_KEYS) {
    if (merged[k] !== undefined) {
      definition[k] = merged[k];
      delete merged[k];
    }
  }
  checkDefinitionLayer(definition);
  return { config: parseLlmConfig(merged), definition, warnings };
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
