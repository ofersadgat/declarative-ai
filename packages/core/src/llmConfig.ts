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
 * Pure data — no provider/DB coupling — so it can be imported by the engine, the server assembly, and
 * the config UI alike.
 */

/** Universal LLM-call params every model accepts. */
export interface LlmConfiguration {
  model: string;
  maxOutputTokens?: number;
  stopSequences?: string[];
}

/** Sampling models — the decoding knobs a temperature-sampling endpoint accepts. */
export interface SamplingConfiguration extends LlmConfiguration {
  temperature?: number;
  topP?: number;
  topK?: number;
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
 *  reasoning model simply has no temperature/top-p branch, §config-as-dimensions). */
export interface ReasoningConfiguration extends LlmConfiguration {
  reasoning?: ReasoningSpec;
}

/** A single RESOLVED LLM call config — a model is sampling XOR reasoning, so the union is the honest
 *  point shape. `enumerateConfigPoints` produces these (capability-pruned per model). */
export type LlmCallConfig = SamplingConfiguration | ReasoningConfiguration;

/** The FLAT superset of the call-config variants — the serialized form (`model` always present; only the
 *  params that apply to the model populated). The shared shape of a stored config JSON: `buildConfigJson`
 *  writes it, `readLlmConfig` reads it back, and `ConfigPoint` (the config-space leaf) is an alias of it. */
export interface FlatLlmConfig extends LlmConfiguration {
  temperature?: number;
  topP?: number;
  topK?: number;
  reasoning?: ReasoningSpec;
}

/** Parse a stored `reasoning` blob into a typed `ReasoningSpec` (dropping malformed fields); undefined when
 *  neither effort nor budget is usable. Keys are reconstructed in a FIXED order so re-serialization is
 *  content-stable. */
export function readReasoningSpec(v: unknown): ReasoningSpec | undefined {
  if (!v || typeof v !== "object") return undefined;
  const o = v as { effort?: unknown; budgetTokens?: unknown };
  const effort = o.effort === "low" || o.effort === "medium" || o.effort === "high" ? o.effort : undefined;
  const budgetTokens = typeof o.budgetTokens === "number" && Number.isFinite(o.budgetTokens) ? o.budgetTokens : undefined;
  if (effort === undefined && budgetTokens === undefined) return undefined;
  return { ...(effort !== undefined ? { effort } : {}), ...(budgetTokens !== undefined ? { budgetTokens } : {}) };
}

/** Read a stored config JSON into a typed `FlatLlmConfig` — the inverse of `buildConfigJson` and the ONE
 *  place a config blob is parsed (replacing the ad-hoc `num()` field-picking that was duplicated across the
 *  executor, the op expander, and the candidate-view resolver). Malformed/unknown fields are dropped;
 *  `model` falls back to `""` (callers validate it is non-empty). */
export function readLlmConfig(json: Record<string, unknown> | undefined): FlatLlmConfig {
  const j = json ?? {};
  const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
  const strArr = (v: unknown): string[] | undefined => (Array.isArray(v) && v.every((x) => typeof x === "string") ? (v as string[]) : undefined);
  return {
    model: typeof j.model === "string" ? j.model : "",
    temperature: num(j.temperature),
    topP: num(j.topP),
    topK: num(j.topK),
    maxOutputTokens: num(j.maxOutputTokens),
    stopSequences: strArr(j.stopSequences),
    reasoning: readReasoningSpec(j.reasoning),
  };
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
