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
 */
import { GENERATED_MODELS } from "./model-catalog-data.generated";
import type { ModelRoute } from "./router";
import type { ProviderSchemaProfile } from "./schema/profile";

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
   * Parameters the model MAY be sent (capability) — OpenRouter snake_case names, e.g.
   * `["temperature","top_p","response_format","structured_outputs"]`. The executor FILTERS the
   * sampling params it sends down to this set, so a param no endpoint accepts (e.g. `temperature`
   * on a reasoning model) is never sent — which is what keeps OpenRouter's `require_parameters`
   * routing from rejecting the call (§5.1). Absent ⇒ unknown (send everything; see the
   * reasoning-family fallback in {@link ModelInfo.supportedParameters}).
   */
  supportedParameters?: readonly string[];
  /** Parameters the model MUST be sent (mandatory) — always included; their absence is an error. */
  requiredParameters?: readonly string[];
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
 * fallback for {@link ModelInfo.supportedParameters} when the §5 refresh hasn't yet recorded the
 * model's real `supportedParameters`; the recorded data always wins when present. Accepts either a
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

/** OpenRouter parameter NAMES for the sampling params the executor may filter out (our key → OR name). */
export const SAMPLING_PARAM_NAMES = {
  temperature: "temperature",
  topP: "top_p",
  topK: "top_k",
  stopSequences: "stop",
  presencePenalty: "presence_penalty",
  frequencyPenalty: "frequency_penalty",
  seed: "seed",
} as const;

/** Fallback `supportedParameters` for a reasoning model with no recorded capabilities: everything a
 *  reasoning endpoint typically takes EXCEPT the sampling params it rejects (so they get filtered). */
const REASONING_FALLBACK_SUPPORTED = [
  "max_tokens",
  "response_format",
  "structured_outputs",
  "reasoning",
  "seed",
  "tools",
  "tool_choice",
];

/** The capability gate for one model's optional params — the SINGLE implementation shared by `plan` (fit
 *  reporting) and `executeStructuredCall` (param filtering), so the dry-run can never drift from what
 *  execution actually sends. An unknown model (no catalog row) accepts everything — the prior behavior. */
export interface ParamAcceptance {
  accepts(key: keyof typeof SAMPLING_PARAM_NAMES): boolean;
  acceptsReasoning: boolean;
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
   * Parameters the model MAY be sent — the recorded `supported_parameters`, else (for an OpenAI reasoning
   * model) its sampling-less fallback set, else `undefined` ⇒ the caller sends everything. Native Claude
   * capabilities live in the row (synthesized by the ingestion path in `model-catalog-source`), not here.
   */
  supportedParameters(model: ModelKeyOf<Rows>): readonly string[] | undefined {
    const row = this.rows.get(model);
    if (row?.supportedParameters && row.supportedParameters.length > 0) return row.supportedParameters;
    if (isReasoningModel(model)) return REASONING_FALLBACK_SUPPORTED;
    return undefined;
  }

  /** Parameters the model MUST be sent (mandatory), or undefined if none recorded. */
  requiredParameters(model: ModelKeyOf<Rows>): readonly string[] | undefined {
    return this.rows.get(model)?.requiredParameters;
  }

  /** The capability gate for one model's optional params (see {@link ParamAcceptance}). Unknown model ⇒
   *  accepts everything (`supportedParameters` undefined). */
  paramAcceptance(model: ModelKeyOf<Rows>): ParamAcceptance {
    const supported = this.supportedParameters(model);
    return {
      accepts: (key) => supported === undefined || supported.includes(SAMPLING_PARAM_NAMES[key]),
      acceptsReasoning: supported === undefined || supported.includes("reasoning"),
    };
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
 * carries PRICE only; capabilities (supportedParameters / schemaProfile) are filled by the refresh +
 * the profile-artifact seed, with heuristic fallbacks until then.
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
export const CORE_SEED_MODELS: ModelInfoInterface[] = [
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
];

/**
 * The default constructor data for {@link ModelInfo} — the committed snapshot produced by
 * `npm run update:model-info` ({@link GENERATED_MODELS} in `model-catalog-data.generated.ts`). It is a
 * strongly-typed `as const` tuple, so `new ModelInfo(DEFAULT_MODELS)` (and any literal construction) gets
 * compile-time-checked model keys; `ModelInfo.instance` widens it to the weak (string-keyed) case so the
 * runtime consumers that pass a `string` model id still compile. Re-run the generator to refresh prices +
 * capabilities from the live sources.
 */
export const DEFAULT_MODELS = GENERATED_MODELS;

/** The union of every `${route}/${model}` key present in the committed snapshot — a compile-time enum of
 *  the models the runtime knows about out of the box. */
export type KnownModelKey = ModelKeyOf<typeof GENERATED_MODELS>;

/** Seed rows (a fresh copy per call). */
export function modelsSeed(): ModelInfoInterface[] {
  return DEFAULT_MODELS.map((r) => ({ ...r }));
}
