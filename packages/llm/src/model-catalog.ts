/**
 * The per-model CATALOG (§5). Originally just a price table; now the single source of everything
 * the runtime knows about a model — cost, capabilities (which parameters it accepts / requires),
 * limits, and its structured-output schema profile. Prices change, providers get added, and models
 * get re-priced or re-capability'd, so this is an **updatable table**, not a frozen constant:
 * `ModelCatalog` supports `upsert`/`remove`/`load`, and is backed by the `models` DB table
 * (db/schema.ts) so the runtime can hydrate/override at startup without a code change. The in-code
 * `DEFAULT_MODELS` is just the seed.
 *
 * Matching is **longest-prefix-wins**, so order of insertion doesn't matter and a more specific
 * dated/variant prefix always beats a coarser one. A concrete `vendor/model` id is just a maximally
 * specific prefix.
 */
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

/** Input/output modalities a model accepts/produces (OpenRouter `architecture.*_modalities`). */
export interface Modalities {
  input?: string[];
  output?: string[];
}

/**
 * Everything the runtime knows about ONE model (matched longest-prefix-wins on `modelPrefix`).
 * The rate fields ({@link RateSet}) drive cost; the capability fields drive routing and the
 * structured-output decision. Every non-rate field is OPTIONAL — a row that only knows a price
 * still works (cost computes; capabilities fall back to heuristics), and the §5 refresh fills the
 * rest in over time.
 */
export interface ModelInfo extends RateSet {
  /** Model-id prefix this row applies to. Longest matching prefix wins. */
  modelPrefix: string;
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
  /** Routing family: "anthropic" (native provider) | "openrouter" (catch-all). Derivable from the id. */
  family?: string;
  /** Display vendor ("OpenAI", "Anthropic", …) — the row label in the picker grid. */
  provider?: string;
  /** Short chip label (the id with any `vendor/` prefix dropped). */
  label?: string;
  /**
   * Provider-NEUTRAL model id, the SAME across every serving route for one underlying model — so the
   * native `claude-opus-4-8` row and the OpenRouter `anthropic/claude-opus-4.8` row share
   * `canonicalId: "claude-opus-4-8"`. Derived by {@link canonicalIdFor} (drop any `vendor/` prefix,
   * normalize dots→hyphens). Lets the picker collapse the same model's routes and lets capability data
   * be reconciled across them. Absent ⇒ derive from `modelPrefix`.
   */
  canonicalId?: string;
  /**
   * WHO SERVES this row — the route the call takes: "anthropic" (native Anthropic API) or "openrouter".
   * This is what distinguishes an OpenRouter-served Opus row from a Claude-API-served one (they carry
   * different prices). Same value as {@link family} today (routing is derived from the id string, not
   * this column); kept as an explicit, queryable field for the catalog/UI. Derived by
   * {@link servingProviderFor}. Absent ⇒ derive from `modelPrefix`.
   */
  servingProvider?: string;

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
   * reasoning-family fallback in {@link ModelCatalog.supportedParametersFor}).
   */
  supportedParameters?: string[];
  /** Parameters the model MUST be sent (mandatory) — always included; their absence is an error. */
  requiredParameters?: string[];
  /**
   * The RESOLVED structured-output schema profile (§5.1) for this model. Stored in the DB as a
   * `json_artifacts` reference (`$base`-composable so a shared base profile is referenced by many
   * models); resolved to a whole object at hydrate time so the engine reads it synchronously.
   * Absent ⇒ the static `profileForModelId` family fallback applies.
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

function bareModelId(modelId: string): string {
  // Drop any `vendor/` routing prefix (e.g. OpenRouter ids).
  return modelId.includes("/") ? modelId.slice(modelId.lastIndexOf("/") + 1) : modelId;
}

/**
 * Heuristic: OpenAI reasoning families (GPT-5*, o1/o3/o4*) reject sampling params (`temperature`,
 * `top_p`, `top_k`) — no OpenRouter endpoint lists them as supported. Used ONLY as a cold-start
 * fallback for {@link ModelCatalog.supportedParametersFor} when the §5 refresh hasn't yet recorded
 * the model's real `supportedParameters`; the recorded data always wins when present.
 */
export function isReasoningModel(modelId: string): boolean {
  const bare = bareModelId(modelId).toLowerCase();
  return /^gpt-5/.test(bare) || /^o[1-9]/.test(bare);
}

/** Provider-NEUTRAL canonical id for a model prefix (drop any `vendor/` prefix, dots→hyphens) — the key
 *  that collapses the same model's native + OpenRouter routes (see {@link ModelInfo.canonicalId}). */
export function canonicalIdFor(modelPrefix: string): string {
  return bareModelId(modelPrefix).toLowerCase().replace(/\./g, "-");
}

/** The serving route for a model prefix — mirrors the router's id-string rule (`router.familyForModel`):
 *  a BARE `claude-*` id routes to the native Anthropic API; anything with a `vendor/` prefix (incl.
 *  `anthropic/claude-…` on OpenRouter) and every bare non-Anthropic id routes via OpenRouter. So the
 *  native and OpenRouter rows for the same model get DIFFERENT `servingProvider`. See {@link ModelInfo.servingProvider}. */
export function servingProviderFor(modelPrefix: string): string {
  return modelPrefix.startsWith("claude-") ? "anthropic" : "openrouter";
}

/**
 * Fill the DERIVED IDENTITY fields onto a row — {@link ModelInfo.canonicalId}, {@link ModelInfo.servingProvider},
 * and (so a bare seed row is self-describing) `family` / `provider` / `label`. Fill-only: values a parser
 * set explicitly are kept. This is what an OpenRouter row needs (its capabilities come from the feed).
 */
export function deriveIdentity(row: ModelInfo): ModelInfo {
  const servingProvider = row.servingProvider ?? servingProviderFor(row.modelPrefix);
  return {
    ...row,
    canonicalId: row.canonicalId ?? canonicalIdFor(row.modelPrefix),
    servingProvider,
    // `family` is the routing target — the same value as `servingProvider`.
    family: row.family ?? servingProvider,
    provider: row.provider ?? displayProviderFor(row.modelPrefix),
    label: row.label ?? bareModelId(row.modelPrefix),
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
 * Display vendor for a model id — the row label in the picker grid. An OpenRouter `vendor/model` id
 * resolves by its vendor slug; a bare id is matched by a known family prefix. Unknown ⇒ "Other", so a
 * newly-priced family still shows up before this table learns about it. Single source of truth for both
 * the §5 refresh (which records it) and `/api/models` (which displays it).
 */
export function displayProviderFor(modelPrefix: string): string {
  if (modelPrefix.includes("/")) {
    const vendor = modelPrefix.slice(0, modelPrefix.indexOf("/")).toLowerCase();
    return VENDOR_NAMES[vendor] ?? vendor.charAt(0).toUpperCase() + vendor.slice(1);
  }
  if (modelPrefix.startsWith("claude")) return "Anthropic";
  if (modelPrefix.startsWith("gpt") || modelPrefix.startsWith("o3") || modelPrefix.startsWith("o4")) return "OpenAI";
  if (modelPrefix.startsWith("gemini") || modelPrefix.startsWith("palm")) return "Google";
  if (modelPrefix.startsWith("llama")) return "Meta";
  if (modelPrefix.startsWith("mistral") || modelPrefix.startsWith("mixtral") || modelPrefix.startsWith("magistral")) return "Mistral";
  if (modelPrefix.startsWith("deepseek")) return "DeepSeek";
  if (modelPrefix.startsWith("grok")) return "xAI";
  if (modelPrefix.startsWith("nova")) return "Amazon";
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

export class ModelCatalog {
  private readonly rows = new Map<string, ModelInfo>();
  private readonly cacheReadMultiplier: number;
  private readonly cacheWriteMultiplier: number;
  private readonly cacheWrite1hMultiplier: number;

  constructor(seed: ModelInfo[] = [], opts: PricingOptions = {}) {
    this.cacheReadMultiplier = opts.cacheReadMultiplier ?? 0.1;
    this.cacheWriteMultiplier = opts.cacheWriteMultiplier ?? 1.25;
    this.cacheWrite1hMultiplier = opts.cacheWrite1hMultiplier ?? 2.0;
    this.load(seed);
  }

  /** Insert or replace a model row (keyed by `modelPrefix`). */
  upsert(row: ModelInfo): void {
    this.rows.set(row.modelPrefix, row);
  }

  /** Bulk upsert — e.g. hydrate from the `models` table at startup. */
  load(rows: ModelInfo[]): void {
    for (const row of rows) this.upsert(row);
  }

  remove(modelPrefix: string): void {
    this.rows.delete(modelPrefix);
  }

  list(): ModelInfo[] {
    return [...this.rows.values()];
  }

  private matchAgainst(id: string): ModelInfo | undefined {
    let best: ModelInfo | undefined;
    for (const row of this.rows.values()) {
      if (id.startsWith(row.modelPrefix) && (!best || row.modelPrefix.length > best.modelPrefix.length)) {
        best = row;
      }
    }
    return best;
  }

  /**
   * Match the FULL id (incl. any `vendor/` prefix) first, so a vendor-specific row —
   * e.g. an OpenRouter price keyed `anthropic/claude-...`, which carries OpenRouter's
   * markup — wins when present. Fall back to the bare id so a direct-provider price
   * still serves as an approximation when no vendor-specific row exists.
   */
  private match(modelId: string): ModelInfo | undefined {
    return this.matchAgainst(modelId) ?? this.matchAgainst(bareModelId(modelId));
  }

  /** The full catalog row for a model id (longest-prefix match), or undefined if unknown. */
  lookup(modelId: string): ModelInfo | undefined {
    return this.match(modelId);
  }

  hasPricing(modelId: string): boolean {
    return this.match(modelId) !== undefined;
  }

  /**
   * Parameters the model MAY be sent — read from the CATALOG (the source of truth): the recorded
   * `supported_parameters`, seeded by migration 0011 for every model up to today and kept current by the
   * §5 refresh. Absent a recorded row, an OpenAI reasoning model falls back to its sampling-less set
   * (cold-start heuristic for models the refresh hasn't reached); any other unknown model returns
   * `undefined` ⇒ the caller sends everything. Native Claude capabilities are NOT computed here — they
   * live in the `models` table (see `model-catalog-source.claudeSupportedParameters`, which only the ingestion
   * path knows), so a fresh Claude model gets correct caps once it's in the table.
   */
  supportedParametersFor(modelId: string): string[] | undefined {
    const row = this.match(modelId);
    if (row?.supportedParameters && row.supportedParameters.length > 0) return row.supportedParameters;
    if (isReasoningModel(modelId)) return REASONING_FALLBACK_SUPPORTED;
    return undefined;
  }

  /** Parameters the model MUST be sent (mandatory), or undefined if none recorded. */
  requiredParametersFor(modelId: string): string[] | undefined {
    return this.match(modelId)?.requiredParameters;
  }

  /** The recorded (already-resolved) structured-output schema profile for a model, if any. */
  schemaProfileFor(modelId: string): ProviderSchemaProfile | undefined {
    return this.match(modelId)?.schemaProfile;
  }

  /** Input/output modalities a model accepts/produces (synced from `architecture.*_modalities`), if known.
   *  Used to gate media inputs and validate output-modality requests. */
  modalitiesFor(modelId: string): Modalities | undefined {
    return this.match(modelId)?.modalities;
  }

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
  computeCost(modelId: string, usage: UsageForCost): number | null {
    const p = this.match(modelId);
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
    row: ModelInfo,
    totalInputTokens: number,
  ): { input: number; output: number; cacheRead: number; cacheWrite: number; cacheWrite1h: number } {
    const lc = row.longContext;
    const set: RateSet =
      lc && totalInputTokens > lc.thresholdTokens ? lc : row;
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
  computeCostUsd(modelId: string, inputTokens: number | null, outputTokens: number | null): number | null {
    return this.computeCost(modelId, { inputTokens, outputTokens });
  }
}

/**
 * Seed catalog — a STARTING point, kept current by the §5 refresh (scrapes the Anthropic docs
 * pricing table + OpenRouter models API into `models`, which hydrates `defaultModelCatalog` at
 * startup). Values verified against https://platform.claude.com/docs/en/about-claude/pricing
 * (2026-06-02). The seed carries PRICE only; capabilities (supportedParameters / schemaProfile)
 * are filled by the refresh + the profile-artifact seed, with heuristic fallbacks until then.
 *
 * NB current Anthropic models (Opus 4.6+/Sonnet 4.6) include the full 1M context window at
 * STANDARD pricing — there is NO long-context premium today, so no row seeds `longContext`
 * (the tier mechanism stays for when a model reintroduces one). Opt-in modifiers NOT seeded
 * (applied by the caller when used): data-residency `inference_geo:"us"` ×1.1, Fast mode,
 * Batch ×0.5. Update via `defaultModelCatalog.upsert(...)` or the `models` table.
 */
const RAW_DEFAULT_MODELS: ModelInfo[] = [
  // Anthropic (input / output $/MTok; cache rates derive 0.1x/1.25x/2x).
  { modelPrefix: "claude-opus-4-8", inputPerMillion: 5, outputPerMillion: 25 },
  { modelPrefix: "claude-opus-4-7", inputPerMillion: 5, outputPerMillion: 25 },
  { modelPrefix: "claude-opus-4-6", inputPerMillion: 5, outputPerMillion: 25 },
  { modelPrefix: "claude-opus-4-5", inputPerMillion: 5, outputPerMillion: 25 },
  { modelPrefix: "claude-opus-4-1", inputPerMillion: 15, outputPerMillion: 75 },
  { modelPrefix: "claude-opus-4-0", inputPerMillion: 15, outputPerMillion: 75 },
  { modelPrefix: "claude-sonnet-4-8", inputPerMillion: 3, outputPerMillion: 15 },
  { modelPrefix: "claude-sonnet-4-6", inputPerMillion: 3, outputPerMillion: 15 },
  { modelPrefix: "claude-sonnet-4-5", inputPerMillion: 3, outputPerMillion: 15 },
  { modelPrefix: "claude-sonnet-4-2", inputPerMillion: 3, outputPerMillion: 15 },
  { modelPrefix: "claude-sonnet-4-0", inputPerMillion: 3, outputPerMillion: 15 },
  { modelPrefix: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 },
  { modelPrefix: "claude-haiku-3-5", inputPerMillion: 0.8, outputPerMillion: 4 },
  { modelPrefix: "claude-3-haiku", inputPerMillion: 0.25, outputPerMillion: 1.25 },

  // OpenAI (direct or via OpenRouter ids).
  { modelPrefix: "gpt-4o-mini", inputPerMillion: 0.15, outputPerMillion: 0.6 },
  { modelPrefix: "gpt-4o", inputPerMillion: 2.5, outputPerMillion: 10 },
  { modelPrefix: "gpt-4.1-nano", inputPerMillion: 0.1, outputPerMillion: 0.4 },
  { modelPrefix: "gpt-4.1-mini", inputPerMillion: 0.4, outputPerMillion: 1.6 },
  { modelPrefix: "gpt-4.1", inputPerMillion: 2, outputPerMillion: 8 },
  { modelPrefix: "gpt-5-nano", inputPerMillion: 0.25, outputPerMillion: 1 },
  { modelPrefix: "gpt-5-mini", inputPerMillion: 1, outputPerMillion: 4 },
  { modelPrefix: "gpt-5", inputPerMillion: 3.75, outputPerMillion: 15 },
  { modelPrefix: "o3-mini", inputPerMillion: 1.1, outputPerMillion: 4.4 },
  { modelPrefix: "o3", inputPerMillion: 10, outputPerMillion: 40 },
  { modelPrefix: "o4-mini", inputPerMillion: 1.1, outputPerMillion: 4.4 },

  // NB other providers (Google, Meta, DeepSeek, Amazon, Mistral, xAI, …) are intentionally NOT
  // seeded here: their authoritative prices + ids + `releasedAt` + capabilities come from the §5
  // OpenRouter refresh (model-catalog-source.ts) into `models`. Hand-seeding them would risk a fabricated
  // price the refresh never corrects (when its real `vendor/model` id differs from the guess).

  // Embeddings (§3.6 kernel).
  { modelPrefix: "text-embedding-3-small", inputPerMillion: 0.02, outputPerMillion: 0 },
  { modelPrefix: "text-embedding-3-large", inputPerMillion: 0.13, outputPerMillion: 0 },
];

/**
 * The seed catalog — PRICE + generic identity only (canonicalId / servingProvider / family / provider /
 * label via {@link deriveIdentity}). It carries NO capabilities: `supported_parameters` is the `models`
 * TABLE's job (seeded by migration 0011, refreshed by §5), and the ONLY code that synthesizes native
 * Claude caps is the ingestion path (`model-catalog-source.claudeSupportedParameters`). At runtime the table is
 * hydrated over this seed; this stays the bootstrap fallback for COST (OpenRouter also self-reports cost).
 */
export const DEFAULT_MODELS: ModelInfo[] = RAW_DEFAULT_MODELS.map(deriveIdentity);

/** The process-wide model catalog. Hydrate/override from `models` at startup + lazily at first read. */
export const defaultModelCatalog = new ModelCatalog(DEFAULT_MODELS);

/** Seed rows for the `models` table (applied with a live DB). */
export function modelsSeed(): ModelInfo[] {
  return DEFAULT_MODELS.map((r) => ({ ...r }));
}

// Convenience free functions delegating to the default catalog.
export function computeCostUsd(
  modelId: string,
  inputTokens: number | null,
  outputTokens: number | null,
): number | null {
  return defaultModelCatalog.computeCostUsd(modelId, inputTokens, outputTokens);
}

/** Billing-accurate cost from a token breakdown (cache-aware). See {@link ModelCatalog.computeCost}. */
export function computeCost(modelId: string, usage: UsageForCost): number | null {
  return defaultModelCatalog.computeCost(modelId, usage);
}

export function hasPricing(modelId: string): boolean {
  return defaultModelCatalog.hasPricing(modelId);
}

/** Parameters a model MAY be sent (capability), with the reasoning-family fallback. */
export function supportedParametersFor(modelId: string): string[] | undefined {
  return defaultModelCatalog.supportedParametersFor(modelId);
}

/** The capability gate for one model's optional params — the SINGLE implementation shared by `plan` (fit
 *  reporting) and `executeStructuredCall` (param filtering), so the dry-run can never drift from what
 *  execution actually sends. An unknown model (no catalog row) accepts everything — the prior behavior. */
export interface ParamAcceptance {
  accepts(key: keyof typeof SAMPLING_PARAM_NAMES): boolean;
  acceptsReasoning: boolean;
}

export function paramAcceptanceFor(modelId: string): ParamAcceptance {
  const supported = defaultModelCatalog.supportedParametersFor(modelId);
  return {
    accepts: (key) => supported === undefined || supported.includes(SAMPLING_PARAM_NAMES[key]),
    acceptsReasoning: supported === undefined || supported.includes("reasoning"),
  };
}

/** Parameters a model MUST be sent (mandatory). */
export function requiredParametersFor(modelId: string): string[] | undefined {
  return defaultModelCatalog.requiredParametersFor(modelId);
}

/** Input/output modalities a model accepts/produces, if known. See {@link ModelCatalog.modalitiesFor}. */
export function modalitiesFor(modelId: string): Modalities | undefined {
  return defaultModelCatalog.modalitiesFor(modelId);
}
