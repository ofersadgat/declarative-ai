/**
 * Keeping the §5 model catalog current — from what each route REPORTS about itself (JaiRA decision
 * 0009). Providers re-price models, launch new ones, and change capabilities; a client refreshes its
 * catalog from those reports rather than waiting on anyone to publish a list.
 *
 * The sources here are the provider ones; an agent's rows come from the agent itself (`claude`'s
 * initialize answer, codex's `model/list` — `@declarative-ai/agents-cli`), and a locally-served model's
 * from its server or its weights (the host). Every source answers the same {@link CatalogSource} shape,
 * and {@link refreshModelCatalog} merges them all into one table.
 *
 *  - **OpenRouter** `/api/v1/models` — public, no key. Prices (1-hour cache writes and long-context
 *    tiers included), context, the parameter names and a per-model `reasoning` object. Its `anthropic/*`
 *    and `openai/*` rows are ALSO the native routes' rows ({@link nativeMirrors}): OpenRouter passes the
 *    providers' prices through — measured 2026-09-24, its Claude rows equal Anthropic's docs on input,
 *    output, cache read and 5-minute write — and neither provider publishes prices of its own.
 *  - **Anthropic** `/v1/models` — needs a key. No prices, but the capabilities OpenRouter cannot know:
 *    exact effort levels, thinking modes (adaptive-only from Claude 4.7 on), context and output limits.
 *    Merged OVER the OpenRouter mirror of the same row, so each field comes from the source that knows it.
 *
 * Everything here is pure except the injected `FetchText`, so it unit-tests against recorded payloads
 * with no network. The orchestrator never overwrites good data with a bad read: a source that fails to
 * fetch, parse or VALIDATE is skipped, and the rows it wrote before stand.
 */
import type { JsonValue, SchemaDocument } from "@declarative-ai/json";
import { createLogger } from "@declarative-ai/log";
import {
  canonicalIdFor,
  deriveIdentity,
  displayProviderFor,
  hasRates,
  keyForModel,
  ModelInfo,
  orderedEfforts,
  parametersFromNames,
  structuredOutputFromNames,
  type ModelInfoInterface,
  type ReasoningCapability,
} from "./model-catalog.js";

const log = createLogger("engine.providers.model-catalog-source");

/** Fetches a URL's body as text. Injected so the core is testable and engine stays fetch-agnostic. */
export type FetchText = (url: string, init?: { headers?: Record<string, string> }) => Promise<string>;

/** A source of catalog rows for some set of models — prices, capabilities, or both. */
export interface CatalogSource {
  readonly name: string;
  /** Fetch + parse to rows. Throws on fetch/parse failure (the orchestrator isolates it). */
  fetchRows(): Promise<ModelInfoInterface[]>;
  /**
   * `false`/unset: a SMALL curated set — any bad row fails the whole batch. `true` (OpenRouter): a
   * LARGE heterogeneous set — drop individual bad rows and keep the good ones, rather than reject
   * hundreds for one.
   */
  readonly lenient?: boolean;
  /**
   * `true`: this source's rows FILL — they refresh a row's rates and supply the fields it lacks, and
   * never overwrite anything else a row states. For a source that describes another route second-hand
   * (OpenRouter's mirror of the native routes), so the route's OWN report — Anthropic's list — keeps its
   * say even on a refresh that could not reach it.
   */
  readonly fills?: boolean;
}

export const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
export const ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models";

// --- OpenRouter -------------------------------------------------------------

/**
 * Per-token cost string → per-million number; `undefined` for missing / `-1` (unavailable) / NaN.
 * Rounds to 6 decimals so the `× 1e6` doesn't leave float noise (e.g. 0.0000004 → 0.4, not 0.3999…).
 */
function perMillion(v: unknown): number | undefined {
  if (typeof v !== "string" && typeof v !== "number") return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return Math.round(n * 1_000_000 * 1_000_000) / 1_000_000;
}

/**
 * Parse OpenRouter's `/api/v1/models` (machine-readable, authoritative for OpenRouter routing).
 * Pricing is PER-TOKEN strings; `-1` means dynamic/unavailable (dropped). Rows are keyed by the
 * full `provider/model` id, so an OpenRouter-routed call gets OpenRouter's rate. This source is
 * `lenient` — odd individual rows are dropped by the orchestrator, not fatal.
 *
 * Beyond price, the feed carries the model's CAPABILITIES — `supported_parameters` and the `reasoning`
 * object (`supported_efforts`, `default_effort`, `mandatory`), which become the row's `parameters`
 * schema the executor filters and fits a call against (§5.1); `context_length`;
 * `top_provider.max_completion_tokens`; and `architecture.{input,output}_modalities`.
 */
export function parseOpenRouterModels(json: string): ModelInfoInterface[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return [];
  }
  const data = (parsed as { data?: unknown }).data;
  if (!Array.isArray(data)) return [];
  const out: ModelInfoInterface[] = [];
  for (const m of data) {
    const model = m as {
      id?: unknown;
      pricing?: Record<string, unknown> & { overrides?: unknown };
      created?: unknown;
      context_length?: unknown;
      supported_parameters?: unknown;
      reasoning?: unknown;
      top_provider?: { max_completion_tokens?: unknown };
      architecture?: { input_modalities?: unknown; output_modalities?: unknown };
    };
    const id = model.id;
    const pricing = model.pricing;
    if (typeof id !== "string" || !pricing) continue;
    const input = perMillion(pricing.prompt);
    const output = perMillion(pricing.completion);
    if (input === undefined || output === undefined) continue;
    const row: ModelInfoInterface = {
      // OpenRouter ids are always `vendor/model`, and they always route through OpenRouter (never the
      // native Anthropic provider — a claude-via-OR id is reached as `openrouter/anthropic/claude-…`).
      route: "openrouter",
      model: id,
      inputPerMillion: input,
      outputPerMillion: output,
      provider: displayProviderFor(id),
      label: id.includes("/") ? id.slice(id.indexOf("/") + 1) : id,
      source: "openrouter-models",
    };
    const read = perMillion(pricing.input_cache_read);
    const write = perMillion(pricing.input_cache_write);
    const write1h = perMillion(pricing.input_cache_write_1h);
    if (read !== undefined) row.cacheReadPerMillion = read;
    if (write !== undefined) row.cacheWritePerMillion = write;
    if (write1h !== undefined) row.cacheWrite1hPerMillion = write1h;
    // OpenRouter lists every rate it charges. A cached READ with no WRITE is a provider whose caching is
    // automatic and free to fill (OpenAI's is) — stated as 0 here, because the class's defaults are
    // Anthropic's ratios and would bill a write at 1.25x input that nobody was charged for.
    if (read !== undefined && write === undefined) {
      row.cacheWritePerMillion = 0;
      row.cacheWrite1hPerMillion = 0;
    }
    const longContext = openRouterLongContext(pricing.overrides);
    if (longContext !== undefined) row.longContext = longContext;
    // OpenRouter reports `created` (UNIX seconds) — the model's release/listing time. Capture it so
    // the admin UI can sort newest-first; it never touches cost.
    const created = model.created;
    if (typeof created === "number" && Number.isFinite(created) && created > 0) row.releasedAt = created;
    // Capabilities + limits (all optional; absent ⇒ heuristic fallbacks apply downstream).
    const ctx = model.context_length;
    if (typeof ctx === "number" && Number.isFinite(ctx) && ctx > 0) row.contextLength = ctx;
    const maxOut = model.top_provider?.max_completion_tokens;
    if (typeof maxOut === "number" && Number.isFinite(maxOut) && maxOut > 0) row.maxOutputTokens = maxOut;
    const supported = model.supported_parameters;
    if (Array.isArray(supported) && supported.every((p) => typeof p === "string") && supported.length > 0) {
      const names = supported as string[];
      const reasoning = openRouterReasoning(model.reasoning);
      row.parameters = parametersFromNames(names, {
        ...(reasoning !== undefined ? { reasoning } : {}),
        ...(row.maxOutputTokens !== undefined ? { maxOutputTokens: row.maxOutputTokens } : {}),
      });
      row.structuredOutput = structuredOutputFromNames(names);
    }
    const inMods = model.architecture?.input_modalities;
    const outMods = model.architecture?.output_modalities;
    const modalities: { input?: string[]; output?: string[] } = {};
    if (Array.isArray(inMods) && inMods.every((x) => typeof x === "string")) modalities.input = inMods as string[];
    if (Array.isArray(outMods) && outMods.every((x) => typeof x === "string")) modalities.output = outMods as string[];
    if (modalities.input || modalities.output) row.modalities = modalities;
    out.push(deriveIdentity(row));
  }
  return out;
}

/**
 * OpenRouter's per-model `reasoning` object as a {@link ReasoningCapability}, or `undefined` when the
 * model has none (its names alone then say whether it reasons).
 *
 * MEASURED 2026-09-24: `{"mandatory":true,"supported_efforts":["max","xhigh","high","medium","low"],
 * "default_effort":"high"}` on `anthropic/claude-opus-5.5`. Levels arrive highest-first and in
 * OpenRouter's spelling, which is ours; one we do not know is dropped rather than guessed at.
 */
function openRouterReasoning(raw: unknown): ReasoningCapability | undefined {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const r = raw as { supported_efforts?: unknown; default_effort?: unknown; mandatory?: unknown };
  const efforts = Array.isArray(r.supported_efforts) ? orderedEfforts(r.supported_efforts.filter((e): e is string => typeof e === "string")) : undefined;
  const fallback = typeof r.default_effort === "string" ? orderedEfforts([r.default_effort])[0] : undefined;
  return {
    ...(efforts !== undefined && efforts.length > 0 ? { efforts } : {}),
    ...(fallback !== undefined ? { defaultEffort: fallback } : {}),
    ...(r.mandatory === true ? { mandatory: true } : {}),
  };
}

/**
 * The long-context tier from OpenRouter's `pricing.overrides` — MEASURED 2026-09-24 on
 * `openai/gpt-5.6-luna-pro`: `[{"min_prompt_tokens":272000,"prompt":"0.0000004","completion":"0.0000018",
 * "input_cache_read":"0.00000004","input_cache_write":"0.0000005"}]`. The first override that names a
 * threshold and both base rates; anything else is not a tier this catalog can price.
 */
function openRouterLongContext(raw: unknown): ModelInfoInterface["longContext"] {
  if (!Array.isArray(raw)) return undefined;
  for (const o of raw) {
    if (o === null || typeof o !== "object") continue;
    const tier = o as Record<string, unknown>;
    const threshold = tier["min_prompt_tokens"];
    const input = perMillion(tier["prompt"]);
    const output = perMillion(tier["completion"]);
    if (typeof threshold !== "number" || input === undefined || output === undefined) continue;
    const read = perMillion(tier["input_cache_read"]);
    const write = perMillion(tier["input_cache_write"]);
    return {
      thresholdTokens: threshold,
      inputPerMillion: input,
      outputPerMillion: output,
      ...(read !== undefined ? { cacheReadPerMillion: read } : {}),
      ...(write !== undefined ? { cacheWritePerMillion: write } : {}),
    };
  }
  return undefined;
}

/**
 * OpenRouter's rows for a provider's OWN models, restated as that provider's native route: `anthropic/…`
 * as `anthropic/<id>` (OpenRouter's dotted `claude-opus-4.8` is the API's hyphenated `claude-opus-4-8`),
 * and `openai/…` as `openai/<id>` (spelled the same on both). Neither provider publishes prices as data,
 * and OpenRouter passes theirs through, so this is where the native routes' prices come from.
 *
 * A variant id (`anthropic/claude-3.7-sonnet:thinking`) is OpenRouter's own routing option, not a model
 * the provider serves, and is not mirrored.
 */
export function nativeMirrors(rows: readonly ModelInfoInterface[]): ModelInfoInterface[] {
  const out: ModelInfoInterface[] = [];
  for (const row of rows) {
    if (row.route !== "openrouter" || row.model.includes(":")) continue;
    const slash = row.model.indexOf("/");
    const vendor = row.model.slice(0, slash);
    const bare = row.model.slice(slash + 1);
    if (vendor === "anthropic") out.push({ ...row, route: "anthropic", model: canonicalIdFor(bare), label: canonicalIdFor(bare) });
    else if (vendor === "openai") {
      // OpenRouter turns a budget into `reasoning.max_tokens`; OpenAI's own API has only levels.
      const parameters = row.parameters !== undefined ? withoutBudget(row.parameters) : undefined;
      out.push({ ...row, route: "openai", model: bare, label: bare, ...(parameters !== undefined ? { parameters } : {}) });
    }
  }
  return out;
}

/** A `parameters` schema whose `reasoning` takes no budget. */
function withoutBudget(schema: SchemaDocument): SchemaDocument {
  const properties = schema["properties"] as Record<string, JsonValue> | undefined;
  const reasoning = properties?.["reasoning"] as { properties?: Record<string, JsonValue> } | undefined;
  if (reasoning?.properties?.["budgetTokens"] === undefined) return schema;
  const { budgetTokens: _dropped, ...rest } = reasoning.properties;
  return { ...schema, properties: { ...properties, reasoning: { ...reasoning, properties: rest } } };
}

/** The OpenRouter models source — its own rows, plus the native routes' mirrors. Lenient. */
/**
 * The OpenRouter sources: its own rows (lenient), and its mirrors of the native routes, which FILL
 * (see {@link CatalogSource.fills}). Two sources over ONE fetch — the mirror reads what the first just
 * fetched, and fetches for itself only when run alone.
 */
export function makeOpenRouterSources(fetchText: FetchText, url: string = OPENROUTER_MODELS_URL): [CatalogSource, CatalogSource] {
  let fetched: ModelInfoInterface[] | undefined;
  const own: CatalogSource = {
    name: "openrouter-models",
    lenient: true,
    async fetchRows() {
      fetched = parseOpenRouterModels(await fetchText(url));
      return fetched;
    },
  };
  const mirrors: CatalogSource = {
    name: "openrouter-native-mirrors",
    lenient: true,
    fills: true,
    async fetchRows() {
      const rows = fetched ?? parseOpenRouterModels(await fetchText(url));
      fetched = undefined;
      return nativeMirrors(rows);
    },
  };
  return [own, mirrors];
}

// --- Anthropic ----------------------------------------------------------------

/** opus-4-7/4-8 reject the sampling knobs (temperature/top_p/top_k) — MIRRORS `@ai-sdk/anthropic`'s
 *  `getModelCapabilities().rejectsSamplingParameters` (v3), which is where that rule lives; Anthropic's
 *  model list does not say. Dots→hyphens so the OpenRouter form matches; substring match covers dated
 *  and vendor ids. */
export function anthropicRejectsSampling(modelId: string): boolean {
  const bare = modelId.toLowerCase().replace(/\./g, "-");
  return bare.includes("claude-opus-4-8") || bare.includes("claude-opus-4-7");
}

/** The slice of one `/v1/models` entry this reads. */
interface AnthropicModel {
  id?: unknown;
  display_name?: unknown;
  created_at?: unknown;
  max_input_tokens?: unknown;
  max_tokens?: unknown;
  capabilities?: {
    effort?: { supported?: unknown } & Record<string, { supported?: unknown } | unknown>;
    thinking?: { supported?: unknown; types?: Record<string, { supported?: unknown } | undefined> };
    structured_outputs?: { supported?: unknown };
    image_input?: { supported?: unknown };
    pdf_input?: { supported?: unknown };
  };
}

const supported = (v: unknown): boolean => v !== null && typeof v === "object" && (v as { supported?: unknown }).supported === true;

/**
 * One `/v1/models` page's entries as native `anthropic/…` rows — capabilities and limits, no prices.
 *
 * MEASURED 2026-09-24 (`claude-opus-5-5`): `max_input_tokens: 1000000`, `max_tokens: 128000`,
 * `capabilities.effort.{low,medium,high,xhigh,max}.supported`, `capabilities.thinking.types` with only
 * `adaptive` supported. So the effort `enum` is exactly the levels marked supported, and a budget is
 * offered only where `enabled` thinking is. A DATED id (`claude-haiku-4-5-20251001`) also yields its
 * alias (`claude-haiku-4-5`), which the API accepts and which is what configs name.
 */
export function parseAnthropicModels(json: string): { rows: ModelInfoInterface[]; hasMore: boolean; lastId?: string } {
  let parsed: { data?: unknown; has_more?: unknown; last_id?: unknown };
  try {
    parsed = JSON.parse(json) as typeof parsed;
  } catch {
    return { rows: [], hasMore: false };
  }
  if (!Array.isArray(parsed.data)) return { rows: [], hasMore: false };
  const rows: ModelInfoInterface[] = [];
  const listed = new Set(parsed.data.map((m) => (m as AnthropicModel).id).filter((id): id is string => typeof id === "string"));
  for (const m of parsed.data as AnthropicModel[]) {
    if (typeof m.id !== "string" || m.id.length === 0) continue;
    const caps = m.capabilities ?? {};
    const maxOut = typeof m.max_tokens === "number" && m.max_tokens > 0 ? m.max_tokens : undefined;
    const efforts = supported(caps.effort)
      ? orderedEfforts(Object.keys(caps.effort ?? {}).filter((level) => supported((caps.effort as Record<string, unknown>)[level])))
      : [];
    const budget = supported(caps.thinking?.types?.["enabled"]);
    const reasons = efforts.length > 0 || supported(caps.thinking);
    const names = ["max_tokens", "stop", "tools", "tool_choice", ...(anthropicRejectsSampling(m.id) ? [] : ["temperature", "top_p", "top_k"])];
    const created = typeof m.created_at === "string" ? Date.parse(m.created_at) : NaN;
    const row: ModelInfoInterface = deriveIdentity({
      route: "anthropic",
      model: m.id,
      provider: "Anthropic",
      label: m.id,
      source: "anthropic-models",
      ...(Number.isFinite(created) ? { releasedAt: Math.floor(created / 1000) } : {}),
      ...(typeof m.max_input_tokens === "number" && m.max_input_tokens > 0 ? { contextLength: m.max_input_tokens } : {}),
      ...(maxOut !== undefined ? { maxOutputTokens: maxOut } : {}),
      parameters: parametersFromNames(names, {
        ...(maxOut !== undefined ? { maxOutputTokens: maxOut } : {}),
        // Anthropic's floor for a thinking budget is 1024, and it must sit below `max_tokens`.
        ...(reasons ? { reasoning: { efforts, budget: budget ? { minimum: 1024, ...(maxOut !== undefined ? { maximum: maxOut - 1 } : {}) } : false } } : {}),
      }),
      structuredOutput: supported(caps.structured_outputs) ? "schema" : false,
      modalities: {
        input: ["text", ...(supported(caps.image_input) ? ["image"] : []), ...(supported(caps.pdf_input) ? ["file"] : [])],
        output: ["text"],
      },
    });
    rows.push(row);
    const alias = m.id.replace(/-\d{8}$/, "");
    if (alias !== m.id && !listed.has(alias)) rows.push({ ...row, model: alias, label: alias });
  }
  const lastId = typeof parsed.last_id === "string" ? parsed.last_id : undefined;
  return { rows, hasMore: parsed.has_more === true, ...(lastId !== undefined ? { lastId } : {}) };
}

/**
 * The Anthropic models source — every page of `/v1/models`, read with the account's key. Strict: the
 * list is small, and a page that reads wrong is a reason to keep what the table has.
 */
export function makeAnthropicModelsSource(fetchText: FetchText, apiKey: string, url: string = ANTHROPIC_MODELS_URL): CatalogSource {
  return {
    name: "anthropic-models",
    async fetchRows() {
      const rows: ModelInfoInterface[] = [];
      let after: string | undefined;
      // Bounded: a listing that never says it is done must not spin a refresh forever.
      for (let page = 0; page < 20; page++) {
        const body = await fetchText(`${url}?limit=100${after !== undefined ? `&after_id=${encodeURIComponent(after)}` : ""}`, {
          headers: { "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
        });
        const read = parseAnthropicModels(body);
        rows.push(...read.rows);
        if (!read.hasMore || read.lastId === undefined) break;
        after = read.lastId;
      }
      return rows;
    },
  };
}

// --- Validation -------------------------------------------------------------

export interface CatalogValidation {
  ok: boolean;
  problems: string[];
}

/** Internal-consistency problems with a SINGLE row (empty array = sane). A row with no rates is sane —
 *  it is priced by another row, or not at all — so only rates that ARE present are checked. */
function rowProblems(r: ModelInfoInterface): string[] {
  const p: string[] = [];
  if (!r.model) p.push("empty model");
  if (r.inputPerMillion === undefined && r.outputPerMillion === undefined) return p;
  if (!hasRates(r)) {
    p.push("half a price (input or output rate missing)");
    return p;
  }
  if (!(r.inputPerMillion > 0)) p.push(`non-positive input ${r.inputPerMillion}`);
  // An output rate of EXACTLY 0 is legitimate, not a bad read: embedding / classifier / input-only
  // models bill no completion tokens (`text-embedding-3-small` is $0 output). So only a NEGATIVE
  // output is invalid, and the "output < input is suspect" heuristic — which assumes a generation model
  // whose completions cost more than its prompt — applies ONLY to models that charge for output.
  if (r.outputPerMillion < 0) p.push(`negative output ${r.outputPerMillion}`);
  else if (r.outputPerMillion > 0 && r.outputPerMillion < r.inputPerMillion) p.push("output < input (suspect)");
  if (r.inputPerMillion > 1000 || r.outputPerMillion > 1000) p.push("implausibly high rate");
  if (r.cacheReadPerMillion !== undefined && r.cacheReadPerMillion >= r.inputPerMillion) p.push("cache read ≥ input (should be a discount)");
  if (r.cacheWritePerMillion !== undefined && r.cacheWrite1hPerMillion !== undefined && r.cacheWrite1hPerMillion < r.cacheWritePerMillion)
    p.push("1h write < 5m write");
  return p;
}

/**
 * Hard sanity gate (STRICT sources) before any write — a bad read must NOT poison the table. Requires a
 * plausible row count and internally-consistent, non-duplicate rows. Any violation fails the WHOLE batch.
 */
export function validateCatalogRows(rows: ModelInfoInterface[], minRows = 3): CatalogValidation {
  const problems: string[] = [];
  if (rows.length < minRows) problems.push(`too few rows: ${rows.length} < ${minRows}`);
  const seen = new Set<string>();
  for (const r of rows) {
    const key = keyForModel(r);
    if (seen.has(key)) problems.push(`[${key}] duplicate model`);
    seen.add(key);
    for (const p of rowProblems(r)) problems.push(`[${key}] ${p}`);
  }
  return { ok: problems.length === 0, problems };
}

/**
 * Per-row filter (LENIENT sources): keep the sane, non-duplicate rows and report what was dropped. Used
 * for large heterogeneous feeds (OpenRouter) where one odd row must not sink the whole refresh.
 */
export function sanitizeCatalogRows(rows: ModelInfoInterface[]): {
  rows: ModelInfoInterface[];
  dropped: { model: string; problems: string[] }[];
} {
  const seen = new Set<string>();
  const kept: ModelInfoInterface[] = [];
  const dropped: { model: string; problems: string[] }[] = [];
  for (const r of rows) {
    const key = keyForModel(r);
    const problems = rowProblems(r);
    if (seen.has(key)) problems.push("duplicate model");
    if (problems.length > 0) {
      dropped.push({ model: key, problems });
      continue;
    }
    seen.add(key);
    kept.push(r);
  }
  return { rows: kept, dropped };
}

// --- Refresh orchestrator ---------------------------------------------------

export interface SourceOutcome {
  name: string;
  fetched: number;
  /** Rows that differed from the current table and were applied. */
  applied: number;
  /** True when the source was skipped (fetch/parse error or validation failure) — the table stands. */
  skipped: boolean;
  error?: string;
  problems?: string[];
}

export interface RefreshReport {
  bySource: SourceOutcome[];
  added: string[];
  updated: string[];
  /** The rows this refresh wrote, merged — what the caller persists to its `models` table. */
  changed: ModelInfoInterface[];
  /** The full row set after refresh. */
  rows: ModelInfoInterface[];
}

/**
 * A source's row laid OVER the table's: every field the source states wins, and every field it does not
 * state is kept. Two sources describe one row from different sides — OpenRouter's mirror prices
 * `anthropic/claude-opus-5-5`, Anthropic's own list says which levels it thinks at — and a whole-row
 * replace would let whichever ran second erase what the first knew.
 */
export function mergeRow(current: ModelInfoInterface | undefined, incoming: ModelInfoInterface, fills = false): ModelInfoInterface {
  if (current === undefined) return incoming;
  const merged: Record<string, unknown> = { ...current };
  for (const [key, value] of Object.entries(incoming)) {
    if (value === undefined) continue;
    // A FILLING source (see `CatalogSource.fills`) states prices and fills gaps, nothing more.
    if (fills && merged[key] !== undefined && !RATE_FIELDS.has(key)) continue;
    merged[key] = value;
  }
  return merged as unknown as ModelInfoInterface;
}

/** The fields a price is made of — what a filling source refreshes on a row that has them already. */
const RATE_FIELDS: ReadonlySet<string> = new Set([
  "inputPerMillion",
  "outputPerMillion",
  "cacheReadPerMillion",
  "cacheWritePerMillion",
  "cacheWrite1hPerMillion",
  "longContext",
]);

/**
 * Fetch every source, validate, and MERGE only the changed rows into `table` (default: the process-wide
 * `ModelInfo.instance`), in the order given — a later source's fields win where two state the same one.
 * A source that throws or fails validation is isolated and skipped: its failure never drops or corrupts
 * what the table holds. Returns a report, with the rows written, so the caller can persist them.
 */
export async function refreshModelCatalog(opts: {
  sources: CatalogSource[];
  table?: ModelInfo;
  /** Min rows a source must yield to be trusted (validation). */
  minRows?: number;
}): Promise<RefreshReport> {
  const table = opts.table ?? ModelInfo.instance;
  const bySource: SourceOutcome[] = [];
  const added: string[] = [];
  const updated: string[] = [];
  const changed = new Map<string, ModelInfoInterface>();

  const minRows = opts.minRows ?? 3;
  for (const source of opts.sources) {
    let fetched: ModelInfoInterface[];
    try {
      fetched = await source.fetchRows();
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      log.warn("catalog source fetch failed; keeping the table as it is", { source: source.name, error });
      bySource.push({ name: source.name, fetched: 0, applied: 0, skipped: true, error });
      continue;
    }

    // Decide the trusted row set: STRICT = all-or-nothing; LENIENT = drop bad rows individually.
    let toApply: ModelInfoInterface[];
    if (source.lenient) {
      const { rows: clean, dropped } = sanitizeCatalogRows(fetched);
      if (dropped.length > 0) log.debug("catalog source dropped rows", { source: source.name, dropped: dropped.length });
      if (clean.length < minRows) {
        log.warn("catalog source yielded too few valid rows; keeping the table as it is", { source: source.name, fetched: fetched.length, valid: clean.length });
        bySource.push({ name: source.name, fetched: fetched.length, applied: 0, skipped: true, problems: [`only ${clean.length} valid rows`] });
        continue;
      }
      toApply = clean;
    } else {
      const check = validateCatalogRows(fetched, minRows);
      if (!check.ok) {
        log.warn("catalog source failed validation; keeping the table as it is", { source: source.name, fetched: fetched.length, problems: check.problems });
        bySource.push({ name: source.name, fetched: fetched.length, applied: 0, skipped: true, problems: check.problems });
        continue;
      }
      toApply = fetched;
    }

    let applied = 0;
    for (const row of toApply) {
      const key = keyForModel(row);
      const current = table.lookup(key);
      const merged = mergeRow(current, row, source.fills === true);
      if (current !== undefined && JSON.stringify(current) === JSON.stringify(merged)) continue;
      if (current === undefined) added.push(key);
      else if (!added.includes(key) && !updated.includes(key)) updated.push(key);
      table.upsert(merged);
      changed.set(key, merged);
      applied++;
    }
    log.info("catalog source applied", { source: source.name, fetched: fetched.length, applied });
    bySource.push({ name: source.name, fetched: fetched.length, applied, skipped: false });
  }

  return { bySource, added, updated, changed: [...changed.values()], rows: table.list() };
}
