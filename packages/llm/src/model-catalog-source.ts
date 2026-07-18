/**
 * Keeping the §5 model catalog current. Providers re-price models, launch new ones, and change
 * capabilities, so a scheduled refresh (the app wires this to a daily Vercel Cron, §0) pulls
 * authoritative data from each provider's published source into the `models` table, which hydrates
 * `defaultModelCatalog` at startup.
 *
 * Two providers, two realities (see also the OpenRouter usage-accounting path in generate.ts):
 *  - **Anthropic** has NO pricing API — prices live in its docs table. `parseAnthropicDocsPricing`
 *    reads that table (markdown OR HTML) into `ModelInfo`s, INCLUDING the exact cache rates.
 *  - **OpenRouter** exposes `/api/v1/models` with machine-readable prices AND capabilities
 *    (supported_parameters, context length, modalities), captured by `parseOpenRouterModels`.
 *
 * Everything here is pure except the injected `FetchText`, so it unit-tests against a recorded
 * fixture with no network. The orchestrator NEVER overwrites good prices with a bad scrape: a
 * source that fails to fetch, parse, or VALIDATE is skipped, and existing prices stand.
 */
import { createLogger } from "./logger";
import { defaultModelCatalog, deriveIdentity, displayProviderFor, type ModelInfo, type ModelCatalog } from "./model-catalog";

const log = createLogger("engine.providers.model-catalog-source");

// --- Native Claude capability synthesis -------------------------------------
// The Anthropic docs table (and the in-code price seed) carry NO capability columns, and the native
// `claude-*` ids aren't on OpenRouter, so THIS module is the ONE place that knows Claude's default
// parameter support. What it synthesizes is written to the `models` TABLE — by the §5 refresh below and
// by the seed generator (`db/genModelsSeedSql.ts`, which imports these) — and the table is the runtime
// source of truth. Nothing reads this at call time; `ModelCatalog.supportedParametersFor` reads the row.

/** Drop any `vendor/` routing prefix (e.g. OpenRouter ids). */
function bareId(modelId: string): string {
  return modelId.includes("/") ? modelId.slice(modelId.lastIndexOf("/") + 1) : modelId;
}

/** Native Anthropic (`claude-*`) id — bare or vendor-prefixed. */
function isClaudeModel(modelId: string): boolean {
  return bareId(modelId).toLowerCase().startsWith("claude");
}

/** opus-4-7/4-8 reject the sampling knobs (temperature/top_p/top_k) — MIRRORS `@ai-sdk/anthropic`'s
 *  `getModelCapabilities().rejectsSamplingParameters` (v3). Dots→hyphens so the OpenRouter dotted form
 *  (`claude-opus-4.8`) matches the native hyphenated one; substring match covers dated/vendor ids. */
export function anthropicRejectsSampling(modelId: string): boolean {
  const bare = bareId(modelId).toLowerCase().replace(/\./g, "-");
  return bare.includes("claude-opus-4-8") || bare.includes("claude-opus-4-7");
}

/** Claude 3.7 Sonnet and every Claude 4+ family support extended thinking (`reasoning`); 3.5 / 3 do not. */
export function claudeSupportsThinking(modelId: string): boolean {
  const p = bareId(modelId).toLowerCase();
  if (/^claude-3-7-sonnet/.test(p)) return true; // old naming: claude-3-7-sonnet
  const m = p.match(/^claude-(?:opus|sonnet|haiku)-(\d+)/); // new naming: claude-{family}-{major}-…
  return m ? Number(m[1]) >= 4 : false;
}

/** Claude input/output modalities: text + image + document(file) in, text out (matches OpenRouter's
 *  `claude-*` rows, which list `file`). */
export const CLAUDE_MODALITIES = { input: ["text", "image", "file"], output: ["text"] } as const;

const CLAUDE_SAMPLING_PARAMS = ["temperature", "top_p", "top_k"];

/**
 * PERMISSIVE default `supported_parameters` a NEW/unseeded Claude model resolves to (accepts sampling +
 * thinking). Anthropic keeps caps STABLE across releases, so this covers the family; a specific model only
 * ever SUBTRACTS from it. Permissive so a wrong guess for a future model fails LOUD (an AI SDK warning you
 * fix in the DB) rather than SILENT (a real sampling dimension quietly dropped). Ordered for stable arrays.
 */
export const CLAUDE_DEFAULT_SUPPORTED = ["max_tokens", ...CLAUDE_SAMPLING_PARAMS, "stop", "tools", "tool_choice", "reasoning"];

/** The `supported_parameters` for a native Claude model: {@link CLAUDE_DEFAULT_SUPPORTED} MINUS the sampling
 *  knobs when {@link anthropicRejectsSampling} (opus-4-7/4-8) and MINUS `reasoning` when the model has no
 *  extended thinking ({@link claudeSupportsThinking}). */
export function claudeSupportedParameters(modelId: string): string[] {
  let params = [...CLAUDE_DEFAULT_SUPPORTED];
  if (anthropicRejectsSampling(modelId)) params = params.filter((p) => !CLAUDE_SAMPLING_PARAMS.includes(p));
  if (!claudeSupportsThinking(modelId)) params = params.filter((p) => p !== "reasoning");
  return params;
}

/**
 * Complete a SEED / scraped row for insertion into the `models` table: generic identity always
 * ({@link deriveIdentity}), PLUS — for a native Claude row with no capability feed — its synthesized
 * `supported_parameters` + modalities. Fill-only. Used by {@link parseAnthropicDocsPricing} (all-Claude
 * scrape rows) and the models-seed generator (`db/genModelsSeedSql.ts`). OpenRouter rows use
 * {@link deriveIdentity} directly (their capabilities come from the feed).
 */
export function completeSeedRow(row: ModelInfo): ModelInfo {
  const out = deriveIdentity(row);
  if (isClaudeModel(row.modelPrefix)) {
    if (out.supportedParameters === undefined) out.supportedParameters = claudeSupportedParameters(row.modelPrefix);
    if (out.modalities === undefined) out.modalities = { input: [...CLAUDE_MODALITIES.input], output: [...CLAUDE_MODALITIES.output] };
  }
  return out;
}

/** Fetches a URL's body as text. Injected so the core is testable and engine stays fetch-agnostic. */
export type FetchText = (url: string) => Promise<string>;

/** A source of authoritative prices for some set of models. */
export interface PricingSource {
  readonly name: string;
  /** Fetch + parse to rows. Throws on fetch/parse failure (the orchestrator isolates it). */
  fetchRows(): Promise<ModelInfo[]>;
  /**
   * `false`/unset (Anthropic scrape): a SMALL curated set — any bad row fails the whole batch
   * (a mis-scrape must not poison billing data). `true` (OpenRouter API): a LARGE heterogeneous
   * set — drop individual bad rows and keep the good ones, rather than reject thousands for one.
   */
  readonly lenient?: boolean;
}

export const ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing";
export const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";

// --- Parsing ----------------------------------------------------------------

/** Pull the first numeric value out of a price cell like `"$6.25 / MTok"` → `6.25`. */
function parseMoney(cell: string): number | undefined {
  const m = cell.replace(/,/g, "").match(/-?\d+(?:\.\d+)?/);
  if (!m) return undefined;
  const n = Number(m[0]);
  return Number.isFinite(n) ? n : undefined;
}

/** Strip tags + decode the handful of entities a docs table uses. */
function stripHtml(s: string): string {
  return s
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

/** Normalize a page (markdown table OR HTML `<table>`) into rows of trimmed cell strings. */
function extractRows(text: string): string[][] {
  if (/<tr\b/i.test(text)) {
    const rows: string[][] = [];
    for (const tr of text.match(/<tr\b[\s\S]*?<\/tr>/gi) ?? []) {
      const cells = [...tr.matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)].map((c) => stripHtml(c[1]!));
      if (cells.length > 0) rows.push(cells);
    }
    return rows;
  }
  // Markdown table: `| a | b | c |` lines; drop the `|---|` separator rows.
  return text
    .split(/\r?\n/)
    .filter((line) => line.includes("|") && !/^\s*\|?[\s:|-]*\|[\s:|-]*$/.test(line))
    .map((line) =>
      line
        .replace(/^\s*\|/, "")
        .replace(/\|\s*$/, "")
        .split("|")
        .map((c) => c.trim()),
    )
    .filter((cells) => cells.length > 1);
}

/** `"Claude Opus 4.8 ([deprecated]...)"` → `"claude-opus-4-8"` (a longest-prefix match key). */
function modelNameToPrefix(name: string): string | undefined {
  const base = stripHtml(name).split("(")[0]!.trim(); // drop "([deprecated]...)" notes
  if (!/^claude\b/i.test(base)) return undefined;
  return base.toLowerCase().replace(/[.\s]+/g, "-").replace(/-+$/g, "");
}

/**
 * Parse Anthropic's docs pricing table into `ModelInfo`s — base input/output PLUS the exact
 * cache rates (5-min write, 1-hour write, read), so cost is billing-exact without relying on
 * the default multipliers. Locates the model-pricing table by its header, then reads each
 * `Claude …` row by column. Returns `[]` if no recognizable table is found (the caller treats
 * an empty/short result as a failed scrape and keeps existing prices).
 *
 * The docs table carries NO capability/modality columns, and the native `claude-*` ids aren't on
 * OpenRouter, so nothing else fills them — {@link completeModelIdentity} stamps the correct
 * per-model `supportedParameters` (via `claudeSupportedParameters`, so opus-4-7/4-8 reject the
 * sampling knobs) + modalities + canonical id / serving provider, the SAME way the seed does.
 */
export function parseAnthropicDocsPricing(text: string): ModelInfo[] {
  const rows = extractRows(text);

  // Find the header row: cells naming the base-input and output columns.
  const headerIdx = rows.findIndex(
    (cells) =>
      cells.some((c) => /base\s*input/i.test(c)) && cells.some((c) => /output/i.test(c)),
  );
  if (headerIdx === -1) return [];
  const header = rows[headerIdx]!.map((c) => c.toLowerCase());
  const col = (re: RegExp): number => header.findIndex((c) => re.test(c));
  const idx = {
    input: col(/base\s*input/),
    write5m: col(/5m|5\s*min|5-min/),
    write1h: col(/1h|1\s*hour|1-hour/),
    read: col(/cache\s*hit|cache\s*read|hits/),
    output: col(/output/),
  };
  if (idx.input === -1 || idx.output === -1) return [];

  const out: ModelInfo[] = [];
  for (const cells of rows.slice(headerIdx + 1)) {
    const prefix = modelNameToPrefix(cells[0] ?? "");
    if (!prefix) continue; // not a model row (notes, blank, next section)
    const input = parseMoney(cells[idx.input] ?? "");
    const output = parseMoney(cells[idx.output] ?? "");
    if (input === undefined || output === undefined) continue;
    const row: ModelInfo = {
      modelPrefix: prefix,
      inputPerMillion: input,
      outputPerMillion: output,
      family: "anthropic",
      provider: "Anthropic",
      label: prefix,
      source: "anthropic-docs",
    };
    const read = idx.read !== -1 ? parseMoney(cells[idx.read] ?? "") : undefined;
    const w5 = idx.write5m !== -1 ? parseMoney(cells[idx.write5m] ?? "") : undefined;
    const w1 = idx.write1h !== -1 ? parseMoney(cells[idx.write1h] ?? "") : undefined;
    if (read !== undefined) row.cacheReadPerMillion = read;
    if (w5 !== undefined) row.cacheWritePerMillion = w5;
    if (w1 !== undefined) row.cacheWrite1hPerMillion = w1;
    // Stamp the derived identity + Claude capabilities/modalities (the docs table has none) — the SAME
    // completion the seed generator applies, so the refresh and the seed produce byte-identical rows.
    out.push(completeSeedRow(row));
  }
  return out;
}

/** The Anthropic-docs pricing source (the user's chosen authority). Strict batch validation. */
export function makeAnthropicDocsSource(fetchText: FetchText, url: string = ANTHROPIC_PRICING_URL): PricingSource {
  return {
    name: "anthropic-docs",
    async fetchRows() {
      return parseAnthropicDocsPricing(await fetchText(url));
    },
  };
}

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
 * full `provider/model` id (matched ahead of the bare id, §pricing), so an OpenRouter-routed call
 * gets OpenRouter's marked-up rate. This source is `lenient` — odd individual rows are dropped by
 * the orchestrator, not fatal. (Note: we PREFER OpenRouter's per-call reported cost; this table is
 * the fallback when a response omits it.)
 *
 * Beyond price, the feed carries the model's CAPABILITIES — `supported_parameters` (which the
 * executor filters outgoing params against, §5.1), `context_length`, `top_provider.max_completion_tokens`,
 * and `architecture.{input,output}_modalities` — so the catalog drives routing/structured-output
 * decisions from data instead of hardcoded family heuristics.
 */
export function parseOpenRouterModels(json: string): ModelInfo[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return [];
  }
  const data = (parsed as { data?: unknown }).data;
  if (!Array.isArray(data)) return [];
  const out: ModelInfo[] = [];
  for (const m of data) {
    const model = m as {
      id?: unknown;
      pricing?: Record<string, unknown>;
      created?: unknown;
      context_length?: unknown;
      supported_parameters?: unknown;
      top_provider?: { max_completion_tokens?: unknown };
      architecture?: { input_modalities?: unknown; output_modalities?: unknown };
    };
    const id = model.id;
    const pricing = model.pricing;
    if (typeof id !== "string" || !pricing) continue;
    const input = perMillion(pricing.prompt);
    const output = perMillion(pricing.completion);
    if (input === undefined || output === undefined) continue;
    const row: ModelInfo = {
      modelPrefix: id,
      inputPerMillion: input,
      outputPerMillion: output,
      // OpenRouter ids are always `vendor/model`, so they route through OpenRouter (never the native
      // Anthropic provider — a claude-via-OR id is non-`claude-` prefixed → openrouter family).
      family: "openrouter",
      provider: displayProviderFor(id),
      label: id.includes("/") ? id.slice(id.indexOf("/") + 1) : id,
      source: "openrouter-models",
    };
    const read = perMillion(pricing.input_cache_read);
    const write = perMillion(pricing.input_cache_write);
    if (read !== undefined) row.cacheReadPerMillion = read;
    if (write !== undefined) row.cacheWritePerMillion = write;
    // OpenRouter reports `created` (UNIX seconds) — the model's release/listing time. Capture it so
    // the admin UI can sort newest-first; it never touches cost.
    const created = model.created;
    if (typeof created === "number" && Number.isFinite(created) && created > 0) row.releasedAt = created;
    // Capabilities + limits (all optional; absent ⇒ heuristic fallbacks apply downstream).
    const supported = model.supported_parameters;
    if (Array.isArray(supported) && supported.every((p) => typeof p === "string") && supported.length > 0) {
      row.supportedParameters = supported as string[];
    }
    const ctx = model.context_length;
    if (typeof ctx === "number" && Number.isFinite(ctx) && ctx > 0) row.contextLength = ctx;
    const maxOut = model.top_provider?.max_completion_tokens;
    if (typeof maxOut === "number" && Number.isFinite(maxOut) && maxOut > 0) row.maxOutputTokens = maxOut;
    const inMods = model.architecture?.input_modalities;
    const outMods = model.architecture?.output_modalities;
    const modalities: { input?: string[]; output?: string[] } = {};
    if (Array.isArray(inMods) && inMods.every((x) => typeof x === "string")) modalities.input = inMods as string[];
    if (Array.isArray(outMods) && outMods.every((x) => typeof x === "string")) modalities.output = outMods as string[];
    if (modalities.input || modalities.output) row.modalities = modalities;
    // Add ONLY the derived identity (canonical id + serving provider = "openrouter" for these `vendor/model`
    // ids) — the feed is authoritative for capabilities, so we never stamp native-Claude caps here.
    out.push(deriveIdentity(row));
  }
  return out;
}

/** The OpenRouter models-API pricing source. Lenient (thousands of heterogeneous models). */
export function makeOpenRouterSource(fetchText: FetchText, url: string = OPENROUTER_MODELS_URL): PricingSource {
  return {
    name: "openrouter-models",
    lenient: true,
    async fetchRows() {
      return parseOpenRouterModels(await fetchText(url));
    },
  };
}

// --- Validation -------------------------------------------------------------

export interface PricingValidation {
  ok: boolean;
  problems: string[];
}

/** Internal-consistency problems with a SINGLE row's rates (empty array = sane). */
function rowProblems(r: ModelInfo): string[] {
  const p: string[] = [];
  if (!r.modelPrefix) p.push("empty modelPrefix");
  if (!(r.inputPerMillion > 0)) p.push(`non-positive input ${r.inputPerMillion}`);
  // An output rate of EXACTLY 0 is legitimate, not a bad scrape: embedding / classifier /
  // input-only models bill no completion tokens (`text-embedding-3-small` is $0 output). So
  // only a NEGATIVE output is invalid, and the "output < input is suspect" heuristic — which
  // assumes a generation model whose completions cost more than its prompt — applies ONLY to
  // models that actually charge for output (output > 0). Without this carve-out every
  // zero-output model in a lenient feed (OpenRouter embeddings) would be wrongly dropped.
  if (r.outputPerMillion < 0) p.push(`negative output ${r.outputPerMillion}`);
  else if (r.outputPerMillion > 0 && r.outputPerMillion < r.inputPerMillion) p.push("output < input (suspect)");
  if (r.inputPerMillion > 1000 || r.outputPerMillion > 1000) p.push("implausibly high rate");
  if (r.cacheReadPerMillion !== undefined && r.cacheReadPerMillion >= r.inputPerMillion)
    p.push("cache read ≥ input (should be a discount)");
  if (
    r.cacheWritePerMillion !== undefined &&
    r.cacheWrite1hPerMillion !== undefined &&
    r.cacheWrite1hPerMillion < r.cacheWritePerMillion
  )
    p.push("1h write < 5m write");
  return p;
}

/**
 * Hard sanity gate (STRICT sources) before any write — a mis-scrape must NOT poison billing data.
 * Requires a plausible row count and internally-consistent, non-duplicate rates. Any violation
 * fails the WHOLE batch (keep the old prices rather than apply a partially-garbled one).
 */
export function validatePricingRows(rows: ModelInfo[], minRows = 3): PricingValidation {
  const problems: string[] = [];
  if (rows.length < minRows) problems.push(`too few rows: ${rows.length} < ${minRows}`);
  const seen = new Set<string>();
  for (const r of rows) {
    if (seen.has(r.modelPrefix)) problems.push(`[${r.modelPrefix}] duplicate modelPrefix`);
    seen.add(r.modelPrefix);
    for (const p of rowProblems(r)) problems.push(`[${r.modelPrefix}] ${p}`);
  }
  return { ok: problems.length === 0, problems };
}

/**
 * Per-row filter (LENIENT sources): keep the sane, non-duplicate rows and report what was
 * dropped. Used for large heterogeneous feeds (OpenRouter) where one odd row must not sink the
 * whole refresh.
 */
export function sanitizePricingRows(rows: ModelInfo[]): {
  rows: ModelInfo[];
  dropped: { modelPrefix: string; problems: string[] }[];
} {
  const seen = new Set<string>();
  const kept: ModelInfo[] = [];
  const dropped: { modelPrefix: string; problems: string[] }[] = [];
  for (const r of rows) {
    const problems = rowProblems(r);
    if (seen.has(r.modelPrefix)) problems.push("duplicate modelPrefix");
    if (problems.length > 0) {
      dropped.push({ modelPrefix: r.modelPrefix, problems });
      continue;
    }
    seen.add(r.modelPrefix);
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
  /** True when the source was skipped (fetch/parse error or validation failure) — old prices stand. */
  skipped: boolean;
  error?: string;
  problems?: string[];
}

export interface RefreshReport {
  bySource: SourceOutcome[];
  added: string[];
  updated: string[];
  /** The full row set after refresh — what the caller persists to the `models` table. */
  rows: ModelInfo[];
}

/** Value-equality on the rate-bearing fields (ignores nothing pricing-relevant). */
function rowsEqual(a: ModelInfo | undefined, b: ModelInfo): boolean {
  return a !== undefined && JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Fetch every source, validate, and apply only the CHANGED rows to `table` (default: the
 * process-wide `defaultModelCatalog`). A source that throws or fails validation is isolated and
 * skipped — its failure never drops or corrupts existing prices. Returns a report (and the full
 * row set) so the caller can persist to the `models` table and log/alert on skips.
 */
export async function refreshModelCatalog(opts: {
  sources: PricingSource[];
  table?: ModelCatalog;
  /** Min rows a source must yield to be trusted (validation). */
  minRows?: number;
}): Promise<RefreshReport> {
  const table = opts.table ?? defaultModelCatalog;
  const bySource: SourceOutcome[] = [];
  const added: string[] = [];
  const updated: string[] = [];

  const minRows = opts.minRows ?? 3;
  for (const source of opts.sources) {
    let fetched: ModelInfo[];
    try {
      fetched = await source.fetchRows();
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      log.warn("pricing source fetch failed; keeping existing prices", { source: source.name, error });
      bySource.push({ name: source.name, fetched: 0, applied: 0, skipped: true, error });
      continue;
    }

    // Decide the trusted row set: STRICT = all-or-nothing; LENIENT = drop bad rows individually.
    let toApply: ModelInfo[];
    if (source.lenient) {
      const { rows: clean, dropped } = sanitizePricingRows(fetched);
      if (dropped.length > 0) log.debug("pricing source dropped rows", { source: source.name, dropped: dropped.length });
      if (clean.length < minRows) {
        log.warn("pricing source yielded too few valid rows; keeping existing prices", {
          source: source.name,
          fetched: fetched.length,
          valid: clean.length,
        });
        bySource.push({ name: source.name, fetched: fetched.length, applied: 0, skipped: true, problems: [`only ${clean.length} valid rows`] });
        continue;
      }
      toApply = clean;
    } else {
      const check = validatePricingRows(fetched, minRows);
      if (!check.ok) {
        log.warn("pricing source failed validation; keeping existing prices", {
          source: source.name,
          fetched: fetched.length,
          problems: check.problems,
        });
        bySource.push({ name: source.name, fetched: fetched.length, applied: 0, skipped: true, problems: check.problems });
        continue;
      }
      toApply = fetched;
    }

    // Apply only the CHANGED rows (snapshot the current table once for an O(n) diff).
    const current = new Map(table.list().map((r) => [r.modelPrefix, r]));
    let applied = 0;
    for (const row of toApply) {
      if (rowsEqual(current.get(row.modelPrefix), row)) continue;
      (current.has(row.modelPrefix) ? updated : added).push(row.modelPrefix);
      table.upsert(row);
      applied++;
    }
    log.info("pricing source applied", { source: source.name, fetched: fetched.length, applied });
    bySource.push({ name: source.name, fetched: fetched.length, applied, skipped: false });
  }

  return { bySource, added, updated, rows: table.list() };
}
