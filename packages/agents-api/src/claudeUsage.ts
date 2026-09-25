/**
 * Claude's usage vocabulary, normalized to the neutral readings (`ContextReading`, `LimitReading`).
 *
 * Every number here has a provider-specific home and this module is the one place that knows it:
 *
 *  - a STREAM `rate_limit_event` — whenever the limits change. MEASURED in ten stored records from
 *    claude 2.1.x: `status`, `resetsAt` (unix seconds), `rateLimitType`, `isUsingOverage`, and never
 *    `utilization`. So the stream gives the state and the reset, not the percent — a SPARSE reading.
 *  - the `get_usage` control request (the SDK's `usage_EXPERIMENTAL…`), answered without a turn by a
 *    claude process started only to ask — every window's `utilization` (0–100) and `resets_at`, plus
 *    the plan. A COMPLETE reading. MEASURED: claude 2.1.223 answers it; 2.1.142 refuses the subtype.
 *  - each assistant message's `usage` — what the conversation holds after that response.
 *  - the result's `modelUsage[model].contextWindow` — the window.
 *  - the `get_context_usage` control request — the per-category breakdown and the auto-compact point.
 *
 * Treat every source as able to vanish: each reader returns `undefined` for a shape it does not
 * recognise, and a missing reading is an ordinary value.
 */
import type { ContextPart, ContextReading, LimitReading, LimitStatus, LimitWindow } from "@declarative-ai/json";
import { limitWindowLabel } from "@declarative-ai/json";

type Bag = Record<string, unknown>;
const bag = (v: unknown): Bag | undefined => (v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Bag) : undefined);
const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" && v.length > 0 ? v : undefined);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** Unix seconds (or milliseconds, or ISO) → ISO-8601. */
function isoOf(v: unknown): string | null {
  const n = num(v);
  if (n !== undefined) return new Date(n < 1e12 ? n * 1000 : n).toISOString();
  const s = str(v);
  if (s === undefined) return null;
  const t = Date.parse(s);
  return Number.isNaN(t) ? null : new Date(t).toISOString();
}

const WINDOW_MINUTES: Record<string, number> = { five_hour: 300, seven_day: 10080, seven_day_opus: 10080, seven_day_sonnet: 10080, seven_day_oauth_apps: 10080 };
const WINDOW_MODEL: Record<string, string> = { seven_day_opus: "opus", seven_day_sonnet: "sonnet" };

function windowOf(id: string, usedPercent: number | null, resetsAt: string | null, status?: LimitStatus): LimitWindow {
  const minutes = WINDOW_MINUTES[id] ?? null;
  return {
    id,
    label: limitWindowLabel(id, minutes),
    minutes,
    usedPercent: usedPercent === null ? null : Math.max(0, Math.min(100, usedPercent)),
    resetsAt,
    ...(WINDOW_MODEL[id] !== undefined ? { model: WINDOW_MODEL[id] } : {}),
    ...(status !== undefined ? { status } : {}),
  };
}

const STATUS: Record<string, LimitStatus> = { allowed: "ok", allowed_warning: "warning", rejected: "exhausted" };

/**
 * A stream `rate_limit_event` → a SPARSE limit reading about the one window it names.
 *
 * `utilization`, when a later claude sends it, is read as a FRACTION (the unified rate-limit headers
 * it mirrors are 0–1) — unverified, since no recorded event has carried it.
 */
export function limitReadingOfRateLimitEvent(event: unknown, route: string, at: string = new Date().toISOString()): LimitReading | undefined {
  const e = bag(event);
  if (e?.["type"] !== "rate_limit_event") return undefined;
  const info = bag(e["rate_limit_info"]);
  if (info === undefined) return undefined;
  const status = STATUS[str(info["status"]) ?? ""] ?? "ok";
  const id = str(info["rateLimitType"]) ?? "unknown";
  const u = num(info["utilization"]);
  const usedPercent = u === undefined ? null : u <= 1 ? u * 100 : u;
  const overage = info["isUsingOverage"] === true || info["overageInUse"] === true;
  return {
    route,
    plan: null,
    windows: id === "unknown" && usedPercent === null && isoOf(info["resetsAt"]) === null ? [] : [windowOf(id, status === "exhausted" && usedPercent === null ? 100 : usedPercent, isoOf(info["resetsAt"]), status)],
    status,
    source: "stream",
    at,
    ...(overage ? { overage: true } : {}),
  };
}

/**
 * Which `rate_limits` keys are plan windows. MEASURED 2026-09-24 (claude 2.1.223, a max account): the
 * object also holds codenamed entries in the SAME `{utilization, resets_at}` shape (`nimbus_quill` at
 * 0 with no reset), `extra_usage` (the credits switch), `spend` (money), `seven_day_breakdown` (the
 * week split by surface), `limits` and `model_scoped` (arrays restating the windows) and a flag. So a
 * window is named for its length — `five_hour`, `seven_day`, or a `seven_day_*` slice of the week —
 * AND carries a figure.
 */
const PLAN_WINDOW = /^(five_hour|seven_day(_[a-z0-9_]+)?)$/;

/** `extra_usage` → the reading's `overage`: on, and not stopped by its spend cap. `undefined` when absent. */
function overageOf(raw: unknown): boolean | undefined {
  const x = bag(raw);
  if (x === undefined || typeof x["is_enabled"] !== "boolean") return undefined;
  return x["is_enabled"] === true && x["spend_limit_reached"] !== true;
}

/** The `get_usage` answer → a COMPLETE limit reading; `undefined` when it carries no windows (signed out, API key). */
export function limitReadingOfClaudeUsage(response: unknown, route: string, at: string = new Date().toISOString()): LimitReading | undefined {
  const r = bag(response);
  const limits = bag(r?.["rate_limits"]);
  if (r === undefined || limits === undefined) return undefined;
  const windows: LimitWindow[] = [];
  for (const [id, raw] of Object.entries(limits)) {
    if (!PLAN_WINDOW.test(id)) continue;
    const w = bag(raw);
    const u = num(w?.["utilization"]);
    if (w === undefined || u === undefined) continue;
    windows.push(windowOf(id, u, isoOf(w["resets_at"])));
  }
  if (windows.length === 0) return undefined;
  const status: LimitStatus = windows.some((w) => (w.usedPercent ?? 0) >= 100) ? "exhausted" : "ok";
  const overage = overageOf(limits["extra_usage"]);
  return { route, plan: str(r["subscription_type"]) ?? null, windows, status, source: "query", at, complete: true, ...(overage !== undefined ? { overage } : {}) };
}

/** An Anthropic `usage` object → the tokens the conversation holds after that response. */
export function contextTokensOfUsage(usage: unknown): number | undefined {
  const u = bag(usage);
  if (u === undefined) return undefined;
  const parts = [u["input_tokens"], u["cache_creation_input_tokens"], u["cache_read_input_tokens"], u["output_tokens"]].map(num);
  if (parts.every((p) => p === undefined)) return undefined;
  return parts.reduce<number>((s, p) => s + (p ?? 0), 0);
}

/** The window claude reported for `model` on a result's `modelUsage`, when it did. */
export function contextWindowOf(modelUsage: unknown, model: string | undefined): number | undefined {
  const byModel = bag(modelUsage);
  if (byModel === undefined) return undefined;
  const exact = model !== undefined ? bag(byModel[model]) : undefined;
  const pick = exact ?? Object.values(byModel).map(bag).find((m) => num(m?.["contextWindow"]) !== undefined);
  return num(pick?.["contextWindow"]);
}

/** What `get_context_usage` adds to a reading: the window, the breakdown, the auto-compact point. */
export type ContextDetail = Partial<Pick<ContextReading, "window" | "breakdown" | "autoCompactAt" | "model" | "used">>;

const SKIP = new Set(["free space", "autocompact buffer"]);

/**
 * The `get_context_usage` answer → breakdown and window.
 *
 * The top level is the provider's own categories ("System prompt", "System tools", "MCP tools",
 * "Memory files", "Skills", "Messages", …), minus the free space and the reserved buffer, which are
 * not things the conversation holds. Where the answer itemizes a category, the items become its
 * `parts`: tools one by one and MCP tools by server, memory files by path, skills by name, and the
 * messages split into tool results by tool, tool calls, your messages, answers and attachments.
 */
export function contextDetailOfClaude(response: unknown): ContextDetail | undefined {
  const r = bag(response);
  if (r === undefined) return undefined;
  const out: ContextDetail = {};
  const max = num(r["maxTokens"]);
  if (max !== undefined) out.window = max;
  const auto = num(r["autoCompactThreshold"]);
  if (auto !== undefined && r["isAutoCompactEnabled"] !== false) out.autoCompactAt = auto;
  const model = str(r["model"]);
  if (model !== undefined) out.model = model;
  const total = num(r["totalTokens"]);
  if (total !== undefined) out.used = total;

  const tools: ContextPart[] = list(r["systemTools"]).map(bag).filter((t): t is Bag => t !== undefined).map((t) => ({ name: str(t["name"]) ?? "tool", tokens: num(t["tokens"]) ?? 0 }));
  const servers = new Map<string, ContextPart[]>();
  let deferred = 0;
  for (const t of list(r["mcpTools"]).map(bag)) {
    if (t === undefined) continue;
    if (t["isLoaded"] === false) {
      deferred += 1;
      continue;
    }
    const server = str(t["serverName"]) ?? "mcp";
    const name = (str(t["name"]) ?? "tool").replace(new RegExp(`^mcp__${server}__`), "");
    servers.set(server, [...(servers.get(server) ?? []), { name, tokens: num(t["tokens"]) ?? 0 }]);
  }
  const mcp: ContextPart[] = [...servers].map(([server, parts]) => ({ name: server, tokens: parts.reduce((s, p) => s + p.tokens, 0), parts }));
  if (deferred > 0) mcp.push({ name: `${deferred} more`, tokens: 0, note: "deferred — not in context yet" });
  const memory: ContextPart[] = list(r["memoryFiles"]).map(bag).filter((f): f is Bag => f !== undefined).map((f) => ({ name: str(f["path"]) ?? "file", tokens: num(f["tokens"]) ?? 0 }));
  const skills = bag(r["skills"]);
  const skillParts: ContextPart[] = list(skills?.["skillFrontmatter"]).map(bag).filter((s): s is Bag => s !== undefined).map((s) => ({ name: str(s["name"]) ?? "skill", tokens: num(s["tokens"]) ?? 0 }));
  const mb = bag(r["messageBreakdown"]);
  const messages: ContextPart[] = [];
  if (mb !== undefined) {
    const byTool = list(mb["toolCallsByType"]).map(bag).filter((t): t is Bag => t !== undefined);
    const results = byTool.map((t) => ({ name: str(t["name"]) ?? "tool", tokens: num(t["resultTokens"]) ?? 0 })).filter((p) => p.tokens > 0).sort((a, b) => b.tokens - a.tokens);
    const resultTotal = num(mb["toolResultTokens"]) ?? results.reduce((s, p) => s + p.tokens, 0);
    if (resultTotal > 0) messages.push({ name: "Tool results", tokens: resultTotal, ...(results.length > 0 ? { parts: results } : {}) });
    const calls = num(mb["toolCallTokens"]);
    if (calls !== undefined && calls > 0) messages.push({ name: "Tool calls", tokens: calls });
    const user = num(mb["userMessageTokens"]);
    if (user !== undefined && user > 0) messages.push({ name: "Your messages", tokens: user });
    const said = num(mb["assistantMessageTokens"]);
    if (said !== undefined && said > 0) messages.push({ name: "Answers", tokens: said });
    const attach = num(mb["attachmentTokens"]);
    if (attach !== undefined && attach > 0) messages.push({ name: "Attachments", tokens: attach });
  }

  const categories: ContextPart[] = [];
  for (const c of list(r["categories"]).map(bag)) {
    if (c === undefined) continue;
    const name = str(c["name"]) ?? "other";
    const key = name.toLowerCase();
    if (SKIP.has(key) || c["isDeferred"] === true) continue;
    const tokens = num(c["tokens"]) ?? 0;
    if (tokens <= 0) continue;
    const parts =
      key.includes("mcp") ? mcp
      : key.includes("tool") ? tools
      : key.includes("memory") ? memory
      : key.includes("skill") ? skillParts
      : key.includes("message") ? messages
      : [];
    categories.push({ name, tokens, ...(parts.length > 0 ? { parts } : {}) });
  }
  if (categories.length > 0) out.breakdown = categories;
  return out;
}

/** A context reading from the response's own tokens, the window claude reported, and the detail `get_context_usage` added. */
export function contextOfClaude(
  used: number,
  model: string,
  window: number | null,
  detail: ContextDetail | undefined,
  at: string = new Date().toISOString(),
): ContextReading {
  return {
    used,
    window: window ?? detail?.window ?? null,
    model,
    ...(detail?.breakdown !== undefined ? { breakdown: detail.breakdown } : {}),
    ...(detail?.autoCompactAt !== undefined ? { autoCompactAt: detail.autoCompactAt } : {}),
    at,
  };
}

/** For a `provider_event` payload: is this the compaction boundary, and what did it say. */
export function compactionOf(event: unknown): { trigger: string; before?: number; after?: number; durationMs?: number } | undefined {
  const e = bag(event);
  if (e?.["type"] !== "system" || e["subtype"] !== "compact_boundary") return undefined;
  const m = bag(e["compact_metadata"]);
  const before = num(m?.["pre_tokens"]);
  const after = num(m?.["post_tokens"]);
  const durationMs = num(m?.["duration_ms"]);
  return {
    trigger: str(m?.["trigger"]) ?? "auto",
    ...(before !== undefined ? { before } : {}),
    ...(after !== undefined ? { after } : {}),
    ...(durationMs !== undefined ? { durationMs } : {}),
  };
}

