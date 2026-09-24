/**
 * Usage readings — how full a conversation's context is, and how much of an account's allowance is
 * spent. Two readings because they have different owners and different lifetimes: a context reading
 * belongs to ONE conversation and changes with every response; a limit reading belongs to an ACCOUNT
 * and changes whenever anything anywhere spends from it.
 *
 * They live at the bottom of the graph because both halves of the system carry them: `exec` streams
 * them as events, `llm` stores a context reading on the turn it follows, and neither may import the
 * other. Everything provider-specific — which event carries which number — stays in the adapters;
 * these shapes are what every adapter normalizes to.
 *
 * The unit of a limit is PERCENT of a window, never tokens: that is what the subscription providers
 * report, and it is the only unit comparable across model families.
 */

/** One slice of what a context holds — a category, and optionally what it is made of. */
export interface ContextPart {
  name: string;
  tokens: number;
  /** The slice broken down further (tools one by one, results by tool), when the provider says. */
  parts?: ContextPart[];
  /** A short note about the slice ("deferred — not in context yet", "14 calls"). */
  note?: string;
}

/** How full one conversation is, read after a response. */
export interface ContextReading {
  /** Tokens the conversation holds now: the last response's input (cached and uncached) plus its output. */
  used: number;
  /** The model's context window; `null` when the provider does not say. */
  window: number | null;
  /** The model that answered, as dispatched. */
  model: string;
  /** What the tokens are, when the provider offers it. */
  breakdown?: ContextPart[];
  /** Where the agent compacts by itself, in tokens, when it does. */
  autoCompactAt?: number;
  /** When it was read (ISO-8601). */
  at: string;
}

/** The provider's own verdict on an account, or one derived from the windows. */
export type LimitStatus = "ok" | "warning" | "exhausted";

/** One allowance window: a five-hour session, a week, a week counting one model only. */
export interface LimitWindow {
  /** The provider's own word for the window (`five_hour`, `seven_day`, `seven_day_opus`, `primary`). */
  id: string;
  /** A human label ("5-hour", "Weekly"). */
  label: string;
  /** The window's length in minutes, when known. */
  minutes: number | null;
  /** How much of it is used, 0–100; `null` when the provider gave the window without a figure. */
  usedPercent: number | null;
  /** When it resets (ISO-8601), when known. */
  resetsAt: string | null;
  /** Present when the window counts one model family only ("opus", "sonnet"). */
  model?: string;
  /** The provider's verdict on THIS window, when it gave one. */
  status?: LimitStatus;
}

/** How much of an account's allowance is spent, read on one route. */
export interface LimitReading {
  /** The route it was read on (`claude-cli`, `codex-cli`, `anthropic`, …). */
  route: string;
  /** The provider's plan word (`max`, `plus`, …); `null` for an API key or when not said. */
  plan: string | null;
  /** Every window the provider reported. May be empty. */
  windows: LimitWindow[];
  /** The provider's own verdict where it gives one, else derived: exhausted at 100. */
  status: LimitStatus;
  /** How it was learned, so a reader can weigh it. */
  source: "stream" | "query" | "headers" | "file";
  /** When it was read (ISO-8601). */
  at: string;
  /**
   * `true` when this reading names EVERY window the account has (a full query); absent or `false`
   * when it is a sparse update about the windows it lists — a stream event names one window and says
   * nothing of the others, so it is merged into what was known rather than replacing it.
   */
  complete?: boolean;
  /** The account is drawing on extra, paid usage — spent does not mean waiting. */
  overage?: boolean;
}

const RANK: Record<LimitStatus, number> = { ok: 0, warning: 1, exhausted: 2 };

/** A window's verdict: the provider's when given, else derived from the figure (100 ⇒ exhausted). */
export function windowStatus(window: LimitWindow): LimitStatus {
  if (window.status !== undefined) return window.status;
  return window.usedPercent !== null && window.usedPercent >= 100 ? "exhausted" : "ok";
}

/** A window whose reset has passed describes a window that no longer exists. */
export function windowIsCurrent(window: LimitWindow, nowMs: number = Date.now()): boolean {
  if (window.resetsAt === null) return true;
  const t = Date.parse(window.resetsAt);
  return Number.isNaN(t) || t > nowMs;
}

/**
 * The account's verdict as of `nowMs`: the worst CURRENT window's. A window whose reset time has
 * passed no longer counts — an account spent until 14:05 is not spent at 14:06, whatever the last
 * reading said.
 */
export function limitStatusAt(reading: LimitReading, nowMs: number = Date.now()): LimitStatus {
  let worst: LimitStatus = "ok";
  for (const w of reading.windows) {
    if (!windowIsCurrent(w, nowMs)) continue;
    const s = windowStatus(w);
    if (RANK[s] > RANK[worst]) worst = s;
  }
  return worst;
}

/** The windows that apply to a call on `model`: the account-wide ones plus any counting that model's family. */
export function windowsForModel(reading: LimitReading, model?: string): LimitWindow[] {
  const m = model?.toLowerCase();
  return reading.windows.filter((w) => w.model === undefined || (m !== undefined && m.includes(w.model.toLowerCase())));
}

/**
 * The window that decides whether the next call on `model` goes: the current one with the most used
 * (an exhausted one first). `undefined` when nothing current applies.
 */
export function tightestWindow(reading: LimitReading, model?: string, nowMs: number = Date.now()): LimitWindow | undefined {
  let best: LimitWindow | undefined;
  let bestKey = -1;
  for (const w of windowsForModel(reading, model)) {
    if (!windowIsCurrent(w, nowMs)) continue;
    const key = RANK[windowStatus(w)] * 1000 + (w.usedPercent ?? -1);
    if (key > bestKey) {
      best = w;
      bestKey = key;
    }
  }
  return best;
}

/** How much of the tightest applicable window is LEFT, 0–100; `null` when no figure is known. */
export function remainingPercent(reading: LimitReading, model?: string, nowMs: number = Date.now()): number | null {
  const figures = windowsForModel(reading, model)
    .filter((w) => windowIsCurrent(w, nowMs))
    .map((w) => (windowStatus(w) === "exhausted" ? 100 : w.usedPercent))
    .filter((p): p is number => p !== null);
  return figures.length === 0 ? null : Math.max(0, 100 - Math.max(...figures));
}

/**
 * Fold a new reading into what was known.
 *
 * A COMPLETE reading replaces the windows outright — except that a figure the new one leaves `null`
 * keeps the last known figure of the same window, since "not said" is not "zero". A SPARSE reading
 * (a stream event about one window) updates only the windows it names. The plan and route follow
 * whichever said them last; `at` is the newest.
 */
export function mergeLimitReadings(prev: LimitReading | null | undefined, next: LimitReading): LimitReading {
  if (prev === null || prev === undefined) return { ...next, status: limitStatusOfWindows(next.windows, next.status) };
  const old = new Map(prev.windows.map((w) => [w.id, w]));
  const merged: LimitWindow[] = [];
  const seen = new Set<string>();
  for (const w of next.windows) {
    const before = old.get(w.id);
    seen.add(w.id);
    merged.push({
      ...before,
      ...w,
      usedPercent: w.usedPercent ?? before?.usedPercent ?? null,
      resetsAt: w.resetsAt ?? before?.resetsAt ?? null,
      minutes: w.minutes ?? before?.minutes ?? null,
    });
  }
  if (next.complete !== true) for (const w of prev.windows) if (!seen.has(w.id)) merged.push(w);
  const plan = next.plan ?? prev.plan;
  const overage = next.overage ?? prev.overage;
  const out: LimitReading = {
    route: next.route,
    plan,
    windows: merged,
    status: "ok",
    source: next.source,
    at: next.at > prev.at ? next.at : prev.at,
    ...(next.complete === true || prev.complete === true ? { complete: true } : {}),
    ...(overage !== undefined ? { overage } : {}),
  };
  out.status = limitStatusOfWindows(merged, next.complete === true ? next.status : undefined);
  return out;
}

function limitStatusOfWindows(windows: readonly LimitWindow[], stated?: LimitStatus): LimitStatus {
  let worst: LimitStatus = stated ?? "ok";
  for (const w of windows) {
    const s = windowStatus(w);
    if (RANK[s] > RANK[worst]) worst = s;
  }
  return worst;
}

/** Human labels for the windows every provider we know reports. */
export function limitWindowLabel(id: string, minutes?: number | null): string {
  switch (id) {
    case "five_hour":
      return "5-hour";
    case "seven_day":
      return "Weekly";
    case "seven_day_opus":
      return "Weekly · Opus";
    case "seven_day_sonnet":
      return "Weekly · Sonnet";
    case "seven_day_oauth_apps":
      return "Weekly · apps";
    case "overage":
      return "Extra usage";
    default:
      if (minutes === 300) return "5-hour";
      if (minutes === 10080) return "Weekly";
      if (typeof minutes === "number" && minutes > 0) return minutes % 1440 === 0 ? `${minutes / 1440}-day` : minutes % 60 === 0 ? `${minutes / 60}-hour` : `${minutes}-minute`;
      return id;
  }
}

/** The weekly window, all models — what a sign-in card shows. */
export function weeklyWindow(reading: LimitReading): LimitWindow | undefined {
  return reading.windows.find((w) => w.model === undefined && (w.id === "seven_day" || w.minutes === 10080));
}

/** The context's fill, 0–100, when the window is known. */
export function contextPercent(reading: ContextReading): number | null {
  return reading.window !== null && reading.window > 0 ? Math.min(100, (reading.used / reading.window) * 100) : null;
}
