/**
 * Codex's usage, read from where codex keeps it: its own session files.
 *
 * `codex exec --json` carries neither reading — its `turn.completed.usage` is the thread's running
 * total, with no window and no rate limits. The session file carries both: one `token_count` event per
 * model response, always persisted, with `info.last_token_usage` against `info.model_context_window`
 * and `rate_limits.{primary,secondary}` as `{ used_percent, window_minutes, resets_at }` plus the plan.
 *
 * The file is `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<stamp>-<thread id>.jsonl`, found from the
 * thread id `thread.started` reports. The format is codex's own and internal, so every reader here
 * returns `undefined` for anything it does not recognise.
 */
import { readdir, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { limitWindowLabel, type ContextReading, type LimitReading, type LimitWindow } from "@declarative-ai/exec";

type Bag = Record<string, unknown>;
const bag = (v: unknown): Bag | undefined => (v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Bag) : undefined);
const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" && v.length > 0 ? v : undefined);

/** Where codex keeps its state: `$CODEX_HOME`, else `~/.codex`. */
export function codexHome(env: NodeJS.ProcessEnv = process.env): string {
  return env["CODEX_HOME"] ?? join(homedir(), ".codex");
}

/** The readings one `token_count` payload holds. */
export interface CodexReadings {
  context?: Omit<ContextReading, "model"> & { model?: string };
  limits?: LimitReading;
}

/**
 * One `token_count` event's payload → a context reading and a COMPLETE limit reading.
 *
 * `rate_limit_reached_type` names the window that ran out; that window is exhausted whatever its
 * figure says.
 */
export function readingsOfCodexTokenCount(payload: unknown, route: string, at: string): CodexReadings {
  const p = bag(payload);
  if (p?.["type"] !== "token_count") return {};
  const out: CodexReadings = {};
  const info = bag(p["info"]);
  const last = bag(info?.["last_token_usage"]);
  const used = num(last?.["total_tokens"]);
  if (used !== undefined) {
    const window = num(info?.["model_context_window"]);
    out.context = { used, window: window ?? null, at };
  }
  const limits = bag(p["rate_limits"]);
  if (limits !== undefined) {
    const reached = str(limits["rate_limit_reached_type"]);
    const windows: LimitWindow[] = [];
    for (const id of ["primary", "secondary"] as const) {
      const w = bag(limits[id]);
      if (w === undefined) continue;
      const minutes = num(w["window_minutes"]) ?? null;
      const usedPercent = num(w["used_percent"]) ?? null;
      const resets = num(w["resets_at"]);
      const spent = reached !== undefined && (reached === id || reached.includes(id) || (usedPercent ?? 0) >= 100);
      windows.push({
        id,
        label: limitWindowLabel(id, minutes),
        minutes,
        usedPercent: spent && usedPercent === null ? 100 : usedPercent,
        resetsAt: resets !== undefined ? new Date(resets < 1e12 ? resets * 1000 : resets).toISOString() : null,
        ...(spent ? { status: "exhausted" as const } : {}),
      });
    }
    if (windows.length > 0) {
      const exhausted = windows.some((w) => w.status === "exhausted" || (w.usedPercent ?? 0) >= 100);
      out.limits = { route, plan: str(limits["plan_type"]) ?? null, windows, status: exhausted ? "exhausted" : "ok", source: "file", at, complete: true };
    }
  }
  return out;
}

/** The day folders under `sessions/`, newest first (YYYY/MM/DD), at most `limit`. */
async function dayDirs(home: string, limit: number): Promise<string[]> {
  const root = join(home, "sessions");
  const out: string[] = [];
  const sorted = async (dir: string): Promise<string[]> => {
    try {
      return (await readdir(dir)).filter((n) => /^\d+$/.test(n)).sort().reverse();
    } catch {
      return [];
    }
  };
  for (const y of await sorted(root)) {
    for (const m of await sorted(join(root, y))) {
      for (const d of await sorted(join(root, y, m))) {
        out.push(join(root, y, m, d));
        if (out.length >= limit) return out;
      }
    }
  }
  return out;
}

/** The rollout file of `threadId`, looked for in the most recent day folders. */
export async function findCodexRollout(threadId: string, home: string = codexHome(), days = 14): Promise<string | undefined> {
  for (const dir of await dayDirs(home, days)) {
    let names: string[];
    try {
      names = await readdir(dir);
    } catch {
      continue;
    }
    const hit = names.find((n) => n.startsWith("rollout-") && n.endsWith(`${threadId}.jsonl`));
    if (hit !== undefined) return join(dir, hit);
  }
  return undefined;
}

/** The newest `token_count` payload in a rollout file, with its timestamp. */
export async function lastCodexTokenCount(file: string, need: "any" | "limits" = "any"): Promise<{ payload: Bag; at: string } | undefined> {
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch {
    return undefined;
  }
  const lines = text.split(/\r?\n/);
  for (let k = lines.length - 1; k >= 0; k--) {
    const line = lines[k]!;
    if (!line.includes('"token_count"')) continue;
    try {
      const o = JSON.parse(line) as Bag;
      const payload = bag(o["payload"]);
      if (payload?.["type"] !== "token_count") continue;
      if (need === "limits" && bag(payload["rate_limits"]) === undefined) continue;
      return { payload, at: str(o["timestamp"]) ?? new Date().toISOString() };
    } catch {
      continue;
    }
  }
  return undefined;
}

/** A finished codex thread's readings, from its own rollout file. */
export async function readCodexThreadReadings(threadId: string, route: string, home: string = codexHome()): Promise<CodexReadings> {
  const file = await findCodexRollout(threadId, home);
  if (file === undefined) return {};
  const last = await lastCodexTokenCount(file);
  return last === undefined ? {} : readingsOfCodexTokenCount(last.payload, route, last.at);
}

/**
 * The account's limits as the NEWEST codex session on this machine last saw them — the only way to
 * refresh codex's numbers without spending a turn. Helps only if some codex session ran since the
 * last reading; the reading's `at` is when that session saw it, so its age shows honestly.
 */
export async function readCodexLimits(options: { home?: string; route?: string; days?: number } = {}): Promise<LimitReading | undefined> {
  const home = options.home ?? codexHome();
  const files: { path: string; name: string }[] = [];
  for (const dir of await dayDirs(home, options.days ?? 7)) {
    try {
      for (const n of await readdir(dir)) if (n.startsWith("rollout-") && n.endsWith(".jsonl")) files.push({ path: join(dir, n), name: n });
    } catch {
      continue;
    }
    if (files.length >= 20) break;
  }
  files.sort((a, b) => (a.name < b.name ? 1 : -1));
  for (const f of files.slice(0, 20)) {
    const last = await lastCodexTokenCount(f.path, "limits");
    if (last === undefined) continue;
    const { limits } = readingsOfCodexTokenCount(last.payload, options.route ?? "codex-cli", last.at);
    if (limits !== undefined) return limits;
  }
  return undefined;
}
