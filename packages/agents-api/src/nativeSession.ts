/**
 * The agent's OWN session file, read back — the concrete {@link AgentSessionReader} for `claude`.
 *
 * Both `claude` transports keep their real log on disk, at
 * `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl` — and that file holds lines that NEVER ride
 * the stream-json wire: `attachment` lines (the agent's own context injections — deferred tools,
 * skill listings, task reminders, structured-output echoes), the rich `toolUseResult` record beside
 * each tool-result turn, `queue-operation` / `last-prompt` / `ai-title` bookkeeping, and the
 * uuid/parentUuid threading and timestamps on every line. A record built from the stream alone is
 * therefore a smaller story than the run, and this module is how the missing half is read.
 *
 * ⚠️ The file is the AGENT'S, not ours: it lives under the operator's config dir, it is pruned on
 * the agent's own schedule, and a resumed session appends to the same file across runs. So a caller
 * that wants the lines durably copies them AT CLOSE ({@link nativeLinesOf} is the fold built for
 * that), rather than keeping the path and reading it later — the path is a pointer into somebody
 * else's garbage-collected heap.
 *
 * Sidechain conversations have their own files: a subagent writes
 * `<project-folder>/<main-session-id>/subagents/agent-<agentId>.jsonl`, with a sibling
 * `agent-<agentId>.meta.json` naming the spawning tool call (`toolUseId`) — which is the key
 * `LlmOutput.sidechains` already uses, so the two records join without guessing.
 * {@link readNativeSidechains} reads them all back by the main session's id.
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { JsonValue } from "@declarative-ai/exec";
import type { AgentSessionReader } from "./seam.js";

/**
 * A working directory as `claude` spells it in a project-folder name: every character that is not a
 * letter or digit becomes `-`. Lossy by construction — `C:\a\b` and `C:/a/b` collide — which is the
 * agent's own choice, faithfully reproduced rather than improved: an "improved" encoding would name
 * a folder the agent never writes.
 */
export function encodeSessionCwd(cwd: string): string {
  return cwd.replace(/[^A-Za-z0-9]/g, "-");
}

/** Where the agent keeps its per-project session files. `CLAUDE_CONFIG_DIR` is the agent's own
 *  override for its whole config tree, so it is honoured here for the same reason a resumed session
 *  honours it: the files are wherever the BINARY put them, not where the default says. */
export function claudeProjectsDir(configDir?: string): string {
  return join(configDir ?? process.env["CLAUDE_CONFIG_DIR"] ?? join(homedir(), ".claude"), "projects");
}

/** The session file for one conversation in one working directory. */
export function nativeSessionPath(providerSessionId: string, cwd: string, configDir?: string): string {
  return join(claudeProjectsDir(configDir), encodeSessionCwd(cwd), `${providerSessionId}.jsonl`);
}

/** A session id is joined into a filesystem path, so an id that could STEP OUT of the projects tree
 *  is refused before it becomes one. Real ids are UUIDs; this is deliberately looser (the format is
 *  the agent's to change) while still closing the traversal. */
function pathSafe(id: string): boolean {
  return id.length > 0 && !/[/\\]/.test(id) && !id.includes("..");
}

/** One file's lines, parsed. A line that is not JSON is returned as the raw string rather than
 *  dropped — an unreadable line is a fact about the file, and hiding it would make a corrupt record
 *  indistinguishable from a shorter one. */
function parseLines(text: string): unknown[] {
  return text
    .split(/\r?\n/)
    .filter((line) => line.length > 0)
    .map((line) => {
      try {
        return JSON.parse(line) as unknown;
      } catch {
        return line;
      }
    });
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * The project folder holding this conversation's files — spelled from the `cwd` when one is given,
 * found by searching every folder for `<id>.jsonl` when the seam forwarded only the id. The id
 * alone still names exactly one conversation; the cwd only says where to look.
 */
async function locateProjectFolder(providerSessionId: string, cwd: string | undefined, configDir?: string): Promise<string | undefined> {
  const root = claudeProjectsDir(configDir);
  if (cwd !== undefined) return join(root, encodeSessionCwd(cwd));
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    return undefined;
  }
  for (const entry of entries) {
    if (entry.isDirectory() && (await fileExists(join(root, entry.name, `${providerSessionId}.jsonl`)))) {
      return join(root, entry.name);
    }
  }
  return undefined;
}

/**
 * Build an {@link AgentSessionReader} over the agent's own files.
 *
 * A missing file reads as `[]`, which the seam documents as indistinguishable from an empty
 * conversation and requires the CALLER to treat honestly (§11: an empty resync is recorded as
 * empty, not passed off as a conversation).
 */
export function nativeSessionReader(options: { configDir?: string } = {}): AgentSessionReader {
  return async (providerSessionId, cwd) => {
    if (!pathSafe(providerSessionId)) return [];
    const folder = await locateProjectFolder(providerSessionId, cwd, options.configDir);
    if (folder === undefined) return [];
    try {
      return parseLines(await readFile(join(folder, `${providerSessionId}.jsonl`), "utf8"));
    } catch {
      return [];
    }
  };
}

/** One subagent's own session file, with what the sibling meta file says about who spawned it. */
export interface NativeSidechainFile {
  /** The agent's id, off the file name (`agent-<agentId>.jsonl`). */
  agentId: string;
  /** The SPAWNING tool call's id, from the meta file — the key `LlmOutput.sidechains` already uses,
   *  so a native sidechain joins the streamed one without guessing. Absent when the meta file is. */
  toolUseId?: string;
  /** The sibling `agent-<agentId>.meta.json`, verbatim — agentType, description, spawnDepth. */
  meta?: JsonValue;
  lines: unknown[];
}

/**
 * Read back every subagent conversation a session spawned.
 *
 * ✅ OBSERVED (claude 2.x): a subagent does not write into the parent's file — it writes its own,
 * under `<project-folder>/<main-session-id>/subagents/`, one `agent-<id>.jsonl` per spawn plus an
 * `agent-<id>.meta.json` naming the spawning `toolUseId`. Nothing on the stream ever carries the
 * attachments or `toolUseResult` records inside those files, so a capture that stops at the main
 * file keeps the main thread's missing half and silently drops the subagents'.
 */
export async function readNativeSidechains(providerSessionId: string, cwd?: string, configDir?: string): Promise<NativeSidechainFile[]> {
  if (!pathSafe(providerSessionId)) return [];
  const folder = await locateProjectFolder(providerSessionId, cwd, configDir);
  if (folder === undefined) return [];
  const dir = join(folder, providerSessionId, "subagents");
  let names: string[];
  try {
    names = (await readdir(dir)).filter((name) => name.startsWith("agent-") && name.endsWith(".jsonl"));
  } catch {
    return []; // no subagents directory — the ordinary case for a run that spawned none
  }
  const out: NativeSidechainFile[] = [];
  for (const name of names.sort()) {
    const agentId = name.slice("agent-".length, -".jsonl".length);
    let lines: unknown[];
    try {
      lines = parseLines(await readFile(join(dir, name), "utf8"));
    } catch {
      continue; // a file that vanished mid-read — the pruning this capture exists to outrun
    }
    let meta: JsonValue | undefined;
    let toolUseId: string | undefined;
    try {
      meta = JSON.parse(await readFile(join(dir, `agent-${agentId}.meta.json`), "utf8")) as JsonValue;
      const spawned = (meta as { toolUseId?: unknown } | null)?.toolUseId;
      if (typeof spawned === "string") toolUseId = spawned;
    } catch {
      // The meta file is the join key, not the record — its absence loses the join, never the lines.
    }
    out.push({ agentId, ...(toolUseId !== undefined ? { toolUseId } : {}), ...(meta !== undefined ? { meta } : {}), lines });
  }
  return out;
}

/** The default reader — the agent's default config dir, resolved per call so a test that sets
 *  `CLAUDE_CONFIG_DIR` is honoured. */
export const readNativeSession: AgentSessionReader = (providerSessionId, cwd) => nativeSessionReader()(providerSessionId, cwd);

/**
 * One native line worth keeping, pinned the way `LlmOutput.providerEvents` pins its events: `index`
 * is how many main-chain message lines preceded it in the file, so a reader can interleave the kept
 * lines with the record's own messages without re-reading the file that may no longer exist.
 */
export interface NativeLine {
  index: number;
  line: JsonValue;
}

/** A parsed line, when it is the record-shaped object every real line is. */
function recordOf(raw: unknown): Record<string, JsonValue> | undefined {
  return raw !== null && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, JsonValue>) : undefined;
}

/**
 * Fold a session file's lines down to what the STREAM never carried — the delta a record keeps.
 *
 * Two rules, by line kind:
 *
 *  - A main-chain `user` / `assistant` line's `message` body already rides the stream verbatim and
 *    is already in the record — so the body is stripped and the ENVELOPE kept: the uuid/parentUuid
 *    threading, the timestamp, and above all `toolUseResult`, the agent's own structured record of a
 *    tool execution (file metadata, split stdout/stderr) that the wire-level `tool_result` block
 *    flattens to text. Every such line advances `index`.
 *  - Every other line — `attachment`, `queue-operation`, `last-prompt`, `ai-title`, whatever a later
 *    release adds — is kept WHOLE, because none of it exists anywhere else. Unknown types are kept
 *    rather than filtered: the vocabulary is the agent's and it grows; a filter here would silently
 *    shrink the record every time it did.
 *
 * `sinceMs` is the RESUMED-session cut. The file spans the whole conversation while a record spans
 * one call, so a capture at close keeps only lines stamped at or after the call began — earlier
 * lines belong to the records of the calls that produced them. Lines the agent leaves unstamped
 * (`ai-title`, `last-prompt`) inherit the last stamp seen, and lines before ANY stamp are kept: at
 * the head of a fresh file that is the capture's own material, and erring toward keeping is the
 * point of capturing at all.
 *
 * `sidechain: true` folds a SUBAGENT's own file, where every line is marked `isSidechain: true` and
 * those lines ARE the file's chain — so they are what `index` counts. Under the default (a main
 * session file) such lines strip their body but advance nothing, since they are not the main
 * thread's messages.
 */
export function nativeLinesOf(lines: readonly unknown[], options: { sinceMs?: number; sidechain?: boolean } = {}): NativeLine[] {
  const since = options.sinceMs;
  const own = options.sidechain === true;
  const out: NativeLine[] = [];
  let index = 0;
  let stampMs: number | undefined;
  for (const raw of lines) {
    const line = recordOf(raw);
    const stamp = typeof line?.["timestamp"] === "string" ? Date.parse(line["timestamp"] as string) : Number.NaN;
    if (!Number.isNaN(stamp)) stampMs = stamp;
    const message = line !== undefined && (line["type"] === "user" || line["type"] === "assistant");
    const ownMessage = message && (line["isSidechain"] === true) === own;
    const kept = since === undefined || stampMs === undefined || stampMs >= since;
    if (kept) {
      if (message) {
        // The body rode the stream (a sidechain's into its own chain) — keep only what did not.
        const { message: _body, ...envelope } = line;
        out.push({ index, line: envelope });
      } else {
        out.push({ index, line: (raw ?? null) as JsonValue });
      }
    }
    if (ownMessage) index += 1;
  }
  return out;
}
