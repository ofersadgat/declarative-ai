/**
 * A replayed conversation written as the agent's OWN session file, so `claude --resume` carries it
 * as real turns instead of one rendered prompt.
 *
 * Replay is the fallback for a call that has history but no remote to resume or copy (a provider
 * switch, a branch with nothing to fork). Rendered into the prompt, that history arrives as ONE user
 * message of `ROLE: <json>` text: every tool call and result flattened into prose, the agent's own
 * turns quoted back at it as something the person said, and the whole of it re-sent as fresh input
 * (JaiRA t-5uwe4jt33x, 2026-11-21: 160 KB in one message). The CLI has no message array on its wire,
 * but it does read its session files, and those are only JSON lines — so the transcript is written
 * where a resume reads it, and the call resumes that.
 *
 * Only a transcript already in the agent's own message shape can go in: a block of a type the
 * Messages API does not take would make the resumed call fail, where a rendered prompt still works.
 * Anything else answers `undefined`, and the caller renders as before.
 *
 * ⚠️ The format is the agent's (see `nativeSession.ts`). This writes the fields a resume reads and
 * nothing else; a line it does not need is not invented.
 */
import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { JsonValue } from "@declarative-ai/exec";
import { nativeSessionPath } from "./nativeSession.js";

/** Block types a resumed `claude` sends back to the Messages API as they are. */
const SEEDABLE_BLOCKS = new Set(["text", "thinking", "redacted_thinking", "tool_use", "tool_result", "image", "document"]);

type Block = Record<string, JsonValue> & { type: string };
type Turn = { role: "user" | "assistant"; content: string | Block[]; model?: string };

/** The result a dangling tool call is given, so the transcript the API reads is well-formed. */
const NO_RESULT = "The tool call was cut off before it returned a result.";

/**
 * The transcript as turns the API accepts, or `undefined` when one of them is not in its shape.
 *
 * Two repairs, both of what a cut leaves behind: a tool call with no result gets one (the API refuses
 * a `tool_use` with no `tool_result` after it), and a result whose call is not in the transcript is
 * dropped (the API refuses that too). A host's own entries (`calledBy`, e.g. an approval it asked on
 * the agent's behalf) were never in the agent's conversation and are left out, as the agent's own
 * file leaves them out.
 */
export function seedTurns(prior: readonly JsonValue[]): Turn[] | undefined {
  const turns: Turn[] = [];
  for (const message of prior) {
    if (message === null || typeof message !== "object" || Array.isArray(message)) return undefined;
    const bag = message as Record<string, JsonValue>;
    if (bag["calledBy"] !== undefined) continue;
    const role = bag["role"];
    if (role !== "user" && role !== "assistant") return undefined;
    const content = bag["content"];
    const model = typeof bag["model"] === "string" ? { model: bag["model"] } : {};
    if (typeof content === "string") {
      if (content.length > 0) turns.push({ role, content, ...model });
      continue;
    }
    if (!Array.isArray(content)) return undefined;
    const blocks: Block[] = [];
    for (const part of content) {
      if (part === null || typeof part !== "object" || Array.isArray(part)) return undefined;
      const type = (part as Record<string, JsonValue>)["type"];
      if (typeof type !== "string" || !SEEDABLE_BLOCKS.has(type)) return undefined;
      blocks.push(part as Block);
    }
    if (blocks.length > 0) turns.push({ role, content: blocks, ...model });
  }
  return pairToolCalls(turns);
}

function pairToolCalls(turns: Turn[]): Turn[] {
  const blocksOf = (t: Turn): Block[] => (typeof t.content === "string" ? [] : t.content);
  const called = new Set(turns.flatMap((t) => (t.role === "assistant" ? blocksOf(t) : [])).filter((b) => b.type === "tool_use").map((b) => String(b["id"])));
  const answered = new Set(turns.flatMap((t) => (t.role === "user" ? blocksOf(t) : [])).filter((b) => b.type === "tool_result").map((b) => String(b["tool_use_id"])));
  const out: Turn[] = [];
  let pending: string[] = [];
  const settle = (): void => {
    if (pending.length === 0) return;
    out.push({ role: "user", content: pending.map((id) => ({ type: "tool_result", tool_use_id: id, content: NO_RESULT, is_error: true })) });
    pending = [];
  };
  for (const turn of turns) {
    if (turn.role === "assistant") {
      for (const b of blocksOf(turn)) if (b.type === "tool_use" && !answered.has(String(b["id"]))) pending.push(String(b["id"]));
      out.push(turn);
      continue;
    }
    // A user turn ends the assistant's run: anything it called and nothing answered is answered here.
    settle();
    if (typeof turn.content === "string") {
      out.push(turn);
      continue;
    }
    const kept = turn.content.filter((b) => b.type !== "tool_result" || called.has(String(b["tool_use_id"])));
    if (kept.length > 0) out.push({ ...turn, content: kept });
  }
  settle();
  return out;
}

/**
 * The session file's lines for these turns: one linear chain, consecutive assistant turns sharing a
 * message id (the agent's own file splits one API message into a line per block, and reads them
 * back as one by that id).
 */
export function seedLines(turns: readonly Turn[], session: { sessionId: string; cwd: string; at?: Date; uuid?: () => string }): JsonValue[] {
  const uuid = session.uuid ?? randomUUID;
  const timestamp = (session.at ?? new Date()).toISOString();
  const lines: JsonValue[] = [];
  let parentUuid: string | null = null;
  let messageId: string | undefined;
  for (const turn of turns) {
    const id = uuid();
    if (turn.role === "assistant") {
      messageId ??= `msg_seed_${uuid().replace(/-/g, "")}`;
    } else {
      messageId = undefined;
    }
    const message: Record<string, JsonValue> =
      turn.role === "assistant"
        ? {
            ...(turn.model !== undefined ? { model: turn.model } : {}),
            id: messageId!,
            type: "message",
            role: "assistant",
            content: turn.content as JsonValue,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
          }
        : { role: "user", content: turn.content as JsonValue };
    lines.push({
      parentUuid,
      isSidechain: false,
      type: turn.role,
      message,
      uuid: id,
      timestamp,
      userType: "external",
      cwd: session.cwd,
      sessionId: session.sessionId,
    });
    parentUuid = id;
  }
  return lines;
}

/**
 * Write `prior` as a new session of the agent's in `cwd`, and answer its id to `--resume` — or
 * `undefined` when the transcript cannot be seeded or the file cannot be written, which leaves the
 * caller its rendered prompt.
 */
export async function writeNativeSeed(prior: readonly JsonValue[], cwd: string, options: { configDir?: string } = {}): Promise<string | undefined> {
  const turns = seedTurns(prior);
  if (turns === undefined || turns.length === 0) return undefined;
  const sessionId = randomUUID();
  const path = nativeSessionPath(sessionId, cwd, options.configDir);
  try {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, seedLines(turns, { sessionId, cwd }).map((line) => JSON.stringify(line)).join("\n") + "\n", { encoding: "utf8", flag: "wx" });
  } catch {
    return undefined;
  }
  return sessionId;
}
