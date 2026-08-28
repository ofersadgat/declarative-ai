/**
 * One conversation, stored once — the ENTRY format (JaiRA's RECORDS.md).
 *
 * A record used to keep the same conversation in three encodings and two derived indexes beside
 * them: `messages` (normalized), `nativeLines` (the agent's own on-disk log), the session outcome's
 * copy of `messages`, plus `toolCalls`/`toolResults`/`thinking` projected out of the first. Measured
 * on one run of JaiRA's `feature` workflow, the DECLARED outputs were 2.4% of the stored bytes, and
 * `toolResults`/`toolCalls` were reconstructible from `messages` 21 times out of 21.
 *
 * Worse than the size: the encodings had no join. A captured line carrying a `toolUseResult` holds
 * no tool-use id, so a reader could only pair it with the message thread by role and order, and the
 * code that did said so ("index arithmetic against the turns is off by one the moment a run
 * begins"). One array removes the join rather than fixing it.
 *
 * ## What is provider-neutral, and what is not
 *
 * The core is what any agent transport has: identity, threading, a timestamp, a role, content
 * blocks, and — for a subagent — which call spawned it. Everything else rides `providerData`
 * VERBATIM and is never interpreted. The rule for putting a field in the core is that a second
 * provider would have the same thing under a different name, not that this one has it.
 */
import type { JsonValue } from "@declarative-ai/json";

/** What a conversation is made of: messages, and the session facts that happened around them. */
export type Entry = MessageEntry | EventEntry;

export interface BaseEntry {
  /**
   * This entry's own id, as the provider gave it.
   *
   * Identity, so a streamed entry and a captured one merge instead of appearing twice. OPTIONAL
   * because not every transport has one: a plain model call returns a list of messages with no ids
   * at all, and synthesizing them would be inventing a fact rather than recording one. Absent, the
   * entry is identified by its position, which is all the ordering a single producer needs.
   */
  uuid?: string;
  /** The entry this one answers or follows — the provider's threading, kept rather than re-derived. */
  parentUuid?: string;
  /** ISO 8601. */
  timestamp: string;
  /**
   * Subagent membership. Absent = the main chain.
   *
   * One object rather than the two flat fields a transport spells it with, because the pair is one
   * fact: this entry belongs to the agent `id`, which the tool call `parentToolUseId` spawned. It is
   * also the join the old shape lacked — a subagent's messages were keyed by the spawning call and
   * its captured lines by the agent id, two key spaces for one conversation.
   */
  sidechain?: { id: string; parentToolUseId?: string };
  /** Which transport produced this — `"anthropic"`, `"openai"`, … */
  provider: string;
  /**
   * Fields with no home in the core, verbatim.
   *
   * ONLY those: never a copy of a field above, and never a per-record invariant. A transport that
   * stamps `sessionId`, `cwd`, `version` and `gitBranch` on every line is saying one thing many
   * times — measured at 6% of one capture, with one or two distinct values each — and those belong
   * on the record, once.
   */
  providerData?: Record<string, JsonValue>;
  /**
   * What the HOST measured about this entry, as distinct from what the provider stamped on it.
   *
   * `timestamp` is the provider's own clock and goes back on the wire; these are numbers only a
   * streaming consumer can take, and nobody can recover afterwards: when the first fragment of this
   * turn appeared, and how long the model spent thinking before it began answering — the
   * "thought for 12 s" a viewer shows.
   *
   * ON THE ENTRY rather than in a parallel array beside the conversation. An array of stamps aligned
   * by index is a join, and a join is the thing this format exists to remove: a shift of one labels
   * every turn with its neighbour's duration, which is worse than no label at all.
   */
  timing?: { at?: number; startedAt?: number; thoughtMs?: number };
}

export interface MessageEntry extends BaseEntry {
  kind: "message";
  /**
   * This turn was still being written when the record was last written.
   *
   * A conversation is ONE array that grows as fragments arrive, so the turn in flight is an entry
   * like any other — there is deliberately no second field holding "the partial" beside the finished
   * ones, because two encodings of one conversation is what this format replaced.
   *
   * What a reader must NOT do is treat it as something the model finished saying. It is not
   * replayable — half an assistant turn is not the exchange that happened — and a viewer renders it
   * as the tail it is. This flag is what carries that, and it is a property OF the entry, so it
   * travels with the thing it describes and disappears when the finished turn replaces it.
   */
  partial?: boolean;
  /**
   * The provider's own role. `"user"` / `"assistant"` are the two every transport has; the AI SDK
   * also writes `"tool"` for a result turn and `"system"` for a preamble, and a provider may add
   * one tomorrow.
   *
   * Typed as a string rather than a closed union for the reason the content is stored verbatim: a
   * role coerced into the nearest known value replays as a different message than the one that was
   * sent, and `"tool"` collapsing to `"user"` is exactly that.
   */
  role: string;
  /**
   * The provider's own content, VERBATIM — its parts, or the bare string it used instead.
   *
   * Both spellings are kept because both are what a provider sent and both go back on the wire. A
   * string is not normalized into one text part here: it would replay as a different request body
   * than the one that was made, and "the record holds what happened" is the property everything else
   * in this format is arranged around. {@link blocksOf} is where a reader gets one shape.
   */
  content: Block[] | string;
}

export interface EventEntry extends BaseEntry {
  kind: "event";
  /**
   * A session fact that is not a message — a context injection, a queued operation, the agent's own
   * name for the conversation.
   *
   * Open by design, and needing no `Other` case the way blocks do: there are no sibling literals to
   * be confused with, so a type this reader has no name for degrades to a row saying its own name
   * rather than to a value the type system cannot narrow.
   */
  event: { type: string; data?: JsonValue };
}

// --- blocks ------------------------------------------------------------------

export interface TextBlock {
  type: "text";
  text: string;
}

export interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  /** Anthropic signs reasoning blocks and the signature must go back byte-identical. */
  signature?: string;
}

export interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: JsonValue;
  /** Which agent asked, where a transport distinguishes the main thread from a subagent. */
  caller?: string;
}

/**
 * A tool's answer, in the two faces it has: what the model SAW, and what the provider RECORDED.
 *
 * Neither derives from the other in general. Measured across 192 captured results in 10 shapes: a
 * Glob that matched nothing rendered `"No files found"` against a structured
 * `{filenames: [], durationMs: 258, numFiles: 0, truncated: false}` — the prose is in neither the
 * object nor the object's fields in the prose. A path error rendered inside `<tool_use_error>` tags
 * against a structured string prefixed `Error: `. And a `Read` rendered the file with `N\t` line
 * numbers against a structured record carrying `filePath`, `numLines` and `totalLines`.
 *
 * So both are kept, and `text` is dropped only where a registered renderer reproduces it from
 * `data` — see {@link renderToolResult}. At least one of the two is always present.
 */
export interface ToolResultBlock {
  type: "tool_result";
  toolUseId: string;
  isError?: boolean;
  /** The provider's structured record, when it kept one. */
  data?: JsonValue;
  /** What the model saw. Absent only when a renderer covers `data`'s shape. */
  text?: string;
}

/**
 * A block whose provider type this version has no case for — stored EXACTLY as it arrived.
 *
 * No synthetic `"unknown"` tag: the stored data would then be a claim the provider never made, and a
 * round-trip would lose the name it actually used. The discrimination lives in
 * {@link isKnownBlock} instead, which is what keeps the known cases precisely typed — a member with
 * `type: string` in the union is assignable to every literal, so `case "text":` would widen to
 * `TextBlock | OtherBlock` and `b.text` would stop being a string.
 */
export interface OtherBlock {
  type: string;
  [key: string]: JsonValue;
}

export type KnownBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock;
export type Block = KnownBlock | OtherBlock;

export const KNOWN_BLOCK_TYPES = ["text", "thinking", "tool_use", "tool_result"] as const;

/**
 * Split the blocks this version knows from the ones it does not.
 *
 * Use it before switching: inside the guard the union discriminates properly and an `assertNever`
 * default fires the day a known type is added without a case; outside it, the block is the
 * provider's and gets rendered by its own name rather than filtered away. Filtering was the
 * alternative, and it quietly shrinks a transcript every time the vocabulary grows.
 */
export function isKnownBlock(block: Block): block is KnownBlock {
  return (KNOWN_BLOCK_TYPES as readonly string[]).includes(block.type);
}

// --- tool-result rendering ---------------------------------------------------

/**
 * Reproduce what the model saw from what the provider recorded, for shapes where the one determines
 * the other.
 *
 * Keyed by the SHAPE of `data`, not by the tool's name, for three reasons the capture gives: 84
 * `Read` calls produced 83 records of the file shape, so a tool does not always yield its shape; a
 * second tool (`mcp__dai__read_file`) plausibly yields the same one; and the captured result carries
 * no tool id to join a name on anyway.
 *
 * Anything unregistered keeps its `text` as it came. That is the fail-safe direction: a new tool, a
 * new provider or an unrecognized variant costs bytes, never fidelity.
 */
export type ToolResultRenderer = (data: JsonValue) => string | undefined;

/** `{ type: "text", file: { filePath, content, numLines, startLine, totalLines } }` — a file read. */
const renderFileRead: ToolResultRenderer = (data) => {
  if (data === null || typeof data !== "object" || Array.isArray(data)) return undefined;
  const file = (data as { file?: unknown }).file;
  if (file === null || typeof file !== "object" || Array.isArray(file)) return undefined;
  const content = (file as { content?: unknown }).content;
  if (typeof content !== "string") return undefined;
  const startRaw = (file as { startLine?: unknown }).startLine;
  const start = typeof startRaw === "number" ? startRaw : 1;
  // Verified against every file read in one run: 82 of 83 reproduce byte-for-byte, and the 83rd is
  // not a rendering mismatch — it is a read whose result never appeared as a tool result at all.
  // No left-padding: zero of them matched a padded variant.
  return content
    .split("\n")
    .map((line, i) => `${start + i}\t${line}`)
    .join("\n");
};

const RENDERERS: readonly ToolResultRenderer[] = [renderFileRead];

/**
 * What a registered renderer makes of `data`, or `undefined` when none covers its shape.
 *
 * The capture calls this to decide whether to keep `text`; a reader calls it to put `text` back.
 */
export function renderToolResult(data: JsonValue | undefined): string | undefined {
  if (data === undefined) return undefined;
  for (const render of RENDERERS) {
    const text = render(data);
    if (text !== undefined) return text;
  }
  return undefined;
}

/** What the model saw, whether it was stored or has to be rendered back from `data`. */
export function toolResultText(block: ToolResultBlock): string | undefined {
  return block.text ?? renderToolResult(block.data);
}

// --- projections -------------------------------------------------------------

/**
 * The wire history, derived — `{ role, content }` per message entry, in order.
 *
 * Computed rather than stored, which is the rule the whole format turns on: the record used to hold
 * this array twice over (once as `messages`, once inside the session outcome) and it was
 * byte-identical both times.
 *
 * The content is handed back UNTOUCHED, which is what makes this a faithful replay rather than a
 * reconstruction — see {@link entriesOfMessages}.
 */
export function messagesOfEntries(entries: readonly Entry[]): Array<{ role: string; content: Block[] | string }> {
  const out: Array<{ role: string; content: Block[] | string }> = [];
  for (const entry of entries) {
    if (entry.kind !== "message" || entry.sidechain !== undefined) continue;
    out.push({ role: entry.role, content: entry.content });
  }
  return out;
}

/** Every tool call in the main chain, in order — what `toolCalls` used to be stored for. */
export function toolUsesOfEntries(entries: readonly Entry[]): ToolUseBlock[] {
  return blocksOfEntries(entries).filter((b): b is ToolUseBlock => b.type === "tool_use");
}

/** Every tool result in the main chain, in order — what `toolResults` used to be stored for. */
export function toolResultsOfEntries(entries: readonly Entry[]): ToolResultBlock[] {
  return blocksOfEntries(entries).filter((b): b is ToolResultBlock => b.type === "tool_result");
}

/** Every reasoning block, in order — what `thinking` used to be stored for. */
export function thinkingOfEntries(entries: readonly Entry[]): ThinkingBlock[] {
  return blocksOfEntries(entries).filter((b): b is ThinkingBlock => b.type === "thinking");
}

/**
 * Every block of the main chain, NORMALIZED — the read side of storing content verbatim.
 *
 * Two spellings live in stored entries because two producers write them, and {@link blockOfPart}
 * is where they become one vocabulary. Doing it here rather than at write time is what keeps the
 * record replayable: the projections get one shape, and the wire still gets the provider's own.
 */
function blocksOfEntries(entries: readonly Entry[]): Block[] {
  const out: Block[] = [];
  for (const entry of entries) {
    if (entry.kind !== "message" || entry.sidechain !== undefined) continue;
    for (const block of blocksOf(entry)) out.push(block);
  }
  return out;
}

/**
 * One message entry's content as BLOCKS, whichever way the provider spelled it.
 *
 * The bare-string form becomes one text block here rather than at write time — a reader wants one
 * shape and the wire wants the original, and only one of those two can be the stored one.
 */
export function blocksOf(entry: MessageEntry): Block[] {
  const raw = entry.content;
  if (typeof raw === "string") return [{ type: "text", text: raw }];
  return raw.map(blockOfPart);
}

// --- building entries from a provider's messages ------------------------------

/** A message as a transport hands it over: a role and either text or a list of parts. */
export interface RawMessage {
  role?: unknown;
  content?: unknown;
}

const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
const rec = (v: unknown): Record<string, unknown> | undefined =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;

/**
 * One content part, normalized to a {@link Block}.
 *
 * TWO spellings arrive here and both are the same four things. The AI SDK writes `reasoning`,
 * `tool-call` and `tool-result`; an Anthropic-shaped agent log writes `thinking`, `tool_use` and
 * `tool_result`. Normalizing at the edge is what lets everything downstream — the transcript, the
 * projections, the renderers — hold one vocabulary instead of asking which producer it came from.
 *
 * A part this function does not recognize is returned VERBATIM as an {@link OtherBlock}. It keeps
 * its own `type`, so a round-trip loses nothing and a reader shows it by the name the provider used.
 */
export function blockOfPart(part: unknown): Block {
  const p = rec(part);
  if (p === undefined) return { type: "text", text: String(part) };
  const type = str(p["type"]);
  switch (type) {
    case "text":
      return { type: "text", text: str(p["text"]) ?? "" };
    case "reasoning":
    case "thinking": {
      const signature = str(p["signature"]) ?? str(rec(p["providerMetadata"])?.["signature"]);
      return {
        type: "thinking",
        thinking: str(p["thinking"]) ?? str(p["text"]) ?? "",
        ...(signature !== undefined ? { signature } : {}),
      };
    }
    case "tool-call":
    case "tool_use": {
      const caller = str(p["caller"]);
      return {
        type: "tool_use",
        id: str(p["id"]) ?? str(p["toolCallId"]) ?? "",
        name: str(p["name"]) ?? str(p["toolName"]) ?? "",
        input: (p["input"] ?? p["args"] ?? null) as JsonValue,
        ...(caller !== undefined ? { caller } : {}),
      };
    }
    case "tool-result":
    case "tool_result": {
      const content = unwrapToolOutput(p["content"] ?? p["output"]);
      const isError = p["is_error"] === true || p["isError"] === true;
      return {
        type: "tool_result",
        toolUseId: str(p["tool_use_id"]) ?? str(p["toolCallId"]) ?? "",
        ...(isError ? { isError: true } : {}),
        ...(typeof content === "string" ? { text: content } : content !== undefined ? { data: content as JsonValue } : {}),
      };
    }
    default:
      return p as unknown as OtherBlock;
  }
}

/**
 * Turn a transport's message list into entries.
 *
 * `at` stamps every entry, because a producer that reports no per-message time still has one time
 * that is true of all of them — the moment the call settled — and an entry with no timestamp cannot
 * be merged with a captured one later.
 */
export function entriesOfMessages(
  messages: readonly RawMessage[],
  options: { provider: string; at: string; sidechain?: { id: string; parentToolUseId?: string } },
): Entry[] {
  const out: Entry[] = [];
  for (const message of messages) {
    const role = typeof message.role === "string" ? message.role : "user";
    const raw = message.content;
    // A bare string IS one text block. Providers use the short form for a plain turn and the list
    // form the moment anything else rides along; the entry holds one shape either way.
    // VERBATIM. The parts are stored exactly as the provider sent them, and normalization happens
    // on READ ({@link blockOfPart}) rather than on the way in. Normalizing here looked tidier and was
    // wrong: an Anthropic reasoning part carries a signature under `providerOptions` that has to come
    // back byte-identical, and every provider keeps that sort of thing under a name of its own. A
    // record that stores a translation cannot replay the conversation it claims to hold.
    //
    // A bare string still becomes one text part, because that IS the same thing in a shape every
    // consumer can walk, and no provider distinguishes the two on the way back.
    const content: Block[] | string = typeof raw === "string" ? raw : Array.isArray(raw) ? (raw as Block[]) : [];
    out.push({
      kind: "message",
      role,
      content,
      timestamp: options.at,
      provider: options.provider,
      ...(options.sidechain !== undefined ? { sidechain: options.sidechain } : {}),
    });
  }
  return out;
}

/**
 * The provider a resolved model id names — `anthropic/claude-…` → `anthropic`.
 *
 * A bare id has no vendor to report and answers `"unknown"`, which is a fact rather than a guess:
 * `providerData`'s contract is that a reader knows whose vocabulary it is looking at, and inventing
 * a vendor there would be worse than admitting the id did not say.
 */
export function providerOf(modelId: string | undefined): string {
  if (modelId === undefined) return "unknown";
  const slash = modelId.indexOf("/");
  return slash > 0 ? modelId.slice(0, slash) : "unknown";
}

/**
 * Strip the AI SDK's `{ type, value }` envelope from a tool result.
 *
 * The SDK tags its output — `{ type: "json", value: … }`, `{ type: "text", value: "…" }` — where an
 * Anthropic-shaped log just carries the value. Keeping the tag would put the same answer in two
 * shapes depending on which producer ran, which is the thing this format exists to stop; and the tag
 * says nothing a reader cannot see from the value itself.
 *
 * Anything that is not that envelope passes through untouched, including an object that happens to
 * have a `value` — the `type` discriminant has to be there too.
 */
function unwrapToolOutput(output: unknown): unknown {
  const o = rec(output);
  if (o === undefined || !("value" in o)) return output;
  const tag = str(o["type"]);
  return tag === "json" || tag === "text" || tag === "error-json" || tag === "error-text" ? o["value"] : output;
}
