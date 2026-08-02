/**
 * The injectable agent-query SEAM — a normalized shape the {@link createClaudeCodeFunction} adapter drives,
 * DECOUPLED from the Claude Agent SDK's concrete types. The default implementation (`sdkQuery.ts`) maps this
 * onto `@anthropic-ai/claude-agent-sdk`'s `query()`; tests inject a fake. Keeping the adapter logic against
 * this interface (not the SDK's) is what lets the whole package be built and tested without the SDK
 * installed and without an API key.
 */
import type { FunctionInputs, JsonSchema, JsonValue, SyncOutputValidator } from "@declarative-ai/exec";

/** The permission mode handed to the delegated agent (its NATIVE profile control). */
export type AgentPermissionMode = "default" | "plan" | "acceptEdits" | "bypassPermissions";

/** A tool-use the agent wants to make, surfaced to our approver via {@link AgentPermissionCallback}. */
export interface AgentToolRequest {
  toolName: string;
  input: FunctionInputs;
}

/** The decision our approver returns for an agent tool-use (mapped to the SDK's allow/deny result). */
export type AgentPermissionDecision = { allow: true } | { allow: false; reason?: string };

/** The callback the agent calls before each gated tool-use — the adapter routes it to `ctx.approve`. */
export type AgentPermissionCallback = (req: AgentToolRequest, opts: { signal: AbortSignal }) => Promise<AgentPermissionDecision>;

/**
 * A tool the adapter INJECTS into the delegated agent (over MCP), so the agent calls OUR implementation —
 * making a `bash`/`read_file` on `claude-code` behave identically to the composed `llm` runtime (the
 * portable-vocabulary goal, DESIGN §5.1, "Tool renames are just overlay bindings"). The `run` closes over the runtime's ctx.
 */
export interface InjectedTool {
  description?: string;
  inputSchema: JsonSchema;
  run: (input: FunctionInputs) => JsonValue | Promise<JsonValue>;
}

/** Options the adapter builds from the op inputs + ctx and hands to the query seam. */
export interface AgentQueryOptions {
  prompt: string;
  /** Working directory (from `ctx.workspace.root`). */
  cwd?: string;
  /** Tool allow-list (the logical names from `runtime.tools`). Note what this MEANS to an agent: it is a
   *  PRE-APPROVAL list, so a name here is not put to `canUseTool`/`--permission-prompt-tool`. */
  allowedTools?: string[];
  /** Tool DENY-list — the `deny` entries of the compiled policy baseline, by the name the agent
   *  addresses. Without this channel a per-tool `deny` that needs no human could not be expressed to a
   *  delegated agent at all; an adapter that cannot honour it must refuse, not drop it. */
  disallowedTools?: string[];
  /** OUR tools to inject into the agent over MCP, keyed by logical name — the agent calls these impls. */
  mcpTools?: Record<string, InjectedTool>;
  permissionMode?: AgentPermissionMode;
  canUseTool?: AgentPermissionCallback;
  /** Boundary validation for INJECTED tool arguments. An agent's tool call arrives as untyped JSON and
   *  no MCP server validates it, so an adapter that injects tools checks each call against the tool's
   *  own `inputSchema` through this seam (`json`'s three lines — no ajv on the agent path). SYNC by
   *  requirement: the gate sits inline in the MCP request handler, and tool `inputSchema`s are inline
   *  documents — the inline family's truth. */
  validator?: SyncOutputValidator;
  abortSignal?: AbortSignal;
  /**
   * The agent's own session to continue, when there is one (DESIGN.md §1.6, "Native fork").
   *
   * This adapter has the primitive the design wants: `resume` continues a conversation server-side,
   * reading zero messages — no replay, no transcript on the wire.
   *
   * ⚠️ A resumed session is bound to the CWD IT WAS CREATED IN. Transcripts live at
   * `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, where `<encoded-cwd>` is the absolute
   * working directory with every non-alphanumeric character replaced by `-`. Resuming from a
   * DIFFERENT cwd silently starts a fresh session rather than failing — so a caller that moved the
   * workspace and kept the handle gets an empty conversation reported as a successful resume.
   */
  resume?: string;
  /**
   * Branch the resumed session instead of continuing it.
   *
   * `resume` + `forkSession` starts a NEW session id seeded with a copy of the original's history and
   * leaves the original untouched — a fork, done server-side at no replay cost. The new id comes back
   * on {@link AgentResult.sessionId}, and recording it is NOT optional: a fork that kept its parent's
   * handle would put two branches into one remote session.
   *
   * Forking branches the CONVERSATION, not the filesystem — both branches see one working directory.
   */
  forkSession?: boolean;
  /**
   * The conversation to REPLAY into this call — the fallback for an adapter that appends natively but
   * cannot branch (SESSIONS.md §6, "Strategies").
   *
   * Codex is the case this exists for: `codex exec resume <id>` continues a conversation server-side,
   * and there is no fork primitive at all. So its append is free and its FORK has to be replayed, which
   * is a per-call decision the adapter makes from `session.mode` — never a caller's.
   *
   * Mutually exclusive with {@link AgentQueryOptions.resume} by construction: replaying a transcript
   * into a session that already contains it duplicates the conversation. An adapter that is handed both
   * must refuse rather than pick one.
   *
   * ⚠️ Replay against a DELEGATED agent is lossy in a way it is not against a message-based provider.
   * There is no message array on the wire — a delegated CLI takes one prompt — so the transcript is
   * rendered into text; and what a delegated agent contributes to a transcript is already an outline
   * (one assistant turn per call, since it keeps its real log server-side). Both halves of that are why
   * the lineage edge records such a branch as summary-seeded rather than as a native fork.
   */
  messages?: readonly JsonValue[];
}

/** The agent's final answer for a run. */
export interface AgentResult {
  text: string;
  costUsd?: number;
  /**
   * The agent's session id as of this run — its own, not ours.
   *
   * Always the id the run ACTUALLY ended in: a new one after a fork, the resumed one otherwise. A
   * value that differs from the handle we resumed means the remote moved underneath us
   * (DESIGN.md §1.6), which is the only way to notice server-side compaction or an out-of-band
   * resume.
   */
  sessionId?: string;
}

/** A normalized message from the agent stream — the adapter only needs the terminal result + any error. */
export interface AgentStreamMessage {
  type: "result" | "assistant" | "other";
  /** Present on the terminal `result` message. */
  result?: AgentResult;
  /** A run-fatal error the agent reported (mapped to a permanent outcome). */
  error?: string;
}

/**
 * Read a provider-side conversation back, for re-syncing after the remote moved (DESIGN.md §1.6).
 *
 * Separate from {@link AgentQuery} because it is a genuinely separate capability, and an optional
 * one: Claude Code has `getSessionMessages()`, Managed Agents has `events.list`, the Messages API has
 * neither — and needs neither, being stateless and therefore unable to diverge. An adapter without
 * one leaves a resync EMPTY, which the lineage edge records rather than passing off as a conversation
 * that happened to be empty.
 *
 * ⚠️ Bound to the same cwd the session was created in, for the same reason `resume` is — transcripts
 * are per-project-directory files. Reading from elsewhere finds nothing, which is indistinguishable
 * from a conversation with no messages.
 */
export type AgentSessionReader = (providerSessionId: string, cwd?: string) => Promise<readonly unknown[]>;

/** The seam: run an agent query and yield its message stream. */
export type AgentQuery = (opts: AgentQueryOptions) => AsyncIterable<AgentStreamMessage>;
