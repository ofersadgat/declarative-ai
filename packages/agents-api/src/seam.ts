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

/** How hard to think, as a level. The neutral `ReasoningSpec.effort`, restated for the reason
 *  {@link AgentReasoning} gives — `xhigh` is in the vocabulary because a delegated agent is the
 *  transport that has such a tier. */
export type AgentEffort = "low" | "medium" | "high" | "xhigh";

/** A tool-use the agent wants to make, surfaced to our approver via {@link AgentPermissionCallback}. */
export interface AgentToolRequest {
  toolName: string;
  input: FunctionInputs;
}

/**
 * The decision our approver returns for an agent tool-use (mapped to the SDK's allow/deny result).
 *
 * `updatedInput` is the wire's "run it with THESE arguments" channel. Ordinary approvals never set it
 * (the transports echo the original input, which the CLI's parse requires); it exists for the one tool
 * family whose ANSWER travels as input — `AskUserQuestion`, where the human's chosen options ride back
 * on the allow (`{...input, answers}`), which is the documented contract for answering it.
 */
export type AgentPermissionDecision = { allow: true; updatedInput?: FunctionInputs } | { allow: false; reason?: string };

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
  /**
   * Which model the agent should use — the PROVIDER-NATIVE name, route prefix already stripped.
   *
   * `sonnet`, not `claude-cli/sonnet`: the prefix chose this transport and has no meaning to the
   * binary, which would reject it as an unknown model. Absent ⇒ the agent's own default, which is the
   * ordinary case and the whole reason an agent needs no configuration.
   *
   * An adapter that cannot honour a specific model must REFUSE rather than drop it. Silently running
   * a different model than the one asked for is the failure this field exists to make impossible —
   * and it is expensive as well as wrong, since the models differ by an order of magnitude in price.
   */
  model?: string;
  /** Working directory (from `ctx.workspace.root`). */
  cwd?: string;
  /**
   * WHICH binary to run — an absolute path, or a name to resolve.
   *
   * Absent ⇒ each transport's own default: the SDK's bundled executable, or the bare `claude` on the
   * PATH. Present, it is resolved through {@link resolveAgentBinary} before it reaches either, because
   * on Windows neither transport spawns through a shell and a bare name is therefore not launchable at
   * all — an npm `claude.cmd` shim fails with `spawn EINVAL` (Node ≥ 20.12) and a bare command name
   * fails as ENOENT even with the executable on the PATH.
   *
   * The point is a host that ships or pins its own build: a workflow must be able to say WHICH agent
   * answered, not just that one did.
   */
  binaryPath?: string;
  /**
   * The environment the agent runs under. Absent ⇒ it inherits this process's.
   *
   * A whole environment rather than a patch, matching what both transports take underneath — so a
   * caller that means to ADD one variable passes `{...process.env, X: "y"}`, and one that means to
   * start clean can. Forwarded verbatim; nothing here interprets a variable's meaning.
   */
  env?: NodeJS.ProcessEnv;
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
  /**
   * How hard the agent should think, in the neutral vocabulary (`ReasoningSpec`, restated here for the
   * reason {@link AgentReasoning} gives).
   *
   * `effort` is the level; `budgetTokens` is an explicit thinking budget. A transport that carries one
   * and not the other honours what it can and REFUSES the rest — dropping a caller's reasoning request
   * gets a cheaper, worse answer than the one that was paid for, and says nothing about it.
   */
  reasoning?: { effort?: AgentEffort; budgetTokens?: number };
  /**
   * Cap on the agent's OWN loop — how many model→tool→model turns it may take.
   *
   * The neutral `maxSteps`, which means the same thing here that it does for an executed tool loop: it
   * is the bound on how long the thing may run before it has to answer with what it has.
   */
  maxSteps?: number;
  /**
   * Whether the agent may use tools at all.
   *
   * Only the two ends of `LlmToolChoice` are expressible to an agent: `auto` (its own judgement, the
   * default) and `none` (answer from what it already knows). `required` and a named tool are choices
   * about ONE model turn, and an agent runs a whole loop — so a transport handed either refuses.
   */
  toolChoice?: "auto" | "none";
  /**
   * A JSON Schema the agent's ANSWER must satisfy — the delegated half of a prompt op's output schema.
   *
   * Both `claude` transports carry it natively (`--json-schema` on argv, `outputFormat: {type:
   * "json_schema"}` through the SDK) and both answer on {@link AgentResult.structured}, retrying inside
   * their own loop until the value validates. That is what makes an agent able to serve a state with a
   * declared output at all: without it the agent answers in prose, the engine finds none of the slots
   * the state declared, and the run fails with "did not produce required output".
   *
   * A transport that cannot constrain its output must REFUSE rather than drop it, like every other
   * field here. Answering in prose when a schema was asked for is not a lesser answer — it is a
   * different one, and the caller cannot tell it apart from a model that ignored the request.
   */
  schema?: JsonValue;
  /**
   * Transport-specific settings, already selected by provider key and passed through VERBATIM.
   *
   * `LlmConfiguration.providerOptions` is documented as "the full escape hatch… passed through to the
   * provider verbatim", and this is its delegated-agent half: `providerOptions.claudeCode` reaches the
   * `claude` transports, `providerOptions.codex` reaches codex. The neutral core stays strict-parsed
   * and nothing agent-specific leaks into `LlmConfiguration`.
   *
   * Each transport documents the keys it understands and REFUSES one it does not, because a setting
   * silently ignored is the failure this whole seam is built to avoid.
   */
  providerOptions?: Record<string, JsonValue>;
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
   * With `forkSession`: cut the copy at this message rather than at wherever the session now stands.
   *
   * `--resume-session-at <message id>` — "only messages up to and including the assistant message
   * with <message.id>". What makes a branch behind the remote's tip one server-side copy instead of a
   * full replay. An adapter without the flag must not accept this silently: copying the whole session
   * for a branch that ends earlier is the bug the option exists to avoid.
   */
  resumeSessionAt?: string;
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

/**
 * One reasoning block the agent emitted, kept whole.
 *
 * Declared here rather than imported from `@declarative-ai/llm` for the same reason `LlmMetrics`
 * restates `ExecMetrics`'s fields: this seam is what `agents-cli` compiles against, and that package
 * has no business depending on the llm layer. The shape is kept structurally compatible by hand, so
 * the executor maps it onto `ReasoningSegment` with a cast and no translation.
 *
 * `providerMetadata` is where the Anthropic thinking-block SIGNATURE rides. That is not decoration: a
 * signed thinking block must come back byte-identical on the next turn or the provider rejects the
 * conversation, which is precisely what a lossy round-trip destroys.
 */
export interface AgentReasoning {
  text: string;
  /** Native provider metadata — open by nature, JSON by construction (§2.2). */
  providerMetadata?: Record<string, JsonValue>;
}

/** A tool the agent invoked mid-run. Structurally `llm`'s `ToolCall` — see {@link AgentReasoning}. */
export interface AgentToolCall {
  toolCallId?: string;
  toolName: string;
  input: JsonValue;
}

/** The result of a tool the agent ran. Structurally `llm`'s `ToolResult`. */
export interface AgentToolResult {
  toolCallId?: string;
  toolName?: string;
  output: JsonValue;
}

/**
 * What a run consumed, in the neutral vocabulary. Structurally `llm`'s `TokenCounts` — see
 * {@link AgentReasoning} for why it is restated rather than imported.
 *
 * The split matters for MONEY, not for curiosity: a cache read is billed at roughly a tenth of the base
 * rate and a 1-hour cache write at roughly twice it, so a single `inputTokens` figure cannot be priced.
 */
export interface AgentTokenCounts {
  /** Total input, INCLUDING cache reads and writes — the provider's billed input. */
  inputTokens?: number;
  outputTokens?: number;
  /** Uncached (fresh) input tokens. */
  noCacheTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  /** The 1-hour-TTL subset of `cacheWriteTokens`. */
  cacheWrite1hTokens?: number;
  reasoningTokens?: number;
  totalTokens?: number;
}

/** The agent's final answer for a run. */
export interface AgentResult {
  text: string;
  /**
   * The SCHEMA-CONSTRAINED answer, when {@link AgentQueryOptions.schema} asked for one.
   *
   * A separate field rather than a parse of `text`, because that is how both transports report it: the
   * terminal message carries prose on `result` AND the constrained value on `structured_output`. The
   * prose summarizes the work; it is not a serialization of the value, so reading the value out of it
   * would parse the wrong thing on every run that produced both.
   *
   * Absent ⇒ no schema was asked for, or the agent could not produce one. The second is a FAILURE to
   * answer rather than an invitation to fall back to the prose — a schema was the request.
   */
  structured?: JsonValue;
  costUsd?: number;
  /**
   * Why the run ENDED, in the same neutral vocabulary a provider call reports
   * (`stop` / `length` / `tool-calls` / `content-filter` / `error` / `unknown`).
   *
   * Absent ⇒ the transport had nothing to say and the caller may assume nothing. What it must not be
   * is a fabricated `"stop"`: a run that hit its turn cap or its budget ceiling produced a PARTIAL
   * answer, and reporting that as a clean stop is how a truncated review reads as a complete one.
   */
  finishReason?: string;
  /** What the run consumed. Absent ⇒ the transport reported nothing, which is a different claim
   *  from zero. */
  usage?: AgentTokenCounts;
  /** The provider's exact usage object, kept so `costUsd` stays recomputable when our reading of the
   *  fields turns out to be wrong or incomplete. Open by nature, JSON by construction (§2.2). */
  rawUsage?: JsonValue;
  /**
   * The agent's session id as of this run — its own, not ours.
   *
   * Always the id the run ACTUALLY ended in: a new one after a fork, the resumed one otherwise. A
   * value that differs from the handle we resumed means the remote moved underneath us
   * (DESIGN.md §1.6), which is the only way to notice server-side compaction or an out-of-band
   * resume.
   */
  sessionId?: string;
  /**
   * The provider-native model that ANSWERED — `gpt-5-codex`, `claude-opus-4-5`.
   *
   * Reported on the terminal message for a transport whose model arrives with its session rather
   * than in an init event (codex names it on `session_configured`); the `claude` adapters report the
   * same fact through `system`/`init` while the turn is still running, and either route ends in the
   * same place. What it is FOR is the settle: a call asked to use "your own default" must not record
   * the placeholder it was asked with, because a persisted model id is read back as a fact about
   * what ran — see `AgentExecutor.resolvedModel`.
   */
  model?: string;
}

/**
 * One normalized message from an agent's stream.
 *
 * It used to carry the terminal result and nothing else — every other message collapsed to
 * `{type: "other"}` — which is why a delegated call reported a fabricated `finishReason: "stop"`, a
 * synthesized one-turn message log, and no tokens at all. None of that was missing at the source: the
 * agent hands back its whole log, on the same wire, and it was being thrown away at the mapping.
 *
 * The variants:
 *  - `result` — the terminal message. Carries {@link AgentResult}.
 *  - `assistant` / `user` — a turn the agent appended, with `message` VERBATIM and the projections
 *    (`thinking`, `toolCalls`, `toolResults`) read off it.
 *  - `partial` — a text delta while the answer is still being written. `delta` is the new text.
 *  - `thinking-partial` — a reasoning delta while the model is still thinking. `delta` is the new
 *    reasoning text. Its own variant rather than a `partial`, because the answer's stream must never
 *    contain reasoning — and a consumer that wants to SHOW thinking as it happens (the point of
 *    streaming a minutes-long turn) still needs the delta somewhere.
 *  - `provider_event` — everything with no neutral home, forwarded opaquely (see {@link event}).
 *  - `other` — a message this mapping recognises and has nothing to say about.
 */
export interface AgentStreamMessage {
  type: "result" | "assistant" | "user" | "partial" | "thinking-partial" | "provider_event" | "other";
  /** Present on the terminal `result` message. */
  result?: AgentResult;
  /** A run-fatal error the agent reported, as prose. */
  error?: string;
  /**
   * The agent's own MACHINE-READABLE failure code, when it gave one —
   * `authentication_failed`, `rate_limit`, `overloaded`, `billing_error`, `server_error`, …
   *
   * It rides beside {@link error} rather than replacing it because the two answer different
   * questions. The prose is what a human reads ("Not logged in · Please run /login"); the code is what
   * decides whether RETRYING is sound. Without it every delegated failure classifies as `permanent`,
   * since an `AgentError` carries no status and no retryable flag — so a transient overload inside the
   * agent's own loop looks like a broken workflow and defeats every retry wrapper above it, which is
   * exactly what {@link AgentExecutor.invoke}'s "classified, not flattened" comment promises not to do.
   */
  errorCode?: string;
  /**
   * The provider's OWN message object for this turn, verbatim.
   *
   * The provider's log rather than a reconstruction, which is what `LlmOutput.messages` is documented
   * to be and what a later replay has to work from. The projections below are a convenience for a
   * consumer that wants one field; this is the thing that goes back on the wire.
   */
  message?: JsonValue;
  /** The plain OUTPUT text this turn contributed — the answer, never the reasoning. It is what
   *  `ReasoningSegment.textOffset` is measured against, so a consumer can place a thinking block
   *  against the text it accompanies. */
  text?: string;
  /** Reasoning blocks this turn emitted, signatures intact. */
  thinking?: AgentReasoning[];
  /** Tools this turn asked to call. */
  toolCalls?: AgentToolCall[];
  /** Tool results this turn carried back. */
  toolResults?: AgentToolResult[];
  /** The new text on a `partial`, or the new reasoning text on a `thinking-partial`. */
  delta?: string;
  /**
   * The tool call this turn belongs to, when it is a SUBAGENT's — the spawning `Task` call's id,
   * off the stream envelope's `parent_tool_use_id`.
   *
   * Absent means the main thread. Dropping this field is not lossy, it is WRONG: a subagent's turns
   * arrive on the same stream, and untagged they interleave into the main conversation
   * indistinguishably — a record that then claims the main thread said things a subagent said.
   */
  parentToolUseId?: string;
  /**
   * A `provider_event`'s payload, forwarded opaquely.
   *
   * The agent emits a great deal that has no neutral home and should not be given one — session init,
   * compaction boundaries, hook lifecycle, task progress, API retries, rate-limit windows, permission
   * denials. Naming each of them in the neutral vocabulary would teach `exec` Claude's own vocabulary
   * for the sake of events it cannot act on. Same precedent as `rawUsage` and `providerMetadata`: open
   * by nature, JSON by construction.
   */
  event?: JsonValue;
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

/**
 * A LIVE agent run: its message stream, plus whatever steering the transport can offer.
 *
 * It extends `AsyncIterable<AgentStreamMessage>`, which is what the seam used to be, so every existing
 * `for await` and every `async function*` fake keeps working unchanged — the methods are additions, not
 * a new shape.
 *
 * They are **optional, never throwing stubs**. Absent MEANS unsupported: `if (run.interrupt)` is the
 * runtime check and `capabilities.sessionSteering` is how a caller knows before the call. codex leaves
 * every one of them undefined, and correctly — SIGINT to a subprocess is a kill, not a graceful turn
 * end, and offering `interrupt` that killed the run would answer "stop and tell me what you found" by
 * throwing the answer away.
 *
 * ⚠️ `interrupt()` is NOT cancellation. The turn ends early, a `result` message still arrives, and the
 * call SUCCEEDS with the partial answer. See {@link ExecControl}.
 */
export interface AgentRun extends AsyncIterable<AgentStreamMessage> {
  /** End the current turn early and let the run settle normally. Idempotent. */
  interrupt?(): Promise<void>;
  /** Add a user message to a run already under way. */
  send?(text: string): Promise<void>;
  /** Change the permission posture for the rest of the run. */
  setPermissionMode?(mode: AgentPermissionMode): Promise<void>;
  /** Change the model used for subsequent responses. */
  setModel?(model: string): Promise<void>;
}

/** The seam: run an agent query and yield its message stream. */
export type AgentQuery = (opts: AgentQueryOptions) => AgentRun;
