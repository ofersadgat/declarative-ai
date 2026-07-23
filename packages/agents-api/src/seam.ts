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
}

/** The agent's final answer for a run. */
export interface AgentResult {
  text: string;
  costUsd?: number;
}

/** A normalized message from the agent stream — the adapter only needs the terminal result + any error. */
export interface AgentStreamMessage {
  type: "result" | "assistant" | "other";
  /** Present on the terminal `result` message. */
  result?: AgentResult;
  /** A run-fatal error the agent reported (mapped to a permanent outcome). */
  error?: string;
}

/** The seam: run an agent query and yield its message stream. */
export type AgentQuery = (opts: AgentQueryOptions) => AsyncIterable<AgentStreamMessage>;
