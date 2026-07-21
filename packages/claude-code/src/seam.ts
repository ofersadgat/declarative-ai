/**
 * The injectable agent-query SEAM — a normalized shape the {@link createClaudeCodeRuntime} adapter drives,
 * DECOUPLED from the Claude Agent SDK's concrete types. The default implementation (`sdkQuery.ts`) maps this
 * onto `@anthropic-ai/claude-agent-sdk`'s `query()`; tests inject a fake. Keeping the adapter logic against
 * this interface (not the SDK's) is what lets the whole package be built and tested without the SDK
 * installed and without an API key.
 */

/** The permission mode handed to the delegated agent (its NATIVE profile control). */
export type AgentPermissionMode = "default" | "plan" | "acceptEdits" | "bypassPermissions";

/** A tool-use the agent wants to make, surfaced to our approver via {@link AgentPermissionCallback}. */
export interface AgentToolRequest {
  toolName: string;
  input: Record<string, unknown>;
}

/** The decision our approver returns for an agent tool-use (mapped to the SDK's allow/deny result). */
export type AgentPermissionDecision = { allow: true } | { allow: false; reason?: string };

/** The callback the agent calls before each gated tool-use — the adapter routes it to `ctx.approve`. */
export type AgentPermissionCallback = (req: AgentToolRequest, opts: { signal: AbortSignal }) => Promise<AgentPermissionDecision>;

/**
 * A tool the adapter INJECTS into the delegated agent (over MCP), so the agent calls OUR implementation —
 * making a `bash`/`read_file` on `claude-code` behave identically to the composed `llm` runtime (the
 * portable-vocabulary goal, RUNTIMES-AND-PERMISSIONS.md §1/§3). The `run` closes over the runtime's ctx.
 */
export interface InjectedTool {
  description?: string;
  inputSchema: Record<string, unknown>;
  run: (input: Record<string, unknown>) => unknown | Promise<unknown>;
}

/** Options the adapter builds from a `RuntimeOp` + ctx and hands to the query seam. */
export interface AgentQueryOptions {
  prompt: string;
  /** Working directory (from `ctx.workspace.root`). */
  cwd?: string;
  /** Tool allow-list (the logical names from `runtime.tools`). */
  allowedTools?: string[];
  /** OUR tools to inject into the agent over MCP, keyed by logical name — the agent calls these impls. */
  mcpTools?: Record<string, InjectedTool>;
  permissionMode?: AgentPermissionMode;
  canUseTool?: AgentPermissionCallback;
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
