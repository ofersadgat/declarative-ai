/**
 * @declarative-ai/agents-cli — delegated agents reached through a CLI subprocess.
 *
 * The sibling of `@declarative-ai/agents-api`: `claude-code` split by INVOCATION MECHANISM
 * (DESIGN §4.4). Both packages drive the same normalized `AgentQuery` seam and produce the
 * same shape of `runtime` registry entry, so a workflow authored against one runs against the other;
 * only how the agent is reached — and therefore how its safety policy is enforced — differs.
 *
 * Two CLIs live here now: `claude` (`./cliQuery`) and `codex` (`./codexQuery`). They share the process
 * seam (`./process`) and the MCP bridge (`./mcpBridge`) and nothing else — a flag vocabulary and a
 * message schema are exactly the parts that do not generalize.
 */
export * from "./cliQuery.js";
export * from "./codexQuery.js";
export * from "./codexRuntime.js";
export * from "./mcpProtocol.js";
export * from "./mcpBridge.js";
export * from "./process.js";
export * from "./runtime.js";
// The CLI agent as an `Executor` (DESIGN §4.4) — this package's half of the agent hierarchy, which is
// what lets a PROMPT op be answered by `claude` or `codex` rather than by a provider endpoint.
export * from "./cliExecutor.js";
