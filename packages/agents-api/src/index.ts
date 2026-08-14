/**
 * @declarative-ai/agents-api — delegated agents reached through an in-process SDK.
 *
 * `claude-code` split by INVOCATION MECHANISM (DESIGN §4.4); the CLI-driven sibling is
 * `@declarative-ai/agents-cli`. This package also owns the normalized `AgentQuery` SEAM both share, so
 * the two adapters differ only in how the agent is reached — and therefore in how its safety policy is
 * enforced (`callback` here, `config` there).
 */
export * from "./seam.js";
// Making a NAMED binary launchable — the Windows resolution both transports run a `binaryPath` through.
export * from "./binary.js";
// Serving OUR tools to an agent over MCP — the half that is the same on both transports, so the CLI
// sibling shares this implementation instead of keeping its own.
export * from "./mcpTools.js";
// Reading an agent's stream — ONE mapping, shared by both transports because both carry the same
// messages: the SDK drives the CLI as a subprocess and passes its lines through untouched.
export * from "./streamMessages.js";
export * from "./runtime.js";
// The agent's OWN session file, read back — the concrete `AgentSessionReader` over
// `~/.claude/projects`, and the fold that keeps what the stream never carried.
export * from "./nativeSession.js";
// The agent as an `Executor` (DESIGN §4.4) — a `PromptExecutor` whose call reaches an agent, which is
// what lets a PROMPT op be answered by one. The registry-entry adapters in `runtime.js` remain, and
// now delegate here rather than carrying their own copy of the session and tool logic.
export * from "./agentExecutor.js";
export * from "./sdkQuery.js";
