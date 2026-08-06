/**
 * @declarative-ai/agents-api — delegated agents reached through an in-process SDK.
 *
 * `claude-code` split by INVOCATION MECHANISM (DESIGN §4.4); the CLI-driven sibling is
 * `@declarative-ai/agents-cli`. This package also owns the normalized `AgentQuery` SEAM both share, so
 * the two adapters differ only in how the agent is reached — and therefore in how its safety policy is
 * enforced (`callback` here, `config` there).
 */
export * from "./seam.js";
export * from "./runtime.js";
// The agent as an `Executor` (DESIGN §4.4) — a `PromptExecutor` whose call reaches an agent, which is
// what lets a PROMPT op be answered by one. The registry-entry adapters in `runtime.js` remain, and
// now delegate here rather than carrying their own copy of the session and tool logic.
export * from "./agentExecutor.js";
export * from "./sdkQuery.js";
