/**
 * @declarative-ai/agents-cli — delegated agents reached through a CLI subprocess.
 *
 * The sibling of `@declarative-ai/agents-api`: `claude-code` split by INVOCATION MECHANISM
 * (DESIGN §4.4). Both packages drive the same normalized `AgentQuery` seam and produce the
 * same shape of `runtime` registry entry, so a workflow authored against one runs against the other;
 * only how the agent is reached — and therefore how its safety policy is enforced — differs.
 */
export * from "./cliQuery";
export * from "./mcpProtocol";
export * from "./mcpBridge";
export * from "./runtime";
