/** The types this package borrows, in one place — so the protocol module names no package path. */
// `OutputValidator` is `json`'s three-line structural seam (re-exported by `exec`), NOT
// `@declarative-ai/validate`: the injected-argument check must not drag ajv onto the agent path, and a
// caller is free to inject any implementation of those three lines.
export type { JsonValue, OutputValidator, SchemaDocument, SyncOutputValidator } from "@declarative-ai/exec";
export type { AgentPermissionDecision, AgentToolRequest, InjectedTool } from "@declarative-ai/agents-api";
// The TRANSPORT-INDEPENDENT half of tool injection. It lives in `agents-api` because both delegated
// transports serve the same tools the same way and this package depends on that one, not the reverse —
// so the descriptor shape, the input gate, and the dispatch are ONE implementation rather than two that
// drift. What stays here is the part that really is the CLI's: the permission-prompt protocol.
export type { McpToolDescriptor, McpToolResult } from "@declarative-ai/agents-api";
export { MCP_SERVER_NAME, injectedToolDescriptors, mcpToolName, runInjectedTool, textResult } from "@declarative-ai/agents-api";
