/** The types this package borrows, in one place — so the protocol module names no package path. */
// `OutputValidator` is `json`'s three-line structural seam (re-exported by `exec`), NOT
// `@declarative-ai/validate`: the injected-argument check must not drag ajv onto the agent path, and a
// caller is free to inject any implementation of those three lines.
export type { JsonValue, OutputValidator, SchemaDocument } from "@declarative-ai/exec";
export type { AgentPermissionDecision, AgentToolRequest, InjectedTool } from "@declarative-ai/agents-api";
