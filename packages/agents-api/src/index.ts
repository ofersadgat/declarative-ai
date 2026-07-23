/**
 * @declarative-ai/agents-api — delegated agents reached through an in-process SDK.
 *
 * `claude-code` split by INVOCATION MECHANISM (DESIGN §4.4); the CLI-driven sibling is
 * `@declarative-ai/agents-cli`. This package also owns the normalized `AgentQuery` SEAM both share, so
 * the two adapters differ only in how the agent is reached — and therefore in how its safety policy is
 * enforced (`callback` here, `config` there).
 */
export * from "./seam";
export * from "./runtime";
export * from "./sdkQuery";
