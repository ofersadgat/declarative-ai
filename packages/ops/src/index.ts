/**
 * @declarative-ai/ops — the typed operation spine (DESIGN §3.1; package graph §2).
 *
 * The op model and function registry imported from findmyprompt, made generic over the reference
 * substrate (ref families), plus the `Signature` ⇄ schema bridge. The JSON vocabulary it is typed by
 * comes from `@declarative-ai/json` and is re-exported here, so a consumer that only speaks ops needs
 * one import. Execution machinery is deliberately absent — ai-exec executes ops through
 * `@declarative-ai/exec`'s `Executor`; findmyprompt keeps its own `runOperation`.
 *
 * Its only non-workspace dependency is `json-schema-to-ts`, and that is TYPES ONLY.
 */
export * from "@declarative-ai/json";
export * from "./model.js";
export * from "./metrics.js";
export * from "./registry.js";
export * from "./metadata.js";
export * from "./signatureSchema.js";
export * from "./typed.js";
