/**
 * @declarative-ai/promptop — running a `PromptOp` through the `Executor` seam.
 *
 * Decision #7: the op SHAPE stays in `ops` (text slots + a config slot + a schema, importing nothing
 * from llm); the LOWERING to an `LlmCallDefinition` is llm-specific and lives here, together with the
 * `Executor` that applies it and the three wrappers that need llm knowledge (token estimates, model
 * pricing, transcript folding).
 *
 * Conceptually the layering is `llm ← promptop ← exec`; in dependency terms `promptop` depends on
 * `exec`, because the prompt executor IMPLEMENTS the interface `exec` defines. Both readings hold —
 * `exec` owns the generic machinery, `promptop` owns the LLM-specific implementation, and nothing in
 * `exec` knows `PromptOp` exists.
 */
export * from "./lowering";
export * from "./executor";
export * from "./wrappers";
