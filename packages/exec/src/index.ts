/**
 * @declarative-ai/exec — the generic execution machinery (DESIGN §2, §3.1/§3.2).
 *
 * ONE seam: `Executor.start(op, ctx)`. This package owns the machinery — handles, outcomes, the
 * augmentable `ExecServices` bundle, composition, memoization, rate limiting, deadlines, retry,
 * sessions — and knows nothing about LLMs, validation, permissions, or filesystems. Those declare
 * their own seams by augmenting `ExecServices` (DESIGN §3.2).
 *
 * It has no dependencies outside the workspace.
 */
export * from "./contract.js";
export * from "./handles.js";
export * from "./operationExecutor.js";
export * from "./resolvedOperation.js";
export * from "./memo.js";
export * from "./hydrate.js";
export * from "./wrappers.js";
export * from "./concurrency.js";
export * from "./deadline.js";
export * from "./retry.js";
// Recording what ran (DESIGN.md §1.6) — the two-phase write, split out of the memo store where
// findmyprompt fused it. A session IS the records sharing a `session.id`.
export * from "./record.js";
