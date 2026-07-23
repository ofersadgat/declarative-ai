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
export * from "./contract";
export * from "./handles";
export * from "./operationExecutor";
export * from "./memo";
export * from "./wrappers";
export * from "./concurrency";
export * from "./deadline";
export * from "./retry";
