/**
 * The classified failure vocabulary shared by every layer.
 *
 * One declaration at the bottom of the graph, so an llm call's failure, an execution's failure, and a
 * stored record's failure are the SAME value — which is what lets the retry loop and the AIMD controller
 * read a classification off any of them without re-deriving one from prose.
 *
 * This file used to carry `Metrics`, `TokenCounts`, `ReasoningSegment`, `ToolCall`, and `ToolResult` as
 * well. None of them belong to a package about JSON: metrics belong to whatever produced the
 * measurement (ops' floor, exec's timing, llm's tokens), and the reasoning/tool trace is model
 * vocabulary that now lives in `@declarative-ai/llm`. They were here only because the old `Outcome`
 * named them all at once.
 */
import type { ErrorClass } from "./classification";

/**
 * A classified failure. `reason` is the REAL underlying cause, human-readable — never a bookkeeping
 * message like "retries exhausted" (see `describeError`).
 *
 * Was `ExecFailure`. The `Exec` prefix was wrong twice over: it lives in `json`, not `exec`, and it is
 * not execution-specific — a provider call and a stored record classify failures the same way. A layer
 * prefix names a layer's CUSTOMIZATION of a base type; this is the base.
 */
export interface Failure {
  classification: ErrorClass;
  /** The real underlying cause, human-readable. */
  reason: string;
  /** Server-advised wait before the next attempt (`retry-after`), ms. */
  retryAfterMs?: number;
  /** True iff this was a 429 rate-limit — feeds AIMD's multiplicative decrease. */
  rateLimited?: boolean;
}
