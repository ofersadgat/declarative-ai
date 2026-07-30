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
import { ERROR_CLASSES, type ErrorClass } from "./classification";

/**
 * A classified failure. `reason` is the REAL underlying cause, human-readable — never a bookkeeping
 * message like "retries exhausted" (see `describeError`).
 *
 * Was `ExecFailure`. The `Exec` prefix was wrong twice over: it lives in `json`, not `exec`, and it is
 * not execution-specific — a provider call and a stored record classify failures the same way. A layer
 * prefix names a layer's CUSTOMIZATION of a base type; this is the base.
 */
export interface Failure<D = never> {
  classification: ErrorClass;
  /** The real underlying cause, human-readable. */
  reason: string;
  /** Server-advised wait before the next attempt (`retry-after`), ms. */
  retryAfterMs?: number;
  /** True iff this was a 429 rate-limit — feeds AIMD's multiplicative decrease. */
  rateLimited?: boolean;
  /**
   * Domain-specific detail, for a layer with more to say than the shared fields carry.
   *
   * Generic in the DETAIL only, and `classification` stays invariant on purpose. This type exists so
   * that an llm call's failure, an execution's failure and a stored record's failure are the SAME
   * value — a parameter able to vary the classification would break `classifyError`, the retry loop
   * and the AIMD controller at once. The parameter is defaulted, so every existing unparameterized
   * `Failure` keeps meaning exactly what it meant.
   */
  detail?: D;
}

/**
 * A failure AS DATA — the JSON Schema a slot declares in order to accept one.
 *
 * A failure already travels as data through the RESULT envelope; this is what lets it travel through
 * the DATA plane, so a workflow can bind one to a slot and branch on it instead of terminating
 * (JaiRA EXPRESSIONS.md §5).
 *
 * The `classification` enum is the point. Because the set is closed, an expression comparing against
 * it can be checked exhaustively — `error.classification === 'policy-denide'` becomes a lint error
 * rather than a comparison that is quietly always false, which is the property `run.cursor` already
 * has. `detail` is deliberately unconstrained: the shared shape is the shared fields.
 */
export const FAILURE_SCHEMA = {
  type: "object",
  properties: {
    classification: { type: "string", enum: [...ERROR_CLASSES] },
    reason: { type: "string" },
    retryAfterMs: { type: "number" },
    rateLimited: { type: "boolean" },
    detail: {},
  },
  required: ["classification", "reason"],
} as const;

/**
 * How a failure appears AS A VALUE: wrapped, `{ error: <failure> }`.
 *
 * Wrapped rather than bare, for two reasons that are easy to miss until they bite:
 *
 *  - **A bare failure is not distinguishable from data that looks like one.** A classifier operation
 *    returning `{ classification, reason }` is an entirely plausible thing for a workflow to have,
 *    and sniffing those fields would read its output as an error. The `error` key is a discriminator
 *    nothing else in the data plane claims.
 *  - **It makes the union writable.** A slot says "a value, or an error" as
 *    `{ anyOf: [ <the value>, <this> ] }` — and a slot that says nothing simply has no branch
 *    requiring `error`, so "unconstrained does not accept a failure" stops being a special case and
 *    becomes the ordinary reading of what the author wrote.
 *
 * It is also the shape a guard reads: `outputs.result.error.classification === 'policy-denied'`.
 */
export const ERROR_VALUE_SCHEMA = {
  type: "object",
  properties: { error: FAILURE_SCHEMA },
  required: ["error"],
} as const;
