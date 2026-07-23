/**
 * Value-or-failure, and the one place it is declared.
 *
 * Every layer returns a result; none of them should re-derive the shape. Before this there were three
 * near-identical records — exec's `Outcome<T>`, ops' `Result<O>`, and llm's `CallOutcome<T>` — each with
 * its own failure type (`CallFailure` was field-for-field identical to `ExecFailure`) and its own metrics
 * record. They differed in what they CARRIED, never in what they MEANT, so the shape lives here and each
 * layer supplies the payload:
 *
 *   exec  ExecResult<O>      = ResultWithMetrics<O, Failure, ExecMetrics>
 *   llm   LlmCallResult<T>   = ResultWithMetrics<LlmOutput<T>, Failure, LlmMetrics>
 *   ops   FunctionResult<O,M>= Result<O, Failure> & { metrics?: M }
 *
 * NO DEFAULTS on `S`, `E`, or `M`. A result's value type is the caller's business, its failure type is
 * the layer's, and its metrics type belongs to whatever produced the measurement — a default on any of
 * them would silently pick one layer's answer for another layer's question. The generic parameter is
 * also what keeps this package free of every vocabulary above it: `S` is pinned at each instantiation,
 * so nothing here ever names an operation, an execution, or a model (§2.2 — a hole is a type parameter,
 * never `unknown`).
 */

/**
 * Value-or-failure. The SUCCESS branch guarantees a value; the FAILURE branch may still carry one — a
 * partial is what makes a failure diagnosable rather than empty (a truncated generation, output that
 * arrived but would not parse).
 *
 * The success branch has NO `error` key, rather than `error?: undefined`. That difference is the whole
 * ergonomics of the type: `error` is not a property of the union, so `if (r.error)` does not compile at
 * all, instead of compiling and silently widening `value` to `S | undefined` the way an optional
 * `error?: undefined` did. A wrong check is now a build failure; the right checks are `isOk(r)` and
 * `"error" in r`, both of which narrow.
 */
export type Result<S, E> = { value: S } | { error: E; value?: S };

/**
 * A result that also REPORTS what the work cost. `M` is a parameter rather than a fixed record because
 * metrics belong to the thing being measured: an execution reports timing and counts, a model call
 * reports tokens and money, and a pure function reports nothing at all. The layer that consumes a
 * measurement constrains `M` to exactly the fields it reads (`M extends ExecMetrics`,
 * `M extends BudgetMetrics`) and stays ignorant of the rest.
 */
export type ResultWithMetrics<S, E, M> = Result<S, E> & { metrics: M };

/**
 * Narrow a {@link Result} to its success branch.
 *
 * Discriminates on `error`, not on the presence of `value`: the failure branch is allowed to carry a
 * partial value, so "has a value" and "succeeded" are different questions.
 */
export function isOk<S, E>(r: Result<S, E>): r is { value: S } {
  // Tolerates an explicit `error: undefined` key as well as its absence, so a result that survived a
  // spread or a round-trip is still read correctly rather than reported as a failure with no failure.
  return !("error" in r) || (r as { error?: E }).error === undefined;
}
