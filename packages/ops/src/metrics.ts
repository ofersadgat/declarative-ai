/**
 * The metrics FLOOR — what every measurement has, and nothing more.
 *
 * Metrics are not one record. They belong to whatever produced the measurement: an execution reports
 * timing and counts (`ExecMetrics`, in `exec`), a model call reports tokens and money (`LlmMetrics`, in
 * `llm`), a budget-aware layer reports spend (`BudgetMetrics`). Each of those is a flat record that
 * SATISFIES this floor structurally; none of them extends another's fields by coincidence.
 *
 * This is the floor rather than the union because the only thing every measurement genuinely shares is
 * how long it took. A consumer constrains `M` to exactly the fields it reads — `M extends Metrics` for
 * timing, `M extends BudgetMetrics` for money — and stays ignorant of everything else the producer
 * chose to report.
 */

/** What every measurement carries. */
export interface Metrics {
  /** Wall-clock duration of the measured work, ms. Required on a completed measurement. */
  durationMs: number;
  /** When the work started (ms epoch), when the producer knows it. */
  startMs?: number;
}

/** In-flight metrics: the row exists before the work completes, so nothing is required yet. */
export type PartialMetrics = Partial<Metrics>;

/**
 * Combine two measurements of the same kind — two attempts of a retried call, or a child folded into
 * its parent. Registered by whatever PRODUCES `M`, because only it knows which fields sum, which take
 * the latest, and which are the first observation.
 *
 * The consumer of a merge never learns what the fields mean: `exec` aggregates retry attempts through
 * this without knowing whether `M` counts tokens or dollars.
 */
export interface MetricsAlgebra<M> {
  merge(a: M, b: M): M;
}

/** Sum two optional numbers, preserving "neither was reported" as `undefined`. */
export function sumOptional(a: number | undefined, b: number | undefined): number | undefined {
  return a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0);
}

/** The floor's own merge: duration sums, the start time is the FIRST observation. A richer `M` builds
 *  its algebra on top of this rather than restating it. */
export function mergeMetrics(a: Metrics, b: Metrics): Metrics {
  const startMs = a.startMs ?? b.startMs;
  return { durationMs: a.durationMs + b.durationMs, ...(startMs !== undefined ? { startMs } : {}) };
}
