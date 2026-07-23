/**
 * The retry ARITHMETIC, extracted from findmyprompt `src/engine/execution/retry.ts`. The retry LOOP is
 * `withRetry` in `./wrappers` — the one that runs in the shipped stack — and this module deliberately
 * carries no second one.
 *
 * There WAS a second one (`retryLoop` + `RetryOptions`), exported from the package index and referenced
 * only by its own test: `withRetry` never called it, so its budget gate and its two short-circuits
 * (`budget-exhausted`, the deadline floor) protected nothing while the shipped wrapper happily slept a
 * full backoff to re-attempt a window it could already have known was closed. Both are now conditions in
 * `withRetry` itself (`isFutile`, `RetryConfig.budget`), and the duplicate is gone. Two implementations
 * of one policy is one implementation plus a decoy.
 *
 * Discipline (unchanged from findmyprompt): only `network-retriable` failures are auto-retried — an
 * `api-retriable` output failure must never be SILENTLY re-rolled (that re-rolls a stochastic output
 * until it passes and biases scores); callers opt in explicitly, via `RetryConfig.validation`.
 */

/** The retry budget gate: every attempt spends real budget, so before each retry we
 *  confirm budget remains. `allowMore` returns false once the cap is (about to be) hit. */
export interface RetryBudget {
  allowMore(): boolean;
}

export const DEFAULT_BASE_BACKOFF_MS = 500;
export const DEFAULT_MAX_BACKOFF_MS = 60_000;

/**
 * How long to wait before the next attempt. A server `retry-after` wins (clamped to
 * `maxBackoffMs`, with a touch of jitter so a fleet doesn't wake in lockstep); otherwise
 * FULL-JITTER exponential backoff: a uniform draw in `[0, min(cap, base·2^attempt)]`.
 * Pure given `random` — directly unit-testable.
 */
export function backoffDelayMs(
  attempt: number,
  retryAfterMs: number | undefined,
  opts: { baseBackoffMs: number; maxBackoffMs: number },
  random: () => number,
): number {
  if (retryAfterMs !== undefined && Number.isFinite(retryAfterMs) && retryAfterMs > 0) {
    const jitter = random() * Math.min(opts.baseBackoffMs, retryAfterMs);
    return Math.min(retryAfterMs + jitter, opts.maxBackoffMs);
  }
  const ceiling = Math.min(opts.maxBackoffMs, opts.baseBackoffMs * 2 ** attempt);
  return Math.floor(random() * ceiling);
}
