/**
 * The ONLY thing in `exec` that knows about tokens.
 *
 * Same quarantine as `budget.ts`, for the same reason: a token estimate is provider vocabulary, and an
 * executor that runs a shell command or a sub-workflow has no tokens to estimate. The limiter SEAM
 * lives here so `exec`'s AIMD controller can be typed by it; deriving an estimate from a prompt is
 * `promptop`'s job, one layer up.
 */

/** One call's estimated token footprint, input/output split — what rate pre-admission is priced on. */
export interface CallTokenEstimate {
  inputTokens: number;
  outputTokens: number;
}

export interface CallEstimate extends CallTokenEstimate {
  modelId?: string;
}

export interface RateLimiter {
  /** Admit one call: take a concurrency slot, wait for rate headroom, run it. */
  schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T>;
  /** Feed the call's outcome back (a 429 halves concurrency; success grows it). */
  reportOutcome(outcome: { rateLimited?: boolean; modelId?: string }): void;
}
