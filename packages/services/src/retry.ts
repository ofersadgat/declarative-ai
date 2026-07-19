/**
 * Generic budget-gated retry, extracted from findmyprompt `src/engine/execution/retry.ts`.
 * The op-model-specific drivers (`runWithRetry`/`runWithRetries` over findmyprompt's
 * content-addressed operations) stay in findmyprompt; this module carries the reusable
 * pieces: the backoff arithmetic and a generic outcome-driven retry loop.
 *
 * Discipline (unchanged from findmyprompt): only `network-retriable` failures are
 * auto-retried — an `api-retriable` output failure must never be silently re-rolled
 * (that re-rolls a stochastic output until it passes and biases scores); callers opt in
 * to output re-rolls explicitly (`retryApiRetriable`).
 */
import type { ExecFailure } from "@declarative-ai/core";
import { isDeadlineFloor } from "./deadline";

/** The retry budget gate: every attempt spends real budget, so before each retry we
 *  confirm budget remains. `allowMore` returns false once the cap is (about to be) hit. */
export interface RetryBudget {
  allowMore(): boolean;
}

export const DEFAULT_BASE_BACKOFF_MS = 500;
export const DEFAULT_MAX_BACKOFF_MS = 60_000;

const realWait = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

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

export interface RetryOptions {
  /** Max retries beyond the first attempt. Default 5. */
  retryCap?: number;
  budget?: RetryBudget;
  /** Also retry `api-retriable` failures (explicit output re-roll opt-in). Default false. */
  retryApiRetriable?: boolean;
  /** Base for full-jitter exponential backoff when no `retry-after` is present. Default 500ms. */
  baseBackoffMs?: number;
  /** Hard cap on any single backoff wait (also clamps an over-long `retry-after`). Default 60s. */
  maxBackoffMs?: number;
  /** Injectable sleep — tests pass a no-op to avoid real timers. */
  waitMs?: (ms: number) => Promise<void>;
  /** Injectable RNG for jitter (default `Math.random`). */
  random?: () => number;
}

/**
 * Drive `attempt(i)` until it succeeds (no failure), fails permanently, or exhausts the
 * cap/budget. The attempt index is passed through so callers can diversify draw scopes
 * per attempt (fresh memo keys). Special reasons short-circuit immediately:
 * budget-exhausted (a re-roll cannot succeed this window) and the deadline floor
 * (every retry would re-hit the floor guard and fail instantly).
 */
export async function retryLoop<T extends { failure?: ExecFailure }>(
  attempt: (attemptIndex: number) => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const cap = options.retryCap ?? 5;
  const baseBackoffMs = options.baseBackoffMs ?? DEFAULT_BASE_BACKOFF_MS;
  const maxBackoffMs = options.maxBackoffMs ?? DEFAULT_MAX_BACKOFF_MS;
  const waitMs = options.waitMs ?? realWait;
  const random = options.random ?? Math.random;
  let last: T | undefined;

  for (let i = 0; i <= cap; i++) {
    const res = await attempt(i);
    last = res;
    if (!res.failure) return res; // success
    const cls = res.failure.classification;
    // out-of-credits / deadline / canceled / permanent are all non-retriable classes and
    // fall out here; the reason-prefix checks below catch the same conditions when they
    // arrive classified as network-retriable (findmyprompt's historical encoding).
    const retriable = cls === "network-retriable" || (options.retryApiRetriable && cls === "api-retriable");
    if (!retriable) return res;
    if (res.failure.reason.startsWith("budget-exhausted")) return res;
    if (isDeadlineFloor(res.failure.reason)) return res;
    if (i >= cap) return res;
    if (options.budget && !options.budget.allowMore()) return res;
    const delay = backoffDelayMs(i, res.failure.retryAfterMs, { baseBackoffMs, maxBackoffMs }, random);
    if (delay > 0) await waitMs(delay);
  }
  return last as T;
}
