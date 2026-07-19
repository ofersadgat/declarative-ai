/**
 * Deadline cutoff, extracted from findmyprompt `src/engine/execution/deadline.ts`
 * (Clock/DeadlineConfig types now live in @declarative-ai/core). A serverless step has no
 * per-step timeout API, so the step computes its own deadline from a captured
 * `stepStartMs` and a configured `maxDurationMs`; the result drives
 * `AbortSignal.timeout(remainingMs)` so a long generation is cut off and salvaged
 * before the platform hard-kills the function.
 *
 * Pure + clock-injectable so the floor-guard and margin arithmetic are deterministically
 * testable without real timers.
 */
import type { Clock, DeadlineConfig } from "@declarative-ai/core";

export type { Clock, DeadlineConfig };

export const systemClock: Clock = { now: () => Date.now() };

export const DEFAULT_SAFETY_MARGIN_MS = 10_000;
export const DEFAULT_FLOOR_MS = 5_000;

/**
 * §time-vs-money — the marker for "this WINDOW ran out of time to make the call", as opposed to
 * running out of MONEY (`out-of-credits`). The two demand OPPOSITE handling:
 *  - out of MONEY → the run cannot continue at all until someone tops up ⇒ HALT (gracefully, resumably).
 *  - out of TIME  → the run is fine, this WINDOW just ended ⇒ YIELD and let the next window run it.
 * Keeping the marker in ONE place is what stops the two floors drifting back together.
 */
export const DEADLINE_FLOOR_REASON = "deadline-floor";

/** True iff a failure reason is the window-deadline floor (see `DEADLINE_FLOOR_REASON`). */
export function isDeadlineFloor(reason: string): boolean {
  return reason.startsWith(DEADLINE_FLOOR_REASON);
}

export interface DeadlineDecision {
  /** Whether enough time remains to start the call. */
  proceed: boolean;
  /** Time budget for the call (what `AbortSignal.timeout` gets). */
  remainingMs: number;
}

/** `remainingMs = stepStartMs + maxDurationMs − margin − now`, measured from the fixed origin. */
export function computeRemainingMs(stepStartMs: number, cfg: DeadlineConfig, now: number): number {
  const margin = cfg.safetyMarginMs ?? DEFAULT_SAFETY_MARGIN_MS;
  return stepStartMs + cfg.maxDurationMs - margin - now;
}

export function deadlineDecision(stepStartMs: number, cfg: DeadlineConfig, now: number): DeadlineDecision {
  const remainingMs = computeRemainingMs(stepStartMs, cfg, now);
  const floor = cfg.floorMs ?? DEFAULT_FLOOR_MS;
  return { proceed: remainingMs >= floor, remainingMs };
}
