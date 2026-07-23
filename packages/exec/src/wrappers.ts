/**
 * The GENERIC executor wrappers — the ones whose behavior is a property of execution itself rather than
 * of any particular op kind. Because the seam is now `start(op, ctx)` (DESIGN §3.2), they
 * apply uniformly to prompt and function ops alike; previously they could not reach anything dispatched
 * through the function registry.
 *
 * The llm-AWARE wrappers — rate limiting (needs a token estimate off a prompt), budget (needs model
 * pricing), session (needs transcript folding) — live in `@declarative-ai/promptop`, one layer up.
 *
 * Deps model: SERVICES and per-deployment POLICY are captured at CONSTRUCTION — explicit and
 * composable. Each wrapper CONSUMES its own trigger (`withDeadline` strips `ctx.deadline`), so a
 * mis-composed stack fails loudly on first use instead of silently degrading.
 */
import type { InlineFamily, Operation, PromptOp, ResolvedValue } from "@declarative-ai/ops";
import { isOk } from "@declarative-ai/ops";
import type { DeadlineConfig, ExecHandle, ExecMetrics, Executor, ExecutorWrapper, ExecServices, ExecResult } from "./contract";
import { forwardCapabilitiesFor } from "./contract";
import { DEADLINE_FLOOR_REASON, deadlineDecision, isDeadlineFloor, systemClock } from "./deadline";
import { abortableDelay, canceledFailure, finishedHandle, permanentFailure, raceWork, withMetrics, wrapHandle } from "./handles";
import type { RetryBudget } from "./retry";
import { DEFAULT_BASE_BACKOFF_MS, DEFAULT_MAX_BACKOFF_MS, backoffDelayMs } from "./retry";

/** True iff a value is an `Executor` (has `.start`) — disambiguates a wrapper's optional trailing
 *  `inner` from its optional config argument (both structurally disjoint from `Executor`). */
export function isExecutor(x: unknown): x is Executor {
  return typeof x === "object" && x !== null && typeof (x as Executor).start === "function";
}

/** Dual-mode dispatch: a wrapper called WITHOUT an inner executor returns the curried
 *  `ExecutorWrapper` (for the `compose(...).with(...)` builder); called WITH one it applies
 *  immediately, so direct nesting reads inside-out. */
export function curryOrApply<RIn, ROut, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>, Out = ResolvedValue>(
  wrap: ExecutorWrapper<RIn, ROut, M, Op, Out>,
  inner?: Executor<RIn, M, Op, Out>,
): ExecutorWrapper<RIn, ROut, M, Op, Out> | Executor<ROut, M, Op, Out> {
  return inner ? wrap(inner) : wrap;
}

/** The ctx SEAMS `withDeadline` consumes — its `config` mirrors these. A seam PROVIDED at construction
 *  drops out of what `.start` requires; an omitted one stays required (`Omit`-tracked). */
type DeadlineSeams = { deadline: DeadlineConfig; stepStartMs: number };

/**
 * Deadline fail-fast + timeout clamp (§time-vs-money). Its `config` mirrors the ctx seams it reads —
 * `{ deadline, stepStartMs }` — and whatever you supply at CONSTRUCTION drops out of what `.start`
 * requires (the rest is read from ctx): so `withDeadline({ deadline })` needs only `stepStartMs` at
 * start, `withDeadline({ deadline, stepStartMs })` needs neither, and `withDeadline()` needs both.
 *
 * Below the start floor it short-circuits with a `deadline` failure and NEVER starts the inner call;
 * otherwise it lowers `ctx.timeoutMs` to the remaining window AND ENFORCES that window on the call in
 * flight — when it expires the inner handle is canceled and the result is a `deadline` failure. The
 * clamp alone was only a floor on STARTING: `ctx.timeoutMs` is a number this package writes and nothing
 * reads, so an executor that ignored it (every one that does not thread it into an `AbortSignal`) ran
 * unbounded past the very window this wrapper exists to keep it inside — which is the serverless
 * hard-kill it is supposed to salvage before. With the per-call budget on {@link ExecServices} rather
 * than on a spec AND on a definition, there is no longer a "definition budget above the spec limit"
 * conflict for the core to refuse — one field, one clamp.
 */
export function withDeadline<R = ExecServices, Out = ResolvedValue>(inner: Executor<R, ExecMetrics, Operation<InlineFamily>, Out>): Executor<R & DeadlineSeams, ExecMetrics, Operation<InlineFamily>, Out>;
export function withDeadline<R = ExecServices, P extends Partial<DeadlineSeams> = {}, Out = ResolvedValue>(
  config?: P,
): ExecutorWrapper<R, R & Omit<DeadlineSeams, keyof P>, ExecMetrics, Operation<InlineFamily>, Out>;
export function withDeadline<R = ExecServices, P extends Partial<DeadlineSeams> = {}, Out = ResolvedValue>(
  config: P,
  inner: Executor<R, ExecMetrics, Operation<InlineFamily>, Out>,
): Executor<R & Omit<DeadlineSeams, keyof P>, ExecMetrics, Operation<InlineFamily>, Out>;
export function withDeadline<R = ExecServices>(
  configOrInner?: Partial<DeadlineSeams> | Executor<R>,
  maybeInner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<DeadlineSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R> | undefined;
  // Body typed against `ExecServices` (unchanged shape); the generic signature ADDS the seam(s) the
  // config does NOT provide to what `start` requires — the cast bridges the two.
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      const { deadline: ctxDeadline, stepStartMs: ctxStep, ...restCtx } = ctx;
      const deadline = config?.deadline ?? ctxDeadline; // construction config wins; else read from ctx
      const stepStartMs = config?.stepStartMs ?? ctxStep;
      if (!deadline) return innerExec.start(op, restCtx);
      if (stepStartMs === undefined) {
        return finishedHandle(
          permanentFailure("the deadline is set without a stepStartMs — deadline arithmetic needs the step-start origin"),
        );
      }
      const clock = ctx.clock ?? systemClock;
      const now = clock.now();
      const decision = deadlineDecision(stepStartMs, deadline, now);
      if (!decision.proceed) {
        return finishedHandle({
          error: {
            classification: "deadline",
            reason: `${DEADLINE_FLOOR_REASON}: ${decision.remainingMs}ms remaining is below the start floor`,
          },
          metrics: { startMs: now, durationMs: 0 },
        });
      }
      const ceiling = Math.min(restCtx.timeoutMs ?? Number.POSITIVE_INFINITY, decision.remainingMs);
      return wrapHandle(
        async (ctl): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
          if (ctl.canceled()) return canceledFailure("canceled before the call started");
          const handle = ctl.started(innerExec.start(op, { ...restCtx, timeoutMs: ceiling }));
          // The window is enforced HERE, not delegated: `timeoutMs` is advisory to an inner executor
          // that may not read it at all. `raceWork` clears the timer on the branch that did not win, so
          // a call finishing in 5ms does not strand a 290s timer holding the event loop. The enforcement
          // timer runs on the SAME clock the start decision used (`clock.wait`), so a virtual clock is not
          // measured against wall-clock `setTimeout` ms; absent a `wait` seam, `raceWork` uses a real timer.
          const raced = await raceWork(handle.result, ceiling, ctl.signal, clock.wait?.bind(clock));
          if (raced.status === "done") return raced.value;
          if (raced.status === "canceled") return canceledFailure("canceled while the call was in flight");
          void handle.cancel(); // stop the overrunning call; do not wait for it to agree
          // Carries the SAME `deadline-floor` marker as the start refusal: both mean "this window ran
          // out of TIME" (yield to the next window), as opposed to running out of money — which is the
          // distinction the marker exists to keep from drifting (§time-vs-money).
          return {
            error: {
              classification: "deadline",
              reason: `${DEADLINE_FLOOR_REASON}: the call exceeded its ${ceiling}ms window and was cut off in flight`,
            },
            metrics: { startMs: now, durationMs: clock.now() - now },
          };
        },
        { signal: ctx.abortSignal, canceledReason: "canceled while the call was in flight" },
      );
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/**
 * The unified opt-in re-attempt policy. One concept, two independent axes:
 *
 * - `transient` — re-attempt a `network-retriable` failure (rate limit / 5xx) with full-jitter
 *   exponential backoff, up to this many EXTRA attempts (a number is the cap; the object form tunes
 *   backoff and injects a test sleep). Re-sends the same op.
 * - `validation` — re-attempt a schema-VALIDATION failure (`api-retriable`) up to `turns` extra
 *   attempts. `feedback: true` appends the concrete validation errors to the prompt before retrying (a
 *   targeted fix — the former `withRepair`); `feedback: false` is a blind re-roll. Off by default,
 *   because a silent re-roll biases stochastic output until it passes.
 *
 * Metrics accumulate across attempts. Both axes compose: after a feedback repair, a subsequent
 * transient failure re-sends the ALREADY-augmented op. A non-retriable failure (or success) stops
 * immediately. The augmented OP itself carries the repair hint, so an inner memoize (keyed on the op
 * hash) keys on exactly what is sent — no separate hash to keep in sync.
 *
 * Now that a function impl resolves a CLASSIFIED failure (§4.2), this reaches function ops too: a 429
 * raised inside a registered async function is retried here instead of being permanently failed.
 */
export interface RetryConfig {
  transient?:
    | number
    | { cap: number; baseBackoffMs?: number; maxBackoffMs?: number; waitMs?: (ms: number) => Promise<void>; random?: () => number };
  validation?: { turns: number; feedback?: boolean };
  /** Every attempt spends real budget, so before each re-attempt the gate is asked whether any remains.
   *  Absent ⇒ ungated (an absent service is a no-op, like the rest of the stack). */
  budget?: RetryBudget;
}

/** The repair suffix appended to a prompt op's `user` text after a rejected output. */
function repairSuffix(errors: string): string {
  return `\n\nYour previous output was rejected: ${errors}. Return ONLY corrected JSON matching the schema.`;
}

/**
 * An output failure a fresh attempt can fix — which is EXACTLY what `api-retriable` means (json's
 * `ErrorClass`: "the API RESPONDED but the result is unusable in a way a FRESH stochastic run can fix:
 * schema-validation reject, truncation, unparseable/empty output").
 *
 * The check is the CLASSIFICATION, not `/validation/i` over the free-text `reason`. Matching prose was
 * wrong in both directions: it missed the truncated and unparseable failures llm classifies
 * `api-retriable` with no such word in them (the `validation` axis silently did nothing for them), and
 * it fired on any unrelated failure whose reason happened to contain the substring. `reason` is a
 * human-readable diagnostic — the classification is the machine-readable channel, and reading the wrong
 * one is the mistake §4.2 removed everywhere else.
 */
function isRepairableOutput(result: ExecResult<ResolvedValue>): boolean {
  return !isOk(result) && result.error.classification === "api-retriable";
}

/** A failure a re-attempt provably cannot fix within this window, whatever its classification says.
 *  Folded in from the deleted second retry implementation: sleeping a full backoff to re-hit a closed
 *  window (or an empty wallet) burns the little time that window had left. */
function isFutile(result: ExecResult<ResolvedValue>): boolean {
  if (isOk(result)) return false;
  return result.error.reason.startsWith("budget-exhausted") || isDeadlineFloor(result.error.reason);
}

/** Append the concrete errors to a prompt op's instruction. A function op has no prompt to amend, so
 *  feedback degrades to a plain re-attempt rather than pretending to repair something. */
function withRepairHint(op: Operation<InlineFamily>, errors: string): Operation<InlineFamily> {
  if (op.kind !== "prompt") return op;
  const prompt: PromptOp<InlineFamily> = { ...op, user: `${op.user}${repairSuffix(errors)}` };
  return prompt;
}

export function withRetry<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(config: RetryConfig): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
export function withRetry<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: RetryConfig,
  inner: Executor<R, M, Operation<InlineFamily>, Out>,
): Executor<R, M, Operation<InlineFamily>, Out>;
export function withRetry<R = ExecServices>(config: RetryConfig, inner?: Executor<R>): ExecutorWrapper<R, R> | Executor<R> {
  const transientCap = typeof config.transient === "number" ? config.transient : (config.transient?.cap ?? 0);
  const b = typeof config.transient === "object" ? config.transient : undefined;
  const baseBackoffMs = b?.baseBackoffMs ?? DEFAULT_BASE_BACKOFF_MS;
  const maxBackoffMs = b?.maxBackoffMs ?? DEFAULT_MAX_BACKOFF_MS;
  const injectedWait = b?.waitMs;
  /** The backoff wait, ALWAYS raced against cancellation. The built-in waiter clears its own timer when
   *  cancel wins ({@link abortableDelay}); an INJECTED `waitMs` is a caller's promise we cannot cancel,
   *  so it is raced instead — the loop stops waiting on it even though it runs to completion. */
  const sleep = async (ms: number, signal: AbortSignal): Promise<void> => {
    if (!injectedWait) return abortableDelay(ms, signal);
    await raceWork(injectedWait(ms), undefined, signal);
  };
  const random = b?.random ?? Math.random;
  const validationTurns = config.validation?.turns ?? 0;
  const feedback = config.validation?.feedback ?? false;

  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      return wrapHandle(
        async (ctl): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
          let accumulated: ExecMetrics | undefined;
          let last: ExecResult<ResolvedValue> | undefined;
          let currentOp = op;
          let transientUsed = 0;
          let validationUsed = 0;
          while (!ctl.canceled()) {
            last = await ctl.started(innerExec.start(currentOp, ctx)).result;
            // Aggregate through the algebra the PRODUCER registered: exec never learns which of `M`'s
            // fields sum, which take the latest, or which are the first observation.
            accumulated = accumulated === undefined ? last.metrics : innerExec.metrics.merge(accumulated, last.metrics);
            if (ctl.canceled() || isOk(last)) break;
            // A closed window or an empty wallet cannot be re-attempted into success, whatever class the
            // failure arrived under; stop before spending the backoff finding that out.
            if (isFutile(last)) break;
            const retriable =
              (last.error.classification === "network-retriable" && transientUsed < transientCap) ||
              (isRepairableOutput(last) && validationUsed < validationTurns);
            if (retriable && config.budget && !config.budget.allowMore()) break;
            if (last.error.classification === "network-retriable" && transientUsed < transientCap) {
              const delay = backoffDelayMs(transientUsed, last.error.retryAfterMs, { baseBackoffMs, maxBackoffMs }, random);
              transientUsed++;
              // RACED against cancellation, and the timer cleared when cancel wins. A server
              // `retry-after: 60` clamped to `maxBackoffMs` is a full minute of `setTimeout`: unraced, a
              // cancel arriving here could not settle for that minute and `cancel()` itself blocked for
              // it, while the un-`unref`'d timer held the event loop open behind both.
              if (delay > 0) await sleep(delay, ctl.signal);
              if (ctl.canceled()) break;
              continue; // re-send `currentOp` (possibly already repair-augmented)
            }
            if (isRepairableOutput(last) && validationUsed < validationTurns) {
              validationUsed++;
              // Hint onto `currentOp`, NOT `op`: the turns ACCUMULATE, so a third attempt carries both
              // the first turn's errors and the second's. Re-basing on the original op each time threw
              // away every earlier hint, leaving the model to rediscover a mistake it had already been
              // told about.
              if (feedback) currentOp = withRepairHint(currentOp, last.error.reason);
              continue;
            }
            break;
          }
          if (!last) return canceledFailure("canceled before the call started");
          return withMetrics<ResolvedValue, ExecMetrics, ExecMetrics>(last, accumulated!);
        },
        { signal: ctx.abortSignal },
      );
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}
