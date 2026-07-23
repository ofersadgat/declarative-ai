/**
 * Handle scaffolding shared by every executor and wrapper. Kept in one place because the never-throws
 * contract is easy to break by accident: a wrapper body that rejects, a cancel that lands in the window
 * BEFORE the inner call starts, an inner event stream that gets swallowed.
 *
 * It is also where CANCELLATION is made real. `ctx.abortSignal` and `handle.cancel()` are two names for
 * the same event, so they are unified here into one per-handle `AbortController`: the caller's signal is
 * LINKED in, `cancel()` aborts the same controller, and either path settles the handle EXACTLY ONCE
 * with a `canceled` failure without waiting for the in-flight work to notice. Three properties every
 * helper below maintains, because each was a live bug:
 *  - a settle happens exactly once — never twice (the guard), never never (the abort path settles the
 *    handle itself rather than awaiting a body parked on an inner call that ignores cancel);
 *  - every timer is CLEARED on the branch that did not win — a pending 60s backoff holds the event loop
 *    open long after the operation settled;
 *  - every listener added to a caller-owned signal is REMOVED when the handle settles — `ctx.abortSignal`
 *    is typically run-scoped, so a leftover listener is one leak per operation and keeps the whole
 *    settled handle graph reachable.
 */
import type { Failure, ResolvedValue } from "@declarative-ai/ops";
import { isOk } from "@declarative-ai/ops";
import type { ExecEvent, ExecHandle, ExecMetrics, ExecResult } from "./contract";

/** An empty, already-completed event stream (for executors that emit no events). */
export function emptyEvents(): AsyncIterable<ExecEvent> {
  return {
    // eslint-disable-next-line require-yield
    async *[Symbol.asyncIterator]() {
      /* no events */
    },
  };
}

/**
 * A zero-cost failure result of the given class.
 *
 * The payload type is `never`, not a generic `O`: a failure that never ran produced no value, and
 * saying so makes these assignable to EVERY `ExecResult<O>`. Left generic, `O` gets inferred from the
 * contextual return type through the union's optional `value?: S`, which silently widens it to
 * `O | undefined` and then fails to assign back — the failure branch's partial makes payload inference
 * lossy at any generic boundary.
 *
 * It carries `{ durationMs: 0 }` and nothing else — no `rawText: ""`, no `finishReason: "error"`. Those
 * were fabricated LLM fields on a result that may have come from a shell command or a sub-workflow;
 * they went away with `Outcome`.
 */
export function failure(classification: Failure["classification"], reason: string): ExecResult<never> {
  return { error: { classification, reason }, metrics: { durationMs: 0 } };
}

/** A zero-cost permanent failure (wrapper-level refusals and normalized wiring faults). */
export function permanentFailure(reason: string): ExecResult<never> {
  return failure("permanent", reason);
}

/** A zero-cost canceled result (cancel landed before any inner call started). */
export function canceledFailure(reason: string): ExecResult<never> {
  return failure("canceled", reason);
}

/**
 * Replace a result's metrics, preserving WHICH BRANCH of the union it is.
 *
 * `{ ...result, metrics }` does not work: spreading a discriminated union widens it to a bag with both
 * `value?` and `error?` optional, which is no longer a `Result` — the success branch stops guaranteeing
 * a value. Every wrapper that re-reports metrics (retry, budget, memo) needs this, so it lives here
 * rather than being re-derived, subtly differently, in each of them.
 */
export function withMetrics<O, MIn extends ExecMetrics, MOut extends ExecMetrics>(
  result: ExecResult<O, MIn>,
  metrics: MOut,
): ExecResult<O, MOut> {
  // Narrowed with `isOk`, not `result.error === undefined`: the union discriminates on an OPTIONAL
  // `error?: undefined` against a non-literal `Failure`, which is not a unit-type discriminant, so only
  // the type predicate narrows it. Raw property checks silently widen `value` to `O | undefined`.
  if (isOk(result)) return { value: result.value, metrics };
  return { error: result.error, ...(result.value !== undefined ? { value: result.value } : {}), metrics };
}

/** A completed handle wrapping a ready result (for wrappers that short-circuit, e.g. deadline fail-fast). */
export function finishedHandle<O, M extends ExecMetrics = ExecMetrics>(result: ExecResult<O, M>): ExecHandle<O, M> {
  return { events: emptyEvents(), result: Promise.resolve(result), cancel: async () => {} };
}

// --- Cancellation primitives ---------------------------------------------------

/**
 * Forward an external signal's abort into `controller`, returning the UNLINK function.
 *
 * The caller MUST call the returned function when its handle settles. `ctx.abortSignal` is normally
 * RUN-scoped — hw hands the same signal to every operation in a workflow — so a listener left on it
 * after the operation finished is a real leak: one dead closure per op, each keeping its whole settled
 * handle graph reachable for the lifetime of the run.
 */
export function linkAbort(controller: AbortController, signal: AbortSignal | undefined): () => void {
  if (!signal) return (): void => {};
  if (signal.aborted) {
    controller.abort();
    return (): void => {};
  }
  const onAbort = (): void => controller.abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return (): void => signal.removeEventListener("abort", onAbort);
}

/** How a {@link raceWork} settled: the work finished, the time budget expired, or cancel landed. */
export type RaceOutcome<T> = { status: "done"; value: T } | { status: "timeout" } | { status: "canceled" };

/**
 * Race `work` against a `ms` time budget AND a cancellation signal, clearing the timer and removing the
 * listener on EVERY branch.
 *
 * That cleanup is the whole reason this exists rather than a `Promise.race`: race leaves both losers
 * running, so a fast completion strands a multi-second timer that holds the event loop open, and a
 * finished operation strands a listener on the caller's (long-lived) signal. `ms` undefined or
 * non-finite means "no time budget" — the pure cancellation race.
 *
 * Rejections propagate: the work's own error channel is not this function's business.
 */
export function raceWork<T>(
  work: Promise<T>,
  ms: number | undefined,
  signal: AbortSignal,
  wait?: (ms: number, waitSignal: AbortSignal) => Promise<void>,
): Promise<RaceOutcome<T>> {
  return new Promise<RaceOutcome<T>>((resolve, reject) => {
    if (signal.aborted) {
      // Observe a `work` we are about to abandon: it may still reject, and with no handler attached that
      // becomes an unhandled rejection (the caller already has its answer — canceled).
      void work.catch(() => undefined);
      resolve({ status: "canceled" });
      return;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    // An injected `wait` (a clock-driven timer, so a deadline is enforced in the SAME time base its
    // remaining window was computed from, not real `setTimeout` ms) is stopped via this controller on the
    // branch that did not win — exactly as the built-in `timer` is cleared.
    const waitAbort = wait ? new AbortController() : undefined;
    const cleanup = (): void => {
      if (timer !== undefined) clearTimeout(timer);
      waitAbort?.abort();
      signal.removeEventListener("abort", onAbort);
    };
    function onAbort(): void {
      cleanup();
      resolve({ status: "canceled" });
    }
    signal.addEventListener("abort", onAbort);
    if (ms !== undefined && Number.isFinite(ms)) {
      if (wait) {
        void wait(ms, waitAbort!.signal).then(() => {
          if (waitAbort!.signal.aborted) return; // stopped early because another branch won — not a timeout
          cleanup();
          resolve({ status: "timeout" });
        });
      } else {
        timer = setTimeout(() => {
          cleanup();
          resolve({ status: "timeout" });
        }, ms);
      }
    }
    work.then(
      (value) => {
        cleanup();
        resolve({ status: "done", value });
      },
      (err: unknown) => {
        cleanup();
        reject(err instanceof Error ? err : new Error(String(err)));
      },
    );
  });
}

/**
 * Sleep `ms`, waking EARLY when `signal` aborts — and CLEARING the timer when it does.
 *
 * A retry backoff is where this matters: with a server `retry-after: 60` the wait is a full minute, so
 * a bare `await waitMs(delay)` makes a cancel land a minute late, and a `Promise.race` that wakes early
 * still leaves the minute-long timer pending (which, un-`unref`'d, holds the process open). Resolves
 * either way — the caller re-checks cancellation rather than distinguishing here.
 */
export function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort(): void {
      clearTimeout(timer);
      resolve();
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/** The control surface {@link wrapHandle} hands a wrapper body: check whether cancel already landed
 *  (BEFORE starting an inner call), race an in-flight wait against cancellation, and register each inner
 *  handle as it starts — registration forwards its events and makes it the cancel target. */
export interface WrapControl<M extends ExecMetrics = ExecMetrics> {
  canceled(): boolean;
  /** Aborted when `handle.cancel()` lands OR the caller's `ctx.abortSignal` fires. A body must RACE
   *  every wait longer than a tick against this (see {@link abortableDelay}/{@link raceWork}); checking
   *  `canceled()` only at the top of a loop cannot end a wait already in flight. */
  readonly signal: AbortSignal;
  started(h: ExecHandle<ResolvedValue, M>): ExecHandle<ResolvedValue, M>;
}

export interface WrapHandleOptions {
  /** The caller's cancellation, i.e. `ctx.abortSignal`. Linked in — and unlinked when the handle
   *  settles — so an external abort cancels exactly like `handle.cancel()` does. */
  signal?: AbortSignal;
  /** The `reason` recorded on the `canceled` failure the abort path settles with. */
  canceledReason?: string;
}

/**
 * The shared handle scaffold for wrappers whose body starts inner handles asynchronously. Every inner
 * handle registered via `ctl.started` has its events forwarded (concatenated, in registration order) to
 * the outer `events` stream.
 *
 * Cancellation (`handle.cancel()` or `options.signal` firing — one internal controller, so the two are
 * indistinguishable to the body):
 *  1. flips `ctl.canceled()`/`ctl.signal` FIRST, so a body that hasn't started its inner call yet
 *     short-circuits and one already parked in a backoff wakes (the pre-start and mid-wait windows a
 *     plain `innerHandle?.cancel()` misses);
 *  2. cancels the current inner handle WITHOUT awaiting it;
 *  3. settles this handle itself with a `canceled` failure.
 *
 * Step 3 is what makes cancel bounded: awaiting the body means awaiting whatever it is parked on, so an
 * inner call that ignores cancellation left `result` pending forever and `cancel()` hanging with it.
 * The settle is guarded, so the body finishing and the abort landing race for it and exactly one wins.
 *
 * A body throw/rejection is normalized into a permanent failure (the `result` promise NEVER rejects,
 * per the contract).
 */
export function wrapHandle<M extends ExecMetrics = ExecMetrics>(
  body: (ctl: WrapControl<M>) => Promise<ExecResult<ResolvedValue, M>>,
  options: WrapHandleOptions = {},
): ExecHandle<ResolvedValue, M> {
  const registered: ExecHandle<ResolvedValue, M>[] = [];
  let current: ExecHandle<ResolvedValue, M> | undefined;
  let done = false;
  const waiters: (() => void)[] = [];
  const wake = (): void => {
    for (const w of waiters.splice(0)) w();
  };

  const abort = new AbortController();
  const unlink = linkAbort(abort, options.signal);

  let settle!: (r: ExecResult<ResolvedValue, M>) => void;
  const result = new Promise<ExecResult<ResolvedValue, M>>((resolve) => {
    settle = resolve;
  });
  // EXACTLY ONCE. A second settle on a promise is silently swallowed, which would hide the real
  // question — whether the bookkeeping (unlink, wake) ran twice — so the guard is explicit.
  const finish = (r: ExecResult<ResolvedValue, M>): void => {
    if (done) return;
    done = true;
    unlink();
    settle(r);
    wake();
  };

  const ctl: WrapControl<M> = {
    canceled: () => abort.signal.aborted,
    signal: abort.signal,
    started(h) {
      current = h;
      registered.push(h);
      if (abort.signal.aborted) void h.cancel();
      wake();
      return h;
    },
  };

  const onAbort = (): void => {
    void current?.cancel(); // best-effort stop of the in-flight inner call — deliberately NOT awaited
    finish(canceledFailure(options.canceledReason ?? "the operation was canceled") as ExecResult<ResolvedValue, M>);
  };
  abort.signal.addEventListener("abort", onAbort, { once: true });

  // If `options.signal` was ALREADY aborted, `linkAbort` aborted `abort` SYNCHRONOUSLY above — before this
  // listener existed, and a listener added to an already-aborted signal never fires. Settle here and DO
  // NOT start the body: without this, a body that awaits before its first `ctl.canceled()` check (e.g.
  // memoize awaiting `cache.get`) would leave `result` pending forever and `cancel()` hanging with it.
  if (abort.signal.aborted) {
    onAbort();
  } else {
    void body(ctl)
      .catch((err: unknown) => permanentFailure(err instanceof Error ? err.message : String(err)) as ExecResult<ResolvedValue, M>)
      .then(finish);
  }

  // `events` is SINGLE-CONSUMER (see {@link ExecHandle.events}): the drain owns its position in the
  // registered-handles list and consumes each inner stream, so a second one would steal events from the
  // first. A second attach is an explicit error rather than a silent hang.
  let consumed = false;
  async function* drain(): AsyncGenerator<ExecEvent> {
    let i = 0;
    for (;;) {
      while (i < registered.length) yield* registered[i++]!.events;
      if (done) return;
      await new Promise<void>((resolve) => {
        waiters.push(resolve);
      });
    }
  }

  return {
    events: {
      [Symbol.asyncIterator](): AsyncIterator<ExecEvent> {
        if (consumed) throw new Error(SINGLE_CONSUMER_REASON);
        consumed = true;
        return drain();
      },
    },
    result,
    cancel: async () => {
      abort.abort();
      // Already settled by the abort listener above; awaiting it is a formality that also lets a caller
      // read the final result off `cancel()`. It can no longer wait on the operation itself.
      await result.catch(() => undefined);
    },
  };
}

/** The message both single-consumer guards raise — one string so the contract reads identically from
 *  either end of the stream. */
export const SINGLE_CONSUMER_REASON =
  "an ExecHandle event stream is SINGLE-CONSUMER: events are delivered to one iterator, so a second `for await` would steal them from the first. Fan out downstream of the single drain (e.g. push into your own broadcaster) rather than attaching twice.";

/**
 * A simple event queue an executor pushes into while it runs.
 *
 * SINGLE-CONSUMER, like the handles that expose it: `buffer`/`waiters` are the QUEUE's, so two iterators
 * would each shift from the same buffer and split the stream between them. The second attach throws
 * ({@link SINGLE_CONSUMER_REASON}) instead of silently delivering half the events to each.
 */
export class EventQueue {
  private buffer: ExecEvent[] = [];
  private waiters: Array<(v: IteratorResult<ExecEvent>) => void> = [];
  private closed = false;
  private consumed = false;

  push(event: ExecEvent): void {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter({ value: event, done: false });
    else this.buffer.push(event);
  }

  close(): void {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) waiter({ value: undefined as never, done: true });
  }

  iterate(): AsyncIterable<ExecEvent> {
    const self = this;
    return {
      [Symbol.asyncIterator](): AsyncIterator<ExecEvent> {
        if (self.consumed) throw new Error(SINGLE_CONSUMER_REASON);
        self.consumed = true;
        return {
          next(): Promise<IteratorResult<ExecEvent>> {
            const buffered = self.buffer.shift();
            if (buffered !== undefined) return Promise.resolve({ value: buffered, done: false });
            if (self.closed) return Promise.resolve({ value: undefined as never, done: true });
            return new Promise((resolve) => self.waiters.push(resolve));
          },
        };
      },
    };
  }
}
