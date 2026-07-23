import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  ExecEvent,
  ExecHandle,
  ExecMetrics,
  ExecResult,
  ExecServices,
  Executor,
  Failure,
  FunctionRegistry,
  InlineFamily,
  Operation,
  ResolvedValue,
} from "../src/index";
import {
  EXEC_METRICS_ALGEBRA,
  EventQueue,
  HOST_CAPABILITIES,
  MapMemoCache,
  OperationExecutor,
  RUNTIME_CAPABILITIES,
  hostFunction,
  isOk,
  pureFunction,
  withDeadline,
  withMemoize,
  withRetry,
} from "../src/index";

const errorOf = <O>(r: ExecResult<O>): Failure | undefined => (isOk(r) ? undefined : r.error);

const CAPS = { ...RUNTIME_CAPABILITIES, memoizable: true, runtime: "edge-safe" as const };

function op(user = "hi"): Operation<InlineFamily> {
  return { kind: "prompt", user, config: { model: "m" }, input: {}, output: { name: "output", kind: "json" } };
}

function fnOp(functionRef: string): Operation<InlineFamily> {
  return { kind: "function", functionRef, input: {}, output: { name: "output", kind: "json" } };
}

function emptyStream(): AsyncIterable<ExecEvent> {
  return {
    // eslint-disable-next-line require-yield
    async *[Symbol.asyncIterator]() {},
  };
}

const failed = (classification: string, reason: string, over: Partial<Failure> = {}): ExecResult<ResolvedValue> => ({
  metrics: { durationMs: 1 },
  error: { classification: classification as never, reason, ...over },
});

/** Let every already-resolved continuation run. Deterministic: no timers, just the microtask queue. */
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

/**
 * An `AbortSignal` that COUNTS its listeners. There is no public API for that on a real signal, and the
 * leak this guards against — a listener left on a caller-owned (run-scoped) signal after the handle
 * settled — is invisible without it.
 */
function spySignal(): { signal: AbortSignal; live: () => number; abort: () => void } {
  const listeners = new Set<() => void>();
  let aborted = false;
  const signal = {
    get aborted(): boolean {
      return aborted;
    },
    addEventListener: (_type: string, fn: () => void): void => void listeners.add(fn),
    removeEventListener: (_type: string, fn: () => void): void => void listeners.delete(fn),
  } as unknown as AbortSignal;
  return {
    signal,
    live: () => listeners.size,
    abort: () => {
      aborted = true;
      for (const fn of [...listeners]) fn();
    },
  };
}

describe("cancellation — ctx.abortSignal and handle.cancel() are the same event", () => {
  it("stops a retry mid-flight instead of running out the whole cap", async () => {
    let calls = 0;
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start() {
        calls++;
        return { events: emptyStream(), result: Promise.resolve(failed("network-retriable", "429")), cancel: async () => {} };
      },
    };
    const controller = new AbortController();
    const stack = withRetry({ transient: { cap: 5, waitMs: async (): Promise<void> => {}, random: () => 0 } }, core);
    const handle = stack.start(op(), { abortSignal: controller.signal });
    controller.abort(); // lands during attempt 1
    const out = await handle.result;
    await flush();
    expect(errorOf(out)?.classification).toBe("canceled");
    expect(calls).toBe(1); // was 6: nothing consumed ctx.abortSignal, so every attempt ran
  });

  it("refuses to dispatch at all when the caller's signal is ALREADY aborted", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("count", pureFunction(() => (calls++, { value: null })));
    const controller = new AbortController();
    controller.abort();
    const out = await new OperationExecutor({ functions }).start(fnOp("count"), { abortSignal: controller.signal }).result;
    expect(errorOf(out)?.classification).toBe("canceled");
    expect(calls).toBe(0);
  });

  it("hands the impl a signal that FIRES, so the work itself can stop", async () => {
    let seen: AbortSignal | undefined;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set(
      "watch",
      hostFunction(async (_inputs, ctx: ExecServices) => {
        seen = ctx.abortSignal;
        await new Promise<void>((r) => ctx.abortSignal?.addEventListener("abort", () => r(), { once: true }));
        return { value: "stopped" };
      }, HOST_CAPABILITIES),
    );
    const handle = new OperationExecutor({ functions }).start(fnOp("watch"), {});
    await flush();
    expect(seen?.aborted).toBe(false);
    await handle.cancel();
    expect(seen?.aborted).toBe(true);
  });

  it("cancel() STOPS the operation rather than awaiting it — an impl that never settles cannot hold it", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    // Ignores its signal entirely: the worst case the handle has to stay bounded against.
    functions.set("hang", hostFunction(() => new Promise(() => {}), HOST_CAPABILITIES));
    const handle = new OperationExecutor({ functions }).start(fnOp("hang"), {});
    await handle.cancel(); // hung forever when cancel was `await result`
    expect(errorOf(await handle.result)?.classification).toBe("canceled");
  });

  it("settles the WRAPPER exactly once even when the inner handle never honors cancel", async () => {
    const stuck: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start() {
        return { events: emptyStream(), result: new Promise<never>(() => {}), cancel: async () => {} };
      },
    };
    const spy = spySignal();
    const handle = withRetry({ transient: 3 }, stuck).start(op(), { abortSignal: spy.signal });
    await flush();
    let settles = 0;
    void handle.result.then(() => settles++);
    spy.abort();
    await flush();
    expect(settles).toBe(1); // "never never": the body is still parked on the inner call
    spy.abort(); // a second abort must not re-settle or re-run the bookkeeping
    await flush();
    expect(settles).toBe(1);
    expect(spy.live()).toBe(0);
  });

  it("removes its listener from the caller's signal once the handle settles NORMALLY", async () => {
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: Promise.resolve({ value: 1, metrics: { durationMs: 1 } }), cancel: async () => {} }),
    };
    const spy = spySignal();
    // Every layer that takes the signal must give it back: a run-scoped signal outlives all of them.
    const stack = withMemoize({ cache: new MapMemoCache() }, withRetry({ transient: 2 }, withDeadline({ deadline: { maxDurationMs: 60_000 }, stepStartMs: 0 }, core)));
    await stack.start(op(), { abortSignal: spy.signal, clock: { now: () => 0 } }).result;
    await flush();
    expect(spy.live()).toBe(0);
  });
});

describe("withRetry — the backoff is raced against cancellation (and its timer cleared)", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("settles immediately on cancel instead of after the full retry-after, and leaves no timer", async () => {
    vi.useFakeTimers();
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      // `retry-after: 60s`, clamped to the default maxBackoffMs — the pathological-but-ordinary case.
      start: () => ({
        events: emptyStream(),
        result: Promise.resolve(failed("network-retriable", "429", { retryAfterMs: 60_000 })),
        cancel: async () => {},
      }),
    };
    const handle = withRetry({ transient: { cap: 5, random: () => 0 } }, core).start(op(), {});
    await vi.advanceTimersByTimeAsync(0); // attempt 1 resolves; the loop enters the backoff
    expect(vi.getTimerCount()).toBe(1);

    let settled = false;
    void handle.result.then(() => (settled = true));
    await vi.advanceTimersByTimeAsync(30_000);
    expect(settled).toBe(false); // still legitimately backing off

    await handle.cancel();
    expect(settled).toBe(true); // was: not until t+60s, with cancel() blocked for the same minute
    expect(vi.getTimerCount()).toBe(0); // the backoff timer was CLEARED, not left holding the loop
    expect(errorOf(await handle.result)?.classification).toBe("canceled");
  });
});

describe("withDeadline — the window is ENFORCED, not merely advertised", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("cuts off an operation that overruns the remaining window", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    let canceled = false;
    const slow: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start() {
        let settle: (r: ExecResult<ResolvedValue>) => void;
        const result = new Promise<ExecResult<ResolvedValue>>((r) => (settle = r));
        // Ignores ctx.timeoutMs, like any executor that does not thread it into an AbortSignal.
        const timer = setTimeout(() => settle({ value: "late", metrics: { durationMs: 300 } }), 300);
        return {
          events: emptyStream(),
          result,
          cancel: async () => {
            canceled = true;
            clearTimeout(timer);
            settle({ error: { classification: "canceled", reason: "cut off" }, metrics: { durationMs: 0 } });
          },
        };
      },
    };
    const stack = withDeadline({ deadline: { maxDurationMs: 50, safetyMarginMs: 0, floorMs: 1 }, stepStartMs: 0 }, slow);
    const handle = stack.start(op(), {});
    await vi.advanceTimersByTimeAsync(200);
    const out = await handle.result;
    expect(errorOf(out)?.classification).toBe("deadline"); // was: {value:"late"} at t+300, past the window
    expect(errorOf(out)?.reason).toMatch(/^deadline-floor/);
    expect(canceled).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears its window timer the moment a fast call returns", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const fast: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: Promise.resolve({ value: 1, metrics: { durationMs: 1 } }), cancel: async () => {} }),
    };
    const out = await withDeadline({ deadline: { maxDurationMs: 300_000 }, stepStartMs: 0 }, fast).start(op(), {}).result;
    expect(out.value).toBe(1);
    expect(vi.getTimerCount()).toBe(0); // no stranded 290s timer holding the event loop open
  });
});

describe("event streams are single-consumer, loudly", () => {
  it("a second iterator over a wrapper handle is an ERROR, not a silent hang", async () => {
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: Promise.resolve({ value: 1, metrics: { durationMs: 1 } }), cancel: async () => {} }),
    };
    const handle: ExecHandle<ResolvedValue> = withRetry({ transient: 0 }, core).start(op(), {});
    const first: ExecEvent[] = [];
    for await (const e of handle.events) first.push(e);
    // Before: the second drain overwrote the first's single `notify` slot and then waited forever.
    expect(() => handle.events[Symbol.asyncIterator]()).toThrow(/single-consumer/i);
  });

  it("a second iterator over an EventQueue is an ERROR, not a split stream", async () => {
    const queue = new EventQueue();
    queue.push({ type: "progress", message: "one" });
    queue.push({ type: "progress", message: "two" });
    queue.close();
    const seen: ExecEvent[] = [];
    for await (const e of queue.iterate()) seen.push(e);
    expect(seen).toHaveLength(2); // both events to the ONE consumer, never one each
    expect(() => queue.iterate()[Symbol.asyncIterator]()).toThrow(/single-consumer/i);
  });
});
