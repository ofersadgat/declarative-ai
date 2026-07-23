import { describe, expect, it } from "vitest";
import type {
  ExecEvent,
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
  HOST_CAPABILITIES,
  MapMemoCache,
  OperationExecutor,
  RUNTIME_CAPABILITIES,
  hostFunction,
  isOk,
  pureFunction,
  runtimeFunction,
  withMemoize,
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

/** A core executor that replays a scripted outcome sequence and records what it was handed. */
function scripted(script: ExecResult<ResolvedValue>[]): { core: Executor; calls: number } {
  const state = { calls: 0 };
  const core: Executor = {
    capabilities: CAPS,
    metrics: EXEC_METRICS_ALGEBRA,
    start() {
      state.calls++;
      return { events: emptyStream(), result: Promise.resolve(script[Math.min(state.calls - 1, script.length - 1)]!), cancel: async () => {} };
    },
  };
  return {
    core,
    get calls() {
      return state.calls;
    },
  };
}

const ok = (value: ResolvedValue = { a: 1 }, metrics: ExecMetrics = { durationMs: 5 }): ExecResult<ResolvedValue> => ({ value, metrics });

describe("withMemoize — start() never throws (the seam handles.ts exists to hold)", () => {
  const streamingOp: Operation<InlineFamily> = {
    kind: "function",
    functionRef: "upload",
    input: {
      file: {
        kind: "blob",
        // The documented (§7.3) live-stream case `hashOperation` deliberately refuses.
        binding: { blob: { getReader: () => ({ read: async () => ({ done: true }), cancel: async () => {}, releaseLock: () => {} }) } },
      },
    },
    output: { name: "output", kind: "json" },
  };

  it("turns a refused operation hash into a FAILED HANDLE, not a throw out of start()", async () => {
    const { core } = scripted([ok()]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    let handle;
    expect(() => (handle = stack.start(streamingOp, {}))).not.toThrow(); // key computation used to run out here
    expect(errorOf(await handle!.result)?.classification).toBe("permanent");
    expect(errorOf(await handle!.result)?.reason).toMatch(/materialize it to a Uint8Array/);
  });

  it("turns a throwing caller-supplied `identify` into a failed handle too", async () => {
    const { core } = scripted([ok()]);
    const stack = withMemoize(
      {
        cache: new MapMemoCache(),
        identify: () => {
          throw new Error("the bundle is not loaded");
        },
      },
      core,
    );
    let handle;
    expect(() => (handle = stack.start(op(), {}))).not.toThrow();
    expect(errorOf(await handle!.result)?.reason).toMatch(/the bundle is not loaded/);
  });
});

describe("withMemoize — the gate reads the DISPATCHED ENTRY's capabilities (§2)", () => {
  const dispatcher = (functions: FunctionRegistry<ExecServices, ExecMetrics>): Executor => new OperationExecutor({ functions });

  it("does NOT memoize an entry that declares memoizable:false", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("clocky", pureFunction(() => ({ value: ++calls }), { memoizable: false }));
    const stack = withMemoize({ cache: new MapMemoCache() }, dispatcher(functions));
    expect((await stack.start(fnOp("clocky"), {}).result).value).toBe(1);
    // The dispatcher's own static record says `memoizable: true`; the ENTRY says otherwise, and it wins.
    expect((await stack.start(fnOp("clocky"), {}).result).value).toBe(2);
  });

  it("still memoizes the entry NEXT to it that declares memoizable:true", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("pure", pureFunction(() => ({ value: ++calls })));
    const stack = withMemoize({ cache: new MapMemoCache() }, dispatcher(functions));
    await stack.start(fnOp("pure"), {}).result;
    await stack.start(fnOp("pure"), {}).result;
    expect(calls).toBe(1);
  });

  it("refuses an ENTRY declaring mutatesWorkspace with no pinned snapshot, though the dispatcher's record says otherwise", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set(
      "edit",
      runtimeFunction(async () => (calls++, { value: "done" }), { ...RUNTIME_CAPABILITIES, memoizable: true, mutatesWorkspace: true }),
    );
    const stack = withMemoize({ cache: new MapMemoCache() }, dispatcher(functions));
    const out = await stack.start(fnOp("edit"), { workspace: { root: "/w" } }).result;
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/pinned workspace snapshot/);
    expect(calls).toBe(0); // was: cached under a key that silently meant "any workspace"
  });

  it("memoizes that same entry once a snapshot IS pinned", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set(
      "edit",
      runtimeFunction(async () => (calls++, { value: "done" }), { ...RUNTIME_CAPABILITIES, memoizable: true, mutatesWorkspace: true }),
    );
    const stack = withMemoize({ cache: new MapMemoCache() }, dispatcher(functions));
    await stack.start(fnOp("edit"), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    await stack.start(fnOp("edit"), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    expect(calls).toBe(1);
  });

  it("a session-capable PROMPT executor no longer makes FUNCTION ops un-memoizable", async () => {
    let calls = 0;
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("pure", pureFunction(() => ({ value: ++calls })));
    const sessionish: Executor = {
      capabilities: { ...CAPS, sessionResume: true },
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: Promise.resolve(ok("from the session layer")), cancel: async () => {} }),
    };
    const exec = new OperationExecutor({ functions, prompt: sessionish });
    // Composition must not refuse the whole registry over one variant's capability...
    const stack = withMemoize({ cache: new MapMemoCache() }, exec);
    await stack.start(fnOp("pure"), {}).result;
    await stack.start(fnOp("pure"), {}).result;
    expect(calls).toBe(1);
    // ...but the op that actually dispatches to it is still refused, now by name.
    const prompted = await stack.start(op(), {}).result;
    expect(errorOf(prompted)?.reason).toMatch(/must not wrap a session layer/);
  });
});

describe("memoKey — the executor is part of the identity", () => {
  it("two executors sharing ONE cache do not serve each other's results", async () => {
    const cache = new MapMemoCache();
    const a = scripted([ok("from A")]);
    const b = scripted([ok("from B")]);
    const stackA = withMemoize({ cache }, a.core);
    const stackB = withMemoize({ cache }, b.core);
    expect((await stackA.start(op(), {}).result).value).toBe("from A");
    // Byte-identical op, different executor (different routing / registry / a stub vs the real thing).
    expect((await stackB.start(op(), {}).result).value).toBe("from B");
    expect(b.calls).toBe(1);
  });

  it("an EXPLICIT namespace is the way to share a cache on purpose (and to survive the process)", async () => {
    const cache = new MapMemoCache();
    const a = scripted([ok("from A")]);
    const b = scripted([ok("from B")]);
    const stackA = withMemoize({ cache, namespace: "summarizer@3" }, a.core);
    const stackB = withMemoize({ cache, namespace: "summarizer@3" }, b.core);
    await stackA.start(op(), {}).result;
    expect((await stackB.start(op(), {}).result).value).toBe("from A");
    expect(b.calls).toBe(0);
  });
});

describe("withMemoize — the fan-out case, and what a hit costs", () => {
  it("dedups CONCURRENT identical calls into one execution", async () => {
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start() {
        calls++;
        return { events: emptyStream(), result: gate.then(() => ok("once")), cancel: async () => {} };
      },
    };
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    const handles = [0, 1, 2, 3, 4].map(() => stack.start(op(), {}));
    release();
    const results = await Promise.all(handles.map((h) => h.result));
    expect(calls).toBe(1); // was 5: the cache is written on COMPLETION, so all five missed and all five ran
    expect(results.map((r) => r.value)).toEqual(["once", "once", "once", "once", "once"]);
  });

  it("reports NO work for a cache hit — an outer retry/budget layer must not sum a run that did not happen", async () => {
    // `costUsd` is a producer's own field: this wrapper is generic in `M` and cannot know what it means,
    // which is exactly why every numeric measurement is zeroed rather than a hand-picked two.
    const expensive = { durationMs: 5_000, startMs: 111, costUsd: 0.004, costSource: "provider" } as unknown as ExecMetrics;
    const { core } = scripted([ok({ a: 1 }, expensive)]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    const first = await stack.start(op(), { clock: { now: () => 900 } }).result;
    expect(first.metrics.durationMs).toBe(5_000); // the run that DID happen reports what it cost
    const hit = await stack.start(op(), { clock: { now: () => 900 } }).result;
    expect(hit.value).toEqual({ a: 1 });
    expect(hit.metrics).toEqual({ durationMs: 0, startMs: 900, costUsd: 0, costSource: "provider" });
  });

  it("a follower on an in-flight call reports no work either", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: gate.then(() => ok("once", { durationMs: 4_000 })), cancel: async () => {} }),
    };
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    const leader = stack.start(op(), { clock: { now: () => 7 } });
    const follower = stack.start(op(), { clock: { now: () => 7 } });
    release();
    expect((await leader.result).metrics.durationMs).toBe(4_000);
    expect((await follower.result).metrics).toEqual({ durationMs: 0, startMs: 7 });
  });

  it("does not cache a FAILURE, and clears the in-flight slot so the next call re-runs", async () => {
    const { core, ...state } = scripted([
      { metrics: { durationMs: 1 }, error: { classification: "permanent", reason: "nope" } },
      ok("second"),
    ]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    expect(errorOf(await stack.start(op(), {}).result)?.reason).toBe("nope");
    expect((await stack.start(op(), {}).result).value).toBe("second");
  });
});

describe("MapMemoCache", () => {
  it("is unbounded by default, and LRU-bounded when asked", async () => {
    const bounded = new MapMemoCache(2);
    bounded.set("a", ok("A"));
    bounded.set("b", ok("B"));
    bounded.get("a"); // refresh recency, so `b` becomes the eviction candidate
    bounded.set("c", ok("C"));
    expect(bounded.get("a")?.value).toBe("A");
    expect(bounded.get("b")).toBeUndefined();
    expect(bounded.get("c")?.value).toBe("C");

    const unbounded = new MapMemoCache();
    for (let i = 0; i < 50; i++) unbounded.set(`k${i}`, ok(i));
    expect(unbounded.get("k0")?.value).toBe(0);
  });
});

describe("OperationExecutor — an impl's own failure still travels as data", () => {
  it("keeps a host impl's classified failure through the cancellation race", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("flaky", hostFunction(async () => ({ error: { classification: "network-retriable" as const, reason: "429" } }), HOST_CAPABILITIES));
    const out = await new OperationExecutor({ functions }).start(fnOp("flaky"), {}).result;
    expect(errorOf(out)).toMatchObject({ classification: "network-retriable", reason: "429" });
  });
});
