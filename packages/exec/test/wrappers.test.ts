import { describe, expect, expectTypeOf, it } from "vitest";
import type { ExecEvent, ExecHandle, Failure, FunctionInputs, FunctionRegistry, ExecMetrics, ExecResult, ExecServices, Executor, InlineFamily, Operation, Ref, ResolvedValue } from "../src/index.js";
import {
  EXEC_METRICS_ALGEBRA,
  isOk,
  hostFunction,
  pureFunction,
  HOST_CAPABILITIES,
  MapMemoCache,
  OperationExecutor,
  PURE_CAPABILITIES,
  RUNTIME_CAPABILITIES,
  compose,
  composeExecutors,
  resolveLiteralInputs,
  withDeadline,
  withMemoize,
  withMetrics,
  withRetry,
} from "../src/index.js";


/** Read a result's failure, or `undefined` when it succeeded. `error` is not a property of the union —
 *  a success branch has no such key — so a test reads it through the narrowing predicate like anyone else. */
const errorOf = <O>(r: ExecResult<O>): Failure | undefined => (isOk(r) ? undefined : r.error);

const CAPS = { ...RUNTIME_CAPABILITIES, memoizable: true, runtime: "edge-safe" as const };

function op(user = "hi"): Operation<InlineFamily> {
  return { kind: "prompt", user, config: { model: "m" }, input: {}, output: { name: "output", kind: "json" } };
}

/** A core executor that replays a scripted outcome sequence and records what it was handed. */
function scripted(script: ExecResult<ResolvedValue>[]): { core: Executor; calls: { op: Operation<InlineFamily>; ctx: ExecServices }[] } {
  const calls: { op: Operation<InlineFamily>; ctx: ExecServices }[] = [];
  const core: Executor = {
    capabilities: CAPS,
    metrics: EXEC_METRICS_ALGEBRA,
    start(o, ctx) {
      calls.push({ op: o, ctx });
      const result = Promise.resolve(script[Math.min(calls.length - 1, script.length - 1)]!);
      return { events: emptyStream(), result, cancel: async () => {} };
    },
  };
  return { core, calls };
}

function emptyStream(): AsyncIterable<ExecEvent> {
  return {
    // eslint-disable-next-line require-yield
    async *[Symbol.asyncIterator]() {},
  };
}

const ok = (over: Partial<ExecResult<ResolvedValue>> = {}): ExecResult<ResolvedValue> => ({ value: { a: 1 }, metrics: { durationMs: 5 }, ...over });
const failed = (classification: string, reason: string): ExecResult<ResolvedValue> => ({
  metrics: { durationMs: 1 },
  error: { classification: classification as never, reason },
});

describe("OperationExecutor — dispatch by op kind (DESIGN §3.1)", () => {
  it("routes a prompt op to the injected prompt executor", async () => {
    const { core, calls } = scripted([ok()]);
    const exec = new OperationExecutor({ functions: new Map(), prompt: core });
    const out = await exec.start(op(), {}).result;
    expect(out.value).toEqual({ a: 1 });
    expect(calls).toHaveLength(1);
  });

  it("fails permanently — and namefully — when a graph has a prompt op but no prompt executor", async () => {
    const exec = new OperationExecutor({ functions: new Map() });
    const out = await exec.start(op(), {}).result;
    expect(errorOf(out)?.reason).toMatch(/no prompt executor is wired in/);
  });

  it("routes a function op to the registry and returns the impl's Result value", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("double", pureFunction((inputs: FunctionInputs) => ({ value: Number(inputs.n) * 2 }), PURE_CAPABILITIES));
    const exec = new OperationExecutor({ functions });
    const out = await exec.start(
      { kind: "function", functionRef: "double", input: { n: { kind: "json", binding: { json: 21 } } }, output: { name: "output", kind: "json" } },
      {},
    ).result;
    expect(out.value).toBe(42);
  });

  it("carries an impl's CLASSIFIED failure through as data — no err.name guessing (§4.2)", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set(
      "flaky",
      hostFunction(
        async () => ({ error: { classification: "network-retriable" as const, reason: "429 from the upstream", rateLimited: true } }),
        HOST_CAPABILITIES,
      ),
    );
    const exec = new OperationExecutor({ functions });
    const out = await exec.start({ kind: "function", functionRef: "flaky", input: {}, output: { name: "output", kind: "json" } }, {}).result;
    expect(errorOf(out)).toMatchObject({ classification: "network-retriable", rateLimited: true });
  });

  it("reports an unresolved producer edge instead of silently skipping the input", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("f", pureFunction(() => ({ value: null })));
    const exec = new OperationExecutor({ functions });
    const out = await exec.start(
      {
        kind: "function",
        functionRef: "f",
        input: { x: { kind: "json", binding: { op: "some-child" } } },
        output: { name: "output", kind: "json" },
      },
      {},
    ).result;
    expect(errorOf(out)?.reason).toMatch(/unresolved binding/);
  });
});

/**
 * A nested operation is the one unresolved binding dispatch can settle by itself — it is a whole
 * document, so running it needs an executor and nothing else. Recursing through THIS executor is what
 * makes the nested op reach the right place by its own kind, decided in one spot instead of by each
 * caller. Everything else that can be unresolved (a `{refs}` tree, a producer edge naming a declared
 * child) is a scope lookup and still has to arrive substituted.
 */
describe("OperationExecutor — embedded operations resolve through dispatch", () => {
  const doubler = (): FunctionRegistry<ExecServices, ExecMetrics> => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("double", pureFunction((inputs: FunctionInputs) => ({ value: Number(inputs.n) * 2 }), PURE_CAPABILITIES));
    functions.set("inc", pureFunction((inputs: FunctionInputs) => ({ value: Number(inputs.n) + 1 }), PURE_CAPABILITIES));
    return functions;
  };

  const call = (fn: string, arg: Ref<InlineFamily>): Operation<InlineFamily> => ({
    kind: "function",
    functionRef: fn,
    input: { n: { kind: "json", binding: arg } },
    output: { name: "output", kind: "json" },
  });

  it("runs an embedded operation and substitutes its output", async () => {
    const exec = new OperationExecutor({ functions: doubler() });
    const out = await exec.start(call("double", { op: call("inc", { json: 20 }) }), {}).result;
    expect(out.value).toBe(42);
  });

  it("recurses, so a tree of calls resolves innermost first", async () => {
    const exec = new OperationExecutor({ functions: doubler() });
    const inner = call("inc", { json: 1 }); // 2
    const middle = call("double", { op: inner }); // 4
    const out = await exec.start(call("double", { op: middle }), {}).result; // 8
    expect(out.value).toBe(8);
  });

  it("sends a nested PROMPT callee to the prompt executor — dispatch is by the nested op's own kind", async () => {
    const { core, calls } = scripted([ok()]);
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    functions.set("id", pureFunction((inputs: FunctionInputs) => ({ value: inputs.n ?? null }), PURE_CAPABILITIES));
    const exec = new OperationExecutor({ functions, prompt: core });
    const out = await exec.start(call("id", { op: op("nested") }), {}).result;
    // The prompt executor ran the NESTED op, and its value became the outer op's input.
    expect(calls.map((c) => (c.op as { user?: string }).user)).toEqual(["nested"]);
    expect(out.value).toEqual({ a: 1 });
  });

  it("carries a nested failure out WHOLE, keeping its classification", async () => {
    const functions = doubler();
    functions.set("boom", pureFunction(() => failed("network-retriable", "429 from upstream") as never, PURE_CAPABILITIES));
    const exec = new OperationExecutor({ functions });
    const out = await exec.start(call("double", { op: call("boom", { json: 1 }) }), {}).result;
    // Not restated as "an input failed": a retriable provider error must stay distinguishable from a
    // wiring mistake, or the retry wrapper above has nothing to act on.
    expect(errorOf(out)?.classification).toBe("network-retriable");
    expect(errorOf(out)?.reason).toMatch(/429 from upstream/);
  });

  it("runs the nested operation but NOT the outer one when the nested fails", async () => {
    const functions = doubler();
    let outerRan = 0;
    let nestedRan = 0;
    functions.set("count", pureFunction(() => { outerRan++; return { value: 0 }; }, PURE_CAPABILITIES));
    functions.set("boom", pureFunction(() => { nestedRan++; return failed("permanent", "no") as never; }, PURE_CAPABILITIES));
    const exec = new OperationExecutor({ functions });
    await exec.start(call("count", { op: call("boom", { json: 1 }) }), {}).result;
    // Both halves matter: the nested op has to have been dispatched at all (otherwise this passes for
    // the wrong reason — an outer op that simply refused an unresolved binding), and the outer op has
    // to have been skipped once it failed.
    expect(nestedRan).toBe(1);
    expect(outerRan).toBe(0);
  });

  it("rolls nested metrics up, so an embedded call's cost is not lost", async () => {
    // The nested op is a PROMPT, scripted to report a duration the outer function cannot produce on
    // its own — so the number can only appear in the outer frame by being merged in.
    const core: Executor = {
      capabilities: CAPS,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: emptyStream(), result: Promise.resolve({ value: 1 as ResolvedValue, metrics: { durationMs: 500 } }), cancel: async () => {} }),
    };
    const exec = new OperationExecutor({ functions: doubler(), prompt: core });
    const out = await exec.start(call("double", { op: op("nested") }), {}).result;
    expect(out.value).toBe(2);
    expect(out.metrics.durationMs).toBeGreaterThanOrEqual(500);
  });

  it("leaves a HIGHER-ORDER edge alone — the definition IS the value there", async () => {
    const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
    let received: unknown;
    functions.set("apply", pureFunction((inputs: FunctionInputs) => { received = inputs.f; return { value: 1 }; }, PURE_CAPABILITIES));
    const exec = new OperationExecutor({ functions });
    const passed = call("inc", { json: 1 });
    await exec.start(
      { kind: "function", functionRef: "apply", input: { f: { kind: "function", binding: { op: passed } } }, output: { name: "output", kind: "json" } },
      {},
    ).result;
    // `inc` was handed over as a document, never run: `kind: "function"` means pass the definition.
    expect(received).toEqual(passed);
  });
});

describe("resolveLiteralInputs — every binding form that is ALREADY a value (ops model.ts:150-155)", () => {
  const child: Operation<InlineFamily> = {
    kind: "prompt",
    user: "summarize {{text}}",
    config: { model: "m" },
    input: {},
    output: { name: "output", kind: "json" },
  };

  it("passes an embedded op DEFINITION through for a `function`/`prompt` kind (higher-order)", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "mapOver",
      input: { each: { kind: "prompt", binding: { op: child } } },
      output: { name: "output", kind: "json" },
    });
    expect("values" in resolved && resolved.values.each).toBe(child);
  });

  it("reads the value straight off a RESOLVED result record", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "f",
      input: {
        prior: {
          kind: "json",
          binding: { result: { source: "child", inputs: [], result: { value: { total: 7 } }, metrics: { durationMs: 3 } } },
        },
      },
      output: { name: "output", kind: "json" },
    });
    expect("values" in resolved && resolved.values.prior).toEqual({ total: 7 });
  });

  it("refuses a result record that FAILED — there is no value to pass", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "f",
      input: {
        prior: {
          kind: "json",
          binding: {
            result: { source: "child", inputs: [], result: { error: { classification: "permanent", reason: "upstream died" } }, metrics: { durationMs: 0 } },
          },
        },
      },
      output: { name: "output", kind: "json" },
    });
    expect("error" in resolved && resolved.error).toMatch(/FAILED \(upstream died\)/);
  });

  it("keeps refusing a producer edge on a DATA kind — running it is the family's job", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "f",
      input: { x: { kind: "json", binding: { op: child } } },
      output: { name: "output", kind: "json" },
    });
    expect("error" in resolved && resolved.error).toMatch(/must be RUN and its output substituted/);
  });

  it("refuses a LOCAL CHILD NAME even at an op kind — a name is not a definition we hold", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "mapOver",
      input: { each: { kind: "function", binding: { op: "some-child" } } },
      output: { name: "output", kind: "json" },
    });
    expect("error" in resolved && resolved.error).toMatch(/names a declared child 'some-child'/);
  });

  it("refuses a ref TREE with the reason it is refused, not a generic 'unresolved binding'", () => {
    const resolved = resolveLiteralInputs({
      kind: "function",
      functionRef: "f",
      input: { x: { kind: "json", binding: { refs: [{ text: "a" }, { text: "b" }] } } },
      output: { name: "output", kind: "json" },
    });
    expect("error" in resolved && resolved.error).toMatch(/ref TREE .* is the ref family's job/);
  });
});

describe("withRetry — now reaching FUNCTION ops too", () => {
  const noWait = { cap: 2, waitMs: async (): Promise<void> => {}, random: (): number => 0 };

  it("retries a transient failure up to the cap, then succeeds (metrics accumulate)", async () => {
    const { core, calls } = scripted([failed("network-retriable" as never, "429"), failed("network-retriable" as never, "429"), ok()]);
    const out = await withRetry({ transient: noWait }, core).start(op(), {}).result;
    expect(calls).toHaveLength(3);
    expect(errorOf(out)).toBeUndefined();
    expect(out.metrics.durationMs).toBe(1 + 1 + 5);
  });

  it("stops at the transient cap and returns the last failure", async () => {
    const { core, calls } = scripted([failed("network-retriable" as never, "429")]);
    const out = await withRetry({ transient: noWait }, core).start(op(), {}).result;
    expect(calls).toHaveLength(3); // 1 attempt + 2 retries
    expect(errorOf(out)?.reason).toBe("429");
  });

  it("does NOT retry a permanent failure on either axis", async () => {
    const { core, calls } = scripted([failed("permanent" as never, "bad input")]);
    const out = await withRetry({ transient: noWait, validation: { turns: 2 } }, core).start(op(), {}).result;
    expect(calls).toHaveLength(1);
    expect(errorOf(out)?.reason).toBe("bad input");
  });

  it("validation.feedback:true appends the concrete errors to the PROMPT (the former withRepair)", async () => {
    const { core, calls } = scripted([failed("api-retriable" as never, "output failed schema validation: /a must be string"), ok()]);
    await withRetry({ validation: { turns: 1, feedback: true } }, core).start(op("summarize"), {}).result;
    expect(calls).toHaveLength(2);
    const second = calls[1]!.op;
    expect(second.kind === "prompt" && second.user).toContain("summarize");
    expect(second.kind === "prompt" && second.user).toContain("must be string");
  });

  it("validation.feedback:false is a blind re-roll (same op, no hint)", async () => {
    const { core, calls } = scripted([failed("api-retriable" as never, "output failed schema validation: x"), ok()]);
    await withRetry({ validation: { turns: 1, feedback: false } }, core).start(op("summarize"), {}).result;
    expect(calls[1]!.op).toEqual(calls[0]!.op);
  });

  it("ACCUMULATES feedback across turns — the third attempt still carries the first turn's errors", async () => {
    const { core, calls } = scripted([failed("api-retriable" as never, "ERR_ONE"), failed("api-retriable" as never, "ERR_TWO"), ok()]);
    await withRetry({ validation: { turns: 2, feedback: true } }, core).start(op("summarize"), {}).result;
    expect(calls).toHaveLength(3);
    const third = calls[2]!.op;
    // Hinting onto the ORIGINAL op each turn re-based the prompt and dropped ERR_ONE, so the model was
    // left to rediscover a mistake it had already been corrected on.
    expect(third.kind === "prompt" && third.user).toContain("ERR_ONE");
    expect(third.kind === "prompt" && third.user).toContain("ERR_TWO");
  });

  it("repairs a TRUNCATED output — the check is the classification, not the word 'validation'", async () => {
    const { core, calls } = scripted([
      failed("api-retriable" as never, 'finishReason "length": output truncated before the JSON closed'),
      ok(),
    ]);
    await withRetry({ validation: { turns: 1, feedback: true } }, core).start(op("summarize"), {}).result;
    expect(calls).toHaveLength(2); // the prose check missed every api-retriable failure without the word
    const second = calls[1]!.op;
    expect(second.kind === "prompt" && second.user).toContain("truncated");
  });

  // The classification is a strict superset of the old prose match in the false-POSITIVE direction (an
  // `api-retriable` failure is by definition one a fresh run can fix), so what still has to be pinned is
  // that a failure of another CLASS is never repaired, however its reason happens to read.
  it("does NOT spend a validation turn on a failure of another class whose reason mentions validation", async () => {
    const { core, calls } = scripted([failed("permanent" as never, "the validation service is unreachable")]);
    await withRetry({ validation: { turns: 3, feedback: true } }, core).start(op(), {}).result;
    expect(calls).toHaveLength(1);
  });
});

describe("withRetry — the short-circuits folded in from the deleted second retry loop", () => {
  const noWait = { cap: 5, waitMs: async (): Promise<void> => {}, random: (): number => 0 };

  it("does not sleep a backoff to re-attempt a window it already knows is CLOSED", async () => {
    const { core, calls } = scripted([failed("network-retriable" as never, "deadline-floor: 300ms remaining is below the start floor")]);
    await withRetry({ transient: noWait }, core).start(op(), {}).result;
    expect(calls).toHaveLength(1);
  });

  it("short-circuits a budget-exhausted failure the same way", async () => {
    const { core, calls } = scripted([failed("network-retriable" as never, "budget-exhausted: the wallet is empty")]);
    await withRetry({ transient: noWait }, core).start(op(), {}).result;
    expect(calls).toHaveLength(1);
  });

  it("stops when the budget gate refuses another attempt", async () => {
    const { core, calls } = scripted([failed("network-retriable" as never, "429")]);
    let attempts = 0;
    await withRetry({ transient: noWait, budget: { allowMore: () => ++attempts < 2 } }, core).start(op(), {}).result;
    expect(calls).toHaveLength(2); // one gated retry, then the gate says no
  });
});

describe("withDeadline", () => {
  it("fails fast with a 'deadline-floor' reason when the window is below the floor", async () => {
    const { core, calls } = scripted([ok()]);
    const out = await withDeadline({ deadline: { maxDurationMs: 10_000 }, stepStartMs: 0 }, core).start(op(), {
      clock: { now: () => 9_000 },
    }).result;
    expect(calls).toHaveLength(0);
    expect(errorOf(out)?.classification).toBe("deadline");
    expect(errorOf(out)?.reason).toMatch(/^deadline-floor/);
  });

  it("bounds the call by CANCELLATION, on the signal an executor already reads", async () => {
    // This replaced a test that asserted the wrapper clamped `ctx.timeoutMs` to the remaining window.
    // That number was a second spelling of a bound the abort channel already carries — the llm layer
    // turned it straight back into `AbortSignal.timeout(...)` and folded it into the signal — and only
    // an executor that knew to read it got the window at all. An abort reaches every executor.
    const { core, calls } = scripted([ok()]);
    const outer = new AbortController();
    await withDeadline({ deadline: { maxDurationMs: 100_000, safetyMarginMs: 0 }, stepStartMs: 0 }, core).start(op(), {
      clock: { now: () => 40_000 },
      abortSignal: outer.signal,
    }).result;
    const seen = calls[0]!.ctx.abortSignal;
    expect(seen).toBeDefined();
    expect(seen).not.toBe(outer.signal); // combined, not passed through
    // …and the CALLER's cancellation still reaches the call through it.
    expect(seen!.aborted).toBe(false);
    outer.abort();
    expect(seen!.aborted).toBe(true);
  });

  it("takes its window at CONSTRUCTION, so no deadline field reaches the core at all", async () => {
    // This replaced a test that put the window on ctx and asserted the wrapper stripped it back off.
    // The strip existed because the window was readable from two places; with one place there is
    // nothing to consume — a window is the wrapper's subject, not a service the core needs to run.
    const { core, calls } = scripted([ok()]);
    await withDeadline({ deadline: { maxDurationMs: 100_000 }, stepStartMs: 0 }, core).start(op(), { clock: { now: () => 0 } })
      .result;
    expect(calls[0]!.ctx).not.toHaveProperty("deadline");
    expect(calls[0]!.ctx).not.toHaveProperty("stepStartMs");
  });
});

describe("withMemoize — keyed on the OPERATION's content hash", () => {
  it("caches the final result; a second identical call skips the inner executor", async () => {
    const { core, calls } = scripted([ok()]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    await stack.start(op(), {}).result;
    const second = await stack.start(op(), {}).result;
    expect(calls).toHaveLength(1);
    expect(second.value).toEqual({ a: 1 });
  });

  it("distinguishes ops whose RESOLVED inputs differ — the op embeds its inputs", async () => {
    const { core, calls } = scripted([ok()]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    await stack.start(op("one"), {}).result;
    await stack.start(op("two"), {}).result;
    expect(calls).toHaveLength(2);
  });

  it("does not cache a failure", async () => {
    const { core, calls } = scripted([failed("permanent" as never, "nope")]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    await stack.start(op(), {}).result;
    await stack.start(op(), {}).result;
    expect(calls).toHaveLength(2);
  });

  it("REFUSES (throws at composition time) to wrap a session layer", () => {
    const sessionish: Executor = { capabilities: { ...CAPS, sessionResume: true }, metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({} as ExecHandle<ResolvedValue>) };
    expect(() => withMemoize({ cache: new MapMemoCache() }, sessionish)).toThrow(/must not wrap a session layer/);
  });

  // A workspace makes the answer snapshot-dependent whether or not the op WRITES to it: reading a
  // checkout is enough. Gating the FOLD on `mutatesWorkspace` replayed a stale result across a changed
  // tree for every read-only workspace op, so the fold is unconditional.
  //
  // The REFUSAL below does read `mutatesWorkspace` — and reads it off the dispatched ENTRY now
  // (`Executor.capabilitiesFor`), not off the dispatching executor's one static record, where it was
  // always false and the per-variant capability could never reach the gate. These two tests pin the
  // leaf-executor case, where the static record IS the whole truth; memo.test.ts pins the dispatcher.
  it("re-runs when the workspace snapshot changed, even for a read-only executor", async () => {
    const { core, calls } = scripted([ok()]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    await stack.start(op(), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    await stack.start(op(), { workspace: { root: "/w", treeHash: "bbb" } }).result;
    expect(calls).toHaveLength(2);
  });

  it("still hits the cache when the workspace snapshot is unchanged", async () => {
    const { core, calls } = scripted([ok()]);
    const stack = withMemoize({ cache: new MapMemoCache() }, core);
    await stack.start(op(), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    await stack.start(op(), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    expect(calls).toHaveLength(1);
  });

  it("REFUSES to memoize a mutatesWorkspace leaf executor with no pinned snapshot", async () => {
    const { core, calls } = scripted([ok()]);
    const mutating: Executor = { ...core, capabilities: { ...CAPS, mutatesWorkspace: true } };
    const stack = withMemoize({ cache: new MapMemoCache() }, mutating);
    const outcome = await stack.start(op(), { workspace: { root: "/w" } }).result;
    expect(errorOf(outcome)?.classification).toBe("permanent");
    expect(errorOf(outcome)?.reason).toMatch(/pinned workspace snapshot/);
    expect(calls).toHaveLength(0); // refused, not silently cached under an "any workspace" key
  });

  it("memoizes a mutatesWorkspace executor once a snapshot IS pinned", async () => {
    const { core, calls } = scripted([ok()]);
    const mutating: Executor = { ...core, capabilities: { ...CAPS, mutatesWorkspace: true } };
    const stack = withMemoize({ cache: new MapMemoCache() }, mutating);
    await stack.start(op(), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    await stack.start(op(), { workspace: { root: "/w", treeHash: "aaa" } }).result;
    expect(calls).toHaveLength(1);
  });
});

describe("composition — two forms + typed requirements", () => {
  it("form 1 (direct nesting) and form 2 (builder) nest identically", async () => {
    const a = scripted([ok()]);
    const b = scripted([ok()]);
    const direct = withMemoize({ cache: new MapMemoCache() }, withRetry({ transient: 1 }, a.core));
    const built = compose(b.core).with(withRetry({ transient: 1 })).with(withMemoize({ cache: new MapMemoCache() }));
    expect((await direct.start(op(), {}).result).value).toEqual((await built.start(op(), {}).result).value);
  });

  it("the variadic convenience stacks the same way and is an Executor", async () => {
    const { core } = scripted([ok()]);
    const stack = composeExecutors(core, withRetry({ transient: 1 }), withMemoize({ cache: new MapMemoCache() }));
    expectTypeOf(stack).toExtend<Executor>();
    expect((await stack.start(op(), {}).result).value).toEqual({ a: 1 });
  });

  it("start() requires exactly the seams the stack consumes (compile-time)", () => {
    const { core } = scripted([ok()]);
    const needsBoth = compose(core).with(withDeadline());
    // @ts-expect-error — `withDeadline()` supplied neither seam, so both are required at start.
    needsBoth.start(op(), {});
    const needsStep = compose(core).with(withDeadline({ deadline: { maxDurationMs: 1 } }));
    // @ts-expect-error — the deadline came from construction, but `stepStartMs` is still required.
    needsStep.start(op(), {});
    compose(core).with(withDeadline({ deadline: { maxDurationMs: 1 }, stepStartMs: 0 })).start(op(), {});
  });
});

/**
 * The session outcome survives a wrapper that re-reports metrics.
 *
 * `SessionOutcome` is a DECLARED field precisely so it cannot be dropped — its own doc says "an
 * undeclared extra field typechecks by leniency and would be dropped by anything that rebuilt the
 * result". `withMetrics` is the thing that rebuilds the result, and it dropped it anyway.
 *
 * The damage was silent and specific. A delegated agent's whole conversation lives server-side, and
 * the one thing it hands back is the `providerSessionId` the run ended in. Lose that and the next
 * call resumes nothing — it opens a fresh remote conversation while the workflow reads as though
 * `session: "review"` had joined them up. Any host wrapping its prompt executor in a repair or retry
 * loop hit this on every single call.
 */
describe("withMetrics — what a re-reported result keeps", () => {
  const metrics = { durationMs: 1 };

  it("carries a success's session outcome across", () => {
    const result = withMetrics(
      { value: "ok", metrics: { durationMs: 0 }, session: { providerSessionId: "prov-1" } } as never,
      metrics,
    );
    expect((result as { session?: unknown }).session).toEqual({ providerSessionId: "prov-1" });
    expect(isOk(result) && result.value).toBe("ok");
  });

  it("carries a FAILURE's session outcome across too", () => {
    // A failed call is still a record: it ran, it cost money, and for a session its turns may already
    // exist remotely — so the handle it ended on matters exactly as much.
    const result = withMetrics(
      { error: { classification: "permanent", reason: "no" }, metrics: { durationMs: 0 }, session: { messages: [1] } } as never,
      metrics,
    );
    expect((result as { session?: unknown }).session).toEqual({ messages: [1] });
  });

  it("adds nothing when the execution reported none", () => {
    const result = withMetrics({ value: "ok", metrics: { durationMs: 0 } }, metrics);
    expect("session" in result).toBe(false);
  });
});
