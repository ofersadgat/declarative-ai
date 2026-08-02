import { describe, expect, expectTypeOf, it } from "vitest";
import type { CallTarget, ExecMetrics, ExecServices, FunctionInputs, FunctionRegistry, InlineFamily, Operation, Ref } from "../src/index.js";
import {
  HOST_CAPABILITIES,
  OperationExecutor,
  PURE_CAPABILITIES,
  carryCall,
  isResolvedCall,
  hashOperation,
  hostFunction,
  isOk,
  isPureTree,
  pureFunction,
  resolveCalls,
  withCall,
  tryRunPureSync,
} from "../src/index.js";

const registry = (): FunctionRegistry<ExecServices, ExecMetrics> => {
  const functions: FunctionRegistry<ExecServices, ExecMetrics> = new Map();
  functions.set("add", pureFunction((i: FunctionInputs) => ({ value: Number(i.a) + Number(i.b) }), PURE_CAPABILITIES));
  functions.set("double", pureFunction((i: FunctionInputs) => ({ value: Number(i.n) * 2 }), PURE_CAPABILITIES));
  functions.set("boom", pureFunction(() => ({ error: { classification: "permanent" as const, reason: "nope" } }), PURE_CAPABILITIES));
  functions.set("throws", pureFunction(() => { throw new Error("impl threw"); }, PURE_CAPABILITIES));
  functions.set("readFile", hostFunction(async () => ({ value: "contents" }), HOST_CAPABILITIES));
  return functions;
};

const fn = (functionRef: string, input: Record<string, Ref<InlineFamily>>): Operation<InlineFamily> => ({
  kind: "function",
  functionRef,
  input: Object.fromEntries(Object.entries(input).map(([k, binding]) => [k, { kind: "json" as const, binding }])),
  output: { name: "value", kind: "json" },
});

const resolved = (op: Operation<InlineFamily>): Operation<InlineFamily> => {
  const out = resolveCalls(op, registry());
  if ("error" in out) throw new Error(`unexpected resolve error: ${out.error}`);
  return out.op;
};

describe("resolveCalls — a resolved operation IS an operation", () => {
  it("carries the entry on the operation itself, in place", () => {
    const op = resolved(fn("add", { a: { json: 1 }, b: { json: 2 } }));
    expect(isResolvedCall(op)).toBe(true);
    expect(op.kind).toBe("function");
  });

  it("resolves nested operations IN THE ORDINARY SLOT — no parallel structure to keep in step", () => {
    const op = resolved(fn("double", { n: { op: fn("add", { a: { json: 1 }, b: { json: 2 } }) } }));
    const nested = (op.input.n?.binding as { op: Operation<InlineFamily> }).op;
    expect(isResolvedCall(nested)).toBe(true);
  });

  it("hashes IDENTICALLY to the unresolved document — the memo key must not depend on pre-resolution", () => {
    const plain = fn("add", { a: { json: 1 }, b: { json: 2 } });
    // Built separately so `withCall`'s in-place define cannot be what makes these agree.
    const other = resolved(fn("add", { a: { json: 1 }, b: { json: 2 } }));
    expect(hashOperation(other)).toBe(hashOperation(plain));
  });

  it("is accepted by an executor that knows nothing about resolution", async () => {
    const exec = new OperationExecutor({ functions: registry() });
    // The whole point of extending rather than wrapping: this call site is unchanged.
    const out = await exec.start(resolved(fn("add", { a: { json: 20 }, b: { json: 22 } })), {}).result;
    expect(out.value).toBe(42);
  });

  it("stops being resolved through a spread — a plain FunctionOp again, and nothing has lied", () => {
    const op = resolved(fn("add", { a: { json: 1 }, b: { json: 2 } }));
    // Spread copies own ENUMERABLE properties, so `call` does not survive. The narrowing simply fails
    // and the holder falls back to a lookup; resolution is an optimization, never a guarantee.
    expect(isResolvedCall({ ...op } as Operation<InlineFamily>)).toBe(false);
  });

  it("narrows the TYPE, so `call` is reachable without an optional access", () => {
    const op: Operation<InlineFamily> = resolved(fn("add", { a: { json: 1 }, b: { json: 2 } }));
    if (!isResolvedCall(op)) throw new Error("expected a resolved op");
    expectTypeOf(op.call).toEqualTypeOf<CallTarget>();
    expect(op.call.kind).toBe("pure");
  });

  it("is what the executor DISPATCHES TO, in preference to the registry", async () => {
    // The carried entry and the registered one disagree on purpose: only the carried answer proves
    // the executor consulted the operation rather than looking the name up again.
    const op = fn("double", { n: { json: 21 } });
    withCall(op as never, pureFunction(() => ({ value: "carried" }), PURE_CAPABILITIES));
    const out = await new OperationExecutor({ functions: registry() }).start(op, {}).result;
    expect(out.value).toBe("carried");
  });

  it("still dispatches an UNRESOLVED op — the executor never requires anyone to have resolved first", async () => {
    const out = await new OperationExecutor({ functions: registry() }).start(fn("double", { n: { json: 21 } }), {}).result;
    expect(out.value).toBe(42);
  });

  it("carryCall re-attaches across a rebuild, which spread would otherwise drop", () => {
    const op = resolved(fn("add", { a: { json: 1 }, b: { json: 2 } }));
    expect(isResolvedCall(carryCall(op, { ...op, input: {} } as Operation<InlineFamily>))).toBe(true);
  });

  it("carryCall leaves a rebuild alone when the source was never resolved", () => {
    const op = fn("add", { a: { json: 1 }, b: { json: 2 } });
    expect(isResolvedCall(carryCall(op, { ...op } as Operation<InlineFamily>))).toBe(false);
  });

  it("reports an unknown name at RESOLVE time, naming the input path", () => {
    const out = resolveCalls(fn("double", { n: { op: fn("nope", {}) } }), registry());
    expect("error" in out && out.error).toMatch(/input 'n': no function 'nope' is registered/);
  });

  it("leaves a higher-order edge alone — the definition is the value, not a callee", () => {
    const passed = fn("double", { n: { json: 1 } });
    const op = resolved({
      kind: "function",
      functionRef: "add",
      input: { f: { kind: "function", binding: { op: passed } }, a: { kind: "json", binding: { json: 1 } } },
      output: { name: "value", kind: "json" },
    });
    expect(isResolvedCall((op.input.f?.binding as { op: Operation<InlineFamily> }).op)).toBe(false);
  });

  it("resolves a prompt op's inputs without giving the prompt itself an entry", () => {
    const op = resolved({
      kind: "prompt",
      user: "hi",
      config: {},
      input: { x: { kind: "json", binding: { op: fn("add", { a: { json: 1 }, b: { json: 1 } }) } } },
      output: { name: "value", kind: "json" },
    });
    expect(isResolvedCall(op)).toBe(false);
    expect(isResolvedCall((op.input.x?.binding as { op: Operation<InlineFamily> }).op)).toBe(true);
  });
});

describe("tryRunPureSync — the hot path, with no async machinery at all", () => {
  it("evaluates a pure tree synchronously", () => {
    const out = tryRunPureSync(resolved(fn("double", { n: { op: fn("add", { a: { json: 20 }, b: { json: 1 } }) } })));
    expect(out && isOk(out) && out.value).toBe(42);
  });

  it("is genuinely synchronous — the answer exists before any microtask could run", () => {
    // No `await` anywhere: if the fast path deferred even to a microtask this would be undefined at
    // this statement, which is the property a per-round guard depends on.
    const out = tryRunPureSync(resolved(fn("add", { a: { json: 1 }, b: { json: 2 } })));
    expect(out && isOk(out) ? out.value : undefined).toBe(3);
  });

  it("declines an UNRESOLVED tree, since the fast path exists to avoid lookups", () => {
    expect(tryRunPureSync(fn("add", { a: { json: 1 }, b: { json: 2 } }))).toBeUndefined();
  });

  it("declines a tree containing a non-pure node, having run nothing", () => {
    const op = resolved(fn("double", { n: { op: fn("readFile", {}) } }));
    expect(isPureTree(op)).toBe(false);
    expect(tryRunPureSync(op)).toBeUndefined();
  });

  it("carries a nested failure out whole, keeping its classification", () => {
    const out = tryRunPureSync(resolved(fn("double", { n: { op: fn("boom", {}) } })));
    expect(out && !isOk(out) && out.error.classification).toBe("permanent");
    expect(out && !isOk(out) && out.error.reason).toBe("nope");
  });

  it("classifies a throwing impl rather than letting it escape into the caller's loop", () => {
    const out = tryRunPureSync(resolved(fn("throws", {})));
    expect(out && !isOk(out) && out.error.reason).toMatch(/impl threw/);
  });
});
