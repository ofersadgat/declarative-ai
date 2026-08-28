import type { ResolvedValue, FunctionResult } from "@declarative-ai/exec";
import type { WorkflowMetrics } from "../src/ports.js";
/**
 * Binding RESOLUTION (§7.4) — turning an operation's `Parameter` bindings into values.
 *
 * The focus here is the LEAF/NODE discrimination in a ref tree. `RefTree` is an untagged union: a
 * `{ text: "hi" }` leaf and an object node with a property called `text` are the same JavaScript
 * shape, so getting the discriminator wrong silently produces the wrong VALUE — no error, no failed
 * validation, just different data flowing down the graph.
 */
import { describe, expect, it } from "vitest";
import type { InlineFamily, Ref } from "@declarative-ai/exec";
import { PENDING } from "../src/expr.js";
import { isResolveError, isResolvedValue, resolveInputs, resolveRef, type ResolutionScope } from "../src/resolve.js";
import { RESOLVER_REFS } from "../src/format.js";

const scope: ResolutionScope = {
  exprContext: { inputs: { n: 2 } },
  childOutputs: () => undefined,
  scopeValue: (name) => (name === "n" ? 2 : undefined),
  optionalInput: (name) => name === "maybe",
  artifact: () => undefined,
  conversation: () => undefined,
};

const value = (ref: Ref<InlineFamily>) => {
  const r = resolveRef(ref, scope);
  if (!isResolvedValue(r)) throw new Error(`expected a value, got ${r === PENDING ? "PENDING" : JSON.stringify(r)}`);
  return r.value;
};

/** The reason a ref REFUSED to resolve — the other half of a total resolver. */
const error = (ref: Ref<InlineFamily>) => {
  const r = resolveRef(ref, scope);
  if (!isResolveError(r)) throw new Error(`expected an error, got ${r === PENDING ? "PENDING" : JSON.stringify(r)}`);
  return r.error;
};

describe("leaf refs", () => {
  it("resolves each literal leaf to its value", () => {
    expect(value({ text: "hi" })).toBe("hi");
    expect(value({ json: { a: 1 } })).toEqual({ a: 1 });
    const bytes = new Uint8Array([1, 2, 3]);
    expect(value({ blob: bytes })).toBe(bytes);
  });
});

describe("ref TREES — leaf vs object node", () => {
  it("resolves an ordinary object arrangement", () => {
    expect(value({ refs: { a: { text: "x" }, b: { json: 2 } } })).toEqual({ a: "x", b: 2 });
  });

  it("resolves an array arrangement, and nests structurally", () => {
    // Nesting inside a tree is plain structure — `{ refs: … }` is the top-level WRAPPER, not a leaf
    // form, so inside a tree it is just an object key like any other.
    expect(value({ refs: [{ text: "x" }, { inner: { json: 1 } }] })).toEqual(["x", { inner: 1 }]);
  });

  // The bug: testing `"text" in node` alone read this whole node as a TEXT leaf and returned
  // `{ text: "hi" }` — the raw ref object — instead of the record the author described.
  it("treats a MULTI-KEY node containing a leaf keyword as an object, not a leaf", () => {
    expect(value({ refs: { text: { text: "hi" }, body: { text: "world" } } })).toEqual({ text: "hi", body: "world" });
  });

  it("treats a single-key `text` node as an object when its value is not a string", () => {
    // `{ text: <subtree> }` cannot be a text leaf — a text leaf holds a string — so the only reading
    // under which it means anything is "an object with a property called text".
    expect(value({ refs: { text: { text: "hi" } } })).toEqual({ text: "hi" });
  });

  it("still resolves a genuine single-key text leaf inside a tree", () => {
    expect(value({ refs: { a: { text: "hi" } } })).toEqual({ a: "hi" });
  });

  it("treats a single-key `blob` node as an object when its value is not bytes", () => {
    expect(value({ refs: { blob: { text: "not bytes" } } })).toEqual({ blob: "not bytes" });
  });

  it("keeps a genuine blob leaf as bytes inside a tree", () => {
    const bytes = new Uint8Array([9]);
    expect(value({ refs: { file: { blob: bytes } } })).toEqual({ file: bytes });
  });

  it("resolves producer leaves nested in a tree", () => {
    expect(value({ refs: { n: { json: 7 }, label: { text: "seven" } } })).toEqual({ n: 7, label: "seven" });
  });

  // A single-key `result` node was accepted as a leaf on the strength of its value being ANY object,
  // so the resolver then dereferenced a record field that was not there and threw.
  it("treats a single-key `result` node as an object when its value is not an operation record", () => {
    expect(value({ refs: { result: { text: "x" } } })).toEqual({ result: "x" });
  });

  it("still resolves a genuine result leaf — a record carrying its own result envelope", () => {
    const record = { source: "producer", inputs: [], result: { value: "recorded" }, metrics: { durationMs: 1 } };
    expect(value({ refs: { prior: { result: record } } })).toEqual({ prior: "recorded" });
  });
});

/**
 * TOTALITY. Resolution is a pure function over authored data, so no input shape may throw: what a tree
 * cannot mean, it must REFUSE. The walk used to fall through to "recurse structurally" for everything
 * it did not recognize — which turned a string into infinite recursion over its own characters, a
 * number or a boolean into `{}` (wrong data, silently, all the way down the graph), and `null` into a
 * TypeError.
 */
describe("ref TREES — primitives, and what a tree cannot contain", () => {
  it("resolves each primitive in a tree position to itself", () => {
    expect(value({ refs: { greeting: "hi", n: 3, ok: true, nothing: null } })).toEqual({ greeting: "hi", n: 3, ok: true, nothing: null });
  });

  it("resolves a bare primitive tree", () => {
    expect(value({ refs: "hi" })).toBe("hi");
    expect(value({ refs: 3 })).toBe(3);
    expect(value({ refs: null })).toBe(null);
  });

  it("mixes primitives and leaves inside arrays and nested objects", () => {
    expect(value({ refs: [1, "two", { text: "three" }, { deep: { flag: false } }] })).toEqual([1, "two", "three", { deep: { flag: false } }]);
  });

  // A producer edge is a `Ref` case, not a tree leaf: running one is the engine's job at the parameter
  // that binds it. Nested in a tree it is an authoring mistake, and must READ as one.
  it("refuses a producer edge nested in a tree", () => {
    expect(error({ refs: { a: { op: "c" } } })).toMatch(/producer edge .* cannot be nested in a ref tree/);
    expect(error({ refs: { a: { op: "c", parameters: {} } } })).toMatch(/producer edge/);
  });

  // The loader lowers sugar at a parameter's `binding` and does not walk into `refs`, so sugar written
  // inside a tree arrives here unlowered.
  it("refuses unlowered binding sugar nested in a tree, naming the keyword", () => {
    expect(error({ refs: { a: { expr: ".inputs.n" } } })).toMatch(/'expr' binding sugar cannot be nested/);
  });

  // A key that merely SHARES a sugar keyword's name is still an ordinary object property — the same
  // rule the leaf/node discrimination already applies to `text`/`blob`/`result`.
  it("keeps a multi-key node carrying a sugar keyword as an ordinary object", () => {
    expect(value({ refs: { a: { child: "c", note: "not sugar" } } })).toEqual({ a: { child: "c", note: "not sugar" } });
  });

  it("RESOLVES a top-level producer edge — only nesting one in a tree is refused", () => {
    // The edge resolves; it does not refuse. A child that has not run is `undefined` (SPEC §3.4),
    // which is the same answer `{ expr: ".children.c.output" }` gives for the same path.
    const r = resolveRef({ op: "c" }, scope);
    expect(isResolveError(r)).toBe(false);
    expect(isResolvedValue(r) && r.value).toBeUndefined();
  });
});

/**
 * `select` projects a named output off a producer's object output — the lowering of
 * `{ child, output }`. The projection is an OWN-property lookup: an inherited hit used to hand a
 * FUNCTION on as the slot's value instead of reporting an output the producer does not declare.
 */
describe("the select resolver projects own properties only", () => {
  const withChild: ResolutionScope = { ...scope, childOutputs: (key) => (key === "c" ? { plan: "doc" } : undefined) };
  const select = (key: string): Ref<InlineFamily> => ({
    op: {
      kind: "function",
      functionRef: "select",
      input: {
        value: { kind: "json", binding: { op: "c" } },
        key: { kind: "text", binding: { text: key } },
      },
      output: { name: "value", kind: "json" },
    },
  });

  it("projects a declared output", () => {
    const r = resolveRef(select("plan"), withChild);
    expect(isResolvedValue(r) && r.value).toBe("doc");
  });

  it("never returns a prototype member — an inherited name is absent, not a function", () => {
    // OWN properties only. `select` reads as `memberOf` does now, so the answer is `undefined`
    // rather than a refusal — but the hazard this guards is unchanged: a FUNCTION off the
    // prototype must never reach a slot as its value.
    for (const key of ["constructor", "toString", "hasOwnProperty", "__proto__"]) {
      const r = resolveRef(select(key), withChild);
      expect(isResolveError(r), key).toBe(false);
      expect(isResolvedValue(r) && r.value, key).toBeUndefined();
    }
  });

  it("is `undefined`, not a refusal, for an output the run legitimately omitted", () => {
    const r = resolveRef(select("absent_optional_output"), withChild);
    expect(isResolvedValue(r) && r.value).toBeUndefined();
  });
});

/**
 * OPERATOR resolvers (EXPRESSIONS.md §2) — the producer-edge form of the expression language.
 *
 * An expression lowered to a tree of these has to mean exactly what the interpreter meant, which is
 * why the semantics live in `expr.ts` (`applyBinary`, `memberOf`) and are called from both. The
 * cases that would drift silently are the ones pinned here: implicit optional chaining, JavaScript
 * equality, and above all LAZINESS.
 */
describe("operator resolvers", () => {
  const ctx = { inputs: { n: 2, s: "abc", o: { a: 1 }, arr: [1, 2] }, run: { iteration: 1 } };
  const opScope: ResolutionScope = {
    exprContext: ctx,
    childOutputs: (key) => (key === "pending" ? PENDING : undefined),
    scopeValue: () => undefined,
    optionalInput: () => false,
    artifact: () => undefined,
    conversation: () => undefined,
  };

  const lit = (v: unknown): Ref<InlineFamily> => ({ json: v as never });
  const edge = (functionRef: string, args: Record<string, Ref<InlineFamily>>): Ref<InlineFamily> => ({
    op: {
      kind: "function",
      functionRef,
      input: Object.fromEntries(Object.entries(args).map(([k, binding]) => [k, { kind: "json", binding }])),
      output: { name: "value", kind: "json" },
    },
  });
  const val = (ref: Ref<InlineFamily>): unknown => {
    const r = resolveRef(ref, opScope);
    if (!isResolvedValue(r)) throw new Error(`expected a value, got ${r === PENDING ? "PENDING" : JSON.stringify(r)}`);
    return r.value;
  };
  /** A producer edge that always fails — proof that a branch was not evaluated. */
  const exploding = edge("no.such.resolver", {});
  /** A producer edge that resolves to PENDING. */
  const pending: Ref<InlineFamily> = { op: "pending" };

  const ctxRoot = (name: string) => edge(RESOLVER_REFS.context, { name: { text: name } });
  const member = (base: Ref<InlineFamily>, prop: string) => edge(RESOLVER_REFS.member, { value: base, prop: { text: prop } });
  const path = (...segments: string[]): Ref<InlineFamily> =>
    segments.slice(1).reduce((base, seg) => member(base, seg), ctxRoot(segments[0]!));

  it("reads a context root, and an unknown one is undefined rather than an error", () => {
    expect(val(ctxRoot("inputs"))).toEqual(ctx.inputs);
    expect(val(ctxRoot("nope"))).toBeUndefined();
  });

  it("projects properties with implicit optional chaining", () => {
    expect(val(path("inputs", "n"))).toBe(2);
    expect(val(path("inputs", "o", "a"))).toBe(1);
    expect(val(path("inputs", "gone", "deeper"))).toBeUndefined();
    expect(val(path("inputs", "s", "length"))).toBe(3);
    expect(val(path("inputs", "arr", "length"))).toBe(2);
  });

  it("reaches nothing off the prototype, exactly as the interpreter does not", () => {
    for (const prop of ["constructor", "toString", "__proto__", "hasOwnProperty"]) {
      expect(val(member(path("inputs", "o"), prop)), prop).toBeUndefined();
      expect(val(member(path("inputs", "s"), prop)), prop).toBeUndefined();
    }
  });

  it("applies JavaScript comparison semantics", () => {
    expect(val(edge(RESOLVER_REFS.eq, { left: lit(1), right: { text: "1" } }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.strictEq, { left: lit(1), right: { text: "1" } }))).toBe(false);
    expect(val(edge(RESOLVER_REFS.ne, { left: lit(1), right: { text: "1" } }))).toBe(false);
    expect(val(edge(RESOLVER_REFS.strictNe, { left: lit(1), right: { text: "1" } }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.lt, { left: lit(2), right: lit(10) }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.le, { left: lit(2), right: lit(2) }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.gt, { left: { text: "b" }, right: { text: "a" } }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.ge, { left: lit(1), right: lit(2) }))).toBe(false);
    expect(val(edge(RESOLVER_REFS.not, { value: lit(0) }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.gt, { left: path("inputs", "n"), right: lit(1) }))).toBe(true);
  });

  /**
   * The reason `and`/`or`/`cond` cannot be ordinary eager operations (EXPRESSIONS.md §6): resolving
   * every bound parameter would turn `false && PENDING` into PENDING — parking a state that used to
   * make progress — and would RUN the branch a conditional did not take.
   */
  it("short-circuits `and` on a determinate-falsy left, past a pending right", () => {
    expect(val(edge(RESOLVER_REFS.and, { left: lit(false), right: pending }))).toBe(false);
    expect(val(edge(RESOLVER_REFS.and, { left: lit(false), right: exploding }))).toBe(false);
    expect(resolveRef(edge(RESOLVER_REFS.and, { left: lit(true), right: pending }), opScope)).toBe(PENDING);
    expect(val(edge(RESOLVER_REFS.and, { left: lit(1), right: { text: "a" } }))).toBe("a");
  });

  it("short-circuits `or` on a determinate-truthy left", () => {
    expect(val(edge(RESOLVER_REFS.or, { left: lit(true), right: exploding }))).toBe(true);
    expect(val(edge(RESOLVER_REFS.or, { left: lit(0), right: { text: "fallback" } }))).toBe("fallback");
  });

  it("evaluates only the taken branch of `cond`", () => {
    expect(val(edge(RESOLVER_REFS.cond, { test: lit(true), then: { text: "yes" }, else: exploding }))).toBe("yes");
    expect(val(edge(RESOLVER_REFS.cond, { test: lit(false), then: exploding, else: { text: "no" } }))).toBe("no");
    expect(resolveRef(edge(RESOLVER_REFS.cond, { test: pending, then: lit(1), else: lit(2) }), opScope)).toBe(PENDING);
  });

  it("propagates PENDING through an operator", () => {
    expect(resolveRef(edge(RESOLVER_REFS.eq, { left: pending, right: lit(1) }), opScope)).toBe(PENDING);
    expect(resolveRef(member(pending, "x"), opScope)).toBe(PENDING);
    expect(resolveRef(edge(RESOLVER_REFS.not, { value: pending }), opScope)).toBe(PENDING);
  });
});

/**
 * Failures reaching the DATA plane (EXPRESSIONS.md §5).
 *
 * A binding that cannot resolve is a failure, and a failure is a value with a type. If the consuming
 * slot declared that it accepts one, it flows in and the operation runs; otherwise this is the
 * implicit unwrap and the operation terminates with it — which is what every binding got before.
 */
describe("a failure flows into a slot that declares it", () => {
  /** A slot declaring it accepts an error. The value is WRAPPED, so the branch requires `error`. */
  const FAILURE = {
    type: "object",
    properties: {
      error: {
        type: "object",
        properties: { classification: { type: "string" }, reason: { type: "string" } },
        required: ["classification", "reason"],
      },
    },
    required: ["error"],
  };
  /**
   * A binding that genuinely cannot resolve. An unset INPUT, not an unrun child: a child that has
   * not run is `undefined` (SPEC §3.4) and so resolves fine — the strictness there belongs to the
   * consuming slot's `optional`/`default`, not to the resolver. An input slot that nothing set is
   * a real fault, and `scope.get` still says so.
   */
  const unresolvable: Ref<InlineFamily> = {
    op: {
      kind: "function",
      functionRef: RESOLVER_REFS.scope,
      input: {
        scope: { kind: "text", binding: { text: "inputs" } },
        name: { kind: "text", binding: { text: "never_set" } },
      },
      output: { name: "value", kind: "json" },
    },
  } as Ref<InlineFamily>;

  it("binds the failure when the slot accepts one", () => {
    const out = resolveInputs({ report: { kind: "json", schema: FAILURE as never, binding: unresolvable } }, scope);
    expect(out).toHaveProperty("values");
    const value = (out as { values: Record<string, unknown> }).values.report as { error: Record<string, unknown> };
    expect(value.error.classification).toBe("permanent");
    expect(value.error.reason).toMatch(/input 'never_set' is not set/);
  });

  it("terminates when the slot does not — the behaviour every binding had before", () => {
    const out = resolveInputs({ report: { kind: "text", schema: { type: "string" }, binding: unresolvable } }, scope);
    expect(out).toHaveProperty("error");
    expect((out as { error: string }).error).toMatch(/input 'never_set' is not set/);
  });

  it("terminates for an untyped slot, so silence never reads as 'I handle errors here'", () => {
    const out = resolveInputs({ report: { kind: "json", binding: unresolvable } }, scope);
    expect(out).toHaveProperty("error");
  });

  it("leaves a resolvable binding completely alone", () => {
    const out = resolveInputs({ n: { kind: "json", schema: FAILURE as never, binding: { json: 7 } } }, scope);
    expect((out as { values: Record<string, unknown> }).values.n).toBe(7);
  });
});

/**
 * A RECORDED failure keeps its classification through the data plane (EXPRESSIONS.md §5).
 *
 * `{result}` is the one binding whose error already carries a real classification. Flattening it to
 * its reason string — which is what happened before — made a `network-retriable` provider error
 * arrive indistinguishable from a wiring mistake, and anything routing on classification would have
 * seen every failure as the same one.
 */
describe("a recorded failure keeps its shape", () => {
  const failedRecord = (classification: string) => ({
    result: {
      source: "producer",
      inputs: [],
      result: { error: { classification, reason: "provider said no" } },
      metrics: { durationMs: 1 },
    },
  }) as unknown as Ref<InlineFamily>;

  /** A slot that declares it handles exactly one kind. */
  const handles = (kind: string) => ({
    type: "object",
    properties: {
      error: {
        type: "object",
        properties: { classification: { type: "string", enum: [kind] }, reason: { type: "string" } },
        required: ["classification", "reason"],
      },
    },
    required: ["error"],
  });

  it("carries the classification, not just the reason", () => {
    const out = resolveInputs(
      { r: { kind: "json", schema: handles("network-retriable") as never, binding: failedRecord("network-retriable") } },
      scope,
    );
    const value = (out as { values: Record<string, unknown> }).values.r as { error: Record<string, unknown> };
    expect(value.error.classification).toBe("network-retriable");
    expect(value.error.reason).toBe("provider said no");
  });

  it("refuses a failure of a kind the slot did not declare", () => {
    const out = resolveInputs(
      { r: { kind: "json", schema: handles("policy-denied") as never, binding: failedRecord("network-retriable") } },
      scope,
    );
    expect(out).toHaveProperty("error");
  });
});
