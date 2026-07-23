import type { ResolvedValue, FunctionResult } from "@declarative-ai/exec";
import type { WorkflowMetrics } from "../src/ports";
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
import { PENDING } from "../src/expr";
import { isResolveError, isResolvedValue, resolveRef, type ResolutionScope } from "../src/resolve";

const scope: ResolutionScope = {
  exprContext: { inputs: { n: 2 } },
  childOutputs: () => undefined,
  scopeValue: (name) => (name === "n" ? 2 : undefined),
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
    expect(error({ refs: { a: { expr: "inputs.n" } } })).toMatch(/'expr' binding sugar cannot be nested/);
    expect(error({ refs: { a: { child: "c" } } })).toMatch(/'child' binding sugar/);
    expect(error({ refs: { a: { child: "c", output: "o" } } })).toMatch(/'child' binding sugar/);
    expect(error({ refs: { a: { input: "n" } } })).toMatch(/'input' binding sugar/);
    expect(error({ refs: { a: { artifact: "doc" } } })).toMatch(/'artifact' binding sugar/);
    expect(error({ refs: { a: { conversation: "s", message: 0 } } })).toMatch(/'conversation' binding sugar/);
  });

  // A key that merely SHARES a sugar keyword's name is still an ordinary object property — the same
  // rule the leaf/node discrimination already applies to `text`/`blob`/`result`.
  it("keeps a multi-key node carrying a sugar keyword as an ordinary object", () => {
    expect(value({ refs: { a: { child: "c", note: "not sugar" } } })).toEqual({ a: { child: "c", note: "not sugar" } });
  });

  it("resolves a TOP-LEVEL producer edge as before — only nesting one in a tree is refused", () => {
    expect(error({ op: "c" })).toMatch(/child 'c' has not run/);
  });
});
