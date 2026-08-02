/**
 * The built-in operation library (JaiRA EXPRESSIONS.md §3), exercised through the LANGUAGE.
 *
 * Written as expressions rather than as direct calls to the implementations, because what matters is
 * the whole path: parse → lower to a producer edge → resolve. A builtin that worked in isolation but
 * did not lower would pass a unit test and fail a workflow.
 */
import { describe, expect, it } from "vitest";
import { parseExpression, PENDING } from "../src/expr.js";
import { lowerExpression } from "../src/lowerExpr.js";
import { isResolvedValue, resolveRef, type ResolutionScope } from "../src/resolve.js";

const scope: ResolutionScope = {
  exprContext: {
    inputs: {
      n: 7,
      s: "  Hello World  ",
      xs: [3, 1, 2, 3],
      o: { a: 1, b: 2 },
      pairs: [["x", 1], ["y", 2]],
      nothing: null,
    },
  },
  childOutputs: () => undefined,
  scopeValue: () => undefined,
  artifact: () => undefined,
  conversation: () => undefined,
};

/** Evaluate an expression the way a binding would. */
function ev(src: string): unknown {
  const r = resolveRef(lowerExpression(parseExpression(src)), scope);
  if (r === PENDING) return PENDING;
  if (!isResolvedValue(r)) throw new Error(`refused to resolve: ${JSON.stringify(r)}`);
  return r.value;
}

describe("arithmetic — the language had none at all", () => {
  it("computes", () => {
    expect(ev("add(.inputs.n, 3)")).toBe(10);
    expect(ev("sub(10, .inputs.n)")).toBe(3);
    expect(ev("mul(.inputs.n, 2)")).toBe(14);
    expect(ev("div(10, 4)")).toBe(2.5);
    expect(ev("mod(.inputs.n, 4)")).toBe(3);
    expect(ev("min(.inputs.n, 3)")).toBe(3);
    expect(ev("max(.inputs.n, 3)")).toBe(7);
    expect(ev("abs(sub(3, 10))")).toBe(7);
    expect(ev("round(2.6)")).toBe(3);
    expect(ev("floor(2.6)")).toBe(2);
    expect(ev("ceil(2.1)")).toBe(3);
  });

  it("composes with comparison, which is the point of having it", () => {
    expect(ev("add(.inputs.n, 3) > 9")).toBe(true);
  });

  /** TOTAL: a synchronous resolver runs inside the scheduling loop, where a throw has nowhere to go. */
  it("answers rather than throwing on nonsense", () => {
    expect(ev("div(1, 0)")).toBe(Infinity);
    expect(Number.isNaN(ev("add(.inputs.s, 1)") as number)).toBe(true);
    expect(ev("at(.inputs.nothing, 0)")).toBeUndefined();
    expect(ev("get(.inputs.nothing, 'a')")).toBeUndefined();
  });
});

describe("dynamic access — member access takes a LITERAL name", () => {
  it("reads a computed key and an index", () => {
    expect(ev("get(.inputs.o, 'a')")).toBe(1);
    expect(ev("get(.inputs.o, concat('a', ''))")).toBe(1);
    expect(ev("at(.inputs.xs, 1)")).toBe(1);
    expect(ev("at(.inputs.xs, -1)")).toBe(3);
    expect(ev("at(.inputs.xs, 99)")).toBeUndefined();
  });

  it("reaches nothing off the prototype, like member access", () => {
    expect(ev("get(.inputs.o, 'constructor')")).toBeUndefined();
    expect(ev("get(.inputs.o, '__proto__')")).toBeUndefined();
  });
});

describe("negative literals", () => {
  /** No unary minus and no subtraction operator, so a leading `-` can only be a sign. */
  it("parses, which they could not before", () => {
    expect(ev("-1")).toBe(-1);
    expect(ev("at(.inputs.xs, -1)")).toBe(3);
    expect(ev("add(-2, 5)")).toBe(3);
    expect(ev("-1 < 0")).toBe(true);
  });
});

describe("arrays", () => {
  it("queries and transforms without mutating", () => {
    expect(ev("len(.inputs.xs)")).toBe(4);
    expect(ev("first(.inputs.xs)")).toBe(3);
    expect(ev("last(.inputs.xs)")).toBe(3);
    expect(ev("isEmpty(.inputs.xs)")).toBe(false);
    expect(ev("sort(.inputs.xs)")).toEqual([1, 2, 3, 3]);
    expect(ev("unique(.inputs.xs)")).toEqual([3, 1, 2]);
    expect(ev("reverse(.inputs.xs)")).toEqual([3, 2, 1, 3]);
    expect(ev("slice(.inputs.xs, 1, 3)")).toEqual([1, 2]);
    expect(ev("join(.inputs.xs, '-')")).toBe("3-1-2-3");
    expect(ev("range(0, 4)")).toEqual([0, 1, 2, 3]);
  });

  /** `append`, not `push` — every value here is immutable, and the source array is untouched. */
  it("appends by returning a new array", () => {
    expect(ev("append(.inputs.xs, 9)")).toEqual([3, 1, 2, 3, 9]);
    expect(ev(".inputs.xs")).toEqual([3, 1, 2, 3]);
  });

  it("bounds `range`, so a typo cannot exhaust memory", () => {
    expect((ev("range(0, 999999999)") as unknown[]).length).toBe(10_000);
  });
});

describe("strings", () => {
  it("transforms", () => {
    expect(ev("trim(.inputs.s)")).toBe("Hello World");
    expect(ev("lower(trim(.inputs.s))")).toBe("hello world");
    expect(ev("upper('ab')")).toBe("AB");
    expect(ev("split(trim(.inputs.s), ' ')")).toEqual(["Hello", "World"]);
    expect(ev("replace(.inputs.s, 'World', 'There')")).toBe("  Hello There  ");
    expect(ev("concat('a', 'b')")).toBe("ab");
    expect(ev("startsWith(trim(.inputs.s), 'Hello')")).toBe(true);
    expect(ev("endsWith(trim(.inputs.s), 'World')")).toBe(true);
  });

  it("means the obvious thing by `contains` for both strings and arrays", () => {
    expect(ev("contains(.inputs.s, 'World')")).toBe(true);
    expect(ev("contains(.inputs.xs, 2)")).toBe(true);
    expect(ev("contains(.inputs.xs, 9)")).toBe(false);
  });

  it("concatenates arrays when given arrays", () => {
    expect(ev("concat(.inputs.xs, range(0, 2))")).toEqual([3, 1, 2, 3, 0, 1]);
  });
});

describe("objects", () => {
  it("reads and reshapes", () => {
    expect(ev("keys(.inputs.o)")).toEqual(["a", "b"]);
    expect(ev("values(.inputs.o)")).toEqual([1, 2]);
    expect(ev("entries(.inputs.o)")).toEqual([["a", 1], ["b", 2]]);
    expect(ev("fromEntries(.inputs.pairs)")).toEqual({ x: 1, y: 2 });
    expect(ev("merge(.inputs.o, fromEntries(.inputs.pairs))")).toEqual({ a: 1, b: 2, x: 1, y: 2 });
    expect(ev("pick(.inputs.o, append(range(0,0), 'a'))")).toEqual({ a: 1 });
    expect(ev("omit(.inputs.o, append(range(0,0), 'a'))")).toEqual({ b: 2 });
  });

  /** `out[key] = v` invokes the inherited setter for `__proto__` and re-parents the object. */
  it("refuses to let `fromEntries` re-parent its result", () => {
    const built = ev("fromEntries(append(range(0,0), append(append(range(0,0), '__proto__'), 'x')))");
    expect(Object.getPrototypeOf(built as object)).toBe(Object.prototype);
  });
});

describe("json, types and null", () => {
  it("round-trips and classifies", () => {
    expect(ev("parse_json('{\"a\":1}')")).toEqual({ a: 1 });
    expect(ev("to_json(.inputs.o)")).toBe('{"a":1,"b":2}');
    expect(ev("typeof(.inputs.xs)")).toBe("array");
    expect(ev("typeof(.inputs.nothing)")).toBe("null");
    expect(ev("typeof(.inputs.n)")).toBe("number");
    expect(ev("isArray(.inputs.xs)")).toBe(true);
    expect(ev("isNull(.inputs.nothing)")).toBe(true);
    expect(ev("coalesce(.inputs.nothing, 'fallback')")).toBe("fallback");
    expect(ev("coalesce(.inputs.n, 'fallback')")).toBe(7);
  });

  it("returns undefined for malformed JSON rather than throwing", () => {
    expect(ev("parse_json('{')")).toBeUndefined();
  });
});
