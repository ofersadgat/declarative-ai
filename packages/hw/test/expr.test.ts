import { describe, expect, it } from "vitest";
import { evaluateExpression as ev, ExprError, parseExpression, PENDING, referencesOf } from "../src/expr";

describe("expression language — literals and identifiers (SPEC §6)", () => {
  it("parses literals", () => {
    expect(ev("42", {})).toBe(42);
    expect(ev("3.25", {})).toBe(3.25);
    expect(ev("1e3", {})).toBe(1000);
    expect(ev("'significant'", {})).toBe("significant");
    expect(ev('"double"', {})).toBe("double");
    expect(ev("'esc\\'aped'", {})).toBe("esc'aped");
    expect(ev("'a\\nb'", {})).toBe("a\nb");
    expect(ev("true", {})).toBe(true);
    expect(ev("false", {})).toBe(false);
    expect(ev("null", {})).toBe(null);
  });

  it("resolves identifiers from the context; missing → undefined", () => {
    expect(ev("x", { x: 7 })).toBe(7);
    expect(ev("missing", {})).toBe(undefined);
  });
});

describe("property access", () => {
  const ctx = {
    inputs: { plan: "doc", nested: { deep: 5 } },
    outputs: { outcome: "clean", weaknesses: ["a", "b"] },
  };

  it("drills through object graphs", () => {
    expect(ev("outputs.outcome", ctx)).toBe("clean");
    expect(ev("inputs.nested.deep", ctx)).toBe(5);
  });

  it("access on undefined/missing/null yields undefined (implicit optional chaining)", () => {
    expect(ev("children.x.outputs.y", ctx)).toBe(undefined);
    expect(ev("inputs.gone.deeper.still", ctx)).toBe(undefined);
    expect(ev("n.prop", { n: null })).toBe(undefined);
    expect(ev("b.prop", { b: 42 })).toBe(undefined);
  });

  it(".length works for arrays and strings", () => {
    expect(ev("outputs.weaknesses.length", ctx)).toBe(2);
    expect(ev("outputs.outcome.length", ctx)).toBe(5);
    expect(ev("outputs.weaknesses.length > 1", ctx)).toBe(true);
  });
});

describe("operators — JavaScript semantics", () => {
  it("equality (loose and strict)", () => {
    expect(ev("1 == '1'", {})).toBe(true);
    expect(ev("1 === '1'", {})).toBe(false);
    expect(ev("1 != '1'", {})).toBe(false);
    expect(ev("1 !== '1'", {})).toBe(true);
    expect(ev("x == null", { x: undefined })).toBe(true);
    expect(ev("x === null", { x: undefined })).toBe(false);
  });

  it("comparison, numeric and string", () => {
    expect(ev("2 < 10", {})).toBe(true);
    expect(ev("'b' > 'a'", {})).toBe(true);
    expect(ev("run.iteration < limits.max_iterations", { run: { iteration: 2 }, limits: { max_iterations: 3 } })).toBe(true);
    expect(ev("x < 1", { x: undefined })).toBe(false); // NaN comparison, as in JS
  });

  it("boolean operators and truthiness", () => {
    expect(ev("!0", {})).toBe(true);
    expect(ev("!'x'", {})).toBe(false);
    expect(ev("1 && 'a'", {})).toBe("a");
    expect(ev("0 || 'fallback'", {})).toBe("fallback");
    expect(ev("'' || 0", {})).toBe(0);
  });

  it("ternary, right-associative", () => {
    expect(ev("x === 'clean' ? 'complete' : 'blocked'", { x: "clean" })).toBe("complete");
    expect(ev("a ? 1 : b ? 2 : 3", { a: false, b: true })).toBe(2);
  });

  it("parentheses and precedence", () => {
    expect(ev("(1 < 2) === true", {})).toBe(true);
    expect(ev("!a && b", { a: false, b: true })).toBe(true);
    expect(ev("a === 1 && b === 2 || c === 3", { a: 1, b: 2, c: 0 })).toBe(true);
    expect(ev("a === 0 && b === 2 || c === 3", { a: 1, b: 2, c: 3 })).toBe(true);
  });
});

describe("purity — rejected constructs", () => {
  const bad = ["f()", "a[0]", "a = 1", "a + b", "a - b", "a * b", "new X", "a; b", "() => 1", "a?.b", "`t`"];
  for (const src of bad) {
    it(`rejects: ${src}`, () => {
      expect(() => parseExpression(src)).toThrow(ExprError);
    });
  }

  it("rejects trailing input and unterminated strings", () => {
    expect(() => parseExpression("a b")).toThrow(ExprError);
    expect(() => parseExpression("'unterminated")).toThrow(ExprError);
  });
});

describe("PENDING propagation (SPEC §6/§10.4)", () => {
  const ctx = {
    children: { review: { outputs: PENDING }, done: { outputs: { report: "r" }, outcome: "success" } },
    flag: false,
    truthy: 1,
  };

  it("member access through PENDING is PENDING", () => {
    expect(ev("children.review.outputs.report", ctx)).toBe(PENDING);
    expect(ev("children.review.outputs", ctx)).toBe(PENDING);
  });

  it("operators touching PENDING yield PENDING", () => {
    expect(ev("children.review.outputs.report === 'x'", ctx)).toBe(PENDING);
    expect(ev("!children.review.outputs", ctx)).toBe(PENDING);
    expect(ev("children.review.outputs.n < 3", ctx)).toBe(PENDING);
    expect(ev("children.review.outputs.ok ? 1 : 2", ctx)).toBe(PENDING);
  });

  it("short-circuits on determinate values only", () => {
    expect(ev("flag && children.review.outputs.ok", ctx)).toBe(false); // false && PENDING
    expect(ev("truthy || children.review.outputs.ok", ctx)).toBe(1); // true || PENDING
    expect(ev("children.review.outputs.ok && flag", ctx)).toBe(PENDING); // PENDING && x
    expect(ev("children.review.outputs.ok || flag", ctx)).toBe(PENDING); // PENDING || x
    expect(ev("truthy && children.review.outputs.ok", ctx)).toBe(PENDING); // true && PENDING
  });

  it("resolved children evaluate normally alongside pending ones", () => {
    expect(ev("children.done.outcome === 'success'", ctx)).toBe(true);
    expect(ev("children.done.outputs.report", ctx)).toBe("r");
  });
});

describe("referencesOf (static analysis)", () => {
  it("collects root-anchored paths", () => {
    const ast = parseExpression("children.critique.outputs.outcome === 'clean' && run.iteration < limits.max_iterations");
    expect(referencesOf(ast)).toEqual([
      ["children", "critique", "outputs", "outcome"],
      ["run", "iteration"],
      ["limits", "max_iterations"],
    ]);
  });

  it("collects from every branch of ternary and unary", () => {
    const ast = parseExpression("!a.b ? c.d : e");
    expect(referencesOf(ast)).toEqual([["a", "b"], ["c", "d"], ["e"]]);
  });
});

describe("spec example expressions evaluate as documented", () => {
  it("§7.3 critique transitions", () => {
    const ctx = {
      outputs: { outcome: "needs_changes" },
      children: { human_review: {}, address_weaknesses: {} },
    };
    expect(ev("children.human_review.outcome === 'success'", ctx)).toBe(false);
    expect(ev("outputs.outcome === 'needs_changes'", ctx)).toBe(true);
  });

  it("§9 planning outcome mapping", () => {
    const ctx = { children: { critique: { outputs: { outcome: "clean" } }, context: { outputs: { plan_doc: "p" } } } };
    expect(ev("children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'", ctx)).toBe("complete");
  });
});
