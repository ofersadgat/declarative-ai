import { describe, expect, it } from "vitest";
import { evaluateExpression as ev, ExprError, parseExpression, PENDING, referencesOf } from "../src/expr.js";

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
    expect(ev(".x", { x: 7 })).toBe(7);
    expect(ev(".missing", {})).toBe(undefined);
  });
});

describe("property access", () => {
  const ctx = {
    inputs: { plan: "doc", nested: { deep: 5 } },
    outputs: { outcome: "clean", weaknesses: ["a", "b"] },
  };

  it("drills through object graphs", () => {
    expect(ev(".outputs.outcome", ctx)).toBe("clean");
    expect(ev(".inputs.nested.deep", ctx)).toBe(5);
  });

  it("access on undefined/missing/null yields undefined (implicit optional chaining)", () => {
    expect(ev(".children.x.outputs.y", ctx)).toBe(undefined);
    expect(ev(".inputs.gone.deeper.still", ctx)).toBe(undefined);
    expect(ev(".n.prop", { n: null })).toBe(undefined);
    expect(ev(".b.prop", { b: 42 })).toBe(undefined);
  });

  it(".length works for arrays and strings", () => {
    expect(ev(".outputs.weaknesses.length", ctx)).toBe(2);
    expect(ev(".outputs.outcome.length", ctx)).toBe(5);
    expect(ev(".outputs.weaknesses.length > 1", ctx)).toBe(true);
  });

  /**
   * Access is an OWN-property lookup. Native lookup falls through to the prototype, so
   * `constructor`, `__proto__` and every prototype method were readable — and a `{ expr }` binding
   * reading one yielded a FUNCTION as a slot value, in a dataflow that is JSON all the way down.
   * `inferExpr`'s `projectProperty` has always said `.length` is the one property a string or an
   * array exposes; this is the evaluator agreeing with its own type-checker.
   */
  it("reaches nothing off the prototype", () => {
    for (const src of [
      ".inputs.plan.constructor",
      ".inputs.plan.toString",
      ".inputs.nested.constructor",
      ".inputs.nested.hasOwnProperty",
      ".inputs.nested.__proto__",
      ".outputs.weaknesses.constructor",
      ".outputs.weaknesses.map",
      ".outputs.weaknesses.join",
    ]) {
      expect(ev(src, ctx), src).toBe(undefined);
    }
  });
});

describe("operators — JavaScript semantics", () => {
  it("equality (loose and strict)", () => {
    expect(ev("1 == '1'", {})).toBe(true);
    expect(ev("1 === '1'", {})).toBe(false);
    expect(ev("1 != '1'", {})).toBe(false);
    expect(ev("1 !== '1'", {})).toBe(true);
    expect(ev(".x == null", { x: undefined })).toBe(true);
    expect(ev(".x === null", { x: undefined })).toBe(false);
  });

  it("comparison, numeric and string", () => {
    expect(ev("2 < 10", {})).toBe(true);
    expect(ev("'b' > 'a'", {})).toBe(true);
    expect(ev(".run.iteration < .limits.max_iterations", { run: { iteration: 2 }, limits: { max_iterations: 3 } })).toBe(true);
    expect(ev(".x < 1", { x: undefined })).toBe(false); // NaN comparison, as in JS
  });

  it("boolean operators and truthiness", () => {
    expect(ev("!0", {})).toBe(true);
    expect(ev("!'x'", {})).toBe(false);
    expect(ev("1 && 'a'", {})).toBe("a");
    expect(ev("0 || 'fallback'", {})).toBe("fallback");
    expect(ev("'' || 0", {})).toBe(0);
  });

  it("ternary, right-associative", () => {
    expect(ev(".x === 'clean' ? 'complete' : 'blocked'", { x: "clean" })).toBe("complete");
    expect(ev(".a ? 1 : .b ? 2 : 3", { a: false, b: true })).toBe(2);
  });

  it("parentheses and precedence", () => {
    expect(ev("(1 < 2) === true", {})).toBe(true);
    expect(ev("!.a && .b", { a: false, b: true })).toBe(true);
    expect(ev(".a === 1 && .b === 2 || .c === 3", { a: 1, b: 2, c: 0 })).toBe(true);
    expect(ev(".a === 0 && .b === 2 || .c === 3", { a: 1, b: 2, c: 3 })).toBe(true);
  });
});

/**
 * Applying an operation (EXPRESSIONS.md §3). The grammar admits a call; the CALLEE is a reference
 * (a local child key, or one resolved along the `path`), which is why it is kept as a dotted path
 * rather than an arbitrary sub-expression and why it never enters the data scope.
 */
describe("calls", () => {
  const callOf = (src: string) => parseExpression(src) as { type: string; op: string; args: unknown[] };

  it("parses a bare and a dotted callee, with any number of arguments", () => {
    expect(callOf("classify()")).toMatchObject({ type: "apply", op: "classify", args: [] });
    expect(callOf("classify(.inputs.issue)").op).toBe("classify");
    expect(callOf("classify(.inputs.issue)").args).toHaveLength(1);
    expect(callOf("lib.review(a, b, 'c')")).toMatchObject({ type: "apply", op: "lib.review" });
    expect(callOf("lib.review(a, b, 'c')").args).toHaveLength(3);
  });

  it("composes with the rest of the language", () => {
    // A call is an ordinary operand: it nests in operators, conditionals and other calls.
    expect(() => parseExpression("classify(.inputs.issue).severity === 'high'")).not.toThrow();
    expect(() => parseExpression("f(a) && g(b)")).not.toThrow();
    expect(() => parseExpression("f(g(h(1)))")).not.toThrow();
    expect(() => parseExpression("cond ? f(a) : g(b)")).not.toThrow();
    expect(() => parseExpression("f(a > 1 ? 'x' : 'y')")).not.toThrow();
  });

  /**
   * A callee may be a full REFERENCE, not just a bare or dotted name — both spellings of the same
   * operation resolve, one through the search path and one explicitly rooted.
   */
  it("parses a rooted reference as a callee", () => {
    expect(callOf("$JAIRA/prompts/customPrompt('my input')")).toMatchObject({
      type: "apply",
      op: "$JAIRA/prompts/customPrompt",
    });
    expect(callOf("$/functions/classify(a)").op).toBe("$/functions/classify");
    expect(callOf("customPrompt('my input')").op).toBe("customPrompt");
    // A property path after the file part is the reference grammar, kept intact.
    expect(callOf("$/lib/ops.review(a)").op).toBe("$/lib/ops.review");
  });

  it("refuses a rooted path that is not called, and a stray separator", () => {
    // Reading DATA uses the leading-dot runtime form; a file path is only ever an operation.
    expect(() => parseExpression("$JAIRA/prompts/customPrompt")).toThrow(/only meaningful called/);
    expect(() => parseExpression("1 / 2")).toThrow(/only meaningful inside an operation reference/);
  });

  it("refuses to call something that is not a name", () => {
    // An operation is NAMED by a reference, so there is no first-class function value to apply.
    expect(() => parseExpression("(a ? f : g)(x)")).toThrow(/only a name may be called/);
    expect(() => parseExpression("'literal'(x)")).toThrow(/only a name may be called/);
  });

  it("reports a malformed argument list", () => {
    expect(() => parseExpression("f(a,")).toThrow(ExprError);
    expect(() => parseExpression("f(a b)")).toThrow(ExprError);
    expect(() => parseExpression("f(")).toThrow(ExprError);
  });

  /**
   * The callee is a REFERENCE, not a data path. Reporting it as one would make `classify(x)` look
   * like a read of an undeclared namespace called `classify` — and the validator would reject it.
   */
  it("keeps the callee out of the data references, but not the arguments", () => {
    const ast = parseExpression("classify(.children.review.outputs.plan, .inputs.n)");
    expect(referencesOf(ast)).toEqual([
      ["children", "review", "outputs", "plan"],
      ["inputs", "n"],
    ]);
  });

  /**
   * `evaluate` is the reference semantics the lowering is checked against, not a second execution
   * path — applying an operation needs a resolved callee and a scope, so it is refused rather than
   * half-implemented.
   */
  it("cannot be interpreted, and says so", () => {
    expect(() => ev("classify(.inputs.n)", { inputs: { n: 1 } })).toThrow(/only the lowered form can run/);
  });

  /**
   * An operator and a call are the SAME node. `a === b` is an application of `op.strictEq`, and the
   * only thing that differs is the name — and where that name resolves.
   */
  it("is the same node an operator parses to", () => {
    expect(parseExpression("a === b")).toMatchObject({ type: "apply", op: "op.strictEq" });
    expect(parseExpression("!a")).toMatchObject({ type: "apply", op: "op.not" });
    expect(parseExpression("a ? b : c")).toMatchObject({ type: "apply", op: "op.cond" });
    expect(parseExpression("a && b")).toMatchObject({ type: "apply", op: "op.and" });
    expect(parseExpression("classify(a)")).toMatchObject({ type: "apply", op: "classify" });
  });
});

describe("purity — rejected constructs", () => {
  const bad = ["a[0]", "a = 1", "a + b", "a - b", "a * b", "new X", "a; b", "() => 1", "a?.b", "`t`"];
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
    expect(ev(".children.review.outputs.report", ctx)).toBe(PENDING);
    expect(ev(".children.review.outputs", ctx)).toBe(PENDING);
  });

  it("operators touching PENDING yield PENDING", () => {
    expect(ev(".children.review.outputs.report === 'x'", ctx)).toBe(PENDING);
    expect(ev("!.children.review.outputs", ctx)).toBe(PENDING);
    expect(ev(".children.review.outputs.n < 3", ctx)).toBe(PENDING);
    expect(ev(".children.review.outputs.ok ? 1 : 2", ctx)).toBe(PENDING);
  });

  it("short-circuits on determinate values only", () => {
    expect(ev(".flag && .children.review.outputs.ok", ctx)).toBe(false); // false && PENDING
    expect(ev(".truthy || .children.review.outputs.ok", ctx)).toBe(1); // true || PENDING
    expect(ev(".children.review.outputs.ok && .flag", ctx)).toBe(PENDING); // PENDING && x
    expect(ev(".children.review.outputs.ok || .flag", ctx)).toBe(PENDING); // PENDING || x
    expect(ev(".truthy && .children.review.outputs.ok", ctx)).toBe(PENDING); // true && PENDING
  });

  it("resolved children evaluate normally alongside pending ones", () => {
    expect(ev(".children.done.outcome === 'success'", ctx)).toBe(true);
    expect(ev(".children.done.outputs.report", ctx)).toBe("r");
  });
});

describe("referencesOf (static analysis)", () => {
  it("collects root-anchored paths", () => {
    const ast = parseExpression(".children.critique.outputs.outcome === 'clean' && .run.iteration < .limits.max_iterations");
    expect(referencesOf(ast)).toEqual([
      ["children", "critique", "outputs", "outcome"],
      ["run", "iteration"],
      ["limits", "max_iterations"],
    ]);
  });

  it("collects from every branch of ternary and unary", () => {
    const ast = parseExpression("!.a.b ? .c.d : .e");
    expect(referencesOf(ast)).toEqual([["a", "b"], ["c", "d"], ["e"]]);
  });

  it("collects nothing from a bare name, which reads no instance data", () => {
    // A bare path is resolved along the search `path` at load, so it is not a runtime reference and
    // must not be reported as one — the same reason a callee has never been collected.
    expect(referencesOf(parseExpression("!a.b ? c.d : e"))).toEqual([]);
    expect(referencesOf(parseExpression("classify(.inputs.issue)"))).toEqual([["inputs", "issue"]]);
  });
});

describe("spec example expressions evaluate as documented", () => {
  it("§7.3 critique transitions", () => {
    const ctx = {
      outputs: { outcome: "needs_changes" },
      children: { human_review: {}, address_weaknesses: {} },
    };
    expect(ev(".children.human_review.outcome === 'success'", ctx)).toBe(false);
    expect(ev(".outputs.outcome === 'needs_changes'", ctx)).toBe(true);
  });

  it("§9 planning outcome mapping", () => {
    const ctx = { children: { critique: { outputs: { outcome: "clean" } }, context: { outputs: { plan_doc: "p" } } } };
    expect(ev(".children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'", ctx)).toBe("complete");
  });
});
