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
    expect(ev(".children.x.output.y", ctx)).toBe(undefined);
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
   * Bracket indexing is SUGAR for `at`: the two spellings parse to the same AST, so there is one
   * semantics — negative indices count from the end because `at`'s already do — and nothing
   * downstream (lowering, inference, static analysis) can treat them differently.
   */
  it("parses `xs[i]` to exactly the AST `at(xs, i)` parses to", () => {
    expect(parseExpression(".inputs.xs[-1]")).toEqual(parseExpression("at(.inputs.xs, -1)"));
    expect(parseExpression(".inputs.xs[0]")).toEqual(parseExpression("at(.inputs.xs, 0)"));
    // A computed index is an ordinary expression…
    expect(parseExpression(".inputs.xs[.inputs.i]")).toEqual(parseExpression("at(.inputs.xs, .inputs.i)"));
    // …indexing chains like any other postfix, on either side…
    expect(parseExpression(".inputs.rows[-1].name")).toEqual(parseExpression("at(.inputs.rows, -1).name"));
    expect(parseExpression("messages(.inputs.s)[-1]")).toEqual(parseExpression("at(messages(.inputs.s), -1)"));
    expect(parseExpression(".inputs.grid[0][1]")).toEqual(parseExpression("at(at(.inputs.grid, 0), 1)"));
    // …and composes with operators.
    expect(parseExpression(".inputs.xs[-1] === 3")).toEqual(parseExpression("at(.inputs.xs, -1) === 3"));
  });

  it("refuses a dangling or misplaced bracket with a real message", () => {
    expect(() => parseExpression(".inputs.xs[-1")).toThrow(ExprError);
    expect(() => parseExpression(".inputs.xs[]")).toThrow(ExprError);
    // An operation reference is only meaningful called (its own rule), never indexed.
    expect(() => parseExpression("$JAIRA/prompts/review[-1]")).toThrow(/cannot be indexed/);
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

  it("reads a rooted path that is not called as a document", () => {
    // An uncalled reference is a DOCUMENT, exactly as a single-segment bare name is: both are
    // `ident`, both lower through `resolveName`, and what comes back may be the file's text, its
    // value, or an operation (the higher-order value of §3.1). This used to throw "only meaningful
    // called", which held only because a `/`-path had nowhere to live in the AST — `reference` is a
    // local string in `member()` and the `(` branch was its one consumer.
    expect(parseExpression("$JAIRA/prompts/customPrompt")).toMatchObject({
      type: "ident",
      name: "$JAIRA/prompts/customPrompt",
    });
    // Which is what lets a template be an ARGUMENT rather than only a callee.
    expect(parseExpression("renderTemplate($/prompts/turns/revise.md, { n: 1 })")).toMatchObject({
      type: "apply",
      op: "renderTemplate",
      args: [{ type: "ident", name: "$/prompts/turns/revise.md" }, { type: "object" }],
    });
    // `1 / 2` is division: no `$` root to the left of the slash.
    expect(parseExpression("1 / 2")).toMatchObject({ type: "apply", op: "div" });
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
    const ast = parseExpression("classify(.children.review.output.plan, .inputs.n)");
    expect(referencesOf(ast)).toEqual([
      ["children", "review", "output", "plan"],
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

/**
 * NAMED arguments, spelled as a spread (SPEC §6.3).
 *
 * The point of the form is that a name reaches a slot positions cannot — so what these pin is that
 * the name survives parsing as a name, and that the two forms compose in one argument list rather
 * than being two calling conventions that happen to share a syntax.
 */
describe("spread arguments", () => {
  const callOf = (src: string) => parseExpression(src) as { type: string; op: string; args: unknown[] };

  it("parses a spread as its own kind of argument, mixed freely with positions", () => {
    expect(callOf("f(...{ mode: 'plan' })").args).toEqual([
      { type: "spread", value: { type: "object", entries: [{ key: "mode", value: { type: "lit", value: "plan" } }] } },
    ]);
    expect(callOf("f(a, ...{ b: 1 }, c)").args).toHaveLength(3);
    expect(callOf("f(a, ...{ b: 1 }, c)").args.map((a) => (a as { type: string }).type)).toEqual([
      "ident",
      "spread",
      "ident",
    ]);
    // The operand is a full expression, not a literal-only form.
    expect(callOf("f(...opts)").args).toEqual([{ type: "spread", value: { type: "ident", name: "opts" } }]);
    expect(callOf("f(...g(1))").args).toEqual([
      { type: "spread", value: { type: "apply", op: "g", args: [{ type: "lit", value: 1 }] } },
    ]);
  });

  it("lexes '...' as one token rather than three property accesses", () => {
    // The failure this guards is silent: three '.' tokens would parse as member access on nothing
    // and produce a baffling message about a property name.
    expect(() => parseExpression("f(...{ a: 1 })")).not.toThrow();
    expect(parseExpression(".inputs.a")).toEqual(parseExpression(".inputs.a"));
    expect(() => parseExpression("f(..)")).toThrow(ExprError);
  });

  it("binds a built-in's parameters by name, and refuses a name it has none for", () => {
    // `at(value, index)` — the same call, twice, reaching the same two slots two ways.
    expect(ev("at(.xs, -1)", { xs: ["a", "b", "c"] })).toBe("c");
    expect(ev("at(...{ value: .xs, index: -1 })", { xs: ["a", "b", "c"] })).toBe("c");
    expect(ev("at(.xs, ...{ index: 0 })", { xs: ["a", "b", "c"] })).toBe("a");
    expect(() => ev("at(...{ nope: 1 })", {})).toThrow(/no parameter 'nope'/);
  });

  it("refuses an operand that names no arguments", () => {
    expect(() => ev("at(...3)", {})).toThrow(/names no arguments/);
    expect(() => ev("at(....xs)", { xs: [1, 2] })).toThrow(/names no arguments/);
  });

  it("reports a reference inside a spread operand, so the read is not hidden", () => {
    expect(referencesOf(parseExpression("f(...{ a: .inputs.x })"))).toEqual([["inputs", "x"]]);
    expect(referencesOf(parseExpression("f(...opts.bag)"))).toEqual([]);
    expect(referencesOf(parseExpression("f(....inputs.bag)"))).toEqual([["inputs", "bag"]]);
  });
});

describe("arithmetic", () => {
  it("is sugar for the built-ins, not a node of its own", () => {
    expect(parseExpression("a + b")).toMatchObject({ type: "apply", op: "add" });
    expect(parseExpression("a - b")).toMatchObject({ type: "apply", op: "sub" });
    expect(parseExpression("a * b")).toMatchObject({ type: "apply", op: "mul" });
    expect(parseExpression(".a / .b")).toMatchObject({ type: "apply", op: "div" });
  });

  it("binds tighter than comparison and looser than member access", () => {
    expect(ev("1 + 2 * 3", {})).toBe(7);
    expect(ev("(1 + 2) * 3", {})).toBe(9);
    expect(ev("1 + 2 < 4", {})).toBe(true);
    expect(ev(".n.v * 2", { n: { v: 4 } })).toBe(8);
  });

  it("keeps a leading minus a SIGN where no value precedes it", () => {
    // The case the sign rule exists for: indexing from the end.
    expect(ev("at(.xs, -1)", { xs: [1, 2, 3] })).toBe(3);
    expect(ev(".xs[-1]", { xs: [1, 2, 3] })).toBe(3);
    expect(ev("-2 + 5", {})).toBe(3);
    // …and a subtraction where one does.
    expect(ev(".n - 1", { n: 10 })).toBe(9);
    expect(ev(".n -1", { n: 10 })).toBe(9);
    expect(ev("(4) - 1", {})).toBe(3);
  });

  it("negates in prefix position", () => {
    expect(ev("-.n", { n: 4 })).toBe(-4);
  });

  /**
   * `/` is both the reference separator and division, and what decides is whether a NAME sits to its
   * left. A bare dotted path is a document; everything else is a value.
   */
  it("divides unless a `$` root opened the path", () => {
    expect(ev(".total / 2", { total: 9 })).toBe(4.5);
    expect(ev("len(.xs) / 2", { xs: [1, 2, 3, 4] })).toBe(2);
    expect(ev("6 / 3", {})).toBe(2);
    // A `$` ROOT is what makes a slash a path separator, so a rooted callee still parses whole…
    expect(parseExpression("$/functions/classify(.x)")).toMatchObject({ type: "apply", op: "$/functions/classify" });
    expect(parseExpression("$JAIRA/lib/review(.x)")).toMatchObject({ type: "apply", op: "$JAIRA/lib/review" });
    // …while two bare names divide, which is what the old "is there a NAME to the left" rule cost.
    // The searched multi-segment spelling (`lib/review`) is gone with it; a SINGLE-segment bare name
    // is untouched and is still a callee, a document, and what `map(xs, classify)` passes.
    // Parse shape only: a bare name is a DOCUMENT, so the interpreter refuses to evaluate one —
    // resolving it is the lowered form's job. What matters here is that the `/` is division.
    expect(parseExpression("lib / review")).toMatchObject({
      type: "apply",
      op: "div",
      args: [{ type: "ident", name: "lib" }, { type: "ident", name: "review" }],
    });
    expect(parseExpression("classify(.x)")).toMatchObject({ type: "apply", op: "classify" });
  });

  it("aggregates over a list", () => {
    const ctx = { scores: [{ total: 3 }, { total: 9 }, { total: 5 }], w: [0.5, 0.5], s: [4, 8] };
    expect(ev("sum(pluck(.scores, 'total'))", ctx)).toBe(17);
    expect(ev("avg(pluck(.scores, 'total'))", ctx)).toBeCloseTo(17 / 3);
    expect(ev("maxBy(.scores, 'total').total", ctx)).toBe(9);
    expect(ev("minBy(.scores, 'total').total", ctx)).toBe(3);
    expect(ev("at(sortBy(.scores, 'total'), -1).total", ctx)).toBe(9);
    expect(ev("find(.scores, 'total', 5).total", ctx)).toBe(5);
    expect(ev("dot(.w, .s)", ctx)).toBe(6);
    expect(ev("all(.flags)", { flags: [true, true] })).toBe(true);
    expect(ev("any(.flags)", { flags: [false, false] })).toBe(false);
    // Empty averages to 0 rather than NaN: every comparison against NaN is false, which would make a
    // score guard silently pass.
    expect(ev("avg(.none)", { none: [] })).toBe(0);
  });

  it("reads a key an identifier cannot spell", () => {
    expect(ev('.cfg."claude-cli"', { cfg: { "claude-cli": 7 } })).toBe(7);
    expect(ev(".cfg.'a.b'", { cfg: { "a.b": 1 } })).toBe(1);
    expect(parseExpression('.cfg."claude-cli"')).toMatchObject({ type: "member", prop: "claude-cli" });
  });
});

describe("purity — rejected constructs", () => {
  // `a[0]` left this list when bracket indexing became sugar for `at`, and `a + b` / `a - b` /
  // `a * b` left it when arithmetic became sugar for `add` / `sub` / `mul` — see the block above.
  const bad = ["a = 1", "new X", "a; b", "() => 1", "a?.b", "`t`"];
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
    children: { review: { output: PENDING }, done: { output: { report: "r" }, outcome: "success" } },
    flag: false,
    truthy: 1,
  };

  it("member access through PENDING is PENDING", () => {
    expect(ev(".children.review.output.report", ctx)).toBe(PENDING);
    expect(ev(".children.review.output", ctx)).toBe(PENDING);
  });

  it("operators touching PENDING yield PENDING", () => {
    expect(ev(".children.review.output.report === 'x'", ctx)).toBe(PENDING);
    expect(ev("!.children.review.output", ctx)).toBe(PENDING);
    expect(ev(".children.review.output.n < 3", ctx)).toBe(PENDING);
    expect(ev(".children.review.output.ok ? 1 : 2", ctx)).toBe(PENDING);
  });

  it("short-circuits on determinate values only", () => {
    expect(ev(".flag && .children.review.output.ok", ctx)).toBe(false); // false && PENDING
    expect(ev(".truthy || .children.review.output.ok", ctx)).toBe(1); // true || PENDING
    expect(ev(".children.review.output.ok && .flag", ctx)).toBe(PENDING); // PENDING && x
    expect(ev(".children.review.output.ok || .flag", ctx)).toBe(PENDING); // PENDING || x
    expect(ev(".truthy && .children.review.output.ok", ctx)).toBe(PENDING); // true && PENDING
  });

  it("resolved children evaluate normally alongside pending ones", () => {
    expect(ev(".children.done.outcome === 'success'", ctx)).toBe(true);
    expect(ev(".children.done.output.report", ctx)).toBe("r");
  });
});

describe("referencesOf (static analysis)", () => {
  it("collects root-anchored paths", () => {
    const ast = parseExpression(".children.critique.output.outcome === 'clean' && .run.iteration < .limits.max_iterations");
    expect(referencesOf(ast)).toEqual([
      ["children", "critique", "output", "outcome"],
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
    const ctx = { children: { critique: { output: { outcome: "clean" } }, context: { output: { plan_doc: "p" } } } };
    expect(ev(".children.critique.output.outcome === 'clean' ? 'complete' : 'blocked'", ctx)).toBe("complete");
  });
});
