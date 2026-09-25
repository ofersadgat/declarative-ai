/**
 * `$ref` EXPRESSIONS — a `$ref` that is not a path is evaluated at LOAD over the documents it names
 * (SPEC §5.4, JaiRA decision 0010 §5.5).
 *
 * What the feature is, one test each:
 *
 *  - arrow lambdas parse (`(t) => …`, `t => …`, `(a, b) => …`, `() => …`) without moving any
 *    existing parse — a parenthesized expression is still grouping;
 *  - an array literal parses and means the same at run time (lowered) as at load;
 *  - the load evaluator reads a reference spelling as the value it names, a lambda parameter as a
 *    value (a parameter shadows a document), and runs `filter`/`map`/… with a lambda;
 *  - `{ "$ref": "...x" }` as an ITEM splices a list into the list it stands in, and the decision's
 *    own example loads end to end through `loadBundle`;
 *  - a failing expression, or one producing the wrong shape for its position, is a load error that
 *    says so.
 */
import { describe, expect, it } from "vitest";
import { evaluate, parseExpression, type Expr } from "../src/expr.js";
import { evaluateLoadExpression } from "../src/loadExpr.js";
import { lowerExpression } from "../src/lowerExpr.js";
import { loadBundle, WorkflowLoadError } from "../src/loader.js";
import { isResolvedValue, resolveRef, type ResolutionScope } from "../src/resolve.js";
import { type Vfs } from "../src/reference.js";
import { validateBundle } from "../src/validate.js";

const PROJECT = "/p/.jaira";
const SHARED = "/s/.jaira";

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        names.add(rest.split("/")[0]!);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

const opts = (files: Record<string, string>) => ({
  defaultRoot: `${PROJECT}/workflows`,
  roots: { JAIRA: PROJECT, BASE: SHARED },
  vfs: vfsOf(files),
});

describe("the grammar: lambdas and array literals", () => {
  it("parses the lambda spellings", () => {
    expect(parseExpression("(t) => t.name")).toEqual({
      type: "lambda",
      params: ["t"],
      body: { type: "member", obj: { type: "ident", name: "t" }, prop: "name" },
    });
    expect(parseExpression("t => t")).toMatchObject({ type: "lambda", params: ["t"] });
    expect(parseExpression("(a, b) => a")).toMatchObject({ type: "lambda", params: ["a", "b"] });
    expect(parseExpression("() => 1")).toMatchObject({ type: "lambda", params: [], body: { type: "lit", value: 1 } });
    // The body reaches as far right as it can, as in JavaScript.
    expect(parseExpression("t => t.a ? 1 : 2")).toMatchObject({ type: "lambda", body: { type: "apply" } });
  });

  it("leaves a parenthesized expression the grouping it always was", () => {
    expect(parseExpression("(a) + 1")).toEqual(parseExpression("a + 1"));
    expect(parseExpression("(.inputs.n)")).toEqual(parseExpression(".inputs.n"));
    expect(() => parseExpression("(a, b) + 1")).toThrow();
  });

  it("refuses a duplicate parameter", () => {
    expect(() => parseExpression("(t, t) => t")).toThrow(/duplicate lambda parameter 't'/);
  });

  it("parses an array literal, and indexing is still indexing", () => {
    expect(parseExpression("['a', 1]")).toEqual({ type: "array", items: [{ type: "lit", value: "a" }, { type: "lit", value: 1 }] });
    expect(parseExpression("[]")).toEqual({ type: "array", items: [] });
    expect(parseExpression(".inputs.xs[0]")).toMatchObject({ type: "apply", op: "at" });
  });

  it("gives an array literal one meaning — interpreted, and lowered and resolved", () => {
    const context = { inputs: { n: 2 } };
    expect(evaluate(parseExpression("[.inputs.n, 'x', [1]]"), context)).toEqual([2, "x", [1]]);
    const scope: ResolutionScope = {
      exprContext: context,
      childOutputs: () => undefined,
      scopeValue: () => undefined,
      optionalInput: () => false,
      artifact: () => undefined,
      conversation: () => undefined,
    };
    const r = resolveRef(lowerExpression(parseExpression("[.inputs.n, 'x', [1]]")), scope);
    expect(isResolvedValue(r) && r.value).toEqual([2, "x", [1]]);
    // `includes` is JavaScript's spelling of `contains`, at run time too.
    const has = resolveRef(lowerExpression(parseExpression("['a', 'b'].includes('b')")), scope);
    expect(isResolvedValue(has) && has.value).toBe(true);
  });

  it("refuses a lambda at run time, naming where one runs", () => {
    expect(() => lowerExpression(parseExpression("filter(.inputs.xs, (x) => x)"))).toThrow(/load-time '\$ref' expression/);
    expect(() => evaluate(parseExpression("(x) => x") as Expr, {})).toThrow(/load-time/);
  });
});

describe("the load evaluator", () => {
  const docs: Record<string, unknown> = {
    "$BASE/wf.transitions": [
      { name: "a", to: "x" },
      { name: "b", to: "y" },
      { name: "c", to: "z" },
    ],
    "lib.ignored": ["c"],
  };
  const run = (src: string): unknown =>
    evaluateLoadExpression(src, {
      reference: (s) => {
        if (!(s in docs)) throw new Error(`no '${s}'`);
        return docs[s];
      },
    });

  it("filters a referenced list with a lambda — the decision's own spelling", () => {
    expect(run("filter($BASE/wf.transitions, (t) => !['b'].includes(t.name))")).toEqual([
      { name: "a", to: "x" },
      { name: "c", to: "z" },
    ]);
  });

  it("reads a dotted bare path as ONE reference, and a parameter's path as properties", () => {
    expect(run("filter($BASE/wf.transitions, t => !lib.ignored.includes(t.name))")).toHaveLength(2);
    expect(run("map($BASE/wf.transitions, (t) => t.to)")).toEqual(["x", "y", "z"]);
  });

  it("takes the receiver spelling, including on a parameter's own path", () => {
    expect(run("$BASE/wf.transitions.filter(t => t.name.startsWith('a'))")).toEqual([{ name: "a", to: "x" }]);
    expect(run("$BASE/wf.transitions.map(t => t.name).join('')")).toBe("abc");
  });

  it("runs the rest of the family", () => {
    expect(run("find($BASE/wf.transitions, t => t.to === 'y').name")).toBe("b");
    expect(run("some($BASE/wf.transitions, t => t.name === 'c')")).toBe(true);
    expect(run("every($BASE/wf.transitions, t => len(t.name) === 1)")).toBe(true);
    expect(run("reduce(['x', 'y'], (acc, s) => acc + len(s), 0)")).toBe(2);
    expect(run("flatMap([1, 2], (n) => [n, n])")).toEqual([1, 1, 2, 2]);
    expect(run("concat($BASE/wf.transitions, [{ name: 'd', to: 'w' }]).length")).toBe(4);
  });

  it("closes over an outer parameter", () => {
    expect(run("map([1, 2], (a) => map([10], (b) => a + b))")).toEqual([[11], [12]]);
  });

  it("refuses what it cannot answer, saying what", () => {
    expect(() => run(".inputs.x")).toThrow(/no instance/);
    expect(() => run("classify($BASE/wf.transitions)")).toThrow(/not something a load-time expression can run/);
    expect(() => run("filter($BASE/wf.transitions, 'name')")).toThrow(/takes a lambda/);
    expect(() => run("filter('abc', (c) => true)")).toThrow(/takes a list/);
  });
});

describe("`$ref` expressions in a document", () => {
  const shared = {
    transitions: [
      { name: "push_main", when: "true", to: "build" },
      { name: "nightly", when: "true", to: "report" },
    ],
  };
  const project = {
    children: { release: { state: "leaf" }, build: { state: "leaf" }, report: { state: "leaf" } },
    sequence: [],
    transitions: [
      { name: "release_push", when: "true", to: "release" },
      { $ref: "...filter($BASE/workflows/system/events.transitions, (t) => !['push_main'].includes(t.name))" },
    ],
  };
  const files = {
    [`${SHARED}/workflows/system/events.json`]: JSON.stringify(shared),
    [`${PROJECT}/workflows/system/events.json`]: JSON.stringify(project),
    [`${PROJECT}/workflows/leaf.json`]: "{}",
  };

  it("splices the base layer's lines, minus the ignored one, after the project's own — end to end", () => {
    const bundle = loadBundle({ "system/events.json": project, "leaf.json": {} }, "system/events", opts(files));
    const transitions = bundle.states["system/events"]!.transitions!;
    expect(transitions.map((t) => t.name)).toEqual(["release_push", "nightly"]);
    expect(transitions.map((t) => t.to)).toEqual(["release", "report"]);
    // Every spliced line is a rule like any other: lowered, and clean.
    expect(transitions.every((t) => t.whenRef !== undefined)).toBe(true);
    expect(validateBundle(bundle).errors).toEqual([]);
  });

  it("splices a plain path, and an expression standing for the whole list", () => {
    const def = { ...project, transitions: [{ $ref: "...$BASE/workflows/system/events.transitions" }, { name: "last", when: "true", to: "release" }] };
    const names = loadBundle({ "system/events.json": def, "leaf.json": {} }, "system/events", opts(files)).states["system/events"]!.transitions!.map((t) => t.name);
    expect(names).toEqual(["push_main", "nightly", "last"]);

    const whole = { ...project, transitions: { $ref: "filter($BASE/workflows/system/events.transitions, t => t.name === 'nightly')" } };
    const only = loadBundle({ "system/events.json": whole, "leaf.json": {} }, "system/events", opts(files)).states["system/events"]!.transitions!;
    expect(only.map((t) => t.name)).toEqual(["nightly"]);
  });

  it("expands what a referenced list holds in its own file's scope", () => {
    // The base layer's line itself splices a fragment that lives beside it.
    const layered = {
      [`${SHARED}/workflows/system/events.json`]: JSON.stringify({
        transitions: [{ $ref: "...$BASE/workflows/system/extra.transitions" }, { name: "own", when: "true", to: "build" }],
      }),
      [`${SHARED}/workflows/system/extra.json`]: JSON.stringify({ transitions: [{ name: "extra", when: "true", to: "report" }] }),
      [`${PROJECT}/workflows/leaf.json`]: "{}",
    };
    const def = { ...project, transitions: [{ $ref: "...$BASE/workflows/system/events.transitions" }] };
    const names = loadBundle({ "system/events.json": def, "leaf.json": {} }, "system/events", opts(layered)).states["system/events"]!.transitions!.map((t) => t.name);
    expect(names).toEqual(["extra", "own"]);
  });

  const refused = (transitions: unknown): string => {
    try {
      loadBundle({ "system/events.json": { ...project, transitions }, "leaf.json": {} }, "system/events", opts(files));
    } catch (e) {
      expect(e).toBeInstanceOf(WorkflowLoadError);
      return (e as Error).message;
    }
    throw new Error("loaded");
  };

  it("refuses an expression that fails, naming it", () => {
    expect(refused([{ $ref: "...filter($BASE/workflows/system/nope.transitions, t => true)" }])).toMatch(/could not be evaluated/);
    expect(refused([{ $ref: "...filter(" }])).toMatch(/neither a path nor an expression that parses/);
  });

  it("refuses a result of the wrong shape for its position", () => {
    expect(refused([{ $ref: "...len($BASE/workflows/system/events.transitions)" }])).toMatch(/must produce a list here, but produced a number/);
    expect(refused({ $ref: "len($BASE/workflows/system/events.transitions)" })).toMatch(/must produce a list here, but produced a number/);
    expect(refused([{ $ref: "...map([1], x => (y) => y)" }])).toMatch(/lambda/);
  });

  it("refuses a spread that is not an item of a list, and one with siblings", () => {
    expect(refused({ $ref: "...$BASE/workflows/system/events.transitions" })).toMatch(/not an item of a list/);
    expect(refused([{ $ref: "...$BASE/workflows/system/events.transitions", name: "x" }])).toMatch(/no sibling keys/);
  });
});
