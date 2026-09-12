/**
 * Expression LOWERING (EXPRESSIONS.md §1), pinned differentially.
 *
 * The lowered tree has to mean exactly what the interpreter meant — for every expression, against
 * every context, including the awkward ones (a child that has not started, a child still running, a
 * missing namespace). Comparing values is the only check that can prove that; comparing structure
 * would just restate the lowering.
 */
import { describe, expect, it } from "vitest";
import type { InlineFamily, Ref } from "@declarative-ai/exec";
import { evaluate, parseExpression, PENDING } from "../src/expr.js";
import { lowerExpression, pathOfRef } from "../src/lowerExpr.js";
import { inferExpression, inferRef } from "../src/inferExpr.js";
import { isResolvedValue, resolveRef, type ResolutionScope } from "../src/resolve.js";

/** Every context an expression below is evaluated against — the awkward cases on purpose. */
const CONTEXTS: Record<string, Record<string, unknown>> = {
  populated: {
    inputs: { issue: "significant", n: 2, flag: false, list: ["a", "b"], nested: { deep: 5 } },
    outputs: { severity: "high", count: 0 },
    children: {
      done: { outputs: { plan: "# Plan", n: 3 }, outcome: "success" },
      failed: { outputs: {}, outcome: "error" },
      running: { output: PENDING, outcome: PENDING },
      unstarted: {},
    },
    run: { iteration: 2, cursor: "done", position: 1 },
    limits: { max_iterations: 3 },
    artifacts: {},
    conversations: {},
  },
  empty: {},
};

const EXPRESSIONS = [
  // literals and identifiers
  "42", "'significant'", "true", "null", "3.25",
  ".inputs", ".missing",
  // property access, including the optional-chaining cases
  ".inputs.issue", ".inputs.nested.deep", ".inputs.gone.deeper.still",
  ".outputs.severity", ".outputs.count",
  ".inputs.list.length", ".inputs.issue.length",
  // prototype must stay unreachable through both paths
  ".inputs.issue.constructor", ".inputs.nested.toString", ".inputs.list.map",
  // children, in all four shapes §1.3 distinguishes
  ".children.done.output.plan", ".children.done.output", ".children.done", ".children.done.outcome",
  ".children.unstarted.output.plan", ".children.unstarted.outcome",
  ".children.failed.outcome", ".children.missing.output.x",
  // operators
  ".inputs.n === 2", ".inputs.n == '2'", ".inputs.n !== 2", ".inputs.n != '2'",
  ".inputs.n < 3", ".inputs.n <= 2", ".inputs.n > 3", ".inputs.n >= 2",
  "!.inputs.flag", "!!.inputs.issue",
  ".run.iteration < .limits.max_iterations",
  // truthiness and short-circuit results
  ".inputs.flag && .inputs.issue", ".inputs.flag || .inputs.issue", ".outputs.count || 'fallback'",
  ".inputs.issue && .outputs.severity",
  // ternary
  ".outputs.severity === 'high' ? 'escalate' : 'continue'",
  ".inputs.flag ? 1 : .inputs.n > 1 ? 2 : 3",
  // combinations
  ".run.cursor === 'done' && .outputs.severity === 'high'",
  ".children.done.outcome === 'success' && .children.done.output.n > 2",
];

function scopeFor(context: Record<string, unknown>): ResolutionScope {
  return {
    exprContext: context,
    childOutputs: () => undefined,
    scopeValue: () => undefined,
    optionalInput: () => false,
    artifact: () => undefined,
    conversation: () => undefined,
  };
}

/** What the lowered tree produces, normalized onto the interpreter's vocabulary. */
function resolved(ref: Ref<InlineFamily>, context: Record<string, unknown>): unknown {
  const r = resolveRef(ref, scopeFor(context));
  if (r === PENDING) return PENDING;
  if (!isResolvedValue(r)) throw new Error(`lowered tree refused to resolve: ${JSON.stringify(r)}`);
  return r.value;
}

describe("lowering preserves the interpreter's semantics", () => {
  for (const [name, context] of Object.entries(CONTEXTS)) {
    describe(`against the ${name} context`, () => {
      for (const src of EXPRESSIONS) {
        it(`${src}`, () => {
          const ast = parseExpression(src);
          const interpreted = evaluate(ast, context);
          const lowered = resolved(lowerExpression(ast), context);
          expect(lowered, src).toEqual(interpreted);
        });
      }
    });
  }
});

describe("PENDING propagates through a lowered tree exactly as it does through the interpreter", () => {
  const ctx = CONTEXTS.populated!;
  for (const src of [
    ".children.running.output",
    ".children.running.output.plan",
    ".children.running.outcome === 'success'",
    "!.children.running.outcome",
    // The short-circuit cases: a determinate side decides, past a pending one.
    "false && .children.running.outcome",
    "true || .children.running.outcome",
    "true && .children.running.outcome",
    ".children.running.outcome ? 'a' : 'b'",
  ]) {
    it(`${src}`, () => {
      const ast = parseExpression(src);
      expect(resolved(lowerExpression(ast), ctx), src).toEqual(evaluate(ast, ctx));
    });
  }
});

describe("messages() — a conversation is read by REF, and its turns are typed", () => {
  const scope = { inputs: { type: "object", properties: { s: { type: "object", properties: { id: { type: "string" } } } } } } as never;
  const infer = (src: string): ReturnType<typeof inferExpression> => inferExpression(parseExpression(src), scope);

  it("lowers onto the conversation resolver, with the ref as its argument", () => {
    const ref = lowerExpression(parseExpression("messages(.inputs.s)"));
    expect(ref).toMatchObject({ op: { kind: "function", functionRef: "conversation.get" } });
  });

  it("infers an array of turns, so reading a turn is checked", () => {
    expect(infer("messages(.inputs.s)").schema).toMatchObject({ type: "array" });
    // `.content` projects to a string; a typo does not project at all, which is the point of
    // typing a closed shape rather than leaving a call's result universal.
    expect(infer("at(messages(.inputs.s), -1).content").unresolved).toEqual([]);
    expect(infer("at(messages(.inputs.s), -1).text").schema).toBeDefined();
  });
});

describe("pathOfRef recovers the reference a lowered sub-tree reads", () => {
  const pathOf = (src: string): string[] | undefined => pathOfRef(lowerExpression(parseExpression(src)));

  it("reads back a root-anchored path", () => {
    expect(pathOf(".children.done.output.plan")).toEqual(["children", "done", "output", "plan"]);
    expect(pathOf(".children.done.output")).toEqual(["children", "done", "output"]);
    expect(pathOf(".children.done")).toEqual(["children", "done"]);
    expect(pathOf(".children.done.outcome")).toEqual(["children", "done", "outcome"]);
    expect(pathOf(".inputs")).toEqual(["inputs"]);
  });

  it("is undefined for anything that is not a path", () => {
    expect(pathOf("42")).toBeUndefined();
    expect(pathOf(".inputs.n === 2")).toBeUndefined();
    expect(pathOf("!.inputs.flag")).toBeUndefined();
  });
});

/**
 * INFERENCE over the lowered tree (EXPRESSIONS.md §1.4).
 *
 * The inferred type IS the leaf's producer schema — it is what makes `isSubschema` binding checking
 * apply to an expression with no special case, and what a `boolean`-strict guard is checked against.
 * So it has to agree with the AST inference exactly: a wrong schema here is a binding mismatch
 * nothing downstream catches, which is the quietest failure in the whole design.
 */
describe("inference over the tree agrees with inference over the AST", () => {
  const scope = {
    inputs: {
      type: "object",
      properties: {
        issue: { type: "string" },
        n: { type: "integer" },
        flag: { type: "boolean" },
        list: { type: "array", items: { type: "string" } },
        groups: { type: "array", items: { type: "array", items: { type: "integer" } } },
        nested: { type: "object", properties: { deep: { type: "number" } } },
      },
    },
    outputs: { type: "object", properties: { severity: { type: "string", enum: ["low", "high"] } } },
    children: {
      type: "object",
      properties: {
        done: {
          type: "object",
          properties: { outputs: { type: "object", properties: { plan: { type: "string" } } }, outcome: { type: "string" } },
        },
      },
    },
    run: { type: "object", properties: { iteration: { type: "integer" }, cursor: { type: "string" } } },
    limits: { type: "object", properties: { max_iterations: { type: "integer" } } },
  } as never;

  const SOURCES = [
    "42", "'high'", "true", "null",
    ".inputs", ".inputs.issue", ".inputs.n", ".inputs.flag",
    ".inputs.list.length", ".inputs.issue.length", ".inputs.nested.deep",
    ".outputs.severity",
    ".children.done.output.plan", ".children.done.outcome", ".children.done.output",
    ".inputs.n === 2", ".inputs.n < 3", "!.inputs.flag",
    ".run.iteration < .limits.max_iterations",
    ".inputs.flag && .inputs.issue", ".inputs.flag || .inputs.issue",
    ".outputs.severity === 'high' ? 'escalate' : 'continue'",
    ".inputs.flag ? .inputs.n : 3",
    ".run.cursor === 'done' && .outputs.severity === 'high'",
    "flatten(.inputs.groups)", "flatten(.inputs.list)",
    // the mistakes the validator has to keep catching
    ".nope", ".inputs.missing", ".inputs.nested.gone", ".children.done.output.absent",
  ];

  for (const src of SOURCES) {
    it(`${src}`, () => {
      const ast = parseExpression(src);
      const fromAst = inferExpression(ast, scope);
      const fromTree = inferRef(lowerExpression(ast), scope);
      expect(fromTree.schema, `${src} — schema`).toEqual(fromAst.schema);
      expect(fromTree.unresolved, `${src} — unresolved`).toEqual(fromAst.unresolved);
    });
  }
});
