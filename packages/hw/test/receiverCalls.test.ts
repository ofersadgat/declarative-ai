/**
 * Receiver calls, and the two builtin spellings that make them read as written (NAMES.md §8).
 *
 * `recv.name(args)` is sugar for `name(recv, args)` and nothing more: it lowers to the same tree, so
 * there is no second semantics to keep in step — which is what these pin, by resolving both spellings
 * and by comparing the lowered trees themselves.
 */
import { describe, expect, it } from "vitest";
import type { InlineFamily, Ref } from "@declarative-ai/exec";
import { parseExpression } from "../src/expr.js";
import { inferExpression } from "../src/inferExpr.js";
import { lowerExpression, type LowerOptions } from "../src/lowerExpr.js";
import { loadBundle } from "../src/loader.js";
import { isResolvedValue, resolveRef, type ResolutionScope } from "../src/resolve.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";

const CONTEXT = {
  inputs: { xs: [3, 9, 4], words: ["b", "a"] },
  any: [
    { model: "small", remaining: 20 },
    { model: "large", remaining: 75 },
    { model: "mid", remaining: 40 },
  ],
};

const scope: ResolutionScope = {
  exprContext: CONTEXT,
  childOutputs: () => undefined,
  scopeValue: () => undefined,
  optionalInput: () => false,
  artifact: () => undefined,
  conversation: () => undefined,
};

const lowered = (src: string, options: LowerOptions = {}): Ref<InlineFamily> => lowerExpression(parseExpression(src), options);
const valueOf = (src: string, options: LowerOptions = {}): unknown => {
  const r = resolveRef(lowered(src, options), scope);
  if (!isResolvedValue(r)) throw new Error(`did not resolve: ${JSON.stringify(r)}`);
  return r.value;
};

describe("recv.name(args) is name(recv, args)", () => {
  it("lowers to the SAME tree as the call form", () => {
    expect(lowered(".inputs.xs.len()")).toEqual(lowered("len(.inputs.xs)"));
    expect(lowered(".inputs.words.join(', ')")).toEqual(lowered("join(.inputs.words, ', ')"));
    expect(lowered(".inputs.xs.slice(1).len()")).toEqual(lowered("len(slice(.inputs.xs, 1))"));
  });

  it("applies to a call's RESULT as well as to a leading-dot read", () => {
    expect(valueOf("sort(.inputs.xs).reverse().at(0)")).toBe(9);
  });

  it("leaves a BARE dotted name what it was — a module symbol, never a receiver", () => {
    const seen: string[] = [];
    const options: LowerOptions = {
      resolveOperation: (name) => {
        seen.push(name);
        return { kind: "function", functionRef: name, input: { doc: { kind: "json" } }, output: { name: "value", kind: "json" } };
      },
    };
    lowered("confidence.score(.inputs.xs)", options);
    expect(seen).toEqual(["confidence.score"]);
  });

  it("stays a call of a callable VALUE when the receiver's declared type has that property", () => {
    const callable = lowered(".inputs.contains(.inputs.xs)", { receiverHas: () => true });
    expect(JSON.stringify(callable)).toContain('"op.apply"');
    // …and with no such property declared, the same text is the operation.
    expect(lowered(".inputs.contains(.inputs.xs)")).toEqual(lowered("contains(.inputs, .inputs.xs)"));
  });

  it("stays a call of a value when the word after the dot names no operation at all", () => {
    expect(JSON.stringify(lowered(".inputs.reviewer(.inputs.xs)"))).toContain('"op.apply"');
  });

  it("is decided from the state's DECLARED inputs when a workflow loads", () => {
    const files: Record<string, StateDef> = {
      root: {
        inputs: {
          xs: { schema: { type: "array", items: { type: "number" } } },
          len: { kind: "function", schema: { input: { value: { schema: { type: "array" } } }, output: { schema: { type: "integer" } } } },
        },
        outputs: {
          counted: { schema: { type: "integer" }, binding: ".inputs.xs.len()" },
          called: { schema: { type: "integer" }, binding: ".inputs.len(.inputs.xs)" },
        },
      } as unknown as StateDef,
    };
    const bundle = loadBundle(files, "root");
    const binding = (name: string): string => JSON.stringify(bundle.states.root!.outputs![name]!.binding);
    expect(binding("counted")).not.toContain("op.apply");
    expect(binding("called")).toContain("op.apply");
    expect(validateBundle(bundle).errors).toEqual([]);
  });

  it("infers the receiver form as the call form, and reports no phantom reference", () => {
    const typed = { inputs: { type: "object", properties: { xs: { type: "array", items: { type: "number" } } } } } as never;
    const result = inferExpression(parseExpression(".inputs.xs.len() > 2"), typed);
    expect(result.schema).toEqual({ type: "boolean" });
    expect(result.unresolved).toEqual([]);
    expect(result.issues).toEqual([]);
  });
});

describe("the two builtin spellings", () => {
  it("`map(xs, 'key')` with a STRING plucks", () => {
    expect(lowered("map(.any, 'remaining')")).toEqual(lowered("pluck(.any, 'remaining')"));
    expect(valueOf(".any.map('remaining')")).toEqual([20, 75, 40]);
  });

  it("`indexOf(xs, v)` finds a value, by value, and answers -1 for none", () => {
    expect(valueOf("indexOf(.inputs.xs, 9)")).toBe(1);
    expect(valueOf("indexOf(.inputs.xs, 5)")).toBe(-1);
    expect(valueOf("indexOf(.any, { model: 'mid', remaining: 40 })")).toBe(2);
  });

  it("`indexOf(xs, op)` means `indexOf(xs, op(xs))`", () => {
    expect(lowered("indexOf(.inputs.xs, max)")).toEqual(lowered("indexOf(.inputs.xs, max(.inputs.xs))"));
    expect(valueOf(".inputs.xs.indexOf(max)")).toBe(1);
    expect(valueOf(".inputs.xs.indexOf(min)")).toBe(0);
  });

  it("gives `max` and `min` an answer for ONE list, and none for an empty one", () => {
    expect(valueOf("max(.inputs.xs)")).toBe(9);
    expect(valueOf("max(2, 7)")).toBe(7);
    expect(valueOf("max(.inputs.missing)")).toBeNaN();
    expect(valueOf("indexOf(slice(.inputs.xs, 0, 0), max)")).toBe(-1);
  });

  it("reads NAMES.md §6's pick as written", () => {
    expect(valueOf(".any[.any.map('remaining').indexOf(max)]")).toEqual({ model: "large", remaining: 75 });
  });
});

describe("inference decides a receiver call the way lowering does", () => {
  // `stats` has a property called `max`. Lowering cannot see a child's output type, so it lowers the
  // BUILT-IN; inference reading that type would check a call of `.stats.max` that never runs.
  const typed = {
    inputs: { type: "object", properties: { f: { type: "object" }, bag: { type: "object", properties: { max: { type: "object" } } } } },
    children: { type: "object", properties: { k: { type: "object", properties: { output: { type: "object", properties: { stats: { type: "object", properties: { max: { type: "number" } } } } } } } } },
  } as never;

  it("takes a built-in's name after a NON-input receiver as the built-in, whatever properties the type has", () => {
    const result = inferExpression(parseExpression(".children.k.output.stats.max()"), typed);
    expect(result.schema).toEqual({ type: "number" });
    expect(result.issues).toEqual([]);
    expect(lowered(".children.k.output.stats.max()")).toEqual(lowered("max(.children.k.output.stats)"));
  });

  it("keeps a property an INPUT declares a call of that value — in both", () => {
    const options: LowerOptions = { receiverHas: (_recv, name) => name === "max" };
    expect(JSON.stringify(lowered(".inputs.bag.max(1)", options))).toContain("op.apply");
    // Inference agrees: it is a call through `.inputs.bag.max`, which here is not callable — said so,
    // rather than typed as the built-in's number.
    const result = inferExpression(parseExpression(".inputs.bag.max(1)"), typed);
    expect(result.schema).not.toEqual({ type: "number" });
  });
});
