/**
 * CALLABLE subtyping (hw SPEC §6.2): a producer of one callable type where a consumer expects
 * another — contravariant in its parameters, covariant in its output — and the binding checker's
 * use of it for a `prompt`/`function`-kind slot.
 */
import { describe, expect, it } from "vitest";
import type { CallableSchema, InlineFamily, Operation } from "@declarative-ai/ops";
import { callableSchemaOf, inlineDeref } from "@declarative-ai/ops";
import { checkBinding, isSubcallable, type CheckerHooks } from "../src/index.js";

const fn = (input: CallableSchema["input"], output?: CallableSchema["output"]): CallableSchema => ({
  kind: "function",
  input,
  ...(output !== undefined ? { output } : {}),
});

describe("isSubcallable", () => {
  it("accepts an identical contract, and a producer that takes more (optionally) or returns less", () => {
    const wanted = fn({ doc: { schema: { type: "string" } } }, { schema: { type: "string" } });
    expect(isSubcallable(wanted, wanted).ok).toBe(true);
    // An extra OPTIONAL parameter on the producer is fine: the consumer never has to fill it.
    expect(isSubcallable(fn({ doc: { schema: { type: "string" } }, mode: { schema: { type: "string" }, optional: true } }, { schema: { type: "string" } }), wanted).ok).toBe(true);
    // COVARIANT output: a producer returning an enum satisfies a consumer expecting a string.
    expect(isSubcallable(fn({ doc: { schema: { type: "string" } } }, { schema: { type: "string", enum: ["a", "b"] } }), wanted).ok).toBe(true);
  });

  it("refuses a kind mismatch, a missing parameter, and a required one the consumer cannot fill", () => {
    const wanted = fn({ doc: { schema: { type: "string" } } });
    expect(isSubcallable({ kind: "prompt", input: wanted.input }, wanted).reason).toMatch(/expects a function but the producer is a prompt/);
    expect(isSubcallable(fn({ text: {} }), wanted).reason).toMatch(/passes 'doc', which the producer does not accept/);
    expect(isSubcallable(fn({ doc: {}, extra: {} }), wanted).reason).toMatch(/requires 'extra', which the consumer does not pass/);
  });

  it("is CONTRAVARIANT in a parameter: what the consumer passes must be acceptable to the producer", () => {
    const passesString = fn({ doc: { schema: { type: "string" } } });
    const takesEnum = fn({ doc: { schema: { type: "string", enum: ["a"] } } });
    expect(isSubcallable(takesEnum, passesString).ok).toBe(false);
    expect(isSubcallable(passesString, takesEnum).ok).toBe(true);
  });

  it("is COVARIANT in the output", () => {
    const returnsString = fn({}, { schema: { type: "string" } });
    const returnsNumber = fn({}, { schema: { type: "number" } });
    expect(isSubcallable(returnsNumber, returnsString).reason).toMatch(/output/);
  });

  it("treats a side that declared nothing as constraining nothing", () => {
    expect(isSubcallable({ kind: "function" }, fn({ doc: { schema: { type: "string" } } })).ok).toBe(true);
    expect(isSubcallable(fn({ doc: {} }), { kind: "function" }).ok).toBe(true);
  });
});

describe("the binding checker on a callable slot", () => {
  const shout: Operation<InlineFamily> = {
    kind: "function",
    functionRef: "shout",
    input: { text: { kind: "text", schema: { type: "string" }, index: 0 } },
    output: { name: "output", kind: "text", schema: { type: "string" } },
  };
  const hooks: CheckerHooks<InlineFamily> = { producer: (ref) => (typeof ref === "string" ? undefined : ref) };

  it("types an UNCALLED operation as its own contract, and checks it against the slot's signature", () => {
    expect(callableSchemaOf(shout, inlineDeref)).toEqual({
      kind: "function",
      input: { text: { kind: "text", schema: { type: "string" }, index: 0 } },
      output: { kind: "text", schema: { type: "string" } },
    });
    const wanted = fn({ text: { schema: { type: "string" } } }, { schema: { type: "string" } });
    expect(checkBinding({ op: shout }, wanted as never, hooks, "input.fn", { kind: "function" })).toEqual([]);
    const wrong = fn({ text: { schema: { type: "number" } } });
    expect(checkBinding({ op: shout }, wrong as never, hooks, "input.fn", { kind: "function" })[0]!.message).toMatch(/parameter 'text'/);
  });

  it("refuses a data value where a callable is expected, and a callable where data is", () => {
    expect(checkBinding({ text: "shout" }, undefined, hooks, "input.fn", { kind: "function" })[0]!.message).toMatch(/expects a function but the producer is a string/);
    expect(checkBinding({ op: shout }, { type: "string" }, hooks, "input.text", { kind: "text" })).toEqual([]); // a CALL's output
  });
});
