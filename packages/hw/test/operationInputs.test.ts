/**
 * Does the state pass the operation what the operation needs? (DESIGN §7)
 *
 * A registered function may declare a `signature`, and the checker compared it in ONE direction
 * only: a document parameter the impl has no slot for. That catches an argument nothing reads —
 * untidy, but harmless. It did not catch the reverse, which is the one that stops a run: a parameter
 * the impl REQUIRES and the state never passes. The call then fails at dispatch, having loaded and
 * linted clean.
 *
 * `children.<key>.inputs` has had this check for as long as it has existed ("required child input
 * 'x' is not wired"). There is no reason mounting a child badly should be a lint error while calling
 * a function badly is not.
 */
import { describe, expect, it } from "vitest";
import { HOST_CAPABILITIES, type FunctionRegistry } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";

/** A registry entry that declares what `review` takes — `text` required, `tone` optional. */
const registry = (required: string[] = ["text"]): FunctionRegistry<never, never> => {
  const map = new Map();
  map.set("review", {
    kind: "host",
    capabilities: HOST_CAPABILITIES,
    impl: async () => ({ value: {} }),
    signature: {
      input: {
        schema: {
          type: "object",
          properties: { text: { type: "string" }, tone: { type: "string" } },
          ...(required.length > 0 ? { required } : {}),
        },
      },
      output: { name: "output", schema: {} },
    },
  });
  return map as FunctionRegistry<never, never>;
};

const state = (input: Record<string, unknown>): Record<string, StateDef> =>
  ({
    s: {
      inputs: { doc: { schema: { type: "string" } } },
      outputs: { out: { schema: { type: "string" }, binding: ".operation.output.out" } },
      operation: { kind: "function", function: "review", input, outputs: { out: { schema: { type: "string" } } } },
    },
  }) as unknown as Record<string, StateDef>;

const errorsFor = (input: Record<string, unknown>, required?: string[]): string[] =>
  validateBundle(loadBundle(state(input), "s"), { functions: registry(required) }).errors.map((e) => e.message);

describe("a required operation input the state never passes", () => {
  it("is an error naming the parameter", () => {
    const [message] = errorsFor({ tone: { schema: { type: "string" }, binding: ".inputs.doc" } });
    expect(message).toMatch(/requires an input 'text', which this state does not pass/);
  });

  it("is clean once the state passes it", () => {
    expect(errorsFor({ text: { schema: { type: "string" }, binding: ".inputs.doc" } })).toEqual([]);
  });

  it("says nothing about an OPTIONAL parameter the state omits", () => {
    // `tone` is declared but not required — leaving it out is the impl's own default, not a fault.
    expect(errorsFor({ text: { schema: { type: "string" }, binding: ".inputs.doc" } }, ["text"])).toEqual([]);
  });

  it("reports every missing parameter, not just the first", () => {
    const messages = errorsFor({}, ["text", "tone"]);
    expect(messages.join("\n")).toMatch(/'text'/);
    expect(messages.join("\n")).toMatch(/'tone'/);
  });

  it("still reports a parameter the impl does not accept — the other direction survives", () => {
    const [message] = errorsFor({
      text: { schema: { type: "string" }, binding: ".inputs.doc" },
      nope: { schema: { type: "string" }, binding: ".inputs.doc" },
    });
    expect(message).toMatch(/declares a parameter 'nope' its registered implementation does not accept/);
  });
});

describe("what must NOT be reported", () => {
  it("says nothing with no registry — the document alone cannot know the signature", () => {
    expect(validateBundle(loadBundle(state({}), "s")).errors).toEqual([]);
  });

  it("says nothing for an entry that declares no signature", () => {
    const map = new Map();
    map.set("review", { kind: "host", capabilities: HOST_CAPABILITIES, impl: async () => ({ value: {} }) });
    const report = validateBundle(loadBundle(state({}), "s"), { functions: map as FunctionRegistry<never, never> });
    expect(report.errors).toEqual([]);
  });

  it("says nothing when the impl takes an OPEN object", () => {
    // No declared property set constrains nothing — there is no signature to disagree with.
    const map = new Map();
    map.set("review", {
      kind: "host",
      capabilities: HOST_CAPABILITIES,
      impl: async () => ({ value: {} }),
      signature: { input: { schema: { type: "object" } }, output: { name: "output", schema: {} } },
    });
    expect(validateBundle(loadBundle(state({}), "s"), { functions: map as FunctionRegistry<never, never> }).errors).toEqual([]);
  });
});
