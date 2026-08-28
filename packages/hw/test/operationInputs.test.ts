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
import { HOST_CAPABILITIES, type FunctionRegistry, type InlineFamily, type JsonSchema, type Parameter } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";

/**
 * A registry entry that declares what `review` takes — `text` required, `tone` optional.
 *
 * The slots are NAMED on the signature itself now, rather than being properties of one object schema
 * this check had to take apart. Optionality is the slot's own `optional`, which is what the old
 * `required` array said less directly.
 */
const registry = (required: string[] = ["text"]): FunctionRegistry<never, never> => {
  const map = new Map();
  const slot = (name: string, schema: JsonSchema): Parameter<InlineFamily> => ({
    kind: "text",
    schema,
    ...(required.includes(name) ? {} : { optional: true }),
  });
  map.set("review", {
    kind: "host",
    capabilities: HOST_CAPABILITIES,
    impl: async () => ({ value: {} }),
    signature: {
      input: { text: slot("text", { type: "string" }), tone: slot("tone", { type: "string" }) },
      output: { name: "output", kind: "json", schema: {} },
    },
  });
  return map as FunctionRegistry<never, never>;
};

const state = (input: Record<string, unknown>): Record<string, StateDef> =>
  ({
    s: {
      inputs: { doc: { schema: { type: "string" } } },
      outputs: { out: { schema: { type: "string" }, binding: ".operation.output.out" } },
      operation: { kind: "function", function: "review", input, output: { out: { schema: { type: "string" } } } },
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

  /**
   * A spread PASSES the slots its operand's type names (SPEC §6.3), and this check has to know it —
   * otherwise the one argument form that cannot be read at load would be the one form that always
   * reports a missing argument.
   */
  it("counts a deferred spread as passing the slots its type names", () => {
    const spread = (bag: JsonSchema): string[] =>
      validateBundle(
        loadBundle(
          {
            s: {
              inputs: { bag: { schema: bag } },
              outputs: { out: { schema: { type: "string" }, binding: ".operation.output.out" } },
              operation: { function: "review(....inputs.bag)", output: { out: { schema: { type: "string" } } } },
            },
          } as unknown as Record<string, StateDef>,
          "s",
        ),
        { functions: registry() },
      ).errors.map((e) => e.message);

    expect(spread({ type: "object", properties: { text: { type: "string" } } })).toEqual([]);
    // And an operand naming only the OPTIONAL slot still leaves the required one unpassed.
    expect(spread({ type: "object", properties: { tone: { type: "string" } } }).join(" ")).toMatch(
      /requires an input 'text'/,
    );
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

/**
 * The check has to survive `operation.function` resolving along the path, and the way it survives is
 * the point: it stopped asking "does the operation declare a slot of this name" — which the callee's
 * own slots now answer YES to unconditionally — and started asking whether anything will actually
 * FILL it.
 */
describe("a required parameter is satisfied by whatever will actually fill it", () => {
  const withCallee = (input: Record<string, unknown>, inputs: Record<string, unknown>) =>
    validateBundle(
      loadBundle(
        {
          s: {
            inputs,
            outputs: { out: { schema: { type: "string" }, binding: ".operation.output.out" } },
            operation: { kind: "function", function: "review", input, output: { out: { schema: { type: "string" } } } },
          },
        } as unknown as Record<string, StateDef>,
        "s",
        // The registry ON the loader too, so the callee's slots are copied onto the operation — the
        // arrangement that made the old spelling of this check unfireable.
        { functions: registry() },
      ),
      { functions: registry() },
    ).errors.map((e) => e.message);

  it("is reported when nothing binds it and no state input shares its name", () => {
    expect(withCallee({}, { doc: { schema: { type: "string" } } }).join(" | ")).toMatch(/requires an input 'text'/);
  });

  it("is clean when a STATE INPUT of that name will fill the free slot", () => {
    // The engine's own rule: `opInputs = { ...instance.inputs, ...resolved.values }`, so a state
    // whose declared input is called `text` fills the callee's `text` with nothing wired.
    expect(withCallee({}, { text: { schema: { type: "string" } } })).toEqual([]);
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

  it("says nothing when the impl declares NO SLOTS", () => {
    // An entry that names no slot constrains nothing — which is what every entry written before
    // signatures existed means, and the reason an empty map is a legal signature rather than an error.
    const map = new Map();
    map.set("review", {
      kind: "host",
      capabilities: HOST_CAPABILITIES,
      impl: async () => ({ value: {} }),
      signature: { input: {}, output: { name: "output", kind: "json", schema: {} } },
    });
    expect(validateBundle(loadBundle(state({}), "s"), { functions: map as FunctionRegistry<never, never> }).errors).toEqual([]);
  });
});
