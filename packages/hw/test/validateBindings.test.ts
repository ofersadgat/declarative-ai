/**
 * Two things the static checker must get right about a BINDING, both of which it used to get wrong in
 * the same direction — disagreeing with the engine that runs the document.
 *
 *  1. **What a binding may name.** A producer edge names a declared child or one of the loader's
 *     synthesized resolvers; `runResolver` refuses anything else. The checker typed an author-embedded
 *     op by the generic "its declared output schema is its type" rule and passed it — so the shape was
 *     authorable, validated clean even under `strict`, and then never resolved at run time.
 *  2. **`.length`.** The expression evaluator documents it as the one meaningful property of an array
 *     or a string, and the inferrer had no projection for it — so `when: "inputs.items.length > 0"`
 *     validated as a reference to nothing and was, in practice, unauthorable.
 */
import { describe, expect, it } from "vitest";
import type { JsonSchema } from "@declarative-ai/exec";
import type { StateDef } from "../src/format";
import { loadBundle } from "../src/loader";
import { validateBundle } from "../src/validate";

const report = (files: Record<string, StateDef>, rootId: string, env = {}) => validateBundle(loadBundle(files, rootId), env);
const messages = (files: Record<string, StateDef>, rootId: string, env = {}) =>
  report(files, rootId, env)
    .errors.map((e) => `${e.path}: ${e.message}`)
    .join("\n");

describe("`.length` on arrays and strings (§7.2)", () => {
  const files = (guard: string, leaf: string, declared: JsonSchema = { type: "integer" }): Record<string, StateDef> => ({
    root: {
      label: "Root",
      inputs: {
        items: { schema: { type: "array", items: { type: "string" } } },
        note: { schema: { type: "string" } },
        config: { schema: { type: "object", properties: { a: { type: "string" } } } },
      },
      outputs: { size: { schema: declared, binding: { expr: leaf } } },
      operation: { kind: "prompt", config: { model: "m" }, prompt: { template: "go" } },
      transitions: [{ to: "terminate.success", when: guard }],
    } as StateDef,
  });

  it("infers an integer for an array's length — in a guard AND in an `{ expr }` leaf", () => {
    expect(messages(files("inputs.items.length > 0", "inputs.items.length"), "root")).toBe("");
  });

  it("does the same for a string", () => {
    expect(messages(files("inputs.note.length > 3", "inputs.note.length"), "root")).toBe("");
  });

  it("types it precisely enough to be CHECKED against the consuming slot", () => {
    // The point of inferring `integer` rather than shrugging to `any`: the assertion on the leaf is now
    // decidable, so a wrong declared type is caught instead of waved through.
    expect(messages(files("inputs.items.length > 0", "inputs.items.length", { type: "string" }), "root")).toMatch(
      /producer type 'integer' not allowed by consumer string/,
    );
  });

  it("does not invent `.length` for an object, which has none", () => {
    expect(messages(files("inputs.config.length > 0", "inputs.items.length"), "root")).toMatch(/'inputs.config.length', which resolves to no declared value/);
  });
});

describe("what a producer edge may name (§7.4)", () => {
  /** A prompt op with one bound input — the binding under test. */
  const withBinding = (binding: unknown): Record<string, StateDef> => ({
    root: {
      label: "Root",
      inputs: { seed: { schema: { type: "string" } } },
      outputs: { answer: { schema: { type: "string" } } },
      operation: {
        kind: "prompt",
        config: { model: "m" },
        prompt: { template: "go" },
        input: { thing: { kind: "json", binding: binding as never } },
      },
      children: { helper: { state: "root/helper", inputs: { seed: { input: "seed" } } } },
      sequence: ["helper"],
    } as StateDef,
    "root/helper": {
      label: "Helper",
      inputs: { seed: { schema: { type: "string" } } },
      outputs: { result: { schema: { type: "string" } } },
      operation: { kind: "prompt", config: { model: "m" }, prompt: { template: "help" } },
    } as StateDef,
  });

  const embedded = { kind: "function", functionRef: "totally_missing", input: {}, output: { name: "value", kind: "json" } };

  it("rejects an author-embedded operation, which nothing would ever run", () => {
    expect(messages(withBinding({ op: embedded }), "root")).toMatch(/embedded operation 'totally_missing', which no binding can run/);
  });

  it("rejects it whether or not a registry is in hand — this is a DOCUMENT error, not a lookup miss", () => {
    for (const env of [{}, { functions: new Map() }, { functions: new Map(), strict: true }]) {
      expect(messages(withBinding({ op: embedded }), "root", env)).toMatch(/no binding can run/);
    }
  });

  it("rejects an operation input that references a CHILD — the operation runs before any child (§7.4)", () => {
    // The operation resolves its inputs before any child runs, so a producer edge to a child here can
    // only ever fail at run time ("child 'helper' has not run"). Caught statically instead. An
    // operation input is a value/scope form or an embedded operation — never a reach into children.
    expect(messages(withBinding({ child: "helper" }), "root")).toMatch(/operation input cannot reference child 'helper'/);
    expect(messages(withBinding({ child: "helper", output: "result" }), "root")).toMatch(/operation input cannot reference child 'helper'/);
  });

  it("still accepts the session-owned resolvers, whose contents are only known at run time", () => {
    expect(messages(withBinding({ artifact: "spec" }), "root")).toBe("");
    expect(messages(withBinding({ conversation: "default" }), "root")).toBe("");
    expect(messages(withBinding({ conversation: "default", message: 0 }), "root")).toBe("");
  });

  it("still accepts the ordinary literal and scope forms", () => {
    expect(messages(withBinding({ text: "literal" }), "root")).toBe("");
    expect(messages(withBinding({ input: "seed" }), "root")).toBe("");
    expect(messages(withBinding({ expr: "inputs.seed" }), "root")).toBe("");
  });
});
