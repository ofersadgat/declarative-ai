/**
 * Two things the static checker must get right about a BINDING, both of which it used to get wrong in
 * the same direction — disagreeing with the engine that runs the document.
 *
 *  1. **What a binding may name.** A producer edge names a declared child or one of the loader's
 *     synthesized resolvers; `runResolver` refuses anything else. The checker typed an author-embedded
 *     op by the generic "its declared output schema is its type" rule and passed it — so the shape was
 *     authorable, validated clean even under `strict`, and then never resolved at run time.
 *  2. **`.length`.** The expression evaluator documents it as the one meaningful property of an array
 *     or a string, and the inferrer had no projection for it — so `when: ".inputs.items.length > 0"`
 *     validated as a reference to nothing and was, in practice, unauthorable.
 */
import { describe, expect, it } from "vitest";
import type { JsonSchema } from "@declarative-ai/exec";
import type { StateDef } from "../src/format.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";

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
      operation: { kind: "prompt", model: "m", prompt: "go" },
      transitions: [{ to: "terminate.success", when: guard }],
    } as StateDef,
  });

  it("infers an integer for an array's length — in a guard AND in an `{ expr }` leaf", () => {
    expect(messages(files(".inputs.items.length > 0", ".inputs.items.length"), "root")).toBe("");
  });

  it("does the same for a string", () => {
    expect(messages(files(".inputs.note.length > 3", ".inputs.note.length"), "root")).toBe("");
  });

  it("types it precisely enough to be CHECKED against the consuming slot", () => {
    // The point of inferring `integer` rather than shrugging to `any`: the assertion on the leaf is now
    // decidable, so a wrong declared type is caught instead of waved through.
    expect(messages(files(".inputs.items.length > 0", ".inputs.items.length", { type: "string" }), "root")).toMatch(
      /producer type 'integer' not allowed by consumer string/,
    );
  });

  it("does not invent `.length` for an object, which has none", () => {
    expect(messages(files(".inputs.config.length > 0", ".inputs.items.length"), "root")).toMatch(/'.inputs.config.length', which resolves to no declared value/);
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
        model: "m",
        prompt: "go",
        input: { thing: { kind: "json", binding: binding as never } },
      },
      children: { helper: { state: "root/helper", inputs: { seed: ".inputs.seed" } } },
      sequence: ["helper"],
    } as StateDef,
    "root/helper": {
      label: "Helper",
      inputs: { seed: { schema: { type: "string" } } },
      outputs: { result: { schema: { type: "string" } } },
      operation: { kind: "prompt", model: "m", prompt: "help" },
    } as StateDef,
  });

  const embedded = { kind: "function", functionRef: "totally_missing", input: {}, output: { name: "value", kind: "json" } };

  /**
   * An author-embedded operation used to be REJECTED here, and that was right at the time: nothing
   * could run one, so binding to it validated clean and then never resolved — the rejection was the
   * honest half of that pair.
   *
   * It is a lowered CALL now (EXPRESSIONS.md §3): the engine runs it before resolution reads the
   * result, so the pair is honest the other way round. Keeping the rejection would fail every
   * workflow that calls anything.
   */
  it("accepts an author-embedded operation, which is a lowered call", () => {
    expect(messages(withBinding({ op: embedded }), "root")).toBe("");
  });

  /**
   * An unregistered CALLEE is reported. The registry check used to run over a state's own
   * `operation.functionRef` only, so a call to a function nobody registered passed lint and then
   * failed mid-run — the one place a missing function is fatal to the whole run.
   */
  it("reports an unregistered callee", () => {
    expect(messages(withBinding({ op: embedded }), "root", { functions: new Map(), strict: true })).toMatch(
      /no function 'totally_missing' is registered/,
    );
  });

  /** Without a registry in hand there is nothing to check against, so it stays quiet. */
  it("says nothing about a callee when no registry was supplied", () => {
    expect(messages(withBinding({ op: embedded }), "root")).toBe("");
  });

  it("rejects an operation input that references a CHILD — the operation runs before any child (§7.4)", () => {
    // The operation resolves its inputs before any child runs, so a producer edge to a child here can
    // only ever fail at run time ("child 'helper' has not run"). Caught statically instead. An
    // operation input is a value/scope form or an embedded operation — never a reach into children.
    expect(messages(withBinding(".children.helper.outputs"), "root")).toMatch(/operation input cannot reference child 'helper'/);
    expect(messages(withBinding(".children.helper.outputs.result"), "root")).toMatch(/operation input cannot reference child 'helper'/);
  });

  it("still accepts the session-owned resolvers, whose contents are only known at run time", () => {
    expect(messages(withBinding(".artifacts.spec"), "root")).toBe("");
  });

  it("still accepts the ordinary literal and scope forms", () => {
    expect(messages(withBinding({ text: "literal" }), "root")).toBe("");
    expect(messages(withBinding(".inputs.seed"), "root")).toBe("");
    expect(messages(withBinding({ expr: ".inputs.seed" }), "root")).toBe("");
  });
});

/**
 * An output's `schema` is a CONSTRAINT, and declaring one is optional (SPEC §6.2).
 *
 * The half that was missing: an undeclared one used to mean the top type, which no typed consumer
 * accepts — so leaving it off made the state unwirable into anything typed, and the complaint
 * landed on the PARENT, about a slot the parent did not write. Optional in name only.
 */
describe("an untyped output adopts the type of what fills it (§6.2)", () => {
  const wiring = (helperOutput: unknown, seed: JsonSchema = { type: "string" }): Record<string, StateDef> => ({
    root: {
      label: "Root",
      inputs: { seed: { schema: seed } },
      outputs: { answer: { schema: { type: "string" }, binding: ".children.helper.outputs.result" } },
      children: { helper: { state: "root/helper", inputs: { seed: ".inputs.seed" } } },
      sequence: ["helper"],
    } as StateDef,
    "root/helper": {
      label: "Helper",
      inputs: { seed: { schema: seed } },
      outputs: { result: helperOutput as never },
    } as StateDef,
  });

  it("takes the binding's type, so a typed consumer accepts it", () => {
    expect(messages(wiring({ binding: ".inputs.seed" }), "root")).toBe("");
  });

  it("takes it precisely enough to be REJECTED when it does not fit", () => {
    // The point of inferring rather than shrugging: undeclared is not unchecked. The consumer wants
    // a string and the child hands out whatever its own `seed` is.
    expect(messages(wiring({ binding: ".inputs.seed" }, { type: "number" }), "root")).toMatch(
      /producer type 'number' not allowed by consumer string/,
    );
  });

  it("still honours a DECLARED schema over the binding's own type", () => {
    expect(messages(wiring({ schema: { type: "number" }, binding: ".inputs.seed" }), "root")).toMatch(
      /producer type 'number' not allowed by consumer string/,
    );
  });

  it("leaves an output with no binding unconstrained — the operation fills it, and untyped", () => {
    // Nothing is connected, so there is nothing to adopt. Unknown, rather than merely undeclared.
    expect(messages(wiring({}), "root")).toMatch(/consumer requires type string but producer declares none/);
  });

  it("terminates when a child mounts its own ancestor", () => {
    // Inference needs the declaring state's scope, and a scope names its children's outputs. A cycle
    // in the mount graph would otherwise recurse forever; the state already being inferred
    // contributes nothing instead.
    const cyclic: Record<string, StateDef> = {
      root: {
        label: "Root",
        inputs: { seed: { schema: { type: "string" } } },
        outputs: { answer: { binding: ".children.loop.outputs.back" } },
        children: { loop: { state: "root/loop", inputs: { seed: ".inputs.seed" } } },
        sequence: ["loop"],
      } as StateDef,
      "root/loop": {
        label: "Loop",
        inputs: { seed: { schema: { type: "string" } } },
        outputs: { back: { binding: ".children.again.outputs.answer" } },
        children: { again: { state: "root", inputs: { seed: ".inputs.seed" } } },
        sequence: ["again"],
      } as StateDef,
    };
    expect(() => messages(cyclic, "root")).not.toThrow();
  });
});

describe("an untyped output is adopted by its OWN state's readers too (§6.2)", () => {
  /** A second output deriving from a first — the documented pattern, and a consumer like any other. */
  const files = (first: unknown): Record<string, StateDef> => ({
    root: {
      label: "Root",
      inputs: { seed: { schema: { type: "string" } } },
      outputs: {
        first: first as never,
        second: { schema: { type: "string" }, binding: ".outputs.first" },
      },
      operation: { kind: "prompt", model: "m", prompt: "go" },
    } as StateDef,
  });

  it("types `.outputs.<name>` from what the first output is bound to", () => {
    expect(messages(files({ binding: ".inputs.seed" }), "root")).toBe("");
  });

  it("and rejects it when the adopted type does not fit", () => {
    expect(messages(files({ binding: { json: 3 } }), "root")).toMatch(/not allowed by consumer string/);
  });
});
