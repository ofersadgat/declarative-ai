/**
 * The `operation.*` expression namespace (SPEC.md §6.1).
 *
 * A state's operation used to be unaddressable — its result landed straight in the state's declared
 * output slots, while a CHILD was reachable as `children.<key>.outputs`. That asymmetry is why
 * engine metadata about the call had nowhere to live except the events journal, which no expression
 * can read.
 *
 * Two properties are worth pinning: the namespace resolves at run time, and it is TYPED by the
 * operation's kind — so `operation.output.session` on a `ui` gate is an authoring error rather than
 * a binding that silently resolves to nothing.
 */
import { describe, expect, it } from "vitest";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { operationNodeSchema } from "../src/operationNode.js";
import type { StateDef } from "../src/format.js";

const errorsFor = (files: Record<string, StateDef>, rootId: string): string[] =>
  validateBundle(loadBundle(files, rootId)).errors.map((e) => `${e.stateId}:${e.path}:${e.message}`);

describe("the schema is a union over the operation's kind", () => {
  it("gives a prompt op the conversation position it ended at", () => {
    const schema = operationNodeSchema("prompt") as { properties: Record<string, unknown> };
    expect(Object.keys(schema.properties)).toContain("output");
  });

  it("gives a function op no session at all", () => {
    // Not "undefined at run time" — absent from the type, so reaching for it is caught at load.
    const schema = operationNodeSchema("function") as { properties: Record<string, unknown> };
    expect(Object.keys(schema.properties)).not.toContain("output");
    expect(Object.keys(schema.properties)).toContain("cost");
  });

  it("gives a state with no operation no namespace", () => {
    // A pure composite has no call, so `operation.cost` there is an unresolved reference rather than
    // an object of unknowns.
    expect(operationNodeSchema(undefined)).toBeUndefined();
  });

  it("is closed, so a typo in a metadata field is a lint error", () => {
    const schema = operationNodeSchema("prompt") as { additionalProperties: boolean };
    expect(schema.additionalProperties).toBe(false);
  });

  it("treats a session ref as opaque — `id` and `end`, and no other structure", () => {
    // `operation.output.session.position` is a plausible thing to reach for given how the notation
    // reads. It must not resolve: only the session store knows a ref has structure. `end` is the one
    // exception, and it is DECLARED — the same conversation with no position, which is the difference
    // between continuing a thread and branching off it.
    type Ref = { additionalProperties: boolean; properties: Record<string, unknown> };
    const schema = operationNodeSchema("prompt") as { properties: { output: { properties: { session: Ref } } } };
    const session = schema.properties.output.properties.session;
    expect(Object.keys(session.properties)).toEqual(["id", "end"]);
    expect(session.additionalProperties).toBe(false);
    // ...and `end` is already unpositioned, so there is nothing further to ask it for.
    const end = session.properties["end"] as Ref;
    expect(Object.keys(end.properties)).toEqual(["id"]);
    expect(end.additionalProperties).toBe(false);
  });
});

describe("the load-time lint", () => {
  const promptLeaf = (): StateDef => ({
    outputs: { answer: { schema: { type: "string" } } },
    operation: { kind: "prompt", prompt: "go", model: "m" },
  });

  it("accepts `.children.<key>.operation.output.session` when the child is a prompt op", () => {
    const files: Record<string, StateDef> = {
      root: {
        children: { plan: { state: "leaf" }, review: { state: "leaf" } },
        sequence: ["plan", "review"],
        outputs: { r: { binding: ".children.review.outputs.answer" } },
      },
      leaf: promptLeaf(),
    };
    files["root"]!.children!["review"]!.state = "consumer";
    files["root"]!.children!["review"]!.inputs = { s: { expr: ".children.plan.operation.output.session" } };
    files["consumer"] = { ...promptLeaf(), inputs: { s: { schema: {} } } };
    expect(errorsFor(files, "root")).toEqual([]);
  });

  it("accepts `.session.end`, and refuses a second hop off it", () => {
    // `.end` is the conversation with no position — the "append to the end" spelling. It is a ref
    // like any other, so asking IT for an end is the same reach-too-far as `.session.position`.
    const files = (expr: string): Record<string, StateDef> => ({
      root: {
        children: { plan: { state: "leaf" }, review: { state: "consumer", inputs: { s: { expr } } } },
        sequence: ["plan", "review"],
        outputs: { r: { binding: ".children.review.outputs.answer" } },
      },
      leaf: promptLeaf(),
      consumer: { ...promptLeaf(), inputs: { s: { schema: {} } } },
    });
    expect(errorsFor(files(".children.plan.operation.output.session.end"), "root")).toEqual([]);
    for (const overreach of [".children.plan.operation.output.session.end.end", ".children.plan.operation.output.session.position"]) {
      expect(errorsFor(files(overreach), "root").some((e) => /resolves to no declared value/.test(e))).toBe(true);
    }
  });

  it("REFUSES it when the child is a function op", () => {
    const files: Record<string, StateDef> = {
      root: {
        children: { gate: { state: "gate" }, review: { state: "leaf" } },
        sequence: ["gate", "review"],
        outputs: { r: { binding: ".children.review.outputs.answer" } },
      },
      gate: { outputs: { decision: { schema: { type: "string" } } }, operation: { kind: "function", function: "choose_option" } },
      leaf: { ...promptLeaf(), inputs: { s: { schema: {} } } },
    };
    files["root"]!.children!["review"]!.state = "leaf";
    files["root"]!.children!["review"]!.inputs = { s: { expr: ".children.gate.operation.output.session" } };
    const errors = errorsFor(files, "root");
    expect(errors.some((e) => /operation.*outputs/.test(e) || /resolves to no declared value/.test(e))).toBe(true);
  });

  it("REFUSES a metadata field the engine does not fill", () => {
    // `attempts` is not declared until the executor reports it. Better a lint error now than a value
    // that is always undefined.
    const files: Record<string, StateDef> = {
      root: {
        children: { plan: { state: "leaf" }, review: { state: "leaf" } },
        sequence: ["plan", "review"],
        outputs: { r: { binding: ".children.review.outputs.answer" } },
      },
      leaf: promptLeaf(),
      consumer: { ...promptLeaf(), inputs: { s: { schema: {} } } },
    };
    files["root"]!.children!["review"]!.state = "consumer";
    files["root"]!.children!["review"]!.inputs = { s: { expr: ".children.plan.operation.attempts" } };
    expect(errorsFor(files, "root").some((e) => /resolves to no declared value/.test(e))).toBe(true);
  });

  it("accepts the common core on either kind", () => {
    const files: Record<string, StateDef> = {
      root: {
        children: { gate: { state: "gate" } },
        outputs: { r: { schema: { type: "number" }, binding: { expr: ".children.gate.operation.cost" } } },
      },
      gate: { outputs: { decision: { schema: { type: "string" } } }, operation: { kind: "function", function: "choose_option" } },
    };
    expect(errorsFor(files, "root")).toEqual([]);
  });

  it("a state can read its OWN operation, and a guard can branch on it", () => {
    const files: Record<string, StateDef> = {
      root: {
        outputs: { answer: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: "go", model: "m" },
        children: { retry: { state: "root" } },
        transitions: [{ to: "terminate.success", when: "operation.outcome === 'success'" }],
      },
    };
    expect(errorsFor(files, "root")).toEqual([]);
  });
});
