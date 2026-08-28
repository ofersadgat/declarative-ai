/**
 * Binding a state output to what the operation RETURNED.
 *
 * The operation's result used to be unaddressable: it landed straight in the state's declared output
 * slots, so there was no way to rename it, put an expression between the call and the slot, or read
 * it from a second output. `operation` was not even a binding namespace — the loader refused any
 * path starting with it, while the same path in a `when` guard loaded cleanly.
 *
 * What is pinned here: the namespace resolves in a BINDING, it is typed by what the operation
 * actually returns, and a name the call never produces is still a load-time error.
 */
import { describe, expect, it } from "vitest";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { operationNodeSchema } from "../src/operationNode.js";
import type { StateDef } from "../src/format.js";

const errorsFor = (files: Record<string, StateDef>, rootId: string): string[] =>
  validateBundle(loadBundle(files, rootId)).errors.map((e) => `${e.stateId}:${e.path}:${e.message}`);

/** A prompt state that returns `report`, with one output bound to it. */
const boundTo = (binding: string): Record<string, StateDef> => ({
  s: {
    outputs: {
      report: { schema: { type: "string" }, binding: ".operation.output.report" },
      summary: { schema: { type: "string" }, binding },
    },
    // The call says what it returns; the state says what it publishes.
    operation: { kind: "prompt", prompt: "go", output: { report: { schema: { type: "string" } } } },
  },
});

describe("`operation` is a binding namespace", () => {
  it("binds an output to a value the call returned", () => {
    expect(errorsFor(boundTo(".operation.output.report"), "s")).toEqual([]);
  });

  it("renames on the way through — the slot need not match the returned name", () => {
    // The capability the produced-output mechanism could not express: what the call returns and what
    // the state publishes are now two names with a binding between them.
    const files = boundTo(".operation.output.report");
    expect(Object.keys(files["s"]!.outputs!)).toContain("summary");
    expect(errorsFor(files, "s")).toEqual([]);
  });

  it("still reads the conversation position, now as one output among the rest", () => {
    expect(
      errorsFor(
        {
          s: {
            outputs: {
              answer: { schema: { type: "string" }, binding: ".operation.output.answer" },
              where: { binding: ".operation.output.session" },
            },
            operation: { kind: "prompt", prompt: "go", output: { answer: { schema: { type: "string" } } } },
          },
        },
        "s",
      ),
    ).toEqual([]);
  });

  it("reports a name the operation never returns", () => {
    // The whole point of typing the namespace rather than leaving it an object of unknowns.
    const [message] = errorsFor(boundTo(".operation.output.nope"), "s");
    expect(message).toMatch(/nope/);
  });

  it("refuses `.operation` on a state with no operation", () => {
    // A pure composite has no call, so there is no node to read.
    const [message] = errorsFor(
      { s: { children: { kid: {} }, sequence: ["kid"], outputs: { x: { binding: ".operation.outcome" } } }, "s/kid": { operation: { kind: "prompt", prompt: "go" } } },
      "s",
    );
    expect(message).toBeDefined();
  });

  it("names something, rather than accepting a bare `.operation`", () => {
    expect(() => loadBundle({ s: { outputs: { x: { binding: ".operation" } }, operation: { kind: "prompt", prompt: "go" } } }, "s")).toThrow(
      /must name something on the operation/,
    );
  });
});

describe("the namespace is typed by what the operation returns", () => {
  it("carries the declared output's properties", () => {
    const schema = operationNodeSchema("function", {
      name: "output",
      kind: "json",
      schema: { type: "object", properties: { report: { type: "string" } } },
    }) as { properties: { output: { properties: Record<string, unknown> } } };
    expect(Object.keys(schema.properties.output.properties)).toEqual(["report"]);
  });

  it("makes a BLOB output the value itself, not a record around it", () => {
    const schema = operationNodeSchema("function", { name: "report", kind: "blob", schema: { type: "string" } }) as {
      properties: { output: { type: string } };
    };
    expect(schema.properties.output.type).toBe("string");
  });

  it("makes an ARRAY output the array itself", () => {
    // `.operation.outputs` IS the list. Wrapping it in a record keyed by the slot's name would make
    // `.operation.output.output` the spelling for "the list", which reads as a field access on it.
    const schema = operationNodeSchema("function", {
      name: "output",
      schema: { items: { type: "string" } },
    }) as { properties: { output: { items: { type: string } } } };
    expect(schema.properties.output.items.type).toBe("string");
  });

  it("keeps `session` reachable on an array output", () => {
    // An array carries named properties, so the position rides along without displacing the list.
    const schema = operationNodeSchema("prompt", {
      name: "output",
      schema: { items: { type: "string" } },
    }) as { properties: { output: { items: { type: string }; properties: Record<string, unknown> } } };
    expect(schema.properties.output.items.type).toBe("string");
    expect(Object.keys(schema.properties.output.properties)).toEqual(["session"]);
  });

  it("keeps `session` alongside the call's own outputs on a prompt op", () => {
    const schema = operationNodeSchema("prompt", {
      name: "output",
      kind: "json",
      schema: { type: "object", properties: { report: { type: "string" } } },
    }) as { properties: { output: { properties: Record<string, unknown> } } };
    expect(Object.keys(schema.properties.output.properties).sort()).toEqual(["report", "session"]);
  });

  it("lets an operation's own `session` output win over the engine's", () => {
    // The author named it. Shadowing theirs would leave their value unreadable by any route, while
    // the engine's position is still reachable from a guard.
    const schema = operationNodeSchema("prompt", {
      name: "output",
      kind: "json",
      schema: { type: "object", properties: { session: { type: "string" } } },
    }) as { properties: { output: { properties: { session: { type?: string } } } } };
    expect(schema.properties.output.properties.session.type).toBe("string");
  });

  it("gives a function op with no declared output no `outputs` at all", () => {
    const schema = operationNodeSchema("function") as { properties: Record<string, unknown> };
    expect(Object.keys(schema.properties)).not.toContain("output");
  });
});

/**
 * The operation declares what it RETURNS, and that is the model's structured-output contract.
 *
 * `operation.input` was always a map of named slots while `output` was a single one, and the
 * contract was derived from the STATE's unbound outputs instead — so the call borrowed its own
 * signature from whatever the state around it happened to publish. `operation.outputs` restores the
 * symmetry: the call says what it returns, the state binds from it.
 */
describe("`operation.outputs` is the call's signature", () => {
  const withDeclaredOutputs = (): Record<string, StateDef> => ({
    s: {
      outputs: {
        summary: { schema: { type: "string" }, binding: ".operation.output.report" },
        rating: { schema: { type: "number" }, binding: ".operation.output.score" },
      },
      operation: {
        kind: "prompt",
        prompt: "go",
        output: {
          report: { schema: { type: "string" } },
          score: { schema: { type: "number" } },
        },
      },
    },
  });

  it("loads, with every state output bound to a declared return", () => {
    expect(errorsFor(withDeclaredOutputs(), "s")).toEqual([]);
  });

  it("becomes the structured-output schema handed to the model", () => {
    // The lowered op carries ONE output — the executor seam takes one — whose schema is the object
    // built from the declared map. This is what a prompt op asks the model for.
    const bundle = loadBundle(withDeclaredOutputs(), "s");
    const output = (bundle.states["s"] as unknown as { operation: { output: { schema: { properties: Record<string, unknown>; required?: string[] } } } })
      .operation.output;
    expect(Object.keys(output.schema.properties).sort()).toEqual(["report", "score"]);
    expect(output.schema.required?.sort()).toEqual(["report", "score"]);
  });

  it("does NOT take the contract from the state's outputs any more", () => {
    // `summary` and `rating` are what the STATE publishes. The model is asked for `report`/`score`.
    const bundle = loadBundle(withDeclaredOutputs(), "s");
    const output = (bundle.states["s"] as unknown as { operation: { output: { schema: { properties: Record<string, unknown> } } } }).operation.output;
    expect(Object.keys(output.schema.properties)).not.toContain("summary");
  });

  it("types `.operation.output.*` from the declaration, so a wrong name is caught", () => {
    const files = withDeclaredOutputs();
    files["s"]!.outputs!["summary"] = { schema: { type: "string" }, binding: ".operation.output.nope" };
    expect(errorsFor(files, "s").join(" ")).toMatch(/nope/);
  });

  it("checks the bound type against what the call declares it returns", () => {
    // `report` is a string; a number slot must not silently accept it.
    const files = withDeclaredOutputs();
    files["s"]!.outputs!["summary"] = { schema: { type: "number" }, binding: ".operation.output.report" };
    expect(errorsFor(files, "s").length).toBeGreaterThan(0);
  });

  it("says a blob return with a ONE-ENTRY map carrying the kind", () => {
    // §4.4: a delegated agent hands back ONE value, not a record. A lone entry that declares its
    // `kind` is how the map says so — the bytes ARE the value, and `report` names it rather than
    // being a field wrapped around it.
    const bundle = loadBundle(
      {
        s: {
          outputs: { report: { binding: ".operation.output" } },
          operation: {
            kind: "function",
            function: "agent",
            output: { report: { kind: "blob", schema: { type: "string", contentMediaType: "text/markdown" } } },
          },
        },
      },
      "s",
    );
    const output = (bundle.states["s"] as unknown as { operation: { output: { kind: string; name: string } } }).operation.output;
    expect(output.kind).toBe("blob");
    expect(output.name).toBe("report");
  });

  it("REFUSES a kind on a multi-entry map, rather than reading it only when the map is short", () => {
    // The cliff this closes: were the kind read only on a lone entry, adding a second output would
    // silently rewrap the return and break every binding onto it, with nothing said.
    // Carried as data, not thrown: an operation that will not build is one state's problem, and the
    // loader reports the rest of the bundle rather than stopping at the first bad one.
    const bundle = loadBundle(
      {
        s: {
          outputs: { report: { binding: ".operation.output.report" } },
          operation: {
            kind: "function",
            function: "agent",
            output: {
              report: { kind: "blob", schema: { type: "string", contentMediaType: "text/markdown" } },
              summary: { schema: { type: "string" } },
            },
          },
        },
      },
      "s",
    );
    expect(bundle.states["s"]?.operationError ?? "").toMatch(/output\.report: declares kind 'blob'/);
  });
});

/**
 * An operation whose output is not an object.
 *
 * The shape that broke the first attempt: `"output": { "schema": { "items": { "type": "string" } } }`
 * on a prompt op — a list of strings, with no properties to name. `.operation.outputs` has to BE the
 * list; a record wrapped around it would make `.operation.output.output` the spelling for the value.
 */
describe("a non-object operation output", () => {
  const listState = (binding: string): Record<string, StateDef> => ({
    s: {
      inputs: { feature_description: { schema: { type: "string" } } },
      outputs: { features: { schema: { type: "array", items: { type: "string" } }, binding } },
      operation: {
        kind: "prompt",
        prompt: "split {{.inputs.feature_description}} into features",
        input: { feature_description: { schema: { type: "string" }, binding: ".inputs.feature_description" } },
        output: { features: { kind: "json", schema: { type: "array", items: { type: "string" } } } },
      },
    },
  });

  it("binds the whole list off `.operation.outputs`", () => {
    expect(errorsFor(listState(".operation.output"), "s")).toEqual([]);
  });

  it("type-checks the list against the slot", () => {
    // A list of strings must not bind into a slot declared as a list of numbers.
    const files = listState(".operation.output");
    files["s"]!.outputs!["features"] = {
      schema: { type: "array", items: { type: "number" } },
      binding: ".operation.output",
    };
    expect(errorsFor(files, "s").length).toBeGreaterThan(0);
  });

  it("still reads the session off it", () => {
    const files = listState(".operation.output");
    files["s"]!.outputs!["where"] = { binding: ".operation.output.session" };
    expect(errorsFor(files, "s")).toEqual([]);
  });

  it("reports a property the list does not have", () => {
    const files = listState(".operation.output");
    files["s"]!.outputs!["features"] = { schema: { type: "string" }, binding: ".operation.output.report" };
    expect(errorsFor(files, "s").length).toBeGreaterThan(0);
  });
});
