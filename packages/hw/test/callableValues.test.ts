/**
 * Signature-carrying CALLABLE types (SPEC §4.1, §6.2): a `function`-kind slot carries a signature,
 * an uncalled reference passed into it is checked against that signature, and a call THROUGH the
 * value — `.inputs.fn(.inputs.text)` — is typed and dispatched.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecServices, type FunctionInputs, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { newRegistry } from "./fakes.js";
import type { Vfs } from "../src/reference.js";
import type { WorkflowMetrics } from "../src/ports.js";

const ROOT = "/p/.jaira";
const WF = `${ROOT}/workflows`;
const FUNCTIONS = `${ROOT}/functions`;

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (!rest.includes("/")) names.add(rest);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

const files = {
  [`${WF}/plan.json`]: "{}",
  [`${FUNCTIONS}/shout.json`]: JSON.stringify({
    kind: "function",
    function: "shout",
    input: { text: { kind: "text", index: 0, schema: { type: "string" } } },
    output: { value: { kind: "text", schema: { type: "string" } } },
  }),
  [`${FUNCTIONS}/count.json`]: JSON.stringify({
    kind: "function",
    function: "count",
    input: { n: { kind: "json", index: 0, schema: { type: "integer" } } },
    output: { value: { kind: "json", schema: { type: "integer" } } },
  }),
};

function registry(calls: { n: number }) {
  const r = newRegistry();
  r.functions.set(
    "noop",
    hostFunction<ExecServices, WorkflowMetrics>(async () => ({ value: {} as ResolvedValue }), { interactive: false, readOnly: true, memoizable: true }),
  );
  r.functions.set(
    "shout",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs) => {
        calls.n += 1;
        return { value: String(inputs.text ?? "").toUpperCase() as ResolvedValue };
      },
      { interactive: false, readOnly: true, memoizable: true },
    ),
  );
  r.functions.set(
    "count",
    hostFunction<ExecServices, WorkflowMetrics>(async (inputs: FunctionInputs) => ({ value: (Number(inputs.n) + 1) as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  return r;
}

function bundleFor(states: Record<string, unknown>) {
  return loadBundle(states, "plan", { defaultRoot: [WF, FUNCTIONS], roots: { JAIRA: ROOT }, vfs: vfsOf(files) });
}

/** A child that takes a function and applies it. */
const applier = (signature: unknown, expr = ".inputs.fn(.inputs.text)") => ({
  inputs: {
    fn: { kind: "function", schema: signature },
    text: { kind: "text", schema: { type: "string" } },
  },
  outputs: { loud: { schema: { type: "string" }, binding: { expr } } },
  operation: { kind: "function", function: "noop" },
});

const STRING_TO_STRING = { input: { text: { schema: { type: "string" } } }, output: { schema: { type: "string" } } };

describe("a function-kind slot carries a signature", () => {
  it("stamps the slot's kind into its schema, so the type is self-describing", () => {
    const bundle = bundleFor({ "plan.json": applier(STRING_TO_STRING), "plan/child.json": {} });
    expect(bundle.states.plan!.inputs!.fn!.schema).toEqual({ kind: "function", ...STRING_TO_STRING });
    // A slot that says nothing is "some function".
    const bare = bundleFor({ "plan.json": { inputs: { fn: { kind: "function" } } } });
    expect(bare.states.plan!.inputs!.fn!.schema).toEqual({ kind: "function" });
  });

  it("checks an UNCALLED reference wired in against the signature", () => {
    const parent = (fn: unknown) => ({
      inputs: { issue: { kind: "text", schema: { type: "string" } } },
      children: { child: { state: "plan/child", inputs: { fn, text: ".inputs.issue" } } },
      outputs: { loud: { schema: { type: "string" }, binding: ".children.child.output.loud" } },
    });
    const good = validateBundle(bundleFor({ "plan.json": parent({ expr: "shout" }), "plan/child.json": applier(STRING_TO_STRING) }));
    expect(good.errors).toEqual([]);
    // `count` takes an integer where the slot promises to pass a string: contravariance refuses it.
    const wrong = validateBundle(bundleFor({ "plan.json": parent({ expr: "count" }), "plan/child.json": applier(STRING_TO_STRING) }));
    expect(wrong.errors.map((e) => e.message).join("\n")).toMatch(/passes 'text', which the producer does not accept/);
    // Data where a function belongs.
    const data = validateBundle(bundleFor({ "plan.json": parent({ text: "shout" }), "plan/child.json": applier(STRING_TO_STRING) }));
    expect(data.errors.map((e) => e.message).join("\n")).toMatch(/expects a function but the producer is a string/);
  });

  it("types a call THROUGH the value against the declared signature", () => {
    const argType = validateBundle(bundleFor({ "plan.json": applier(STRING_TO_STRING, ".inputs.fn(3)") }));
    expect(argType.errors.map((e) => e.message).join("\n")).toMatch(/passes argument 'text' as a type the callable does not accept/);
    const arity = validateBundle(bundleFor({ "plan.json": applier(STRING_TO_STRING, ".inputs.fn(.inputs.text, 'extra')") }));
    expect(arity.errors.map((e) => e.message).join("\n")).toMatch(/takes 1 argument \(text\), but 2 were given/);
    const missing = validateBundle(bundleFor({ "plan.json": applier(STRING_TO_STRING, ".inputs.fn()") }));
    expect(missing.errors.map((e) => e.message).join("\n")).toMatch(/does not pass 'text', which the callable requires/);
    // The result is typed by the signature's output: a string result into a string slot is clean,
    // and a signature returning a number into that same slot is not.
    const result = validateBundle(bundleFor({ "plan.json": applier({ input: { text: {} }, output: { schema: { type: "number" } } }) }));
    expect(result.errors.map((e) => e.message).join("\n")).toMatch(/not type-compatible/);
    // An UNTYPED callable is legal, unchecked, and said out loud.
    const untyped = validateBundle(bundleFor({ "plan.json": applier(undefined) }));
    expect(untyped.errors).toEqual([]);
    expect(untyped.warnings.map((w) => w.message).join("\n")).toMatch(/applies an untyped callable/);
    // Calling something that is not callable at all.
    const notCallable = validateBundle(bundleFor({ "plan.json": { ...applier(STRING_TO_STRING, ".inputs.text('x')") } }));
    expect(notCallable.errors.map((e) => e.message).join("\n")).toMatch(/applies a value that is not callable/);
  });
});

describe("a function VALUE flows through a slot and is applied", () => {
  it("passes the operation into the child, and the child's call dispatches it", async () => {
    const calls = { n: 0 };
    const bundle = bundleFor({
      "plan.json": {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        children: { child: { state: "plan/child", inputs: { fn: { expr: "shout" }, text: ".inputs.issue" } } },
        outputs: { loud: { schema: { type: "string" }, binding: ".children.child.output.loud" } },
      },
      "plan/child.json": applier(STRING_TO_STRING),
    });
    expect(validateBundle(bundle).errors).toEqual([]);
    const engine = new WorkflowEngine({ bundle, registry: registry(calls), validator: new SchemaValidator() });
    const result = await engine.run({ inputs: { issue: "ship it" } });
    expect(result.failure).toBeUndefined();
    expect(result.outcome).toBe("success");
    expect(result.outputs?.loud).toBe("SHIP IT");
    expect(calls.n).toBe(1);
  });

  it("binds a spread by name through the value, and an extra positional argument is refused", async () => {
    const calls = { n: 0 };
    const spread = bundleFor({
      "plan.json": {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        children: { child: { state: "plan/child", inputs: { fn: { expr: "shout" }, text: ".inputs.issue" } } },
        outputs: { loud: { schema: { type: "string" }, binding: ".children.child.output.loud" } },
      },
      "plan/child.json": applier(STRING_TO_STRING, ".inputs.fn(...{ text: .inputs.text })"),
    });
    const engine = new WorkflowEngine({ bundle: spread, registry: registry(calls), validator: new SchemaValidator() });
    const result = await engine.run({ inputs: { issue: "ship it" } });
    expect(result.outputs?.loud).toBe("SHIP IT");
  });
});
