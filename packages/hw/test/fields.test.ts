/**
 * COMPUTED FIELDS (SPEC §5.3): every value in a state file may be a binding, evaluated once at
 * instance entry in dependency order, journaled as `value.settled`, and read back under its
 * authored name. The motivating case is `title` (§5.2): a prompt asked once per instance for a
 * display name, which a board shows instead of the state's static label.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecServices, type FunctionInputs, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { FakePromptExecutor, modelOf, newRegistry, ok, promptTail, type Script } from "./fakes.js";
import type { Vfs } from "../src/reference.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import type { LoadedInstance } from "../src/load.js";

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
  /** The title prompt: asked for a name, answers with one. */
  [`${FUNCTIONS}/title.json`]: JSON.stringify({
    kind: "prompt",
    prompt: "Name this: {{.inputs.issue}}",
    model: "titler",
    input: { issue: { kind: "text", index: 0, schema: { type: "string" } } },
    output: { title: { schema: { type: "string" } } },
  }),
  [`${FUNCTIONS}/wrap.json`]: JSON.stringify({
    kind: "function",
    function: "wrap",
    input: { text: { kind: "text", index: 0, schema: { type: "string" } } },
    output: { loud: { schema: { type: "string" } } },
  }),
};

function bundleFor(states: Record<string, unknown>) {
  return loadBundle(states, "plan", { defaultRoot: [WF, FUNCTIONS], roots: { JAIRA: ROOT }, vfs: vfsOf(files) });
}

function registry() {
  const r = newRegistry();
  r.functions.set(
    "wrap",
    hostFunction<ExecServices, WorkflowMetrics>(async (inputs: FunctionInputs) => ({ value: { loud: String(inputs.text ?? "").toUpperCase() } as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  return r;
}

/** The title prompt answers with a name; the main call answers by its model. */
const script: Script = (call) => {
  if (modelOf(call) === "titler") return ok({ title: `Ship: ${promptTail(call).replace("Name this: ", "")}` });
  return ok({ done: modelOf(call) });
};

async function run(states: Record<string, unknown>, inputs: Record<string, ResolvedValue>, prompt: Script = script) {
  const bundle = bundleFor(states);
  const report = validateBundle(bundle);
  expect(report.errors).toEqual([]);
  const persistence = new InMemoryPersistence();
  const executor = new FakePromptExecutor(prompt);
  const engine = new WorkflowEngine({ bundle, registry: registry(), prompt: executor, validator: new SchemaValidator(), persistence });
  const result = await engine.run({ inputs });
  const settled = persistence.events.map(({ event }) => event).filter((e): e is Extract<EngineEvent, { type: "value.settled" }> => e.type === "value.settled");
  return { result, settled, calls: executor.calls };
}

/** A root whose title is asked of a prompt, and whose model depends on the title. */
const titled = {
  label: "Planning",
  title: { binding: { expr: "title(.inputs.issue).title", environment: { session: null }, failureValue: "Untitled" } },
  inputs: { issue: { kind: "text", schema: { type: "string" } } },
  outputs: { done: { schema: { type: "string" } } },
  operation: {
    kind: "prompt",
    prompt: "Work on {{.inputs.issue}}",
    model: { expr: "startsWith(.title, 'Ship') ? 'fast' : 'slow'" },
  },
};

describe("a computed title (SPEC §5.2)", () => {
  it("is evaluated once at entry, journaled with its value, and readable by the fields that depend on it", async () => {
    const { result, settled, calls } = await run({ "plan.json": titled }, { issue: "the docs" });
    expect(result.outcome).toBe("success");
    expect(settled.map((e) => [e.field, e.outcome, e.value])).toEqual([
      ["title", "value", "Ship: the docs"],
      ["operation.config.model", "value", "fast"],
    ]);
    // The title was asked ONCE, before the main call, which ran under the model the title chose.
    expect(calls.map(modelOf)).toEqual(["titler", "fast"]);
    expect(result.outputs?.done).toBe("fast");
  });

  it("stands its failureValue in when the prompt fails, and says so in the journal", async () => {
    const failing: Script = (call) => (modelOf(call) === "titler" ? { error: { classification: "permanent", reason: "no title today" }, metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" } } : script(call));
    const { result, settled } = await run({ "plan.json": titled }, { issue: "the docs" }, failing);
    expect(result.outcome).toBe("success");
    const title = settled.find((e) => e.field === "title")!;
    expect(title).toMatchObject({ outcome: "error", value: "Untitled", fallback: true });
    expect(title.error).toMatch(/no title today/);
    expect(settled.find((e) => e.field === "operation.config.model")?.value).toBe("slow");
  });

  it("fails the instance when there is no failureValue to stand in", async () => {
    const { title: _t, ...rest } = titled;
    const bare = { ...rest, title: { binding: { expr: "title(.inputs.issue).title" } } };
    const failing: Script = (call) => (modelOf(call) === "titler" ? { error: { classification: "permanent", reason: "no title today" }, metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" } } : script(call));
    const { result, settled, calls } = await run({ "plan.json": bare }, { issue: "the docs" }, failing);
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/field 'title': no title today/);
    expect(settled.find((e) => e.field === "title")).toMatchObject({ outcome: "error" });
    // The operation never ran: it waits for nothing it does not read, but the instance failed first.
    expect(calls.map(modelOf)).toEqual(["titler"]);
  });

  it("is not paid for again by a LOADED instance that brings its settled fields", async () => {
    const bundle = bundleFor({ "plan.json": titled });
    const executor = new FakePromptExecutor(script);
    const persistence = new InMemoryPersistence();
    const engine = new WorkflowEngine({ bundle, registry: registry(), prompt: executor, validator: new SchemaValidator(), persistence });
    const loaded: LoadedInstance = { id: "root-1", stateId: "plan", inputs: { issue: "the docs" }, live: true, fields: { title: "Ship: loaded", "operation.config.model": "fast" } };
    const result = await engine.loadRun(loaded);
    expect(result.outcome).toBe("success");
    expect(executor.calls.map(modelOf)).toEqual(["fast"]);
    expect(persistence.events.some(({ event }) => event.type === "value.settled")).toBe(false);
    // A field the description LACKS is settled afresh — the one case a run stopped mid-evaluation leaves.
    const partial = new FakePromptExecutor(script);
    const again = new WorkflowEngine({ bundle, registry: registry(), prompt: partial, validator: new SchemaValidator() });
    await again.loadRun({ ...loaded, id: "root-2", fields: { title: "Ship: loaded" } });
    expect(partial.calls.map(modelOf)).toEqual(["fast"]);
  });
});

describe("every value is a binding (SPEC §5.3)", () => {
  it("computes a label, a limit, a system prompt off the model, a wrapped knob, and a wrapped arg", async () => {
    const { result, settled, calls } = await run(
      {
        "plan.json": {
        label: { expr: "concat('Plan for ', .inputs.issue)" },
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { done: { schema: { type: "string" } } },
        limits: { max_iterations: { expr: "len(.inputs.issue)" } },
        operation: {
          kind: "prompt",
          prompt: "Work on {{.inputs.issue}}",
          model: { expr: "'fast'" },
          system: { expr: "concat(concat('Model: ', .operation.config.model), concat(' for ', .label))" },
          providerOptions: { effort: { binding: { expr: ".limits.max_iterations" } } },
        },
        },
      },
      { issue: "the docs" },
    );
    expect(result.outcome).toBe("success");
    expect(new Map(settled.map((e) => [e.field, e.value]))).toEqual(
      new Map<string, unknown>([
        ["label", "Plan for the docs"],
        ["limits.max_iterations", 8],
        ["operation.config.model", "fast"],
        ["operation.system", "Model: fast for Plan for the docs"],
        ["operation.config.providerOptions.effort", 8],
      ]),
    );
    const call = calls[0]!;
    expect(call.op.system).toBe("Model: fast for Plan for the docs");
    expect((call.op.config as { providerOptions?: { effort?: number } }).providerOptions?.effort).toBe(8);
  });

  it("binds a callee: `function` from an input, its arguments checked against the input's signature", async () => {
    const child = {
      inputs: {
        fn: { kind: "function", schema: { input: { text: { schema: { type: "string" } } } } },
        text: { kind: "text", schema: { type: "string" } },
      },
      outputs: { loud: { schema: { type: "string" } } },
      operation: { kind: "function", function: { expr: ".inputs.fn" }, args: { text: { binding: ".inputs.text" } } },
    };
    const parent = {
      inputs: { issue: { kind: "text", schema: { type: "string" } } },
      children: { child: { state: "plan/child", inputs: { fn: { expr: "wrap" }, text: ".inputs.issue" } } },
      outputs: { loud: { schema: { type: "string" }, binding: ".children.child.output.loud" } },
    };
    const { result, settled } = await run({ "plan.json": parent, "plan/child.json": child }, { issue: "ship it" });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.loud).toBe("SHIP IT");
    expect(settled.find((e) => e.field === "operation.function")?.value).toMatchObject({ kind: "function", functionRef: "wrap" });
    // An argument the signature has no parameter for is a lint error.
    const wrong = validateBundle(
      bundleFor({
        "plan.json": parent,
        "plan/child.json": { ...child, operation: { ...child.operation, args: { nope: { binding: ".inputs.text" } } } },
      }),
    );
    expect(wrong.errors.map((e) => e.message).join("\n")).toMatch(/passes 'nope', which the bound callable does not accept/);
  });

  it("binds a prompt: a template string computed at entry", async () => {
    const { result, calls } = await run(
      {
        "plan.json": {
          inputs: { issue: { kind: "text", schema: { type: "string" } } },
          outputs: { done: { schema: { type: "string" } } },
          operation: { kind: "prompt", prompt: { expr: "concat('Please handle ', .inputs.issue)" }, model: "fast" },
        },
      },
      { issue: "the docs" },
    );
    expect(result.outcome).toBe("success");
    expect(promptTail(calls[0]!)).toBe("Please handle the docs");
  });

  it("refuses a field that reads what does not exist at entry, and a cycle between fields", () => {
    const readsChild = validateBundle(
      bundleFor({
        "plan.json": { label: { expr: ".children.c.output.x" }, children: { c: { state: "plan/c" } } },
        "plan/c.json": {},
      }),
    );
    expect(readsChild.errors.map((e) => e.message).join("\n")).toMatch(/evaluated at entry, before any child runs/);
    const readsResult = validateBundle(bundleFor({ "plan.json": { label: { expr: ".operation.outcome" }, operation: { kind: "function", function: "wrap" } } }));
    expect(readsResult.errors.map((e) => e.message).join("\n")).toMatch(/before the operation runs/);
    expect(() => bundleFor({ "plan.json": { label: { expr: ".description" }, description: { expr: ".label" } } })).toThrow(/fields form a cycle/);
  });

  it("computes a slot's default at entry from the inputs that were provided, and an output's at termination", async () => {
    const { result } = await run(
      {
        "plan.json": {
          inputs: {
            issue: { kind: "text", schema: { type: "string" } },
            summary: { kind: "text", schema: { type: "string" }, default: { expr: "concat('About ', .inputs.issue)" } },
          },
          outputs: {
            done: { schema: { type: "string" } },
            echo: { schema: { type: "string" }, binding: ".inputs.summary" },
            fallback: { schema: { type: "string" }, default: { expr: "concat(.inputs.summary, '!')" } },
          },
          operation: { kind: "prompt", prompt: "Work on {{.inputs.summary}}", model: "fast" },
        },
      },
      { issue: "the docs" },
    );
    expect(result.outcome).toBe("success");
    expect(result.outputs?.echo).toBe("About the docs");
    expect(result.outputs?.fallback).toBe("About the docs!");
    // An input's default resolves before any field settles, so it may not read one.
    const readsField = validateBundle(bundleFor({ "plan.json": { label: { expr: "'x'" }, inputs: { a: { schema: { type: "string" }, default: { expr: ".label" } } } } }));
    expect(readsField.errors.map((e) => e.message).join("\n")).toMatch(/before the state's fields settle/);
  });

  it("binds the permission profile, a mount's async flag, and gives a binding's call a fresh session", async () => {
    const { result, settled, calls } = await run(
      {
        "plan.json": {
          inputs: { issue: { kind: "text", schema: { type: "string" } }, quick: { kind: "json", schema: { type: "boolean" }, default: true } },
          title: { binding: { expr: "title(.inputs.issue).title", environment: { session: null } } },
          outputs: { done: { schema: { type: "string" }, binding: ".children.work.output.done" } },
          children: { work: { state: "plan/work", async: { expr: ".inputs.quick" }, inputs: { issue: ".inputs.issue", quick: ".inputs.quick" } } },
          // An environment layer's expression is evaluated in the scope of the state whose operation
          // it ends up in — the child's — so it reads the CHILD's `quick`, wired from the parent's.
          environment: { permissions: { profile: { expr: ".inputs.quick ? 'read-only' : 'full'" } } },
        },
        "plan/work.json": {
          inputs: { issue: { kind: "text", schema: { type: "string" } }, quick: { kind: "json", schema: { type: "boolean" } } },
          outputs: { done: { schema: { type: "string" } } },
          operation: { kind: "prompt", prompt: "Work on {{.inputs.issue}}", model: "fast" },
        },
      },
      { issue: "the docs" },
    );
    expect(result.outcome).toBe("success");
    expect(result.outputs?.done).toBe("fast");
    // The child's permission profile was computed from the parent's input it inherited the layer under.
    expect(settled.find((e) => e.field === "environment.permissions.profile")?.value).toBe("read-only");
    // The title call ran in a conversation of its own; the child's op declared none and got none.
    const title = calls.find((c) => modelOf(c) === "titler")!;
    expect(title.ctx.session).toBeDefined();
    // A session NAME on a binding is refused at load.
    const named = validateBundle(bundleFor({ "plan.json": { label: { expr: "'x'", environment: { session: "shared" } } } }));
    expect(named.errors.map((e) => e.message).join("\n")).toMatch(/must be null .* or a ref expression/);
  });

  it("type-checks a field and its failureValue against the field's own type", () => {
    const wrongType = validateBundle(bundleFor({ "plan.json": { limits: { max_iterations: { expr: "'three'" } } } }));
    expect(wrongType.errors.map((e) => e.message).join("\n")).toMatch(/not type-compatible/);
    const wrongFallback = validateBundle(bundleFor({ "plan.json": { label: { expr: "'x'", failureValue: 3 } } }));
    expect(wrongFallback.errors.map((e) => e.message).join("\n")).toMatch(/failureValue does not satisfy/);
  });
});
