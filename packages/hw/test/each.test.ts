/**
 * A mount that FANS OUT (WORKFLOWS.md §6.2): `each: true` on one of its wires enters the child once
 * per element of the bound array, under ONE child record whose outputs are the elements' outputs in
 * element order. What the tests pin is the contract an author writes against — how many times the
 * child ran and with what, what the parent reads back, what `.each` says, what an empty array does,
 * how a failure names its element, and that a stopped fan-out comes back as the batch it was.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import type { JsonValue } from "@declarative-ai/json";
import { SchemaValidator } from "@declarative-ai/validate";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

interface Outcome {
  outcome: string;
  reason?: string;
  outputs?: Record<string, unknown>;
  /** Every `build` call, in the order the leaf was DISPATCHED, with the inputs it was handed. */
  calls: Array<Record<string, unknown>>;
  /** When each call finished — overlap is a claim about intervals, which `calls` alone cannot make. */
  finished: string[];
  events: EngineEvent[];
}

/**
 * Run (or load) a workflow whose leaves call `build`, which records what it was handed and answers
 * with a value derived from it — so the parent's collected outputs say which element produced which.
 */
async function run(
  files: Record<string, StateDef>,
  rootId: string,
  options: { inputs?: Record<string, ResolvedValue>; loaded?: LoadedInstance; fail?: string; delayMs?: Record<string, number> } = {},
): Promise<Outcome> {
  const calls: Array<Record<string, unknown>> = [];
  const finished: string[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "build",
    hostFunction(async (inputs: Record<string, unknown>) => {
      calls.push(inputs);
      const name = String(inputs.name ?? "");
      const delay = options.delayMs?.[name];
      if (delay !== undefined) await new Promise((resolve) => setTimeout(resolve, delay));
      finished.push(name);
      if (options.fail === name) return { error: { classification: "permanent" as const, reason: `${name} cannot be built` } };
      return ok({ doc: `doc for ${name}`, size: name.length }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({ bundle: loadBundle(files, rootId), registry, validator: new SchemaValidator(), persistence });
  const result = options.loaded !== undefined ? await engine.loadRun(options.loaded, { inputs: options.inputs ?? {} }) : await engine.run({ inputs: options.inputs ?? {} });
  return {
    outcome: result.outcome,
    ...(result.failure ? { reason: result.failure.reason } : {}),
    outputs: result.outputs as Record<string, unknown> | undefined,
    calls,
    finished,
    events: persistence.events.map(({ event }) => event),
  };
}

/** The child that is fanned out: one component in, one doc out. */
const COMPONENT_SCHEMA = { type: "object", properties: { name: { type: "string" } }, required: ["name"] } as const;

const COMPONENT: StateDef = {
  label: "Component",
  inputs: {
    component: { schema: COMPONENT_SCHEMA },
    flow: { schema: { type: "string" }, optional: true },
    position: { schema: { type: "integer" }, optional: true },
  },
  // No bindings: the operation fills these by name, as any leaf's does.
  outputs: { doc: { schema: { type: "string" } }, size: { schema: { type: "integer" } } },
  operation: {
    kind: "function",
    function: "build",
    input: {
      name: { kind: "json", binding: { expr: ".inputs.component.name" } },
      flow: { kind: "json", binding: ".inputs.flow" },
      position: { kind: "json", binding: ".inputs.position" },
    },
  },
};

/** Root → `component`, fanned out over the root's `components` input. */
const FAN: Record<string, StateDef> = {
  root: {
    label: "Root",
    inputs: {
      components: { schema: { type: "array", items: COMPONENT_SCHEMA } },
      flows: { schema: { type: "array", items: { type: "string" } }, optional: true },
    },
    outputs: {
      docs: { schema: { type: "array", items: { type: "string" } }, binding: ".children.component.output.doc" },
      sizes: { schema: { type: "array", items: { type: "integer" } }, binding: ".children.component.output.size" },
    },
    children: {
      component: {
        state: "root/component",
        inputs: {
          component: { expr: ".inputs.components", each: true },
          flow: ".inputs.flows[.each.index]",
          position: ".each.axis.component",
        },
      },
    },
    sequence: ["component"],
  },
  "root/component": COMPONENT,
};

const THREE = [{ name: "divider" }, { name: "badge" }, { name: "toggle" }];

describe("a mount with `each: true` on a wire", () => {
  it("enters the child once per element, in order, handing each its element", async () => {
    const { outcome, calls, outputs } = await run(FAN, "root", { inputs: { components: THREE, flows: ["list", "row", "row"] } });
    expect(outcome).toBe("success");
    expect(calls.map((c) => c.name)).toEqual(["divider", "badge", "toggle"]);
    // The parent reads every output as an ARRAY in element order — one record per key, as always.
    expect(outputs).toEqual({ docs: ["doc for divider", "doc for badge", "doc for toggle"], sizes: [7, 5, 6] });
  });

  it("lets the mount's other wires read `.each.index` and `.each.axis.<input>`", async () => {
    const { calls } = await run(FAN, "root", { inputs: { components: THREE, flows: ["list", "row", "row"] } });
    // `flow` is a parallel array indexed by the element's number; `position` is the axis coordinate,
    // which with one axis is the same number.
    expect(calls.map((c) => [c.flow, c.position])).toEqual([
      ["list", 0],
      ["row", 1],
      ["row", 2],
    ]);
  });

  it("enters nothing for an empty array, and the outputs read as empty arrays", async () => {
    const { outcome, calls, outputs, events } = await run(FAN, "root", { inputs: { components: [] } });
    expect(outcome).toBe("success");
    expect(calls).toEqual([]);
    expect(outputs).toEqual({ docs: [], sizes: [] });
    // No instance of the child was ever entered — the skip is the fan-out over nothing, not a guard.
    expect(events.filter((e) => e.type === "instance.entered" && (e as { childKey?: string }).childKey === "component")).toEqual([]);
  });

  it("journals each element's entry with its position", async () => {
    const { events } = await run(FAN, "root", { inputs: { components: THREE } });
    const entered = events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered" && e.childKey === "component");
    expect(entered.map((e) => e.element)).toEqual([0, 1, 2]);
    // Three DIFFERENT instances under one key and one parent — siblings, not passes.
    expect(new Set(entered.map((e) => e.instanceId)).size).toBe(3);
    expect(new Set(entered.map((e) => e.parentInstanceId)).size).toBe(1);
  });

  it("runs the elements in sequence and stops at the first that fails, naming it", async () => {
    const { outcome, reason, calls } = await run(FAN, "root", { inputs: { components: THREE }, fail: "badge" });
    expect(outcome).toBe("error");
    expect(reason).toMatch(/child 'component' element 1: badge cannot be built/);
    // `toggle` was never entered: a doomed batch is not finished for the sake of it.
    expect(calls.map((c) => c.name)).toEqual(["divider", "badge"]);
  });

  it("runs the elements concurrently when the mount is `async`", async () => {
    const files: Record<string, StateDef> = {
      ...FAN,
      root: { ...FAN.root!, children: { component: { ...FAN.root!.children!.component!, async: true } } },
    };
    const { outcome, finished, outputs } = await run(files, "root", { inputs: { components: THREE }, delayMs: { divider: 60, badge: 5, toggle: 30 } });
    expect(outcome).toBe("success");
    // The slow first element finishes LAST — they overlapped — and the outputs are still in ELEMENT order.
    expect(finished).toEqual(["badge", "toggle", "divider"]);
    expect(outputs?.docs).toEqual(["doc for divider", "doc for badge", "doc for toggle"]);
  });

  it("refuses a value that is not an array, through the mount's own seam", async () => {
    // The wire resolves to the array's LENGTH — a number where a list was promised.
    const files: Record<string, StateDef> = {
      ...FAN,
      root: {
        ...FAN.root!,
        children: { component: { state: "root/component", inputs: { component: { expr: ".inputs.components.length", each: true } } } },
      },
    };
    const { outcome, reason, events } = await run(files, "root", { inputs: { components: THREE } });
    expect(outcome).toBe("error");
    expect(reason).toMatch(/input 'component' is marked each, so it must be an array, and it resolved to a number/);
    // The refusal is an `instance.blocked` on the MOUNT — the same event a single mount's failed
    // wiring raises — so a rule on the mount could have answered it.
    expect(events.some((e) => e.type === "instance.blocked" && (e as { childKey?: string }).childKey === "component")).toBe(true);
  });
});

describe("several `each` wires on one mount", () => {
  /** A child taking a component AND a theme — two axes. */
  const PAIR: StateDef = {
    label: "Pair",
    inputs: {
      component: { schema: COMPONENT_SCHEMA },
      theme: { schema: { type: "string" } },
      position: { schema: { type: "object" }, optional: true },
    },
    outputs: { doc: { schema: { type: "string" } } },
    operation: {
      kind: "function",
      function: "build",
      input: {
        name: { kind: "json", binding: "concat(concat(.inputs.component.name, '/'), .inputs.theme)" },
        position: { kind: "json", binding: ".inputs.position" },
      },
    },
  };
  const GRID: Record<string, StateDef> = {
    root: {
      label: "Root",
      inputs: {
        components: { schema: { type: "array", items: COMPONENT_SCHEMA } },
        themes: { schema: { type: "array", items: { type: "string" } } },
      },
      outputs: { docs: { schema: { type: "array" }, binding: ".children.pair.output.doc" } },
      children: {
        pair: {
          state: "root/pair",
          inputs: {
            component: { expr: ".inputs.components", each: true },
            theme: { expr: ".inputs.themes", each: true },
            position: { expr: "{ index: .each.index, c: .each.axis.component, t: .each.axis.theme }" },
          },
        },
      },
      sequence: ["pair"],
    },
    "root/pair": PAIR,
  };

  it("take the cartesian product in row-major order, the first wire being the outer axis", async () => {
    const { outcome, reason, calls, outputs } = await run(GRID, "root", {
      inputs: { components: [{ name: "badge" }, { name: "toggle" }], themes: ["light", "dark"] },
    });
    expect(reason).toBeUndefined();
    expect(outcome).toBe("success");
    expect(calls.map((c) => c.name)).toEqual(["badge/light", "badge/dark", "toggle/light", "toggle/dark"]);
    expect(outputs?.docs).toEqual(["doc for badge/light", "doc for badge/dark", "doc for toggle/light", "doc for toggle/dark"]);
    // `.each.index` is the flat position; `.each.axis.<input>` the coordinate along that wire.
    expect(calls.map((c) => c.position)).toEqual([
      { index: 0, c: 0, t: 0 },
      { index: 1, c: 0, t: 1 },
      { index: 2, c: 1, t: 0 },
      { index: 3, c: 1, t: 1 },
    ]);
  });

  it("produce nothing when any axis is empty", async () => {
    const { outcome, calls, outputs } = await run(GRID, "root", { inputs: { components: [{ name: "badge" }], themes: [] } });
    expect(outcome).toBe("success");
    expect(calls).toEqual([]);
    expect(outputs?.docs).toEqual([]);
  });
});

describe("what the loader and validator say about `each`", () => {
  const messages = (files: Record<string, StateDef>, rootId: string): string =>
    validateBundle(loadBundle(files, rootId), {})
      .errors.map((e) => `${e.path}: ${e.message}`)
      .join("\n");

  it("accepts the fan-out above as written", () => {
    expect(messages(FAN, "root")).toBe("");
  });

  it("refuses `each` anywhere but a mount's inputs", () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        inputs: { items: { schema: { type: "array" } } },
        outputs: { first: { schema: {}, binding: { expr: ".inputs.items", each: true } as never } },
        operation: { kind: "prompt", model: "m", prompt: "go" },
      },
    };
    expect(() => loadBundle(files, "root")).toThrow(/'each' is only legal on a child mount's inputs/);
  });

  it("checks an `each` wire as an ARRAY of what the child declares", () => {
    // `components` is a list of strings; the child wants an object per element.
    const files: Record<string, StateDef> = {
      ...FAN,
      root: { ...FAN.root!, inputs: { components: { schema: { type: "array", items: { type: "string" } } } } },
    };
    expect(messages(files, "root")).toMatch(/children\.component\.inputs\.component: .*not allowed by consumer/);
  });

  it("refuses `.each` on a mount that does not fan out, and in an output", () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        inputs: { component: { schema: { type: "object" } } },
        outputs: { where: { schema: {}, binding: ".each.index" } },
        children: { component: { state: "root/component", inputs: { component: ".inputs.component", position: ".each.index" } } },
        sequence: ["component"],
      },
      "root/component": COMPONENT,
    };
    const report = messages(files, "root");
    expect(report).toMatch(/children\.component\.inputs\.position: '\.each' is only readable in the wiring of a child mount that fans out/);
    expect(report).toMatch(/outputs\.where: '\.each' is only readable/);
  });

  it("types the fanned-out child's outputs as arrays in the parent", () => {
    // A consumer declared as a STRING cannot take what is now a list of strings.
    const files: Record<string, StateDef> = {
      ...FAN,
      root: { ...FAN.root!, outputs: { docs: { schema: { type: "string" }, binding: ".children.component.output.doc" } } },
    };
    expect(messages(files, "root")).toMatch(/outputs\.docs: .*producer type 'array' not allowed by consumer string/);
  });

  it("names an axis that does not exist", () => {
    const files: Record<string, StateDef> = {
      ...FAN,
      root: { ...FAN.root!, children: { component: { ...FAN.root!.children!.component!, inputs: { ...FAN.root!.children!.component!.inputs, position: ".each.axis.flows" } } } },
    };
    expect(messages(files, "root")).toMatch(/children\.component\.inputs\.position: .*\.each\.axis\.flows/);
  });
});

describe("a stopped fan-out, loaded", () => {
  /** A terminated element as its description: entered under `element`, its call completed. */
  const element = (id: string, element: number, name: string): LoadedInstance => ({
    id,
    stateId: "root/component",
    childKey: "component",
    occurrence: 0,
    element,
    inputs: { component: { name } },
    live: false,
    outcome: "success",
    operation: { value: { doc: `recorded ${name}`, size: name.length } as ResolvedValue },
  });

  it("comes back as one record holding every element's outputs, dispatching nothing", async () => {
    const { outcome, calls, outputs } = await run(FAN, "root", {
      inputs: { components: THREE },
      loaded: {
        id: "i-root",
        stateId: "root",
        inputs: { components: THREE },
        live: true,
        cursor: 0,
        unanswered: ["component"],
        children: [element("i-0", 0, "divider"), element("i-1", 1, "badge"), element("i-2", 2, "toggle")],
      },
    });
    expect(outcome).toBe("success");
    expect(calls).toEqual([]);
    expect(outputs).toEqual({ docs: ["recorded divider", "recorded badge", "recorded toggle"], sizes: [7, 5, 6] });
  });

  it("enters the elements the stopped run never reached, and only those", async () => {
    // Stopped after the first element, mid-batch: the second is live (entered, never settled) and
    // the third was never entered. The live one re-dispatches; the third runs for the first time.
    const { outcome, calls, outputs } = await run(FAN, "root", {
      inputs: { components: THREE },
      loaded: {
        id: "i-root",
        stateId: "root",
        inputs: { components: THREE },
        live: true,
        cursor: 0,
        children: [
          element("i-0", 0, "divider"),
          { id: "i-1", stateId: "root/component", childKey: "component", occurrence: 0, element: 1, inputs: { component: { name: "badge" } }, live: true },
        ],
      },
    });
    expect(outcome).toBe("success");
    expect(calls.map((c) => c.name)).toEqual(["badge", "toggle"]);
    expect(outputs?.docs).toEqual(["recorded divider", "doc for badge", "doc for toggle"]);
  });
});
