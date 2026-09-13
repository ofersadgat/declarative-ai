/**
 * What a fan-out element BECOMES (WORKFLOWS.md §6.3): `each: "inline"` is `each: true`; `"task"`
 * and `"split"` hand the elements to the host's `fanOut` hook instead of entering them. What the
 * tests pin is the contract on both sides of that seam — what the host is handed, what its answer
 * does to the mount's record, that a split ends the state it is in unless a rule says otherwise,
 * that a run on the far side of a split narrows the list to its own element, and that a stopped
 * hosted fan-out comes back through the same hook with its recorded rows.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { newRegistry, ok } from "./fakes.js";
import { combineElements, WorkflowEngine, type FanOutOutcome, type FanOutRequest, type SplitEntry } from "../src/engine.js";
import { loadBundle, WorkflowLoadError } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

interface Outcome {
  outcome: string;
  reason?: string;
  outputs?: Record<string, unknown>;
  /** Every `build` call the engine dispatched itself — an inline element's leaf. */
  calls: Array<Record<string, unknown>>;
  /** Every request the host's hook received. */
  requests: FanOutRequest[];
  events: EngineEvent[];
}

async function run(
  files: Record<string, StateDef>,
  rootId: string,
  options: {
    inputs?: Record<string, ResolvedValue>;
    loaded?: LoadedInstance;
    split?: SplitEntry[];
    /** What the host answers. Absent means no host at all. */
    host?: (request: FanOutRequest) => Promise<FanOutOutcome>;
  } = {},
): Promise<Outcome> {
  const calls: Array<Record<string, unknown>> = [];
  const requests: FanOutRequest[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "build",
    hostFunction(async (inputs: Record<string, unknown>) => {
      calls.push(inputs);
      const name = String((inputs.component as { name?: string } | undefined)?.name ?? "");
      return ok({ doc: `doc for ${name}`, size: name.length }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    validator: new SchemaValidator(),
    persistence,
    ...(options.split !== undefined ? { split: options.split } : {}),
    ...(options.host !== undefined
      ? {
          fanOut: async (request: FanOutRequest) => {
            requests.push(request);
            return options.host!(request);
          },
        }
      : {}),
  });
  const result = options.loaded !== undefined ? await engine.loadRun(options.loaded, { inputs: options.inputs ?? {} }) : await engine.run({ inputs: options.inputs ?? {} });
  return {
    outcome: result.outcome,
    ...(result.failure ? { reason: result.failure.reason } : {}),
    outputs: result.outputs as Record<string, unknown> | undefined,
    calls,
    requests,
    events: persistence.events.map(({ event }) => event),
  };
}

const COMPONENT_SCHEMA = { type: "object", properties: { name: { type: "string" }, id: { type: "string" } }, required: ["name"] } as const;

const COMPONENT: StateDef = {
  label: "Component",
  inputs: { component: { schema: COMPONENT_SCHEMA } },
  outputs: { doc: { schema: { type: "string" } }, size: { schema: { type: "integer" } } },
  operation: { kind: "function", function: "build" },
};

/** A leaf that records it ran — what a sibling AFTER a split must never reach in the parent. */
const AFTER: StateDef = {
  label: "After",
  // Untyped: an inline batch hands it the docs gathered, a split's one element hands it one doc.
  inputs: { docs: { schema: {}, optional: true } },
  outputs: { doc: { schema: { type: "string" } }, size: { schema: { type: "integer" } } },
  operation: { kind: "function", function: "build" },
};

/** Root → `component` fanned out with the given kind, then `after` reading the gathered docs. */
function fan(each: unknown, extra: Record<string, unknown> = {}, mount: Record<string, unknown> = {}): Record<string, StateDef> {
  return {
    root: {
      label: "Root",
      inputs: { components: { schema: { type: "array", items: COMPONENT_SCHEMA } } },
      outputs: {
        // Untyped on purpose: an inline or task batch reads `doc` back as an array, a split's one
        // element reads it back as the string it is, and this fixture serves all three.
        docs: { schema: {}, binding: ".children.component.output.doc", optional: true },
        last: { schema: { type: "string" }, binding: ".children.after.output.doc", optional: true },
      },
      children: {
        component: { state: "root/component", inputs: { component: { expr: ".inputs.components", each, ...extra } }, ...mount },
        after: { state: "root/after", inputs: { docs: ".children.component.output.doc" } },
      },
      sequence: ["component", "after"],
    },
    "root/component": COMPONENT,
    "root/after": AFTER,
  };
}

const THREE = [
  { name: "divider", id: "d" },
  { name: "badge", id: "b" },
  { name: "toggle", id: "t" },
];

/** The component names the engine's own `build` calls were handed — `after`'s call carries none. */
function componentsOf(calls: ReadonlyArray<Record<string, unknown>>): string[] {
  return calls.flatMap((call) => {
    const component = call.component as { name?: string } | undefined;
    return component?.name !== undefined ? [component.name] : [];
  });
}

describe("`each: \"inline\"` is `each: true`", () => {
  it("enters the child once per element in this run, and no host is asked", async () => {
    const { outcome, calls, requests, outputs } = await run(fan("inline"), "root", { inputs: { components: THREE } });
    expect(outcome).toBe("success");
    expect(requests).toEqual([]);
    expect(componentsOf(calls)).toEqual(["divider", "badge", "toggle"]);
    expect(outputs?.docs).toEqual(["doc for divider", "doc for badge", "doc for toggle"]);
  });

  it("loads to the same mount as `true` does", () => {
    const spelled = loadBundle(fan("inline"), "root").states.root!.children!.component!;
    const flagged = loadBundle(fan(true), "root").states.root!.children!.component!;
    expect(spelled.each).toEqual(["component"]);
    expect(spelled.eachKind).toBeUndefined();
    expect(spelled).toEqual(flagged);
  });
});

describe("what the loader says about the three kinds", () => {
  it("records a hosted kind, the axis expression, and the element fields with their defaults", () => {
    const mount = loadBundle(fan("task", { title: "name" }), "root").states.root!.children!.component!;
    expect(mount.eachKind).toBe("task");
    expect(mount.eachExprs).toEqual({ component: ".inputs.components" });
    expect(mount.spawn).toEqual({ id: "id", title: "name", requires: "requires", start: "manual" });
  });

  it("refuses a value that is none of the three", () => {
    expect(() => loadBundle(fan("elsewhere"), "root")).toThrow(WorkflowLoadError);
    expect(() => loadBundle(fan("elsewhere"), "root")).toThrow(/must be true, "inline", "task" or "split"/);
    expect(() => loadBundle(fan(false), "root")).toThrow(/must be true, "inline", "task" or "split"/);
  });

  it("refuses element fields on an inline wire, and `start` on a task wire", () => {
    expect(() => loadBundle(fan("inline", { title: "name" }), "root")).toThrow(/mean nothing on each: "inline"/);
    expect(() => loadBundle(fan("task", { start: "when_ready" }), "root")).toThrow(/'start' is only meaningful on each: "split"/);
    expect(() => loadBundle(fan("split", { start: "later" }), "root")).toThrow(/'start' must be "manual" or "when_ready"/);
  });

  it("refuses a mount whose each wires name different kinds, and a split over two lists", () => {
    const files = fan("task");
    const mount = files.root!.children!.component as { inputs: Record<string, unknown> };
    mount.inputs.other = { expr: ".inputs.components", each: "split" };
    expect(() => loadBundle(files, "root")).toThrow(/disagrees with the mount's other each wire/);
    const two = fan("split");
    (two.root!.children!.component as { inputs: Record<string, unknown> }).inputs.other = { expr: ".inputs.components", each: "split" };
    expect(() => loadBundle(two, "root")).toThrow(/each: "split" is on one list/);
  });

  it("still refuses `each` of any spelling off a mount's inputs", () => {
    const files = fan("inline");
    (files.root as { outputs: Record<string, unknown> }).outputs.stray = { schema: { type: "array" }, binding: { expr: ".inputs.components", each: "task" } };
    expect(() => loadBundle(files, "root")).toThrow(/'each' is only legal on a child mount's inputs/);
  });
});

describe("`each: \"task\"` — the elements go to the host, and the run waits for them", () => {
  it("hands the host every element with its inputs resolved, and reads back what the host gathered", async () => {
    const host = async (request: FanOutRequest): Promise<FanOutOutcome> => {
      const terms = request.elements.map((element) => ({
        outcome: "success" as const,
        outputs: { doc: `made ${(element.inputs.component as { name: string }).name}`, size: 1 as ResolvedValue },
      }));
      return combineElements(request.key, undefined, terms);
    };
    const { outcome, calls, requests, outputs } = await run(fan("task", { title: "name" }), "root", { inputs: { components: THREE }, host });
    expect(outcome).toBe("success");
    // Nothing entered here: the engine dispatched only `after`, which read the gathered docs.
    expect(calls).toHaveLength(1);
    expect(requests).toHaveLength(1);
    const request = requests[0]!;
    expect(request.kind).toBe("task");
    expect(request.key).toBe("component");
    expect(request.state).toBe("root/component");
    expect(request.stateId).toBe("root");
    expect(request.occurrence).toBe(0);
    expect(request.async).toBe(false);
    expect(request.spawn).toEqual({ id: "id", title: "name", requires: "requires", start: "manual" });
    expect(request.exprs).toEqual({ component: ".inputs.components" });
    expect(request.elements.map((e) => e.inputs)).toEqual(THREE.map((component) => ({ component })));
    expect(outputs).toEqual({ docs: ["made divider", "made badge", "made toggle"], last: "doc for " });
  });

  it("fails the state, naming the mount, when there is no host to hand the elements to", async () => {
    const { outcome, reason } = await run(fan("task"), "root", { inputs: { components: THREE } });
    expect(outcome).toBe("error");
    expect(reason).toMatch(/child 'component' is each: "task", and this engine has no host/);
  });

  it("is a failed child when the host answers with a failure — handled by a rule like any other", async () => {
    const files = fan("task");
    const host = async (): Promise<FanOutOutcome> => ({ outcome: "error", failure: { classification: "permanent", reason: "element 1 failed" } });
    const { outcome, reason } = await run(files, "root", { inputs: { components: THREE }, host });
    expect(outcome).toBe("error");
    expect(reason).toMatch(/element 1 failed/);
  });

  it("asks the host again with the recorded rows when a stopped run is loaded, never re-making them", async () => {
    const host = async (request: FanOutRequest): Promise<FanOutOutcome> => {
      // The list travels beside the rows, re-read, so a batch stopped short can be finished — and
      // the rows win for every element they cover.
      expect(request.elements).toHaveLength(3);
      const rows = request.loaded ?? [];
      return combineElements(
        request.key,
        undefined,
        [...rows].sort((a, b) => (a.element ?? 0) - (b.element ?? 0)).map((row) => ({ outcome: "success" as const, outputs: { doc: `kept ${(row.inputs.component as { name: string }).name}` } })),
      );
    };
    const element = (id: string, index: number, name: string): LoadedInstance => ({
      id,
      stateId: "root/component",
      childKey: "component",
      occurrence: 0,
      element: index,
      inputs: { component: { name } },
      live: index === 2,
    });
    const { outcome, reason, requests, outputs } = await run(fan("task"), "root", {
      inputs: { components: THREE },
      host,
      loaded: {
        id: "i-root",
        stateId: "root",
        inputs: { components: THREE },
        live: true,
        cursor: 0,
        children: [element("i-0", 0, "divider"), element("i-1", 1, "badge"), element("i-2", 2, "toggle")],
      },
    });
    expect({ outcome, reason }).toEqual({ outcome: "success" });
    expect(requests).toHaveLength(1);
    expect(requests[0]!.loaded?.map((row) => row.id)).toEqual(["i-0", "i-1", "i-2"]);
    expect(outputs?.docs).toEqual(["kept divider", "kept badge", "kept toggle"]);
  });
});

describe("`each: \"split\"` — the elements go to the host, and this run ends", () => {
  const made = async (request: FanOutRequest): Promise<FanOutOutcome> => ({
    outcome: "success",
    outputs: { tasks: request.elements.map((element, index) => ({ index, id: (element.inputs.component as { id: string }).id })) as ResolvedValue },
  });

  it("ends the state successfully after the mount, so the sibling after it never runs here", async () => {
    const { outcome, calls, requests, outputs } = await run(fan("split"), "root", { inputs: { components: THREE }, host: made });
    expect(outcome).toBe("success");
    expect(requests).toHaveLength(1);
    expect(requests[0]!.kind).toBe("split");
    expect(calls).toEqual([]);
    // The parent's bound outputs read what the host answered, and `after` was never entered.
    expect(outputs?.last).toBeUndefined();
  });

  it("lets a rule on the mount say where to go instead", async () => {
    const files = fan("split", {}, { transitions: [{ to: "after" }] });
    const { outcome, calls } = await run(files, "root", { inputs: { components: THREE }, host: made });
    expect(outcome).toBe("success");
    expect(calls).toHaveLength(1);
  });

  it("on the far side of the split, narrows the list to this run's element and runs it inline", async () => {
    const { outcome, reason, calls, requests, outputs } = await run(fan("split"), "root", {
      inputs: { components: THREE },
      host: made,
      split: [{ expr: ".inputs.components", index: 1 }],
    });
    expect({ outcome, reason }).toEqual({ outcome: "success" });
    expect(requests).toEqual([]);
    // One element ran here, then `after` — the sequence CONTINUES on the far side of the split, and
    // the mount reads as an ordinary mount: the element's own doc, not a one-element array of it.
    expect(componentsOf(calls)).toEqual(["badge"]);
    expect(calls).toHaveLength(2);
    expect(outputs?.docs).toEqual("doc for badge");
  });

  it("journals the narrowed element as element 0 of a one-element batch", async () => {
    const { events } = await run(fan("split"), "root", { inputs: { components: THREE }, host: made, split: [{ expr: ".inputs.components", index: 2 }] });
    const entered = events.filter((e) => e.type === "instance.entered" && e.childKey === "component");
    expect(entered.map((e) => (e.type === "instance.entered" ? [e.element, e.inputs] : undefined))).toEqual([[0, { component: THREE[2] }]]);
  });

  it("blocks the mount when this run's element is no longer in the list", async () => {
    const { outcome, reason } = await run(fan("split"), "root", { inputs: { components: THREE }, host: made, split: [{ expr: ".inputs.components", index: 7 }] });
    expect(outcome).toBe("error");
    expect(reason).toMatch(/element 7 of the list, which now has 3/);
  });

  it("narrows the same list at every later mount that splits on it", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        inputs: { components: { schema: { type: "array", items: COMPONENT_SCHEMA } } },
        outputs: {
          first: { schema: { type: "string" }, binding: ".children.first.output.doc", optional: true },
          second: { schema: { type: "string" }, binding: ".children.second.output.doc", optional: true },
        },
        children: {
          first: { state: "root/component", inputs: { component: { expr: ".inputs.components", each: "split" } } },
          second: { state: "root/component", inputs: { component: { expr: ".inputs.components", each: "split" } } },
        },
        sequence: ["first", "second"],
      },
      "root/component": COMPONENT,
    };
    const { outcome, calls, requests, outputs } = await run(files, "root", {
      inputs: { components: THREE },
      host: made,
      split: [{ expr: ".inputs.components", index: 0 }],
    });
    expect(outcome).toBe("success");
    expect(requests).toEqual([]);
    expect(calls).toHaveLength(2);
    expect(outputs).toEqual({ first: "doc for divider", second: "doc for divider" });
  });
});
