/**
 * Resume as CONSTRUCTION (Identity and Resume §04): a stopped run is loaded from a description —
 * the tree, the recorded operation values, the live spine — and the evaluation loop is re-entered
 * where each instance's own fields say it stands. Nothing already answered runs again.
 *
 * The property under test is not "the same outputs come out" — that would pass just as well if the
 * calls were re-made. It is that the CALLS DO NOT HAPPEN. Every workflow here counts its own
 * dispatches, because an operation with side effects is the thing a resume must not run twice, and a
 * count is the only assertion that says so directly.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import type { JsonValue } from "@declarative-ai/json";
import { SchemaValidator } from "@declarative-ai/validate";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

interface RunOutcome {
  outcome: string;
  outputs: Record<string, unknown> | undefined;
  metrics: { childLlmCalls: number; childCost: number };
  /** Which states actually DISPATCHED, in order — the assertion that matters. */
  dispatched: string[];
  /** Every `instance.entered` id, in order — the live spine re-states its entry; history never does. */
  entered: string[];
}

/**
 * Load a workflow whose leaves are `tally` functions.
 *
 * `tally` takes the answer it should give and records that it was asked, so a loaded operation is
 * distinguishable from a re-run one by absence rather than by value.
 */
async function load(files: Record<string, StateDef>, rootId: string, loaded: LoadedInstance): Promise<RunOutcome> {
  const dispatched: string[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "tally",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const { name, answer } = inputs as { name: string; answer: JsonValue };
      dispatched.push(name);
      return ok({ answer }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    validator: new SchemaValidator(),
    persistence,
  });
  const result = await engine.loadRun(loaded);
  return {
    outcome: result.outcome,
    outputs: result.outputs as Record<string, unknown> | undefined,
    metrics: { childLlmCalls: result.metrics.childLlmCalls, childCost: result.metrics.childCost },
    dispatched,
    entered: persistence.events.filter(({ event }) => event.type === "instance.entered").map(({ event }) => (event as { instanceId: string }).instanceId),
  };
}

const leaf = (name: string, answer: JsonValue): StateDef => ({
  label: name,
  outputs: { answer: { schema: {} } },
  operation: { kind: "function", function: "tally", args: { name, answer } },
});

/** A terminated leaf as its description: entered, its call completed, its end recorded. */
const done = (id: string, stateId: string, childKey: string, value: JsonValue, occurrence = 0): LoadedInstance => ({
  id,
  stateId,
  childKey,
  occurrence,
  inputs: {},
  live: false,
  outcome: "success",
  operation: { value: { answer: value } as ResolvedValue },
});

/** Root → two leaves in a sequence, the second reading the first. */
const FLAT: Record<string, StateDef> = {
  root: {
    label: "Root",
    outputs: {
      first: { schema: {}, binding: ".children.a.output.answer" },
      second: { schema: {}, binding: ".children.b.output.answer" },
    },
    children: { a: { state: "root/a" }, b: { state: "root/b" } },
    sequence: ["a", "b"],
  },
  "root/a": leaf("a", "from a"),
  "root/b": leaf("b", "from b"),
};

/** Root → `loop` → `tick`, where `loop` re-enters `tick` until it is told to stop. */
const LOOP: Record<string, StateDef> = {
  root: {
    label: "Root",
    outputs: { last: { schema: {}, binding: ".children.loop.output.last" } },
    children: { loop: { state: "root/loop" } },
    sequence: ["loop"],
  },
  "root/loop": {
    label: "Loop",
    outputs: { last: { schema: {}, binding: ".children.tick.output.answer" } },
    children: { tick: { state: "root/loop/tick" } },
    sequence: ["tick"],
    transitions: [
      { to: "tick", when: ".run.cursor === 'tick' && .children.tick.output.answer === 'again' && .run.iteration < .limits.max_iterations" },
      { to: "terminate.success", when: ".run.cursor === 'tick' && .children.tick.output.answer !== 'again'" },
    ],
    limits: { max_iterations: 5 },
  },
  // The same operation every time, so its content hash is identical on every iteration — which is
  // precisely why an occurrence is part of the description rather than derivable from the op.
  "root/loop/tick": leaf("tick", "again"),
};

describe("a loaded run", () => {
  it("re-dispatches nothing when the description already answers everything", async () => {
    // The run stopped after `b` finished and before the round that would have answered it: both
    // leaves terminated, the root live with `b` still owed an evaluation.
    const result = await load(FLAT, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 1,
      unanswered: ["b"],
      children: [done("i-a", "root/a", "a", "recorded a"), done("i-b", "root/b", "b", "recorded b")],
    });

    // NOTHING ran. This is the safety property: an operation that already happened — a file written,
    // a changeset applied — is not dispatched on the way to the end.
    expect(result.dispatched).toEqual([]);
    expect(result.outcome).toBe("success");
    // …and the recorded values flowed through each state's declared outputs and the root's bindings
    // exactly as dispatched ones would — a state's outputs are a pure function of its recorded
    // operation value and the pinned definition, recomputed rather than stored.
    expect(result.outputs).toEqual({ first: "recorded a", second: "recorded b" });
    // Only the live spine re-states its entry; the terminated leaves are history, journaled once by
    // the run that did the work and never again.
    expect(result.entered).toEqual(["i-root"]);
    // Spend belongs to the run that paid it. A dispatched `tally` reports cost through `ok()`;
    // rolling recorded metrics in here would bill this run for calls it did not make.
    expect(result.metrics.childCost).toBe(0);
    expect(result.metrics.childLlmCalls).toBe(0);
  });

  it("runs for real from the point the description stops", async () => {
    // The frontier is not declared anywhere — it is wherever the description's answers end.
    const result = await load(FLAT, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 0,
      unanswered: ["a"],
      children: [done("i-a", "root/a", "a", "recorded a")],
    });
    expect(result.dispatched).toEqual(["b"]);
    expect(result.outputs).toEqual({ first: "recorded a", second: "from b" });
  });

  it("dispatches an active leaf again when its call never settled", async () => {
    // `b` was ENTERED and its operation was cut mid-flight: live, no completed operation. The loaded
    // run re-issues the call — and because the instance keeps its recorded id, the scoped record id
    // recomputes identically and a durable store reopens the interrupted ask rather than adding one.
    const result = await load(FLAT, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 1,
      children: [
        done("i-a", "root/a", "a", "recorded a"),
        { id: "i-b", stateId: "root/b", childKey: "b", inputs: {}, live: true },
      ],
    });
    expect(result.dispatched).toEqual(["b"]);
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ first: "recorded a", second: "from b" });
    // The live spine — and only the live spine — re-enters under its recorded ids.
    expect(result.entered).toEqual(["i-root", "i-b"]);
  });

  it("holds a loop's iterations apart by occurrence, though their operations are identical", async () => {
    // Three recorded rounds, the last saying "stop", and the loop still owing the evaluation that
    // would have read it.
    const result = await load(LOOP, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 0,
      children: [
        {
          id: "i-loop",
          stateId: "root/loop",
          childKey: "loop",
          inputs: {},
          live: true,
          cursor: 0,
          index: 2,
          iteration: 2,
          unanswered: ["tick"],
          children: [
            done("i-t0", "root/loop/tick", "tick", "again", 0),
            done("i-t1", "root/loop/tick", "tick", "again", 1),
            done("i-t2", "root/loop/tick", "tick", "stop", 2),
          ],
        },
      ],
    });
    expect(result.outcome).toBe("success");
    expect(result.dispatched).toEqual([]);
    // The record read is the LAST occurrence's — the loop's `.children.tick` is its newest entry.
    expect(result.outputs).toEqual({ last: "stop" });
  });

  it("re-enters the loop for real once the description runs out", async () => {
    // Round one recorded and unanswered; it says "again", so the loaded evaluation transitions back
    // into `tick` and the loop runs LIVE — fresh instances, fresh occurrences — until `max_iterations`.
    const result = await load(LOOP, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 0,
      children: [
        {
          id: "i-loop",
          stateId: "root/loop",
          childKey: "loop",
          inputs: {},
          live: true,
          cursor: 0,
          unanswered: ["tick"],
          children: [done("i-t0", "root/loop/tick", "tick", "again", 0)],
        },
      ],
    });
    expect(result.dispatched.length).toBeGreaterThan(0);
    expect(result.dispatched.every((name) => name === "tick")).toBe(true);
  });
});

describe("what a loaded operation reports", () => {
  it("carries the conversation position the recorded call ended at", async () => {
    // `operation.output.session` has to read on a loaded state as it did on the original: a later
    // state binding `{"expr": "...operation.output.session"}` is binding to a position, and a resume
    // that dropped it would resolve that expression to nothing.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { session: { schema: {}, binding: ".children.a.operation.output.session" } },
        children: { a: { state: "root/a" } },
        sequence: ["a"],
      },
      "root/a": leaf("a", "x"),
    };
    const result = await load(files, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 0,
      unanswered: ["a"],
      children: [
        {
          id: "i-a",
          stateId: "root/a",
          childKey: "a",
          inputs: {},
          live: false,
          outcome: "success",
          operation: { value: { answer: "x" } as ResolvedValue, sessionRef: "plan@7" },
        },
      ],
    });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ session: { id: "plan@7", end: { id: "plan" } } });
  });

  it("fails loudly when a recorded value does not satisfy the state's outputs", async () => {
    // Unreachable in principle — the definition is pinned, so a shape that satisfied it once still
    // does — which is exactly why it must be a failure rather than a quiet re-dispatch if it happens.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        children: { a: { state: "root/a" } },
        sequence: ["a"],
      },
      "root/a": {
        label: "a",
        outputs: { answer: { schema: { type: "string" } } },
        operation: { kind: "function", function: "tally", args: { name: "a", answer: "x" } },
      },
    };
    const result = await load(files, "root", {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 0,
      unanswered: ["a"],
      children: [done("i-a", "root/a", "a", 42)],
    });
    expect(result.outcome).toBe("error");
    expect(result.dispatched).toEqual([]);
  });
});
