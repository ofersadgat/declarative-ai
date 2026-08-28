/**
 * Resume as a deterministic FAST-FORWARD: a run walks itself again, and every operation a stopped
 * run already answered is taken rather than dispatched.
 *
 * The property under test is not "the same outputs come out" — that would pass just as well if the
 * calls were re-made. It is that the CALLS DO NOT HAPPEN. Every workflow here counts its own
 * dispatches, because an operation with side effects is the thing a resume must not run twice, and a
 * count is the only assertion that says so directly.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import type { JsonValue } from "@declarative-ai/json";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { InstanceAddress, ReplayedOperation, ReplaySource, WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** The canonical string form of an address, and the key these tests index by. */
function key(address: InstanceAddress): string {
  return address.map((step) => `${step.childKey}#${step.occurrence}`).join("/");
}

/** A source backed by a plain map of address key → what that operation returned. */
function sourceOf(answers: Record<string, ResolvedValue>, asked: string[] = []): ReplaySource {
  return {
    operationAt(address: InstanceAddress): ReplayedOperation | undefined {
      asked.push(key(address));
      const value = answers[key(address)];
      return value === undefined ? undefined : { value };
    },
  };
}

interface RunOutcome {
  outcome: string;
  outputs: Record<string, unknown> | undefined;
  /** Which states actually DISPATCHED, in order — the assertion that matters. */
  dispatched: string[];
}

/**
 * Run a workflow whose leaves are `tally` functions.
 *
 * `tally` takes the answer it should give and records that it was asked, so a replayed operation is
 * distinguishable from a re-run one by absence rather than by value.
 */
async function run(
  files: Record<string, StateDef>,
  rootId: string,
  replay?: ReplaySource,
): Promise<RunOutcome> {
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
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    ...(replay !== undefined ? { replay } : {}),
  });
  const result = await engine.run({ inputs: {} });
  return {
    outcome: result.outcome,
    outputs: result.outputs as Record<string, unknown> | undefined,
    dispatched,
  };
}

const leaf = (name: string, answer: JsonValue): StateDef => ({
  label: name,
  outputs: { answer: { schema: {} } },
  operation: { kind: "function", function: "tally", args: { name, answer } },
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
  // precisely why the address cannot be that hash.
  "root/loop/tick": leaf("tick", "again"),
};

describe("a run with no replay source", () => {
  it("dispatches everything, as it always did", async () => {
    const result = await run(FLAT, "root");
    expect(result.outcome).toBe("success");
    expect(result.dispatched).toEqual(["a", "b"]);
    expect(result.outputs).toEqual({ first: "from a", second: "from b" });
  });
});

describe("a resumed run", () => {
  it("takes a recorded answer instead of making the call", async () => {
    const asked: string[] = [];
    const result = await run(FLAT, "root", sourceOf({ "a#0": { answer: "recorded a" }, "b#0": { answer: "recorded b" } }, asked));

    // NOTHING ran. This is the safety property: an operation that already happened — a file written,
    // a changeset applied — is not dispatched on the way back to where the run stopped.
    expect(result.dispatched).toEqual([]);
    expect(result.outcome).toBe("success");
    // …and the recorded values flowed through the state's declared outputs and the root's bindings
    // exactly as dispatched ones would, which is what makes a replayed state indistinguishable
    // downstream from one that ran.
    expect(result.outputs).toEqual({ first: "recorded a", second: "recorded b" });
    expect(asked).toEqual(["a#0", "b#0"]);
  });

  it("runs for real from the point the answers stop", async () => {
    // The frontier is not declared anywhere — it is wherever the source starts saying `undefined`.
    const result = await run(FLAT, "root", sourceOf({ "a#0": { answer: "recorded a" } }));
    expect(result.dispatched).toEqual(["b"]);
    expect(result.outputs).toEqual({ first: "recorded a", second: "from b" });
  });

  it("addresses a loop's iterations apart, though their operations are identical", async () => {
    const asked: string[] = [];
    // Two recorded rounds, then nothing: the third `tick` is the frontier and runs for real, and
    // because it answers "again" the loop keeps going until the recorded pattern is exhausted.
    const result = await run(
      LOOP,
      "root",
      sourceOf(
        {
          "loop#0/tick#0": { answer: "again" },
          "loop#0/tick#1": { answer: "again" },
          "loop#0/tick#2": { answer: "stop" },
        },
        asked,
      ),
    );
    expect(result.outcome).toBe("success");
    expect(result.dispatched).toEqual([]);
    expect(result.outputs).toEqual({ last: "stop" });
    // The root is never asked — it has no operation — so the addresses seen are exactly the leaves',
    // each iteration under its own occurrence.
    expect(asked).toEqual(["loop#0/tick#0", "loop#0/tick#1", "loop#0/tick#2"]);
  });

  it("re-enters the loop for real once the record runs out", async () => {
    const result = await run(LOOP, "root", sourceOf({ "loop#0/tick#0": { answer: "again" } }));
    // Round one replayed, round two dispatched — and round two says "again" too, so the loop keeps
    // running live until `max_iterations` stops it.
    expect(result.dispatched.length).toBeGreaterThan(0);
    expect(result.dispatched.every((name) => name === "tick")).toBe(true);
  });
});

describe("what a replayed operation reports", () => {
  it("carries the conversation position the recorded call ended at", async () => {
    // `operation.output.session` has to read on a replayed state as it did on the original: a later
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
    const engine = new WorkflowEngine({
      bundle: loadBundle(files, "root"),
      registry: newRegistry(),
      replay: {
        operationAt: (address) =>
          key(address) === "a#0"
            ? { value: { answer: "x" }, session: { position: "plan@7", conversation: "plan" } }
            : undefined,
      },
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ session: { id: "plan@7", end: { id: "plan" } } });
  });

  it("charges nothing — the run that paid for the call was the other one", async () => {
    const engine = new WorkflowEngine({
      bundle: loadBundle(FLAT, "root"),
      registry: newRegistry(),
      replay: sourceOf({ "a#0": { answer: "a" }, "b#0": { answer: "b" } }),
    });
    const result = await engine.run({ inputs: {} });
    // A dispatched `tally` reports `costUsd: 0.01` through `ok()`. Rolling recorded metrics in here
    // would bill a resumed run for calls it did not make, and every roll-up over the task would
    // then double-count.
    expect(result.metrics.childCost).toBe(0);
    expect(result.metrics.childLlmCalls).toBe(0);
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
    const result = await run(files, "root", sourceOf({ "a#0": { answer: 42 } }));
    expect(result.outcome).toBe("error");
    expect(result.dispatched).toEqual([]);
  });
});
