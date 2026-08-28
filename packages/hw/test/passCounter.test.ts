/**
 * `run.index` vs `run.iteration` (SPEC §3.4): every transition, versus the ones that go BACK.
 *
 * The split exists because a forward jump used to spend a loop budget. A state whose first child
 * says `{ "when": …, "to": "draft" }` skips ahead, and under one counter that skip was
 * indistinguishable from a re-plan — so `run.iteration < limits.max_iterations` allowed one fewer
 * pass than it said, silently, and only in the branch where the jump fired.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { EngineEvent, WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** Run the bundle and report every `transition.taken`, in order. */
async function transitionsOf(files: Record<string, StateDef>, rootId: string) {
  const events: EngineEvent[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "mark",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const { name } = inputs as { name: string };
      return ok({ done: name }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({ bundle: loadBundle(files, rootId), registry, onEvent: (e) => void events.push(e) });
  const result = await engine.run({ inputs: {} });
  const taken = events.filter((e): e is Extract<EngineEvent, { type: "transition.taken" }> => e.type === "transition.taken");
  return { outcome: result.outcome, taken: taken.map((e) => ({ to: e.to, index: e.index, iteration: e.iteration })) };
}

const marker = (name: string): StateDef => ({
  label: name,
  outputs: { done: { schema: { type: "string" } } },
  operation: { kind: "function", function: "mark", args: { name } },
});

describe("run.index counts transitions; run.iteration counts passes", () => {
  it("a FORWARD jump spends an index and no iteration", async () => {
    // `a` skips `b` and lands on `c`. Nothing has looped, so no pass has been taken.
    const { outcome, taken } = await transitionsOf(
      {
        root: {
          label: "Root",
          children: {
            a: { state: "root/a", transitions: [{ to: "c" }] },
            b: { state: "root/b" },
            c: { state: "root/c" },
          },
        },
        "root/a": marker("a"),
        "root/b": marker("b"),
        "root/c": marker("c"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(taken).toEqual([{ to: "c", index: 1, iteration: 0 }]);
  });

  it("a BACKWARD jump spends both", async () => {
    // `b` goes back to `a` twice, then falls through. Three transitions, two of them passes.
    const { outcome, taken } = await transitionsOf(
      {
        root: {
          label: "Root",
          limits: { max_iterations: 2 },
          children: {
            a: { state: "root/a" },
            b: { state: "root/b", transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }] },
          },
        },
        "root/a": marker("a"),
        "root/b": marker("b"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(taken).toEqual([
      { to: "a", index: 1, iteration: 1 },
      { to: "a", index: 2, iteration: 2 },
    ]);
  });

  it("a child re-entering ITSELF is a pass — that is the loop everyone means", async () => {
    const { outcome, taken } = await transitionsOf(
      {
        root: {
          label: "Root",
          limits: { max_iterations: 1 },
          children: { a: { state: "root/a", transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }] } },
        },
        "root/a": marker("a"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(taken).toEqual([{ to: "a", index: 1, iteration: 1 }]);
  });

  it("the forward jump no longer eats the loop budget", async () => {
    // The shape of `feature/product`: `a` may skip ahead to `b`, and `c` re-plans back to `b`.
    // Under one counter the skip spent a pass and only TWO re-plans fitted a budget of three.
    const { outcome, taken } = await transitionsOf(
      {
        root: {
          label: "Root",
          limits: { max_iterations: 3 },
          children: {
            a: { state: "root/a", transitions: [{ to: "b" }] },
            b: { state: "root/b" },
            c: { state: "root/c", transitions: [{ to: "b", when: ".run.iteration < .limits.max_iterations" }] },
          },
        },
        "root/a": marker("a"),
        "root/b": marker("b"),
        "root/c": marker("c"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    // One forward jump, then a full three passes — not two.
    expect(taken.map((t) => t.iteration)).toEqual([0, 1, 2, 3]);
    expect(taken.map((t) => t.index)).toEqual([1, 2, 3, 4]);
  });

  it("terminating is a transition, never a pass", async () => {
    const { outcome, taken } = await transitionsOf(
      {
        root: { label: "Root", children: { a: { state: "root/a", transitions: [{ to: "terminate.success" }] } } },
        "root/a": marker("a"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(taken).toEqual([{ to: "terminate.success", index: 1, iteration: 0 }]);
  });
});
