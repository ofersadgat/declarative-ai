/**
 * The derived sequence at RUN time (§6): what actually executes when no `sequence` is written, and
 * what a transition into a member does to the cursor.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/**
 * Run a workflow whose leaves are `mark` functions, and report the order they ran in.
 *
 * Each mark logs `enter <name>`/`exit <name>` around a delay, so the log shows OVERLAP and not just
 * ordering — `enter a | enter b | exit b | exit a` is two children running at once.
 */
async function runMarking(files: Record<string, StateDef>, rootId: string): Promise<{ order: string[]; log: string[]; outcome: string }> {
  const order: string[] = [];
  const log: string[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "mark",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const { name, delayMs } = inputs.config as { name: string; delayMs?: number };
      order.push(name);
      log.push(`enter ${name}`);
      if (delayMs !== undefined) await new Promise((resolve) => setTimeout(resolve, delayMs));
      log.push(`exit ${name}`);
      return ok({ done: name }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({ bundle: loadBundle(files, rootId), registry });
  const result = await engine.run({ inputs: {} });
  return { order, log, outcome: result.outcome };
}

const marker = (name: string, delayMs?: number): StateDef => ({
  label: name,
  outputs: { done: { schema: { type: "string" } } },
  operation: { kind: "function", function: "mark", args: { name, ...(delayMs !== undefined ? { delayMs } : {}) } },
});

describe("a sequence nobody wrote", () => {
  it("runs every child in declaration order", async () => {
    const { order, outcome } = await runMarking(
      {
        root: {
          label: "Root",
          children: { a: { state: "root/a" }, b: { state: "root/b" }, c: { state: "root/c" } },
        },
        "root/a": marker("a"),
        "root/b": marker("b"),
        "root/c": marker("c"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["a", "b", "c"]);
  });

  it("runs them ONE AT A TIME — a plain child is not concurrent", async () => {
    // `a` is slow and `b` is fast: overlapping them would interleave as
    // `enter a | enter b | exit b | exit a`, which is what happened before the cursor held.
    const { log } = await runMarking(
      {
        root: { label: "Root", children: { a: { state: "root/a" }, b: { state: "root/b" } } },
        "root/a": marker("a", 30),
        "root/b": marker("b", 1),
      },
      "root",
    );
    expect(log).toEqual(["enter a", "exit a", "enter b", "exit b"]);
  });

  it("lets `async: true` overlap — the one thing the flag means", async () => {
    const { log } = await runMarking(
      {
        root: { label: "Root", children: { a: { state: "root/a", async: true }, b: { state: "root/b" } } },
        "root/a": marker("a", 30),
        "root/b": marker("b", 1),
      },
      "root",
    );
    expect(log).toEqual(["enter a", "enter b", "exit b", "exit a"]);
  });

  it("runs nothing when the sequence is explicitly empty", async () => {
    const { order, outcome } = await runMarking(
      {
        root: { label: "Root", children: { a: { state: "root/a" } }, sequence: [] },
        "root/a": marker("a"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual([]);
  });
});

describe("a transition is a jump in the sequence", () => {
  /** Three children; the operation's answer decides which one control jumps to. */
  const files = (answer: string): Record<string, StateDef> => ({
    root: {
      label: "Root",
      outputs: { pick: { schema: { type: "string" } } },
      operation: { kind: "function", function: "choose", args: { answer } },
      children: { a: { state: "root/a" }, b: { state: "root/b" }, c: { state: "root/c" } },
      transitions: [
        { to: "terminate.success", when: ".children.b.outcome === 'success'" },
        { to: "b", when: ".outputs.pick === 'b'" },
      ],
    },
    "root/a": marker("a"),
    "root/b": marker("b"),
    "root/c": marker("c"),
  });

  async function run(answer: string): Promise<string[]> {
    const order: string[] = [];
    const registry = newRegistry();
    registry.functions.set(
      "mark",
      hostFunction(async (inputs: Record<string, unknown>) => {
        order.push((inputs.config as { name: string }).name);
        return ok({ done: "x" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    registry.functions.set(
      "choose",
      hostFunction(
        async (inputs: Record<string, unknown>) =>
          ok({ pick: (inputs.config as { answer: string }).answer }) as ExecResult<ResolvedValue, WorkflowMetrics>,
        HOST,
      ),
    );
    const engine = new WorkflowEngine({ bundle: loadBundle(files(answer), "root"), registry });
    await engine.run({ inputs: {} });
    return order;
  }

  it("jumps FORWARD past the members it skipped, and does not come back for them", async () => {
    // `b` is index 1: `a` is skipped, and the guard on `b` then terminates before `c`.
    expect(await run("b")).toEqual(["b"]);
  });

  it("walks the spine one child at a time, evaluating transitions between them", async () => {
    // No transition matches after `a`, so the cursor advances to `b` — and the guard on `b` then
    // terminates before `c` is ever entered. Under the old overlapping advance all three had
    // already started before the first evaluation round, and `c` ran for nothing.
    expect(await run("none")).toEqual(["a", "b"]);
  });

  it("lets a guard read where the cursor is", async () => {
    // "if the cursor is at b and the answer was 'none', skip to c" — the position IS the condition.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { pick: { schema: { type: "string" } } },
        operation: { kind: "function", function: "choose", args: { answer: "none" } },
        children: { a: { state: "root/a" }, b: { state: "root/b" }, c: { state: "root/c" } },
        transitions: [
          { to: "terminate.success", when: ".run.cursor === 'c'" },
          { to: "c", when: ".run.cursor === 'a' && .outputs.pick === 'none'" },
        ],
      },
      "root/a": marker("a"),
      "root/b": marker("b"),
      "root/c": marker("c"),
    };
    const order: string[] = [];
    const registry = newRegistry();
    registry.functions.set(
      "mark",
      hostFunction(async (inputs: Record<string, unknown>) => {
        order.push((inputs.config as { name: string }).name);
        return ok({ done: "x" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    registry.functions.set(
      "choose",
      hostFunction(
        async (inputs: Record<string, unknown>) =>
          ok({ pick: (inputs.config as { answer: string }).answer }) as ExecResult<ResolvedValue, WorkflowMetrics>,
        HOST,
      ),
    );
    const engine = new WorkflowEngine({ bundle: loadBundle(files, "root"), registry });
    await engine.run({ inputs: {} });
    expect(order).toEqual(["a", "c"]);
  });

  it("holds the cursor while the jumped-to child runs, so the rest does not start alongside it", async () => {
    // Without the hold, entering `b` would fall straight through to `c` in the same round — and an
    // either/or state would run both of its branches.
    const order = await run("b");
    expect(order).not.toContain("c");
  });
});
