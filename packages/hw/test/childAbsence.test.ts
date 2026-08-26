/**
 * A child that HAS NOT RUN is `undefined` (SPEC §3.4), not a resolution failure.
 *
 * The two spellings of the same path used to disagree: `".children.c.outputs.x"` lowers to a
 * producer edge and REFUSED, while `{ expr: ".children.c.outputs.x" }` lowers to a `member` chain
 * and yielded `undefined`. Since a guard is always lowered as an expression and a wire usually is
 * not, the split read as "guards are lenient, wiring is strict" — which is not a rule anyone wrote.
 *
 * Whether an absent value is ACCEPTABLE here is the consuming slot's to say (`optional`/`default`,
 * §7.2's named opt-out). Whether it is REACHABLE at all is the validator's. Neither is the
 * resolver's, and this is what pins that down.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** Run a workflow whose leaves record the inputs they were entered with. */
async function run(files: Record<string, StateDef>): Promise<{
  outcome: string;
  reason?: string;
  seen: Record<string, unknown>;
  order: string[];
}> {
  const seen: Record<string, unknown> = {};
  const order: string[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "mark",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const { name, later, delayMs } = inputs as { name: string; later?: unknown; delayMs?: number };
      order.push(name);
      seen[name] = later;
      if (delayMs !== undefined) await new Promise((resolve) => setTimeout(resolve, delayMs));
      return ok({ done: name }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({ bundle: loadBundle(files, "root"), registry });
  const result = await engine.run({ inputs: {} });
  return { outcome: result.outcome, reason: result.failure?.reason, seen, order };
}

/** A leaf that reports the value its `later` slot was entered with. */
const leaf = (name: string, opts: { optional?: boolean; delayMs?: number } = {}): StateDef =>
  ({
    label: name,
    inputs: { later: { schema: {}, ...(opts.optional === true ? { optional: true } : {}) } },
    outputs: { done: { schema: { type: "string" } } },
    operation: {
      kind: "function",
      function: "mark",
      // `args` is JSON config, `input` is where a BINDING goes — the leaf has to read the slot it
      // was entered with, not the string ".inputs.later".
      args: { name, ...(opts.delayMs !== undefined ? { delayMs: opts.delayMs } : {}) },
      // Read as an EXPRESSION: `.inputs.later` lowers to `scope.get`, which refuses an input that
      // was never set — deliberate, and a different question from the one under test here.
      input: { later: { binding: { expr: ".inputs.later" }, optional: true } },
    },
  }) as StateDef;

/** `b` reads `c`, which runs AFTER it. `wire` is the spelling under test. */
const files = (wire: unknown, opts: { optional?: boolean } = {}): Record<string, StateDef> => ({
  root: {
    label: "Root",
    children: { b: { state: "root/b", inputs: { later: wire } }, c: { state: "root/c" } },
  } as StateDef,
  "root/b": leaf("b", opts),
  "root/c": leaf("c", { optional: true }),
});

const EDGE = ".children.c.outputs.done";

describe("a child that has not run (SPEC §3.4)", () => {
  it("resolves to `undefined`, and an OPTIONAL slot absorbs it", async () => {
    const r = await run(files(EDGE, { optional: true }));
    expect(r.outcome).toBe("success");
    expect(r.seen["b"]).toBeUndefined();
    expect(r.order).toEqual(["b", "c"]);
  });

  it("means the SAME THING written as an expression — the two forms agree", async () => {
    const asEdge = await run(files(EDGE, { optional: true }));
    const asExpr = await run(files({ expr: EDGE }, { optional: true }));
    expect(asExpr.outcome).toBe(asEdge.outcome);
    expect(asExpr.seen["b"]).toBe(asEdge.seen["b"]);
    expect(asExpr.order).toEqual(asEdge.order);
  });

  it("is still an error when the slot did NOT opt out — absence is refused where it matters", async () => {
    const r = await run(files(EDGE));
    expect(r.outcome).toBe("error");
    // `b` is never ENTERED: a required slot with nothing to fill it blocks the child, exactly as
    // before. What changed is only that the refusal now comes from the SLOT, not the resolver.
    expect(r.order).not.toContain("b");
  });
});

describe("a block names the MOUNT, not just the state", () => {
  it("carries the parent and the child key, so a reader is told WHERE it happened", async () => {
    // `instanceId` is -1: nothing became an instance, which is the event. The mount is therefore the
    // only address the block has — and one state definition is mounted under several keys in several
    // parents, so the definition's id alone does not say which of them could not be entered.
    const events: Array<Record<string, unknown>> = [];
    const registry = newRegistry();
    registry.functions.set(
      "mark",
      hostFunction(async () => ok({ done: "x" }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST),
    );
    const engine = new WorkflowEngine({
      bundle: loadBundle(files(EDGE), "root"),
      registry,
      onEvent: (event: Record<string, unknown>) => events.push(event),
    } as never);
    await engine.run({ inputs: {} });
    const blocked = events.find((e) => e["type"] === "instance.blocked");
    expect(blocked).toMatchObject({ stateId: "root/b", childKey: "b", parentInstanceId: 1 });
  });
});

describe("in flight is not absent", () => {
  it("PARKS on an async child rather than reading it as `undefined`", async () => {
    // The distinction §6 draws: a child that has STARTED but not finished is PENDING, so the
    // consumer waits for the value. Collapsing that to `undefined` would hand a consumer nothing
    // and call it success — the silent-wrong-answer this whole change has to not introduce.
    const r = await run({
      root: {
        label: "Root",
        children: {
          slow: { state: "root/slow", async: true },
          reader: { state: "root/reader", inputs: { later: ".children.slow.outputs.done" } },
        },
      } as StateDef,
      "root/slow": leaf("slow", { optional: true, delayMs: 20 }),
      "root/reader": leaf("reader", { optional: true }),
    });
    expect(r.outcome).toBe("success");
    expect(r.seen["reader"]).toBe("slow");
  });
});
