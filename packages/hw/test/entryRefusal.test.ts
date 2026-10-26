/**
 * A child the HOST refuses to enter (`EngineConfig.entryRefusal`) is a blocked child.
 *
 * The host's reason for a state it has switched off reaches the journal as `instance.blocked`, the
 * child's record fails, and the failure takes the seam a wiring failure takes: unhandled, the parent
 * fails; named by a rule on the mount, the run carries on. Nothing of a refused child runs — not its
 * operation, not a call its wiring makes, not one element of a fan-out.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

const leaf = (name: string): StateDef =>
  ({
    label: name,
    outputs: { done: { schema: { type: "string" } } },
    operation: { kind: "function", function: "mark", args: { name } },
  }) as StateDef;

async function run(
  files: Record<string, StateDef>,
  refuse: (stateId: string) => string | undefined,
  inputs: Record<string, ResolvedValue> = {},
): Promise<{ outcome: string; reason?: string; order: string[]; events: Array<Record<string, unknown>>; asked: string[] }> {
  const order: string[] = [];
  const asked: string[] = [];
  const events: Array<Record<string, unknown>> = [];
  const registry = newRegistry();
  registry.functions.set(
    "mark",
    hostFunction(async (inputs: Record<string, unknown>) => {
      order.push((inputs as { name: string }).name);
      return ok({ done: (inputs as { name: string }).name }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, "root"),
    registry,
    entryRefusal: (stateId) => {
      asked.push(stateId);
      return refuse(stateId);
    },
    onEvent: (event) => events.push(event as unknown as Record<string, unknown>),
  });
  const result = await engine.run({ inputs });
  return { outcome: result.outcome, reason: result.failure?.reason, order, events, asked };
}

const sequence = (mount: Record<string, unknown> = {}): Record<string, StateDef> => ({
  root: {
    label: "Root",
    children: { a: { state: "root/a" }, b: { state: "root/b", ...mount }, c: { state: "root/c" } },
  } as StateDef,
  "root/a": leaf("a"),
  "root/b": leaf("b"),
  "root/c": leaf("c"),
});

const OFF = (stateId: string): string | undefined => (stateId === "root/b" ? "'root/b' is disabled" : undefined);

describe("a refused entry (EngineConfig.entryRefusal)", () => {
  it("blocks the child with the host's words, and an unhandled block fails the run", async () => {
    const r = await run(sequence(), OFF);
    expect(r.outcome).toBe("error");
    expect(r.order).toEqual(["a"]);
    const blocked = r.events.find((e) => e["type"] === "instance.blocked");
    expect(blocked).toMatchObject({ stateId: "root/b", childKey: "b", reason: "'root/b' is disabled" });
    expect(r.events.some((e) => e["type"] === "instance.entered" && e["stateId"] === "root/b")).toBe(false);
  });

  it("is a child failure a rule on the mount may handle", async () => {
    const r = await run(sequence({ transitions: [{ to: "c", when: ".children.b.outcome === 'error'" }] }), OFF);
    expect(r.outcome).toBe("success");
    expect(r.order).toEqual(["a", "c"]);
  });

  it("asks at every entry, so an answer that changes mid-run reaches the next one", async () => {
    let off = false;
    const r = await run(
      {
        root: { label: "Root", children: { a: { state: "root/a", transitions: [{ to: "b" }] }, b: { state: "root/b" } }, sequence: ["a"] } as StateDef,
        "root/a": leaf("a"),
        "root/b": leaf("b"),
      },
      (stateId) => {
        if (stateId === "root/a") off = true;
        return off && stateId === "root/b" ? "switched off while a ran" : undefined;
      },
    );
    expect(r.outcome).toBe("error");
    expect(r.order).toEqual(["a"]);
    expect(r.asked).toEqual(["root/a", "root/b"]);
  });

  it("refuses a fan-out once, for the mount, entering no element", async () => {
    const r = await run(
      {
        root: { label: "Root", inputs: { names: { schema: { type: "array", items: { type: "string" } } } }, children: { b: { state: "root/b", inputs: { name: { $expr: ".inputs.names", each: true } } } } } as StateDef,
        "root/b": { ...leaf("b"), inputs: { name: { schema: { type: "string" } } } } as StateDef,
      },
      OFF,
      { names: ["x", "y"] },
    );
    expect(r.outcome).toBe("error");
    expect(r.order).toEqual([]);
    expect(r.events.filter((e) => e["type"] === "instance.blocked")).toHaveLength(1);
  });

  it("does not ask about the run's root", async () => {
    const r = await run(sequence(), () => undefined);
    expect(r.outcome).toBe("success");
    expect(r.asked).toEqual(["root/a", "root/b", "root/c"]);
  });
});
