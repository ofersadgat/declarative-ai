/**
 * A run that cannot finish must FAIL, and a number nothing can represent must not travel.
 *
 * Both halves of one incident. `explore/verdict` computed `winner - runnerUp` over a single-candidate
 * list, so the subtraction was `0 - undefined` and the output was NaN. `typeof NaN === "number"`, so
 * it satisfied `{"type":"number"}` at every slot it crossed; `JSON.stringify(NaN)` is `null`, so the
 * journal recorded a plausible null and said nothing. Four hops later a memo key was canonicalized —
 * RFC 8785 has no spelling for NaN — and the throw landed inside the detached promise `enterChild`
 * creates for a child. Nothing held a handler for it, so the parent went on waiting for a `notify`
 * that only the normal path signals: the run hung with the process idle and the heartbeat ticking.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { EngineEvent, WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** Fails the test rather than hanging it, so a regression reads as "stalled" and not as a timeout. */
async function within<T>(p: Promise<T>, ms = 4000): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`STALLED: the run never settled in ${ms}ms`)), ms)),
  ]);
}

function engineOver(files: Record<string, StateDef>, record?: (e: EngineEvent) => void): WorkflowEngine {
  const registry = newRegistry();
  registry.functions.set("noop", hostFunction(async () => ok({}) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
  // Produces exactly what `verdict` produced: a well-typed number that is not a number.
  registry.functions.set("nan", hostFunction(async () => ok({ margin: 0 / 0 }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
  return new WorkflowEngine({
    bundle: loadBundle(files, "root"),
    registry,
    ...(record !== undefined ? { persistence: { record } } : {}),
  });
}

const leaf = (fn: string, outputs: Record<string, unknown>): StateDef =>
  ({ outputs, operation: { kind: "function", function: fn } }) as StateDef;

describe("a throw inside a child's own execution", () => {
  /** The journal is the one host callback the engine lets throw — so it stands in for any of them. */
  const files: Record<string, StateDef> = {
    root: { children: { c: { state: "root/c" } } } as StateDef,
    "root/c": leaf("noop", { done: { schema: {}, optional: true } }),
  };

  it("fails the run instead of stranding it", async () => {
    const engine = engineOver(files, (e) => {
      if (e.type === "instance.entered" && e.stateId === "root/c") throw new Error("journal exploded");
    });
    const result = await within(engine.run({ inputs: {} }));
    expect(result.outcome).toBe("error");
    // The cause travels, rather than the bare "terminated with error" the parent used to report.
    expect(result.failure?.reason).toContain("journal exploded");
  });

  it("still succeeds when nothing throws — the guard is not a behaviour change", async () => {
    const result = await within(engineOver(files).run({ inputs: {} }));
    expect(result.outcome).toBe("success");
  });
});

describe("a non-finite number", () => {
  it("is refused by the slot that produces it, naming the slot", async () => {
    const files: Record<string, StateDef> = {
      root: { children: { c: { state: "root/c" } } } as StateDef,
      "root/c": leaf("nan", { margin: { schema: { type: "number" } } }),
    };
    const result = await within(engineOver(files).run({ inputs: {} }));
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toContain("margin");
    expect(result.failure?.reason).toContain("NaN");
  });

  /**
   * The dangerous half. A slot check cannot reach this: a guard is READ and thrown away, so a NaN
   * here never crosses a boundary anything validates — it just answers `false` to both a comparison
   * and its negation, and the run takes whichever branch was written second.
   */
  it("is refused in a transition GUARD, where it would otherwise silently pick a branch", async () => {
    const files: Record<string, StateDef> = {
      root: {
        inputs: { xs: { schema: {} } },
        children: {
          // `sum(pluck(xs,'total'))` is the documented idiom — and one row without `total` is enough.
          a: { state: "root/a", transitions: [{ when: "sum(pluck(.inputs.xs, 'total')) > 0", to: "b" }] },
          b: { state: "root/b" },
        },
      } as StateDef,
      "root/a": leaf("noop", { done: { schema: {}, optional: true } }),
      "root/b": leaf("noop", { done: { schema: {}, optional: true } }),
    };
    const result = await within(engineOver(files).run({ inputs: { xs: [{ total: 1 }, { other: 9 }] } as never }));
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toContain("NaN");
  });

  it("leaves an ordinary comparison alone — the guard is not a tax on arithmetic that works", async () => {
    const files: Record<string, StateDef> = {
      root: {
        inputs: { xs: { schema: {} } },
        children: {
          a: { state: "root/a", transitions: [{ when: "sum(pluck(.inputs.xs, 'total')) > 0", to: "b" }] },
          b: { state: "root/b" },
        },
      } as StateDef,
      "root/a": leaf("noop", { done: { schema: {}, optional: true } }),
      "root/b": leaf("noop", { done: { schema: {}, optional: true } }),
    };
    const result = await within(engineOver(files).run({ inputs: { xs: [{ total: 1 }, { total: 2 }] } as never }));
    expect(result.outcome).toBe("success");
  });

  it("is refused even where the slot is unconstrained — schemas cannot express this", async () => {
    const files: Record<string, StateDef> = {
      root: { children: { c: { state: "root/c" } } } as StateDef,
      "root/c": leaf("nan", { margin: { schema: {} } }),
    };
    const result = await within(engineOver(files).run({ inputs: {} }));
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toContain("NaN");
  });
});
