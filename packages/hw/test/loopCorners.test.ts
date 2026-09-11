/**
 * Corners of a child's pass history and of a rule's own wiring, exercised with FUNCTION ops.
 *
 * Function ops rather than prompts on purpose: the behaviour under test is the engine's — when a
 * pass opens, what a reset leaves behind, what a rule can see at the moment it fires — and a
 * scripted function makes each of those observable without a model in the way.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { BindingDecl, StateDef } from "../src/format.js";
import type { EngineEvent, WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };
const result = (v: unknown) => ok(v as never) as ExecResult<ResolvedValue, WorkflowMetrics>;

/** A registry whose functions are declared inline, plus the log every one of them writes to. */
function harness(fns: Record<string, (inputs: Record<string, unknown>, log: string[]) => unknown>) {
  const log: string[] = [];
  const registry = newRegistry();
  for (const [name, fn] of Object.entries(fns)) {
    registry.functions.set(
      name,
      hostFunction(async (inputs: Record<string, unknown>) => result(await fn(inputs, log)), HOST),
    );
  }
  return { registry, log };
}

async function run(files: Record<string, StateDef>, registry: ReturnType<typeof harness>["registry"], root = "root") {
  const events: EngineEvent[] = [];
  const engine = new WorkflowEngine({ bundle: loadBundle(files, root), registry, onEvent: (e) => void events.push(e) });
  const outcome = await engine.run({ inputs: {} });
  return { outcome, events };
}

/** A leaf that records that it ran and answers with whatever it was told to. */
const leaf = (fn: string, outputs: Record<string, unknown>, inputs?: Record<string, unknown>): StateDef =>
  ({
    label: fn,
    ...(inputs ? { inputs } : {}),
    outputs: outputs as StateDef["outputs"],
    operation: { kind: "function", function: fn },
  }) as StateDef;

const str = { schema: { type: "string" } } as const;
const strs = { schema: { type: "array", items: { type: "string" } } } as const;

// ---------------------------------------------------------------------------
// history: what a pass leaves behind
// ---------------------------------------------------------------------------

describe("history under a plain loop", () => {
  /** `a` then `b`; `b` sends control back to `a` twice. Three passes of both. */
  const loop = (bInputs?: Record<string, BindingDecl>): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 2 },
      children: {
        a: { state: "root/a" },
        b: {
          state: "root/b",
          ...(bInputs ? { inputs: bInputs } : {}),
          transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }],
        },
      },
    },
    "root/a": leaf("tick", { n: str }),
    "root/b": leaf("read", { n: str }, { seen: { ...strs, optional: true }, count: { schema: { type: "number" }, optional: true } }),
  });

  it("grows one entry per pass and keeps the OLDEST first", async () => {
    const seen: number[] = [];
    let i = 0;
    const { registry } = harness({
      tick: () => ({ n: `a${++i}` }),
      read: (inputs) => {
        seen.push((inputs as { count?: number }).count ?? -1);
        return { n: "b" };
      },
    });
    const { outcome } = await run(loop({ count: { expr: ".children.a.length" } }), registry);
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual([1, 2, 3]);
  });

  it("lets a guard compare THIS pass with the one before it", async () => {
    // The comparison history exists for: a loop that stops when nothing changed.
    let i = 0;
    const files = loop();
    files["root"]!.children!["b"]!.transitions = [
      { to: "a", when: ".children.a[-1].output.n !== .children.a[-2].output.n && .run.iteration < .limits.max_iterations" },
    ];
    const { registry, log } = harness({
      tick: (_i, l) => {
        i += 1;
        // Third pass repeats the second: the guard should stop there.
        const n = i >= 3 ? "same" : `a${i}`;
        l.push(n);
        return { n };
      },
      read: () => ({ n: "b" }),
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(log).toEqual(["a1", "a2", "same"]);
  });

  it("reads a pass by absolute index as well as from the end", async () => {
    const pairs: Array<[unknown, unknown]> = [];
    let i = 0;
    const files = loop();
    files["root"]!.children!["b"]!.inputs = {
      first: { expr: ".children.a[0].output.n" },
      latest: { expr: ".children.a[-1].output.n" },
    };
    files["root/b"] = leaf("read", { n: str }, { first: { ...str, optional: true }, latest: { ...str, optional: true } });
    const { registry } = harness({
      tick: () => ({ n: `a${++i}` }),
      read: (inputs) => {
        pairs.push([(inputs as { first?: unknown }).first, (inputs as { latest?: unknown }).latest]);
        return { n: "b" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // `[0]` is always the first pass; `[-1]` moves with the loop.
    expect(pairs).toEqual([
      ["a1", "a1"],
      ["a1", "a2"],
      ["a1", "a3"],
    ]);
  });

  it("a child no rule names and no sequence lists has a ROW per pass and a record in none", async () => {
    // The cursor only walks `sequence`, and no rule names `helper` either, so nothing ever enters
    // it. It still occupies an index in every pass — that is what keeps index `i` meaning one pass
    // across every key — and `.length` therefore counts PASSES rather than runs.
    const seen: unknown[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["a", "b"],
        children: {
          helper: { state: "root/helper", async: true },
          a: { state: "root/a" },
          b: {
            state: "root/b",
            inputs: { count: { expr: ".children.helper.length" } },
            transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/helper": leaf("help", { n: str }),
      "root/a": leaf("tick", { n: str }),
      "root/b": leaf("read", { n: str }, { count: { schema: { type: "number" }, optional: true } }),
    };
    const { registry, log } = harness({
      help: (_i, l) => {
        l.push("helped");
        return { n: "helped" };
      },
      tick: () => ({ n: "a" }),
      read: (inputs) => {
        seen.push((inputs as { count?: number }).count);
        return { n: "b" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual([1, 2]);
    // …and every one of those rows is a hole: it never ran.
    expect(log).toEqual([]);
  });
});

describe("history when a pass does not complete cleanly", () => {
  it("keeps a FAILED pass addressable, so the next one can see what broke", async () => {
    const seen: unknown[] = [];
    let attempt = 0;
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        children: {
          // The PARENT handles the failure: a taken rule on the child's mount that NAMES its outcome is what
          // stops an errored child taking the state down with it (SPEC §3.3); the cursor cannot do that by walking.
          work: { state: "root/work", transitions: [{ to: "check", when: ".children.work.outcome === 'error'" }, { to: "check" }] },
          check: {
            state: "root/check",
            inputs: { last: { expr: ".children.work[-1].outcome" } },
            transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/work": { label: "work", outputs: { n: str }, operation: { kind: "function", function: "flaky" } },
      "root/check": leaf("read", { n: str }, { last: { ...str, optional: true } }),
    };
    const { registry } = harness({
      flaky: () => {
        attempt += 1;
        if (attempt === 1) throw new Error("first attempt fails");
        return { n: "ok" };
      },
      read: (inputs) => {
        seen.push((inputs as { last?: unknown }).last);
        return { n: "checked" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // The first pass's outcome is READABLE — that is the point of keeping it — and the second's is
    // its own. A failure is a fact about a pass, not a hole where one used to be.
    expect(seen).toEqual(["error", "success"]);
  });

  it("an ASYNC child still running when the pass turns over is aborted, and its row survives", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["slow", "fast"],
        children: {
          slow: { state: "root/slow", async: true },
          fast: {
            state: "root/fast",
            inputs: { count: { expr: ".children.slow.length" } },
            transitions: [{ to: "slow", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/fast": leaf("read", { n: str }, { count: { schema: { type: "number" }, optional: true } }),
    };
    const seen: unknown[] = [];
    const { registry } = harness({
      slow: async () => {
        await new Promise((r) => setTimeout(r, 15));
        return { n: "slow" };
      },
      read: (inputs) => {
        seen.push((inputs as { count?: number }).count);
        return { n: "fast" };
      },
    });
    const { outcome, events } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // A pass turned over while `slow` was in flight: the reset supersedes it, and the row it left
    // is still one row of history rather than a gap.
    expect(events.some((e) => e.type === "child.superseded" && e.childKey === "slow")).toBe(true);
    expect(seen[seen.length - 1]).toBe(2);
  });
});

describe("nested loops each keep their own passes", () => {
  it("an inner loop's history does not leak into the outer one's", async () => {
    const inner: number[] = [];
    const outer: number[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        children: {
          phase: { state: "root/phase" },
          judge: {
            state: "root/judge",
            inputs: { count: { expr: ".children.phase.length" } },
            transitions: [{ to: "phase", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/phase": {
        label: "phase",
        limits: { max_iterations: 2 },
        outputs: { n: { ...str, binding: ".children.step.output.n" } },
        children: {
          step: { state: "root/phase/step", inputs: { count: { expr: ".children.step.length" } } },
          gate: {
            state: "root/phase/gate",
            transitions: [{ to: "step", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/phase/step": leaf("inner", { n: str }, { count: { schema: { type: "number" }, optional: true } }),
      "root/phase/gate": leaf("gate", { n: str }),
      "root/judge": leaf("outer", { n: str }, { count: { schema: { type: "number" }, optional: true } }),
    };
    const { registry } = harness({
      inner: (inputs) => {
        inner.push((inputs as { count?: number }).count ?? -1);
        return { n: "step" };
      },
      gate: () => ({ n: "gate" }),
      outer: (inputs) => {
        outer.push((inputs as { count?: number }).count ?? -1);
        return { n: "judge" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // The inner state is a FRESH INSTANCE each time the outer loops, so its history restarts —
    // passes belong to an instance, not to a state id.
    expect(inner).toEqual([1, 2, 3, 1, 2, 3]);
    expect(outer).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// a rule's own wiring
// ---------------------------------------------------------------------------

describe("a rule's wiring, in the corners", () => {
  const twoWays = (aInputs?: Record<string, BindingDecl>, bInputs?: Record<string, BindingDecl>): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 2 },
      children: {
        work: { state: "root/work", inputs: { why: { text: "mount" } } },
        a: {
          state: "root/a",
          transitions: [
            {
              to: "work",
              when: ".children.a.output.go === 'yes' && .run.iteration < .limits.max_iterations",
              ...(aInputs ? { inputs: aInputs } : {}),
            },
          ],
        },
        b: {
          state: "root/b",
          transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations", ...(bInputs ? { inputs: bInputs } : {}) }],
        },
      },
    },
    "root/work": leaf("work", { n: str }, { why: { ...str, optional: true } }),
    "root/a": leaf("sayA", { go: str }),
    "root/b": leaf("sayB", { note: str }),
  });

  it("two rules to ONE target each hand over their own reason", async () => {
    const seen: unknown[] = [];
    let round = 0;
    const { registry } = harness({
      work: (inputs) => {
        seen.push((inputs as { why?: unknown }).why);
        return { n: "worked" };
      },
      // `a` fires on the second round only, so both rules get a turn at the same child.
      sayA: () => ({ go: ++round >= 2 ? "yes" : "no" }),
      sayB: () => ({ note: "b" }),
    });
    const { outcome } = await run(
      twoWays({ why: { text: "a sent you" } }, { why: { text: "b sent you" } }),
      registry,
    );
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual(["mount", "b sent you", "a sent you"]);
  });

  it("PARKS when the rule's own value is not ready, without spending the pass", async () => {
    // The rule is about a value an async sibling has not produced. Taking the branch anyway would
    // enter the target with the input missing, which is the silence this whole design is against.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["slow", "work", "b"],
        children: {
          slow: { state: "root/slow", async: true },
          work: { state: "root/work", inputs: { why: { text: "mount" } } },
          b: {
            state: "root/b",
            transitions: [
              {
                to: "work",
                when: ".run.iteration < .limits.max_iterations",
                inputs: { why: { expr: ".children.slow.output.n" } },
              },
            ],
          },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/work": leaf("work", { n: str }, { why: { ...str, optional: true } }),
      "root/b": leaf("sayB", { note: str }),
    };
    const seen: unknown[] = [];
    const { registry } = harness({
      slow: async () => {
        await new Promise((r) => setTimeout(r, 20));
        return { n: "slow finished" };
      },
      work: (inputs) => {
        seen.push((inputs as { why?: unknown }).why);
        return { n: "worked" };
      },
      sayB: () => ({ note: "b" }),
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // It waited, then handed over the settled value — rather than entering with nothing.
    expect(seen).toEqual(["mount", "slow finished"]);
  });

  it("overrides ONLY the names it gives; the rest still come from the mount", async () => {
    const seen: Array<Record<string, unknown>> = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        children: {
          work: { state: "root/work", inputs: { why: { text: "mount why" }, who: { text: "mount who" } } },
          b: {
            state: "root/b",
            transitions: [
              { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { text: "rule why" } } },
            ],
          },
        },
      },
      "root/work": leaf("work", { n: str }, { why: { ...str, optional: true }, who: { ...str, optional: true } }),
      "root/b": leaf("sayB", { note: str }),
    };
    const { registry } = harness({
      work: (inputs) => {
        seen.push({ why: (inputs as { why?: unknown }).why, who: (inputs as { who?: unknown }).who });
        return { n: "worked" };
      },
      sayB: () => ({ note: "b" }),
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual([
      { why: "mount why", who: "mount who" },
      { why: "rule why", who: "mount who" },
    ]);
  });

  it("works on a FORWARD jump too — a rule is a rule, not a loop feature", async () => {
    const seen: unknown[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        children: {
          a: { state: "root/a", transitions: [{ to: "c", inputs: { why: { text: "skipped ahead" } } }] },
          b: { state: "root/b" },
          c: { state: "root/c", inputs: { why: { text: "mount" } } },
        },
      },
      "root/a": leaf("sayA", { go: str }),
      "root/b": leaf("sayB", { note: str }),
      "root/c": leaf("work", { n: str }, { why: { ...str, optional: true } }),
    };
    const { registry, log } = harness({
      sayA: () => ({ go: "yes" }),
      sayB: (_i, l) => {
        l.push("b ran");
        return { note: "b" };
      },
      work: (inputs) => {
        seen.push((inputs as { why?: unknown }).why);
        return { n: "worked" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(log).toEqual([]); // `b` was skipped
    expect(seen).toEqual(["skipped ahead"]);
  });

  it("a rule on the STATE's own list wires the same way a child mount's does", async () => {
    const seen: unknown[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        outputs: { pick: { ...str, binding: ".operation.output.pick" } },
        operation: { kind: "function", function: "choose" },
        children: { work: { state: "root/work", inputs: { why: { text: "mount" } } } },
        transitions: [
          {
            to: "work",
            when: ".run.iteration < .limits.max_iterations",
            // NOT `.outputs.pick`: a bound output resolves when the state finishes, so mid-run it
            // reads as absent. `.operation.output` is what this call actually returned.
            inputs: { why: { expr: ".operation.output.pick" } },
          },
        ],
      },
      "root/work": leaf("work", { n: str }, { why: { ...str, optional: true } }),
    };
    const { registry } = harness({
      choose: () => ({ pick: "state rule" }),
      work: (inputs) => {
        seen.push((inputs as { why?: unknown }).why);
        return { n: "worked" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(seen).toContain("state rule");
  });
});

describe("the lint over history and rule wiring", () => {
  const errorsFor = (files: Record<string, StateDef>): string[] =>
    validateBundle(loadBundle(files, "root")).errors.map((e) => e.message);

  const base = (): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 1 },
      children: {
        work: { state: "root/work" },
        b: { state: "root/b", transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations" }] },
      },
    },
    "root/work": leaf("work", { n: str }, { why: { ...str, optional: true } }),
    "root/b": leaf("sayB", { note: str }),
  });

  it("accepts an indexed read of a child's history", () => {
    const files = base();
    files["root"]!.children!["work"]!.inputs = { why: { expr: ".children.b[-2].output.note" } };
    expect(errorsFor(files)).toEqual([]);
  });

  it("type-checks an indexed read against the slot", () => {
    const files = base();
    // `note` is a string; the slot wants a string. Ask for the whole PASS instead and it is not.
    files["root"]!.children!["work"]!.inputs = { why: { expr: ".children.b[-2].output" } };
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });

  it("refuses a rule that names an output the target does not publish", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { expr: ".children.b.output.nope" } } },
    ];
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });
});
