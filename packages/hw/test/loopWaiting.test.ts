/**
 * Loops where something WAITS — an async child mid-flight, a guard that defers, a park that must
 * not spend a pass — and the counters that have to survive all three.
 *
 * The counters are the point. `index` and `iteration` are decremented again when an entry parks,
 * and a pass that opened is popped, so a rule that could not be taken leaves no trace of having
 * tried. Getting that wrong is invisible in the happy path and shortens every loop budget under
 * load, which is the worst combination available.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { EngineEvent, WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };
const str = { schema: { type: "string" } } as const;
const num = { schema: { type: "number" } } as const;

function harness(fns: Record<string, (inputs: Record<string, unknown>) => unknown | Promise<unknown>>) {
  const registry = newRegistry();
  for (const [name, fn] of Object.entries(fns)) {
    registry.functions.set(
      name,
      hostFunction(async (inputs: Record<string, unknown>) => ok((await fn(inputs)) as never) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST),
    );
  }
  return registry;
}

async function run(files: Record<string, StateDef>, registry: ReturnType<typeof harness>) {
  const events: EngineEvent[] = [];
  const engine = new WorkflowEngine({ bundle: loadBundle(files, "root"), registry, onEvent: (e) => void events.push(e) });
  const outcome = await engine.run({ inputs: {} });
  const taken = events.filter((e): e is Extract<EngineEvent, { type: "transition.taken" }> => e.type === "transition.taken");
  return { outcome, events, taken };
}

const leaf = (fn: string, outputs: Record<string, unknown>, inputs?: Record<string, unknown>): StateDef =>
  ({ label: fn, ...(inputs ? { inputs } : {}), outputs, operation: { kind: "function", function: fn } }) as StateDef;

describe("counters under a park", () => {
  /**
   * `b` returns to `a` while an ASYNC sibling is still running, and the rule's wiring reads that
   * sibling. The first attempt cannot be answered, so it parks — and must leave the budget alone.
   */
  it("a parked rule spends NO pass and no step", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["slow", "a", "b"],
        children: {
          slow: { state: "root/slow", async: true },
          a: { state: "root/a", inputs: { why: { text: "mount" } } },
          b: {
            state: "root/b",
            transitions: [
              {
                to: "a",
                when: ".run.iteration < .limits.max_iterations",
                inputs: { why: { expr: ".children.slow.output.n" } },
              },
            ],
          },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/a": leaf("a", { n: str }, { why: { ...str, optional: true } }),
      "root/b": leaf("b", { n: str }),
    };
    const seen: unknown[] = [];
    const registry = harness({
      slow: async () => {
        await new Promise((r) => setTimeout(r, 25));
        return { n: "settled" };
      },
      a: (i) => {
        seen.push((i as { why?: unknown }).why);
        return { n: "a" };
      },
      b: () => ({ n: "b" }),
    });
    const { outcome, taken } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // ONE transition, at iteration 1 — not two, and not iteration 2. The park left no mark.
    expect(taken.map((t) => ({ to: t.to, index: t.index, iteration: t.iteration }))).toEqual([
      { to: "a", index: 1, iteration: 1 },
    ]);
    expect(seen).toEqual(["mount", "settled"]);
  });

  it("does not open a pass it then abandons — history counts entries, not attempts", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["slow", "a", "b"],
        children: {
          slow: { state: "root/slow", async: true },
          a: { state: "root/a" },
          b: {
            state: "root/b",
            inputs: { count: { expr: ".children.a.length" } },
            transitions: [
              { to: "a", when: ".run.iteration < .limits.max_iterations", inputs: { why: { expr: ".children.slow.output.n" } } },
            ],
          },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/a": leaf("a", { n: str }, { why: { ...str, optional: true } }),
      "root/b": leaf("b", { n: str }, { count: { ...num, optional: true } }),
    };
    const counts: unknown[] = [];
    const registry = harness({
      slow: async () => {
        await new Promise((r) => setTimeout(r, 25));
        return { n: "settled" };
      },
      a: () => ({ n: "a" }),
      b: (i) => {
        counts.push((i as { count?: number }).count);
        return { n: "b" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // Two passes of `a`, because one rule was taken — the parked attempt added no row.
    expect(counts).toEqual([1, 2]);
  });
});

describe("an async child across a pass boundary", () => {
  it("the pass that turns over supersedes it, and the NEW pass runs it again", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["slow", "gate"],
        children: {
          slow: { state: "root/slow", async: true },
          gate: { state: "root/gate", transitions: [{ to: "slow", when: ".run.iteration < .limits.max_iterations" }] },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/gate": leaf("gate", { n: str }),
    };
    let runs = 0;
    const registry = harness({
      slow: async () => {
        runs += 1;
        await new Promise((r) => setTimeout(r, 10));
        return { n: `slow${runs}` };
      },
      gate: () => ({ n: "gate" }),
    });
    const { outcome, events } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(runs).toBe(2);
    expect(events.filter((e) => e.type === "child.superseded" && e.childKey === "slow")).toHaveLength(1);
  });

  it("a child outside the sequence that nothing TRANSITIONS to has rows and no records", async () => {
    // Sequence membership decides what the CURSOR walks — not what can run. Nothing names
    // `watcher`, in the sequence or in any rule, so it never runs; it still occupies an index in
    // every pass, which is what `.length` counting PASSES rather than runs means. A child a rule
    // DOES name runs perfectly well outside the sequence — see the test below.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        sequence: ["a", "gate"],
        children: {
          watcher: { state: "root/watcher", async: true },
          a: { state: "root/a" },
          gate: { state: "root/gate", transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }] },
        },
      },
      "root/watcher": leaf("watch", { n: str }),
      "root/a": leaf("a", { n: str }),
      "root/gate": leaf("gate", { n: str }),
    };
    let watched = 0;
    const registry = harness({
      watch: async () => {
        watched += 1;
        await new Promise((r) => setTimeout(r, 5));
        return { n: "watched" };
      },
      a: () => ({ n: "a" }),
      gate: () => ({ n: "gate" }),
    });
    const { outcome, events } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    expect(watched).toBe(0);
    // Never entered, so never superseded either: the reset only names what it can find.
    expect(events.some((e) => e.type === "child.superseded" && e.childKey === "watcher")).toBe(false);
  });
});

describe("what a pass boundary does to the cursor", () => {
  it("a backward jump re-runs the tail, and a forward one leaves the head skipped", async () => {
    const order: string[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 1 },
        children: {
          a: { state: "root/a" },
          b: { state: "root/b", transitions: [{ to: "d", when: ".run.iteration < 1" }] },
          c: { state: "root/c" },
          d: { state: "root/d", transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }] },
        },
      },
      "root/a": leaf("mark", { n: str }, { name: str }),
      "root/b": leaf("mark", { n: str }, { name: str }),
      "root/c": leaf("mark", { n: str }, { name: str }),
      "root/d": leaf("mark", { n: str }, { name: str }),
    };
    for (const k of ["a", "b", "c", "d"]) {
      (files["root"]!.children as Record<string, { inputs?: unknown }>)[k]!.inputs = { name: { text: k } };
    }
    const registry = harness({
      mark: (i) => {
        order.push((i as { name: string }).name);
        return { n: "ok" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // a, b, (skip c) d — then back to a, and this time `b`'s forward rule is spent so `c` runs.
    expect(order).toEqual(["a", "b", "d", "a", "b", "c", "d"]);
  });

  it("a self-loop on the ONLY child still accumulates history", async () => {
    const counts: unknown[] = [];
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 2 },
        children: {
          only: {
            state: "root/only",
            inputs: { count: { expr: ".children.only.length" } },
            transitions: [{ to: "only", when: ".run.iteration < .limits.max_iterations" }],
          },
        },
      },
      "root/only": leaf("only", { n: str }, { count: { ...num, optional: true } }),
    };
    const registry = harness({
      only: (i) => {
        counts.push((i as { count?: number }).count);
        return { n: "only" };
      },
    });
    const { outcome } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // `.length` is the number of PASSES, including the one now opening — not the number of times
    // the child ran. Reading its own history, a self-loop sees a row for the pass it is being
    // entered in, whose record does not exist yet.
    expect(counts).toEqual([1, 2, 3]);
  });
});

describe("limits still bound a loop that carries values", () => {
  it("stops at max_iterations however much the rule hands over", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { max_iterations: 3 },
        children: {
          work: { state: "root/work", inputs: { all: { json: [] } } },
          judge: {
            state: "root/judge",
            transitions: [
              {
                to: "work",
                when: ".run.iteration < .limits.max_iterations",
                inputs: { all: { expr: "concat(.children.work.output.all, .children.judge.output.found)" } },
              },
            ],
          },
        },
      },
      "root/work": {
        label: "work",
        inputs: { all: { schema: { type: "array", items: { type: "string" } }, optional: true } },
        outputs: { all: { schema: { type: "array", items: { type: "string" } } } },
        operation: { kind: "function", function: "work" },
      } as StateDef,
      "root/judge": leaf("judge", { found: { schema: { type: "array", items: { type: "string" } } } }),
    };
    const sizes: number[] = [];
    let round = 0;
    const registry = harness({
      work: (i) => {
        const all = ((i as { all?: string[] }).all ?? []) as string[];
        sizes.push(all.length);
        return { all };
      },
      judge: () => ({ found: [`f${++round}`] }),
    });
    const { outcome, taken } = await run(files, registry);
    expect(outcome.outcome).toBe("success");
    // Four runs of `work`: the first, then one per pass, and the budget stops it at three passes.
    expect(sizes).toEqual([0, 1, 2, 3]);
    expect(taken.map((t) => t.iteration)).toEqual([1, 2, 3]);
  });
});

describe("a child outside the sequence that a RULE names", () => {
  /**
   * SPEC §3.3's `repair` shape. Sequence membership decides what the cursor walks; a transition may
   * name any declared child, and entering one that is not a member deliberately does NOT move the
   * cursor — so when it finishes, the spine resumes exactly where it left off.
   */
  const files = (): Record<string, StateDef> => ({
    root: {
      label: "Root",
      sequence: ["implement", "review"],
      children: {
        implement: {
          state: "root/implement",
          transitions: [{ to: "repair", when: ".children.implement.output.ok === 'no'" }],
        },
        // Not a sequence member. Reachable only by the rule above.
        repair: { state: "root/repair" },
        review: { state: "root/review" },
      },
    },
    "root/implement": leaf("implement", { ok: str }),
    "root/repair": leaf("repair", { n: str }),
    "root/review": leaf("review", { n: str }),
  });

  it("RUNS it, and the spine resumes at the member the cursor was heading for", async () => {
    const order: string[] = [];
    const registry = harness({
      implement: () => {
        order.push("implement");
        return { ok: "no" };
      },
      repair: () => {
        order.push("repair");
        return { n: "repaired" };
      },
      review: () => {
        order.push("review");
        return { n: "reviewed" };
      },
    });
    const { outcome } = await run(files(), registry);
    expect(outcome.outcome).toBe("success");
    // `repair` ran, and `review` still followed: entering a non-member left the cursor alone.
    expect(order).toEqual(["implement", "repair", "review"]);
  });

  it("gives it a real record in the pass it ran in, not a hole", async () => {
    const seen: unknown[] = [];
    const f = files();
    (f["root"]!.children as Record<string, { inputs?: unknown }>)["review"]!.inputs = {
      fixed: { expr: ".children.repair.output.n" },
    };
    f["root/review"] = leaf("review", { n: str }, { fixed: { ...str, optional: true } });
    const registry = harness({
      implement: () => ({ ok: "no" }),
      repair: () => ({ n: "repaired" }),
      review: (i) => {
        seen.push((i as { fixed?: unknown }).fixed);
        return { n: "reviewed" };
      },
    });
    const { outcome } = await run(f, registry);
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual(["repaired"]);
  });

  it("is NOT cleared by a reset, because the reset only names sequence members", async () => {
    // A loop turns the spine over; `repair` is not on it, so its record carries into the new pass
    // exactly as any other non-member's does.
    const counts: unknown[] = [];
    const f = files();
    f["root"]!.limits = { max_iterations: 1 };
    (f["root"]!.children as Record<string, { transitions?: unknown; inputs?: unknown }>)["review"] = {
      state: "root/review",
      inputs: { fixed: { expr: ".children.repair.output.n" } },
      transitions: [{ to: "implement", when: ".run.iteration < .limits.max_iterations" }],
    } as never;
    f["root/review"] = leaf("review", { n: str }, { fixed: { ...str, optional: true } });
    let repairs = 0;
    const registry = harness({
      // Only the FIRST pass fails, so the rule to `repair` fires once.
      implement: () => ({ ok: repairs === 0 ? "no" : "yes" }),
      repair: () => {
        repairs += 1;
        return { n: "repaired" };
      },
      review: (i) => {
        counts.push((i as { fixed?: unknown }).fixed);
        return { n: "reviewed" };
      },
    });
    const { outcome } = await run(f, registry);
    expect(outcome.outcome).toBe("success");
    expect(repairs).toBe(1);
    // The second pass still sees the repair: a non-member survives the reset that cleared the spine.
    expect(counts).toEqual(["repaired", "repaired"]);
  });
});
