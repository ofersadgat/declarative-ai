/**
 * A transition may wire the child it enters — `TransitionDecl.inputs`.
 *
 * The mount says what a child ALWAYS takes; a transition says what it takes WHEN IT ARRIVES THIS
 * WAY. Those separate the moment a child can be re-entered from more than one place, which is the
 * ordinary shape of a workflow that sends work back: several later states can each return to one
 * earlier one, each with its own evidence for doing so.
 *
 * A mount cannot answer that. It can only name a value true of every arrival, and coalescing
 * candidates there does not work either — after a loop, EVERY earlier sibling has a value from the
 * previous pass, so "the finding that sent me here" is unanswerable at the mount. The rule that
 * fired is the only thing that knows.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { BindingDecl, StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** `work` records what it was handed; `a` and `b` each send control back with their own evidence. */
function bundleOf(files: Record<string, StateDef>) {
  const seen: Array<unknown> = [];
  const registry = newRegistry();
  registry.functions.set(
    "work",
    hostFunction(async (inputs: Record<string, unknown>) => {
      seen.push((inputs as { why?: unknown }).why);
      return ok({ n: String(seen.length) }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  for (const name of ["sayA", "sayB"]) {
    registry.functions.set(
      name,
      hostFunction(async () => ok({ note: `${name} spoke` }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST),
    );
  }
  return { engine: new WorkflowEngine({ bundle: loadBundle(files, "root"), registry }), seen };
}

const leaf = (fn: string, out: string): StateDef => ({
  label: fn,
  outputs: { [out]: { schema: { type: "string" } } },
  operation: { kind: "function", function: fn },
});

/** `work` → `a` → `b`, with `b` returning to `work` once. Each returner names its own reason. */
const files = (aInputs?: Record<string, BindingDecl>, bInputs?: Record<string, BindingDecl>): Record<string, StateDef> => ({
  root: {
    label: "Root",
    limits: { max_iterations: 1 },
    children: {
      work: { state: "root/work", inputs: { why: { text: "first time" } } },
      a: {
        state: "root/a",
        transitions: [{ to: "work", when: ".run.iteration < 0", ...(aInputs ? { inputs: aInputs } : {}) }],
      },
      b: {
        state: "root/b",
        transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations", ...(bInputs ? { inputs: bInputs } : {}) }],
      },
    },
  },
  "root/work": {
    label: "work",
    inputs: { why: { schema: { type: "string" }, optional: true } },
    outputs: { n: { schema: { type: "string" } } },
    operation: { kind: "function", function: "work" },
  },
  "root/a": leaf("sayA", "note"),
  "root/b": leaf("sayB", "note"),
});

describe("a transition wires the child it enters", () => {
  it("hands the target the value the RULE THAT FIRED names", async () => {
    const { engine, seen } = bundleOf(
      files({ why: { text: "a sent you back" } }, { why: { text: "b sent you back" } }),
    );
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    // `b` fired, so `b`'s reason arrived — not `a`'s, and not the mount's. A mount could not have
    // said this: both returners are equally true of the child.
    expect(seen).toEqual(["first time", "b sent you back"]);
  });

  it("reads the CURRENT pass — the same one its guard read", async () => {
    // The rule's two halves are one statement. `when` asked about `b`'s outcome and `inputs` hands
    // over `b`'s note, and both mean the `b` that just ran: the wiring resolves before the pass
    // opens and before the reset, so the plain spelling is the right one and `[-2]` would reach
    // past the thing the rule is about.
    const { engine, seen } = bundleOf(files(undefined, { why: { expr: ".children.b.output.note" } }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(seen).toEqual(["first time", "sayB spoke"]);
  });

  it("still sees the TARGET's previous record, because the reset has not run yet", async () => {
    // What this buys beyond tidiness: at the moment a rule is evaluated the child it is about to
    // re-enter has not been cleared, so its last output is readable — and accumulating across
    // passes needs no history indexing at all.
    const { engine, seen } = bundleOf(files(undefined, { why: { expr: ".children.work.output.n" } }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(seen).toEqual(["first time", "1"]);
  });

  it("falls back to the MOUNT for every name it does not restate", async () => {
    // A transition is an override, not a replacement: the child's other inputs stay the mount's
    // problem, so a rule that has one thing to say says one thing.
    const { engine, seen } = bundleOf(files(undefined, undefined));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(seen).toEqual(["first time", "first time"]);
  });

  it("says nothing on the SEQUENCE path — walking to a child is not a reason", async () => {
    // The cursor reaches a child by running out of work, which carries no evidence at all. Only a
    // taken transition supplies overrides.
    const { engine, seen } = bundleOf(files({ why: { text: "never fires" } }, { why: { text: "b sent you back" } }));
    await engine.run({ inputs: {} });
    expect(seen[0]).toBe("first time");
  });
});

describe("the lint over a transition's wiring", () => {
  const errorsFor = (files: Record<string, StateDef>): string[] =>
    validateBundle(loadBundle(files, "root")).errors.map((e) => e.message);

  it("refuses a name the target does not declare", () => {
    const messages = errorsFor(files(undefined, { nope: { text: "x" } }));
    expect(messages.some((m) => /'work' declares no input 'nope'/.test(m))).toBe(true);
  });

  it("refuses inputs on a rule that TERMINATES — there is no child to pass them to", () => {
    const bundle = files();
    bundle["root"]!.children!["b"]!.transitions = [{ to: "terminate.success", inputs: { why: { text: "x" } } }];
    expect(errorsFor(bundle).some((m) => /terminates the state; there is no child/.test(m))).toBe(true);
  });

  it("type-checks the value against the slot, exactly as a mount's wiring is checked", () => {
    const messages = errorsFor(files(undefined, { why: { json: 42 } }));
    expect(messages.length).toBeGreaterThan(0);
  });

  it("accepts a well-formed one", () => {
    expect(errorsFor(files(undefined, { why: { text: "b sent you back" } }))).toEqual([]);
  });
});

describe("accumulating across passes needs no history at all", () => {
  /**
   * The rule fires BEFORE the reset, so the child it is about to re-enter still holds its last
   * record. A rule can therefore hand over "what I just found, on top of what you already had" as
   * one expression — no indexing, no walking the passes, and the growing list lives where every
   * other value lives.
   */
  it("hands the target its own previous list with the new findings appended", async () => {
    const seen: string[][] = [];
    const registry = newRegistry();
    registry.functions.set(
      "collect",
      hostFunction(async (inputs: Record<string, unknown>) => {
        const all = ((inputs as { all?: string[] }).all ?? []) as string[];
        seen.push([...all]);
        // Republished, which is what makes it readable next time round.
        return ok({ all }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    let round = 0;
    registry.functions.set(
      "judge",
      hostFunction(async () => {
        round += 1;
        return ok({ found: [`finding ${round}`] }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    const engine = new WorkflowEngine({
      bundle: loadBundle(
        {
          root: {
            label: "Root",
            limits: { max_iterations: 2 },
            children: {
              collect: { state: "root/collect" },
              judge: {
                state: "root/judge",
                transitions: [
                  {
                    to: "collect",
                    when: ".run.iteration < .limits.max_iterations",
                    // BOTH sides are the current pass: `judge` just ran, and `collect` has not been
                    // cleared yet. One expression, no history.
                    inputs: { all: { expr: "concat(.children.collect.output.all, .children.judge.output.found)" } },
                  },
                ],
              },
            },
          },
          "root/collect": {
            label: "collect",
            inputs: { all: { schema: { type: "array", items: { type: "string" } }, optional: true } },
            outputs: { all: { schema: { type: "array", items: { type: "string" } } } },
            operation: { kind: "function", function: "collect" },
          },
          "root/judge": {
            label: "judge",
            outputs: { found: { schema: { type: "array", items: { type: "string" } } } },
            operation: { kind: "function", function: "judge" },
          },
        },
        "root",
      ),
      registry,
    });
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(seen).toEqual([[], ["finding 1"], ["finding 1", "finding 2"]]);
  });
});
