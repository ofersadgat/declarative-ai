/**
 * What goes wrong, and how loudly — bad wiring, bad values, and calls that throw.
 *
 * The rule this file is written against: a fault must be REPORTED, at the earliest point that can
 * name it. The failure this whole line of work started from was the opposite — a wire that read as
 * correct, resolved to nothing, and let a state run with an input missing — so the cases below are
 * mostly about making sure the quiet paths are the ones that are supposed to be quiet.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
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
      hostFunction(
        async (inputs: Record<string, unknown>) => ok((await fn(inputs)) as never) as ExecResult<ResolvedValue, WorkflowMetrics>,
        HOST,
      ),
    );
  }
  return registry;
}

async function run(files: Record<string, StateDef>, registry: ReturnType<typeof harness>) {
  const events: EngineEvent[] = [];
  const engine = new WorkflowEngine({ bundle: loadBundle(files, "root"), registry, onEvent: (e) => void events.push(e) });
  const outcome = await engine.run({ inputs: {} });
  return { outcome, events };
}

const leaf = (fn: string, outputs: Record<string, unknown>, inputs?: Record<string, unknown>): StateDef =>
  ({ label: fn, ...(inputs ? { inputs } : {}), outputs, operation: { kind: "function", function: fn } }) as StateDef;

const errorsFor = (files: Record<string, StateDef>): string[] =>
  validateBundle(loadBundle(files, "root")).errors.map((e) => e.message);

// ---------------------------------------------------------------------------
// wiring that cannot be right
// ---------------------------------------------------------------------------

describe("inputs the loader and the lint refuse", () => {
  const base = (): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 1 },
      children: {
        work: { state: "root/work", inputs: { why: { text: "mount" } } },
        b: { state: "root/b", transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations" }] },
      },
    },
    "root/work": leaf("work", { n: str }, { why: str }),
    "root/b": leaf("b", { n: str }),
  });

  it("a REQUIRED child input nobody wires is named at load", () => {
    const files = base();
    delete (files["root"]!.children!["work"] as { inputs?: unknown }).inputs;
    expect(errorsFor(files).some((m) => /required child input 'why' is not wired/.test(m))).toBe(true);
  });

  it("a mount naming an input the child does not declare is named", () => {
    const files = base();
    files["root"]!.children!["work"]!.inputs = { why: { text: "x" }, nope: { text: "y" } };
    expect(errorsFor(files).some((m) => /declares no input 'nope'/.test(m))).toBe(true);
  });

  it("a RULE naming an input the child does not declare is named the same way", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { nope: { text: "y" } } },
    ];
    expect(errorsFor(files).some((m) => /'work' declares no input 'nope'/.test(m))).toBe(true);
  });

  it("a rule handing a NUMBER to a string slot is a type error, not a runtime surprise", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { json: 42 } } },
    ];
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });

  it("a rule reading an output the source does not publish is named", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { expr: ".children.b.output.nope" } } },
    ];
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });

  it("a rule reading a child that does not exist is named", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { expr: ".children.ghost.output.n" } } },
    ];
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });

  it("a guard that does not infer to BOOLEAN is refused — no truthiness", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [{ to: "work", when: ".children.b.output.n" }];
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });

  it("a rule naming a target that is neither a child nor a terminate.* is named", () => {
    const files = base();
    files["root"]!.children!["b"]!.transitions = [{ to: "nowhere" }];
    expect(errorsFor(files).some((m) => /neither a declared child nor a terminate/.test(m))).toBe(true);
  });

  it("an expression that does not parse fails at LOAD, naming the binding", () => {
    const files = base();
    files["root"]!.children!["work"]!.inputs = { why: { expr: ".children.b.output.n ===" } };
    expect(() => loadBundle(files, "root")).toThrow(/does not parse/);
  });

  it("a MOUNT reading a later sibling is refused unless the slot opts out", () => {
    // §7.2 reachability, asked AT THE MOUNT: `work` runs before `b`, so a wire on `work` naming
    // `b` is not proven on every path that reaches it — indexed or not. This is why every loop's
    // feedback input is declared OPTIONAL, which IS the opt-out, and it is the strongest argument
    // for putting the wiring on the rule instead: a rule is evaluated at a point where its source
    // demonstrably ran.
    const required = base();
    required["root"]!.children!["work"]!.inputs = { why: { expr: ".children.b[-1].output.n" } };
    expect(errorsFor(required).some((m) => /not proven to have run/.test(m))).toBe(true);

    const optional = base();
    optional["root"]!.children!["work"]!.inputs = { why: { expr: ".children.b[-1].output.n" } };
    optional["root/work"] = leaf("work", { n: str }, { why: { ...str, optional: true } });
    expect(errorsFor(optional)).toEqual([]);
  });

  it("a RULE reading the same sibling needs no opt-out at all", () => {
    // The difference the feature buys: by the time the rule fires, `b` has run — so the value is
    // proven, and the slot does not have to be widened to accept its absence.
    const files = base();
    files["root"]!.children!["b"]!.transitions = [
      { to: "work", when: ".run.iteration < .limits.max_iterations", inputs: { why: { expr: ".children.b.output.n" } } },
    ];
    expect(errorsFor(files)).toEqual([]);
  });
});

describe("outputs the loader refuses", () => {
  it("a KIND on a multi-entry operation output says where it belongs", () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { a: str, b: str },
        operation: {
          kind: "function",
          function: "work",
          output: { a: { kind: "blob", schema: { type: "string" } }, b: str },
        },
      } as StateDef,
    };
    const bundle = loadBundle(files, "root");
    expect(bundle.states["root"]?.operationError ?? "").toMatch(/declares kind 'blob'/);
  });

  it("a state output bound to a name the call never returns is named", () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { n: { ...str, binding: ".operation.output.nope" } },
        operation: { kind: "function", function: "work", output: { n: str } },
      } as StateDef,
    };
    expect(errorsFor(files).length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// values that are wrong at RUN time
// ---------------------------------------------------------------------------

describe("a call that answers with the wrong thing", () => {
  const one = (outputs: Record<string, unknown>): Record<string, StateDef> => ({
    root: { label: "Root", children: { work: { state: "root/work" } } },
    "root/work": leaf("work", outputs),
  });

  it("fails the state when a REQUIRED output is missing, naming it", async () => {
    const { outcome } = await run(one({ n: str }), harness({ work: () => ({ other: "x" }) }));
    expect(outcome.outcome).toBe("error");
    expect(JSON.stringify(outcome)).toMatch(/did not produce required output 'n'/);
  });

  it("fails the state when an output does not match its schema", async () => {
    const { outcome } = await run(one({ n: num }), harness({ work: () => ({ n: "not a number" }) }));
    expect(outcome.outcome).toBe("error");
  });

  it("accepts an OPTIONAL output the call omits", async () => {
    const files = one({ n: str, extra: { ...str, optional: true } });
    const { outcome } = await run(files, harness({ work: () => ({ n: "fine" }) }));
    expect(outcome.outcome).toBe("success");
  });
});

describe("a call that throws", () => {
  const files = (handled: boolean): Record<string, StateDef> => ({
    root: {
      label: "Root",
      children: {
        work: { state: "root/work", ...(handled ? { transitions: [{ to: "after" }] } : {}) },
        after: { state: "root/after" },
      },
    },
    "root/work": leaf("boom", { n: str }),
    "root/after": leaf("after", { n: str }),
  });

  it("takes the whole state down when NOTHING handles it", async () => {
    const { outcome } = await run(
      files(false),
      harness({
        boom: () => {
          throw new Error("exploded");
        },
        after: () => ({ n: "after" }),
      }),
    );
    expect(outcome.outcome).toBe("error");
  });

  it("is handled by a rule on its mount, and the state carries on", async () => {
    const order: string[] = [];
    const { outcome } = await run(
      files(true),
      harness({
        boom: () => {
          order.push("boom");
          throw new Error("exploded");
        },
        after: () => {
          order.push("after");
          return { n: "after" };
        },
      }),
    );
    expect(outcome.outcome).toBe("success");
    expect(order).toEqual(["boom", "after"]);
  });

  it("leaves the failed child's OUTCOME readable, which is what a handler is for", async () => {
    const seen: unknown[] = [];
    const f = files(true);
    (f["root"]!.children as Record<string, { inputs?: unknown }>)["after"]!.inputs = {
      why: { expr: ".children.work.outcome" },
    };
    f["root/after"] = leaf("after", { n: str }, { why: { ...str, optional: true } });
    const { outcome } = await run(
      f,
      harness({
        boom: () => {
          throw new Error("exploded");
        },
        after: (i) => {
          seen.push((i as { why?: unknown }).why);
          return { n: "after" };
        },
      }),
    );
    expect(outcome.outcome).toBe("success");
    expect(seen).toEqual(["error"]);
  });
});

// ---------------------------------------------------------------------------
// failures inside a loop
// ---------------------------------------------------------------------------

describe("a failure inside a loop", () => {
  /** `work` fails on the first pass; the mount's rule handles it and `judge` sends it back. */
  const files = (): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 1 },
      children: {
        work: { state: "root/work", transitions: [{ to: "judge" }] },
        judge: {
          state: "root/judge",
          inputs: { last: { expr: ".children.work[-1].outcome" }, count: { expr: ".children.work.length" } },
          transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations" }],
        },
      },
    },
    "root/work": leaf("work", { n: str }),
    "root/judge": leaf("judge", { n: str }, { last: { ...str, optional: true }, count: { ...num, optional: true } }),
  });

  it("does not stop the loop, and the next pass can read that it failed", async () => {
    const seen: Array<[unknown, unknown]> = [];
    let attempt = 0;
    const { outcome } = await run(
      files(),
      harness({
        work: () => {
          attempt += 1;
          if (attempt === 1) throw new Error("first pass fails");
          return { n: "recovered" };
        },
        judge: (i) => {
          seen.push([(i as { last?: unknown }).last, (i as { count?: unknown }).count]);
          return { n: "judged" };
        },
      }),
    );
    expect(outcome.outcome).toBe("success");
    // Pass 0 errored and is still a row; pass 1 succeeded. History records what happened, not only
    // what worked.
    expect(seen).toEqual([
      ["error", 1],
      ["success", 2],
    ]);
  });

  it("a rule reading the FAILED pass's output hands over nothing, and an optional slot stays empty", async () => {
    const seen: unknown[] = [];
    const f = files();
    (f["root"]!.children as Record<string, { transitions?: unknown }>)["judge"]!.transitions = [
      {
        to: "work",
        when: ".run.iteration < .limits.max_iterations",
        inputs: { prior: { expr: ".children.work.output.n" } },
      },
    ] as never;
    f["root/work"] = leaf("work", { n: str }, { prior: { ...str, optional: true } });
    let attempt = 0;
    const { outcome } = await run(
      f,
      harness({
        work: (i) => {
          seen.push((i as { prior?: unknown }).prior);
          attempt += 1;
          if (attempt === 1) throw new Error("first pass fails");
          return { n: "recovered" };
        },
        judge: () => ({ n: "judged" }),
      }),
    );
    expect(outcome.outcome).toBe("success");
    // A failed pass produced no output, so the rule had nothing to hand over — and the OPTIONAL
    // slot is the author saying that is acceptable. On a required slot the entry would refuse.
    expect(seen).toEqual([undefined, undefined]);
  });
});

describe("a required input a rule cannot fill", () => {
  /**
   * The whole point of the exercise: a missing value must never be silence. `prior` is REQUIRED and
   * only the rule can fill it, so when the rule has nothing the entry is REFUSED — the child does
   * not run with the input absent.
   *
   * Whether the STATE then fails is a separate question, and the answer is the ordinary one: a
   * refused entry is a child failure, and a child failure ends the state unless a rule handles it.
   */
  const files = (handled: boolean): Record<string, StateDef> => ({
    root: {
      label: "Root",
      limits: { max_iterations: 1 },
      children: {
        // No mount wiring for `prior`: the rule is the only thing that can fill it. (With a mount
        // value it would fall back, which is the override rule working and not what this is about.)
        work: { state: "root/work", ...(handled ? { transitions: [{ to: "judge" }] } : {}) },
        judge: {
          state: "root/judge",
          transitions: [
            {
              to: "work",
              when: ".run.iteration < .limits.max_iterations",
              inputs: { prior: { expr: ".children.work.output.n" } },
            },
          ],
        },
      },
    },
    // REQUIRED, and with a default it would be filled rather than refused.
    "root/work": leaf("work", { n: str }, { prior: str }),
    "root/judge": leaf("judge", { n: str }),
  });

  it("refuses the entry and says which input was missing", async () => {
    let ran = 0;
    const { outcome } = await run(
      files(false),
      harness({
        work: () => {
          ran += 1;
          return { n: "ok" };
        },
        judge: () => ({ n: "judged" }),
      }),
    );
    // It never ran: an unfillable required input is not something a state proceeds past.
    expect(ran).toBe(0);
    expect(outcome.outcome).toBe("error");
    expect(JSON.stringify(outcome)).toMatch(/required input 'prior' missing/);
  });

  it("is a refusal, not a termination — the child never became an instance", async () => {
    let ran = 0;
    const { outcome, events } = await run(
      files(true),
      harness({
        work: () => {
          ran += 1;
          return { n: "ok" };
        },
        judge: () => ({ n: "judged" }),
      }),
    );
    // A rule on its mount takes responsibility, so the state carries on. What must not happen —
    // and does not — is `work` running with `prior` unset.
    expect(outcome.outcome).toBe("success");
    expect(ran).toBe(0);
    // And there is no terminated event for it: the entry was refused BEFORE an instance existed,
    // so it did not fail so much as never start. Reading the journal for a failure here would find
    // nothing, which is worth knowing.
    expect(events.some((e) => e.type === "instance.entered" && e.stateId === "root/work")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// limits
// ---------------------------------------------------------------------------

describe("limits are still limits", () => {
  it("a timeout terminates the state, whatever the loop was doing", async () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        limits: { timeout: 0.05, max_iterations: 100 },
        children: {
          slow: { state: "root/slow" },
          judge: { state: "root/judge", transitions: [{ to: "slow", when: ".run.iteration < .limits.max_iterations" }] },
        },
      },
      "root/slow": leaf("slow", { n: str }),
      "root/judge": leaf("judge", { n: str }),
    };
    const { outcome } = await run(
      files,
      harness({
        slow: async () => {
          await new Promise((r) => setTimeout(r, 20));
          return { n: "slow" };
        },
        judge: () => ({ n: "judge" }),
      }),
    );
    expect(outcome.outcome).toBe("timeout");
  });

  it("warns about a cycle with no iteration guard, where the SEQUENCE was authored", () => {
    // Gated on an authored `sequence`: a derived one is the declaration order, and warning about
    // every unguarded rule in a state nobody sequenced would be noise on the common case.
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        sequence: ["a", "b"],
        children: {
          a: { state: "root/a" },
          b: { state: "root/b", transitions: [{ to: "a" }] },
        },
      },
      "root/a": leaf("a", { n: str }),
      "root/b": leaf("b", { n: str }),
    };
    const warnings = validateBundle(loadBundle(files, "root")).warnings.map((w) => w.message);
    expect(warnings.some((m) => /can cycle; add limits.max_iterations or a run.iteration guard/.test(m))).toBe(true);
  });
});
