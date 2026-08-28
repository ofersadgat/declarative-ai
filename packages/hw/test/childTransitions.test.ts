/**
 * Transitions written on a CHILD MOUNT (SPEC §3.3): considered when that child finishes, ahead of the
 * state's own list, and only for the round its completion triggered.
 *
 * What they buy is the thing a state-level list cannot say without repeating itself — "when THIS
 * child ends, go there" — which is why the interesting cases here are all about the narrowing: which
 * round the list is eligible in, which list wins when both match, and what taking one does to a
 * failure the child reported.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/**
 * A callee document for `probe(…)`, so a guard here can embed a CALL.
 *
 * A call lowers at LOAD time against a resolver, which needs the search path and a filesystem — hence
 * the little VFS rather than just a registered function. Counting `probe`'s invocations is how these
 * tests observe that a guard was actually prepared and evaluated, rather than inferring it from which
 * transition happened to win.
 */
const FUNCTIONS = "/p/.jaira/functions";
const CALLEES: Record<string, string> = {
  [`${FUNCTIONS}/probe.json`]: JSON.stringify({
    kind: "function",
    function: "probe",
    input: { text: { kind: "text", index: 0 } },
  }),
  // The same probe, slow — a guard whose call is still in flight while a SIBLING finishes. That is
  // the window where "a child that completes is evaluated" is at risk, and a fast call never opens it.
  [`${FUNCTIONS}/slowProbe.json`]: JSON.stringify({
    kind: "function",
    function: "slowProbe",
    input: { text: { kind: "text", index: 0 } },
  }),
};
const VFS = {
  list: (dir: string): string[] =>
    Object.keys(CALLEES)
      .filter((path) => path.startsWith(`${dir}/`))
      .map((path) => path.slice(dir.length + 1))
      .filter((rest) => !rest.includes("/")),
  read: (path: string): string | undefined => CALLEES[path],
};

/** Run a workflow of `mark` leaves and report which ran, in order. A leaf marked `fail` reports a
 *  classified failure — errors are DATA (§4.2), so this is the shape a real failing child has. */
async function runMarking(
  files: Record<string, StateDef>,
  rootId: string,
): Promise<{ order: string[]; finished: string[]; outcome: string; reason?: string; probes: string[] }> {
  const order: string[] = [];
  // When each leaf FINISHED, as distinct from when it started. Overlap is a claim about the interval
  // between the two, and `order` alone cannot express it: an async child is started and not awaited,
  // so how many microtasks its input resolution takes decides which impl body runs first — a fact
  // about the number of bound slots, not about the scheduler.
  const finished: string[] = [];
  const probes: string[] = [];
  const registry = newRegistry();
  // Each leaf's output carries a RUN COUNT, so a looped child produces a different value every pass.
  // Without that the call memo — content-addressed on the resolved arguments — would answer the
  // second iteration's guard from the first's result, and a count could not see the second at all.
  const runs = new Map<string, number>();
  registry.functions.set(
    "mark",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const { name, fail, delayMs } = inputs as { name: string; fail?: boolean; delayMs?: number };
      order.push(name);
      const n = (runs.get(name) ?? 0) + 1;
      runs.set(name, n);
      if (delayMs !== undefined) await new Promise((resolve) => setTimeout(resolve, delayMs));
      finished.push(name);
      if (fail === true) return { error: { classification: "permanent" as const, reason: `${name} failed` } };
      return ok({ done: `${name}#${n}` }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  registry.functions.set(
    "probe",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const text = String(inputs.text ?? "");
      probes.push(text);
      return ok(text.toUpperCase()) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  registry.functions.set(
    "slowProbe",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const text = String(inputs.text ?? "");
      probes.push(text);
      await new Promise((resolve) => setTimeout(resolve, 40));
      return ok(text.toUpperCase()) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const bundle = loadBundle(files, rootId, { defaultRoot: [FUNCTIONS], roots: { JAIRA: "/p/.jaira", PROJECT: "/p" }, vfs: VFS });
  const engine = new WorkflowEngine({ bundle, registry });
  const result = await engine.run({ inputs: {} });
  return { order, finished, probes, outcome: result.outcome, ...(result.failure ? { reason: result.failure.reason } : {}) };
}

const marker = (name: string, fail = false, delayMs?: number): StateDef => ({
  label: name,
  outputs: { done: { schema: { type: "string" } } },
  operation: {
    kind: "function",
    function: "mark",
    args: { name, ...(fail ? { fail } : {}), ...(delayMs !== undefined ? { delayMs } : {}) },
  },
});

const leaves = (...names: string[]): Record<string, StateDef> =>
  Object.fromEntries(names.map((n) => [`root/${n}`, marker(n)]));

describe("a transition written on a child mount", () => {
  it("fires when that child finishes, diverting the sequence", async () => {
    const { order, outcome } = await runMarking(
      {
        root: {
          children: {
            a: { state: "root/a", transitions: [{ to: "c" }] },
            b: { state: "root/b" },
            c: { state: "root/c" },
          },
          sequence: ["a", "b", "c"],
        },
        ...leaves("a", "b", "c"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    // `b` is skipped: the jump moved the cursor to `c`, exactly as a state-level transition would.
    expect(order).toEqual(["a", "c"]);
  });

  it("is considered ONLY in the round that child's completion triggered", async () => {
    // `a` says "then b". Once `b` has finished, `a`'s list must be spent — a record stays `done`
    // forever, so a list that stayed eligible would send the run back into `b` after every `b`, and
    // this workflow would never terminate.
    const { order, outcome } = await runMarking(
      {
        root: { children: { a: { state: "root/a", transitions: [{ to: "b" }] }, b: { state: "root/b" } }, sequence: ["a", "b"] },
        ...leaves("a", "b"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["a", "b"]);
  });

  it("beats the state's own list, which is the more general statement", async () => {
    // Both lists match in the same round. The mount's wins, because "when a ends" is the narrower
    // claim — and a narrower rule a broader one could pre-empt is a rule nobody can rely on.
    const { order } = await runMarking(
      {
        root: {
          children: {
            a: { state: "root/a", transitions: [{ to: "specific" }] },
            general: { state: "root/general" },
            specific: { state: "root/specific", transitions: [{ to: "terminate.success" }] },
          },
          sequence: ["a"],
          transitions: [{ to: "general" }],
        },
        ...leaves("a", "general", "specific"),
      },
      "root",
    );
    expect(order).toEqual(["a", "specific"]);
  });

  it("guards read the enclosing state's scope, naming the child they are written on", async () => {
    const files = (guard: string): Record<string, StateDef> => ({
      root: {
        children: { a: { state: "root/a", transitions: [{ to: "taken", when: guard }] }, b: { state: "root/b" }, taken: { state: "root/taken" } },
        sequence: ["a", "b", "taken"],
      },
      ...leaves("a", "b", "taken"),
    });
    // One scope, spelled the same way it is spelled anywhere else in the state: the guard fires, the
    // cursor jumps to `taken`, and `b` stays skipped.
    expect((await runMarking(files(".children.a.output.done == 'a#1'"), "root")).order).toEqual(["a", "taken"]);
    expect((await runMarking(files(".children.a.output.done == 'nope'"), "root")).order).toEqual(["a", "b", "taken"]);
  });

  it("HANDLES the child's failure, rather than merely reacting to it", async () => {
    // Unhandled, a child that terminates with error takes the state down with it (SPEC §3.3). Routing
    // it from the mount is what handling looks like — and the point of writing it there is that it
    // cannot be confused with any OTHER child's failure.
    const files = (transitions?: { to: string }[]): Record<string, StateDef> => ({
      root: {
        children: { risky: { state: "root/risky", ...(transitions ? { transitions } : {}) }, recover: { state: "root/recover" } },
        sequence: ["risky"],
      },
      "root/risky": marker("risky", true),
      "root/recover": marker("recover"),
    });
    const unhandled = await runMarking(files(), "root");
    expect(unhandled.outcome).toBe("error");
    expect(unhandled.reason).toContain("no transition handled it");

    const handled = await runMarking(files([{ to: "recover" }]), "root");
    expect(handled.outcome).toBe("success");
    expect(handled.order).toEqual(["risky", "recover"]);
  });

  it("terminates the state, when that is what the mount says to do", async () => {
    const { order, outcome } = await runMarking(
      {
        root: {
          children: { a: { state: "root/a", transitions: [{ to: "terminate.success" }] }, b: { state: "root/b" } },
          sequence: ["a", "b"],
        },
        ...leaves("a", "b"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["a"]);
  });

  it("runs SPEC §3.3's worked example the way the spec walks through it", async () => {
    // Verbatim from the spec, so the two cannot drift: the recovery child is NOT a sequence member,
    // so entering it leaves the cursor where it was and the spine resumes at `review` afterwards.
    const files = (fails: boolean): Record<string, StateDef> => ({
      root: {
        children: {
          implement: { state: "root/implement", transitions: [{ to: "repair", when: ".children.implement.outcome === 'error'" }] },
          repair: { state: "root/repair" },
          review: { state: "root/review" },
        },
        sequence: ["implement", "review"],
      },
      "root/implement": marker("implement", fails),
      "root/repair": marker("repair"),
      "root/review": marker("review"),
    });
    const failed = await runMarking(files(true), "root");
    expect(failed.outcome).toBe("success"); // the transition HANDLED the failure
    expect(failed.order).toEqual(["implement", "repair", "review"]);

    const clean = await runMarking(files(false), "root");
    expect(clean.order).toEqual(["implement", "review"]);
  });

  it("does not divert a sibling's round — a child answers for its own completion only", async () => {
    // `b`'s list would match on data that is already true when `a` finishes. It must not fire then:
    // the round belongs to `a`, and `b` has not run.
    const { order } = await runMarking(
      {
        root: {
          children: {
            a: { state: "root/a" },
            b: { state: "root/b", transitions: [{ to: "elsewhere", when: ".children.a.output.done == 'a#1'" }] },
            elsewhere: { state: "root/elsewhere" },
          },
          sequence: ["a", "b"],
        },
        ...leaves("a", "b", "elsewhere"),
      },
      "root",
    );
    expect(order).toEqual(["a", "b", "elsewhere"]);
  });
});

/**
 * The edges — where "only in the round that child's completion triggered" meets async children,
 * timeouts, parking, and the calls a guard may embed.
 */
describe("a child mount's transitions, at the edges", () => {
  it("fires when an ASYNC child finishes, not when it starts", async () => {
    // An async child does not hold the cursor, so the spine walks past it while it runs. Its rule
    // must wait for the completion, or `late` would be entered before `slow` had produced anything.
    const { order, finished, outcome } = await runMarking(
      {
        root: {
          children: {
            slow: { state: "root/slow", async: true, transitions: [{ to: "late" }] },
            quick: { state: "root/quick" },
            late: { state: "root/late" },
          },
          sequence: ["slow", "quick"],
        },
        "root/slow": marker("slow", false, 20),
        "root/quick": marker("quick"),
        "root/late": marker("late"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    // `quick` runs while `slow` is still going; `late` only after `slow` reports. Read off COMPLETION
    // rather than entry: both are started before either finishes, so which of the two impl bodies
    // runs first is a microtask race that says nothing about the schedule. What the rule promises is
    // that `late` is not entered until `slow` has reported.
    expect(finished).toEqual(["quick", "slow", "late"]);
    expect(order.indexOf("late")).toBe(2);
  });

  /**
   * THE GUARANTEE: a child that finishes has its list evaluated, once, after it finished.
   *
   * Which ROUND is not the interesting part — two async children may land in one round or in two, and
   * nothing an author writes should depend on that. What must hold either way is that neither is
   * skipped. The call in each guard is how that is observed: `probe` runs when a guard is prepared,
   * so two invocations mean two lists were genuinely evaluated, and one means a completion was
   * swallowed. Both guards are false, so nothing fires and nothing pre-empts the other.
   */
  it("evaluates the list of EVERY child that finished — two async children that land together", async () => {
    const { order, outcome, probes } = await runMarking(
      {
        root: {
          children: {
            a: { state: "root/a", async: true, transitions: [{ to: "never", when: "probe(.children.a.output.done) === 'NOPE'" }] },
            b: { state: "root/b", async: true, transitions: [{ to: "never", when: "probe(.children.b.output.done) === 'NOPE'" }] },
            never: { state: "root/never" },
          },
          sequence: ["a", "b"],
        },
        "root/a": marker("a", false, 10),
        "root/b": marker("b", false, 10),
        "root/never": marker("never"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect([...probes].sort()).toEqual(["a#1", "b#1"]);
    expect(order).toEqual(["a", "b"]); // neither guard fired
  });

  it("evaluates a child that finishes WHILE a round is already running", async () => {
    // The window the previous test cannot force. `a`'s guard call takes 40ms, so the round it
    // triggered is still in flight when `b` finishes at 10ms. `b`'s rule is true and nothing else
    // reaches `after`, so if that completion is written off as "already covered by the round in
    // progress" — whose guards were collected before `b` existed — `after` never runs at all.
    const { order, outcome } = await runMarking(
      {
        root: {
          children: {
            a: { state: "root/a", async: true, transitions: [{ to: "never", when: "slowProbe(.children.a.output.done) === 'NOPE'" }] },
            b: { state: "root/b", async: true, transitions: [{ to: "after", when: "probe(.children.b.output.done) === 'B#1'" }] },
            after: { state: "root/after" },
            never: { state: "root/never" },
          },
          sequence: ["a", "b"],
        },
        "root/a": marker("a", false, 1),
        "root/b": marker("b", false, 10),
        "root/after": marker("after"),
        "root/never": marker("never"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toContain("after");
    expect(order).not.toContain("never");
  });

  it("checks a LOOPED child's list once per pass", async () => {
    // Four completions, four evaluations — the eligibility is per completion, not per child. The
    // output carries a pass counter so each pass calls with a different argument: the call memo is
    // content-addressed, so identical arguments would (correctly) be answered once and this count
    // could not see the later passes at all.
    const { order, probes, outcome } = await runMarking(
      {
        root: {
          children: {
            retry: {
              state: "root/retry",
              transitions: [{ to: "retry", when: "probe(.children.retry.output.done) !== 'STOP' && .run.iteration < 3" }],
            },
          },
          sequence: ["retry"],
          limits: { max_iterations: 3 },
        },
        "root/retry": marker("retry"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["retry", "retry", "retry", "retry"]);
    expect(probes).toEqual(["retry#1", "retry#2", "retry#3", "retry#4"]);
  });

  it("handles a child that TIMED OUT, not just one that errored", async () => {
    const { order, outcome } = await runMarking(
      {
        root: {
          children: {
            stuck: { state: "root/stuck", transitions: [{ to: "recover", when: ".children.stuck.outcome === 'timeout'" }] },
            recover: { state: "root/recover" },
          },
          sequence: ["stuck"],
        },
        "root/stuck": { ...marker("stuck", false, 200), limits: { timeout: 0.01 } },
        "root/recover": marker("recover"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["stuck", "recover"]);
  });

  it("re-grants eligibility on each completion, so a mount can loop its own child", async () => {
    // The eligibility is spent per COMPLETION, not once per child — a re-entered child gets a fresh
    // instance and a fresh round. `run.iteration` is what stops it, exactly as at state level.
    const { order, outcome } = await runMarking(
      {
        root: {
          children: { retry: { state: "root/retry", transitions: [{ to: "retry", when: ".run.iteration < 3" }] } },
          sequence: ["retry"],
          limits: { max_iterations: 3 },
        },
        "root/retry": marker("retry"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    expect(order).toEqual(["retry", "retry", "retry", "retry"]);
  });

  // A call embedded in a mount's guard runs only in that child's round — in `callDispatch.test.ts`,
  // which owns the callee-document harness a lowered call needs.

  it("KEEPS eligibility when the target parks, because that round decided nothing", async () => {
    // `trigger`'s rule enters `join`, whose inputs read a still-running async child: the entry parks.
    // Parking is not an answer, so the rule survives to the round that can answer it. Spending it
    // there would strand `join`, which nothing else reaches — it is not a sequence member.
    const { finished, outcome } = await runMarking(
      {
        root: {
          children: {
            slow: { state: "root/slow", async: true },
            trigger: { state: "root/trigger", transitions: [{ to: "join" }] },
            join: { state: "root/join", inputs: { from: ".children.slow.output.done" } },
          },
          sequence: ["slow", "trigger"],
        },
        "root/slow": marker("slow", false, 30),
        "root/trigger": marker("trigger"),
        "root/join": { ...marker("join"), inputs: { from: { schema: { type: "string" } } } },
      },
      "root",
    );
    expect(outcome).toBe("success");
    // By COMPLETION, since `slow` and `trigger` are both in flight before either finishes and which
    // impl body runs first is a microtask race (see `runMarking`). What matters is that `join` is
    // last: it could not have been entered until the child its inputs read had reported.
    expect(finished).toEqual(["trigger", "slow", "join"]);
  });

  it("spends the round on a guard that reads a child still running — PENDING is skipped, not deferred", async () => {
    // The sharp edge of "only in the round its completion triggered": a guard over an unresolved
    // async child evaluates to PENDING, PENDING is skipped, and the round is spent. The rule does not
    // wait for a later round, because a later round is not the one this child ended in. A rule that
    // has to see another child's result belongs on THAT child's mount.
    const { order, outcome } = await runMarking(
      {
        root: {
          children: {
            slow: { state: "root/slow", async: true },
            trigger: { state: "root/trigger", transitions: [{ to: "never", when: ".children.slow.output.done === 'slow#1'" }] },
            never: { state: "root/never" },
          },
          sequence: ["slow", "trigger"],
        },
        "root/slow": marker("slow", false, 30),
        "root/trigger": marker("trigger"),
        "root/never": marker("never"),
      },
      "root",
    );
    expect(outcome).toBe("success");
    // Sorted, because the claim is about WHICH children ran and not in what order: `never` is the
    // one the rule would have entered had PENDING deferred instead of being skipped.
    expect([...order].sort()).toEqual(["slow", "trigger"]);
  });
});

describe("the lint over a child mount's transitions", () => {
  const bundleWith = (transitions: { to: string; when?: string }[]): ReturnType<typeof validateBundle> =>
    validateBundle(
      loadBundle(
        {
          root: { children: { a: { state: "root/a", transitions }, b: { state: "root/b" } }, sequence: ["a", "b"] },
          ...leaves("a", "b"),
        },
        "root",
      ),
    );

  it("reports an unknown target against the line the author wrote", () => {
    const { errors } = bundleWith([{ to: "nowhere" }]);
    expect(errors.map((e) => e.path)).toContain("children.a.transitions[0].to");
    expect(errors[0]!.message).toContain("neither a declared child nor a terminate.* outcome");
  });

  it("holds a guard to the same boolean rule the state's own list follows", () => {
    const { errors } = bundleWith([{ to: "b", when: ".children.a.output.done" }]);
    expect(errors.map((e) => e.path)).toContain("children.a.transitions[0].when");
    expect(errors[0]!.message).toContain("must infer to boolean");
  });

  it("accepts a well-formed one", () => {
    expect(bundleWith([{ to: "b", when: ".children.a.output.done == 'a#1'" }]).errors).toEqual([]);
  });
});
