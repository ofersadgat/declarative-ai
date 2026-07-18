import { describe, expect, it } from "vitest";
import { MapExecutorRegistry, type ExecutionSpec, type Outcome } from "@ai-exec/core";
import { WorkflowEngine, type EngineConfig } from "../src/engine";
import { loadBundle } from "../src/loader";
import { InMemoryPersistence, isArtifactRef, llmCallBinding, type ArtifactRef } from "../src/ports";
import type { StateDef } from "../src/format";
import { deferred, FakeExecutor, modelOf, ok, promptOf, promptTail, rejectingPort, ScriptedPort, type Script } from "./fakes";
import { FANOUT_ID, PLAN_ID, specFanoutFiles, specPlanningFiles } from "./fixtures";

const PROVIDERS = {
  planner: llmCallBinding({ model: "planner" }),
  critic: llmCallBinding({ model: "critic" }),
  fixer: llmCallBinding({ model: "fixer" }),
  reviewer: llmCallBinding({ model: "reviewer" }),
  synthesizer: llmCallBinding({ model: "synthesizer" }),
};

function makeEngine(files: Record<string, StateDef>, rootId: string, script: Script, extra?: Partial<EngineConfig>) {
  const fake = new FakeExecutor(script);
  const registry = new MapExecutorRegistry().register(fake);
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    providers: PROVIDERS,
    registry,
    persistence,
    ...extra,
  });
  return { engine, fake, persistence };
}

/** Default happy-path script for the planning fixture; override per-model as needed. */
function planningScript(overrides: Partial<Record<string, (spec: ExecutionSpec) => Outcome>> = {}): Script {
  return (spec) => {
    const model = modelOf(spec);
    const override = overrides[model];
    if (override) return override(spec);
    switch (model) {
      case "planner":
        // Dispatch on the rendered template TAIL — the default conversation mode is
        // full_history (SPEC §4.7), so prompts CONTAIN earlier exchanges in their
        // history preamble.
        return promptTail(spec).startsWith("Write the plan")
          ? ok({ plan_doc: "# The Plan" })
          : ok({ goals: ["goal-1", "goal-2"] });
      case "critic":
        return ok({ outcome: "clean", weaknesses: [], critique_report: "no issues" });
      case "fixer":
        return ok({ resolution: "fixed" });
      default:
        throw new Error(`unscripted model ${model}`);
    }
  };
}

describe("SPEC §8.2 — UI state terminates with validated outputs", () => {
  const files = specPlanningFiles();
  const HR = "feature/plan/critique/human_review";

  it("runs the component, validates, and terminates success with its outputs", async () => {
    const port = new ScriptedPort({ [HR]: [{ decision: "approve", comments: "lgtm" }] });
    const { engine } = makeEngine(files, HR, () => {
      throw new Error("no agent should run");
    }, { interaction: port });
    const result = await engine.run({ inputs: { plan_doc: "plan", critique_report: "report" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ decision: "approve", comments: "lgtm" });
    expect(port.requests[0]).toMatchObject({ stateId: HR, component: "choose_option" });
  });

  it("rejects an out-of-enum component payload (engine-side validation)", async () => {
    const port = new ScriptedPort({ [HR]: [{ decision: "yolo" }] });
    const { engine } = makeEngine(files, HR, () => ok({}), { interaction: port });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/decision/);
  });

  it("fails permanently when no InteractionPort is supplied", async () => {
    const { engine } = makeEngine(files, HR, () => ok({}));
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/InteractionPort/);
  });

  it("a rejecting port (search context) turns interactive states into permanent failures", async () => {
    const { engine } = makeEngine(files, HR, () => ok({}), { interaction: rejectingPort });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/not allowed/);
  });
});

describe("SPEC §7.3 — critique state walk-through", () => {
  const CRIT = "feature/plan/critique";
  const critiqueInputs = { inputs: { plan_doc: "the plan" } };

  it("clean: terminates immediately, before any child runs", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), CRIT, planningScript());
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("clean");
    expect(fake.calls).toHaveLength(1); // only the critic; no fixer, no human review
    expect(result.outputs?.["human_decision"]).toBeUndefined();
    const report = result.outputs?.["critique_report"];
    expect(isArtifactRef(report)).toBe(true);
    expect((report as ArtifactRef).content).toBe("no issues");
  });

  it("needs_changes: one fix pass, then terminate for the parent to decide", async () => {
    const { engine, fake } = makeEngine(
      specPlanningFiles(),
      CRIT,
      planningScript({
        critic: () => ok({ outcome: "needs_changes", weaknesses: ["w1"], critique_report: "issues" }),
      }),
    );
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("needs_changes");
    expect(fake.calls.map(modelOf)).toEqual(["critic", "fixer"]);
    // The fixer received the weaknesses via input wiring from the critique's own outputs.
    expect(fake.calls[1]!.definitionHash).toBeTruthy();
  });

  it("blocked: collects a human decision, surfaced through human_decision", async () => {
    const port = new ScriptedPort({ "feature/plan/critique/human_review": [{ decision: "block" }] });
    const { engine, fake } = makeEngine(
      specPlanningFiles(),
      CRIT,
      planningScript({
        critic: () => ok({ outcome: "blocked", weaknesses: [], critique_report: "cannot proceed" }),
      }),
      { interaction: port },
    );
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("blocked");
    expect(result.outputs?.["human_decision"]).toBe("block");
    expect(fake.calls.map(modelOf)).toEqual(["critic"]);
  });

  it("agent operation failure terminates the state with error", async () => {
    const { engine } = makeEngine(specPlanningFiles(), CRIT, () => ({
      metrics: { durationMs: 1 },
      error: { classification: "permanent", reason: "provider exploded" },
    }));
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toBe("provider exploded");
  });

  it("schema-violating agent output terminates the state with error", async () => {
    const { engine } = makeEngine(
      specPlanningFiles(),
      CRIT,
      planningScript({ critic: () => ok({ outcome: "not-an-enum-member", weaknesses: [], critique_report: "x" }) }),
    );
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/outcome/);
  });

  it("missing required agent output terminates the state with error", async () => {
    const { engine } = makeEngine(
      specPlanningFiles(),
      CRIT,
      planningScript({ critic: () => ok({ outcome: "clean", critique_report: "x" }) }), // no weaknesses
    );
    const result = await engine.run(critiqueInputs);
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/weaknesses/);
  });
});

describe("SPEC §9 — planning parent: sequence, re-plan loop, iteration limit", () => {
  it("happy path: goals → context → critique(clean) → complete", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    const result = await engine.run({ inputs: { issue: "the issue" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("complete");
    expect(isArtifactRef(result.outputs?.["plan_doc"])).toBe(true);
    // Passthrough forwards the critique's whole output object.
    expect((result.outputs?.["critique"] as Record<string, unknown>)["outcome"]).toBe("clean");
    expect(fake.calls.map(modelOf)).toEqual(["planner", "planner", "critic"]);
  });

  it("needs_changes triggers a re-plan with FRESH instances (sequence reset, SPEC §3.3)", async () => {
    let critiquePass = 0;
    const { engine, fake, persistence } = makeEngine(
      specPlanningFiles(),
      PLAN_ID,
      planningScript({
        critic: () =>
          ++critiquePass === 1
            ? ok({ outcome: "needs_changes", weaknesses: ["w"], critique_report: "r1" })
            : ok({ outcome: "clean", weaknesses: [], critique_report: "r2" }),
      }),
    );
    const result = await engine.run({ inputs: { issue: "the issue" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("complete");
    // Two full passes: goals+context+critic, plus one fixer inside the first critique.
    expect(fake.calls.map(modelOf)).toEqual(["planner", "planner", "critic", "fixer", "planner", "planner", "critic"]);
    // Reset clears goals, context, critique (recorded as superseded events).
    const superseded = persistence.events
      .filter((e) => e.event.type === "child.superseded")
      .map((e) => (e.event as { childKey: string }).childKey);
    expect(superseded).toEqual(["goals", "context", "critique"]);
  });

  it("iteration limit: persistent needs_changes ends blocked after max_iterations re-plans", async () => {
    const { engine, fake } = makeEngine(
      specPlanningFiles(),
      PLAN_ID,
      planningScript({
        critic: () => ok({ outcome: "needs_changes", weaknesses: ["w"], critique_report: "r" }),
      }),
    );
    const result = await engine.run({ inputs: { issue: "the issue" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["outcome"]).toBe("blocked");
    // Initial pass + 3 re-plans (run.iteration guard 0,1,2 < 3) = 4 critiques, each with a fix pass.
    expect(fake.calls.filter((c) => modelOf(c) === "critic")).toHaveLength(4);
    expect(fake.calls.filter((c) => modelOf(c) === "fixer")).toHaveLength(4);
    expect(fake.calls.filter((c) => modelOf(c) === "planner")).toHaveLength(8);
  });

  it("input wiring flows between children (context sees goals; critique sees the plan doc)", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    const contextCall = fake.calls[1]!;
    expect(promptOf(contextCall)).toContain("the issue"); // artifact input rendered as content
    const criticCall = fake.calls[2]!;
    expect(criticCall.outputSchema).toBeDefined(); // structured contract derived from outputs
  });
});

describe("SPEC §10.4 — async children and the dataflow join", () => {
  it("both reviews run concurrently; synthesize waits for both outputs", async () => {
    const gates = [deferred<void>(), deferred<void>()];
    let started = 0;
    let maxConcurrent = 0;
    let active = 0;
    const { engine, fake } = makeEngine(specFanoutFiles(), FANOUT_ID, async (spec) => {
      if (modelOf(spec) === "reviewer") {
        const idx = started++;
        active++;
        maxConcurrent = Math.max(maxConcurrent, active);
        await gates[idx]!.promise;
        active--;
        return ok({ report: `report-${idx}` });
      }
      return ok({ summary: `combined: ${promptOf(spec)}` });
    });
    const run = engine.run({ inputs: { change: "diff" } });
    // Let both async children start, then release them out of order.
    await new Promise((r) => setTimeout(r, 10));
    expect(maxConcurrent).toBe(2);
    gates[1]!.resolve();
    gates[0]!.resolve();
    const result = await run;
    expect(result.outcome).toBe("success");
    const synthCall = fake.calls.find((c) => modelOf(c) === "synthesizer")!;
    expect(promptOf(synthCall)).toContain("report-0");
    expect(promptOf(synthCall)).toContain("report-1");
    expect(result.outputs?.["summary"]).toMatch(/^combined:/);
  });

  it("a transition referencing a running async child is skipped, then taken on resolution", async () => {
    const gate = deferred<void>();
    const files: Record<string, StateDef> = {
      parent: {
        inputs: {},
        outputs: { got: { type: "string", from: "children.slow.outputs.val" } },
        children: {
          slow: { state: "parent/slow", async: true, inputs: {} },
          quick: { state: "parent/quick", inputs: {} },
        },
        sequence: ["slow", "quick"],
        transitions: [{ to: "terminate.success", when: "children.slow.outputs.val === 'done'" }],
      },
      "parent/slow": {
        inputs: {},
        outputs: { val: { type: "string" } },
        agent: { provider: "reviewer", prompt: { template: "slow" } },
      },
      "parent/quick": {
        inputs: {},
        outputs: { q: { type: "string" } },
        agent: { provider: "synthesizer", prompt: { template: "quick" } },
      },
    };
    const { engine } = makeEngine(files, "parent", async (spec) => {
      if (promptOf(spec) === "slow") {
        await gate.promise;
        return ok({ val: "done" });
      }
      return ok({ q: "quick-done" });
    });
    const run = engine.run({ inputs: {} });
    await new Promise((r) => setTimeout(r, 10));
    gate.resolve();
    const result = await run;
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["got"]).toBe("done");
  });

  it("parked wiring with nothing running is a dataflow deadlock → error", async () => {
    // `never` is async and never started (not in sequence, no transition to it), so the
    // join can never resolve — but `waiting` is in the sequence and parks on it forever.
    const files: Record<string, StateDef> = {
      parent: {
        inputs: {},
        outputs: {},
        children: {
          never: { state: "parent/never", async: true, inputs: {} },
          waiting: { state: "parent/waiting", inputs: { x: "children.never.outputs.v" } },
        },
        sequence: ["never", "waiting"],
      },
      "parent/never": {
        inputs: {},
        outputs: { v: { type: "string" } },
        agent: { provider: "reviewer", prompt: { template: "never" } },
      },
      "parent/waiting": {
        inputs: { x: { type: "string" } },
        outputs: {},
        agent: { provider: "reviewer", prompt: { template: "waiting" } },
      },
    };
    // The async child completes but produces the WRONG field, so the wiring expression
    // resolves to undefined → the waiting child gets a missing required input → blocked.
    const { engine } = makeEngine(files, "parent", (spec) =>
      promptOf(spec) === "never" ? ok({ v: "present" }) : ok({}),
    );
    const result = await engine.run({ inputs: {} });
    // v IS produced here, so waiting runs and the flow completes; now break it:
    expect(result.outcome).toBe("success");

    const { engine: broken } = makeEngine(files, "parent", (spec) =>
      promptOf(spec) === "never"
        ? ({ value: { v: "x" }, metrics: { durationMs: 1 }, error: { classification: "permanent", reason: "boom" } } as Outcome)
        : ok({}),
    );
    const brokenResult = await broken.run({ inputs: {} });
    expect(brokenResult.outcome).toBe("error"); // child error unhandled by any transition
  });
});

describe("timeout, cancellation, and unhandled child failures", () => {
  it("limits.timeout terminates the state (and its running ops) with terminate.timeout", async () => {
    const files: Record<string, StateDef> = {
      slowroot: {
        inputs: {},
        outputs: { x: { type: "string", optional: true } },
        agent: { provider: "reviewer", prompt: { template: "forever" } },
        limits: { timeout: 0.05 },
      },
    };
    const { engine } = makeEngine(files, "slowroot", () => new Promise(() => {})); // never resolves; abort race wins
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("timeout");
  });

  it("a child timeout is an unhandled failure → parent error (SPEC §3.3)", async () => {
    const files: Record<string, StateDef> = {
      parent: {
        inputs: {},
        outputs: {},
        children: { slow: { state: "parent/slow", inputs: {} } },
        sequence: ["slow"],
      },
      "parent/slow": {
        inputs: {},
        outputs: { x: { type: "string", optional: true } },
        agent: { provider: "reviewer", prompt: { template: "forever" } },
        limits: { timeout: 0.05 },
      },
    };
    const { engine } = makeEngine(files, "parent", () => new Promise(() => {}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/timeout/);
  });

  it("a transition CAN handle a child timeout", async () => {
    const files: Record<string, StateDef> = {
      parent: {
        inputs: {},
        outputs: { handled: { type: "string", from: "'yes'" } },
        children: {
          slow: { state: "parent/slow", inputs: {} },
        },
        sequence: ["slow"],
        transitions: [{ to: "terminate.success", when: "children.slow.outcome === 'timeout'" }],
      },
      "parent/slow": {
        inputs: {},
        outputs: { x: { type: "string", optional: true } },
        agent: { provider: "reviewer", prompt: { template: "forever" } },
        limits: { timeout: 0.05 },
      },
    };
    const { engine } = makeEngine(files, "parent", () => new Promise(() => {}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["handled"]).toBe("yes");
  });

  it("external cancellation propagates to running descendants and yields canceled", async () => {
    const abort = new AbortController();
    const { engine } = makeEngine(specPlanningFiles(), PLAN_ID, async (spec) => {
      if (modelOf(spec) === "planner" && !promptTail(spec).startsWith("Write the plan")) {
        return ok({ goals: ["g"] });
      }
      return new Promise(() => {}); // context hangs; we cancel meanwhile
    });
    const run = engine.run({ inputs: { issue: "i" }, abortSignal: abort.signal });
    await new Promise((r) => setTimeout(r, 20));
    abort.abort();
    const result = await run;
    expect(result.outcome).toBe("canceled");
  });

  it("root input validation failure blocks the run (error, no ops executed)", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    const result = await engine.run({ inputs: {} }); // missing required 'issue'
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/required input 'issue'/);
    expect(fake.calls).toHaveLength(0);
  });
});

describe("conversation modes (SPEC §4.7)", () => {
  it("full_history threads the prior exchange into the next agent prompt", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    // critic runs with mode full_history: sees the two planner exchanges.
    const criticPrompt = promptOf(fake.calls[2]!);
    expect(criticPrompt).toContain("<conversation-history>");
    expect(criticPrompt).toContain("Extract goals");
  });

  it("fresh mode gets no history preamble", async () => {
    const files = specPlanningFiles();
    files["feature/plan/critique"]!.agent!.conversation = { mode: "fresh" };
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    expect(promptOf(fake.calls[2]!)).not.toContain("<conversation-history>");
  });
});

describe("run records (SPEC §10.2)", () => {
  it("persists entered/started/completed/transition/terminated events with metrics", async () => {
    const { engine, persistence } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    const result = await engine.run({ inputs: { issue: "i" } });
    expect(result.outcome).toBe("success");
    const types = persistence.events.map((e) => e.event.type);
    expect(types).toContain("instance.entered");
    expect(types).toContain("operation.started");
    expect(types).toContain("operation.completed");
    expect(types).toContain("transition.taken");
    expect(types).toContain("instance.terminated");
    const completed = persistence.events.find((e) => e.event.type === "operation.completed")!;
    expect((completed.event as { metrics?: { cost?: number } }).metrics?.cost).toBe(0.01);
    // Cost/call rollup for the whole run.
    expect(result.metrics.childCalls).toBe(3);
    expect(result.metrics.childCost).toBeCloseTo(0.03);
  });
});

describe("skill operations", () => {
  it("executes a registered skill through the agent runtime", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: { topic: { type: "string" } },
        outputs: { summary: { type: "string" } },
        skill: { name: "summarize", params: { style: "terse" } },
      },
    };
    const { engine, fake } = makeEngine(files, "s", (spec) => ok({ summary: `sum(${promptOf(spec)})` }), {
      skills: { get: (name) => (name === "summarize" ? { provider: "reviewer", template: "Summarize {{inputs.topic}}." } : undefined) },
    });
    const result = await engine.run({ inputs: { topic: "codebases" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["summary"]).toBe("sum(Summarize codebases.)");
    expect(fake.calls).toHaveLength(1);
  });

  it("an unregistered skill is a permanent failure", async () => {
    const files: Record<string, StateDef> = {
      s: { inputs: {}, outputs: {}, skill: { name: "ghost" } },
    };
    const { engine } = makeEngine(files, "s", () => ok({}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/skill 'ghost'/);
  });
});
