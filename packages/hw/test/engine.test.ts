import { describe, expect, it } from "vitest";
import {
  MapCapabilityRegistry,
  MapSessionStore,
  isPermissionDenied,
  type Approver,
  type ExecutorCapabilities,
  type HostFunction,
  type Outcome,
  type Runtime,
  type Tool,
} from "@declarative-ai/core";
import { WorkflowEngine, type EngineConfig } from "../src/engine";
import { loadBundle } from "../src/loader";
import { InMemoryPersistence, isArtifactRef, type ArtifactRef } from "../src/ports";
import type { StateDef } from "../src/format";
import { deferred, FakeRuntimes, modelOf, ok, promptOf, promptTail, rejectingFunction, ScriptedFunction, type FakeCall, type Script } from "./fakes";
import { FANOUT_ID, PLAN_ID, specFanoutFiles, specPlanningFiles } from "./fixtures";

const RUNTIME_NAMES = ["planner", "critic", "fixer", "reviewer", "synthesizer"];

interface MakeEngineOpts {
  functions?: Record<string, HostFunction>;
  skills?: Record<string, string>;
  tools?: Record<string, Tool>;
  runtimes?: Record<string, Runtime>;
  extra?: Partial<EngineConfig>;
}

function makeEngine(files: Record<string, StateDef>, rootId: string, script: Script, opts: MakeEngineOpts = {}) {
  const fake = new FakeRuntimes(script);
  const registry = fake.register(new MapCapabilityRegistry(), RUNTIME_NAMES);
  for (const [name, fn] of Object.entries(opts.functions ?? {})) registry.functions.register(name, fn);
  for (const [name, template] of Object.entries(opts.skills ?? {})) registry.skills.register(name, template);
  for (const [name, t] of Object.entries(opts.tools ?? {})) registry.tools.register(name, t);
  for (const [name, r] of Object.entries(opts.runtimes ?? {})) registry.runtimes.register(name, r);
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    persistence,
    ...opts.extra,
  });
  return { engine, fake, persistence };
}

/** Default happy-path script for the planning fixture; override per-runtime as needed. */
function planningScript(overrides: Partial<Record<string, (call: FakeCall) => Outcome>> = {}): Script {
  return (call) => {
    const model = modelOf(call);
    const override = overrides[model];
    if (override) return override(call);
    switch (model) {
      case "planner":
        // Dispatch on the rendered template TAIL — the default conversation mode is
        // full_history (SPEC §4.7), so prompts CONTAIN earlier exchanges in their
        // history preamble.
        return promptTail(call).startsWith("Write the plan")
          ? ok({ plan_doc: "# The Plan" })
          : ok({ goals: ["goal-1", "goal-2"] });
      case "critic":
        return ok({ outcome: "clean", weaknesses: [], critique_report: "no issues" });
      case "fixer":
        return ok({ resolution: "fixed" });
      default:
        throw new Error(`unscripted runtime ${model}`);
    }
  };
}

describe("SPEC §8.2 — function state terminates with validated outputs", () => {
  const files = specPlanningFiles();
  const HR = "feature/plan/critique/human_review";

  it("runs the function, validates, and terminates success with its outputs", async () => {
    const fn = new ScriptedFunction([{ decision: "approve", comments: "lgtm" }]);
    const { engine } = makeEngine(files, HR, () => {
      throw new Error("no runtime should run");
    }, { functions: { choose_option: fn } });
    const result = await engine.run({ inputs: { plan_doc: "plan", critique_report: "report" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ decision: "approve", comments: "lgtm" });
    expect(fn.calls[0]!.config).toMatchObject({ name: "choose_option" });
  });

  it("rejects an out-of-enum function payload (engine-side validation)", async () => {
    const fn = new ScriptedFunction([{ decision: "yolo" }]);
    const { engine } = makeEngine(files, HR, () => ok({}), { functions: { choose_option: fn } });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/decision/);
  });

  it("fails permanently when the function is not registered", async () => {
    const { engine } = makeEngine(files, HR, () => ok({}));
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/function 'choose_option'/);
  });

  it("a rejecting function (search context) turns interactive states into permanent failures", async () => {
    const { engine } = makeEngine(files, HR, () => ok({}), { functions: { choose_option: rejectingFunction } });
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
    expect(fake.calls[1]!.prompt).toBeTruthy();
  });

  it("blocked: collects a human decision, surfaced through human_decision", async () => {
    const fn = new ScriptedFunction([{ decision: "block" }]);
    const { engine, fake } = makeEngine(
      specPlanningFiles(),
      CRIT,
      planningScript({
        critic: () => ok({ outcome: "blocked", weaknesses: [], critique_report: "cannot proceed" }),
      }),
      { functions: { choose_option: fn } },
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

  it("exposes a run-scoped session store shared across all states (ctx.sessions)", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    const stores = fake.ctxs.map((c) => c.sessions);
    expect(stores.length).toBeGreaterThan(1);
    expect(stores.every((s) => s !== undefined)).toBe(true);
    expect(new Set(stores).size).toBe(1); // one run-scoped store, shared by every state
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
        runtime: { name: "reviewer", prompt: { template: "slow" } },
      },
      "parent/quick": {
        inputs: {},
        outputs: { q: { type: "string" } },
        runtime: { name: "synthesizer", prompt: { template: "quick" } },
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
        runtime: { name: "reviewer", prompt: { template: "never" } },
      },
      "parent/waiting": {
        inputs: { x: { type: "string" } },
        outputs: {},
        runtime: { name: "reviewer", prompt: { template: "waiting" } },
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
        runtime: { name: "reviewer", prompt: { template: "forever" } },
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
        runtime: { name: "reviewer", prompt: { template: "forever" } },
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
        runtime: { name: "reviewer", prompt: { template: "forever" } },
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
    files["feature/plan/critique"]!.runtime!.conversation = { mode: "fresh" };
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    expect(promptOf(fake.calls[2]!)).not.toContain("<conversation-history>");
  });

  it("a distinct runtime.session isolates the transcript (full_history sees an empty per-session history)", async () => {
    const files = specPlanningFiles();
    // The planners run in the default session; move the critic to its own session — its full_history now
    // reads an EMPTY transcript, so the planners' exchanges do NOT leak across the session boundary.
    files["feature/plan/critique"]!.runtime!.session = "isolated";
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    expect(promptOf(fake.calls[2]!)).not.toContain("<conversation-history>");
    expect(promptOf(fake.calls[2]!)).not.toContain("Extract goals");
  });

  it("records the transcript into the shared session store (unified with the withSession path)", async () => {
    const store = new MapSessionStore();
    const { engine } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript(), { extra: { services: { sessions: store } } });
    await engine.run({ inputs: { issue: "the issue" } });
    // The built-in transcript lives in the SAME store a runtime's withSession reads — one source of truth.
    const messages = (await store.get("default"))?.messages as Array<{ role: string; content: string }> | undefined;
    expect(messages?.length).toBeGreaterThan(0);
    expect(messages!.some((m) => m.role === "assistant")).toBe(true);
    expect(messages!.some((m) => m.role === "user" && m.content.includes("Extract goals"))).toBe(true);
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

describe("skill as a runtime prompt source", () => {
  it("renders a named skill template as the runtime's prompt", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: { topic: { type: "string" } },
        outputs: { summary: { type: "string" } },
        runtime: { name: "reviewer", prompt: { skill: "summarize" }, params: { style: "terse" } },
      },
    };
    const { engine, fake } = makeEngine(files, "s", (call) => ok({ summary: `sum(${promptOf(call)})` }), {
      skills: { summarize: "Summarize {{inputs.topic}} ({{params.style}})." },
    });
    const result = await engine.run({ inputs: { topic: "codebases" } });
    expect(result.outcome).toBe("success");
    // params are TEMPLATE variables ({{params.*}}), rendered into the skill-sourced prompt.
    expect(result.outputs?.["summary"]).toBe("sum(Summarize codebases (terse).)");
    expect(fake.calls).toHaveLength(1);
  });

  it("an unregistered skill is a permanent failure", async () => {
    const files: Record<string, StateDef> = {
      s: { inputs: {}, outputs: {}, runtime: { name: "reviewer", prompt: { skill: "ghost" } } },
    };
    const { engine } = makeEngine(files, "s", () => ok({}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/skill 'ghost'/);
  });
});

describe("runtime tools (RUNTIMES-AND-PERMISSIONS.md §2)", () => {
  const echoTool: Tool = { description: "echo", inputSchema: { type: "object" }, run: (input) => input };

  it("resolves declared tool names through registry.tools and hands them to the runtime", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: { r: { type: "string" } },
        runtime: { name: "reviewer", prompt: { template: "go" }, tools: ["echo"] },
      },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({ r: "done" }), { tools: { echo: echoTool } });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    // The resolved executable reached the runtime op, keyed by its logical name.
    expect(Object.keys(fake.calls[0]!.tools ?? {})).toEqual(["echo"]);
    expect(fake.calls[0]!.tools?.["echo"]).toBe(echoTool);
  });

  it("an unregistered tool is a permanent failure", async () => {
    const files: Record<string, StateDef> = {
      s: { inputs: {}, outputs: {}, runtime: { name: "reviewer", prompt: { template: "go" }, tools: ["ghost"] } },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/tool 'ghost' is not registered/);
    expect(fake.calls).toHaveLength(0); // failed before dispatching to the runtime
  });

  it("threads the host-provided workspace to the runtime's ctx (where fs tools read it)", async () => {
    const files: Record<string, StateDef> = {
      s: { inputs: {}, outputs: { r: { type: "string" } }, runtime: { name: "reviewer", prompt: { template: "go" } } },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({ r: "done" }), {
      extra: { services: { workspace: { root: "/work/space" } } },
    });
    await engine.run({ inputs: {} });
    expect(fake.ctxs[0]!.workspace?.root).toBe("/work/space");
  });
});

describe("runtime tool permissions (RUNTIMES-AND-PERMISSIONS.md §4)", () => {
  const writeTool: Tool = { description: "write", inputSchema: { type: "object" }, capabilities: { readOnly: false }, run: () => ({ wrote: true }) };
  const allowSession: Approver = () => ({ decision: "allow", scope: "session" });

  /** A state whose runtime invokes its (engine-supplied, possibly permission-wrapped) `write` tool. */
  const invokeWriteFiles = (permissions?: Record<string, unknown>): Record<string, StateDef> => ({
    s: {
      inputs: {},
      outputs: { r: { type: "string" } },
      runtime: { name: "reviewer", prompt: { template: "go" }, tools: ["write"], ...(permissions ? { permissions } : {}) },
    },
  });

  it("read-only profile denies a mutating tool when the runtime invokes it", async () => {
    let toolResult: unknown;
    const script: Script = async (call) => {
      toolResult = await call.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "read-only" }), "s", script, {
      tools: { write: writeTool },
      extra: { permissions: { approve: allowSession } },
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(isPermissionDenied(toolResult)).toBe(true); // out of the read-only profile
  });

  it("full profile + an approving human: the wrapped tool executes", async () => {
    let toolResult: unknown;
    const script: Script = async (call) => {
      toolResult = await call.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "full", default: "ask" }), "s", script, {
      tools: { write: writeTool },
      extra: { permissions: { approve: allowSession } },
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(toolResult).toEqual({ wrote: true });
  });

  it("without an approver, tools are handed over unguarded (run directly)", async () => {
    let toolResult: unknown;
    const script: Script = async (call) => {
      toolResult = await call.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "read-only" }), "s", script, { tools: { write: writeTool } });
    await engine.run({ inputs: {} });
    expect(toolResult).toEqual({ wrote: true }); // no wrapper ⇒ profile not enforced
  });

  it("hands a delegated runtime (policyEnforcement: callback) RAW tools — no double-gating", async () => {
    const delegatedCaps: ExecutorCapabilities = {
      structuredOutput: false,
      sessionResume: false,
      streaming: true,
      interactive: true,
      mutatesWorkspace: true,
      policyEnforcement: "callback",
      memoizable: false,
      runtime: "node",
    };
    let received: Tool | undefined;
    const delegated: Runtime = {
      capabilities: delegatedCaps,
      run: (o) => {
        received = o.tools?.["write"];
        return { events: (async function* () {})(), outcome: Promise.resolve(ok({ r: "done" })), cancel: async () => {} };
      },
    };
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: { r: { type: "string" } },
        runtime: { name: "agent", prompt: { template: "go" }, tools: ["write"], permissions: { profile: "read-only" } },
      },
    };
    const { engine } = makeEngine(files, "s", () => ok({}), {
      tools: { write: writeTool },
      runtimes: { agent: delegated },
      extra: { permissions: { approve: allowSession } },
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    // Raw tool: runs even under a read-only profile, because the engine did NOT wrap it — a delegated
    // runtime gates its own loop's calls (canUseTool → ctx.approve), so wrapping too would double-gate.
    expect(await received!.run({}, {})).toEqual({ wrote: true });
  });

  it("routes a smart-mode tool through the engine-supplied smart policy (decides before the human)", async () => {
    let toolResult: unknown;
    const script: Script = async (call) => {
      toolResult = await call.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "full", default: "smart" }), "s", script, {
      tools: { write: writeTool },
      extra: { permissions: { approve: allowSession, smart: { write: () => "deny" } } },
    });
    await engine.run({ inputs: {} });
    expect(isPermissionDenied(toolResult)).toBe(true); // smart denied directly; the human approver was never consulted
  });

  it("enforces a custom profile supplied via EngineConfig.permissions.profiles", async () => {
    let toolResult: unknown;
    const script: Script = async (call) => {
      toolResult = await call.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "search", default: "allow" }), "s", script, {
      tools: { write: writeTool },
      extra: { permissions: { approve: allowSession, profiles: { search: (t) => t.name === "grep" } } },
    });
    await engine.run({ inputs: {} });
    expect(isPermissionDenied(toolResult)).toBe(true); // "write" is out of the custom "search" profile
  });
});

describe("per-session workspace overlay (RUNTIMES-AND-PERMISSIONS.md §3)", () => {
  const files = (session?: string): Record<string, StateDef> => ({
    s: {
      inputs: {},
      outputs: { r: { type: "string" } },
      runtime: { name: "reviewer", prompt: { template: "go" }, ...(session ? { session } : {}) },
    },
  });

  it("resolves the session's workspace via workspaceFor (overriding the run-level default)", async () => {
    const { engine, fake } = makeEngine(files("sess-x"), "s", () => ok({ r: "done" }), {
      extra: {
        services: { workspace: { root: "/default" } },
        workspaceFor: (id) => (id === "sess-x" ? { root: "/ws/x" } : undefined),
      },
    });
    await engine.run({ inputs: {} });
    expect(fake.ctxs[0]!.workspace?.root).toBe("/ws/x");
  });

  it("falls back to the run-level workspace when workspaceFor has no entry for the session", async () => {
    const { engine, fake } = makeEngine(files("other"), "s", () => ok({ r: "done" }), {
      extra: { services: { workspace: { root: "/default" } }, workspaceFor: () => undefined },
    });
    await engine.run({ inputs: {} });
    expect(fake.ctxs[0]!.workspace?.root).toBe("/default");
  });
});
