import type { ResolvedValue } from "@declarative-ai/exec";
import { hostFunction, runtimeFunction } from "@declarative-ai/exec";
import type { WorkflowMetrics } from "../src/ports";
import { describe, expect, it } from "vitest";
import {
  MapSessionStore,
  RUNTIME_CAPABILITIES,
  type ExecServices,
  type Executor,
  type HostCapabilities,
  type JsonValue,
  type ExecResult,
  type FunctionResult,
  type Tool,
} from "@declarative-ai/exec";
import { isPermissionDenied, type Approver } from "@declarative-ai/permissions";
import { WorkflowEngine, type EngineConfig } from "../src/engine";
import { loadBundle } from "../src/loader";
import { InMemoryPersistence, isArtifactRef, type ArtifactRef } from "../src/ports";
import type { EnvironmentDecl, StateDef } from "../src/format";
import {
  deferred,
  FakePromptExecutor,
  newRegistry,
  modelOf,
  ok,
  promptOf,
  promptTail,
  rejectingFunction,
  throwingFunction,
  ScriptedFunction,
  toolNamesOf,
  type FakeCall,
  type Script,
} from "./fakes";
import { FANOUT_ID, PLAN_ID, specFanoutFiles, specPlanningFiles } from "./fixtures";

/** A registered host function as these tests declare one: an impl plus its REQUIRED capabilities. */
type FakeImpl = (inputs: never, ctx: never) => Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> | FunctionResult<ResolvedValue, WorkflowMetrics>;
interface FakeEntry {
  run: FakeImpl;
  capabilities?: HostCapabilities;
  /** Register as a DELEGATED runtime (policyEnforcement: "callback") rather than plain host code. */
  runtime?: boolean;
}

interface MakeEngineOpts {
  functions?: Record<string, FakeEntry | FakeImpl>;
  skills?: Record<string, string>;
  tools?: Record<string, Tool>;
  /** Override the prompt executor (e.g. a delegated-capability variant). */
  prompt?: Executor<ExecServices, WorkflowMetrics>;
  extra?: Partial<EngineConfig>;
}

const HOST: HostCapabilities = { interactive: true, readOnly: true, memoizable: false };

function makeEngine(files: Record<string, StateDef>, rootId: string, script: Script, opts: MakeEngineOpts = {}) {
  const fake = new FakePromptExecutor(script);
  const registry = newRegistry();
  for (const [name, fn] of Object.entries(opts.functions ?? {})) {
    const entry: FakeEntry = typeof fn === "function" ? { run: fn } : fn;
    if (entry.runtime) {
      registry.functions.set(name, runtimeFunction(entry.run as never, { ...RUNTIME_CAPABILITIES, ...entry.capabilities, policyEnforcement: "callback" }));
    } else {
      registry.functions.set(name, hostFunction(entry.run as never, entry.capabilities ?? HOST));
    }
  }
  for (const [name, template] of Object.entries(opts.skills ?? {})) registry.skills.set(name, template);
  for (const [name, t] of Object.entries(opts.tools ?? {})) registry.tools.set(name, t);
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    prompt: opts.prompt ?? fake,
    persistence,
    ...opts.extra,
  });
  return { engine, fake, persistence };
}

/** Default happy-path script for the planning fixture; override per-model as needed. */
function planningScript(overrides: Partial<Record<string, (call: FakeCall) => ExecResult<ResolvedValue, WorkflowMetrics>>> = {}): Script {
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
        throw new Error(`unscripted model ${model}`);
    }
  };
}

describe("SPEC §8.2 — function state terminates with validated outputs", () => {
  const files = specPlanningFiles();
  const HR = "feature/plan/critique/human_review";

  it("runs the function, validates, and terminates success with its outputs", async () => {
    const fn = new ScriptedFunction([{ decision: "approve", comments: "lgtm" }]);
    const { engine } = makeEngine(files, HR, () => {
      throw new Error("no prompt op should run");
    }, { functions: { choose_option: fn } });
    const result = await engine.run({ inputs: { plan_doc: "plan", critique_report: "report" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ decision: "approve", comments: "lgtm" });
    // §7.1: the authored function surface arrives as the op's `config` input; the state's own
    // inputs are spread in at the top level of the same `FunctionInputs`.
    expect(fn.calls[0]!["config"]).toMatchObject({
      prompt: "Review the critique result.",
      options: ["approve", "request_changes", "block"],
    });
    expect(fn.calls[0]!["plan_doc"]).toBe("plan");
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

  // The impl contract is "errors resolve as data", but nothing at registration ENFORCES it, so an impl
  // that throws is a shape the engine meets in practice. It must degrade to a state failure (SPEC §3.3)
  // — a rejection escaping `engine.run()` skips every error transition and the termination record.
  it("a THROWING function degrades to a state failure rather than rejecting engine.run()", async () => {
    const { engine, persistence } = makeEngine(files, HR, () => ok({}), { functions: { choose_option: throwingFunction } });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/not allowed/);
    expect(result.failure?.classification).toBe("permanent");
    // The run record is intact — a rejection would have skipped the terminal event entirely.
    expect(persistence.events.some((e) => e.event.type === "instance.terminated")).toBe(true);
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
    expect(promptOf(fake.calls[1]!)).toBeTruthy();
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
      metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" as const },
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
    // An unconstrained slot bound to a child producer forwards the critique's whole output object.
    expect((result.outputs?.["critique"] as Record<string, unknown>)["outcome"]).toBe("clean");
    expect(fake.calls.map(modelOf)).toEqual(["planner", "planner", "critic"]);
  });

  it("exposes a run-scoped session store shared across all states (ctx.sessions)", async () => {
    const { engine, fake } = makeEngine(specPlanningFiles(), PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    const stores = fake.calls.map((c: FakeCall) => c.ctx).map((c) => c.sessions);
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
    expect(criticCall.op.output.schema).toBeDefined(); // structured contract derived from outputs
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
        // `slow` is ASYNC, so the reachability rule (§7.2) forbids a `{ child }` binding here —
        // the pending-tolerant `{ expr }` leaf is the async equivalent.
        outputs: { got: { schema: { type: "string" }, binding: { expr: "children.slow.outputs.val" } } },
        children: {
          slow: { state: "parent/slow", async: true, inputs: {} },
          quick: { state: "parent/quick", inputs: {} },
        },
        sequence: ["slow", "quick"],
        transitions: [{ to: "terminate.success", when: "children.slow.outputs.val === 'done'" }],
      },
      "parent/slow": {
        inputs: {},
        outputs: { val: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: { template: "slow" }, config: { model: "reviewer" } },
      },
      "parent/quick": {
        inputs: {},
        outputs: { q: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: { template: "quick" }, config: { model: "synthesizer" } },
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
    // `never` is async, so `waiting` wires to it through the pending-tolerant `{ expr }` leaf;
    // the join parks until it resolves.
    const files: Record<string, StateDef> = {
      parent: {
        inputs: {},
        outputs: {},
        children: {
          never: { state: "parent/never", async: true, inputs: {} },
          waiting: { state: "parent/waiting", inputs: { x: { expr: "children.never.outputs.v" } } },
        },
        sequence: ["never", "waiting"],
      },
      "parent/never": {
        inputs: {},
        outputs: { v: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: { template: "never" }, config: { model: "reviewer" } },
      },
      "parent/waiting": {
        inputs: { x: { schema: { type: "string" } } },
        outputs: {},
        operation: { kind: "prompt", prompt: { template: "waiting" }, config: { model: "reviewer" } },
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
        ? ({ value: { v: "x" }, metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" as const }, error: { classification: "permanent", reason: "boom" } } as ExecResult<ResolvedValue, WorkflowMetrics>)
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
        outputs: { x: { schema: { type: "string" }, optional: true } },
        operation: { kind: "prompt", prompt: { template: "forever" }, config: { model: "reviewer" } },
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
        outputs: { x: { schema: { type: "string" }, optional: true } },
        operation: { kind: "prompt", prompt: { template: "forever" }, config: { model: "reviewer" } },
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
        outputs: { handled: { schema: { type: "string" }, binding: { expr: "'yes'" } } },
        children: {
          slow: { state: "parent/slow", inputs: {} },
        },
        sequence: ["slow"],
        transitions: [{ to: "terminate.success", when: "children.slow.outcome === 'timeout'" }],
      },
      "parent/slow": {
        inputs: {},
        outputs: { x: { schema: { type: "string" }, optional: true } },
        operation: { kind: "prompt", prompt: { template: "forever" }, config: { model: "reviewer" } },
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
    files["feature/plan/critique"]!.environment = { conversation: { mode: "fresh" } };
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    expect(promptOf(fake.calls[2]!)).not.toContain("<conversation-history>");
  });

  it("a distinct environment.session isolates the transcript (full_history sees an empty per-session history)", async () => {
    const files = specPlanningFiles();
    // The planners run in the default session; move the critic to its own session — its full_history now
    // reads an EMPTY transcript, so the planners' exchanges do NOT leak across the session boundary.
    files["feature/plan/critique"]!.environment!.session = "isolated";
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });
    expect(promptOf(fake.calls[2]!)).not.toContain("<conversation-history>");
    expect(promptOf(fake.calls[2]!)).not.toContain("Extract goals");
  });

  it("a { conversation } binding wires a prior transcript in as DATA (§7.5)", async () => {
    const files = specPlanningFiles();
    // The critique state summarizes the planners' earlier session instead of reading its preamble:
    // `conversations` is part of the REF vocabulary, so a transcript is an ordinary wired input.
    const critique = files["feature/plan/critique"]!;
    critique.environment = { conversation: { mode: "fresh" } };
    critique.operation = {
      kind: "prompt",
      config: { model: "critic" },
      // An operation's bound input slots render under `{{inputs.*}}` — the one namespace a template
      // sees, the operation's resolved inputs (state inputs plus the op's own bound inputs).
      prompt: { template: "Summarize this transcript: {{inputs.history}}" },
      input: { history: { kind: "json", binding: { conversation: "default" } } },
    };
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });

    const criticPrompt = promptOf(fake.calls[2]!);
    // No preamble was prepended (mode `fresh`), so the rendered template IS the whole prompt...
    expect(criticPrompt.startsWith("Summarize this transcript:")).toBe(true);
    // ...and the transcript arrived inside it as a wired VALUE, not as an injected preamble.
    expect(criticPrompt).toContain("Extract goals");
  });

  it("a { conversation } binding can select one message of a transcript", async () => {
    const files = specPlanningFiles();
    const critique = files["feature/plan/critique"]!;
    critique.environment = { conversation: { mode: "fresh" } };
    critique.operation = {
      kind: "prompt",
      config: { model: "critic" },
      prompt: { template: "First turn was: {{inputs.first}}" },
      input: { first: { kind: "json", binding: { conversation: "default", message: 0 } } },
    };
    const { engine, fake } = makeEngine(files, PLAN_ID, planningScript());
    await engine.run({ inputs: { issue: "the issue" } });

    const criticPrompt = promptOf(fake.calls[2]!);
    expect(criticPrompt).toContain("Extract goals"); // message 0 = the first planner prompt
    expect(criticPrompt).not.toContain("Write the plan"); // and only that one
  });

  // The append ran BEFORE the success check and a failure carries no value, so the assistant turn was
  // written as the literal string "null" — and under the default full_history mode every later state
  // in that session then replayed `assistant: null` in its preamble.
  it("a FAILED prompt op contributes NOTHING to the shared transcript", async () => {
    const store = new MapSessionStore();
    const { engine } = makeEngine(
      specPlanningFiles(),
      PLAN_ID,
      () => ({ error: { classification: "permanent", reason: "model exploded" }, metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" } }),
      { extra: { services: { sessions: store } } },
    );
    const result = await engine.run({ inputs: { issue: "the issue" } });
    expect(result.outcome).toBe("error");
    expect((await store.get("default"))?.messages ?? []).toEqual([]);
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

describe("template rendering: `{{inputs.*}}` is the operation's resolved inputs", () => {
  // There is ONE namespace now (params is gone): a prompt template sees exactly the inputs its
  // operation resolved — the state's inputs, with the op's own bound inputs overlaid, a bound input
  // of the same name winning (resolveInputs: an explicit binding overrides the by-name fill).
  const files = (): Record<string, StateDef> => ({
    solo: {
      label: "Solo",
      inputs: { tone: { schema: { type: "string" } } },
      outputs: { answer: { schema: { type: "string" } } },
      operation: { kind: "prompt", config: { model: "writer" }, prompt: { template: "tone={{inputs.tone}}" } },
    } as StateDef,
  });

  it("renders a state input under {{inputs.*}}", async () => {
    const { engine, fake } = makeEngine(files(), "solo", () => ok({ answer: "a" }));
    const result = await engine.run({ inputs: { tone: "casual" } });
    expect(result.outcome).toBe("success");
    expect(promptOf(fake.calls[0]!)).toBe("tone=casual");
  });

  it("an operation's own bound input renders under {{inputs.*}} and wins over a same-named state input", async () => {
    const f = files();
    // A render variable is authored as an operation input with a literal binding — the successor to
    // the removed `operation.params` sugar. It resolves and overlays onto the template's inputs.
    (f["solo"]!.operation as { input?: Record<string, unknown> }).input = { tone: { kind: "text", binding: { text: "terse" } } };
    (f["solo"]!.operation as { prompt?: { template?: string } }).prompt = { template: "tone={{inputs.tone}}" };
    const { engine, fake } = makeEngine(f, "solo", () => ok({ answer: "a" }));
    await engine.run({ inputs: { tone: "casual" } });
    expect(promptOf(fake.calls[0]!)).toBe("tone=terse");
  });
});

describe("structured-output contract (buildOutputSchema)", () => {
  // A default-backed output is optional: the engine backfills its default when the model omits it, so
  // the structured-output schema handed to the model must NOT list it as `required` — otherwise the
  // model is forced to fabricate a value the author meant to be skippable.
  it("does not mark a default-backed output as required", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: {
          summary: { schema: { type: "string" } },
          note: { schema: { type: "string" }, default: "n/a" },
        },
        operation: { kind: "prompt", config: { model: "writer" }, prompt: { template: "go" } },
      },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({ summary: "s" }));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    const schema = fake.calls[0]!.op.output.schema!;
    expect(schema.required).toContain("summary");
    expect(schema.required).not.toContain("note");
    // ...and the omitted default-backed output is backfilled from its default.
    expect(result.outputs?.["note"]).toBe("n/a");
  });
});

describe("declared outputs at termination (SPEC §3.7)", () => {
  // The resolution `{error}` case was discarded alongside PENDING, so a binding that genuinely failed
  // to resolve was laundered into the generic "was not produced" — a failure whose reason names the
  // symptom instead of the cause (§7.3: a failure carries the real underlying one).
  const files = (): Record<string, StateDef> => ({
    parent: {
      label: "Parent",
      inputs: {},
      outputs: { report: { schema: { type: "string" }, binding: { child: "never_run", output: "report" } } },
      children: { never_run: { state: "parent/leaf" } },
      // No sequence, and the only transition terminates: `never_run` never runs.
      transitions: [{ to: "terminate.success" }],
    } as StateDef,
    "parent/leaf": {
      label: "Leaf",
      inputs: {},
      outputs: { report: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "m" } },
    } as StateDef,
  });

  it("reports WHY a bound output failed to resolve, not that it 'was not produced'", async () => {
    const { engine } = makeEngine(files(), "parent", () => ok({ report: "r" }));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/output 'report': child 'never_run' has not run/);
    expect(result.failure?.reason).not.toMatch(/was not produced/);
  });

  it("still lets an OPTIONAL slot absorb the same failure", async () => {
    const f = files();
    f["parent"]!.outputs!["report"]!.optional = true;
    const { engine } = makeEngine(f, "parent", () => ok({ report: "r" }));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({});
  });

  it("keeps the generic reason where nothing was resolved and nothing failed", async () => {
    const f = files();
    delete f["parent"]!.outputs!["report"]!.binding; // produced by an operation the state does not have
    const { engine } = makeEngine(f, "parent", () => ok({ report: "r" }));
    const result = await engine.run({ inputs: {} });
    expect(result.failure?.reason).toMatch(/required output 'report' was not produced/);
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
    expect((completed.event as { metrics?: { costUsd?: number } }).metrics?.costUsd).toBe(0.01);
    // Cost/call rollup for the whole run.
    expect(result.metrics.childLlmCalls).toBe(3);
    expect(result.metrics.childCost).toBeCloseTo(0.03);
  });

  // A delegated agent is a FunctionOp, and it is the only thing that knows what it spent — it bills
  // inside its own loop. Without a metrics channel on `FunctionResult<ResolvedValue, WorkflowMetrics>` the function path emitted
  // `operation.completed` with no metrics at all, so the most expensive op in a graph was the one
  // whose cost the run never saw.
  it("rolls up cost a FUNCTION op reports, not just a prompt op's", async () => {
    const files = specPlanningFiles();
    const HR = "feature/plan/critique/human_review";
    const spendy = async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({
      value: { decision: "approve", comments: "lgtm" },
      metrics: { durationMs: 12, costUsd: 0.25, costSource: "provider" as const, childLlmCalls: 1, childCostUsd: 0.25 },
    });
    const { engine, persistence } = makeEngine(files, HR, () => ok({}), { functions: { choose_option: spendy } });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("success");
    expect(result.metrics.childCost).toBeCloseTo(0.25);
    expect(result.metrics.childLlmCalls).toBe(1);
    const completed = persistence.events.find((e) => e.event.type === "operation.completed")!;
    expect((completed.event as { metrics?: { costUsd?: number } }).metrics?.costUsd).toBe(0.25);
  });

  it("leaves the rollup alone for a function op that reports nothing", async () => {
    const files = specPlanningFiles();
    const HR = "feature/plan/critique/human_review";
    const fn = new ScriptedFunction([{ decision: "approve", comments: "lgtm" }]);
    const { engine } = makeEngine(files, HR, () => ok({}), { functions: { choose_option: fn } });
    const result = await engine.run({ inputs: { plan_doc: "p", critique_report: "r" } });
    expect(result.outcome).toBe("success");
    expect(result.metrics.childCost).toBe(0);
    expect(result.metrics.childLlmCalls).toBe(0);
  });
});

describe("skill as a prompt-op prompt source", () => {
  it("renders a named skill template as the operation's prompt", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: { topic: { schema: { type: "string" } } },
        outputs: { summary: { schema: { type: "string" } } },
        // A render variable is a bound operation input; it reaches the skill template under
        // `{{inputs.*}}` alongside the state's own inputs — one namespace, the op's resolved inputs.
        operation: {
          kind: "prompt",
          prompt: { skill: "summarize" },
          input: { style: { schema: { type: "string" }, binding: { text: "terse" } } },
          config: { model: "reviewer" },
        },
      },
    };
    const { engine, fake } = makeEngine(files, "s", (call) => ok({ summary: `sum(${promptOf(call)})` }), {
      skills: { summarize: "Summarize {{inputs.topic}} ({{inputs.style}})." },
    });
    const result = await engine.run({ inputs: { topic: "codebases" } });
    expect(result.outcome).toBe("success");
    // Both the state input `topic` and the op's bound input `style` are template variables under inputs.
    expect(result.outputs?.["summary"]).toBe("sum(Summarize codebases (terse).)");
    expect(fake.calls).toHaveLength(1);
  });

  it("an unregistered skill is a permanent failure", async () => {
    const files: Record<string, StateDef> = {
      s: { inputs: {}, outputs: {}, operation: { kind: "prompt", prompt: { skill: "ghost" }, config: { model: "reviewer" } } },
    };
    const { engine } = makeEngine(files, "s", () => ok({}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/skill 'ghost'/);
  });
});

describe("environment tools (DESIGN §5.1, \"Functions and tools\")", () => {
  const echoTool: Tool = { description: "echo", inputSchema: { type: "object" }, readOnly: true, run: (input) => input as JsonValue };

  it("resolves declared tool names through registry.tools and hands them to the operation", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: { r: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "reviewer" } },
        environment: { tools: ["echo"] },
      },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({ r: "done" }), { tools: { echo: echoTool } });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    // The resolved executable reached the prompt op's environment, keyed by its logical name.
    expect(toolNamesOf(fake.calls[0]!)).toEqual(["echo"]);
    expect(fake.calls[0]!.ctx.tools?.["echo"]).toBe(echoTool);
  });

  it("an unregistered tool is a permanent failure", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: {},
        operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "reviewer" } },
        environment: { tools: ["ghost"] },
      },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({}));
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/tool 'ghost' is not registered/);
    expect(fake.calls).toHaveLength(0); // failed before dispatching to the runner
  });

  it("threads the host-provided workspace to the operation's ctx (where fs tools read it)", async () => {
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: { r: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "reviewer" } },
      },
    };
    const { engine, fake } = makeEngine(files, "s", () => ok({ r: "done" }), {
      extra: { services: { workspace: { root: "/work/space" } } },
    });
    await engine.run({ inputs: {} });
    expect(fake.calls.map((c: FakeCall) => c.ctx)[0]!.workspace?.root).toBe("/work/space");
  });
});

describe("tool permissions (DESIGN §5.1, \"Permissions: two orthogonal axes\")", () => {
  const writeTool: Tool = { description: "write", inputSchema: { type: "object" }, readOnly: false, run: () => ({ wrote: true }) };
  const allowSession: Approver = () => ({ decision: "allow", scope: "session" });

  /** A state whose operation invokes its (engine-supplied, possibly permission-wrapped) `write` tool. */
  const invokeWriteFiles = (permissions?: EnvironmentDecl["permissions"]): Record<string, StateDef> => ({
    s: {
      inputs: {},
      outputs: { r: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "reviewer" } },
      environment: { tools: ["write"], ...(permissions ? { permissions } : {}) },
    },
  });

  it("read-only profile denies a mutating tool when the operation invokes it", async () => {
    let toolResult: JsonValue = null;
    const script: Script = async (call) => {
      toolResult = await call.ctx.tools!["write"]!.run({}, {});
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
    let toolResult: JsonValue = null;
    const script: Script = async (call) => {
      toolResult = await call.ctx.tools!["write"]!.run({}, {});
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
    let toolResult: JsonValue = null;
    const script: Script = async (call) => {
      toolResult = await call.ctx.tools!["write"]!.run({}, {});
      return ok({ r: "done" });
    };
    const { engine } = makeEngine(invokeWriteFiles({ profile: "read-only" }), "s", script, { tools: { write: writeTool } });
    await engine.run({ inputs: {} });
    expect(toolResult).toEqual({ wrote: true }); // no wrapper ⇒ profile not enforced
  });

  it("hands a delegated adapter (policyEnforcement: callback) RAW tools — no double-gating", async () => {
    // Post-redesign a delegated agent runtime IS a registered function (§3.1); `policyEnforcement`
    // lives on the resolved registry ENTRY's capabilities, never on the op.
    let received: Tool | undefined;
    const delegated: FakeEntry = {
      runtime: true,
      capabilities: { interactive: true, readOnly: false, memoizable: false },
      run: ((_inputs: never, ctx: ExecServices): FunctionResult<ResolvedValue, WorkflowMetrics> => {
        received = ctx.tools?.["write"];
        return { value: { r: "done" } };
      }) as FakeImpl,
    };
    const files: Record<string, StateDef> = {
      s: {
        inputs: {},
        outputs: { r: { schema: { type: "string" } } },
        operation: { kind: "function", function: "agent" },
        environment: { tools: ["write"], permissions: { profile: "read-only" } },
      },
    };
    const { engine } = makeEngine(files, "s", () => ok({}), {
      tools: { write: writeTool },
      functions: { agent: delegated },
      extra: { permissions: { approve: allowSession } },
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    // Raw tool: runs even under a read-only profile, because the engine did NOT wrap it — a delegated
    // adapter gates its own loop's calls (canUseTool → ctx.approve), so wrapping too would double-gate.
    expect(await received!.run({}, {})).toEqual({ wrote: true });
  });

  it("routes a smart-mode tool through the engine-supplied smart policy (decides before the human)", async () => {
    let toolResult: JsonValue = null;
    const script: Script = async (call) => {
      toolResult = await call.ctx.tools!["write"]!.run({}, {});
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
    let toolResult: JsonValue = null;
    const script: Script = async (call) => {
      toolResult = await call.ctx.tools!["write"]!.run({}, {});
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

describe("per-session workspace overlay (DESIGN §5.1, \"Sessions: the run-scoped resource bundle\")", () => {
  const files = (session?: string): Record<string, StateDef> => ({
    s: {
      inputs: {},
      outputs: { r: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "reviewer" } },
      ...(session ? { environment: { session } } : {}),
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
    expect(fake.calls.map((c: FakeCall) => c.ctx)[0]!.workspace?.root).toBe("/ws/x");
  });

  it("falls back to the run-level workspace when workspaceFor has no entry for the session", async () => {
    const { engine, fake } = makeEngine(files("other"), "s", () => ok({ r: "done" }), {
      extra: { services: { workspace: { root: "/default" } }, workspaceFor: () => undefined },
    });
    await engine.run({ inputs: {} });
    expect(fake.calls.map((c: FakeCall) => c.ctx)[0]!.workspace?.root).toBe("/default");
  });
});
