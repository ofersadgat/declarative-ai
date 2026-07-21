import { describe, expect, it } from "vitest";
import { MapCapabilityRegistry, memoKey, type ExecEvent, type ExecServices, type ExecutionSpec, type HostFunction } from "@declarative-ai/core";
import { SchemaValidator } from "@declarative-ai/services";
import {
  createHierarchicalWorkflowExecutor,
  workflowMemoKey,
  type HierarchicalWorkflowDefinition,
} from "../src/executor";
import { loadBundle, snapshotHash } from "../src/loader";
import { FakeRuntimes, modelOf, ok, promptTail, ScriptedFunction, type Script } from "./fakes";
import { PLAN_ID, specPlanningFiles } from "./fixtures";

const RUNTIME_NAMES = ["planner", "critic", "fixer"];
const CTX: ExecServices = { validator: new SchemaValidator() };

function planningDefinition(): HierarchicalWorkflowDefinition {
  return { rootId: PLAN_ID, states: specPlanningFiles() };
}

const happyScript: Script = (call) => {
  switch (modelOf(call)) {
    case "planner":
      return promptTail(call).startsWith("Write the plan") ? ok({ plan_doc: "# The Plan" }) : ok({ goals: ["g1"] });
    case "critic":
      return ok({ outcome: "clean", weaknesses: [], critique_report: "no issues" });
    default:
      return ok({ resolution: "fixed" });
  }
};

/** Build a capability registry (runtimes + any interactive/host functions) for the executor options. */
function makeRegistry(script: Script, functions: Record<string, HostFunction> = {}): { registry: MapCapabilityRegistry; fake: FakeRuntimes } {
  const fake = new FakeRuntimes(script);
  const registry = fake.register(new MapCapabilityRegistry(), RUNTIME_NAMES);
  for (const [name, fn] of Object.entries(functions)) registry.functions.register(name, fn);
  return { registry, fake };
}

function specFor(inputs: Record<string, unknown>, overrides?: Partial<ExecutionSpec>): ExecutionSpec {
  return { kind: "hierarchical-workflow", definition: planningDefinition(), inputs, ...overrides };
}

describe("hierarchical-workflow executor", () => {
  it("runs the SPEC §9 planning workflow end-to-end and rolls up metrics", async () => {
    const { registry, fake } = makeRegistry(happyScript);
    const executor = createHierarchicalWorkflowExecutor({ registry });
    const handle = executor.start(specFor({ issue: "the issue" }), CTX);

    const seen: ExecEvent[] = [];
    const consume = (async () => {
      for await (const e of handle.events) seen.push(e);
    })();
    const outcome = await handle.outcome;
    await consume;

    expect(outcome.error).toBeUndefined();
    const value = outcome.value as Record<string, unknown>;
    expect(value["outcome"]).toBe("complete");
    expect(outcome.artifacts?.some((a) => a.content === "# The Plan")).toBe(true);
    expect(outcome.metrics.childCalls).toBe(3);
    expect(outcome.metrics.childCost).toBeCloseTo(0.03);
    expect(outcome.metrics.cost).toBeCloseTo(0.03); // composite cost includes children
    expect(fake.calls).toHaveLength(3);
    // Event stream: progress + one child_outcome per completed operation.
    expect(seen.filter((e) => e.type === "child_outcome")).toHaveLength(3);
    expect(seen.some((e) => e.type === "progress" && e.message.includes("entered feature/plan"))).toBe(true);
  });

  it("rejects an invalid bundle with the validation report", async () => {
    const definition = planningDefinition();
    (definition.states as ReturnType<typeof specPlanningFiles>)[PLAN_ID]!.transitions!.push({ to: "nowhere" });
    const { registry, fake } = makeRegistry(happyScript);
    const executor = createHierarchicalWorkflowExecutor({ registry });
    const outcome = await executor.start({ kind: "hierarchical-workflow", definition, inputs: { issue: "i" } }, CTX).outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/validation failed/);
    expect(outcome.error?.reason).toMatch(/nowhere/);
    expect(fake.calls).toHaveLength(0);
  });

  it("no interaction policy: an interactive definition runs when the gate isn't reached, fails when it is reached unregistered", async () => {
    // The clean path never reaches human_review → succeeds even with no interactive function registered.
    // Whether/how interaction flows is the designer's composition, not an executor mode.
    const { registry: unreachedReg } = makeRegistry(happyScript);
    const unreached = createHierarchicalWorkflowExecutor({ registry: unreachedReg });
    const unreachedOutcome = await unreached.start(specFor({ issue: "i" }), CTX).outcome;
    expect(unreachedOutcome.error).toBeUndefined();

    // A run that DOES reach the function state with the function unregistered → permanent failure at that
    // state (how a search context refuses a human gate: by not registering it — no policy knob needed).
    const blockedScript: Script = (call) =>
      modelOf(call) === "critic" ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" }) : happyScript(call);
    const { registry: reachedReg } = makeRegistry(blockedScript);
    const reached = createHierarchicalWorkflowExecutor({ registry: reachedReg });
    const reachedOutcome = await reached.start(specFor({ issue: "i" }), CTX).outcome;
    expect(reachedOutcome.error?.classification).toBe("permanent");
    expect(reachedOutcome.error?.reason).toMatch(/function 'choose_option'/);
  });

  it("interactive path works when the function is registered (blocked critique → human decision)", async () => {
    const script: Script = (call) =>
      modelOf(call) === "critic" ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" }) : happyScript(call);
    const { registry } = makeRegistry(script, { choose_option: new ScriptedFunction([{ decision: "block" }]) });
    const executor = createHierarchicalWorkflowExecutor({ registry });
    const outcome = await executor.start(specFor({ issue: "i" }), CTX).outcome;
    expect(outcome.error).toBeUndefined();
    const value = outcome.value as Record<string, unknown>;
    expect(value["outcome"]).toBe("blocked");
    expect((value["critique"] as Record<string, unknown>)["human_decision"]).toBe("block");
  });

  it("cancel() yields a canceled outcome; spec timeout yields deadline", async () => {
    const hanging: Script = () => new Promise(() => {});
    {
      const { registry } = makeRegistry(hanging);
      const executor = createHierarchicalWorkflowExecutor({ registry });
      const handle = executor.start(specFor({ issue: "i" }), CTX);
      setTimeout(() => void handle.cancel(), 20);
      const outcome = await handle.outcome;
      expect(outcome.error?.classification).toBe("canceled");
    }
    {
      const { registry } = makeRegistry(hanging);
      const executor = createHierarchicalWorkflowExecutor({ registry });
      const handle = executor.start(specFor({ issue: "i" }, { limits: { timeoutMs: 40 } }), CTX);
      const outcome = await handle.outcome;
      expect(outcome.error?.classification).toBe("deadline");
    }
  });

  it("enforces a spec-level output contract on top of per-state validation", async () => {
    const { registry } = makeRegistry(happyScript);
    const executor = createHierarchicalWorkflowExecutor({ registry });
    const outcome = await executor
      .start(
        specFor(
          { issue: "i" },
          {
            outputSchema: {
              type: "object",
              properties: { outcome: { type: "string", enum: ["blocked"] } }, // demands blocked; run yields complete
              required: ["outcome"],
            },
          },
        ),
        CTX,
      )
      .outcome;
    expect(outcome.error?.classification).toBe("api-retriable");
    expect(outcome.error?.reason).toMatch(/spec contract/);
    expect((outcome.value as Record<string, unknown>)["outcome"]).toBe("complete"); // value preserved on failure
  });

  it("workflowMemoKey matches core memoKey over (kind, snapshot hash, inputs)", () => {
    const spec = specFor({ issue: "i" });
    const definitionHash = snapshotHash(loadBundle(specPlanningFiles(), PLAN_ID));
    expect(workflowMemoKey(spec)).toBe(memoKey({ kind: "hierarchical-workflow", definitionHash, inputs: spec.inputs }));
    // Same workflow version + same inputs ⇒ same key; any change ⇒ new key.
    expect(workflowMemoKey(specFor({ issue: "i" }))).toBe(workflowMemoKey(spec));
    expect(workflowMemoKey(specFor({ issue: "j" }))).not.toBe(workflowMemoKey(spec));
  });
});
