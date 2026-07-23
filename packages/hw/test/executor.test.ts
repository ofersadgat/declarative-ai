import { hostFunction } from "@declarative-ai/exec";
import type { ResolvedValue, FunctionResult } from "@declarative-ai/exec";
import type { WorkflowMetrics } from "../src/ports";
import { describe, expect, it } from "vitest";
import {
  hashOperation,
  memoKey,
  type ExecEvent,
  type ExecServices,
  type InlineFamily,
  type JsonSchema,
  type JsonValue,
  type CapabilityRegistry,
  type Operation,
} from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { createWorkflowExecutor, workflowMemoKey, type HierarchicalWorkflowDefinition } from "../src/executor";
import { loadBundle, snapshotHash } from "../src/loader";
import { FakePromptExecutor, INTERACTIVE, modelOf, newRegistry, ok, promptTail, ScriptedFunction, type Script, errorOf } from "./fakes";
import { PLAN_ID, specPlanningFiles } from "./fixtures";

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

/** Build a capability registry + the prompt executor the engine dispatches `PromptOp`s to. */
function makeRegistry(
  script: Script,
  functions: Record<string, ScriptedFunction> = {},
): { registry: CapabilityRegistry<WorkflowMetrics>; fake: FakePromptExecutor } {
  const fake = new FakePromptExecutor(script);
  const registry = newRegistry();
  for (const [name, fn] of Object.entries(functions)) registry.functions.set(name, hostFunction(fn.run, INTERACTIVE));
  return { registry, fake };
}

/**
 * A workflow run is started by a FUNCTION OP whose bound inputs are the workflow's inputs — there is no
 * `ExecutionSpec` any more, and the bundle is held by the executor rather than re-supplied per run
 * (DESIGN §7).
 */
function opFor(inputs: Record<string, JsonValue>, outputSchema?: JsonSchema): Operation<InlineFamily> {
  return {
    kind: "function",
    functionRef: "planning-workflow",
    input: Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, { kind: "json" as const, binding: { json: v } }])),
    output: { name: "output", kind: "json", ...(outputSchema !== undefined ? { schema: outputSchema } : {}) },
  };
}

describe("hierarchical-workflow executor", () => {
  it("runs the SPEC §9 planning workflow end-to-end and rolls up metrics", async () => {
    const { registry, fake } = makeRegistry(happyScript);
    const executor = createWorkflowExecutor({ definition: planningDefinition(), registry, prompt: fake });
    const handle = executor.start(opFor({ issue: "the issue" }), CTX);

    const seen: ExecEvent[] = [];
    const consume = (async () => {
      for await (const e of handle.events) seen.push(e);
    })();
    const outcome = await handle.result;
    await consume;

    expect(errorOf(outcome)).toBeUndefined();
    const value = outcome.value as Record<string, unknown>;
    expect(value["outcome"]).toBe("complete");
    // No `artifacts` side channel any more (§7.1): a produced artifact is an output SLOT.
    expect(JSON.stringify(outcome.value)).toContain("# The Plan");
    expect(outcome.metrics.childLlmCalls).toBe(3);
    expect(outcome.metrics.childCostUsd).toBeCloseTo(0.03);
    expect(outcome.metrics.costUsd).toBeCloseTo(0.03); // composite cost includes children
    expect(fake.calls).toHaveLength(3);
    // Event stream: progress + one child_outcome per completed operation.
    expect(seen.filter((e) => e.type === "child_result")).toHaveLength(3);
    expect(seen.some((e) => e.type === "progress" && e.message.includes("entered feature/plan"))).toBe(true);
  });

  it("rejects an invalid bundle with the validation report", async () => {
    const definition = planningDefinition();
    (definition.states as ReturnType<typeof specPlanningFiles>)[PLAN_ID]!.transitions!.push({ to: "nowhere" });
    const { registry, fake } = makeRegistry(happyScript);
    const executor = createWorkflowExecutor({ definition, registry, prompt: fake });
    const outcome = await executor.start(opFor({ issue: "i" }), CTX).result;
    expect(errorOf(outcome)?.classification).toBe("permanent");
    expect(errorOf(outcome)?.reason).toMatch(/validation failed/);
    expect(errorOf(outcome)?.reason).toMatch(/nowhere/);
    expect(fake.calls).toHaveLength(0);
  });

  it("no interaction policy: an interactive definition runs when the gate isn't reached, fails when it is reached unregistered", async () => {
    // The clean path never reaches human_review → succeeds even with no interactive function registered.
    // Whether/how interaction flows is the designer's composition, not an executor mode.
    const { registry: unreachedReg, fake: unreachedFake } = makeRegistry(happyScript);
    const unreached = createWorkflowExecutor({ definition: planningDefinition(), registry: unreachedReg, prompt: unreachedFake });
    const unreachedOutcome = await unreached.start(opFor({ issue: "i" }), CTX).result;
    expect(errorOf(unreachedOutcome)).toBeUndefined();

    // A run that DOES reach the function state with the function unregistered → permanent failure at that
    // state (how a search context refuses a human gate: by not registering it — no policy knob needed).
    const blockedScript: Script = (call) =>
      modelOf(call) === "critic" ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" }) : happyScript(call);
    const { registry: reachedReg, fake: reachedFake } = makeRegistry(blockedScript);
    const reached = createWorkflowExecutor({ definition: planningDefinition(), registry: reachedReg, prompt: reachedFake });
    const reachedOutcome = await reached.start(opFor({ issue: "i" }), CTX).result;
    expect(errorOf(reachedOutcome)?.classification).toBe("permanent");
    expect(errorOf(reachedOutcome)?.reason).toMatch(/function 'choose_option'/);
  });

  it("interactive path works when the function is registered (blocked critique → human decision)", async () => {
    const script: Script = (call) =>
      modelOf(call) === "critic" ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" }) : happyScript(call);
    const { registry, fake } = makeRegistry(script, { choose_option: new ScriptedFunction([{ decision: "block" }]) });
    const executor = createWorkflowExecutor({ definition: planningDefinition(), registry, prompt: fake });
    const outcome = await executor.start(opFor({ issue: "i" }), CTX).result;
    expect(errorOf(outcome)).toBeUndefined();
    const value = outcome.value as Record<string, unknown>;
    expect(value["outcome"]).toBe("blocked");
    expect((value["critique"] as Record<string, unknown>)["human_decision"]).toBe("block");
  });

  it("cancel() yields a canceled outcome; spec timeout yields deadline", async () => {
    const hanging: Script = () => new Promise(() => {});
    {
      const { registry, fake } = makeRegistry(hanging);
      const executor = createWorkflowExecutor({ definition: planningDefinition(), registry, prompt: fake });
      const handle = executor.start(opFor({ issue: "i" }), CTX);
      setTimeout(() => void handle.cancel(), 20);
      const outcome = await handle.result;
      expect(errorOf(outcome)?.classification).toBe("canceled");
    }
    {
      const { registry, fake } = makeRegistry(hanging);
      const executor = createWorkflowExecutor({ definition: planningDefinition(), registry, prompt: fake });
      const handle = executor.start(opFor({ issue: "i" }), { ...CTX, timeoutMs: 40 });
      const outcome = await handle.result;
      expect(errorOf(outcome)?.classification).toBe("deadline");
    }
  });

  it("enforces the OPERATION's output contract on top of per-state validation", async () => {
    const { registry, fake } = makeRegistry(happyScript);
    const executor = createWorkflowExecutor({ definition: planningDefinition(), registry, prompt: fake });
    // The contract lives on the op's OUTPUT SLOT — the same place every other op declares it, which is
    // what let `spec.outputSchema` go away (§5.1).
    const op = opFor({ issue: "i" }, {
      type: "object",
      properties: { outcome: { type: "string", enum: ["blocked"] } }, // demands blocked; the run yields complete
      required: ["outcome"],
    });
    const outcome = await executor.start(op, CTX).result;
    expect(errorOf(outcome)?.classification).toBe("api-retriable");
    expect(errorOf(outcome)?.reason).toMatch(/operation's contract/);
    expect((outcome.value as Record<string, unknown>)["outcome"]).toBe("complete"); // value preserved on failure
  });

  it("workflowMemoKey folds the snapshot hash with the operation hash", () => {
    const definition = planningDefinition();
    const op = opFor({ issue: "i" });
    const snapshot = snapshotHash(loadBundle(specPlanningFiles(), PLAN_ID));
    expect(workflowMemoKey(definition, op)).toBe(memoKey({ operationHash: `${snapshot}:${hashOperation(op)}` }));
    // Same workflow version + same inputs ⇒ same key; any change ⇒ new key.
    expect(workflowMemoKey(definition, opFor({ issue: "i" }))).toBe(workflowMemoKey(definition, op));
    expect(workflowMemoKey(definition, opFor({ issue: "j" }))).not.toBe(workflowMemoKey(definition, op));
  });
});
