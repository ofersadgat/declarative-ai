import { describe, expect, it } from "vitest";
import { MapExecutorRegistry, memoKey, type ExecEvent, type ExecutionSpec, type ExecServices } from "@declarative-ai/core";
import { SchemaValidator } from "@declarative-ai/services";
import {
  createHierarchicalWorkflowExecutor,
  workflowMemoKey,
  type HierarchicalWorkflowDefinition,
} from "../src/executor";
import { loadBundle, snapshotHash } from "../src/loader";
import { llmCallBinding } from "../src/ports";
import { FakeExecutor, modelOf, ok, promptTail, ScriptedPort, type Script } from "./fakes";
import { PLAN_ID, specPlanningFiles } from "./fixtures";

const PROVIDERS = {
  planner: llmCallBinding({ model: "planner" }),
  critic: llmCallBinding({ model: "critic" }),
  fixer: llmCallBinding({ model: "fixer" }),
};

function planningDefinition(): { definition: HierarchicalWorkflowDefinition } {
  const states = specPlanningFiles();
  const definition = { rootId: PLAN_ID, states };
  return { definition };
}

const happyScript: Script = (spec) => {
  switch (modelOf(spec)) {
    case "planner":
      return promptTail(spec).startsWith("Write the plan") ? ok({ plan_doc: "# The Plan" }) : ok({ goals: ["g1"] });
    case "critic":
      return ok({ outcome: "clean", weaknesses: [], critique_report: "no issues" });
    default:
      return ok({ resolution: "fixed" });
  }
};

function makeCtx(script: Script): { ctx: ExecServices; fake: FakeExecutor } {
  const fake = new FakeExecutor(script);
  return {
    ctx: { registry: new MapExecutorRegistry().register(fake), validator: new SchemaValidator() },
    fake,
  };
}

function specFor(inputs: Record<string, unknown>, overrides?: Partial<ExecutionSpec>): ExecutionSpec {
  const { definition } = planningDefinition();
  return { kind: "hierarchical-workflow", definition, inputs, ...overrides };
}

describe("hierarchical-workflow executor", () => {
  it("runs the SPEC §9 planning workflow end-to-end and rolls up metrics", async () => {
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const { ctx, fake } = makeCtx(happyScript);
    const handle = executor.start(specFor({ issue: "the issue" }), ctx);

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
    const { definition } = planningDefinition();
    const states = definition.states as ReturnType<typeof specPlanningFiles>;
    states[PLAN_ID]!.transitions!.push({ to: "nowhere" });
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const { ctx, fake } = makeCtx(happyScript);
    const outcome = await executor
      .start({ kind: "hierarchical-workflow", definition, inputs: { issue: "i" } }, ctx)
      .outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/validation failed/);
    expect(outcome.error?.reason).toMatch(/nowhere/);
    expect(fake.calls).toHaveLength(0);
  });

  it("interaction policy: eager refuses interactive definitions up front; lazy runs until a ui state is reached", async () => {
    // Eager (search context): the planning bundle contains human_review → refuse before spending.
    const eager = createHierarchicalWorkflowExecutor({ providers: PROVIDERS, interactionPolicy: "eager" });
    const { ctx: eagerCtx, fake: eagerFake } = makeCtx(happyScript);
    const eagerOutcome = await eager.start(specFor({ issue: "i" }), eagerCtx).outcome;
    expect(eagerOutcome.error?.classification).toBe("permanent");
    expect(eagerOutcome.error?.reason).toMatch(/interactive states/);
    expect(eagerFake.calls).toHaveLength(0);

    // Lazy (default): the clean path never reaches human_review → succeeds without a port.
    const lazy = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const { ctx: lazyCtx } = makeCtx(happyScript);
    const lazyOutcome = await lazy.start(specFor({ issue: "i" }), lazyCtx).outcome;
    expect(lazyOutcome.error).toBeUndefined();

    // Lazy + a run that DOES reach the ui state → permanent failure at that state.
    const blockedScript: Script = (spec) =>
      modelOf(spec) === "critic" ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" }) : happyScript(spec);
    const { ctx: reachedCtx } = makeCtx(blockedScript);
    const reachedOutcome = await lazy.start(specFor({ issue: "i" }), reachedCtx).outcome;
    expect(reachedOutcome.error?.classification).toBe("permanent");
    expect(reachedOutcome.error?.reason).toMatch(/InteractionPort/);
  });

  it("interactive path works when a port is supplied (blocked critique → human decision)", async () => {
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const script: Script = (spec) =>
      modelOf(spec) === "critic"
        ? ok({ outcome: "blocked", weaknesses: [], critique_report: "stuck" })
        : happyScript(spec);
    const { ctx } = makeCtx(script);
    const port = new ScriptedPort({ "feature/plan/critique/human_review": [{ decision: "block" }] });
    const outcome = await executor.start(specFor({ issue: "i" }, { interaction: port }), ctx).outcome;
    expect(outcome.error).toBeUndefined();
    const value = outcome.value as Record<string, unknown>;
    expect(value["outcome"]).toBe("blocked");
    expect((value["critique"] as Record<string, unknown>)["human_decision"]).toBe("block");
  });

  it("cancel() yields a canceled outcome; spec timeout yields deadline", async () => {
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const hanging: Script = () => new Promise(() => {});
    {
      const { ctx } = makeCtx(hanging);
      const port = new ScriptedPort({});
      const handle = executor.start(specFor({ issue: "i" }, { interaction: port }), ctx);
      setTimeout(() => void handle.cancel(), 20);
      const outcome = await handle.outcome;
      expect(outcome.error?.classification).toBe("canceled");
    }
    {
      const { ctx } = makeCtx(hanging);
      const port = new ScriptedPort({});
      const handle = executor.start(
        specFor({ issue: "i" }, { interaction: port, limits: { timeoutMs: 40 } }),
        ctx,
      );
      const outcome = await handle.outcome;
      expect(outcome.error?.classification).toBe("deadline");
    }
  });

  it("enforces a spec-level output contract on top of per-state validation", async () => {
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const { ctx } = makeCtx(happyScript);
    const port = new ScriptedPort({});
    const outcome = await executor
      .start(
        specFor(
          { issue: "i" },
          {
            interaction: port,
            outputSchema: {
              type: "object",
              properties: { outcome: { type: "string", enum: ["blocked"] } }, // demands blocked; run yields complete
              required: ["outcome"],
            },
          },
        ),
        ctx,
      )
      .outcome;
    expect(outcome.error?.classification).toBe("api-retriable");
    expect(outcome.error?.reason).toMatch(/spec contract/);
    expect((outcome.value as Record<string, unknown>)["outcome"]).toBe("complete"); // value preserved on failure
  });

  it("workflowMemoKey matches core memoKey over (kind, snapshot hash, inputs)", () => {
    const spec = specFor({ issue: "i" });
    const definitionHash = snapshotHash(loadBundle(specPlanningFiles(), PLAN_ID));
    expect(workflowMemoKey(spec)).toBe(
      memoKey({ kind: "hierarchical-workflow", definitionHash, inputs: spec.inputs }),
    );
    // Same workflow version + same inputs ⇒ same key; any change ⇒ new key.
    expect(workflowMemoKey(specFor({ issue: "i" }))).toBe(workflowMemoKey(spec));
    expect(workflowMemoKey(specFor({ issue: "j" }))).not.toBe(workflowMemoKey(spec));
  });

  it("requires ctx.registry", async () => {
    const executor = createHierarchicalWorkflowExecutor({ providers: PROVIDERS });
    const port = new ScriptedPort({});
    const outcome = await executor
      .start(specFor({ issue: "i" }, { interaction: port }), { validator: new SchemaValidator() })
      .outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/ctx.registry/);
  });
});
