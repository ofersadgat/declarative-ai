/**
 * Whose ask is whose — the engine names the asking INSTANCE on every approval and question request.
 *
 * A host places one approver and one question channel on a run's services, so a parked request
 * could not say which instance made it. That mattered the moment a directed skip interrupted one
 * agent while an `async` sibling kept running: the host could not withdraw the interrupted agent's
 * asks without taking the sibling's too. The engine stamps `instanceId` on the approver and the asker
 * it hands each operation — the gate's escalation included — and keeps a stamp already present, so
 * the innermost instance of a nested run is the one named.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type ExecServices, type FunctionInputs, type ResolvedValue } from "@declarative-ai/exec";
import type { PermissionDecision, PermissionRequest, UserQuestionRequest } from "@declarative-ai/permissions";
import { SchemaValidator } from "@declarative-ai/validate";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";

const leaf = (name: string): StateDef => ({
  label: name,
  outputs: { answer: { schema: {} } },
  operation: { kind: "function", function: "ask", args: { name } },
});

const FILES: Record<string, StateDef> = {
  root: {
    label: "Root",
    children: { left: { state: "root/left" }, phase: { state: "root/phase" } },
    sequence: ["left", "phase"],
  },
  "root/left": leaf("left"),
  "root/phase": { label: "Phase", children: { inner: { state: "root/phase/inner" } }, sequence: ["inner"] },
  "root/phase/inner": leaf("inner"),
};

function harness() {
  const approvals: PermissionRequest[] = [];
  const questions: UserQuestionRequest[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "ask",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs, ctx: ExecServices) => {
        const name = (inputs as { name: string }).name;
        await ctx.approve?.({ tool: "bash", input: { command: name }, sessionId: "s" });
        await ctx.askUser?.({ questions: [{ question: `which way, ${name}?`, options: [{ label: "this" }] }], sessionId: "s" });
        // The GATE — what a delegated agent consults — escalates to the same approver.
        await ctx.gate?.check({ name: "an_agent_builtin" }, {});
        // A stamp already present is the innermost instance's, and is kept.
        await ctx.approve?.({ tool: "bash", input: {}, sessionId: "s", instanceId: "already-named" });
        return ok({ answer: name }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      },
      { interactive: false, readOnly: true, memoizable: false },
    ),
  );
  const bundle = loadBundle(FILES, "root", { functions: registry.functions });
  const persistence = new InMemoryPersistence();
  let n = 0;
  const engine = new WorkflowEngine({
    bundle,
    registry,
    validator: new SchemaValidator(),
    persistence,
    newInstanceId: () => `i-${++n}`,
    services: {
      approve: (req): PermissionDecision => {
        approvals.push(req);
        return { decision: "allow", scope: "once" };
      },
      askUser: (req) => {
        questions.push(req);
        return Promise.resolve(undefined);
      },
    },
  });
  const idOf = (key: string): string | undefined =>
    persistence.events.map(({ event }) => event).find((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered" && e.childKey === key)?.instanceId;
  return { engine, approvals, questions, idOf };
}

describe("an approval or a question names the instance that asked", () => {
  it("on the approver, the question channel and the gate's escalation alike — at every depth", async () => {
    const h = harness();
    expect((await h.engine.run({ inputs: {} })).outcome).toBe("success");
    const left = h.idOf("left")!;
    const inner = h.idOf("inner")!;
    expect(left).toBeDefined();
    expect(inner).toBeDefined();
    expect(h.questions.map((q) => [q.questions[0]!.question, q.instanceId])).toEqual([
      ["which way, left?", left],
      ["which way, inner?", inner],
    ]);
    expect(h.approvals.map((a) => a.instanceId)).toEqual([left, left, "already-named", inner, inner, "already-named"]);
  });
});
