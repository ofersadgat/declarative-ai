/**
 * A rule that answers the state's OWN operation (`after: "operation"`, SPEC §3.3): eligible in the round
 * that operation's completion triggered, once — as a rule on a child's mount is eligible only in the
 * round that child's end triggered. A rule on the state's plain list is consulted in every round, so one
 * that enters a child after the operation, guarded on the operation's output, would enter it again in
 * any later round that found nothing else to do.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type JsonValue, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { compileScript } from "../src/scriptCompile.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { newRegistry, ok } from "./fakes.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

function run(files: Record<string, StateDef>, root: string) {
  const registry = newRegistry();
  let works = 0;
  registry.functions.set("go", hostFunction(async () => ok({ go: true }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
  registry.functions.set(
    "work",
    hostFunction(async () => {
      works += 1;
      // A runaway re-entry would never end: stop it, and let the count say what happened.
      if (works > 5) return { error: { classification: "permanent", reason: "entered again and again" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } } as never;
      return ok({ done: true as JsonValue }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const persistence = new InMemoryPersistence();
  const bundle = loadBundle(files, root);
  const engine = new WorkflowEngine({ bundle, registry, validator: new SchemaValidator(), persistence } as never);
  return { engine, bundle, works: () => works, events: (): EngineEvent[] => persistence.events.map(({ event }) => event) };
}

/** A state that runs its own operation, then a child its rule picks from the operation's output. */
const files = (after: boolean): Record<string, StateDef> => ({
  s: {
    operation: { function: "go", args: {} },
    children: { work: { state: "w" } },
    transitions: [{ to: "work", when: ".operation.output.go", ...(after ? { after: "operation" as const } : {}) }],
  },
  w: { operation: { function: "work", args: {} } },
});

describe("a rule after the state's own operation", () => {
  it("is taken in the round the operation's completion triggered, and never again", async () => {
    const { engine, works } = run(files(true), "s");
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(works()).toBe(1);
  });

  it("on the state's plain list, is consulted in every round — the re-entry `after` exists to prevent", async () => {
    const { engine, works } = run(files(false), "s");
    const result = await engine.run({ inputs: {} });
    expect(works()).toBeGreaterThan(1);
    expect(result.outcome).toBe("error");
  });

  it("is still owed when a run stopped between the operation and its first round, and taken on resume", async () => {
    const { engine, works } = run(files(true), "s");
    const loaded: LoadedInstance = { id: "root-1", stateId: "s", inputs: {}, live: true, index: 0, operation: { value: { go: true } }, children: [] };
    const result = await engine.loadRun(loaded);
    expect(result.outcome).toBe("success");
    expect(works()).toBe(1);
  });

  it("is not taken again by a resumed run that had answered it", async () => {
    const { engine, works } = run(files(true), "s");
    const loaded: LoadedInstance = {
      id: "root-1",
      stateId: "s",
      inputs: {},
      live: true,
      index: 1,
      operation: { value: { go: true } },
      children: [{ id: "work-1", stateId: "w", childKey: "work", inputs: {}, live: false, outcome: "success", operation: { value: { done: true } } }],
    };
    const result = await engine.loadRun(loaded);
    expect(result.outcome).toBe("success");
    expect(works()).toBe(0);
  });

  it("is refused on a child's mount, on a standing rule, and on a state with no operation", () => {
    const bad: Record<string, StateDef> = {
      s: {
        children: { work: { state: "w", transitions: [{ to: "terminate.success", after: "operation" }] } },
        transitions: [{ to: "work", after: "operation" }, { to: "work", when: "true", standing: true, after: "operation" } as never],
      },
      w: { operation: { function: "work", args: {} } },
    };
    const messages = validateBundle(loadBundle(bad, "s")).errors.map((e) => e.message);
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/only the state's own list can answer its operation/),
        expect.stringMatching(/a standing rule answers nothing in particular/),
        expect.stringMatching(/this state has no operation for the rule to answer/),
      ]),
    );
  });

  it("is how a compiled script writes the rules after its own code", async () => {
    const code = `
import { llm, input } from "@declarative-ai/hw/script";
export default async function chat(first: string): Promise<string> {
  const said = await llm(first);
  await input();
  return said;
}
`;
    const compiled = await compileScript({ file: "/wf/chat.ts", stateId: "chat", vfs: { read: (p: string) => (p === "/wf/chat.ts" ? code : undefined), list: () => [] }, requirePath: ["/wf"] });
    const root = compiled.documents["chat"]! as StateDef;
    expect(root.transitions!.length).toBeGreaterThan(0);
    expect(root.transitions!.every((t) => t.after === "operation")).toBe(true);
  });
});
