/**
 * The engine's outward-facing ports (DESIGN §7). Apps supply implementations:
 * JaiRA provides durable persistence and renderer-backed interactive `functions`;
 * findmyprompt provides the bundled in-memory persistence and scripted functions.
 * Interactive/host behavior now lives in the `CapabilityRegistry<WorkflowMetrics>` (`registry.functions`,
 * @declarative-ai/exec); the engine only ever talks to these interfaces.
 */
import type { BudgetMetrics, ExecMetrics, Failure, ResolvedValue } from "@declarative-ai/exec";

/**
 * What a workflow run accounts for: execution timing and counts, PLUS spend.
 *
 * hw reads money — it rolls a subtree's cost into the parent's run record — so it constrains its `M`
 * to say so, rather than exec inventing a cost field for everyone. An executor whose metrics carry no
 * cost cannot be run by this engine, and that is a compile error rather than a silent zero.
 */
export type WorkflowMetrics = ExecMetrics & BudgetMetrics;

/** How two workflow measurements combine: durations and spend add, the start is the first observation. */
export function mergeWorkflowMetrics(a: WorkflowMetrics, b: WorkflowMetrics): WorkflowMetrics {
  return {
    durationMs: a.durationMs + b.durationMs,
    startMs: a.startMs ?? b.startMs,
    childLlmCalls: (a.childLlmCalls ?? 0) + (b.childLlmCalls ?? 0),
    costUsd: a.costUsd + b.costUsd,
    costSource: b.costSource,
    childCostUsd: (a.childCostUsd ?? 0) + (b.childCostUsd ?? 0),
  };
}
import type { TerminationOutcome } from "./format";

/** An artifact value flowing through workflow inputs/outputs (SPEC §4.6). For
 *  llm-backed states the content travels inline; process units use paths.
 *
 *  A type ALIAS rather than an interface, deliberately: an artifact ref is one of the values that
 *  flows through a state's inputs/outputs, and only an alias gets the implicit index signature that
 *  makes it a `ResolvedValue` without a cast. */
export type ArtifactRef = {
  artifact: true;
  name: string;
  format?: string;
  content?: string;
  path?: string;
};

export function isArtifactRef(v: unknown): v is ArtifactRef {
  return typeof v === "object" && v !== null && (v as { artifact?: unknown }).artifact === true;
}

/** Engine events — the run-record stream (SPEC §10.2). Consumed live by the
 *  executor/UI and persisted by the `Persistence` port. */
export type OperationKind = "prompt" | "function";

export type EngineEvent =
  | { type: "instance.entered"; instanceId: number; stateId: string; childKey?: string; parentInstanceId?: number; inputs: Record<string, ResolvedValue> }
  | { type: "instance.blocked"; instanceId: number; stateId: string; reason: string }
  | { type: "operation.started"; instanceId: number; stateId: string; op: OperationKind }
  | { type: "operation.completed"; instanceId: number; stateId: string; op: OperationKind; metrics?: WorkflowMetrics }
  | { type: "operation.failed"; instanceId: number; stateId: string; op: OperationKind; failure: Failure }
  | { type: "transition.taken"; instanceId: number; stateId: string; to: string; iteration: number }
  | { type: "child.superseded"; instanceId: number; stateId: string; childKey: string }
  | { type: "instance.terminated"; instanceId: number; stateId: string; outcome: TerminationOutcome; failure?: Failure };

/**
 * Durable run recording (SPEC §10.2/§10.3). The engine calls `record` at every step;
 * implementations persist (JaiRA: SQLite) or buffer (in-memory). v1 executes runs
 * in-process; cross-restart resume is built by consumers on top of this record
 * stream plus the snapshot hash.
 */
export interface Persistence {
  record(event: EngineEvent, atMs: number): void;
}

/** The bundled in-memory persistence — embedding & tests. */
export class InMemoryPersistence implements Persistence {
  readonly events: Array<{ event: EngineEvent; atMs: number }> = [];
  record(event: EngineEvent, atMs: number): void {
    this.events.push({ event, atMs });
  }
}

// Operation dispatch goes through the typed `CapabilityRegistry<WorkflowMetrics>` (@declarative-ai/exec) BY OP KIND
// (API.md, "Operation dispatch & ports"): a `PromptOp` runs through `registry.prompt` (the llm leaf runner), a `FunctionOp`
// through `registry.functions` — including sub-workflows, composite units, and delegated agent adapters,
// which are registered async functions like any other (§3.1). `registry.skills` maps a skill name to its
// prompt template; `registry.tools` maps a logical tool name to its executable.
