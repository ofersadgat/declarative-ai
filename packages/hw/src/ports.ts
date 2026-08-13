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

/**
 * How well a cost figure is known — `mergeLlmMetrics`'s ranking, restated here for the same reason
 * `durationMs` is: hw does not import the llm layer, and the two records are kept compatible by hand.
 */
const COST_SOURCE_RANK: Record<WorkflowMetrics["costSource"], number> = { provider: 2, table: 1, unknown: 0 };

/** How two workflow measurements combine: durations and spend add, the start is the first observation. */
export function mergeWorkflowMetrics(a: WorkflowMetrics, b: WorkflowMetrics): WorkflowMetrics {
  return {
    durationMs: a.durationMs + b.durationMs,
    startMs: a.startMs ?? b.startMs,
    childLlmCalls: (a.childLlmCalls ?? 0) + (b.childLlmCalls ?? 0),
    costUsd: a.costUsd + b.costUsd,
    // The BETTER-KNOWN source, not the right-hand one. Taking `b` blindly meant any zero-cost step
    // merged after a real call downgraded that call's provenance to whatever the step happened to
    // claim — a provider-authoritative figure reported as a guess, and the roll-up saying so. It also
    // makes `empty()` below a true identity: `unknown` ranks lowest, so folding it in changes nothing.
    costSource: COST_SOURCE_RANK[a.costSource] >= COST_SOURCE_RANK[b.costSource] ? a.costSource : b.costSource,
    childCostUsd: (a.childCostUsd ?? 0) + (b.childCostUsd ?? 0),
  };
}

/** The neutral {@link WorkflowMetrics}: no time, no spend, and no claim about where a price came from. */
export function emptyWorkflowMetrics(): WorkflowMetrics {
  return { durationMs: 0, costUsd: 0, costSource: "unknown" };
}
import type { TerminationOutcome } from "./format.js";

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
  /**
   * `operationId` is the content hash of the op AS DISPATCHED (`hashOperation` over the value the
   * executor stack received) — which is exactly the id `withRecord` gives an UNPLACED record, so a
   * journal row and its operation record share a key at last. A PLACED call's record is keyed by
   * its session position instead, and its join was always `metrics.sessionRef`; the hash is
   * stamped regardless, so every settled operation event names the call it settled. Absent on
   * `operation.started` deliberately (it fires before input resolution, so the dispatched op — the
   * thing the hash is OF — does not exist yet), and on a `failed` that never reached dispatch.
   */
  | { type: "operation.completed"; instanceId: number; stateId: string; op: OperationKind; operationId?: string; metrics?: WorkflowMetrics }
  | { type: "operation.failed"; instanceId: number; stateId: string; op: OperationKind; operationId?: string; failure: Failure }
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
