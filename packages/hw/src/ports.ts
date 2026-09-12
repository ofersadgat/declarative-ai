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
  | {
      type: "instance.entered";
      instanceId: string;
      stateId: string;
      childKey?: string;
      parentInstanceId?: string;
      /**
       * Which ELEMENT this instance is, when the mount fans out (`each: true` on one of its wires):
       * the row-major position across every `each` axis, 0-based. Absent for an ordinary entry.
       *
       * Carried on the event because a reader of the journal cannot otherwise tell a fan-out's third
       * element from a loop's third pass — both are the third entry under one key in one parent — and
       * the two mean opposite things: a pass SUPERSEDES the one before it, an element sits beside it.
       */
      element?: number;
      inputs: Record<string, ResolvedValue>;
    }
  /**
   * A child that could not be ENTERED, because its input wiring did not resolve.
   *
   * There is no `instanceId` — nothing became an instance, which is the whole event. So the MOUNT is
   * the only address it has: `stateId` names the state definition, and one definition is mounted under
   * several keys in several parents (`explore` sits under all six phases of the feature workflow).
   * Without the parent and the key, a reader is told a block happened somewhere and not where.
   */
  | { type: "instance.blocked"; stateId: string; childKey?: string; parentInstanceId?: string; reason: string }
  | { type: "operation.started"; instanceId: string; stateId: string; op: OperationKind }
  /**
   * The record for this call EXISTS — emitted from the record layer's own callback, at the moment
   * the row is written and its position claimed, before the provider call is made. That order is
   * the invariant: this event can never name a row that was not written, and a crash between the
   * insert and the call leaves an open row for the recovery sweep rather than an event pointing at
   * nothing. Not emitted when the position claim is refused — no row exists, and the typed
   * `positionTaken` failure already carries the fork signal.
   */
  | { type: "operation.dispatched"; instanceId: string; stateId: string; op: OperationKind; operationId: string }
  /**
   * `operationId` is the SCOPED id — the op's content hash folded with the dispatch site
   * `(instanceId, sequence)`, exactly the id `withRecord` keys the record by (`scopedOperationId`),
   * so a journal row and its operation record share a key that cannot collide: the site never
   * repeats, a loop's next iteration is a new instance, and a guard's third round computes the same
   * id as its first. Both layers compute it independently from the same parts; no id is ever handed
   * across a boundary to be agreed on. Absent on `operation.started` deliberately (it fires before
   * input resolution, so the dispatched op — the thing the hash is OF — does not exist yet), and on
   * a `failed` that never reached dispatch.
   */
  | { type: "operation.completed"; instanceId: string; stateId: string; op: OperationKind; operationId?: string; metrics?: WorkflowMetrics }
  /**
   * `metrics` is present exactly when the operation actually RAN — a post-dispatch failure, where
   * the call was made and the money was spent. A pre-dispatch failure (unresolvable inputs, no
   * executor wired) carries none, because nothing ran to measure.
   *
   * It closes two gaps, both silent. A failed call's SPEND was invisible: an agent that burned a
   * dollar and then failed its output schema reported nothing, so every roll-up under-counted. And
   * `metrics.sessionRef` is the journal's join to the conversation a call ran in — carried on
   * completion since the beginning and dropped here — so a failed call's transcript sat in the
   * record store with nothing in the journal pointing at it.
   */
  | {
      type: "operation.failed";
      instanceId: string;
      stateId: string;
      op: OperationKind;
      operationId?: string;
      failure: Failure;
      metrics?: WorkflowMetrics;
    }
  /**
   * A DEFERRED call was started and the round moved on without it — the state is now waiting on
   * something outside the run (`HostCapabilities.deferred`).
   *
   * Recorded because a wait is otherwise invisible in the journal: the last thing written is the
   * operation that completed, and a run parked on a person reads exactly like one that hung. The
   * pair also gives a reader the DURATION of the wait, which is the number anybody asking "why did
   * this take a day" wants.
   */
  | { type: "call.waiting"; instanceId: string; stateId: string; call: string; operationId: string }
  | {
      type: "call.settled";
      instanceId: string;
      stateId: string;
      call: string;
      operationId: string;
      /** `error` covers a cancelled wait too — a failure is data, and this says which kind arrived. */
      outcome: "value" | "error";
    }
  /**
   * A COMPUTED FIELD settled (SPEC §5.3) — `title`, a bound `config.model`, any value position the
   * author wrote a binding in. Between `instance.entered` and this the field is PENDING, which a
   * board renders as it sees fit; after it the value is what the instance runs with.
   *
   * Carries the VALUE, because this is the journal's copy of it: a resumed run reads its settled
   * fields back from here rather than recomputing them, which is what "evaluated once" means across
   * a restart — and what keeps a title from being paid for twice and coming back different. A field
   * that FAILED and stood a `failureValue` in says so with `fallback`, and its `error` beside it.
   */
  | {
      type: "value.settled";
      instanceId: string;
      stateId: string;
      /** The field's authored path — `title`, `operation.config.model`, `environment.tools`. */
      field: string;
      outcome: "value" | "error";
      value?: ResolvedValue;
      error?: string;
      /** The value is the binding's `failureValue`, standing in for one it could not compute. */
      fallback?: boolean;
    }
  /** `index` counts every transition; `iteration` counts only the backward ones — the passes. */
  | { type: "transition.taken"; instanceId: string; stateId: string; to: string; index: number; iteration: number }
  | { type: "child.superseded"; instanceId: string; stateId: string; childKey: string }
  | { type: "instance.terminated"; instanceId: string; stateId: string; outcome: TerminationOutcome; failure?: Failure };

/**
 * Durable run recording (SPEC §10.2/§10.3). The engine calls `record` at every step;
 * implementations persist (JaiRA: SQLite) or buffer (in-memory). v1 executes runs
 * in-process; cross-restart resume is built by consumers on top of this record
 * stream plus the snapshot hash — see {@link ReplaySource}, which is the other half of that
 * sentence: this port writes the stream, that one reads it back.
 */
export interface Persistence {
  record(event: EngineEvent, atMs: number): void;
}

/** One step of an instance's address: which declared child, and which entry under it. */
export interface InstanceAddressStep {
  childKey: string;
  /** 0-based, in entry order — a loop's second iteration is occurrence 1. */
  occurrence: number;
  /**
   * The element position under a mount that FANS OUT — one entry of the mount is one occurrence, and
   * its elements share that occurrence and differ here. Absent for an ordinary entry, and absent is
   * not 0: an ordinary child and the first element of a fan-out are different addresses.
   */
  element?: number;
}

/**
 * Where an instance sits in the tree, root-first. The empty address IS the root.
 *
 * Deliberately neither the instance id nor the operation's content hash. Ids are minted fresh
 * (UUIDv7) as the engine walks and are then DURABLE — a loaded run keeps them — but a definition's
 * position is the one name that means the same thing in two runs of one pinned definition, which
 * is what a projection folding runs together compares by.
 */
export type InstanceAddress = readonly InstanceAddressStep[];

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
