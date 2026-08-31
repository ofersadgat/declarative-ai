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
  | { type: "instance.entered"; instanceId: string; stateId: string; childKey?: string; parentInstanceId?: string; inputs: Record<string, ResolvedValue> }
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
   * `operationId` is the content hash of the op AS DISPATCHED (`hashOperation` over the value the
   * executor stack received) — which is exactly the id `withRecord` gives an UNPLACED record, so a
   * journal row and its operation record share a key at last. A PLACED call's record is keyed by
   * its session position instead, and its join was always `metrics.sessionRef`; the hash is
   * stamped regardless, so every settled operation event names the call it settled. Absent on
   * `operation.started` deliberately (it fires before input resolution, so the dispatched op — the
   * thing the hash is OF — does not exist yet), and on a `failed` that never reached dispatch.
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
}

/**
 * Where an instance sits in the tree, root-first. The empty address IS the root.
 *
 * Deliberately neither the instance id nor the operation's content hash. Ids are minted fresh
 * (UUIDv7) as the engine walks, so a second walk mints its own and the two never agree; the content
 * hash collides whenever a loop dispatches the identical operation twice, which is exactly why an
 * operation record carries an attempt number. A position in the tree is stable across runs of one
 * pinned definition, which is the only comparison a replay ever makes.
 */
export type InstanceAddress = readonly InstanceAddressStep[];

/** What a previous run recorded for one state's operation. */
export interface ReplayedOperation {
  /** What the call returned, exactly as the executor produced it. */
  value: ResolvedValue;
  /**
   * The conversation position the call ended at, for an operation that ran in one.
   *
   * Supplied so `operation.output.session` reads on a replayed state as it did on the original — a
   * later state threading `{"expr": ".children.plan.operation.output.session"}` binds to a position,
   * and a resume that dropped it would resolve that expression to nothing. Absent means the
   * operation had no conversation, and the namespace is absent exactly as it would have been.
   */
  session?: { position: string; conversation: string };
  /** The model that actually served it, for `operation.model`. */
  model?: string;
  /**
   * The content id of the call this answer was read out of.
   *
   * Stamped onto the replayed `operation.completed` so the resumed run's own journal still names the
   * record behind the answer. hw does not interpret it: whether an id minted under an earlier run
   * means anything under a later one is a question about how the host scopes its record store.
   */
  operationId?: string;
}

/**
 * The answers a stopped run already produced, for a run re-entering it.
 *
 * Resume here is a deterministic FAST-FORWARD rather than a rehydration: the run starts at the root
 * as usual, and at every operation this source can answer, the answer is taken instead of the call
 * being made. The engine therefore rebuilds its own instance tree by walking it again, and resolves
 * each state's declared outputs with its own `finish()` — so no part of an instance has to be
 * reconstructed from outside, and no engine logic gets a second implementation to drift from.
 *
 * The last clause is the safety property the whole shape exists for. An operation with side effects
 * — a file written, a changeset applied, an agent editing a worktree — is not re-executed on the way
 * back to where the run stopped, because it is not dispatched at all.
 *
 * A source answering `undefined` is saying "this one still has to run", which is how the frontier is
 * expressed: there is no separate declaration of where to resume, only the point at which the
 * answers run out.
 */
export interface ReplaySource {
  operationAt(address: InstanceAddress): ReplayedOperation | undefined;
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
