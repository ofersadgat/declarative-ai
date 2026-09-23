/**
 * DIRECTED transitions (SPEC §3.3) — a transition the HOST injects into a run on somebody's behalf.
 *
 * Every other transition is taken because a rule fired: a guard the author wrote came true. A
 * directed one is taken because a person said "go there" — by dragging a card, or by saying it to a
 * conversation that holds the workflow as a tool — and the host hands the engine the move. The
 * engine owns what that means, because only it can do the three things in the right order:
 *
 *  - **decide before it interrupts.** A skip interrupts the running child, and an interrupted
 *    operation may well COMPLETE (an agent's `interrupt()` ends the turn early and the call succeeds
 *    with the partial answer). If the jump were decided after that landed, the child's completion
 *    would run an ordinary evaluation round and walk the run into the next member. So the child is
 *    recorded `skipped` and the transition is journaled first, and only then is anything aborted —
 *    by which point nothing the child does can move the run.
 *  - **hold.** A move asked for while the source is still running, without `skip`, is kept for that
 *    instance and taken when the running child ends, ahead of every rule.
 *  - **record what was stepped over.** The interrupted child and every sequence member between the
 *    cursor and the target that was never entered end `skipped`, which an expression reads as
 *    `outcome: "skipped"` rather than as the absence a never-run child is.
 *
 * This module is the vocabulary and the PORT. The port exists because a host does not hold the
 * engine: `WorkflowExecutor` builds one per execution, inside the call. The host makes a port, hands
 * it to the executor, and speaks to whichever engine is attached — or to none yet, in which case the
 * move is queued and claimed by the instance it names as the loaded run builds it, which is how a
 * FINISHED run takes a move: by being loaded with one waiting for it.
 */
import type { ResolvedValue } from "@declarative-ai/exec";

/** Who asked for a directed transition: a person directly, or a control conversation on their behalf. */
export type TransitionAsker = "person" | "control";

export interface DirectedTransition {
  /**
   * The composite whose child is entered, by durable instance id. Absent ⇒ the run's root instance.
   *
   * On a LOADED run this may name an instance that had already terminated: it and every ancestor are
   * reopened to take the move (see the module header).
   */
  instanceId?: string;
  /** A declared child key of that instance — never a `terminate.*` outcome. */
  to: string;
  /**
   * The WAY DOWN from `to`, when the target is nested deeper: the child keys beneath it, one per
   * level (`to: "ux", path: ["item", "draft"]` is `ux → item → draft`).
   *
   * A composite that has not been entered has no instance id, so no move can name it ahead of time.
   * The engine takes the way down itself: the moment `to` is entered, the next step is directed at
   * the instance it just made — `{ to: path[0], path: path.slice(1) }`, same `by` and `skip` — and so
   * on to the end. Each step is its own directed transition, journaled on the composite that takes it
   * with the rest of the way still to go (`transition.taken`'s `descent`), which is what lets a LOADED
   * run pick the way down up where a stopped one left it (`LoadedInstance.descent`). What comes before
   * the named child inside each composite is stepped over as it is at the top.
   *
   * Absent or empty ⇒ `to` is the target.
   */
  path?: readonly string[];
  /**
   * What the asker hands the TARGET, over the mount's wiring per name — exactly a transition's
   * `inputs`. With a {@link path}, the target is the LAST state on the way down; the composites
   * entered on the way are handed nothing but what a standing rule wires.
   */
  inputs?: Record<string, ResolvedValue>;
  by: TransitionAsker;
  /**
   * Interrupt instead of wait. Absent/false ⇒ the move is HELD while a sync child holds the cursor and
   * taken when that child ends; true ⇒ the running sequence children are recorded `skipped` and
   * aborted, and the move is taken at once. Carried to every step of a {@link path}.
   */
  skip?: boolean;
}

/**
 * The rest of a directed move's way down, as a journal row and a loaded instance carry it: the child
 * keys still to go beneath the state just entered, and what the asker hands the last of them.
 */
export interface DirectedDescent {
  path: readonly string[];
  inputs?: Record<string, ResolvedValue>;
  by: TransitionAsker;
  skip?: boolean;
}

/** The step a descent takes next, directed at the instance that was just entered on the way down. */
export function nextStepOf(instanceId: string, descent: DirectedDescent): DirectedTransition {
  const [to, ...rest] = descent.path;
  return {
    instanceId,
    to: to!,
    ...(rest.length > 0 ? { path: rest } : {}),
    ...(descent.inputs !== undefined ? { inputs: descent.inputs } : {}),
    by: descent.by,
    ...(descent.skip === true ? { skip: true } : {}),
  };
}

/** The descent a move hands the state it enters — `undefined` when that state is the target. */
export function descentOf(move: DirectedTransition): DirectedDescent | undefined {
  if (move.path === undefined || move.path.length === 0) return undefined;
  return {
    path: [...move.path],
    ...(move.inputs !== undefined ? { inputs: move.inputs } : {}),
    by: move.by,
    ...(move.skip === true ? { skip: true } : {}),
  };
}

/**
 * What became of a move the moment it was handed over.
 *
 *  - `queued`  — no engine is attached yet; the move waits on the port for the run that loads.
 *  - `held`    — the instance has it, and a running child stands in the way (no `skip`).
 *  - `taking`  — the instance has it and nothing stands in the way; its loop takes it next.
 *  - `refused` — it names no live instance, or `to` is not a child of the one it names.
 */
export type DirectedOutcome = { status: "queued" | "held" | "taking" } | { status: "refused"; reason: string };

/** The half of the engine the port speaks to. */
export interface DirectedTarget {
  direct(move: DirectedTransition): DirectedOutcome;
}

/**
 * The host's handle on "the engine running this, whichever it is" — see the module header.
 *
 * One port serves one run at a time. A move handed over while nothing is attached is queued in
 * arrival order; the engine drains the queue when it attaches and each move is claimed by the
 * instance it names when that instance is built.
 */
export class DirectedTransitions {
  private target: DirectedTarget | undefined;
  private queue: DirectedTransition[] = [];

  /** Hand the run a move. Never throws: a move that cannot be taken is `refused` with the reason. */
  direct(move: DirectedTransition): DirectedOutcome {
    if (this.target !== undefined) return this.target.direct(move);
    this.queue.push(move);
    return { status: "queued" };
  }

  /** Moves handed over and not yet claimed by any engine. */
  queued(): readonly DirectedTransition[] {
    return [...this.queue];
  }

  /** ENGINE-side: start receiving moves, and take what queued up before there was anyone to give them to. */
  attach(target: DirectedTarget): DirectedTransition[] {
    this.target = target;
    const queued = this.queue;
    this.queue = [];
    return queued;
  }

  /** ENGINE-side: the run ended. A later move queues for the next run rather than reaching a dead one. */
  detach(target: DirectedTarget): void {
    if (this.target === target) this.target = undefined;
  }
}

/**
 * The reason a skipped child's abort signal carries.
 *
 * An `Error` named `AbortError` on purpose: everything downstream that classifies a cancellation
 * does it by that name, and a skip IS a cancellation to the call it interrupts. The extra field is
 * what lets the engine tell "somebody stepped past this" from "the run was stopped" when it names
 * the instance's outcome.
 */
export class SkipAbort extends Error {
  readonly skipped = true;
  constructor() {
    super("skipped by a directed transition");
    this.name = "AbortError";
  }
}

export function isSkipAbort(reason: unknown): reason is SkipAbort {
  return reason instanceof Error && (reason as { skipped?: unknown }).skipped === true;
}
