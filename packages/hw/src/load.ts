/**
 * Loading a stopped run — the description a machine is CONSTRUCTED from (Identity and Resume §04).
 *
 * Resume used to be a deterministic fast-forward: the engine started at the root and re-walked the
 * entire workflow, taking a recorded answer at each operation instead of dispatching. It rebuilt the
 * tree by redoing the walk that produced it, because nothing stored WAS a machine state. That had
 * three observed costs: it journaled a second walk (a whole second instance tree, folded against
 * the first), guard calls re-executed (the call cache was in-memory and per-run), and nothing
 * survived to be pointed at.
 *
 * The evaluation loop is MARKOVIAN over the instance's own fields — `opRun`, `cursor`, the children
 * records, `justFinished`, `heldFor` — which is what makes loading possible at all: reconstruct the
 * fields and re-enter the loop. This module is the description's shape; the construction lives in
 * the engine (`WorkflowEngine.loadRun`), because it is made of the same private pieces the walk is.
 *
 * The host builds a {@link LoadedInstance} tree from its journal joined to its record store — the
 * log carries the tree, the inputs and the transitions; the records carry the outputs. Neither
 * alone is sufficient, and together they are complete. The engine then re-dispatches ONLY the
 * active leaves: an instance loaded live whose operation never honestly settled re-dispatches it —
 * and because its id and site are the recorded ones, the scoped record id recomputes identically
 * and the store REOPENS the interrupted record rather than inserting a second ask.
 */
import type { Failure, ResolvedValue } from "@declarative-ai/exec";
import type { TerminationOutcome } from "./format.js";
import type { WorkflowMetrics } from "./ports.js";

/** A completed operation's recorded outcome — what the loaded instance's op already answered. */
export interface LoadedOperation {
  /** The value the call produced — what `acceptOpOutputs` is fed, exactly as a live settle feeds it. */
  value: ResolvedValue;
  metrics?: WorkflowMetrics;
  /** The model that answered, when the journal recorded one — `operation.model` in expressions. */
  model?: string;
  /** The conversation position the call ended at (`<session>@<seq+1>`) — `operation.output.session`. */
  sessionRef?: string;
}

/**
 * One instance of the stopped run, as the journal recorded it.
 *
 * `live: true` marks the spine that continues: an instance that was entered and never terminated.
 * Everything else is history the parent reads — its outputs are recomputed by the engine from the
 * recorded operation value plus the pinned definition, because a state's declared outputs are a
 * pure function of those (`finish()` resolves each slot the same way it did the first time).
 */
export interface LoadedInstance {
  /** The DURABLE id the journal minted — kept, which is the whole point of loading. */
  id: string;
  stateId: string;
  childKey?: string;
  /** Which entry under `(parent, childKey)` this was — the address's occurrence half. Default 0. */
  occurrence?: number;
  /**
   * Which ELEMENT of a fanned-out mount this was (`InstanceAddressStep.element`). The elements of one
   * entry share an `occurrence` and are told apart here; a loaded fan-out is rebuilt from them as one
   * child record, exactly as the live engine keeps one record per key.
   */
  element?: number;
  inputs: Record<string, ResolvedValue>;
  /** Transitions taken / backward passes, as `transition.taken` recorded them. */
  index?: number;
  iteration?: number;
  /** The sequence cursor — where the spine walk stands. */
  cursor?: number;
  /** Entered and never terminated: this instance CONTINUES. */
  live: boolean;
  /** How a terminated instance ended. Ignored when `live`. */
  outcome?: TerminationOutcome;
  failure?: Failure;
  /** The state's own operation, when it COMPLETED. A live instance without one re-dispatches. */
  operation?: LoadedOperation;
  /**
   * Children that finished and were never answered by an evaluation round — finished after the
   * parent's last advancement (a transition taken, or a later child entered). These seed
   * `justFinished`, so the first loaded round answers exactly what the stopped run still owed.
   */
  unanswered?: readonly string[];
  /**
   * The CALL SITES this instance had dispatched from, `(content key, sequence)` — rebuilt from its
   * records so a re-evaluated guard recomputes the same scoped id and finds its own answer. A site
   * whose content key cannot be recomputed (a rendered prompt differs from its authored form) is
   * simply absent; the dispatch mints a fresh site above `nextSite` and pays again, which is a cost
   * and never a collision.
   */
  sites?: ReadonlyArray<readonly [string, number]>;
  /** The next fresh site number — at least one past every recorded sequence. */
  nextSite?: number;
  /**
   * The instance's SETTLED FIELDS (SPEC §5.3), by authored path, as `value.settled` recorded them —
   * a title, a bound model, whatever the author computed. Used verbatim: a loaded instance does not
   * re-evaluate a field the stopped run already paid for, which is what "once" means across a
   * restart. A field absent here (the run stopped mid-evaluation) is evaluated afresh.
   */
  fields?: Readonly<Record<string, ResolvedValue>>;
  /** In entry order. Superseded instances are omitted; their entries survive in `occurrence`. */
  children?: readonly LoadedInstance[];
}
