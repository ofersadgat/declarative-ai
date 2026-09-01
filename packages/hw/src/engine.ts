/**
 * The hierarchical-workflow engine — SPEC §3 (semantics), §10 (lifecycle).
 *
 * One `WorkflowEngine` executes one workflow run in-process. The evaluation loop is a
 * faithful transcription of SPEC §3.3:
 *
 *   1. Entering a state creates a fresh instance; declared inputs are resolved and
 *      validated (failure blocks the state → error termination to the parent).
 *   2. The engine runs the highest-priority unrun operation: ui, agent, skill, then
 *      child states in sequence order.
 *   3. When an operation completes, transitions are evaluated in declared order; the
 *      first whose `when` is true is taken (PENDING-valued conditions are skipped).
 *   4. A taken transition enters a child or terminates the state.
 *   5. No match → next operation. Async children start without blocking (§10.4);
 *      their completion — like every child completion — triggers another evaluation
 *      round. A child whose input wiring evaluates to PENDING parks until the
 *      referenced outputs resolve (the dataflow join).
 *   6. No operations, no match: wait for running children; when none remain,
 *      terminate.success.
 *
 * Sequence resets (SPEC §3.3): a transition to a sequence member clears the recorded
 * results of that member and every later member (superseded — history is preserved in
 * the event record), cancels any of them still running, and default ordering resumes.
 * Entering a member MOVES THE CURSOR to it, and child selection is "first member at or
 * after the cursor with no live record" — so a transition is a jump in either direction:
 * backwards it re-runs the tail, forwards it leaves the members it skipped skipped.
 * The cursor then HOLDS on that member while it runs, unless the child is `async`;
 * one child at a time is the default, and concurrency is something an author asks for.
 *
 * Unhandled failures (SPEC §3.3): an unrecoverable operation failure terminates the
 * state with error; a child that terminated with error/timeout and is not handled by
 * any transition in the following evaluation round does the same.
 */
import {
  type CapabilityRegistry,
  type Failure,
  type ExecMetrics,
  type ExecServices,
  type Executor,
  type FunctionInputs,
  type FunctionRegistry,
  type SyncOutputValidator,
  type InlineFamily,
  type JsonSchema,
  type JsonValue,
  type FunctionOp,
  type NamedParameter,
  type Operation,
  type Parameter,
  type PromptOp,
  type RefKind,
  type RegisteredFunction,
  type SessionStore,
  type Ref,
  type ResolvedValue,
  type Tool,
  type Workspace,
  type Clock,
  type OperationScope,
  MapSessionStore,
  createOperationExecutor,
  hashOperation,
  isOk,
  resolveCalls,
  scopedOperationId,
} from "@declarative-ai/exec";
import type { InstanceAddress, WorkflowMetrics } from "./ports.js";
import type { LoadedInstance } from "./load.js";
import {
  createToolGate,
  PermissionLedger,
  planExitTool,
  withPermission,
  type Approver,
  type PermissionBaseline,
  type PermissionMode,
  type ToolGate,
  type ExecPolicy,
  type ProfileRule,
  type SmartApprover,
} from "@declarative-ai/permissions";
import { SchemaValidator } from "@declarative-ai/validate";
import { isPending, parseExpression, PENDING } from "./expr.js";
import { lowerExpression } from "./lowerExpr.js";
import type {
  ExecEnvironmentDecl,
  LoadedChild,
  LoadedState,
  LoadedTransition,
  SlotMeta,
  TerminationOutcome,
  WorkflowBundle,
} from "./format.js";
import { bindElement, bindInputs, embeddedOpsOf, higherOrderEdgesOf, higherOrderOf, isResolvedValue, isResolveError, resolveEmbedded, resolveInputs, resolveOperationInputs, resolveRef, type ResolutionScope, type Resolved } from "./resolve.js";
import { isByteStream, materialize, MaterializeError } from "./materialize.js";
import {
  RUN_RESOURCE_KEY,
  isSessionExpr,
  publishedSession,
  resolveSession,
  sessionFromExpr,
  type PublishedSession,
  type SessionBinding,
  type SessionDecl,
} from "./session.js";
import type { OperationNode } from "./operationNode.js";
import { isFannedOut } from "./fanout.js";
import { isArtifactRef, type ArtifactRef, type EngineEvent, type OperationKind, type Persistence } from "./ports.js";
import { uuidv7 } from "./ids.js";

/**
 * A round that cannot finish yet: a transition guard is waiting on a deferred call.
 *
 * A sentinel rather than a string outcome so it cannot be confused with a state id or an outcome
 * name at any of the three places a transition result is read.
 */
const WAITING: unique symbol = Symbol("ai-exec/hw transition waiting");

/**
 * A guard that could not be evaluated at all — distinct from one that evaluated to false.
 *
 * A sentinel for the same reason {@link WAITING} is: the three existing answers are a state id, a
 * wait, and nothing-matched, and "the question is broken" is none of those.
 */
const GUARD_FAILED: unique symbol = Symbol("ai-exec/hw transition guard failed");

/**
 * What a call's memo remembers: the value it produced, or the failure it produced.
 *
 * Both are DATA (§5) and both are serializable, which is what lets a host back this with something
 * durable. `PENDING` is deliberately not here — it is a scheduling state, not an answer.
 */
/** {@link hashOperation}, total: undefined for an op that has no stable content identity (a live
 *  stream input) rather than the throw a memo wants. The events stamped from this simply omit the
 *  id, matching the record layer's own fallback for the same op. */
function tryHashOperation(op: Operation<InlineFamily>): string | undefined {
  try {
    return hashOperation(op);
  } catch {
    return undefined;
  }
}

/** The scoped id an event stamps — {@link scopedOperationId} over a hash that may be absent. */
function tryScopedId(hash: string | undefined, scope: OperationScope): string | undefined {
  return hash === undefined ? undefined : scopedOperationId(hash, scope);
}

/** A recorded `sessionRef` (`<session>@<seq>`) back as the published node a loaded op carries. */
function publishedOfRef(ref: string | undefined): PublishedSession | undefined {
  if (ref === undefined) return undefined;
  const at = ref.lastIndexOf("@");
  return publishedSession(ref, at > 0 ? ref.slice(0, at) : ref);
}

export type CallResult = { value: ResolvedValue } | { error: string; failure?: Failure };

export interface EngineConfig {
  bundle: WorkflowBundle;
  /** The typed capability registry: `functions` (host code, sub-workflows, and delegated agent
   *  adapters alike — ONE map of discriminated entries, DESIGN §5.1), `skills` (named prompt
   *  templates), `tools` (executables an agent may call mid-loop). */
  registry: CapabilityRegistry<WorkflowMetrics>;
  /** The `Executor` a `PromptOp` dispatches to (`@declarative-ai/promptop`). Absent ⇒ a prompt state
   *  fails with that reason. Typed as a plain `Executor`, so the engine never learns that a prompt op
   *  has an llm lowering — dispatch is by OP KIND and nothing more (§4.1). */
  prompt?: Executor<ExecServices, WorkflowMetrics>;
  /**
   * The executor a CALL dispatches through (EXPRESSIONS.md §3).
   *
   * Supplying one is how a call gets the wrapper stack — retry, rate limiting, budget, and a
   * content-addressed `withMemoize`. Without it the engine invokes the registry entry directly,
   * which runs the operation but skips every one of those.
   *
   * Separate from `prompt` because it dispatches BOTH kinds: a call's callee may be either, and the
   * point of the seam is that one composed stack covers both.
   */
  operations?: Executor<ExecServices, WorkflowMetrics>;
  /**
   * What was already paid for, by SCOPED id — the durable half of call answering (Identity and
   * Resume §04).
   *
   * A repeat is answered by identity: a guard re-evaluated on round three computes the same scoped
   * id as its first round (same instance, same site, same content), and this seam is where a LOADED
   * run finds the answer its stopped predecessor recorded. The engine keeps its own in-run map
   * beside it, so within one run a repeat costs nothing whether or not a host supplies this.
   *
   * Sync on purpose: it is read during binding resolution, which cannot suspend (`renderTemplate`
   * resolves inside a `String.replace` callback). Only COMPLETED calls should be answered — a
   * failed record is a retry, not an answer.
   */
  answers?: (scopedId: string) => CallResult | undefined;
  /** SYNC by requirement: slot validation runs mid-walk (`validateSlotValue`) and cannot suspend;
   *  hw schemas are inline documents, so a sync validator is the inline family's truth. */
  validator?: SyncOutputValidator;
  persistence?: Persistence;
  /** Forwarded to runtimes/functions (rate limiter, meter, ...) as their `services`. `validator`/session
   *  store are supplied by the engine. */
  services?: ExecServices;
  /**
   * The conversation store this run's transcripts live in.
   *
   * Config rather than a field on `services`, because it is the ENGINE's dependency and not an
   * executor's: the engine resolves each operation's position from it and hands the executor the
   * position. Nothing below needs the store — a host composing a session layer gives that layer its
   * own copy at construction, which is where the code that forks with it can see it.
   */
  sessions?: SessionStore;
  clock?: Clock;
  /**
   * Mints one instance id per instance entered — UUIDv7 over the engine clock by default.
   *
   * An instance id is DURABLE: minted once, never reused, and meaningless to parse — the id a
   * journal event carries is the id a later load points at, which a per-walk counter could never
   * be. Injectable for the same reason `clock` is: a test that asserts on event payloads needs
   * ids it can predict.
   */
  newInstanceId?: () => string;
  onEvent?: (event: EngineEvent) => void;
  /** Tool-call permissions (DESIGN §5.1, "Permissions: two orthogonal axes"). `approve` collects a human decision on `ask`
   *  (the interactive gate); absent ⇒ a state's tools run UNGUARDED. `baseline` is the workflow-wide default
   *  policy; `process` is the host-owned overlay carrying `always` decisions across runs in one process;
   *  `smart` maps a tool name to its `smart`-mode policy (arg-inspecting; escalates to `ask` when uncertain). */
  permissions?: {
    approve?: Approver;
    baseline?: PermissionBaseline;
    process?: Map<string, PermissionMode>;
    smart?: Record<string, SmartApprover>;
    /** Custom profiles by name (DESIGN §5.1, "Permissions: two orthogonal axes") — a `runtime.permissions.profile`
     *  naming one of these gates tools by it instead of the built-in read-only/plan/full. A
     *  {@link ProfileTable} answers for tools the host never registered; a predicate cannot. */
    profiles?: Record<string, ProfileRule>;
    /** The host's own per-call narrowing — see `ExecPolicy.scopeOf`. */
    scopeOf?: ExecPolicy["scopeOf"];
  };
  /** Per-session workspace resolver (DESIGN §5.1, "Sessions: the run-scoped resource bundle"): maps a `runtime.session` id to the
   *  workspace that session's tools act within, so fan-out branches can isolate (e.g. per-worktree). Returns
   *  `undefined` ⇒ fall back to the single run-level `services.workspace`. Absent ⇒ always the run-level one. */
  workspaceFor?: (sessionId: string) => Workspace | undefined;
}

export interface WorkflowRunOptions {
  inputs: Record<string, ResolvedValue>;
  abortSignal?: AbortSignal;
}

export interface WorkflowRunResult {
  outcome: TerminationOutcome;
  outputs?: Record<string, ResolvedValue>;
  failure?: Failure;
  artifacts: ArtifactRef[];
  metrics: { childLlmCalls: number; childCost: number; durationMs: number };
}

interface TerminationRecord {
  outcome: TerminationOutcome;
  outputs?: Record<string, ResolvedValue>;
  failure?: Failure;
  /** What the state's operation reported, carried up so a parent can read it (SPEC.md §6.1). */
  operation?: OperationNode;
}

interface ChildRecord {
  instanceId: string;
  status: "running" | "done";
  outcome?: TerminationOutcome;
  outputs?: Record<string, ResolvedValue>;
  /** The child's own operation node — what `children.<key>.operation.*` reads (SPEC.md §6.1). */
  operation?: OperationNode;
  /**
   * Why it ended, when it ended badly.
   *
   * Carried on the record rather than left in the termination alone because the parent reports an
   * unhandled child failure from HERE, and "terminated with error" on its own names the outcome
   * without naming the cause — which is the difference between a message someone can act on and one
   * they have to reproduce first.
   */
  failure?: Failure;
  abort: AbortController;
  promise: Promise<void>;
}

/** One state instance (SPEC §3.4) — results never leak across instances. */
interface Instance {
  id: string;
  stateId: string;
  def: LoadedState;
  childKey?: string;
  parent?: Instance;
  inputs: Record<string, ResolvedValue>;
  /** Operation-produced outputs accumulated so far. */
  outputs: Record<string, ResolvedValue>;
  /**
   * The resource bundle this instance's subtree runs in — workspace, permission ledger, and the
   * scope a `"session"` approval covers (DESIGN §5.1).
   *
   * Carried on the INSTANCE rather than recomputed per operation because it is inherited: an
   * operation that declares no session runs in whatever bundle encloses it, all the way up to the
   * run's own. That inheritance is what keeps a worktree and its approvals alive across a retry or
   * a loop iteration, neither of which changes what the author DECLARED (DESIGN.md §5.1).
   */
  resourceKey: string;
  /**
   * What this instance's own operation reported — `operation.*` in an expression (SPEC.md §6.1).
   *
   * Accumulated on the instance rather than derived from the events journal, because an expression
   * cannot read a journal: that inaccessibility is the whole reason this namespace exists.
   */
  operation?: OperationNode;
  /**
   * Where this instance sits in the tree — the key a {@link ReplaySource} is asked with.
   *
   * Carried rather than computed on demand because it is only knowable on the way IN: an occurrence
   * counts entries under one key in one parent, so it has to be taken when the entry happens. After
   * the fact the parent's `children` map holds one record per key and the count is gone.
   */
  address: InstanceAddress;
  /**
   * How many times each child key has been entered under this instance, ever.
   *
   * Distinct from `children`, which holds the LIVE record per key: a sequence reset deletes the
   * entry there, and a loop's iterations would all read as occurrence 0. This one only counts up, so
   * two iterations of one child are two addresses.
   */
  entries: Map<string, number>;
  /**
   * The CALL SITES this instance has dispatched from, by content key — the sequence half of an
   * operation's scope (`ExecServices.scope`).
   *
   * Sequence 0 is reserved for the state's own operation; every call site gets a number of its own
   * on first sight. Keyed by the call's content hash so a re-evaluation at one site — a guard's
   * third round asking the same question — carries the same scope and therefore the same record id,
   * while a different ask never shares one. Per instance, so a loop's next iteration (a new
   * instance) shares nothing.
   */
  sites: Map<string, number>;
  /** The next call-site sequence number — 1-based, 0 being the state's own operation. */
  nextSite: number;
  /**
   * Transitions this instance has taken — EVERY one, forward jumps and the exit included.
   *
   * What `run.iteration` used to count, and what an author almost never meant by it: a forward
   * `{ "when": …, "to": "draft" }` that skips a child spends one, so a `max_iterations` budget
   * written against the old spelling was short by however many jumps the spine happened to take.
   * Kept under the name that says what it is.
   */
  index: number;
  /**
   * PASSES: transitions backward into a sequence member at or before the one most recently
   * entered.
   *
   * The number a re-plan loop is actually written against, and the index into a child's history:
   * one pass is one row across every child, which holds only if the counter moves when the spine
   * goes back and stays put when it goes on.
   */
  iteration: number;
  /** Whether the state's single operation has run (§7.1: a state has ONE operation). */
  opRun: boolean;
  /**
   * Live child records by child key; `undefined`/absent = never ran or superseded.
   *
   * ALWAYS the last entry of {@link passes} — the two are one object, not two copies, so every
   * existing read and write through `children` lands on the current pass without knowing there are
   * others.
   */
  children: Map<string, ChildRecord>;
  /**
   * Every pass, oldest first — `.children.<key>` is the sequence of records this key held, one per
   * pass, and `[-1]` is the live one.
   *
   * A pass begins on a BACKWARD transition (see `Instance.iteration`) by copying the current map, so
   * a child the reset does not clear carries forward BY REFERENCE: index `i` means the same pass for
   * every key, which is what lets a guard compare a draft with the critique that judged it. Before
   * this, the sequence reset deleted the record outright and a loop's earlier passes were
   * unreachable — the wire was authorable and resolved to nothing, every time.
   */
  passes: Array<Map<string, ChildRecord>>;
  /**
   * How far along `sequence` the cursor has moved. A transition into a sequence member is a JUMP: it
   * sets the cursor there, so members BEFORE it stay skipped. Scanning from 0 for the first member
   * with no record instead — which is what this replaced — made a forward jump fall back and run
   * everything it had just jumped over, on the next round.
   */
  cursor: number;
  /**
   * The SYNC child the cursor is currently parked on. The cursor does not advance past it while it
   * runs — that is what `async` means, and the only thing it means.
   *
   * This used to be unset for sequence entries, so the loop entered the next member the instant the
   * previous one had a *record*: every plain child in a sequence ran concurrently, and `async: true`
   * changed nothing anywhere. Ordering came out right only when dataflow happened to park a consumer
   * on its producer.
   */
  heldFor?: string;
  /**
   * Why a transition GUARD refused, when one did.
   *
   * Carried on the instance because `firstMatchingTransition` answers in a vocabulary of outcomes
   * ("take this one", "wait", "nothing matched") that has nowhere to put a sentence.
   */
  guardFailure?: Failure;
  /** The child most recently ENTERED — what `run.cursor` reports to a guard. */
  entered?: string;
  /**
   * Children that finished since the last evaluation round — whose own `transitions` are eligible
   * THIS round and no other (`ChildDecl.transitions`).
   *
   * A list rather than a flag on the record, because eligibility is about the round and not about the
   * child: a record stays `done` forever, so reading the status instead would make an unconditional
   * child transition fire again on every later round, and a state with two such children could never
   * reach its second one.
   */
  justFinished: string[];
  /** Child keys whose error/timeout termination has not yet been handled by a transition. */
  unhandledFailures: Set<string>;
  abort: AbortController;
  timedOut: boolean;
  notify: Notifier;
  /**
   * The deferred calls this instance has demanded, by cache key — what a taken transition CONSUMES.
   *
   * See {@link WorkflowEngine.consumeDeferred}: an event is not a value, so the answer to "did the
   * user drag it" cannot outlive the round that acted on it.
   */
  deferredKeys: Set<string>;
}

/**
 * A DEFERRED call in flight — one whose registered function declared `deferred` and has not settled.
 *
 * The engine STARTS it and reads it later, rather than awaiting it inside the round that needed it.
 * That is the whole difference between the two kinds of embedded call, and it exists because a call
 * that waits on the world may never finish: awaiting one inside a round would hold the round open,
 * and a round holding open is a state that can neither report what it is waiting for nor be woken by
 * anything else that happens to it.
 *
 * Keyed by the same content hash its result will be cached under, so the second round to demand the
 * same call finds it in flight instead of starting a second one.
 */
/**
 * What resolution reports back when it asks for a call's result and there is none.
 *
 * `inFlight` distinguishes the two reasons a call has no result: it has not been started, or it has
 * been started and is waiting. The first is work to do; the second is what the round is waiting FOR.
 */
type CallDemand = (op: Operation<InlineFamily>, key: string, inFlight: boolean) => void;

interface DeferredCall {
  /** The instance whose binding demanded it — what gets woken when it settles, and what cancels it. */
  instance: Instance;
  /** The op as dispatched (arguments bound), for diagnostics and for {@link WorkflowEngine.waitingOn}. */
  op: Operation<InlineFamily>;
  /** Stop it. Settles the call with a `canceled` failure if it had not settled already. */
  cancel: () => Promise<void>;
  /** Resolves when the call has settled and its result is in the cache. Never rejects. */
  settled: Promise<void>;
}

class Notifier {
  private waiters: Array<() => void> = [];
  signal(): void {
    const w = this.waiters;
    this.waiters = [];
    for (const resolve of w) resolve();
  }
  wait(): Promise<void> {
    return new Promise((resolve) => this.waiters.push(resolve));
  }
}

/**
 * A template hole. The leading dot is optional HERE and required by the grammar — deliberately.
 *
 * Recognizing `{{inputs.x}}` and then failing to lower it is what turns an unmigrated hole into a
 * load-time error naming the fix. A regex that demanded the dot would leave the hole unrecognized
 * and render it as literal text into the prompt, which is the same mistake silently.
 */
const TEMPLATE_REF = /\{\{\s*(\.?[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)\s*\}\}/g;


/** One conversation turn — a `ModelMessage`-compatible shape, so the built-in `conversationMode` transcript
 *  and the llm session path share ONE representation in a record's payload. */
type Turn = { role: "user" | "assistant"; content: string };


/**
 * The resource bundle an instance runs in — workspace, permission ledger, approval scope.
 *
 * Only a NAME names a bundle. A session REF arrived through data flow from an operation that may
 * live anywhere in the tree, so it says nothing about which workspace this state should act in; and
 * `null` / absent say nothing about resources at all. In every one of those cases the enclosing
 * instance's bundle is inherited, bottoming out at the run's own.
 *
 * This is what makes the bundle survive replay: a retry and a loop iteration are new instances, but
 * neither changes what the author DECLARED, so both land on the same key their predecessor did.
 */
/**
 * The engine's view of a finished call, for the `operation.*` namespace (SPEC.md §6.1).
 *
 * `usage` is passed through as the measurement record rather than re-shaped, so a metric an executor
 * starts reporting reaches expressions without a second mapping to keep in sync. `cost` is lifted out
 * of it because money is the field everyone asks for by name.
 *
 * `provider` and `attempts` are NOT here yet, deliberately. Neither reaches hw's seam today — they
 * are things the executor knows and does not report — so declaring them would give the lint a field
 * it could never resolve. They arrive with the executor-reported delta (DESIGN.md §1.6), and
 * {@link operationNodeSchema} gains them at the same time, so the type never promises more than the
 * engine fills.
 */
function operationNodeOf(
  outcome: TerminationOutcome,
  metrics: WorkflowMetrics | undefined,
  model: string | undefined,
  session?: PublishedSession,
  /** What the call RETURNED — see {@link operationOutputOf}. */
  returned?: ResolvedValue,
): OperationNode {
  // The END position, not the start: you append AT a position but do not know where the call
  // finished until the provider resolves, and "append after me" / "fork after me" both want the end.
  const output = operationOutputOf(returned, session);
  return {
    outcome,
    ...(metrics !== undefined ? { usage: metrics as unknown as Record<string, JsonValue>, cost: metrics.costUsd ?? 0 } : {}),
    ...(model !== undefined ? { model } : {}),
    ...(output !== undefined ? { output } : {}),
  };
}

/**
 * What `operation.output` holds: the value the call returned, with `session` alongside it.
 *
 * The value ITSELF, not a record built around it — an object return exposes its own properties, an
 * array return IS the array, a blob return is the bytes. This mirrors {@link outputSchemaOf}, and
 * the two have to agree or the type would promise a shape the engine never writes.
 *
 * `session` is attached where the container can hold a property, which is an object or an array —
 * both carry named properties in JS, and the array case is exactly why this is not a plain spread.
 * A scalar return cannot carry one; the author's value wins there rather than being wrapped to make
 * room for the engine's, since wrapping would break every binding that reads the value.
 *
 * Written UNDER a returned value of the same name, for the same reason.
 */
function operationOutputOf(value: ResolvedValue | undefined, session?: PublishedSession): JsonValue | undefined {
  const position = session === undefined ? undefined : ({ id: session.id, end: { id: session.end.id } } as JsonValue);
  if (value === undefined) return position === undefined ? undefined : ({ session: position } as JsonValue);
  if (position === undefined) return value as JsonValue;
  // Not a container — a string, a number, bytes, a live stream. Nothing to hang a property on.
  if (value === null || typeof value !== "object" || value instanceof Uint8Array || isByteStream(value)) {
    return value as JsonValue;
  }
  if (Array.isArray(value)) {
    // A copy, so the returned array is not mutated under whoever else is holding it.
    const withSession = [...value] as unknown as Record<string, JsonValue>;
    if (withSession["session"] === undefined) withSession["session"] = position;
    return withSession as unknown as JsonValue;
  }
  const record = value as Record<string, JsonValue>;
  return { ...(record["session"] === undefined ? { session: position } : {}), ...record } as JsonValue;
}

/** One pass's view of a child, as an expression reads it. `undefined` = it did not run in that pass. */
function passView(rec: ChildRecord | undefined): Record<string, unknown> | undefined {
  if (rec === undefined) return undefined;
  // IN FLIGHT is PENDING and not absence: a consumer of a running child WAITS, where a consumer of
  // one that never ran proceeds without it. Collapsing the two is what silently drops an optional
  // input instead of parking on it.
  if (rec.status === "running") return { output: PENDING, outcome: PENDING, operation: PENDING };
  return { output: rec.outputs ?? {}, outcome: rec.outcome, operation: rec.operation ?? {} };
}

/**
 * A child key as an expression sees it: the array of its passes, which ALSO answers as its last one.
 *
 * `.children.critique[-2].output` reads the pass before this one; `.children.critique.output` reads
 * the current pass, because the live view's own properties are hung on the array. One value serves
 * both spellings, so neither the interpreter nor the producer resolver needs a rule about which is
 * meant — `memberOf` finds an own property, `at` indexes, and the array is the array either way.
 *
 * The same move `operationOutputOf` makes for a prompt op returning a list with `session` on it: a
 * JS array carries named properties, and using that is what keeps the two readings one object rather
 * than two that can drift.
 */
function passesOf(instance: Instance, key: string): unknown {
  const views = instance.passes.map((pass) => passView(pass.get(key)));
  const live = views[views.length - 1];
  const array = views as unknown as Record<string, unknown>;
  // `{}` for a key that has never run, matching what a never-entered child has always read as — so
  // `.children.k.outcome` is `undefined` rather than an error before `k` has been anywhere.
  for (const [name, value] of Object.entries(live ?? {})) array[name] = value;
  return array;
}

/** The model an operation resolved to — read off the config the call was actually made with. */
function modelOfOp(op: Operation<InlineFamily>): string | undefined {
  const config = (op as { config?: unknown }).config;
  if (config === null || typeof config !== "object" || Array.isArray(config)) return undefined;
  const model = (config as { model?: unknown }).model;
  return typeof model === "string" ? model : undefined;
}

/**
 * This instance's address, counting the entry as it takes it.
 *
 * MUTATES the parent's tally, which is the only way an occurrence can be right: it means "how many
 * times this key had been entered before now", and only the moment of entry knows that. Reading it
 * back off `children` later would answer 0 every time, because a sequence reset deletes the record
 * a loop's previous iteration left there.
 */
function addressOf(parent: Instance | undefined, childKey: string | undefined): InstanceAddress {
  if (parent === undefined || childKey === undefined) return [];
  const occurrence = parent.entries.get(childKey) ?? 0;
  parent.entries.set(childKey, occurrence + 1);
  return [...parent.address, { childKey, occurrence }];
}

function resourceKeyFor(def: LoadedState, parent: Instance | undefined): string {
  // `scopeSession` rather than `environment.session`: the latter exists only on a state that declares
  // an operation, and declaring a session on a composite ROOT is the ordinary way to give a whole
  // subtree one bundle.
  const declared = def.scopeSession;
  if (typeof declared === "string" && declared !== "") return declared;
  return parent?.resourceKey ?? RUN_RESOURCE_KEY;
}

export class WorkflowEngine {
  private readonly validator: SyncOutputValidator;
  private readonly clock: Clock;
  private readonly newInstanceId: () => string;
  /** A run-level configuration failure (e.g. a required `function` is not registered):
   *  aborts the whole run rather than looping as a state-level outcome a transition
   *  might keep re-entering. */
  private fatal?: Failure;
  private rootAbort?: AbortController;
  private readonly artifacts: ArtifactRef[] = [];
  /** RUN-SCOPED session store: states sharing a logical `sessionId` continue the same conversation when the
   *  llm executor is composed with `withSession` (opt-in; orthogonal to the built-in `conversationMode`
   *  preamble). Exposed to child executors via `ctx.sessions`; an app-provided store takes precedence. */
  private readonly sessionStore = new MapSessionStore();
  /** Synchronous mirror of every transcript this run has read or written, keyed by session id — the
   *  read side of `{ conversation }` bindings, which resolve synchronously. */
  private readonly transcripts = new Map<string, Turn[]>();
  /** RUN-SCOPED permission ledger (DESIGN §5.1, "Persistence granularity — a scope chain"): owns the session/run overlays and the
   *  authored baseline; the host-owned `process` overlay is injected so `always` decisions cross runs. */
  private readonly permissions: PermissionLedger;
  private childLlmCalls = 0;
  private childCost = 0;

  /**
   * Spend accumulated so far, readable MID-RUN.
   *
   * A run that crashes has usually already paid for the children it completed — a failed call still
   * costs money — so the caller needs the running total rather than a zero from a result it never got.
   */
  spentSoFar(): { childLlmCalls: number; childCost: number } {
    return { childLlmCalls: this.childLlmCalls, childCost: this.childCost };
  }

  constructor(private readonly config: EngineConfig) {
    this.validator = config.validator ?? new SchemaValidator();
    this.clock = config.clock ?? { now: () => Date.now() };
    // Timestamped off `this.clock` rather than `Date.now()`, so an id's time half and the journal's
    // timestamps cannot disagree about when one entry happened under a virtual clock.
    this.newInstanceId = config.newInstanceId ?? (() => uuidv7(this.clock.now()));
    this.permissions = new PermissionLedger({
      // Falling back to the services seam for the same reason `resolveTools` does for the approver:
      // `createWorkflowExecutor` forwards the caller's compiled policy as `services.policy` and never
      // sets `permissions`, so without the fallback the production ledger resolved every mode against
      // an EMPTY baseline while the UI displayed the policy as in force.
      baseline: config.permissions?.baseline ?? config.services?.policy?.baseline,
      process: config.permissions?.process,
    });
  }

  async run(options: WorkflowRunOptions): Promise<WorkflowRunResult> {
    const start = this.clock.now();
    const rootDef = this.config.bundle.states[this.config.bundle.rootId];
    if (!rootDef) throw new Error(`root state '${this.config.bundle.rootId}' missing from bundle`);
    const abort = new AbortController();
    if (options.abortSignal) {
      if (options.abortSignal.aborted) abort.abort();
      else options.abortSignal.addEventListener("abort", () => abort.abort(), { once: true });
    }
    this.rootAbort = abort;
    const inputs = this.resolveRootInputs(rootDef, options.inputs);
    let record: TerminationRecord;
    if ("error" in inputs) {
      record = { outcome: "error", failure: { classification: "permanent", reason: inputs.error } };
    } else {
      record = await this.runInstance(this.config.bundle.rootId, rootDef, inputs.values, abort, undefined, undefined);
    }
    if (this.fatal) {
      record = { outcome: "error", failure: this.fatal };
    }
    return {
      outcome: record.outcome,
      outputs: record.outputs,
      failure: record.failure,
      artifacts: this.artifacts,
      metrics: { childLlmCalls: this.childLlmCalls, childCost: this.childCost, durationMs: this.clock.now() - start },
    };
  }

  /**
   * Continue a STOPPED run from its description — loading, not replaying (Identity and Resume §04).
   *
   * The description is the journal joined to the record store: the log carries the tree, the inputs
   * and the transitions; the records carry the outputs. The machine is CONSTRUCTED from it — every
   * instance keeps its recorded id, every terminated child becomes the record its parent reads, and
   * the evaluation loop is re-entered exactly where the instance's own fields say it stands. Only
   * the active leaves dispatch again: an instance whose operation never honestly settled re-issues
   * it, and because the id and the site are the recorded ones, the scoped record id recomputes
   * identically and the store REOPENS the cut record rather than inserting a second ask.
   *
   * What this is not: a re-walk. Nothing already answered is journaled again (only the live spine
   * re-states its `instance.entered`, so the continuing run's journal can stand on its own), no
   * guard already paid for pays again (`EngineConfig.answers` serves repeats by scoped identity),
   * and the tree keeps the very ids the conversation records point at.
   */
  async loadRun(loaded: LoadedInstance, options: WorkflowRunOptions = { inputs: {} }): Promise<WorkflowRunResult> {
    const start = this.clock.now();
    const abort = new AbortController();
    if (options.abortSignal) {
      if (options.abortSignal.aborted) abort.abort();
      else options.abortSignal.addEventListener("abort", () => abort.abort(), { once: true });
    }
    this.rootAbort = abort;
    const rootDef = this.config.bundle.states[loaded.stateId];
    let record: TerminationRecord;
    if (!rootDef) {
      record = { outcome: "error", failure: { classification: "permanent", reason: `loaded root state '${loaded.stateId}' missing from bundle` } };
    } else {
      record = await this.resumeInstance(loaded, rootDef, abort, undefined);
    }
    if (this.fatal) {
      record = { outcome: "error", failure: this.fatal };
    }
    return {
      outcome: record.outcome,
      outputs: record.outputs,
      failure: record.failure,
      artifacts: this.artifacts,
      metrics: { childLlmCalls: this.childLlmCalls, childCost: this.childCost, durationMs: this.clock.now() - start },
    };
  }

  // --- events ---------------------------------------------------------------

  private emit(event: EngineEvent): void {
    // The JOURNAL is allowed to throw. It is the durable record, and a write that failed silently
    // would leave a replay missing an event it has no way to know it is missing — so the failure
    // travels, and `enterChild` turns it into a run failure with a reason rather than a stall.
    this.config.persistence?.record(event, this.clock.now());
    // An OBSERVER is not. `onEvent` is a tap for whoever is watching; a listener that throws is a
    // bug in the listener, and taking the run down with it would make watching a run change it.
    try {
      this.config.onEvent?.(event);
    } catch {
      // Nothing to report it with that is not itself an observer.
    }
  }

  // --- instance loop --------------------------------------------------------

  private async runInstance(
    stateId: string,
    def: LoadedState,
    inputs: Record<string, ResolvedValue>,
    abort: AbortController,
    childKey: string | undefined,
    parent: Instance | undefined,
    // Minted by the CALLER rather than here, because the caller needs it first: `enterChild` stamps
    // it on the child record before this promise is even constructed, which is what lets a child
    // that CRASHES be terminated under the id it was entered under instead of a sentinel.
    id: string = this.newInstanceId(),
  ): Promise<TerminationRecord> {
    const instance: Instance = {
      id,
      stateId,
      def,
      childKey,
      parent,
      inputs,
      outputs: {},
      // Resolved once, on entry, from the parent's bundle and this state's own declaration — so a
      // subtree that declares nothing shares its enclosing bundle rather than minting one per state.
      resourceKey: resourceKeyFor(def, parent),
      // Taken on the way in, for the reason the field documents: an occurrence is a count of entries
      // and there is nothing left to count once the entry is over. A root has no child key and
      // therefore no step, so the run itself is the empty address.
      address: addressOf(parent, childKey),
      entries: new Map(),
      sites: new Map(),
      nextSite: 1,
      index: 0,
      iteration: 0,
      opRun: false,
      // One pass to start with, and `children` IS it — see the field docs. Assigned below, because
      // the two names have to reach the same Map object.
      children: undefined as unknown as Map<string, ChildRecord>,
      passes: [],
      cursor: 0,
      justFinished: [],
      unhandledFailures: new Set(),
      abort,
      timedOut: false,
      notify: new Notifier(),
      deferredKeys: new Set(),
    };
    // Pass 0, with `children` and `passes[0]` the same Map — the invariant every read depends on.
    instance.children = new Map();
    instance.passes.push(instance.children);
    this.emit({
      type: "instance.entered",
      instanceId: instance.id,
      stateId,
      childKey,
      parentInstanceId: parent?.id,
      inputs: shallowRedactArtifacts(inputs),
    });

    // An input authored with a BINDING (not just a static default, and not wired by the parent)
    // resolves here, once, against the state's own scope — the by-name fill and the parent wire only
    // populate FREE inputs. Without this a bound input, which validation type-checks and fan-out
    // counts as a real consumer, would read as `undefined` everywhere `{ input: … }`/`inputs.*` is used.
    this.resolveInputBindings(instance);

    // SPEC §5 limits.timeout (seconds) → terminate.timeout.
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (def.limits?.timeout !== undefined) {
      timer = setTimeout(() => {
        instance.timedOut = true;
        instance.abort.abort();
        instance.notify.signal();
      }, def.limits.timeout * 1000);
    }

    try {
      const loopRecord = await this.evaluationLoop(instance);
      // Attached HERE, at the one place a record leaves this instance, rather than at each of the
      // dozen `{ outcome: … }` returns inside the loop — every one of which would otherwise have to
      // remember, and a forgotten one is a `children.<key>.operation` that is silently empty.
      const record: TerminationRecord =
        instance.operation !== undefined ? { ...loopRecord, operation: instance.operation } : loopRecord;
      this.emit({
        type: "instance.terminated",
        instanceId: instance.id,
        stateId,
        outcome: record.outcome,
        failure: record.failure,
      });
      return record;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      await this.cancelRunningChildren(instance);
      // A wait outlives nothing. An instance that has terminated — succeeded, failed, been superseded
      // by a sequence reset — must not leave a registration standing, or a person is left looking at
      // a request for a run that no longer exists.
      await this.cancelDeferredCalls(instance);
    }
  }

  // --- loading (Identity and Resume §04) ------------------------------------

  /**
   * One loaded instance as a live `Instance` — the construction `loadRun` is made of.
   *
   * Everything the evaluation loop reads is on the instance, which is what makes loading possible
   * at all: reconstruct the fields and re-enter the loop. The recorded id, sites and entry counts
   * are kept verbatim, because they are what the record ids and addresses were computed from.
   */
  private buildLoadedInstance(loaded: LoadedInstance, def: LoadedState, abort: AbortController, parent: Instance | undefined): Instance {
    const sites = new Map<string, number>((loaded.sites ?? []).map(([key, seq]) => [key, seq]));
    let nextSite = Math.max(1, loaded.nextSite ?? 1);
    for (const seq of sites.values()) nextSite = Math.max(nextSite, seq + 1);
    const entries = new Map<string, number>();
    for (const child of loaded.children ?? []) {
      if (child.childKey === undefined) continue;
      entries.set(child.childKey, Math.max(entries.get(child.childKey) ?? 0, (child.occurrence ?? 0) + 1));
    }
    // `run.cursor` reports the child most recently ENTERED, and the description's children are in
    // entry order — so the last one IS it. Without this a loaded loop reads `run.cursor` as nothing
    // and a guard written against it never fires again.
    const entered = (loaded.children ?? []).at(-1)?.childKey;
    const instance: Instance = {
      id: loaded.id,
      stateId: loaded.stateId,
      def,
      childKey: loaded.childKey,
      parent,
      inputs: loaded.inputs,
      outputs: {},
      resourceKey: resourceKeyFor(def, parent),
      address:
        parent === undefined || loaded.childKey === undefined
          ? []
          : [...parent.address, { childKey: loaded.childKey, occurrence: loaded.occurrence ?? 0 }],
      entries,
      sites,
      nextSite,
      index: loaded.index ?? 0,
      iteration: loaded.iteration ?? 0,
      opRun: false,
      children: undefined as unknown as Map<string, ChildRecord>,
      passes: [],
      cursor: loaded.cursor ?? 0,
      justFinished: [...(loaded.unanswered ?? [])],
      unhandledFailures: new Set(),
      abort,
      timedOut: false,
      notify: new Notifier(),
      deferredKeys: new Set(),
    };
    instance.children = new Map();
    instance.passes.push(instance.children);
    if (entered !== undefined) instance.entered = entered;
    // The COMPLETED operation, fed through exactly the path a live settle takes — the node, then
    // `acceptOpOutputs` — so a loaded state is indistinguishable downstream from one that ran. Spend
    // is deliberately NOT rolled up: the recorded metrics belong to the run that paid them.
    if (loaded.operation !== undefined && def.operation) {
      instance.opRun = true;
      instance.operation = operationNodeOf(
        "success",
        loaded.operation.metrics,
        loaded.operation.model ?? modelOfOp(def.operation),
        publishedOfRef(loaded.operation.sessionRef),
        loaded.operation.value,
      );
      const failure = this.acceptOpOutputs(instance, def.operation.kind === "prompt" ? "prompt" : "function", loaded.operation.value, def.operation.output.kind);
      // A recorded value failing this state's own contract should be unreachable — the definition
      // is pinned — which is exactly why it must be loud rather than smoothed into a re-dispatch.
      // Aborting too, as the other fatal site does: a load standing on a corrupt record must not
      // keep dispatching the states downstream of it.
      if (failure !== undefined) {
        this.fatal ??= { classification: "permanent", reason: `loaded operation of '${loaded.stateId}' no longer satisfies its outputs: ${failure.reason}` };
        this.rootAbort?.abort();
      }
    }
    return instance;
  }

  /**
   * A terminated instance of the stopped run, as the child record its parent reads.
   *
   * Outputs are RECOMPUTED, not loaded: a state's declared outputs are a pure function of its
   * recorded operation value, its children and the pinned definition, and `finish` resolves them
   * the same way it did the first time. Nothing is journaled — this is history, not work.
   */
  private loadTerminated(loaded: LoadedInstance, def: LoadedState, abort: AbortController, parent: Instance | undefined): ChildRecord {
    const instance = this.buildLoadedInstance(loaded, def, abort, parent);
    for (const child of loaded.children ?? []) {
      if (child.childKey === undefined) continue;
      const childDef = this.config.bundle.states[child.stateId];
      if (!childDef) continue;
      // A live child under a terminated parent cannot exist — termination cancels the subtree — so
      // whatever the description says, it is read as history here.
      instance.children.set(child.childKey, this.loadTerminated(child, childDef, abort, instance));
    }
    const term: TerminationRecord =
      loaded.outcome === "success"
        ? this.finish(instance, "success")
        : { outcome: loaded.outcome ?? "error", ...(loaded.failure !== undefined ? { failure: loaded.failure } : {}) };
    return {
      instanceId: loaded.id,
      status: "done",
      outcome: term.outcome,
      ...(term.outputs !== undefined ? { outputs: term.outputs } : {}),
      ...(term.failure ?? loaded.failure ? { failure: term.failure ?? loaded.failure } : {}),
      ...(instance.operation !== undefined ? { operation: instance.operation } : {}),
      abort: new AbortController(),
      promise: Promise.resolve(),
    };
  }

  /**
   * Continue one LIVE instance — `runInstance`'s loaded twin.
   *
   * Its `instance.entered` is re-stated in the continuing journal (with the SAME id), so this run's
   * own log can stand alone; nothing terminated is journaled again. The loop is entered with an
   * evaluation owed exactly when the stopped run owed one: the operation had completed (or the
   * state has none) and no sync child holds the cursor — "guards run on load", which is what
   * re-parks a state on the deferred question it was waiting for.
   */
  private async resumeInstance(loaded: LoadedInstance, def: LoadedState, abort: AbortController, parent: Instance | undefined): Promise<TerminationRecord> {
    const instance = this.buildLoadedInstance(loaded, def, abort, parent);
    this.emit({
      type: "instance.entered",
      instanceId: instance.id,
      stateId: instance.stateId,
      childKey: instance.childKey,
      parentInstanceId: parent?.id,
      inputs: shallowRedactArtifacts(instance.inputs),
    });
    this.resolveInputBindings(instance);

    for (const child of loaded.children ?? []) {
      const key = child.childKey;
      if (key === undefined) continue;
      const childDef = this.config.bundle.states[child.stateId];
      if (!child.live) {
        if (!childDef) continue;
        instance.children.set(key, this.loadTerminated(child, childDef, abort, instance));
        continue;
      }
      // A live child continues exactly the way `enterChild` starts one: its own abort wired to the
      // parent's, its record stamped before its promise exists, its crash a failure and never a stall.
      const childAbort = new AbortController();
      const onParentAbort = (): void => childAbort.abort();
      if (instance.abort.signal.aborted) childAbort.abort();
      else instance.abort.signal.addEventListener("abort", onParentAbort, { once: true });
      const record: ChildRecord = { instanceId: child.id, status: "running", abort: childAbort, promise: Promise.resolve() };
      const run = async (): Promise<void> => {
        let term: TerminationRecord;
        if (!childDef) {
          term = { outcome: "error", failure: { classification: "permanent", reason: `unknown state '${child.stateId}'` } };
        } else {
          term = await this.resumeInstance(child, childDef, childAbort, instance);
          term = await this.materializeFanOut(instance, key, term);
        }
        record.status = "done";
        record.outcome = term.outcome;
        record.outputs = term.outputs;
        record.failure = term.failure;
        record.operation = term.operation;
        if (instance.children.get(key) === record) {
          if (!instance.justFinished.includes(key)) instance.justFinished.push(key);
          if (term.outcome === "error" || term.outcome === "timeout") instance.unhandledFailures.add(key);
        }
        instance.notify.signal();
      };
      instance.children.set(key, record);
      record.promise = run().catch((e: unknown) => {
        const failure: Failure = {
          classification: "permanent",
          reason: `child '${key}' crashed: ${e instanceof Error ? e.message : String(e)}`,
        };
        record.status = "done";
        record.outcome = "error";
        record.failure = failure;
        try {
          this.emit({ type: "instance.terminated", instanceId: record.instanceId, stateId: child.stateId, outcome: "error", failure });
        } catch {
          // Nothing left to report it with.
        }
        if (instance.children.get(key) === record) {
          if (!instance.justFinished.includes(key)) instance.justFinished.push(key);
          instance.unhandledFailures.add(key);
        }
        instance.notify.signal();
      });
      void record.promise.finally(() => {
        instance.abort.signal.removeEventListener("abort", onParentAbort);
      });
      // The cursor holds for a running SYNC child, exactly as it did when the child first entered.
      // `instance.entered` is NOT set here: `buildLoadedInstance` already read it off the last child
      // in entry order, which stays right when a terminated child entered after an async live one.
      const decl = instance.def.children?.[key];
      if (decl !== undefined && decl.async !== true) instance.heldFor = key;
    }
    // A finished child no round ever answered still owes an unhandled-failure mark when it ended badly.
    for (const key of instance.justFinished) {
      const rec = instance.children.get(key);
      if (rec !== undefined && (rec.outcome === "error" || rec.outcome === "timeout")) instance.unhandledFailures.add(key);
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    if (def.limits?.timeout !== undefined) {
      timer = setTimeout(() => {
        instance.timedOut = true;
        instance.abort.abort();
        instance.notify.signal();
      }, def.limits.timeout * 1000);
    }

    const initialEvaluation = (instance.opRun || def.operation === undefined) && instance.heldFor === undefined;
    try {
      const loopRecord = await this.evaluationLoop(instance, initialEvaluation);
      const record: TerminationRecord =
        instance.operation !== undefined ? { ...loopRecord, operation: instance.operation } : loopRecord;
      this.emit({
        type: "instance.terminated",
        instanceId: instance.id,
        stateId: instance.stateId,
        outcome: record.outcome,
        failure: record.failure,
      });
      return record;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      await this.cancelRunningChildren(instance);
      await this.cancelDeferredCalls(instance);
    }
  }

  private async evaluationLoop(instance: Instance, initialEvaluation = false): Promise<TerminationRecord> {
    const def = instance.def;
    // SPEC §3.3: transitions are evaluated when an operation completes or a child
    // terminates — not on bare entry (the first operation runs first) and not when an
    // async child merely starts. One final evaluation runs before success-termination.
    //
    // A LOADED instance may start with an evaluation owed: its operation already completed, or its
    // guards were mid-question when the run stopped — "guards run on load" (Identity and Resume
    // §04), which is what re-parks a state on the deferred call it was waiting for.
    let evaluationDue = initialEvaluation;
    for (;;) {
      if (instance.timedOut) return this.finish(instance, "timeout");
      if (instance.abort.signal.aborted) return this.finish(instance, "canceled");

      // WHICH children this round answers for, fixed BEFORE anything is awaited.
      //
      // A round runs its guards' calls and only then evaluates, so it spans an await — and a child
      // can finish inside that gap. Reading the set again at evaluation time would sweep that child
      // in without its guards ever having been prepared: its calls would be missing, its guard would
      // read PENDING, and the round would then mark it answered. Its completion would be silently
      // lost. So the round takes a SNAPSHOT, answers exactly that, and leaves a late arrival to the
      // next round, which is the guarantee: every completion is evaluated in the first round that
      // starts after it.
      let eligible: readonly string[] = [];

      // A guard may CALL an operation, and evaluation is synchronous — so its calls run here, exactly
      // as an operation's input calls run before `resolveInputs`. The memo means a guard re-evaluated
      // over many rounds pays for its call once.
      if (evaluationDue) {
        eligible = this.finishedInRunOrder(instance);
        const guardFailure = await this.runGuardCalls(instance, eligible);
        if (guardFailure !== undefined) return this.finish(instance, "error", guardFailure);
      }

      // (3)/(4) Transition evaluation, declared order, PENDING-skipping.
      if (evaluationDue) {
        evaluationDue = false;
        // Whatever the cursor was waiting on has resolved by the time an evaluation round runs; if
        // no transition handles it, the cursor is free to walk on from where it stopped.
        instance.heldFor = undefined;
        const step = this.takeTransition(instance, eligible);
        // A rule that cannot be evaluated ends the state. Carrying on would let the next rule answer
        // a question this one was supposed to decide.
        if (step === "guard-failed") return this.finish(instance, "error", guardFailureOf(instance));
        if (step === "terminated-success") return await this.finishSuccess(instance);
        if (step === "terminated-error") return this.finish(instance, "error", errorOf(instance, "terminate.error"));
        if (step === "terminated-canceled") return this.finish(instance, "canceled");
        if (step === "terminated-timeout") return this.finish(instance, "timeout");
        if (step === "waiting") {
          // A guard asked for something and has not been answered. Nothing later in the list may run
          // ahead of that answer, so the round ends here and resumes — from the top of the same list,
          // with the same children still eligible — when the call settles or a child completes.
          if (!(await this.waitForProgress(instance))) {
            return this.finish(instance, "error", {
              classification: "permanent",
              reason: "a transition is waiting on a call that is no longer running",
            });
          }
          evaluationDue = true;
          continue;
        }
        if (step === "entered" || step === "parked") continue;
        // "none": fall through — but a child failure no transition handled is fatal (SPEC §3.3).
        if (instance.unhandledFailures.size > 0) {
          const key = [...instance.unhandledFailures][0]!;
          const rec = instance.children.get(key);
          return this.finish(instance, "error", {
            classification: "permanent",
            reason: unhandledChildReason(key, rec),
          });
        }
      }

      // (2)/(5) The state's operation, then sequence children. A state has ONE operation (§7.1);
      // dispatch is by OP KIND — `PromptOp` → the prompt runner, `FunctionOp` → the function
      // registry (host code, sub-workflows, and delegated agents alike, §3.1).
      // A state whose operation never resolved (§5) is normally stopped by validation before a run
      // starts. If one reaches here anyway, fail loudly — an unbuilt operation looks exactly like a
      // pure composite, and silently doing nothing is the one outcome that must not happen.
      if (def.operationError !== undefined) {
        return this.finish(instance, "error", {
          classification: "permanent",
          reason: `state '${instance.stateId}' has no runnable operation: ${def.operationError}`,
        });
      }
      if (def.operation && !instance.opRun) {
        instance.opRun = true;
        const failure = await this.runOperation(instance, def.operation);
        if (failure) return this.finish(instance, failureOutcome(instance), failure);
        evaluationDue = true;
        continue;
      }

      // Next sequence child = first member AT OR AFTER the cursor with no live record (implements
      // the reset-and-resume cursor, see module header). The cursor bound is what makes a transition
      // into a later member a jump rather than a detour.
      //
      // The cursor HOLDS while the child it points at is a running SYNC child, so the rest of the
      // spine does not start alongside it. `async: true` is the opt-out and the only opt-out —
      // concurrency is a thing an author asks for, not the default a plain sequence falls into.
      const sequence = def.sequence ?? [];
      const held = instance.heldFor !== undefined;
      const nextKey = held ? undefined : sequence.slice(instance.cursor).find((k) => !instance.children.has(k));
      if (nextKey !== undefined) {
        const entered = this.enterChild(instance, nextKey);
        if (entered === "parked") {
          // Dataflow join (SPEC §10.4): wait for a resolution, or deadlock → error.
          if (await this.waitForAnyChild(instance)) {
            evaluationDue = true; // a child completed → evaluation round first
            continue;
          }
          return this.finish(instance, "error", {
            classification: "permanent",
            reason: `child '${nextKey}' is parked on unresolvable inputs (dataflow deadlock)`,
          });
        }
        // An async child falls through without waiting and WITHOUT triggering evaluation
        // (SPEC §10.4), so the cursor moves straight to the next member; a sync child now holds the
        // cursor, and the next pass drops to the running-children branch below, whose wake sets
        // evaluationDue.
        continue;
      }

      // (6) Nothing left to run: wait for running children; when none remain, run one
      // final evaluation round, then terminate.success.
      if (this.hasRunningChildren(instance)) {
        await this.waitForAnyChild(instance);
        evaluationDue = true;
        continue;
      }
      // A child that finished while an earlier round was in flight is still owed one. It gets a REAL
      // round — guard calls and all — rather than the bare check below, which resolves guards without
      // running anything and would park on a call this child's rule has not made yet.
      if (instance.justFinished.length > 0) {
        evaluationDue = true;
        continue;
      }
      const final = this.takeTransition(instance, this.finishedInRunOrder(instance));
      if (final === "guard-failed") return this.finish(instance, "error", guardFailureOf(instance));
      if (final === "waiting") {
        // The state has nothing left to run and would terminate — except that a guard is waiting on
        // an answer, and a state that terminates while a person is being asked where it should go has
        // answered the question itself. So it waits here instead, which is what makes this a state
        // PAUSED on a decision rather than one that quietly succeeded.
        if (await this.waitForProgress(instance)) {
          evaluationDue = true;
          continue;
        }
        return this.finish(instance, "error", {
          classification: "permanent",
          reason: "a transition is waiting on a call that is no longer running",
        });
      }
      if (final === "terminated-success") return await this.finishSuccess(instance);
      if (final === "terminated-error") return this.finish(instance, "error", errorOf(instance, "terminate.error"));
      if (final === "terminated-canceled") return this.finish(instance, "canceled");
      if (final === "terminated-timeout") return this.finish(instance, "timeout");
      if (final === "entered") {
        continue;
      }
      if (final === "parked") {
        // Nothing running, so pending references can never resolve.
        return this.finish(instance, "error", {
          classification: "permanent",
          reason: "transition target parked with no running children (dataflow deadlock)",
        });
      }
      if (instance.unhandledFailures.size > 0) {
        const key = [...instance.unhandledFailures][0]!;
        const rec = instance.children.get(key);
        return this.finish(instance, "error", {
          classification: "permanent",
          reason: unhandledChildReason(key, rec),
        });
      }
      return await this.finishSuccess(instance);
    }
  }

  /** Evaluate transitions once; take the first match (SPEC §3.3 step 3–4). */
  private takeTransition(
    instance: Instance,
    /** The children this round answers for — snapshotted before the round awaited anything. */
    eligible: readonly string[],
  ): "none" | "entered" | "parked" | "waiting" | "guard-failed" | "terminated-success" | "terminated-error" | "terminated-canceled" | "terminated-timeout" {
    const taken = this.firstMatchingTransition(instance, eligible);
    // WAITING is not an answer, so the round consumes nothing: the children this round was to answer
    // for are still owed an answer, and they get it from the round that runs when the call settles.
    // Consuming here would lose their completions entirely, exactly as a park would.
    if (taken === WAITING) return "waiting";
    if (taken === GUARD_FAILED) return "guard-failed";
    // A child's list is eligible for the round its completion triggered and no other, so the round
    // consumes it — whether or not anything matched. Only the SNAPSHOT is consumed: a child that
    // finished while this round was awaiting its guard calls has not been answered by it, and wiping
    // the whole list would lose that completion entirely.
    const consumeEligibility = (): void => {
      const answered = new Set(eligible);
      instance.justFinished = instance.justFinished.filter((key) => !answered.has(key));
    };
    if (!taken) {
      consumeEligibility();
      return "none";
    }
    // A PASS is a step BACKWARD, and the comparison is against the member most recently ENTERED
    // rather than against `cursor`: the cursor sits at 0 before anything has run, which would read
    // the very first entry into `sequence[0]` as a loop back onto itself.
    const isPass = this.isBackwardJump(instance, taken.to);
    // The rule's own wiring, resolved in the world its GUARD read — before the pass opens and
    // before the reset (see `resolveTransitionInputs`), and before ANY of this is spent or
    // journalled. A rule waiting on a value has not fired, so nothing here may act as though it had.
    const handed = this.resolveTransitionInputs(instance, taken.inputRefs);
    if (handed === PENDING) {
      if (!this.hasRunningChildren(instance)) consumeEligibility();
      return "parked";
    }
    instance.index++;
    if (isPass) instance.iteration++;
    this.consumeDeferred(instance);
    this.emit({
      type: "transition.taken",
      instanceId: instance.id,
      stateId: instance.stateId,
      to: taken.to,
      index: instance.index,
      iteration: instance.iteration,
    });
    instance.unhandledFailures.clear(); // a taken transition handles preceding child failures
    if (taken.to.startsWith("terminate.")) {
      consumeEligibility();
      return `terminated-${taken.to.slice("terminate.".length) as TerminationOutcome}` as const;
    }
    // The new pass opens BEFORE the entry, so the sequence reset inside `enterChild` clears members
    // out of the new map and leaves the old one whole. That is the whole mechanism: history survives
    // because the reset is no longer the only copy it could delete from.
    if (isPass) {
      instance.children = new Map(instance.children);
      instance.passes.push(instance.children);
    }
    const entered = this.enterChild(instance, taken.to, handed);
    if (entered === "parked") {
      instance.index--; // the entry did not actually happen
      if (isPass) {
        instance.iteration--;
        instance.passes.pop();
        instance.children = instance.passes[instance.passes.length - 1]!;
      }
      // A park is not an answer, so the eligibility survives — but only while something is still
      // running that could resolve what the target waits on. With nothing running it can never
      // resolve, and holding the eligibility open would spin the loop re-parking forever instead of
      // reporting the dataflow deadlock.
      if (!this.hasRunningChildren(instance)) consumeEligibility();
      return "parked";
    }
    consumeEligibility();
    return "entered";
  }

  /**
   * Whether `to` steps BACK into the sequence — the test that makes a transition a PASS.
   *
   * At or before the member most recently entered, so a child re-entering itself counts: `draft`
   * transitioning to `draft` is the loop everyone means by the word. `terminate.*` and a child
   * outside the sequence are neither forward nor back; they are not passes.
   *
   * Measured against `entered` rather than `cursor` because the cursor is 0 before any child has
   * run, which would read the first ordinary entry into `sequence[0]` as a loop.
   */
  private isBackwardJump(instance: Instance, to: string): boolean {
    const sequence = instance.def.sequence ?? [];
    const target = sequence.indexOf(to);
    if (target < 0 || instance.entered === undefined) return false;
    const from = sequence.indexOf(instance.entered);
    return from >= 0 && target <= from;
  }

  /**
   * The first transition that fires: each just-finished child's own list, then the state's (§3.3).
   *
   * Child lists come FIRST because they are the more specific statement — "when THIS child ends, go
   * there" against "when anything happens, consider this" — and a specific rule that a general one
   * could pre-empt is a rule the author cannot rely on.
   *
   * Two children finishing before the same round are walked in the order the state RUNS them: the
   * sequence, then any child the sequence omits, in declaration order. Nothing about async completion
   * order is stable enough to branch on, so the tie is broken by something an author wrote down.
   */
  private firstMatchingTransition(
    instance: Instance,
    eligible: readonly string[],
    // The RULE that fired travels with the answer, not just its target: a transition may carry its
    // own wiring for the child it enters, and which rule matched is the only thing that knows it.
  ): { to: string; inputRefs?: Record<string, Ref<InlineFamily>> } | typeof WAITING | typeof GUARD_FAILED | undefined {
    // One flag, reset before each guard: whether resolving THIS guard reached a deferred call that is
    // still waiting. That is the difference between the two kinds of PENDING a guard can produce —
    // see the WAITING branch below.
    let deferred = false;
    const scope = this.scopeFor(instance, (_op, _key, inFlight) => {
      if (inFlight) deferred = true;
    });
    const firstOf = (
      transitions: readonly LoadedTransition[] | undefined,
    ): { to: string } | typeof WAITING | typeof GUARD_FAILED | undefined => {
      for (const t of transitions ?? []) {
        // A guard that failed to lower never fires: validation blocks the run, and reading it as
        // unconditional would be the worst possible interpretation of a typo.
        if (t.whenError !== undefined) continue;
        if (t.whenRef === undefined) return { to: t.to, ...(t.inputRefs !== undefined ? { inputRefs: t.inputRefs } : {}) };
        deferred = false;
        const r = resolveRef(t.whenRef, scope);
        if (isPending(r)) {
          /**
           * PENDING ON A DEFERRED CALL STOPS THE LIST — it does not skip.
           *
           * Every other PENDING means "not yet, ask again next round", and skipping is right for it:
           * a guard reading a running child is a question about data, the next transition is a
           * different question, and the round can answer that one meanwhile.
           *
           * A guard waiting on a deferred call is not a question about data. It is a DECISION that
           * has been asked for and not yet made — a person has been shown a drop target, an event has
           * been subscribed to — and every transition after it is a rule about what to do given that
           * decision. Letting a later rule fire while the answer is outstanding would take a branch
           * the author wrote to be considered only if the wait came back false, and would leave a
           * request standing for a state that has already moved on.
           *
           * So the round returns here, the eligibility is NOT consumed (`takeTransition`), and the
           * same list is walked again from the top when the call settles.
           */
          if (deferred) return WAITING;
          continue; // skipped this round (SPEC §6/§10.4)
        }
        // A guard that REFUSED stops the state; it does not quietly fail to match.
        //
        // This used to be impossible, and the comment here said so: every operator's failure case was
        // "producer is missing X" — a malformed tree the loader cannot emit — and reading a missing
        // namespace or property yields `undefined` rather than refusing, so no guard could error on
        // DATA. Comparing against a non-finite number is the first that can (see `resolve.ts`), which
        // means there is now a fourth outcome and it needs somewhere to go.
        //
        // Skipping would reproduce in the engine exactly the bug the refusal exists to stop: a rule
        // that cannot be evaluated would read as a rule that did not apply, and the run would fall
        // through to whatever was written next as though the question had been answered.
        if (isResolveError(r)) {
          instance.guardFailure = r.failure ?? {
            classification: "permanent",
            reason: `a transition guard to '${t.to}' could not be evaluated: ${r.error}`,
          };
          return GUARD_FAILED;
        }
        if (isResolvedValue(r) && r.value) return { to: t.to, ...(t.inputRefs !== undefined ? { inputRefs: t.inputRefs } : {}) };
      }
      return undefined;
    };
    for (const key of eligible) {
      const taken = firstOf(instance.def.children?.[key]?.transitions);
      if (taken) return taken;
    }
    return firstOf(instance.def.transitions);
  }

  /** The children eligible this round, in the order the state runs them — see the caller. */
  private finishedInRunOrder(instance: Instance): string[] {
    if (instance.justFinished.length === 0) return [];
    const eligible = new Set(instance.justFinished);
    const order = [
      ...(instance.def.sequence ?? []),
      // An authored `sequence` may omit children; they still run (by transition) and still finish.
      ...Object.keys(instance.def.children ?? {}).filter((k) => !(instance.def.sequence ?? []).includes(k)),
    ];
    return order.filter((key) => eligible.has(key));
  }

  /**
   * Enter a child, however control got here — the sequence cursor or a transition.
   *
   * A SYNC child holds the cursor until it resolves (SPEC §10.4); an `async` one does not, which is
   * the entire difference between the two and the only place the flag is read.
   */
  private enterChild(instance: Instance, key: string, overrides?: Record<string, ResolvedValue>): "started" | "parked" {
    const decl = instance.def.children?.[key];
    if (!decl) throw new Error(`${instance.stateId}: transition/sequence names undeclared child '${key}'`);

    // Sequence reset (SPEC §3.3): entering a sequence member clears it and every
    // later member; children outside the sequence keep their results.
    const sequence = instance.def.sequence ?? [];
    const seqIndex = sequence.indexOf(key);
    if (seqIndex >= 0) {
      // The entry IS the cursor move — backwards it re-runs the tail, forwards it skips the head.
      instance.cursor = seqIndex;
      for (let i = seqIndex; i < sequence.length; i++) {
        const k = sequence[i]!;
        const rec = instance.children.get(k);
        if (rec) {
          if (rec.status === "running") rec.abort.abort();
          instance.children.delete(k);
          this.emit({ type: "child.superseded", instanceId: instance.id, stateId: instance.stateId, childKey: k });
        }
      }
    }

    // The transition's own wiring, over the mount's, per NAME. Only a TAKEN transition supplies
    // any: the sequence cursor reaches a child by walking, which says nothing about why.
    const resolved = this.resolveChildInputs(instance, decl, overrides);
    if (resolved === PENDING) return "parked";

    instance.entered = key;
    if (decl.async !== true) instance.heldFor = key;

    // Re-entering a child (SPEC §3.4) creates a fresh instance; a stale running
    // instance under the same key is canceled and replaced.
    const prior = instance.children.get(key);
    if (prior?.status === "running") prior.abort.abort();

    const childDef = this.config.bundle.states[decl.state];
    const childAbort = new AbortController();
    /**
     * Cancelling the parent cancels this child — and the wiring is UNHOOKED when the child ends.
     *
     * `{ once: true }` is not a cleanup. It unregisters when the listener FIRES, and on any run that
     * is not cancelled it never fires, so the listener and the `childAbort` it closes over are held
     * by the parent's signal for as long as the parent lives. One per child ENTRY, which for a
     * sequence is one per member and for a loop is one per iteration: a state looping thirty times
     * left thirty dead listeners on a signal that was never going to fire, and Node said so —
     * `MaxListenersExceededWarning: 11 abort listeners added to [AbortSignal]` — at which point the
     * warning is the only thing anyone sees, and it names a symptom rather than this line.
     *
     * The removal is hung off the child's own promise below rather than written into `run()`,
     * because `run()` has two endings — its normal path and the `catch` that turns a crash into a
     * failed child — and a cleanup that only covers one of them is the same bug with better odds.
     */
    const onParentAbort = (): void => childAbort.abort();
    if (instance.abort.signal.aborted) childAbort.abort();
    else instance.abort.signal.addEventListener("abort", onParentAbort, { once: true });

    const record: ChildRecord = {
      // Minted at ENTRY, before the child's promise exists: the crash handler below emits
      // `instance.terminated` from this record, and the id it names must be the id the child was
      // entered under — a child that crashes before assigning anything still has a name.
      instanceId: this.newInstanceId(),
      status: "running",
      abort: childAbort,
      promise: Promise.resolve(),
    };

    const run = async (): Promise<void> => {
      let term: TerminationRecord;
      if (!childDef) {
        term = { outcome: "error", failure: { classification: "permanent", reason: `unknown state '${decl.state}'` } };
      } else if ("error" in resolved && typeof resolved.error === "string") {
        // Input validation failure blocks the state (SPEC §3.3/§4.1); surfaced to the
        // parent as an error termination it can branch on.
        this.emit({
          type: "instance.blocked",
          stateId: decl.state,
          childKey: key,
          parentInstanceId: instance.id,
          reason: resolved.error,
        });
        term = { outcome: "error", failure: { classification: "permanent", reason: resolved.error } };
      } else {
        term = await this.runInstance(decl.state, childDef, resolved.values!, childAbort, key, instance, record.instanceId);
        // Fan-out (§7.3, rule 2) is decided at BIND time: if this producer's blob output feeds two
        // consumers, drain it ONCE here, at the producer's completion, so both siblings read the bytes
        // rather than racing to read one stream. A single-consumer output is left a live stream to pipe.
        term = await this.materializeFanOut(instance, key, term);
      }
      record.status = "done";
      record.outcome = term.outcome;
      record.outputs = term.outputs;
      // Kept alongside the outcome so the parent can say WHY an unhandled failure ended the state.
      // Without it the report is "terminated with error", which names the outcome and loses the one
      // sentence that identifies the slot, the call, or the value actually responsible.
      record.failure = term.failure;
      // The child's operation node, so `children.<key>.operation.*` reads what its call reported —
      // including, for a prompt op, the conversation position it ended at (SPEC.md §6.1).
      record.operation = term.operation;
      // Only while this record is still the live one: a superseded child's completion is not an event
      // its own transitions get to answer, for the same reason its failure is not one the state has to
      // handle — the run has already moved past it.
      if (instance.children.get(key) === record) {
        if (!instance.justFinished.includes(key)) instance.justFinished.push(key);
        if (term.outcome === "error" || term.outcome === "timeout") instance.unhandledFailures.add(key);
      }
      instance.notify.signal();
    };

    instance.children.set(key, record);
    // A THROW anywhere in the child's own execution is a run FAILURE, never a stall.
    //
    // `run()` is deliberately not awaited — that is what lets the parent carry on and a sibling run
    // alongside it — so an exception inside it had nowhere to go: the promise rejected, nothing held
    // a handler for it, and the parent went on waiting for a `notify` that only the normal path ever
    // signals. The run hung with the process idle, the heartbeat still ticking, and the rejection
    // printed to a console nobody was reading.
    //
    // Anything can raise in there: a host's journal callback, a memo key that cannot be canonicalized
    // because a bound argument is NaN, a bug in the engine itself. None of them is a reason to stop
    // answering, and every one of them is a reason to fail this child the way any other permanent
    // failure fails it — so a transition can handle it, and the run ends with a reason attached.
    record.promise = run().catch((e: unknown) => {
      const failure: Failure = {
        classification: "permanent",
        reason: `child '${key}' crashed: ${e instanceof Error ? e.message : String(e)}`,
      };
      record.status = "done";
      record.outcome = "error";
      record.failure = failure;
      // Defensively, because the callback that writes the journal is itself a candidate for having
      // been what threw: the record above is what the parent actually reads, so losing this event
      // costs the trail, not the outcome.
      try {
        this.emit({
          type: "instance.terminated",
          instanceId: record.instanceId,
          stateId: decl.state,
          outcome: "error",
          failure,
        });
      } catch {
        // Nothing left to report it with.
      }
      if (instance.children.get(key) === record) {
        if (!instance.justFinished.includes(key)) instance.justFinished.push(key);
        instance.unhandledFailures.add(key);
      }
      instance.notify.signal();
    });
    // The child is over, however it ended: the parent's signal has nothing left to tell it. Safe
    // because `childAbort` is only reached two ways — through this listener, and directly through
    // `record.abort` for a supersede — and the second does not go through the parent at all. See
    // {@link onParentAbort} for what holding it instead used to cost.
    record.promise = record.promise.finally(() => {
      instance.abort.signal.removeEventListener("abort", onParentAbort);
    });
    return "started";
  }

  /**
   * Wait for anything this instance is waiting ON — a child completing, or a deferred call settling.
   *
   * The two are one wait because they wake the same way (`notify`) and mean the same thing to the
   * loop: something has happened, run another evaluation round. Returns false when there is nothing
   * to wait for, which the caller must treat as a fault rather than as a wait of zero length —
   * looping on it would spin.
   */
  private async waitForProgress(instance: Instance): Promise<boolean> {
    // Both checks are synchronous and nothing is awaited between them and `wait()`, so a call that
    // settles "just now" cannot signal into the gap and be missed.
    if (!this.hasRunningChildren(instance) && this.deferredFor(instance).length === 0) return false;
    await instance.notify.wait();
    return true;
  }

  /** Wait for any child completion signal. Returns false immediately when nothing is running. */
  private async waitForAnyChild(instance: Instance): Promise<boolean> {
    if (!this.hasRunningChildren(instance)) return false;
    const wait = instance.notify.wait();
    await wait;
    return true;
  }

  private hasRunningChildren(instance: Instance): boolean {
    for (const rec of instance.children.values()) if (rec.status === "running") return true;
    return false;
  }

  private async cancelRunningChildren(instance: Instance): Promise<void> {
    const running: Promise<void>[] = [];
    for (const rec of instance.children.values()) {
      if (rec.status === "running") {
        rec.abort.abort();
        running.push(rec.promise);
      }
    }
    await Promise.allSettled(running);
  }

  // --- termination ----------------------------------------------------------

  /**
   * Terminate successfully, having first run any call embedded in an OUTPUT binding.
   *
   * A derived output resolves at termination (SPEC §3.7) and `finish` is synchronous, so a call in
   * one has to be run before it — the same "engine produces, resolution reads" division the
   * operation's own inputs follow, applied at the other place bindings resolve.
   */
  private async finishSuccess(instance: Instance): Promise<TerminationRecord> {
    const failure = await this.runEmbeddedOps(instance, instance.def.outputs ?? {});
    if (failure !== undefined) return { outcome: "error", failure };
    return this.finish(instance, "success");
  }

  private finish(instance: Instance, outcome: TerminationOutcome, failure?: Failure): TerminationRecord {
    if (outcome !== "success") {
      return { outcome, failure };
    }
    // Resolve declared outputs (SPEC §3.7: resolved when the state terminates). An output with a
    // BINDING is derived from it (what `from` used to express); one without is produced by the
    // operation and already accumulated.
    const scope = this.scopeFor(instance);
    const outputs: Record<string, ResolvedValue> = {};
    for (const [name, slot] of Object.entries(instance.def.outputs ?? {})) {
      const meta = instance.def.slotMeta?.[`outputs.${name}`];
      let value: ResolvedValue | undefined;
      // Why a bound output produced nothing, when resolution actually FAILED. Kept rather than
      // discarded: reporting "was not produced" for a binding that named a child which never ran
      // launders the real cause, and §7.3's rule is that a failure carries the underlying one. A
      // `default`/`optional` slot still absorbs it — the reason is only surfaced where the state
      // would otherwise fail anyway.
      let reason: string | undefined;
      if (slot.binding !== undefined) {
        const r = resolveRef(slot.binding, scope);
        // A canceled async child resolves to nothing rather than blocking termination.
        if (isPending(r)) value = undefined;
        else if (isResolveError(r)) reason = r.error;
        else value = r.value;
      } else {
        value = instance.outputs[name];
      }
      if (value === undefined) value = meta?.default;
      if (value === undefined) {
        if (meta?.optional !== true) {
          return {
            outcome: "error",
            failure: {
              classification: "permanent",
              reason: reason !== undefined ? `output '${name}': ${reason}` : `required output '${name}' was not produced`,
            },
          };
        }
        continue;
      }
      // An artifact-typed output carries CONTENT; registering it is what turns that content into a
      // referenceable artifact. `acceptOpOutputs` did this on the way in, which only ever covered an
      // output the operation filled DIRECTLY — so an output that reaches the same value through a
      // binding got a raw string where a produced one got a ref. Now every output binds, so the
      // registration belongs here, where all of them pass.
      if (isArtifactSlot(slot) && typeof value === "string") {
        outputs[name] = this.registerArtifact(instance, name, slot, value);
        continue;
      }
      const err = this.validateSlotValue(name, slot, value);
      if (err) return { outcome: "error", failure: { classification: "permanent", reason: err } };
      outputs[name] = value;
    }
    return { outcome: "success", outputs };
  }

  // --- expression context / resolution scope --------------------------------

  private exprContext(instance: Instance): Record<string, unknown> {
    const children: Record<string, unknown> = {};
    for (const key of Object.keys(instance.def.children ?? {})) {
      children[key] = passesOf(instance, key);
    }
    const artifacts: Record<string, unknown> = {};
    for (const a of this.artifacts) artifacts[a.name] = a;
    return {
      inputs: instance.inputs,
      outputs: instance.outputs,
      // The state's own call as a value (SPEC.md §6.1). `{}` before it has run, so a guard reading
      // `operation.outcome` gets `undefined` rather than throwing — the same shape a never-entered
      // child gets.
      operation: instance.operation ?? {},
      children,
      // `run.cursor` is the child the cursor is ON: the one most recently ENTERED, not the one about
      // to be. Transitions are evaluated after an operation completes or a child terminates, so "we
      // are at x" means x has run — reporting the next member instead would make
      // `run.cursor === 'a'` true before `a` had done anything, and a guard on it would skip the
      // very child it named. Empty string before any child is entered, so a comparison is false
      // rather than an error.
      run: {
        index: instance.index,
        iteration: instance.iteration,
        cursor: instance.entered ?? "",
        position: instance.entered !== undefined ? (instance.def.sequence?.indexOf(instance.entered) ?? -1) : -1,
      },
      limits: { ...(instance.def.limits ?? {}) },
      artifacts,
    };
  }

  /** The run-scoped view binding resolution needs (§7.4) — this instance's data addresses. */
  private scopeFor(instance: Instance, demand?: CallDemand): ResolutionScope {
    return {
      exprContext: this.exprContext(instance),
      // A lowered CALL reads its result here, exactly as a child read reads `childOutputs`:
      // resolution never runs anything, and `undefined` (not yet run) parks the consumer.
      operationResult: (op) => {
        const key = hashOperation(op);
        // The deferred half FIRST, and it is a different half on purpose — see `deferredResults`.
        const hit = this.deferredResults.get(key) ?? this.answerFor(instance, key);
        // A MISS IS THE DEMAND. Resolution asked for a call's result and there is none, so this is
        // where the engine learns which calls the expression it is resolving actually needs — and
        // learning it HERE rather than by walking the tree up front is what makes the demand
        // short-circuit correct: `.inputs.severity > 2 && waits()` never asks the resolver for the
        // right-hand side when the left is false, so the right-hand side never runs. A static walk
        // could not tell the difference, and paid for both.
        if (hit === undefined) demand?.(op, key, this.deferredCalls.has(key));
        return hit;
      },
      childOutputs: (key) => {
        const rec = instance.children.get(key);
        if (!rec) return undefined;
        // A child already run ⇒ REUSE its outputs; that reuse IS findmyprompt's memo semantics,
        // in memory. Still running ⇒ PENDING (the dataflow join parks on it).
        if (rec.status === "running") return PENDING;
        return asJsonRecord(rec.outputs ?? {});
      },
      scopeValue: (name) => {
        const v = instance.inputs[name];
        return v === undefined ? undefined : (v as JsonValue);
      },
      optionalInput: (name) => {
        if (instance.def.inputs?.[name] === undefined) return false;
        const meta = instance.def.slotMeta?.[`inputs.${name}`];
        return meta?.optional === true || meta?.default !== undefined;
      },
      artifact: (name) => {
        const found = this.artifacts.find((a) => a.name === name);
        return found === undefined ? undefined : (found as unknown as JsonValue);
      },
      // A previous conversation read as DATA (§7.5): the whole transcript, or one message of it.
      // Served from the mirror the engine keeps as it reads/writes transcripts — binding resolution
      // is synchronous, and a session this run has not touched has nothing to read anyway.
      conversation: (session, message) => {
        const turns = this.transcripts.get(session);
        if (!turns) return undefined;
        if (message === undefined) return turns as unknown as JsonValue;
        const turn = turns[message];
        return turn === undefined ? undefined : (turn as unknown as JsonValue);
      },
    };
  }

  /**
   * One expression source → its lowered producer tree, cached for the run.
   *
   * Guards and `{{…}}` template holes are the last two places an expression was still INTERPRETED at
   * run time, against a context, while every other expression in the system had become a tree
   * resolved against a scope (EXPRESSIONS.md §1). Two evaluators for one language is exactly the
   * drift that let the interpreter and its own type-checker disagree about prototype properties
   * (§12), so there is now one.
   *
   * Lowering is cached, as parsing was: a guard is re-evaluated every scheduling round.
   */
  private readonly lowered = new Map<string, Ref<InlineFamily>>();
  private exprRef(src: string): Ref<InlineFamily> {
    let ref = this.lowered.get(src);
    if (!ref) {
      ref = lowerExpression(parseExpression(src));
      this.lowered.set(src, ref);
    }
    return ref;
  }

  // --- input resolution -----------------------------------------------------

  private resolveRootInputs(
    def: LoadedState,
    provided: Record<string, ResolvedValue>,
  ): { values: Record<string, ResolvedValue> } | { error: string } {
    const values: Record<string, ResolvedValue> = {};
    for (const [name, slot] of Object.entries(def.inputs ?? {})) {
      const meta = def.slotMeta?.[`inputs.${name}`];
      let v = provided[name];
      if (v === undefined) v = meta?.default;
      if (v === undefined) {
        // A bound input is filled after entry by `resolveInputBindings` (it resolves against the
        // instance's own scope); an optional input may stay unset. Neither is "missing".
        if (slot.binding !== undefined || meta?.optional === true) continue;
        return { error: `required input '${name}' missing` };
      }
      const err = this.validateSlotValue(name, slot, v);
      if (err) return { error: err };
      values[name] = v;
    }
    return { values };
  }

  /**
   * Resolve a child's declared inputs from the parent's wiring (§7.4). Every wire is a base
   * `Ref<InlineFamily>` after loading, so this is one uniform resolution — no expression/literal
   * branch. PENDING ⇒ parked (the dataflow join); `{error}` ⇒ blocked; else the resolved values.
   */
  /**
   * A taken rule's own wiring, resolved in the world its GUARD saw.
   *
   * Called before the pass opens and before the sequence reset, which is the whole point: a rule
   * reads one conversation. `when` asks whether the review said revise and `inputs` hands over what
   * the review found, and those are the same review — so `.children.review.output.findings` means
   * the pass that fired, exactly as the guard's `.children.review.output.verdict` did.
   *
   * Resolving it after the entry instead would have made the CURRENT pass mean the one being
   * abandoned in the guard and the one being started in the wiring, and an author would have had to
   * write `[-2]` in one half of an object whose other half says nothing of the kind.
   *
   * A PENDING here parks the transition, as a child input does: the rule is about a value that has
   * not settled, and taking the branch without it would enter the target with the input missing.
   */
  private resolveTransitionInputs(
    instance: Instance,
    refs: Record<string, Ref<InlineFamily>> | undefined,
  ): Record<string, ResolvedValue> | typeof PENDING | undefined {
    if (refs === undefined) return undefined;
    const scope = this.scopeFor(instance);
    const values: Record<string, ResolvedValue> = {};
    for (const [name, ref] of Object.entries(refs)) {
      const r = resolveRef(ref, scope);
      if (isPending(r)) return PENDING;
      // A resolve ERROR is left for the mount to report against the slot it belongs to: an override
      // that answers nothing falls back, and the child's own requirement is what says whether that
      // is a problem. Reporting here would name the rule for a fault in the value.
      if (isResolveError(r)) continue;
      if (r.value !== undefined) values[name] = r.value;
    }
    return values;
  }

  private resolveChildInputs(
    instance: Instance,
    decl: LoadedChild,
    /**
     * A taken transition's own wiring, ALREADY RESOLVED, which wins per NAME.
     *
     * Values rather than refs because the two halves are resolved at different moments on purpose:
     * the rule's wiring reads the world its guard read, and the mount's reads the world the child
     * is entering. See `TransitionDecl.inputs` and the call site in `takeTransition`.
     */
    overrides?: Record<string, ResolvedValue>,
  ): typeof PENDING | { values?: Record<string, ResolvedValue>; error?: string } {
    const childDef = this.config.bundle.states[decl.state];
    if (!childDef) return { error: `unknown state '${decl.state}'` };
    const scope = this.scopeFor(instance);
    const values: Record<string, ResolvedValue> = {};
    for (const [name, slot] of Object.entries(childDef.inputs ?? {})) {
      const meta = childDef.slotMeta?.[`inputs.${name}`];
      // Per NAME, so a transition restates only what it changes and everything else still comes
      // from the mount — the child's other inputs do not become this transition's problem.
      const handed = overrides?.[name];
      const wire = decl.inputs?.[name];
      let v: ResolvedValue | undefined = handed;
      if (v === undefined && wire !== undefined) {
        const r = resolveRef(wire, scope);
        if (isPending(r)) return PENDING;
        if (isResolveError(r)) return { error: `${decl.state}: input '${name}': ${r.error}` };
        v = r.value;
      }
      if (v === undefined) v = meta?.default;
      if (v === undefined) {
        // A bound input resolves after entry (`resolveInputBindings`); an optional input may stay unset.
        if (slot.binding !== undefined || meta?.optional === true) continue;
        return { error: `${decl.state}: required input '${name}' missing` };
      }
      const err = this.validateSlotValue(name, slot, v, decl.state);
      if (err) return { error: err };
      values[name] = v;
    }
    return { values };
  }

  /**
   * Resolve any input slot that carries a BINDING into `instance.inputs`, so a producer-backed input
   * is not silently `undefined` (the by-name fill and the parent wire populate only FREE inputs).
   * Resolved against the instance's own scope at entry; a value already present — provided by the
   * parent's wire or a static default — wins (it is the more specific value), and a binding that is
   * PENDING or unresolvable at entry leaves the slot unset, the same graceful outcome as an unwired
   * optional input.
   */
  private resolveInputBindings(instance: Instance): void {
    const inputs = instance.def.inputs;
    if (inputs === undefined) return;
    const scope = this.scopeFor(instance);
    for (const [name, slot] of Object.entries(inputs)) {
      if (instance.inputs[name] !== undefined) continue; // already wired or defaulted
      const binding = slot.binding;
      if (binding === undefined) continue;
      const r = resolveRef(binding, scope);
      if (isResolvedValue(r)) instance.inputs[name] = r.value;
    }
  }

  /**
   * Validate a value against a declared slot's schema (tier-3 boundary validation, §4). An
   * ARTIFACT slot is checked structurally (an artifact ref, or inline string content); a slot with
   * no schema constrains nothing.
   */
  private validateSlotValue(name: string, slot: Parameter<InlineFamily>, value: ResolvedValue, statePrefix?: string): string | undefined {
    const label = statePrefix ? `${statePrefix}: '${name}'` : `'${name}'`;
    if (isArtifactSlot(slot)) {
      // BYTES are the canonical form of a blob leaf (§7) — the whole point of the kind. A STREAM over
      // those bytes is equally valid (§7.3): materialization is deferred, so the slot must let the live
      // stream through rather than reject it here and force an eager drain at every boundary. A ref and
      // inline string content are the other two accepted forms.
      if (value instanceof Uint8Array || isByteStream(value)) return undefined;
      if (!isArtifactRef(value) && typeof value !== "string") {
        return `${label} expects an artifact (bytes, a byte stream, an artifact ref, or inline string content)`;
      }
      return undefined;
    }
    // NaN and the infinities are refused HERE, ahead of the schema and whether there is one.
    //
    // JSON Schema cannot do this: `typeof NaN === "number"`, so NaN satisfies `{"type":"number"}` and
    // travels as a well-typed value. It then survives every hop — one state's output is the next
    // one's input — until something needs its CANONICAL form, and canonical JSON (RFC 8785) has no
    // spelling for it. That throw lands wherever the value finally got hashed, naming a memo key
    // rather than the slot that produced a number nothing can represent.
    //
    // The journal makes it worse by hiding it: `JSON.stringify(NaN)` is `null`, so the recorded
    // event shows a plausible null and the trail says nothing happened.
    //
    // So: fail at the boundary the bad value CROSSES, and name the slot that produced it.
    const nonFinite = nonFiniteAt(value);
    if (nonFinite !== undefined) return `${label} is ${nonFinite.what}${nonFinite.path}, which is not a representable JSON number`;
    const schema = slot.schema;
    if (schema === undefined || Object.keys(schema).length === 0) return undefined; // unconstrained
    if (isArtifactRef(value)) return undefined; // an artifact carries its own identity, not the slot's shape
    const res = this.validator.validateValue(schema, value as JsonValue);
    return res.ok ? undefined : `${label} failed validation: ${res.errors ?? "invalid"}`;
  }

  // --- operations -----------------------------------------------------------

  /**
   * Run the state's operation (§7.4): resolve its bindings against the run context, then dispatch
   * the RESOLVED op by kind — a `PromptOp` to `registry.prompt` (the llm leaf runner), a
   * `FunctionOp` to `registry.functions`. Sub-workflows, composite units, and delegated agent
   * runtimes are all FunctionOps; nothing about the op distinguishes them, only the resolved
   * registry entry's capabilities (§3.1). Conversation preambles, sessions, and permission gating
   * attach exactly where they did before — only the payload shape and wiring resolution changed.
   */
  private async runOperation(instance: Instance, op: Operation<InlineFamily>): Promise<Failure | undefined> {
    const kind: OperationKind = op.kind === "prompt" ? "prompt" : "function";
    this.emit({ type: "operation.started", instanceId: instance.id, stateId: instance.stateId, op: kind });
    // `operationId` and `metrics` both arrive from the dispatch paths, and for the same reason: they
    // exist only once the call was actually MADE. See the EngineEvent comment — a pre-dispatch
    // failure has no dispatched op to hash and nothing to measure, and saying so by omission is more
    // honest than reporting zeros for a call that never happened.
    const fail = (failure: Failure, operationId?: string, metrics?: WorkflowMetrics): Failure => {
      this.emit({
        type: "operation.failed",
        instanceId: instance.id,
        stateId: instance.stateId,
        op: kind,
        ...(operationId !== undefined ? { operationId } : {}),
        failure,
        ...(metrics !== undefined ? { metrics } : {}),
      });
      return failure;
    };

    // Resolve every BOUND input; FREE slots are filled by name from the state's own inputs (the
    // model's §3.8 rule). Bound values win, because a binding is what the author wrote on THIS
    // operation while the state's inputs are the general scope it draws from — the "explicit value
    // overrides a binding" rule applies one level up, where a parent wires into a child's inputs
    // (`resolveChildInputs`). A PENDING producer means the operation depends on an async child that
    // has not resolved: a blocked operation rather than a park, since a state's operation runs once.
    // Run any operation EMBEDDED in a binding first — a lowered call (EXPRESSIONS.md §3). Resolution
    // is synchronous and re-run every round, so it reads results rather than producing them; this is
    // the same division of labour a child already has, and the memo below is what stops a guard
    // paying for the same call twice.
    const embeddedFailure = await this.runEmbeddedOps(instance, op.input, op.spread);
    if (embeddedFailure !== undefined) return fail(embeddedFailure);
    const resolved = resolveOperationInputs(op, this.scopeFor(instance));
    if (isPending(resolved)) {
      return fail({ classification: "permanent", reason: "operation inputs depend on a child that has not resolved" });
    }
    if ("error" in resolved) return fail({ classification: "permanent", reason: resolved.error });
    const opInputs: FunctionInputs = { ...instance.inputs, ...resolved.values };

    return op.kind === "prompt"
      ? this.runPromptOp(instance, op, opInputs, fail)
      : this.runFunctionOp(instance, op, opInputs, fail);
  }

  /**
   * The transitions this round may evaluate, IN THE ORDER it will evaluate them — each eligible
   * child's own list, in the order the state runs its children, then the state's own.
   *
   * The same order `firstMatchingTransition` walks, and shared with it deliberately: the round PREPARES
   * guards in one pass and EVALUATES them in another, and two passes that disagreed about the order
   * would prepare a rule the evaluation never reaches.
   *
   * A CHILD's list is included only when that child is eligible this round, which is the same rule that
   * decides whether it is evaluated. A guard may embed a call, and preparing it runs that call — so
   * including every child's list unconditionally would dispatch the call behind "if the review found
   * nothing, summarize" before the review had run, and pay for an answer about data that did not exist
   * yet.
   */
  private orderedTransitions(instance: Instance, eligible: readonly string[]): LoadedTransition[] {
    const out: LoadedTransition[] = [];
    for (const key of eligible) out.push(...(instance.def.children?.[key]?.transitions ?? []));
    out.push(...(instance.def.transitions ?? []));
    return out;
  }

  /** The dispatcher built when the host supplies no {@link EngineConfig.operations}. */
  private ownOperations?: Executor<ExecServices, WorkflowMetrics>;
  /**
   * The executor an operation dispatches THROUGH — always, whether or not a host supplied one.
   *
   * Dispatch by op kind belongs in exactly one place, and `OperationExecutor` is that place
   * (`prompt` → the prompt executor, `function` → a registry lookup). Calling `runFunction` from
   * here instead put a second dispatch site in the engine and, worse, put it BELOW the seam a
   * wrapper composes at: retry, deadline and memoize stopped at the registry boundary and never
   * reached a function op at all. A host that wants those composes them around the dispatcher and
   * supplies the result; absent one the engine builds the plain dispatcher, so there is one code
   * path either way rather than a wrapped path and a raw one.
   *
   * The cast is the one `wiring.ts` already uses at this boundary: `OperationExecutor` is generic in
   * the base `ExecMetrics` while the engine works in `WorkflowMetrics`, which extends it.
   */
  private get operations(): Executor<ExecServices, WorkflowMetrics> {
    if (this.config.operations) return this.config.operations;
    this.ownOperations ??= createOperationExecutor({
      functions: this.config.registry.functions as unknown as FunctionRegistry<ExecServices, ExecMetrics>,
      ...(this.config.prompt !== undefined ? { prompt: this.config.prompt as unknown as Executor } : {}),
    }) as unknown as Executor<ExecServices, WorkflowMetrics>;
    return this.ownOperations;
  }

  /** State operations with their callees already resolved, by state id. */
  private readonly resolvedOps = new Map<string, Operation<InlineFamily>>();
  /**
   * A state's operation with every name in it resolved against the registry — done ONCE per state
   * rather than on every dispatch.
   *
   * The executor can resolve a name itself and always will when nobody did it first, so this is purely
   * a hoist: the registry cannot change mid-run, and a state that runs many times (or an operation
   * whose bindings embed calls re-evaluated each round) would otherwise pay the same lookups over and
   * over.
   *
   * An unresolvable name falls back to the UNRESOLVED operation rather than failing here. Reporting it
   * is the validator's job — it already walks every binding a state carries, guards included — and
   * dispatch produces the run-fatal error with the message it always did. Turning a hoist into a new
   * failure point would make an optimization change behaviour.
   */
  private operationFor(instance: Instance, op: Operation<InlineFamily>): Operation<InlineFamily> {
    const cached = this.resolvedOps.get(instance.stateId);
    if (cached !== undefined) return cached;
    const out = resolveCalls(op, this.config.registry.functions);
    const resolved = "error" in out ? op : out.op;
    this.resolvedOps.set(instance.stateId, resolved);
    return resolved;
  }

  /**
   * Deferred calls in flight, by the cache key their result will land under.
   *
   * Run-wide rather than per-instance because the memo is: two states waiting on the same event with
   * the same arguments are waiting on the same call, and starting it twice would register two
   * interests where the author wrote one.
   */
  private readonly deferredCalls = new Map<string, DeferredCall>();

  /**
   * What deferred calls RETURNED, held apart from the call memo — and dropped when a transition acts
   * on them.
   *
   * An event is not a memo. A call answer says "this is what this callee computes for these
   * arguments", which is stable for the life of a run and may be backed by something durable; "did
   * the user drag this card" is stable for exactly as long as nobody has acted on the answer.
   * Putting one in the other made a state that moved on a drag re-enter its target on every
   * following round — the guard kept reading the same `true` — and would have replayed a person's
   * decision into a resumed run.
   */
  private readonly deferredResults = new Map<string, CallResult>();

  /**
   * Answers to embedded operations, keyed by SCOPED id — this run's own half of what
   * {@link EngineConfig.answers} holds durably (Identity and Resume §04, "a repeat is answered by
   * identity").
   *
   * A call site's key is the resolved op's content hash, so a guard re-evaluated over many rounds
   * asks at the same site with the same content and computes the same scoped id — one execution,
   * however many rounds. The scope is what a LOADED run leans on: its instances keep their recorded
   * ids and sites, so the same id recomputes there and the host's `answers` seam serves what the
   * stopped run already paid for.
   */
  private readonly ownAnswers = new Map<string, CallResult>();

  /** The answer this instance's site already has for this content — in-run first, then the host's. */
  private answerFor(instance: Instance, key: string): CallResult | undefined {
    const sid = scopedOperationId(key, this.callSiteScope(instance, key));
    return this.ownAnswers.get(sid) ?? this.config.answers?.(sid);
  }

  private rememberAnswer(instance: Instance, key: string, outcome: CallResult): void {
    this.ownAnswers.set(scopedOperationId(key, this.callSiteScope(instance, key)), outcome);
  }

  /**
   * Run every operation embedded in these bindings, innermost first, recording each result.
   *
   * Deliberately NOT `runFunctionOp`: that one belongs to a state's own operation — it emits
   * `operation.started` against the instance, reads the environment off `instance.def`, and hands its
   * result to `acceptOpOutputs`, which writes the INSTANCE's outputs. A call has no state, no
   * declared environment of its own, and a result belonging to a binding. It borrows the enclosing
   * instance's environment (its session, tools and permissions are the ones in force where the call
   * is written) and returns its value to the binding, with no identity in the run record.
   */
  private async runEmbeddedOps(
    instance: Instance,
    input: Record<string, Parameter<InlineFamily>>,
    /** A spread's operand is an argument like any other, and may itself be a call (`f(...g())`). */
    spread: readonly Ref<InlineFamily>[] = [],
  ): Promise<Failure | undefined> {
    for (const binding of [...Object.values(input).map((p) => p.binding), ...spread]) {
      if (!binding) continue;
      // HIGHER-ORDER first (§3.5): one application per element, and how many there are is not known
      // until the array resolves — so this cannot be a static walk like `embeddedOpsOf` is.
      const higherFailure = await this.runHigherOrder(instance, binding);
      if (higherFailure !== undefined) return higherFailure;
      for (const { op, parameters } of embeddedOpsOf(binding)) {
        const scope = this.scopeFor(instance);
        // ONE definition of a call's identity, shared with resolution (`resolveEmbedded`): the op
        // with its arguments bound in. Two copies of that rule would hash differently and the memo
        // would never hit.
        const resolved = resolveEmbedded(op, parameters, scope);
        if (isPending(resolved)) {
          return { classification: "permanent", reason: "a call's argument depends on a child that has not resolved" };
        }
        if ("error" in resolved) return { classification: "permanent", reason: resolved.error };

        const key = hashOperation(resolved.op);
        if (this.answerFor(instance, key) !== undefined) continue;
        const outcome = await this.runEmbeddedOp(instance, resolved.op, undefined, key);
        // PENDING is a scheduling state, not an answer — nothing to remember, and nothing a durable
        // cache could serialize.
        if (outcome !== PENDING) this.rememberAnswer(instance, key, outcome);
      }
    }
    return undefined;
  }


  /**
   * Run the calls this round's guards actually NEED — demanded by resolution, in evaluation order,
   * and no further than the rule that decides the round.
   *
   * Two things this is not. It is not a WALK over every guard's binding tree: that ran both sides of
   * every operator, so `.inputs.severity > 2 && confirm()` asked for a confirmation at severity 1 —
   * for a computation a wasted call, and for a call that WAITS a request shown to somebody who should
   * never have seen it. Resolution drives it instead: resolving a guard reports each call whose result
   * is missing, those run, and the guard is resolved again, because a call's value can unlock the next
   * demand as `a() && b()` does.
   *
   * And it is not a pass over ALL the guards. Evaluation stops at the first rule that fires and at the
   * first rule that waits (`firstMatchingTransition`), so every rule behind those two is a rule about a
   * decision this round will not reach. Preparing one would start a wait for a move the engine could
   * not act on if somebody made it — two offers on a board for one decision, one of which does nothing.
   * So this walk stops exactly where the evaluation will:
   *
   *  - an UNCONDITIONAL rule fires ⇒ nothing behind it is prepared;
   *  - a rule whose guard is TRUE fires ⇒ likewise;
   *  - a rule waiting on a deferred call holds the round ⇒ likewise;
   *  - anything else (false, or pending on a running child) ⇒ carry on to the next rule.
   *
   * The cost is that two guards' calls no longer overlap, which is the honest arithmetic: the second
   * one's call is needed only if the first is false. Calls demanded by ONE guard still run together.
   */
  private async runGuardCalls(instance: Instance, eligible: readonly string[]): Promise<Failure | undefined> {
    /**
     * What this round has already run, so a call that answers `PENDING` for a reason of its own — a
     * document-level binding that has not resolved — is not demanded a second time.
     *
     * Termination normally comes from the result landing in the cache or the call registering as in
     * flight. A call that does neither would otherwise be re-demanded on every pass, and because
     * nothing in that loop awaits anything real it would starve the event loop rather than merely
     * spin: no timer would fire, and the run would hang with the process pinned.
     */
    const started = new Set<string>();
    for (const transition of this.orderedTransitions(instance, eligible)) {
      // A guard that failed to lower never fires, so the evaluation skips it and so does this.
      if (transition.whenError !== undefined) continue;
      // Unconditional: it fires, and nothing behind it will be asked anything.
      if (transition.whenRef === undefined) return undefined;
      const binding = transition.whenRef;

      for (;;) {
        const fresh = new Map<string, Operation<InlineFamily>>();
        let waiting = false;
        // HIGHER-ORDER stays eager (§3.5): its applications are demanded one element at a time —
        // `resolveRef` returns PENDING at the first element with no result — so a demand-driven pass
        // would run a `map` over twenty elements in twenty rounds instead of one.
        const higherFailure = await this.runHigherOrder(instance, binding);
        if (higherFailure !== undefined) return higherFailure;
        const resolved = resolveRef(binding, this.scopeFor(instance, (op, key, inFlight) => {
          if (inFlight) waiting = true;
          else if (!started.has(key)) fresh.set(key, op);
        }));

        if (fresh.size === 0) {
          // Nothing left to run for this rule, so its answer is the round's answer about it — and the
          // two kinds of PENDING part company here exactly as they do in the evaluation. Pending on a
          // deferred call HOLDS the round; pending on a running child is skipped, and the next rule
          // gets its turn.
          if (isPending(resolved)) {
            if (waiting) return undefined;
            break;
          }
          if (isResolvedValue(resolved) && resolved.value) return undefined; // it fires
          break; // false, or an error — the next rule gets its turn
        }
        for (const key of fresh.keys()) started.add(key);
        const results = await Promise.all([...fresh].map(async ([key, op]) => [key, await this.startCall(instance, op)] as const));
        // A DEFERRED call answers PENDING here — it has been started, not finished — and there is
        // nothing to remember about a scheduling state.
        for (const [key, outcome] of results) if (outcome !== PENDING) this.rememberAnswer(instance, key, outcome);
      }
    }
    return undefined;
  }

  /**
   * Run one demanded call: awaited here if it computes, started and left running if it waits.
   *
   * The fork is the registered function's own declaration (`HostCapabilities.deferred`) and not a
   * property of where the call appears, so one function behaves the same way in a guard, in an
   * input binding and as a state's whole operation.
   */
  private async startCall(instance: Instance, op: Operation<InlineFamily>): Promise<Resolved> {
    const key = hashOperation(op);
    if (!this.isDeferred(op)) return this.runEmbeddedOp(instance, op, undefined, key);
    const already = this.deferredCalls.get(key);
    if (already !== undefined) return PENDING;

    const callName = op.kind === "function" ? op.functionRef : "prompt";
    let cancel: () => Promise<void> = async () => {};
    const settled = (async () => {
      const outcome = await this.runEmbeddedOp(
        instance,
        op,
        (handle) => {
          cancel = () => handle.cancel();
        },
        key,
      );
      this.deferredCalls.delete(key);
      if (outcome !== PENDING) this.deferredResults.set(key, outcome);
      this.emit({
        type: "call.settled",
        instanceId: instance.id,
        stateId: instance.stateId,
        call: callName,
        operationId: key,
        outcome: outcome !== PENDING && "error" in outcome ? "error" : "value",
      });
      // WAKE THE WAITER, exactly as a child completion does. Without this the loop would sit in
      // `waitForProgress` holding a result nobody had been told about.
      instance.notify.signal();
    })();
    this.deferredCalls.set(key, { instance, op, cancel: () => cancel(), settled });
    instance.deferredKeys.add(key);
    this.emit({ type: "call.waiting", instanceId: instance.id, stateId: instance.stateId, call: callName, operationId: key });
    return PENDING;
  }

  /** Whether a call WAITS rather than computes — its registry entry's own statement (§2). */
  private isDeferred(op: Operation<InlineFamily>): boolean {
    if (op.kind !== "function") return false;
    const entry = this.config.registry.functions.get(op.functionRef);
    // A `pure` entry is deterministic glue over its arguments (§2) and has nothing to wait on, which
    // is why the capability is not on its record to read.
    if (entry === undefined || entry.kind === "pure") return false;
    return entry.capabilities.deferred === true;
  }

  /** The deferred calls this instance is waiting on, oldest first. */
  private deferredFor(instance: Instance): DeferredCall[] {
    return [...this.deferredCalls.values()].filter((c) => c.instance === instance);
  }

  /**
   * A taken transition CONSUMES the waits this instance was holding.
   *
   * Two things go, and both for the same reason — the state has acted, so the question it asked is
   * answered or moot:
   *
   *  - a settled result is FORGOTTEN, or the guard that read it would read the same `true` on every
   *    following round and re-take the same transition forever. The run loop is fast and the guard
   *    would never change its mind, so this is not a slow leak but a spin.
   *  - a call still IN FLIGHT is cancelled. It can only be one the round never reached — an earlier
   *    rule fired first — and an offer nobody withdrew would sit on a person's screen belonging to a
   *    state that has moved on.
   *
   * The next round re-demands whatever is still written down, which registers a FRESH wait. That is
   * what makes a rule like "let them drag it again" mean what it says rather than firing on the memory
   * of the last drag.
   */
  private consumeDeferred(instance: Instance): void {
    for (const key of instance.deferredKeys) {
      this.deferredResults.delete(key);
      const inFlight = this.deferredCalls.get(key);
      // Not awaited: cancellation settles the call, and its settle handler tidies up after itself.
      // Blocking a transition on the teardown of a question nobody is answering would be the wait all
      // over again.
      if (inFlight?.instance === instance) void inFlight.cancel();
    }
    instance.deferredKeys.clear();
  }

  /**
   * Stop every deferred call this instance started, and wait for them to settle.
   *
   * A state that has terminated is not waiting for anything any more, and a registration left behind
   * would keep a request on somebody's screen for a run that has ended.
   */
  private async cancelDeferredCalls(instance: Instance): Promise<void> {
    const waiting = this.deferredFor(instance);
    await Promise.allSettled(waiting.map(async (c) => c.cancel()));
    await Promise.allSettled(waiting.map((c) => c.settled));
  }

  /**
   * Run the per-element applications a higher-order edge needs (§3.5).
   *
   * `Promise.all` over the elements: the expensive case is an LLM call each, and the executor stack
   * already owns rate limiting and budget, so throttling here would be a second, worse copy of it.
   * Each element is keyed and memoized independently, so a re-run pays only for elements whose values
   * changed — and an element's FAILURE is recorded as a result like any other, because a failure is
   * data (§5) and the consuming slot decides what it means.
   */
  private async runHigherOrder(instance: Instance, binding: Ref<InlineFamily>): Promise<Failure | undefined> {
    for (const node of higherOrderEdgesOf(binding)) {
      const higher = higherOrderOf(node);
      if (higher === undefined) continue;
      const source = resolveRef(higher.value, this.scopeFor(instance));
      if (isPending(source)) return { classification: "permanent", reason: `'${higher.name}' waits on a child that has not resolved` };
      if (isResolveError(source)) return { classification: "permanent", reason: `'${higher.name}': ${source.error}` };
      if (!Array.isArray(source.value)) return { classification: "permanent", reason: `'${higher.name}' expects an array` };

      // A FOLD runs in sequence: each step's argument is the previous step's result, so there is
      // nothing to parallelize and nothing to dedupe — the chain is the point.
      if (higher.name === "reduce") {
        const seedBinding = (node as { op: Operation<InlineFamily> }).op;
        const seedParam = seedBinding.kind === "function" ? seedBinding.input.initial?.binding : undefined;
        const seed = seedParam ? resolveRef(seedParam, this.scopeFor(instance)) : { value: null as ResolvedValue };
        if (isPending(seed)) return { classification: "permanent", reason: "'reduce' waits on a child that has not resolved" };
        if (isResolveError(seed)) return { classification: "permanent", reason: `'reduce': ${seed.error}` };
        let acc = seed.value as JsonValue;
        for (const element of source.value) {
          const bound = bindElement(higher.op, element as unknown as JsonValue, acc);
          const key = hashOperation(bound);
          let outcome = this.answerFor(instance, key);
          if (outcome === undefined) {
            const run = await this.runEmbeddedOp(instance, bound, undefined, key);
            if (run === PENDING) return { classification: "permanent", reason: "'reduce' step did not resolve" };
            this.rememberAnswer(instance, key, run);
            outcome = run;
          }
          // A failed step stops the fold — there is no accumulator to carry forward. The failure
          // stays recorded, so resolution reports it rather than re-running.
          if ("error" in outcome) break;
          acc = outcome.value as JsonValue;
        }
        continue;
      }

      // Deduped BY HASH, not just filtered against the cache: two equal elements produce the same
      // key and both miss while neither has run yet, so filtering alone would run the identical
      // application twice — the exact thing the memo exists to prevent.
      const pending = new Map<string, Operation<InlineFamily>>();
      for (const element of source.value) {
        const bound = bindElement(higher.op, element as unknown as JsonValue);
        const key = hashOperation(bound);
        if (this.answerFor(instance, key) === undefined) pending.set(key, bound);
      }
      const results = await Promise.all(
        [...pending].map(async ([key, bound]) => [key, await this.runEmbeddedOp(instance, bound, undefined, key)] as const),
      );
      for (const [key, outcome] of results) if (outcome !== PENDING) this.rememberAnswer(instance, key, outcome);
    }
    return undefined;
  }

  /**
   * Run ONE embedded operation and return what the binding should see.
   *
   * `onHandle` hands the live handle back before the call is awaited — the one thing a DEFERRED call
   * needs that an ordinary one does not, since the only way to stop a wait is to cancel the call that
   * is doing the waiting.
   */
  private async runEmbeddedOp(
    instance: Instance,
    op: Operation<InlineFamily>,
    onHandle?: (handle: { cancel: () => Promise<void> }) => void,
    /** The caller's content key for this call — what makes a re-evaluation share its site. */
    siteKey?: string,
  ): Promise<Resolved> {
    const env = instance.def.environment ?? {};
    const resourceKey = instance.resourceKey;
    // Its arguments are already bound into `op.input` as literals (`resolveEmbedded`), so this reads
    // them back out as values.
    const literal = resolveInputs(op.input, this.scopeFor(instance));
    if (isPending(literal)) return PENDING;
    if ("error" in literal) return { error: literal.error };

    // ONE dispatch, through the same executor a state's operation uses. This used to be three
    // branches — a composed stack when the host wired one, else a prompt path, else a registry call —
    // which is dispatch-by-op-kind written a second time, in the layer above the executor that exists
    // to do exactly that. Collapsing them also settles two ways the branches had drifted apart:
    //
    //  - TOOLS. The composed branch passed `delegates: false` unconditionally, so a delegated agent
    //    reached through a call got policy-gated tools where the same adapter reached as a state's
    //    operation got raw ones and enforced through its own callback. The entry's capabilities decide
    //    it here, as they do everywhere else.
    //  - FREE SLOTS. The registry branch merged `instance.inputs` into the callee's inputs; the
    //    composed branch passed the op's own bindings alone. The op's own bindings win, because the
    //    memo key is `hashOperation` of exactly that op — a merged input the key never saw could
    //    return a result computed under different values. A callee is its own operation with its own
    //    parameters, and a lowered call binds every one of them.
    //
    // The prompt callee keeps its OWN output contract and gets no conversation preamble and no
    // transcript append: a call is a COMPUTATION embedded in a binding, not a turn in the enclosing
    // state's conversation. It still runs under the session in force where it is written, so its tools
    // and permissions are the ones the author expects.
    const entry = op.kind === "function" ? this.config.registry.functions.get(op.functionRef) : undefined;
    if (op.kind === "function" && !entry) return { error: `no function '${op.functionRef}' is registered` };
    const delegates =
      op.kind === "prompt"
        ? this.delegatesPolicy(this.operations, op)
        : entry?.kind === "runtime" && entry.capabilities.policyEnforcement === "callback";
    const toolsOrFailure = this.resolveTools(env, resourceKey, delegates);
    if ("failure" in toolsOrFailure) return { error: toolsOrFailure.failure.reason };
    const rendered = op.kind === "prompt" ? { ...op, user: this.renderTemplate(op.user, instance, literal.values) } : op;
    // A call gets a site of its own — sequence 0 is the state's operation, and a call written into a
    // binding or a guard is a different place in the instance, so the two never share an identity.
    const scope = this.callSiteScope(instance, siteKey);
    let outcome;
    try {
      const handle = this.operations.start(
        rendered,
        await this.servicesFor(resourceKey, instance, toolsOrFailure.tools, undefined, toolsOrFailure.gate, scope, op.kind),
      );
      onHandle?.(handle);
      outcome = await handle.result;
    } catch (e) {
      return { error: `executor rejected: ${(e as Error).message}` };
    }
    this.childLlmCalls += (op.kind === "prompt" ? 1 : 0) + (outcome.metrics.childLlmCalls ?? 0);
    // `?? 0` for the same reason the state path needs it: the dispatcher frames every execution with
    // its own timing and always reports metrics, so an impl that costs nothing arrives without a
    // `costUsd` and adding `undefined` would make the run total NaN.
    this.childCost += outcome.metrics.costUsd ?? 0;
    // A failure travels as DATA (§5): the binding decides whether it flows or terminates.
    return isOk(outcome)
      ? { value: (outcome.value ?? null) as ResolvedValue }
      : { error: outcome.error.reason, failure: outcome.error };
  }

  /** Dispatch a `FunctionOp` through the function registry (§7.4). */
  private async runFunctionOp(
    instance: Instance,
    op: FunctionOp<InlineFamily>,
    opInputs: FunctionInputs,
    fail: (f: Failure, operationId?: string, metrics?: WorkflowMetrics) => Failure,
  ): Promise<Failure | undefined> {
    const entry: RegisteredFunction<ExecServices, WorkflowMetrics> | undefined = this.config.registry.functions.get(op.functionRef);
    if (!entry) {
      // Run-fatal, not a state outcome: a transition could otherwise keep re-entering the state (e.g.
      // §7.3's blocked → human_review) and loop forever.
      const failure = fail({
        classification: "permanent",
        reason: `state '${instance.stateId}' requires function '${op.functionRef}' but no such function is registered`,
      });
      this.fatal = failure;
      this.rootAbort?.abort();
      return failure;
    }

    // The execution ENVIRONMENT (session, tools, permissions) is a sibling of the op, never part of
    // it (§7.1). A delegated adapter enforces policy through its own callback, so its tools stay raw.
    const env = instance.def.environment ?? {};
    // The entry's capabilities are REQUIRED and total per variant (§2), so this reads a definite value
    // instead of falling through an `undefined` and silently defaulting the permission gate.
    const delegates = entry.kind === "runtime" && entry.capabilities.policyEnforcement === "callback";
    /**
     * The CONVERSATION a delegated agent runs in (DESIGN.md §1.6, SESSIONS.md §6).
     *
     * This used to be prompt-only, and the omission was silent in the worst way: an agent adapter
     * reads `ctx.session` to decide between resuming a handle and starting fresh, so with no request
     * stated here every delegated call started a NEW provider conversation while the workflow read as
     * though `session: "review"` had joined them up. Nothing failed; the agent just never remembered.
     *
     * `sessionResume` is the gate because it is the entry's own statement that it HAS a transcript
     * worth placing. A pure host function has none, and minting a conversation for one would put
     * empty records in the store for every helper call.
     */
    let session: SessionBinding | undefined;
    if (entry.kind === "runtime" && entry.capabilities.sessionResume) {
      const resolved = this.sessionFor(instance);
      // A `{ expr }` session that cannot be read is PERMANENT: retrying re-evaluates the same
      // expression against the same data and fails the same way.
      if ("error" in resolved) return fail({ classification: "permanent", reason: resolved.error });
      session = resolved;
    }
    // Tools, workspace and permissions key on the RESOURCE bundle, never on the conversation
    // position — a position moves on every call, and a `"session"`-scoped approval that moved with
    // it would cover exactly one operation (DESIGN.md §5.1). The two agree except when a session was
    // named by an EXPRESSION, where only the resolved binding knows the name it evaluated to.
    const resourceKey = session?.resourceKey ?? instance.resourceKey;
    const toolsOrFailure = this.resolveTools(env, resourceKey, delegates);
    if ("failure" in toolsOrFailure) return fail(toolsOrFailure.failure);

    const scope = this.stateOpScope(instance);
    const services = await this.servicesFor(resourceKey, instance, toolsOrFailure.tools, session, toolsOrFailure.gate, scope, "function");
    // Errors are DATA (§4.2): the impl RESOLVES value-or-failure, so a 429 raised inside a registered
    // function keeps its classification instead of being reconstructed from `err.name` — which is what
    // made every non-`AbortError` permanently failed, retry machinery and all.
    //
    // `bindInputs` writes the resolved inputs onto the op first: the executor reads them off the op
    // and has no view of the instance they were resolved against.
    // Hashed HERE, over exactly the value the executor stack receives, and folded with the same
    // scope the record layer was handed — so the settled events and the record share a key
    // (`scopedOperationId`), each side computing it independently. Undefined when the op cannot be
    // hashed (a live stream input, which `hashOperation` refuses by design).
    const dispatched = bindInputs(this.operationFor(instance, op), opInputs);
    const operationId = tryScopedId(tryHashOperation(dispatched), scope);
    const outcome = await this.operations.start(dispatched, services).result;
    // An impl that reports what it cost (a delegated agent bills inside its own loop) rolls up here,
    // exactly as a prompt op's outcome does — otherwise the spend of the most expensive thing in the
    // graph is the one thing the run's metrics never see. `childLlmCalls` counts LLM calls: a prompt op
    // IS one such call — hence the `1 +` on that path — but a function op is NOT. A pure helper or any
    // non-LLM function makes zero, so an ABSENT count means zero LLM calls, not one that went unreported.
    // (The field was `childCalls` when it came from findmyprompt, where a function only ever ran in
    // service of a call, so a missing count implied 1; here that assumption invents a call that never ran.)
    const metrics = outcome.metrics;
    if (metrics) {
      this.childLlmCalls += metrics.childLlmCalls ?? 0;
      // `?? 0` because the dispatcher ALWAYS reports metrics — it frames every execution with its own
      // `startMs`/`durationMs` — where the impl called directly reported none at all. An impl that
      // costs nothing therefore now arrives as timing without a `costUsd`, and adding `undefined`
      // would turn the run's rollup into NaN. `WorkflowMetrics` types the field as required, which is
      // true of a metrics record an impl BUILDS and not of one the dispatcher frames around it.
      this.childCost += metrics.costUsd ?? 0;
    }
    // The operation NODE carries no `session`, which is the same fact `operationNodeSchema` states in
    // the type: `operation.output.session` is prompt-only, so reaching for it on a `ui` gate is an
    // authoring error rather than a runtime undefined.
    //
    // Not the same claim as "a function op has no conversation" — a delegated agent plainly does, and
    // now runs in one (above). Publishing its END POSITION as a node output is a separate change: the
    // loader has no registry, so it cannot tell a delegated adapter from a host helper, and widening
    // the type for every function op would trade a load-time error for a value that is usually
    // undefined. Wiring one agent's conversation into a later state is done by NAME today.
    instance.operation = operationNodeOf(
      isOk(outcome) ? "success" : "error",
      metrics,
      modelOfOp(op),
      undefined,
      isOk(outcome) ? outcome.value : undefined,
    );
    if (instance.abort.signal.aborted || instance.timedOut) {
      // The CUT is visible now. This used to return with no event at all, so a stopped call's
      // journal ended at `operation.started` and a run parked on a person read exactly like one
      // that hung. The call settled — the executor classifies a cut as `interrupted` and keeps the
      // partial — so the journal says so, and the loop top still owns what happens to the instance.
      this.emitAbortedSettle(instance, "function", operationId, outcome, metrics);
      return undefined; // loop top handles
    }
    if (!isOk(outcome)) return fail(outcome.error, operationId, metrics);
    // The op's declared output KIND decides how its value is read — a `blob` output IS the value
    // (bytes), any other kind is a record of named outputs. Omitting it here left the blob branch
    // unreachable from the function path, so a function op producing a `Uint8Array` failed with "did
    // not produce required output" about the file it had just produced (§7.1).
    const failure = this.acceptOpOutputs(instance, "function", outcome.value, op.output.kind);
    if (failure) return fail(failure, operationId, metrics);
    this.emit({
      type: "operation.completed",
      instanceId: instance.id,
      stateId: instance.stateId,
      op: "function",
      ...(operationId !== undefined ? { operationId } : {}),
      ...(metrics !== undefined ? { metrics } : {}),
    });
    return undefined;
  }

  /** Dispatch a `PromptOp` through the registered prompt runner (§6/§7.4). */
  private async runPromptOp(
    instance: Instance,
    op: PromptOp<InlineFamily>,
    opInputs: FunctionInputs,
    fail: (f: Failure, operationId?: string, metrics?: WorkflowMetrics) => Failure,
  ): Promise<Failure | undefined> {
    const promptExecutor = this.config.prompt;
    if (!promptExecutor) {
      return fail({ classification: "permanent", reason: "this workflow contains a prompt state but no prompt executor is wired in (EngineConfig.prompt)" });
    }

    const env = instance.def.environment ?? {};
    // The two halves one `sessionId` used to be (DESIGN.md §1.6).
    //
    // `session.id` is the CONVERSATION — which transcript this call joins. A declared name joins that
    // stream; `null` and absent each start a fresh one. Absent no longer falls back to a shared
    // "default", because an implicit process-wide transcript is what drove unbounded context growth:
    // the SPEC §4.7 "threads across states" behaviour is now something an author asks for by naming a
    // session, not something every undeclared state opts into.
    //
    // `session.resourceKey` is the RESOURCE BUNDLE — workspace, permission ledger, approval scope. It
    // is inherited from the enclosing instance and does not move when the conversation does, which is
    // what keeps one worktree and one set of approvals across a retry, a loop iteration or a fork.
    const session = this.sessionFor(instance);
    // A `{ expr }` session that cannot be read is PERMANENT: retrying re-evaluates the same
    // expression against the same instance data and fails the same way.
    if ("error" in session) return fail({ classification: "permanent", reason: session.error });

    // The `user` slot holds the prompt text. A REUSABLE prompt is a reference to a file, resolved
    // at load time (REFERENCES.md §7.1), so by here there is only ever one kind of prompt — which is
    // what let `registry.skills` and its half-built resolution path be deleted outright.
    const rendered = this.renderTemplate(op.user, instance, opInputs);
    /**
     * Mirrored for `{ conversation }` bindings (§7.5), which resolve synchronously and cannot await —
     * and for nothing else. The prompt does NOT carry a rendering of the conversation any more.
     *
     * It used to, under `environment.conversation.mode`, and that mode was answering a question the
     * layer below already answers better: `applySession` resumes a provider session, branches one
     * server-side, or replays the turns as messages, choosing by the adapter's own capabilities. The
     * preamble was a second copy of whichever of those had already happened — and because it travelled
     * inside `op.user`, the session layer recorded it as the turn that was asked, so the NEXT preamble
     * rendered it back. One measured run went 8.4k → 403k → 1.13M → 2.55M characters over four passes
     * and died holding three passes of finished work.
     */
    await this.readTranscript(session.id);

    // Whether THIS call's tools stay raw is the answering executor's fact, not the op kind's. A prompt
    // op dispatches on its model prefix, and a `claude-cli/…` route is a delegated agent exactly as a
    // registered adapter is — it runs its own loop and authorizes through its native callback. This
    // used to be hard-coded `false`, which had two silent consequences on an agent-served prompt state:
    // the tools it was handed were permission-WRAPPED (a second gate under the callback that is the
    // gate), and — worse — the {@link ToolGate} was never published at all, so the state's authored
    // modes, its `permissions.profile` and the run's ledger were all invisible to the one enforcement
    // channel a delegated transport has. The agent ran under nothing but its own defaults.
    const toolsOrFailure = this.resolveTools(env, session.resourceKey, this.delegatesPolicy(promptExecutor, op));
    if ("failure" in toolsOrFailure) return fail(toolsOrFailure.failure);
    const tools = toolsOrFailure.tools;

    // The op the runner receives is the authored one with its bindings RESOLVED: the rendered
    // prompt in `user`, and the state's produced outputs as the structured-output contract.
    //
    // A BLOB-kind op output is the exception (§7.1): the operation produces bytes, not a JSON record
    // of named outputs, so overwriting its schema with the object contract would ask a model for JSON
    // and then hand back a file. Its schema is left alone and the bytes fill the single produced slot.
    const producedSlots = this.producedOutputSlots(instance.def);
    const produced = op.output.kind === "blob" ? undefined : buildOutputSchema(producedSlots, instance.def);
    const resolvedOp: PromptOp<InlineFamily> = {
      ...op,
      user: rendered,
      output: { ...op.output, ...(produced !== undefined ? { schema: produced } : {}) },
    };

    // The per-call ENVIRONMENT the old `PromptOpEnvironment` carried — tools, the time budget,
    // cancellation — are `ExecServices` fields now, which is why that type could be deleted outright.
    // The gate rides along exactly as it does on the function path: inert for a composed transport,
    // and the ONLY carrier of authored modes and the session profile for a delegated one.
    const scope = this.stateOpScope(instance);
    const services = await this.servicesFor(session.resourceKey, instance, tools, session, toolsOrFailure.gate, scope, "prompt");
    // An authored `limits.timeout` reaches the call as CANCELLATION. It used to be published as
    // `services.timeoutMs`, which only an executor that knew to read it honoured — and which the llm
    // layer turned straight back into `AbortSignal.timeout(...)` anyway. Folding it into the signal
    // bounds every executor, including ones that read nothing but `abortSignal`.
    if (instance.def.limits?.timeout !== undefined) {
      const bound = AbortSignal.timeout(instance.def.limits.timeout * 1000);
      services.abortSignal = services.abortSignal ? AbortSignal.any([services.abortSignal, bound]) : bound;
    }
    // Hashed over the op the executor stack receives and folded with the dispatch scope — the
    // settled events' join to its record, computed by each side independently.
    const operationId = tryScopedId(tryHashOperation(resolvedOp), scope);
    let outcome;
    try {
      outcome = await promptExecutor.start(resolvedOp, services).result;
    } catch (e) {
      // An executor must not reject for unit failures; a rejection is a bug — normalized here so the
      // workflow still degrades per SPEC §3.3.
      return fail({ classification: "permanent", reason: `prompt executor rejected: ${(e as Error).message}` }, operationId);
    }
    this.childLlmCalls += 1 + (outcome.metrics.childLlmCalls ?? 0);
    this.childCost += outcome.metrics.costUsd;
    // Recorded whether the call SUCCEEDED or not, and before the checks below can return: a failed
    // call is exactly when a guard most wants to read what it cost and where the conversation ended
    // up, and a node written only on the happy path would be missing then.
    const published = this.publish(services.session, session);
    instance.operation = operationNodeOf(
      isOk(outcome) ? "success" : "error",
      outcome.metrics,
      modelOfOp(resolvedOp),
      published,
      isOk(outcome) ? outcome.value : undefined,
    );
    if (instance.abort.signal.aborted || instance.timedOut) {
      // Same as the function path: the cut settles in the journal instead of vanishing after
      // `operation.started`, and the loop top still owns the instance's fate.
      this.emitAbortedSettle(instance, "prompt", operationId, outcome, outcome.metrics);
      return undefined; // loop top handles
    }

    // A FAILED call contributes nothing to the transcript. It ran before this check and a failure
    // carries no `value`, so the assistant turn was the literal string "null" — and under the default
    // `full_history` mode every later state in the session then read that back in its preamble.
    if (!isOk(outcome)) return fail(outcome.error, operationId, outcome.metrics);

    // Conversation artifact (SPEC §4.7): the exchange is already in the session, because the session
    // layer RECORDED the call — one write, not two. The engine used to synthesize a user turn and a
    // stringified assistant turn here, which threw away every tool call and reasoning part in between
    // and is exactly what the append-only model replaced. All that remains is re-reading, so a
    // `{ conversation }` binding in this state's outputs sees what the call just added.
    await this.refreshTranscript(published.id, session.id, published.end.id);

    const failure = this.acceptOpOutputs(instance, "prompt", (outcome.value ?? null) as ResolvedValue, op.output.kind);
    if (failure) return fail(failure, operationId, outcome.metrics);
    this.emit({
      type: "operation.completed",
      instanceId: instance.id,
      stateId: instance.stateId,
      op: "prompt",
      ...(operationId !== undefined ? { operationId } : {}),
      metrics: outcome.metrics,
    });
    return undefined;
  }

  /**
   * Whether the executor that will answer this op enforces policy through its OWN callback — the
   * per-op question `capabilitiesFor` exists to answer.
   *
   * A function op's answer is on its registry entry; a PROMPT op's is on whichever route its model
   * prefix dispatches to, and only the executor tree knows that. One static record standing in for a
   * router whose routes disagree (a provider enforces nothing; a `claude-cli` route enforces by
   * callback) is exactly the degradation {@link Executor.capabilitiesFor} documents — so this reads
   * the per-op answer and falls back to the static record only where no per-op answer exists.
   */
  private delegatesPolicy(executor: Executor<ExecServices, WorkflowMetrics>, op: Operation<InlineFamily>): boolean {
    return (executor.capabilitiesFor?.(op) ?? executor.capabilities).policyEnforcement === "callback";
  }

  /**
   * Resolve the environment's declared tool NAMES through `registry.tools` into executables, and
   * apply the permission gate (DESIGN §5.1, "Enforcement").
   *
   * Two shapes, per the executing entry's `policyEnforcement`:
   *
   *  - COMPOSED (`config`/`none`): every tool is WRAPPED, so a call is authorized by profile × mode
   *    on its way through. The plan-exit gate is injected while the session is in `plan`.
   *  - DELEGATED (`callback`): the tools stay RAW — the adapter runs its own loop and authorizes
   *    through its native callback, and wrapping here as well would double-gate — and a {@link ToolGate}
   *    is published instead. The gate is what carries the authored modes across that boundary. Without
   *    it "raw" meant the state's own `permissions` block went NOWHERE: the adapter read the
   *    workflow-wide baseline only, so a per-tool `deny` or `smart` written on the state was
   *    configured, displayed, and ignored.
   */
  private resolveTools(
    env: ExecEnvironmentDecl,
    sessionId: string,
    delegatesPermissions: boolean,
  ): { tools?: Record<string, Tool>; gate?: ToolGate } | { failure: Failure } {
    const approve = this.config.permissions?.approve;
    /**
     * The GATE's escalation channel, falling back to the services seam.
     *
     * `createWorkflowExecutor` forwards the caller's ctx as `config.services` and never sets
     * `config.permissions` — so reading only `permissions.approve`, as this used to, meant the
     * PRODUCTION path never built a gate at all: a delegated adapter's callback fell through to the
     * bare approver, and every authored mode, `smart` rule and `permissions.profile` on that path
     * was configured, displayed, and ignored.
     *
     * The COMPOSED wrapping below still keys on `permissions.approve` alone, deliberately: a host
     * that publishes an approver on services but no engine-level permissions gets exactly the tool
     * behaviour it always had (raw, self-gating where the tool chooses to), while the gate — which
     * only a delegated adapter consults — now exists to carry the profile and the modes across.
     */
    const escalate = approve ?? this.config.services?.approve;
    // Seeded whichever way the policy is enforced. It used to happen on the wrapping path only, so a
    // DELEGATED state authoring `profile: "read-only"` ran under `full` — the ledger's default —
    // before any of the rest of this had a chance to matter.
    if (env.permissions?.profile) this.permissions.seedProfile(sessionId, env.permissions.profile);

    const named = env.tools ?? [];
    const tools: Record<string, Tool> = {};
    for (const name of named) {
      const tool = this.config.registry.tools.get(name);
      if (!tool) return { failure: { classification: "permanent", reason: `tool '${name}' is not registered` } };
      tools[name] = tool;
    }
    const some = Object.keys(tools).length > 0;
    if (!escalate) return some ? { tools } : {};

    /**
     * The gate, built for EVERY enforcement style rather than only the delegated one.
     *
     * A composed runtime never calls it — its tools are wrapped below and the wrapping is the gate —
     * so publishing it there is inert. What it is not is useless: `ToolGate.profile` is the only way
     * the session's profile reaches an adapter at all, and `codex exec` acts on the profile and on
     * nothing else. Withholding it from `config` adapters would keep the one transport that has no
     * per-tool gate from honouring the one thing it CAN honour.
     *
     * Built even with NO declared tools, because a delegated agent's own built-ins are the calls that
     * most need answering for — they are the ones we never registered and cannot wrap.
     */
    const smart = this.config.permissions?.smart ?? this.config.services?.policy?.smart;
    const customProfiles = this.config.permissions?.profiles ?? this.config.services?.policy?.profiles;
    // The host's own per-call narrowing (a path sandbox, a URL allow-list), read off the policy the
    // same way `smart` and `profiles` are. It composes as a narrowing inside `decideToolCall`, so a
    // policy that supplies none changes nothing.
    const scopeOf = this.config.permissions?.scopeOf ?? this.config.services?.policy?.scopeOf;
    const gate = createToolGate({
      ledger: this.permissions,
      sessionId,
      approve: escalate,
      tools: Object.fromEntries(Object.entries(tools).map(([name, tool]) => [name, { readOnly: tool.readOnly }])),
      ...(env.permissions !== undefined ? { authored: env.permissions } : {}),
      ...(smart !== undefined ? { smart } : {}),
      ...(customProfiles !== undefined ? { profiles: customProfiles } : {}),
      ...(scopeOf !== undefined ? { scopeOf } : {}),
    });

    if (delegatesPermissions) return { ...(some ? { tools } : {}), gate };
    // Composed, but the host wired no engine-level approver: the tools stay RAW — the documented
    // "without an approver, tools are handed over unguarded" rule, unchanged — and the gate still
    // travels for anything that reads only it.
    if (!approve) return { ...(some ? { tools } : {}), gate };
    if (!some) return { gate };
    // OWN entries only. These are authored/host-supplied maps keyed by TOOL NAME, so a tool called
    // `constructor` or `toString` used to resolve its permission mode — and its smart-approval rule
    // — to a prototype member, handing a FUNCTION to a permission decision. Far-fetched input, but
    // "does it fail open?" is not a question worth leaving open on this path.
    const authoredTools = env.permissions?.tools;
    const authoredMode = (name: string): PermissionMode | undefined =>
      (authoredTools !== undefined && Object.hasOwn(authoredTools, name) ? authoredTools[name] : undefined) ?? env.permissions?.default;
    const guarded: Record<string, Tool> = {};
    for (const [name, tool] of Object.entries(tools)) {
      guarded[name] = withPermission(tool, {
        ledger: this.permissions,
        sessionId,
        toolName: name,
        approve,
        authoredMode: authoredMode(name),
        smart: smart !== undefined && Object.hasOwn(smart, name) ? smart[name] : undefined,
        profiles: customProfiles,
        // The SAME narrowing the gate applies — a wrapped tool and a delegated one are two routes to
        // one decision, and binding only one would be a sandbox with a door in it.
        scopeOf,
        // This state's own authored block, so a host layering a floor under a per-state table can
        // see both at the moment of decision.
        ...(env.permissions !== undefined ? { authored: env.permissions } : {}),
      });
    }
    if (this.permissions.resolveProfile(sessionId) === "plan") {
      guarded["exit_plan"] = planExitTool({ ledger: this.permissions, sessionId, approve });
    }
    return { tools: guarded, gate };
  }

  /** The services one operation runs with: its resource bundle's workspace, its tools, its cancellation. */
  /**
   * The settle event for a call that was running when the run was cut — emitted only when the call
   * HONESTLY settled.
   *
   * A completed answer that raced the abort is still a completion, and an `interrupted` failure is
   * the transport saying the turn was cut with its partial kept — both are facts worth a journal
   * line, where the old path ended every cut call's trail at `operation.started`. A `canceled`
   * failure is different: it is the abort's own unwinding reported back, nothing about the call
   * settled, and staying silent keeps the "started and never settled" signature the frontier reads.
   */
  private emitAbortedSettle(
    instance: Instance,
    op: OperationKind,
    operationId: string | undefined,
    outcome: { value?: unknown; error?: Failure },
    metrics: WorkflowMetrics | undefined,
  ): void {
    if (isOk(outcome as never)) {
      this.emit({
        type: "operation.completed",
        instanceId: instance.id,
        stateId: instance.stateId,
        op,
        ...(operationId !== undefined ? { operationId } : {}),
        ...(metrics !== undefined ? { metrics } : {}),
      });
      return;
    }
    const failure = (outcome as { error: Failure }).error;
    if (failure.classification !== "interrupted") return;
    this.emit({
      type: "operation.failed",
      instanceId: instance.id,
      stateId: instance.stateId,
      op,
      ...(operationId !== undefined ? { operationId } : {}),
      failure,
      ...(metrics !== undefined ? { metrics } : {}),
    });
  }

  /** The scope of the state's own operation — site 0, by definition (one op per state, SPEC §7.1). */
  private stateOpScope(instance: Instance): OperationScope {
    return { instanceId: instance.id, sequence: 0 };
  }

  /**
   * The scope of a CALL made from inside this instance — a guard, an embedded op, a deferred wait.
   *
   * Keyed by the call's content hash, assigned on first sight: a re-evaluation at one site (a
   * guard's third round asking the same question) reuses its number and therefore its record
   * identity, while a different ask never shares one. An unkeyable call (a live stream input, which
   * `hashOperation` refuses) gets a fresh anonymous site — it names this dispatch and claims nothing
   * about content, which is exactly true of a stream.
   */
  private callSiteScope(instance: Instance, siteKey: string | undefined): OperationScope {
    if (siteKey === undefined) return { instanceId: instance.id, sequence: instance.nextSite++ };
    let sequence = instance.sites.get(siteKey);
    if (sequence === undefined) {
      sequence = instance.nextSite++;
      instance.sites.set(siteKey, sequence);
    }
    return { instanceId: instance.id, sequence };
  }

  private async servicesFor(
    resourceKey: string,
    instance: Instance,
    tools?: Record<string, Tool>,
    session?: SessionBinding,
    /** The delegated permission gate, when this call has one — see {@link resolveTools}. */
    gate?: ToolGate,
    /** The dispatch site this call is made from — with `opKind`, arms the record layer's seam. */
    scope?: OperationScope,
    opKind?: OperationKind,
  ): Promise<ExecServices> {
    const services = this.childServices();
    if (gate !== undefined) services.gate = gate;
    // The dispatch scope and its callback — set or CLEARED unconditionally, because `childServices`
    // copies the host-provided bundle, and for a sub-workflow that bundle IS the parent dispatch's
    // ctx: an inherited scope would stamp the parent's site onto every nested call.
    if (scope !== undefined && opKind !== undefined) {
      services.scope = scope;
      // hw's half of the dispatch seam: the record layer fires this AFTER the row is inserted and
      // BEFORE the provider call, so this event never names a row that was not written.
      services.onDispatch = (dispatch): void =>
        this.emit({
          type: "operation.dispatched",
          instanceId: instance.id,
          stateId: instance.stateId,
          op: opKind,
          operationId: dispatch.id,
        });
    } else {
      delete services.scope;
      delete services.onDispatch;
    }
    // RESOLVED HERE, not stated as a request for a layer below to resolve.
    //
    // The engine used to publish `ctx.sessionRequest` — which conversation, and whether to branch —
    // and leave the lookup to a composed layer, on the principle that a requester must not claim to
    // know where a conversation currently sits. The principle holds; what changed is that resolving
    // IMMEDIATELY BEFORE dispatch is not claiming anything: nothing is stored, nothing is guessed, and
    // the answer is a frame old rather than a stack-depth old.
    //
    // What it buys is that the executor is handed a POSITION — a provider handle and an append/fork
    // decision, which are the only session facts a prompt call actually consumes — instead of a
    // request it has no use for. A services bundle should carry what the executor needs to make the
    // call, and `sessionRequest` never met that test.
    const sessions = this.sessions();
    if (session !== undefined && sessions !== undefined) {
      services.session = await sessions.resolve({
        ref: session.id,
        ...(session.fork ? { fork: true } : {}),
        // Stable across replays, so a re-run lands on the conversation it landed on before rather
        // than minting a second one beside it (DESIGN.md §5.1).
        seed: `${instance.stateId}:${session.id}`,
      });
    }
    // Per-BUNDLE workspace (DESIGN §5.1, "Sessions: the run-scoped resource bundle"): states sharing a
    // resource key share one; a fan-out can isolate each branch (e.g. its own worktree) via
    // `workspaceFor`. Falls back to the run-level `services.workspace` when the host provides none.
    //
    // Keyed on the resource bundle rather than the conversation, deliberately: a conversation position
    // changes on every call, so keying a workspace on it would hand each operation its own worktree —
    // which §5.1 rules out: forking branches the CONVERSATION, not the filesystem.
    const workspace = this.config.workspaceFor?.(resourceKey) ?? services.workspace;
    if (workspace !== services.workspace) services.workspace = workspace;
    if (tools !== undefined) services.tools = tools;
    // A registered async function's only channel to the caller is the ctx, so cancellation rides here.
    services.abortSignal = instance.abort.signal;
    return services;
  }

  /**
   * What one operation's `session` declaration resolves to for THIS instance (DESIGN.md §1.6).
   *
   * The four-way order in the document collapses to the three cases {@link resolveSession} handles,
   * because the environment merge has already run: an ancestor's `environment.session` arrives as
   * this operation's own `session`, and a nearer `null` has already beaten it.
   *
   * `positionOf` is the seam the instance-scoped invariant (DESIGN.md §1.6) lives behind. It answers
   * "where does THIS instance think that named stream currently is", and it must never become a
   * global name → head map: a restarted state has to re-resolve to the position it started from, so
   * that the append the failed attempt made turns the retry into a fork from the right place rather
   * than stacking it on top of the failure. Until positions exist (step 5), a name IS its own
   * position and every instance agrees — which is exactly today's behaviour, and the reason this step
   * can land before the store does.
   */
  private sessionFor(instance: Instance): SessionBinding | { error: string } {
    const env = instance.def.environment ?? {};
    let declared = env.session as SessionDecl | undefined;
    // The `{ expr }` spelling is the only one evaluated rather than read, and it has to be, because
    // a ref is a RUN-TIME value: `children.plan.operation.output.session` does not exist until
    // `plan` has run, so a static field could never carry one. Evaluated against THIS instance, so
    // a re-entered or looped state re-reads the position its own attempt should continue from.
    if (isSessionExpr(declared)) {
      const { expr } = declared;
      const resolved = resolveRef(this.exprRef(expr), this.scopeFor(instance));
      // PENDING means the producing operation is still in flight. That is a wiring mistake rather
      // than something to wait on here: the consumer's own dataflow join is what parks on a running
      // producer, and by the time an operation is being dispatched its inputs have settled.
      if (isPending(resolved)) return { error: `session expression '${expr}' reads an operation that has not finished` };
      if (isResolveError(resolved)) return { error: `session expression '${expr}': ${resolved.error}` };
      const outcome = sessionFromExpr(expr, resolved.value);
      if ("error" in outcome) return outcome;
      declared = outcome.session;
    }
    return resolveSession(declared, env.fork === true, {
      instanceId: instance.id,
      inheritedResourceKey: instance.resourceKey,
      positionOf: () => undefined,
    });
  }

  /**
   * What a finished call publishes as `operation.output.session` — see {@link PublishedSession}.
   *
   * Built from what the STORE resolved, never from the declaration: the declaration says which
   * conversation to join, and only the resolution knows where that landed — including when a taken
   * position turned the call into a fork, where the branch is the conversation a later state must
   * continue and the declared name would send it back to the trunk.
   *
   * The position published is the one AFTER this call's own, because a reader wanting "carry on from
   * here" must not land on the slot this turn occupies — continuing there would see it taken and fork,
   * turning every hand-off into a branch. Derived from `at` rather than by asking for the head, so it
   * is exact whether or not a session layer is composed to write the record.
   *
   * With no resolved session — a run whose store minted nothing for this call — both halves fall back
   * to the declared id, which is what this published before positions existed at all.
   *
   * And a store that cannot spell the position falls back to the conversation id, because THIS IS
   * BOOKKEEPING ABOUT A CALL THAT ALREADY RAN. A host supplies its own store, built against whatever
   * version of the contract it last compiled against, so `refAt` is a method that may simply not be
   * there — as it was not the day it was added, where the throw failed the run permanently at the end
   * of its first completed call and discarded the answer with it. An unpositioned id is a legitimate
   * ref rather than a fudge: every store must accept one as naming that conversation AT ITS HEAD,
   * which for a call that just settled is the slot after it — the same place, resolved later.
   */
  private publish(resolved: ExecServices["session"], declared: SessionBinding): PublishedSession {
    if (resolved === undefined) return publishedSession(declared.id, declared.id);
    try {
      return publishedSession(this.sessions().refAt({ id: resolved.at.id, seq: resolved.at.seq + 1 }), resolved.at.id);
    } catch {
      return publishedSession(resolved.at.id, resolved.at.id);
    }
  }

  /** The `ExecServices` operations run with: caller services + engine validator + the run's session
   *  store — the SAME store the built-in transcript uses, so `withSession` and the preamble share one
   *  source (states sharing a logical `sessionId` continue one conversation; an app store wins). */
  private childServices(): ExecServices {
    return {
      ...this.config.services,
      validator: this.validator,
      // A delegated adapter reads this to route its native permission callback through our approval
      // UI; the engine wraps a composed runtime's tools directly, so this is inert for a prompt op.
      // `approve` is `@declarative-ai/permissions`' seam on `ExecServices` — `exec` does not know it
      // exists (DESIGN §3.2).
      approve: this.config.permissions?.approve ?? this.config.services?.approve,
    };
  }

  /**
   * Merge an operation's structured result into the instance outputs (SPEC §3.3 step 3:
   * "its outputs are validated"). Provided fields are type-checked; artifact-typed
   * fields arrive as inline content and are registered as artifacts. Once the state's
   * operation has run, required produced fields must all be present.
   */
  private acceptOpOutputs(instance: Instance, op: OperationKind, value: ResolvedValue, outputKind?: RefKind): Failure | undefined {
    const produced = this.producedOutputSlots(instance.def);
    // Nothing to distribute. Every output binds — from `.operation.output`, a child, an expression —
    // so the call's result reaches them through the operation NODE rather than by being poured into
    // slots here. Returning early also keeps the blob rule below from firing on a state that
    // declares no produced slot at all, which is now every state.
    if (Object.keys(produced).length === 0) return undefined;

    // A BLOB-kind operation output is the WHOLE value, not a record of named outputs (§7.1) — the
    // bytes go straight into the state's single produced slot. Without this the `Uint8Array` fell
    // through the record path as an empty object and the state failed with "did not produce required
    // output", which is exactly what a generated file DID produce.
    //
    // A blob output that is a live STREAM is stored here AS a stream, deliberately NOT drained: one
    // downstream consumer can then pipe it un-materialized (§7.4). The drain, when required, happens
    // where it is DECIDED — a fan-out at the producer's completion, the run result at the executor
    // boundary, a memo key before hashing — never eagerly at every op that produces bytes.
    if (outputKind === "blob") {
      const names = Object.keys(produced);
      if (names.length !== 1) {
        return {
          classification: "permanent",
          reason: `state '${instance.stateId}' has a blob operation output, which fills exactly ONE produced output slot, but the state declares ${names.length}`,
        };
      }
      const name = names[0]!;
      const slot = produced[name]!;
      if (isArtifactSlot(slot) && typeof value === "string") {
        instance.outputs[name] = this.registerArtifact(instance, name, slot, value);
      } else {
        instance.outputs[name] = value;
      }
      return undefined;
    }

    // A whole-value stream is NOT a record of named outputs — the `getReader`-bearing object would
    // otherwise be walked as one (typeof "object", not array, not `Uint8Array`) and yield an empty
    // record. Guarded so a stream only ever reaches the blob branch above.
    const record: Record<string, ResolvedValue> =
      value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Uint8Array) && !isByteStream(value)
        ? (value as Record<string, ResolvedValue>)
        : {};
    for (const [name, slot] of Object.entries(produced)) {
      let v: ResolvedValue | undefined = record[name];
      if (v === undefined) continue;
      if (isArtifactSlot(slot) && typeof v === "string") {
        v = this.registerArtifact(instance, name, slot, v);
      } else {
        const err = this.validateSlotValue(name, slot, v);
        if (err) return { classification: "api-retriable", reason: `operation output ${err}` };
      }
      instance.outputs[name] = v;
    }
    for (const name of Object.keys(produced)) {
      const meta = instance.def.slotMeta?.[`outputs.${name}`];
      if (meta?.optional !== true && meta?.default === undefined && instance.outputs[name] === undefined) {
        return { classification: "api-retriable", reason: `${op} operation did not produce required output '${name}'` };
      }
    }
    return undefined;
  }

  /**
   * Drain the blob outputs of a just-completed child that this state FANS OUT (§7.3, rule 2). Runs in
   * the producer's completion path — the one async point where every consumer is still downstream — so a
   * single read serves them all. A drain failure fails the producer's delivery non-retriably: the
   * consumers depended on bytes that will never come, so surfacing it as the child's error termination is
   * the honest outcome (the parent's transitions then handle it like any child failure).
   */
  private async materializeFanOut(instance: Instance, key: string, term: TerminationRecord): Promise<TerminationRecord> {
    const fanOut = instance.def.fanOut;
    const outputs = term.outputs;
    if (fanOut === undefined || outputs === undefined || term.outcome !== "success") return term;
    for (const [name, value] of Object.entries(outputs)) {
      if (!isByteStream(value) || !isFannedOut(fanOut, key, name)) continue;
      try {
        outputs[name] = await materialize(value, instance.abort.signal, `state '${instance.stateId}' child '${key}' output '${name}'`);
      } catch (e) {
        const reason = e instanceof MaterializeError ? e.message : `state '${instance.stateId}' child '${key}' output '${name}': ${(e as Error).message}`;
        return { outcome: "error", failure: { classification: "permanent", reason } };
      }
    }
    return term;
  }

  /** Register inline artifact CONTENT as a session artifact and return the ref that stands for it. */
  private registerArtifact(instance: Instance, name: string, slot: Parameter<InlineFamily>, content: string): ArtifactRef {
    const format = artifactFormat(slot);
    const ref: ArtifactRef = {
      artifact: true,
      name: `${instance.stateId.replace(/\//g, ".")}#${instance.id}.${name}`,
      ...(format !== undefined ? { format } : {}),
      content,
    };
    this.artifacts.push(ref);
    return ref;
  }

  /** Output slots the OPERATION produces: everything not derived from a binding (§7.1 — what an
   *  output's `from` expression used to express is now that slot's binding). */
  private producedOutputSlots(def: LoadedState): Record<string, NamedParameter<InlineFamily>> {
    const out: Record<string, NamedParameter<InlineFamily>> = {};
    for (const [name, slot] of Object.entries(def.outputs ?? {})) {
      if (slot.binding === undefined) out[name] = slot;
    }
    return out;
  }

  // --- prompts & conversation -----------------------------------------------

  /**
   * `{{path.to.value}}` interpolation against the instance context. Artifact refs render as their
   * content; arrays/objects as JSON.
   *
   * `opInputs` is the operation's RESOLVED inputs (the state's inputs plus the op's own bound inputs).
   * They become the template's `{{.inputs.*}}` scope — authored render variables ride bound input slots
   * (loader §3.1), so a prompt sees exactly the inputs its operation resolved, nothing more.
   */
  private renderTemplate(template: string, instance: Instance, opInputs?: Record<string, ResolvedValue>): string {
    const base = this.exprContext(instance);
    const ctx = opInputs
      ? { ...base, inputs: { ...(base.inputs as Record<string, unknown>), ...opInputs } }
      : base;
    // The operation's own resolved inputs shadow the instance's for the duration of the render —
    // which is what makes `{{.inputs.style}}` reach a bound render variable rather than a state input.
    const scope: ResolutionScope = { ...this.scopeFor(instance), exprContext: ctx };
    return template.replace(TEMPLATE_REF, (_m, path: string) => {
      // Lowering happens OUTSIDE the catch: a hole that cannot be lowered is an authoring error
      // (`{{inputs.x}}` missing its dot, a name that resolves nowhere), while a hole that lowers and
      // does not resolve is legitimately empty for this render. Swallowing both made the first look
      // like the second — an empty substitution where the prompt silently lost a variable.
      const ref = this.exprRef(path);
      let v: unknown;
      try {
        const r = resolveRef(ref, scope);
        if (!isResolvedValue(r)) return "";
        v = r.value;
      } catch {
        return "";
      }
      if (isPending(v) || v === undefined || v === null) return "";
      if (isArtifactRef(v)) return v.content ?? v.path ?? v.name;
      if (typeof v === "object") return JSON.stringify(v);
      return String(v);
    });
  }

  /**
   * The run's session store — the SINGLE transcript home, and the engine's own.
   *
   * It used to be republished on every child's `ExecServices` so a composed llm layer could find it.
   * It is not published any more: the engine RESOLVES positions itself (`servicesFor`) and hands each
   * executor the position, so nothing downstream needs the store. A host that composes a session layer
   * passes the store to that layer at construction, where the layer that forks with it can see it.
   */
  private sessions(): SessionStore {
    return this.config.sessions ?? this.sessionStore;
  }

  /**
   * Read a session's transcript, mirroring it for synchronous `{ conversation }` binding resolution
   * (§7.5 — a transcript is addressable DATA).
   *
   * Messages are DERIVED from the session's records — a session is not a separate store, it is the
   * records sharing a `session.id`. With no store wired, an empty transcript, which is what a session
   * nobody has written to looks like anyway.
   */
  private async readTranscript(sessionRef: string): Promise<Turn[]> {
    const messages = (await this.sessions().messages(sessionRef)) ?? [];
    const turns = messages as Turn[];
    this.transcripts.set(sessionRef, turns);
    return turns;
  }

  /**
   * Re-read a session after a call, so `{ conversation }` bindings see what it just added.
   *
   * The engine NO LONGER WRITES the transcript. It used to append its own turns, which was the second
   * of two mechanisms doing one job — and once a composed session layer records the call, both would
   * write, land on the same position, and the loser would fork. What the engine keeps is the read: it
   * mirrors the conversation for synchronous binding resolution (§7.5), because bindings resolve
   * without awaiting.
   *
   * A run with no session layer composed records nothing, and the transcript stays empty. That is
   * correct rather than a gap: without one there is no conversation.
   *
   * Mirrored under every ref that NAMES this conversation as it now stands — the position the call
   * ended at, the conversation unpositioned, and the ref the call was declared with. They are three
   * spellings of one content the moment a call returns, and an author holding any of them (a wired
   * `.operation.output.session`, its `.end`, or the name they wrote) reads the same transcript.
   * Mirroring only the declared one is what made `messages(.children.plan.operation.output.session)`
   * report the conversation as unavailable: the ref that flowed as data was never a key.
   */
  private async refreshTranscript(at: string, ...aliases: string[]): Promise<void> {
    const turns = await this.readTranscript(at);
    for (const alias of aliases) this.transcripts.set(alias, turns);
  }
}

// --- helpers -----------------------------------------------------------------

/**
 * An ARTIFACT slot (SPEC §4.6) — a durable work product whose content travels inline for llm-backed
 * states. It is simply a `blob`-KIND slot now (DESIGN §3.7): the bespoke `x-artifact: true`
 * marker existed only because artifact slots had no kind, and `kindFor` derives `blob` from JSON
 * Schema's own `contentEncoding`/`contentMediaType` instead. The slot's `contentMediaType` IS the
 * artifact's content format.
 */
/**
 * Why an unhandled child failure ended the state, in one line.
 *
 * The outcome alone ("terminated with error") names WHAT happened and never why, so the cause had to
 * be reconstructed from the journal — or, for a child that crashed rather than failed, from a console.
 */
/** The reason a guard refused, with a fallback so the outcome is never a failure with no sentence. */
function guardFailureOf(instance: Instance): Failure {
  return instance.guardFailure ?? { classification: "permanent", reason: "a transition guard could not be evaluated" };
}

function unhandledChildReason(key: string, rec: ChildRecord | undefined): string {
  const outcome = rec?.outcome ?? "error";
  const cause = rec?.failure?.reason;
  return cause === undefined
    ? `child '${key}' terminated with ${outcome} and no transition handled it`
    : `child '${key}' terminated with ${outcome} and no transition handled it: ${cause}`;
}

/**
 * The first NaN or Infinity anywhere in a slot value, with the path that reaches it.
 *
 * Deep rather than top-level: a state's output is as often a record or a list of scores as a bare
 * number, and one unrepresentable member poisons the whole value the moment anything canonicalizes
 * it. Binary leaves are skipped — they are bytes, not numbers, and walking them would be pointless
 * work on the largest values in the system.
 */
function nonFiniteAt(
  value: unknown,
  path = "",
  /**
   * Cycle and depth protection, because this runs on EVERY slot value and a slot value is not
   * guaranteed to be JSON. A user `.ts` function's return enters the dataflow unconverted when its
   * schema needs no marshalling (`marshalOut`), so a circular object can reach here — and a check
   * added to stop a run hanging must not become the thing that exhausts the stack.
   */
  seen: Set<object> = new Set(),
  depth = 0,
): { what: string; path: string } | undefined {
  if (typeof value === "number") {
    if (Number.isFinite(value)) return undefined;
    const what = Number.isNaN(value) ? "NaN" : value > 0 ? "Infinity" : "-Infinity";
    return { what, path: path === "" ? "" : ` at ${path}` };
  }
  if (value === null || typeof value !== "object") return undefined;
  if (value instanceof Uint8Array || isByteStream(value)) return undefined;
  // Past this, unrepresentable numbers are not what the value's problem is.
  if (depth > 200 || seen.has(value)) return undefined;
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        const found = nonFiniteAt(value[i], `${path}[${i}]`, seen, depth + 1);
        if (found !== undefined) return found;
      }
      return undefined;
    }
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      const found = nonFiniteAt(v, path === "" ? k : `${path}.${k}`, seen, depth + 1);
      if (found !== undefined) return found;
    }
    return undefined;
  } finally {
    seen.delete(value);
  }
}

function isArtifactSlot(slot: Parameter<InlineFamily>): boolean {
  return slot.kind === "blob";
}

/** The declared content format of an artifact slot — its media type. */
function artifactFormat(slot: Parameter<InlineFamily>): string | undefined {
  const media = slot.schema?.contentMediaType;
  return typeof media === "string" ? media : typeof slot.schema?.format === "string" ? slot.schema.format : undefined;
}

/** JSON view of a resolved record — the resolution scope reads JSON, and engine-internal values are
 *  JSON by construction (they came from validated inputs, operation outputs, or literals). */
function asJsonRecord(values: Record<string, ResolvedValue>): Record<string, JsonValue> {
  return values as Record<string, JsonValue>;
}

/** The state failed while flagged canceled/timed-out? Loop top decides; here we always report error. */
function failureOutcome(instance: Instance): TerminationOutcome {
  if (instance.timedOut) return "timeout";
  if (instance.abort.signal.aborted) return "canceled";
  return "error";
}

/** Failure payload for an author-directed `terminate.error` transition. */
function errorOf(instance: Instance, target: string): Failure {
  return { classification: "permanent", reason: `state '${instance.stateId}' transitioned to ${target}` };
}

/** The structured-output contract a prompt operation must satisfy: the state's produced output
 *  slots as one object schema. An artifact slot asks for its content as a string. */
function buildOutputSchema(slots: Record<string, NamedParameter<InlineFamily>>, def: LoadedState): JsonSchema | undefined {
  const names = Object.keys(slots);
  if (names.length === 0) return undefined;
  const properties: Record<string, JsonValue> = {};
  const required: string[] = [];
  for (const [name, slot] of Object.entries(slots)) {
    const meta = def.slotMeta?.[`outputs.${name}`];
    if (isArtifactSlot(slot)) {
      const format = artifactFormat(slot);
      properties[name] = { type: "string", description: `Artifact content${format !== undefined ? ` (${format})` : ""}.` };
    } else {
      const doc: Record<string, JsonValue> = { ...((slot.schema ?? {}) as Record<string, JsonValue>) };
      if (meta?.description !== undefined) doc.description = meta.description;
      properties[name] = doc;
    }
    // A default-backed output is optional in the structured-output contract: the engine backfills the
    // default when the model omits it (see `finish`, `acceptOpOutputs`), so requiring the model to
    // produce it would force a fabricated value and can trip strict schema validation. This matches
    // the `optional !== true && default === undefined` rule every other optionality check applies.
    if (meta?.optional !== true && meta?.default === undefined) required.push(name);
  }
  return { type: "object", properties, required, additionalProperties: true };
}

/** Keep event payloads readable: inline artifact contents are elided (or kept, for UI display). */
function shallowRedactArtifacts(values: Record<string, ResolvedValue>, keepContent = false): Record<string, ResolvedValue> {
  const out: Record<string, ResolvedValue> = {};
  for (const [k, v] of Object.entries(values)) {
    out[k] = isArtifactRef(v) && !keepContent ? { artifact: true, name: v.name, ...(v.format !== undefined ? { format: v.format } : {}) } : v;
  }
  return out;
}
