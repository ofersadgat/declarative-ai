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
  MapSessionStore,
  createOperationExecutor,
  hashOperation,
  isOk,
  resolveCalls,
} from "@declarative-ai/exec";
import type { WorkflowMetrics } from "./ports.js";
import {
  createToolGate,
  PermissionLedger,
  planExitTool,
  withPermission,
  type Approver,
  type PermissionBaseline,
  type PermissionMode,
  type ToolGate,
  type ProfilePredicate,
  type SmartApprover,
} from "@declarative-ai/permissions";
import { SchemaValidator } from "@declarative-ai/validate";
import { isPending, parseExpression, PENDING } from "./expr.js";
import { lowerExpression } from "./lowerExpr.js";
import type {
  ConversationMode,
  ExecEnvironmentDecl,
  LoadedChild,
  LoadedState,
  SlotMeta,
  TerminationOutcome,
  WorkflowBundle,
} from "./format.js";
import { bindElement, bindInputs, embeddedOpsOf, higherOrderEdgesOf, higherOrderOf, isResolvedValue, isResolveError, resolveEmbedded, resolveInputs, resolveRef, type ResolutionScope, type Resolved } from "./resolve.js";
import { isByteStream, materialize, MaterializeError } from "./materialize.js";
import { RUN_RESOURCE_KEY, isSessionExpr, resolveSession, sessionFromExpr, type SessionBinding, type SessionDecl } from "./session.js";
import type { OperationNode } from "./operationNode.js";
import { isFannedOut } from "./fanout.js";
import { isArtifactRef, type ArtifactRef, type EngineEvent, type OperationKind, type Persistence } from "./ports.js";

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

export type CallResult = { value: ResolvedValue } | { error: string; failure?: Failure };

/**
 * A content-addressed store of call results, keyed by `hashOperation` of the RESOLVED operation.
 *
 * Sync on purpose: it is read during binding resolution, which cannot suspend (`renderTemplate`
 * resolves inside a `String.replace` callback). A host wanting a remote cache warms it between runs
 * rather than awaiting inside one.
 */
export interface CallCache {
  get(key: string): CallResult | undefined;
  set(key: string, value: CallResult): void;
}

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
   * Where the results of CALLS are remembered (EXPRESSIONS.md §3).
   *
   * The question a memo has to answer is "would someone else making the identical call reuse this
   * answer?" — so the key is content-addressed: `hashOperation` over the RESOLVED op, which embeds
   * its argument values, is exactly "this callee with these arguments". An in-run `Map` is the
   * default and answers it only within one run; a host that wants an identical call to be reused
   * across runs, tasks or processes supplies a durable one.
   */
  callCache?: CallCache;
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
    /** Custom profile predicates by name (DESIGN §5.1, "Permissions: two orthogonal axes") — a `runtime.permissions.profile`
     *  naming one of these gates tools by its predicate instead of the built-in read-only/plan/full. */
    profiles?: Record<string, ProfilePredicate>;
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
  instanceId: number;
  status: "running" | "done";
  outcome?: TerminationOutcome;
  outputs?: Record<string, ResolvedValue>;
  /** The child's own operation node — what `children.<key>.operation.*` reads (SPEC.md §6.1). */
  operation?: OperationNode;
  abort: AbortController;
  promise: Promise<void>;
}

/** One state instance (SPEC §3.4) — results never leak across instances. */
interface Instance {
  id: number;
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
  iteration: number;
  /** Whether the state's single operation has run (§7.1: a state has ONE operation). */
  opRun: boolean;
  /** Live child records by child key; `undefined`/absent = never ran or superseded. */
  children: Map<string, ChildRecord>;
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
  /** The child most recently ENTERED — what `run.cursor` reports to a guard. */
  entered?: string;
  /** Child keys whose error/timeout termination has not yet been handled by a transition. */
  unhandledFailures: Set<string>;
  abort: AbortController;
  timedOut: boolean;
  notify: Notifier;
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
  session?: SessionBinding,
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
function operationOutputOf(value: ResolvedValue | undefined, session?: SessionBinding): JsonValue | undefined {
  const position = session === undefined ? undefined : { id: session.id };
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

/** The model an operation resolved to — read off the config the call was actually made with. */
function modelOfOp(op: Operation<InlineFamily>): string | undefined {
  const config = (op as { config?: unknown }).config;
  if (config === null || typeof config !== "object" || Array.isArray(config)) return undefined;
  const model = (config as { model?: unknown }).model;
  return typeof model === "string" ? model : undefined;
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
  private nextInstanceId = 1;
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

  // --- events ---------------------------------------------------------------

  private emit(event: EngineEvent): void {
    this.config.persistence?.record(event, this.clock.now());
    this.config.onEvent?.(event);
  }

  // --- instance loop --------------------------------------------------------

  private async runInstance(
    stateId: string,
    def: LoadedState,
    inputs: Record<string, ResolvedValue>,
    abort: AbortController,
    childKey: string | undefined,
    parent: Instance | undefined,
  ): Promise<TerminationRecord> {
    const instance: Instance = {
      id: this.nextInstanceId++,
      stateId,
      def,
      childKey,
      parent,
      inputs,
      outputs: {},
      // Resolved once, on entry, from the parent's bundle and this state's own declaration — so a
      // subtree that declares nothing shares its enclosing bundle rather than minting one per state.
      resourceKey: resourceKeyFor(def, parent),
      iteration: 0,
      opRun: false,
      children: new Map(),
      cursor: 0,
      unhandledFailures: new Set(),
      abort,
      timedOut: false,
      notify: new Notifier(),
    };
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
    }
  }

  private async evaluationLoop(instance: Instance): Promise<TerminationRecord> {
    const def = instance.def;
    // SPEC §3.3: transitions are evaluated when an operation completes or a child
    // terminates — not on bare entry (the first operation runs first) and not when an
    // async child merely starts. One final evaluation runs before success-termination.
    let evaluationDue = false;
    for (;;) {
      if (instance.timedOut) return this.finish(instance, "timeout");
      if (instance.abort.signal.aborted) return this.finish(instance, "canceled");

      // A guard may CALL an operation, and evaluation is synchronous — so its calls run here, exactly
      // as an operation's input calls run before `resolveInputs`. The memo means a guard re-evaluated
      // over many rounds pays for its call once.
      if (evaluationDue) {
        const guardFailure = await this.runEmbeddedOps(instance, WorkflowEngine.guardParamsOf(def));
        if (guardFailure !== undefined) return this.finish(instance, "error", guardFailure);
      }

      // (3)/(4) Transition evaluation, declared order, PENDING-skipping.
      if (evaluationDue) {
        evaluationDue = false;
        // Whatever the cursor was waiting on has resolved by the time an evaluation round runs; if
        // no transition handles it, the cursor is free to walk on from where it stopped.
        instance.heldFor = undefined;
        const step = this.takeTransition(instance);
        if (step === "terminated-success") return await this.finishSuccess(instance);
        if (step === "terminated-error") return this.finish(instance, "error", errorOf(instance, "terminate.error"));
        if (step === "terminated-canceled") return this.finish(instance, "canceled");
        if (step === "terminated-timeout") return this.finish(instance, "timeout");
        if (step === "entered" || step === "parked") continue;
        // "none": fall through — but a child failure no transition handled is fatal (SPEC §3.3).
        if (instance.unhandledFailures.size > 0) {
          const key = [...instance.unhandledFailures][0]!;
          const rec = instance.children.get(key);
          return this.finish(instance, "error", {
            classification: "permanent",
            reason: `child '${key}' terminated with ${rec?.outcome ?? "error"} and no transition handled it`,
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
      const held = instance.heldFor !== undefined && instance.children.get(instance.heldFor)?.status === "running";
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
      const final = this.takeTransition(instance);
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
          reason: `child '${key}' terminated with ${rec?.outcome ?? "error"} and no transition handled it`,
        });
      }
      return await this.finishSuccess(instance);
    }
  }

  /** Evaluate transitions once; take the first match (SPEC §3.3 step 3–4). */
  private takeTransition(
    instance: Instance,
  ): "none" | "entered" | "parked" | "terminated-success" | "terminated-error" | "terminated-canceled" | "terminated-timeout" {
    const taken = this.firstMatchingTransition(instance);
    if (!taken) return "none";
    instance.iteration++;
    this.emit({
      type: "transition.taken",
      instanceId: instance.id,
      stateId: instance.stateId,
      to: taken.to,
      iteration: instance.iteration,
    });
    instance.unhandledFailures.clear(); // a taken transition handles preceding child failures
    if (taken.to.startsWith("terminate.")) {
      return `terminated-${taken.to.slice("terminate.".length) as TerminationOutcome}` as const;
    }
    const entered = this.enterChild(instance, taken.to);
    if (entered === "parked") {
      instance.iteration--; // the entry did not actually happen
      return "parked";
    }
    return "entered";
  }

  private firstMatchingTransition(instance: Instance): { to: string } | undefined {
    const transitions = instance.def.transitions ?? [];
    if (transitions.length === 0) return undefined;
    const scope = this.scopeFor(instance);
    for (const t of transitions) {
      // A guard that failed to lower never fires: validation blocks the run, and reading it as
      // unconditional would be the worst possible interpretation of a typo.
      if (t.whenError !== undefined) continue;
      if (t.whenRef === undefined) return { to: t.to };
      const r = resolveRef(t.whenRef, scope);
      if (isPending(r)) continue; // skipped this round (SPEC §6/§10.4)
      // A lowered expression cannot yield an ERROR on data: every operator's failure case is
      // "producer is missing X", a malformed tree the loader cannot emit, and reading a missing
      // namespace or property yields `undefined` rather than refusing. So there is no fourth
      // outcome to give a bespoke path to — a non-value simply does not take the transition.
      if (isResolvedValue(r) && r.value) return { to: t.to };
    }
    return undefined;
  }

  /**
   * Enter a child, however control got here — the sequence cursor or a transition.
   *
   * A SYNC child holds the cursor until it resolves (SPEC §10.4); an `async` one does not, which is
   * the entire difference between the two and the only place the flag is read.
   */
  private enterChild(instance: Instance, key: string): "started" | "parked" {
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

    const resolved = this.resolveChildInputs(instance, decl);
    if (resolved === PENDING) return "parked";

    instance.entered = key;
    if (decl.async !== true) instance.heldFor = key;

    // Re-entering a child (SPEC §3.4) creates a fresh instance; a stale running
    // instance under the same key is canceled and replaced.
    const prior = instance.children.get(key);
    if (prior?.status === "running") prior.abort.abort();

    const childDef = this.config.bundle.states[decl.state];
    const childAbort = new AbortController();
    if (instance.abort.signal.aborted) childAbort.abort();
    else instance.abort.signal.addEventListener("abort", () => childAbort.abort(), { once: true });

    const record: ChildRecord = {
      instanceId: -1,
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
        this.emit({ type: "instance.blocked", instanceId: -1, stateId: decl.state, reason: resolved.error });
        term = { outcome: "error", failure: { classification: "permanent", reason: resolved.error } };
      } else {
        term = await this.runInstance(decl.state, childDef, resolved.values!, childAbort, key, instance);
        // Fan-out (§7.3, rule 2) is decided at BIND time: if this producer's blob output feeds two
        // consumers, drain it ONCE here, at the producer's completion, so both siblings read the bytes
        // rather than racing to read one stream. A single-consumer output is left a live stream to pipe.
        term = await this.materializeFanOut(instance, key, term);
      }
      record.status = "done";
      record.outcome = term.outcome;
      record.outputs = term.outputs;
      // The child's operation node, so `children.<key>.operation.*` reads what its call reported —
      // including, for a prompt op, the conversation position it ended at (SPEC.md §6.1).
      record.operation = term.operation;
      if ((term.outcome === "error" || term.outcome === "timeout") && instance.children.get(key) === record) {
        instance.unhandledFailures.add(key);
      }
      instance.notify.signal();
    };

    instance.children.set(key, record);
    record.promise = run();
    return "started";
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
      const rec = instance.children.get(key);
      if (!rec) children[key] = {};
      else if (rec.status === "running") children[key] = { outputs: PENDING, outcome: PENDING, operation: PENDING };
      else children[key] = { outputs: rec.outputs ?? {}, outcome: rec.outcome, operation: rec.operation ?? {} };
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
        iteration: instance.iteration,
        cursor: instance.entered ?? "",
        position: instance.entered !== undefined ? (instance.def.sequence?.indexOf(instance.entered) ?? -1) : -1,
      },
      limits: { ...(instance.def.limits ?? {}) },
      artifacts,
    };
  }

  /** The run-scoped view binding resolution needs (§7.4) — this instance's data addresses. */
  private scopeFor(instance: Instance): ResolutionScope {
    return {
      exprContext: this.exprContext(instance),
      // A lowered CALL reads its result here, exactly as a child read reads `childOutputs`:
      // resolution never runs anything, and `undefined` (not yet run) parks the consumer.
      operationResult: (op) => this.callCache.get(hashOperation(op)),
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
  private resolveChildInputs(
    instance: Instance,
    decl: LoadedChild,
  ): typeof PENDING | { values?: Record<string, ResolvedValue>; error?: string } {
    const childDef = this.config.bundle.states[decl.state];
    if (!childDef) return { error: `unknown state '${decl.state}'` };
    const scope = this.scopeFor(instance);
    const values: Record<string, ResolvedValue> = {};
    for (const [name, slot] of Object.entries(childDef.inputs ?? {})) {
      const meta = childDef.slotMeta?.[`inputs.${name}`];
      const wire = decl.inputs?.[name];
      let v: ResolvedValue | undefined;
      if (wire !== undefined) {
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
    // `operationId` arrives from the dispatch paths once the DISPATCHED op exists — see the
    // EngineEvent comment on why a pre-dispatch failure carries none.
    const fail = (failure: Failure, operationId?: string): Failure => {
      this.emit({
        type: "operation.failed",
        instanceId: instance.id,
        stateId: instance.stateId,
        op: kind,
        ...(operationId !== undefined ? { operationId } : {}),
        failure,
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
    const embeddedFailure = await this.runEmbeddedOps(instance, op.input);
    if (embeddedFailure !== undefined) return fail(embeddedFailure);
    const resolved = resolveInputs(op.input, this.scopeFor(instance));
    if (isPending(resolved)) {
      return fail({ classification: "permanent", reason: "operation inputs depend on a child that has not resolved" });
    }
    if ("error" in resolved) return fail({ classification: "permanent", reason: resolved.error });
    const opInputs: FunctionInputs = { ...instance.inputs, ...resolved.values };

    return op.kind === "prompt"
      ? this.runPromptOp(instance, op, opInputs, fail)
      : this.runFunctionOp(instance, op, opInputs, fail);
  }

  /** A state's guards as parameter-shaped bindings, so one walker serves guards and slots alike. */
  private static guardParamsOf(def: LoadedState): Record<string, Parameter<InlineFamily>> {
    const out: Record<string, Parameter<InlineFamily>> = {};
    (def.transitions ?? []).forEach((t, i) => {
      if (t.whenRef !== undefined) out[`when${i}`] = { kind: "json", binding: t.whenRef };
    });
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

  /** The cache backing {@link EngineConfig.callCache} when the host supplies none. */
  private readonly ownCallCache = new Map<string, CallResult>();
  private get callCache(): CallCache {
    return this.config.callCache ?? { get: (k: string) => this.ownCallCache.get(k), set: (k: string, v: CallResult) => void this.ownCallCache.set(k, v) };
  }

  /**
   * Results of embedded operations, keyed by the RESOLVED op's content hash.
   *
   * The hash is `hashOperation`, which is the same identity `withMemoize` keys on — and because a
   * resolved op embeds its argument values, it IS "this callee with these arguments". So a call
   * appearing in a guard costs one execution however many rounds the guard is evaluated over, and two
   * syntactically different expressions that compute the same thing share one result.
   */

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
  private async runEmbeddedOps(instance: Instance, input: Record<string, Parameter<InlineFamily>>): Promise<Failure | undefined> {
    for (const param of Object.values(input)) {
      if (!param.binding) continue;
      // HIGHER-ORDER first (§3.5): one application per element, and how many there are is not known
      // until the array resolves — so this cannot be a static walk like `embeddedOpsOf` is.
      const higherFailure = await this.runHigherOrder(instance, param.binding);
      if (higherFailure !== undefined) return higherFailure;
      for (const { op, parameters } of embeddedOpsOf(param.binding)) {
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
        if (this.callCache.get(key) !== undefined) continue;
        const outcome = await this.runEmbeddedOp(instance, resolved.op);
        // PENDING is a scheduling state, not an answer — nothing to remember, and nothing a durable
        // cache could serialize.
        if (outcome !== PENDING) this.callCache.set(key, outcome);
      }
    }
    return undefined;
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
          let outcome = this.callCache.get(key);
          if (outcome === undefined) {
            const run = await this.runEmbeddedOp(instance, bound);
            if (run === PENDING) return { classification: "permanent", reason: "'reduce' step did not resolve" };
            this.callCache.set(key, run);
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
        if (this.callCache.get(key) === undefined) pending.set(key, bound);
      }
      const results = await Promise.all(
        [...pending].map(async ([key, bound]) => [key, await this.runEmbeddedOp(instance, bound)] as const),
      );
      for (const [key, outcome] of results) if (outcome !== PENDING) this.callCache.set(key, outcome);
    }
    return undefined;
  }

  /** Run ONE embedded operation and return what the binding should see. */
  private async runEmbeddedOp(instance: Instance, op: Operation<InlineFamily>): Promise<Resolved> {
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
    let outcome;
    try {
      outcome = await this.operations.start(
        rendered,
        await this.servicesFor(resourceKey, instance, toolsOrFailure.tools, undefined, toolsOrFailure.gate),
      ).result;
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
    fail: (f: Failure, operationId?: string) => Failure,
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

    const services = await this.servicesFor(resourceKey, instance, toolsOrFailure.tools, session, toolsOrFailure.gate);
    // Errors are DATA (§4.2): the impl RESOLVES value-or-failure, so a 429 raised inside a registered
    // function keeps its classification instead of being reconstructed from `err.name` — which is what
    // made every non-`AbortError` permanently failed, retry machinery and all.
    //
    // `bindInputs` writes the resolved inputs onto the op first: the executor reads them off the op
    // and has no view of the instance they were resolved against.
    // Hashed HERE, over exactly the value the executor stack receives, because that is the id an
    // unplaced record gets (`withRecord`: no position ⇒ `hashOperation(op)`) — the join the
    // settled events carry. Undefined when the op cannot be hashed (a live stream input, which
    // `hashOperation` refuses by design): such a call's record has no content id either.
    const dispatched = bindInputs(this.operationFor(instance, op), opInputs);
    const operationId = tryHashOperation(dispatched);
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
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles
    if (!isOk(outcome)) return fail(outcome.error, operationId);
    // The op's declared output KIND decides how its value is read — a `blob` output IS the value
    // (bytes), any other kind is a record of named outputs. Omitting it here left the blob branch
    // unreachable from the function path, so a function op producing a `Uint8Array` failed with "did
    // not produce required output" about the file it had just produced (§7.1).
    const failure = this.acceptOpOutputs(instance, "function", outcome.value, op.output.kind);
    if (failure) return fail(failure, operationId);
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
    fail: (f: Failure, operationId?: string) => Failure,
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
    const transcript = await this.readTranscript(session.id);
    const preamble = this.conversationPreamble(env.conversation?.mode ?? "full_history", transcript, env.conversation?.artifacts);
    const prompt = preamble ? `${preamble}\n\n${rendered}` : rendered;

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
      user: prompt,
      output: { ...op.output, ...(produced !== undefined ? { schema: produced } : {}) },
    };

    // The per-call ENVIRONMENT the old `PromptOpEnvironment` carried — tools, the time budget,
    // cancellation — are `ExecServices` fields now, which is why that type could be deleted outright.
    // The gate rides along exactly as it does on the function path: inert for a composed transport,
    // and the ONLY carrier of authored modes and the session profile for a delegated one.
    const services = await this.servicesFor(session.resourceKey, instance, tools, session, toolsOrFailure.gate);
    // An authored `limits.timeout` reaches the call as CANCELLATION. It used to be published as
    // `services.timeoutMs`, which only an executor that knew to read it honoured — and which the llm
    // layer turned straight back into `AbortSignal.timeout(...)` anyway. Folding it into the signal
    // bounds every executor, including ones that read nothing but `abortSignal`.
    if (instance.def.limits?.timeout !== undefined) {
      const bound = AbortSignal.timeout(instance.def.limits.timeout * 1000);
      services.abortSignal = services.abortSignal ? AbortSignal.any([services.abortSignal, bound]) : bound;
    }
    // Hashed over the op the executor stack receives — the settled events' join to its record.
    const operationId = tryHashOperation(resolvedOp);
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
    instance.operation = operationNodeOf(
      isOk(outcome) ? "success" : "error",
      outcome.metrics,
      modelOfOp(resolvedOp),
      session,
      isOk(outcome) ? outcome.value : undefined,
    );
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles

    // A FAILED call contributes nothing to the transcript. It ran before this check and a failure
    // carries no `value`, so the assistant turn was the literal string "null" — and under the default
    // `full_history` mode every later state in the session then read that back in its preamble.
    if (!isOk(outcome)) return fail(outcome.error, operationId);

    // Conversation artifact (SPEC §4.7): the exchange is already in the session, because the session
    // layer RECORDED the call — one write, not two. The engine used to synthesize a user turn and a
    // stringified assistant turn here, which threw away every tool call and reasoning part in between
    // and is exactly what the append-only model replaced. All that remains is re-reading, so a
    // `{ conversation }` binding in this state's outputs sees what the call just added.
    await this.refreshTranscript(session.id);

    const failure = this.acceptOpOutputs(instance, "prompt", (outcome.value ?? null) as ResolvedValue, op.output.kind);
    if (failure) return fail(failure, operationId);
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
    const gate = createToolGate({
      ledger: this.permissions,
      sessionId,
      approve: escalate,
      tools: Object.fromEntries(Object.entries(tools).map(([name, tool]) => [name, { readOnly: tool.readOnly }])),
      ...(env.permissions !== undefined ? { authored: env.permissions } : {}),
      ...(smart !== undefined ? { smart } : {}),
      ...(customProfiles !== undefined ? { profiles: customProfiles } : {}),
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
      });
    }
    if (this.permissions.resolveProfile(sessionId) === "plan") {
      guarded["exit_plan"] = planExitTool({ ledger: this.permissions, sessionId, approve });
    }
    return { tools: guarded, gate };
  }

  /** The services one operation runs with: its resource bundle's workspace, its tools, its cancellation. */
  private async servicesFor(
    resourceKey: string,
    instance: Instance,
    tools?: Record<string, Tool>,
    session?: SessionBinding,
    /** The delegated permission gate, when this call has one — see {@link resolveTools}. */
    gate?: ToolGate,
  ): Promise<ExecServices> {
    const services = this.childServices();
    if (gate !== undefined) services.gate = gate;
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
   */
  private async refreshTranscript(sessionRef: string): Promise<void> {
    await this.readTranscript(sessionRef);
  }

  private conversationPreamble(mode: ConversationMode, transcript: Turn[], artifactNames?: string[]): string {
    switch (mode) {
      case "fresh":
        return "";
      case "selected_artifacts": {
        const wanted = new Set(artifactNames ?? []);
        const parts = this.artifacts
          .filter((a) => wanted.size === 0 || wanted.has(a.name))
          .map((a) => `<artifact name="${a.name}">\n${a.content ?? ""}\n</artifact>`);
        return parts.join("\n");
      }
      case "summary":
      // v1: no summarizer wired — degrade to full history (documented in DESIGN §7).
      // eslint-disable-next-line no-fallthrough
      case "full_history": {
        if (transcript.length === 0) return "";
        // `content` is a string for engine-recorded turns; be defensive if a `withSession` writer stored parts.
        const lines = transcript.map((m) => `${m.role}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`);
        return `<conversation-history>\n${lines.join("\n")}\n</conversation-history>`;
      }
    }
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
