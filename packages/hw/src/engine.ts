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
 * Child selection is simply "first sequence member with no live record", which
 * implements cursor-reset-and-resume without a separate cursor.
 *
 * Unhandled failures (SPEC §3.3): an unrecoverable operation failure terminates the
 * state with error; a child that terminated with error/timeout and is not handled by
 * any transition in the following evaluation round does the same.
 */
import {
  type CapabilityRegistry,
  type Failure,
  type ExecServices,
  type Executor,
  type FunctionInputs,
  type OutputValidator,
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
  type ResolvedValue,
  type Tool,
  type Workspace,
  type Clock,
  MapSessionStore,
  isOk,
  runFunction,
} from "@declarative-ai/exec";
import type { WorkflowMetrics } from "./ports";
import {
  PermissionLedger,
  planExitTool,
  withPermission,
  type Approver,
  type PermissionBaseline,
  type PermissionMode,
  type ProfilePredicate,
  type SmartApprover,
} from "@declarative-ai/permissions";
import { SchemaValidator } from "@declarative-ai/validate";
import { evaluate, isPending, parseExpression, PENDING, type Expr } from "./expr";
import type {
  ConversationMode,
  EnvironmentDecl,
  LoadedChild,
  LoadedState,
  SlotMeta,
  TerminationOutcome,
  WorkflowBundle,
} from "./format";
import { skillNameOf } from "./loader";
import { isResolvedValue, isResolveError, resolveInputs, resolveRef, type ResolutionScope } from "./resolve";
import { isByteStream, materialize, MaterializeError } from "./materialize";
import { isFannedOut } from "./fanout";
import { isArtifactRef, type ArtifactRef, type EngineEvent, type OperationKind, type Persistence } from "./ports";

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
  validator?: OutputValidator;
  persistence?: Persistence;
  /** Forwarded to runtimes/functions (rate limiter, meter, ...) as their `services`. `validator`/session
   *  store are supplied by the engine. */
  services?: ExecServices;
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
}

interface ChildRecord {
  instanceId: number;
  status: "running" | "done";
  outcome?: TerminationOutcome;
  outputs?: Record<string, ResolvedValue>;
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
  iteration: number;
  /** Whether the state's single operation has run (§7.1: a state has ONE operation). */
  opRun: boolean;
  /** Live child records by child key; `undefined`/absent = never ran or superseded. */
  children: Map<string, ChildRecord>;
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

const TEMPLATE_REF = /\{\{\s*([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)\s*\}\}/g;

/** The logical session a runtime op falls back to when its state declares no `runtime.session` — so a plain
 *  workflow behaves as ONE shared session (transcript, workspace, permissions) across all its states. */
const DEFAULT_SESSION = "default";

/** One conversation turn — a `ModelMessage`-compatible shape, so the built-in `conversationMode` transcript
 *  and the llm `withSession` path share ONE representation in `SessionState.messages`. */
type Turn = { role: "user" | "assistant"; content: string };

export class WorkflowEngine {
  private readonly validator: OutputValidator;
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
      baseline: config.permissions?.baseline,
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
      iteration: 0,
      opRun: false,
      children: new Map(),
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
      const record = await this.evaluationLoop(instance);
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

      // (3)/(4) Transition evaluation, declared order, PENDING-skipping.
      if (evaluationDue) {
        evaluationDue = false;
        const step = this.takeTransition(instance);
        if (step === "terminated-success") return this.finish(instance, "success");
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
      if (def.operation && !instance.opRun) {
        instance.opRun = true;
        const failure = await this.runOperation(instance, def.operation);
        if (failure) return this.finish(instance, failureOutcome(instance), failure);
        evaluationDue = true;
        continue;
      }

      // Next sequence child = first member with no live record (implements the
      // reset-and-resume cursor, see module header).
      const nextKey = (def.sequence ?? []).find((k) => !instance.children.has(k));
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
        // Async children fall through without waiting and WITHOUT triggering
        // evaluation (SPEC §10.4); sync children wait via the running-children
        // branch below, whose wake sets evaluationDue.
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
      if (final === "terminated-success") return this.finish(instance, "success");
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
      return this.finish(instance, "success");
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
    const ctx = this.exprContext(instance);
    for (const t of transitions) {
      if (t.when === undefined) return { to: t.to };
      const v = evaluate(this.parse(t.when), ctx);
      if (isPending(v)) continue; // skipped this round (SPEC §6/§10.4)
      if (v) return { to: t.to };
    }
    return undefined;
  }

  /** Sync children block the loop via waitForAnyChild; async ones don't (SPEC §10.4). */
  private enterChild(instance: Instance, key: string): "started" | "parked" {
    const decl = instance.def.children?.[key];
    if (!decl) throw new Error(`${instance.stateId}: transition/sequence names undeclared child '${key}'`);

    // Sequence reset (SPEC §3.3): entering a sequence member clears it and every
    // later member; children outside the sequence keep their results.
    const sequence = instance.def.sequence ?? [];
    const seqIndex = sequence.indexOf(key);
    if (seqIndex >= 0) {
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
      else if (rec.status === "running") children[key] = { outputs: PENDING, outcome: PENDING };
      else children[key] = { outputs: rec.outputs ?? {}, outcome: rec.outcome };
    }
    const artifacts: Record<string, unknown> = {};
    for (const a of this.artifacts) artifacts[a.name] = a;
    return {
      inputs: instance.inputs,
      outputs: instance.outputs,
      children,
      run: { iteration: instance.iteration },
      limits: { ...(instance.def.limits ?? {}) },
      artifacts,
      conversations: Object.fromEntries(this.transcripts),
    };
  }

  /** The run-scoped view binding resolution needs (§7.4) — this instance's data addresses. */
  private scopeFor(instance: Instance): ResolutionScope {
    return {
      exprContext: this.exprContext(instance),
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

  private readonly parsed = new Map<string, Expr>();
  private parse(src: string): Expr {
    let ast = this.parsed.get(src);
    if (!ast) {
      ast = parseExpression(src);
      this.parsed.set(src, ast);
    }
    return ast;
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
    const fail = (failure: Failure): Failure => {
      this.emit({ type: "operation.failed", instanceId: instance.id, stateId: instance.stateId, op: kind, failure });
      return failure;
    };

    // Resolve every BOUND input; FREE slots are filled by name from the state's own inputs (the
    // model's §3.8 rule). Bound values win, because a binding is what the author wrote on THIS
    // operation while the state's inputs are the general scope it draws from — the "explicit value
    // overrides a binding" rule applies one level up, where a parent wires into a child's inputs
    // (`resolveChildInputs`). A PENDING producer means the operation depends on an async child that
    // has not resolved: a blocked operation rather than a park, since a state's operation runs once.
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

  /** Dispatch a `FunctionOp` through the function registry (§7.4). */
  private async runFunctionOp(
    instance: Instance,
    op: FunctionOp<InlineFamily>,
    opInputs: FunctionInputs,
    fail: (f: Failure) => Failure,
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
    const sessionId = env.session ?? DEFAULT_SESSION;
    // The entry's capabilities are REQUIRED and total per variant (§2), so this reads a definite value
    // instead of falling through an `undefined` and silently defaulting the permission gate.
    const delegates = entry.kind === "runtime" && entry.capabilities.policyEnforcement === "callback";
    const toolsOrFailure = this.resolveTools(env, sessionId, delegates);
    if ("failure" in toolsOrFailure) return fail(toolsOrFailure.failure);

    const services = this.servicesFor(sessionId, instance, toolsOrFailure.tools);
    // Errors are DATA (§4.2): the impl RESOLVES value-or-failure, so a 429 raised inside a registered
    // function keeps its classification instead of being reconstructed from `err.name` — which is what
    // made every non-`AbortError` permanently failed, retry machinery and all.
    const outcome = await runFunction(entry, opInputs, services);
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
      this.childCost += metrics.costUsd;
    }
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles
    if (!isOk(outcome)) return fail(outcome.error);
    // The op's declared output KIND decides how its value is read — a `blob` output IS the value
    // (bytes), any other kind is a record of named outputs. Omitting it here left the blob branch
    // unreachable from the function path, so a function op producing a `Uint8Array` failed with "did
    // not produce required output" about the file it had just produced (§7.1).
    const failure = this.acceptOpOutputs(instance, "function", outcome.value, op.output.kind);
    if (failure) return fail(failure);
    this.emit({
      type: "operation.completed",
      instanceId: instance.id,
      stateId: instance.stateId,
      op: "function",
      ...(metrics !== undefined ? { metrics } : {}),
    });
    return undefined;
  }

  /** Dispatch a `PromptOp` through the registered prompt runner (§6/§7.4). */
  private async runPromptOp(
    instance: Instance,
    op: PromptOp<InlineFamily>,
    opInputs: FunctionInputs,
    fail: (f: Failure) => Failure,
  ): Promise<Failure | undefined> {
    const promptExecutor = this.config.prompt;
    if (!promptExecutor) {
      return fail({ classification: "permanent", reason: "this workflow contains a prompt state but no prompt executor is wired in (EngineConfig.prompt)" });
    }

    const env = instance.def.environment ?? {};
    // The logical SESSION this operation runs under (DESIGN §5.1, "Sessions: the run-scoped resource bundle"): the sharing key
    // for its owned resources — conversation transcript, workspace, permissions. Absent ⇒ the run's
    // default session, so a plain workflow is ONE shared session (SPEC §4.7 threads across states).
    const sessionId = env.session ?? DEFAULT_SESSION;

    // The `user` slot holds an inline template, or a `skill:` reference resolved through
    // `registry.skills` (the two authored prompt sources, §7.1).
    let template = op.user;
    const skill = skillNameOf(op.user);
    if (skill !== undefined) {
      const skillTemplate = this.config.registry.skills.get(skill);
      if (skillTemplate === undefined) return fail({ classification: "permanent", reason: `skill '${skill}' is not registered` });
      template = skillTemplate;
    }
    const rendered = this.renderTemplate(template, instance, opInputs);
    const transcript = await this.readTranscript(sessionId);
    const preamble = this.conversationPreamble(env.conversation?.mode ?? "full_history", transcript, env.conversation?.artifacts);
    const prompt = preamble ? `${preamble}\n\n${rendered}` : rendered;

    const toolsOrFailure = this.resolveTools(env, sessionId, false);
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
    const services = this.servicesFor(sessionId, instance, tools);
    if (instance.def.limits?.timeout !== undefined) services.timeoutMs = instance.def.limits.timeout * 1000;
    let outcome;
    try {
      outcome = await promptExecutor.start(resolvedOp, services).result;
    } catch (e) {
      // An executor must not reject for unit failures; a rejection is a bug — normalized here so the
      // workflow still degrades per SPEC §3.3.
      return fail({ classification: "permanent", reason: `prompt executor rejected: ${(e as Error).message}` });
    }
    this.childLlmCalls += 1 + (outcome.metrics.childLlmCalls ?? 0);
    this.childCost += outcome.metrics.costUsd;
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles

    // A FAILED call contributes nothing to the transcript. It ran before this check and a failure
    // carries no `value`, so the assistant turn was the literal string "null" — and under the default
    // `full_history` mode every later state in the session then read that back in its preamble.
    if (!isOk(outcome)) return fail(outcome.error);

    // Conversation artifact (SPEC §4.7): every SUCCEEDED prompt operation appends its exchange to the
    // session. The assistant turn is the op's OUTPUT VALUE. It used to prefer the model's raw text, but
    // that stops at the prompt executor — and for a text-output op the output value IS that text, so
    // the two agree wherever it mattered.
    await this.appendTranscript(sessionId, [
      { role: "user", content: prompt },
      { role: "assistant", content: typeof outcome.value === "string" ? outcome.value : JSON.stringify(outcome.value ?? null) },
    ]);

    const failure = this.acceptOpOutputs(instance, "prompt", (outcome.value ?? null) as ResolvedValue, op.output.kind);
    if (failure) return fail(failure);
    this.emit({ type: "operation.completed", instanceId: instance.id, stateId: instance.stateId, op: "prompt", metrics: outcome.metrics });
    return undefined;
  }

  /**
   * Resolve the environment's declared tool NAMES through `registry.tools` into executables, and
   * apply the permission gate (DESIGN §5.1, "Enforcement"): with an approver configured, wrap
   * every tool so a call is authorized by profile × mode; seed the authored profile once; inject the
   * plan-exit gate while the session is in `plan`. A DELEGATED adapter (`policyEnforcement:
   * "callback"`) authorizes its own loop's calls through `ctx.approve`, so it gets RAW tools —
   * wrapping there too would double-gate.
   */
  private resolveTools(
    env: EnvironmentDecl,
    sessionId: string,
    delegatesPermissions: boolean,
  ): { tools?: Record<string, Tool> } | { failure: Failure } {
    if (env.tools === undefined || env.tools.length === 0) return {};
    const tools: Record<string, Tool> = {};
    for (const name of env.tools) {
      const tool = this.config.registry.tools.get(name);
      if (!tool) return { failure: { classification: "permanent", reason: `tool '${name}' is not registered` } };
      tools[name] = tool;
    }
    const approve = this.config.permissions?.approve;
    if (!approve || delegatesPermissions) return { tools };

    if (env.permissions?.profile) this.permissions.seedProfile(sessionId, env.permissions.profile);
    const authoredMode = (name: string): PermissionMode | undefined => env.permissions?.tools?.[name] ?? env.permissions?.default;
    const smartFor = this.config.permissions?.smart;
    const profiles = this.config.permissions?.profiles;
    const guarded: Record<string, Tool> = {};
    for (const [name, tool] of Object.entries(tools)) {
      guarded[name] = withPermission(tool, {
        ledger: this.permissions,
        sessionId,
        toolName: name,
        approve,
        authoredMode: authoredMode(name),
        smart: smartFor?.[name],
        profiles,
      });
    }
    if (this.permissions.resolveProfile(sessionId) === "plan") {
      guarded["exit_plan"] = planExitTool({ ledger: this.permissions, sessionId, approve });
    }
    return { tools: guarded };
  }

  /** The services one operation runs with: its session's workspace, its tools, its cancellation. */
  private servicesFor(sessionId: string, instance: Instance, tools?: Record<string, Tool>): ExecServices {
    const services = this.childServices();
    // Per-session workspace (DESIGN §5.1, "Sessions: the run-scoped resource bundle"): states sharing a `sessionId` share one; a
    // fan-out can isolate each branch (e.g. its own worktree) via `workspaceFor`. Falls back to the
    // run-level `services.workspace` when the host provides none for this id.
    const workspace = this.config.workspaceFor?.(sessionId) ?? services.workspace;
    if (workspace !== services.workspace) services.workspace = workspace;
    if (tools !== undefined) services.tools = tools;
    // A registered async function's only channel to the caller is the ctx, so cancellation rides here.
    services.abortSignal = instance.abort.signal;
    return services;
  }

  /** The `ExecServices` operations run with: caller services + engine validator + the run's session
   *  store — the SAME store the built-in transcript uses, so `withSession` and the preamble share one
   *  source (states sharing a logical `sessionId` continue one conversation; an app store wins). */
  private childServices(): ExecServices {
    return {
      ...this.config.services,
      validator: this.validator,
      sessions: this.sessions(),
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
   * They become the template's `{{inputs.*}}` scope — authored render variables ride bound input slots
   * (loader §3.1), so a prompt sees exactly the inputs its operation resolved, nothing more.
   */
  private renderTemplate(template: string, instance: Instance, opInputs?: Record<string, ResolvedValue>): string {
    const base = this.exprContext(instance);
    const ctx = opInputs
      ? { ...base, inputs: { ...(base.inputs as Record<string, unknown>), ...opInputs } }
      : base;
    return template.replace(TEMPLATE_REF, (_m, path: string) => {
      let v: unknown;
      try {
        v = evaluate(this.parse(path), ctx);
      } catch {
        return "";
      }
      if (isPending(v) || v === undefined || v === null) return "";
      if (isArtifactRef(v)) return v.content ?? v.path ?? v.name;
      if (typeof v === "object") return JSON.stringify(v);
      return String(v);
    });
  }

  /** The run's session store — the SINGLE transcript home (an app-provided `services.sessions` wins), shared
   *  with the llm `withSession` path via `ctx.sessions` (childServices). */
  private sessions(): SessionStore {
    return this.config.services?.sessions ?? this.sessionStore;
  }

  /** Read `sessionId`'s transcript from the session store (`SessionState.messages`), mirroring it for
   *  synchronous `{ conversation }` binding resolution (§7.5 — a transcript is addressable DATA). */
  private async readTranscript(sessionId: string): Promise<Turn[]> {
    const state = await this.sessions().get(sessionId);
    const turns = Array.isArray(state?.messages) ? (state.messages as Turn[]) : [];
    this.transcripts.set(sessionId, turns);
    return turns;
  }

  /** Append turns to `sessionId`'s transcript in the session store, preserving any other `SessionState`. */
  private async appendTranscript(sessionId: string, turns: Turn[]): Promise<void> {
    const store = this.sessions();
    const state = (await store.get(sessionId)) ?? {};
    const messages = [...(Array.isArray(state.messages) ? state.messages : []), ...turns];
    await store.put(sessionId, { ...state, messages });
    this.transcripts.set(sessionId, messages as Turn[]);
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
