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
  type ExecFailure,
  type ExecMetrics,
  type ExecServices,
  type ExecutionSpec,
  type ExecutorRegistry,
  type InteractionPort,
  type OutputValidator,
  type Clock,
  hashCanonical,
} from "@ai-exec/core";
import { SchemaValidator } from "@ai-exec/services";
import { evaluate, isPending, parseExpression, PENDING, type Expr } from "./expr";
import type {
  AgentConfig,
  ChildDecl,
  ConversationMode,
  FieldSchema,
  OutputFieldSchema,
  StateDef,
  TerminationOutcome,
  WorkflowBundle,
} from "./format";
import {
  isArtifactRef,
  type ArtifactRef,
  type EngineEvent,
  type Persistence,
  type ProviderBinding,
  type SkillResolver,
} from "./ports";

export interface EngineConfig {
  bundle: WorkflowBundle;
  /** `agent.provider` name → executor binding. */
  providers: Record<string, ProviderBinding>;
  registry: ExecutorRegistry;
  validator?: OutputValidator;
  interaction?: InteractionPort;
  skills?: SkillResolver;
  persistence?: Persistence;
  /** Forwarded to child executors (rate limiter, meter, ...). `registry`/`validator`
   *  are supplied by the engine. */
  services?: ExecServices;
  clock?: Clock;
  onEvent?: (event: EngineEvent) => void;
  /** Default repair turns for agent/skill operations (see core ExecutionSpec.repairTurns). */
  repairTurns?: number;
}

export interface WorkflowRunOptions {
  inputs: Record<string, unknown>;
  abortSignal?: AbortSignal;
}

export interface WorkflowRunResult {
  outcome: TerminationOutcome;
  outputs?: Record<string, unknown>;
  failure?: ExecFailure;
  artifacts: ArtifactRef[];
  metrics: { childCalls: number; childCost: number; durationMs: number };
}

interface TerminationRecord {
  outcome: TerminationOutcome;
  outputs?: Record<string, unknown>;
  failure?: ExecFailure;
}

interface ChildRecord {
  instanceId: number;
  status: "running" | "done";
  outcome?: TerminationOutcome;
  outputs?: Record<string, unknown>;
  abort: AbortController;
  promise: Promise<void>;
}

/** One state instance (SPEC §3.4) — results never leak across instances. */
interface Instance {
  id: number;
  stateId: string;
  def: StateDef;
  childKey?: string;
  parent?: Instance;
  inputs: Record<string, unknown>;
  params: Record<string, unknown>;
  /** Operation-produced outputs accumulated so far. */
  outputs: Record<string, unknown>;
  /** Raw ui component results (`ui.*` namespace, SPEC §6.1). */
  uiResults: Record<string, unknown>;
  iteration: number;
  opsRun: { ui: boolean; agent: boolean; skill: boolean };
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

export class WorkflowEngine {
  private readonly validator: OutputValidator;
  private readonly clock: Clock;
  private nextInstanceId = 1;
  /** A run-level configuration failure (e.g. no InteractionPort configured at all):
   *  aborts the whole run rather than looping as a state-level outcome a transition
   *  might keep re-entering. */
  private fatal?: ExecFailure;
  private rootAbort?: AbortController;
  private readonly artifacts: ArtifactRef[] = [];
  private readonly conversation: Array<{ role: "user" | "assistant"; content: string }> = [];
  private childCalls = 0;
  private childCost = 0;

  constructor(private readonly config: EngineConfig) {
    this.validator = config.validator ?? new SchemaValidator();
    this.clock = config.clock ?? { now: () => Date.now() };
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
      metrics: { childCalls: this.childCalls, childCost: this.childCost, durationMs: this.clock.now() - start },
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
    def: StateDef,
    inputs: Record<string, unknown>,
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
      params: defaultsOf(def.params),
      outputs: {},
      uiResults: {},
      iteration: 0,
      opsRun: { ui: false, agent: false, skill: false },
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

      // (2)/(5) Highest-priority unrun operation: ui, agent, skill, sequence children.
      if (def.ui && !instance.opsRun.ui) {
        instance.opsRun.ui = true;
        const failure = await this.runUiOperation(instance);
        if (failure) return this.finish(instance, failureOutcome(instance), failure);
        evaluationDue = true;
        continue;
      }
      if (def.agent && !instance.opsRun.agent) {
        instance.opsRun.agent = true;
        const failure = await this.runAgentOperation(instance, def.agent);
        if (failure) return this.finish(instance, failureOutcome(instance), failure);
        evaluationDue = true;
        continue;
      }
      if (def.skill && !instance.opsRun.skill) {
        instance.opsRun.skill = true;
        const failure = await this.runSkillOperation(instance);
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

  private finish(instance: Instance, outcome: TerminationOutcome, failure?: ExecFailure): TerminationRecord {
    if (outcome !== "success") {
      return { outcome, failure };
    }
    // Resolve declared outputs (SPEC §3.7: resolved when the state terminates).
    const ctx = this.exprContext(instance);
    const outputs: Record<string, unknown> = {};
    for (const [name, schema] of Object.entries(instance.def.outputs ?? {})) {
      let value: unknown;
      if (schema.from !== undefined) {
        const v = evaluate(this.parse(schema.from), ctx);
        value = isPending(v) ? undefined : v; // canceled async children resolve to nothing
      } else {
        value = instance.outputs[name];
      }
      if (value === undefined) {
        if (!schema.optional) {
          return {
            outcome: "error",
            failure: { classification: "permanent", reason: `required output '${name}' was not produced` },
          };
        }
        continue;
      }
      if (schema.type !== "passthrough") {
        const err = this.validateFieldValue(name, schema, value);
        if (err) return { outcome: "error", failure: { classification: "permanent", reason: err } };
      }
      outputs[name] = value;
    }
    return { outcome: "success", outputs };
  }

  // --- expression context ---------------------------------------------------

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
      params: instance.params,
      ui: instance.uiResults,
      children,
      run: { iteration: instance.iteration },
      limits: { ...(instance.def.limits ?? {}) },
      artifacts,
      conversations: {},
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
    def: StateDef,
    provided: Record<string, unknown>,
  ): { values: Record<string, unknown> } | { error: string } {
    const values: Record<string, unknown> = {};
    for (const [name, schema] of Object.entries(def.inputs ?? {})) {
      let v = provided[name];
      if (v === undefined) v = schema.default;
      if (v === undefined) {
        if (schema.optional) continue;
        return { error: `required input '${name}' missing` };
      }
      const err = this.validateFieldValue(name, schema, v);
      if (err) return { error: err };
      values[name] = v;
    }
    return { values };
  }

  /** PENDING ⇒ parked (dataflow join); `{error}` ⇒ blocked; else resolved values. */
  private resolveChildInputs(
    instance: Instance,
    decl: ChildDecl,
  ): typeof PENDING | { values?: Record<string, unknown>; error?: string } {
    const childDef = this.config.bundle.states[decl.state];
    if (!childDef) return { error: `unknown state '${decl.state}'` };
    const ctx = this.exprContext(instance);
    const values: Record<string, unknown> = {};
    for (const [name, schema] of Object.entries(childDef.inputs ?? {})) {
      const wiring = decl.inputs?.[name];
      let v: unknown;
      if (wiring === undefined) {
        v = schema.default;
      } else if (typeof wiring === "string") {
        v = evaluate(this.parse(wiring), ctx);
        if (isPending(v)) return PENDING;
      } else {
        v = wiring.value;
      }
      if (v === undefined) v = schema.default;
      if (v === undefined) {
        if (schema.optional) continue;
        return { error: `${decl.state}: required input '${name}' missing` };
      }
      const err = this.validateFieldValue(name, schema, v, decl.state);
      if (err) return { error: err };
      values[name] = v;
    }
    return { values };
  }

  private validateFieldValue(name: string, schema: FieldSchema, value: unknown, statePrefix?: string): string | undefined {
    const label = statePrefix ? `${statePrefix}: '${name}'` : `'${name}'`;
    if (schema.type === "artifact") {
      if (!isArtifactRef(value) && typeof value !== "string") {
        return `${label} expects an artifact (or inline string content)`;
      }
      return undefined;
    }
    if (schema.type === "passthrough") return undefined;
    const doc: Record<string, unknown> = { type: schema.type };
    if (schema.enum) doc["enum"] = schema.enum;
    if (schema.items) doc["items"] = schema.items;
    if (schema.properties) doc["properties"] = schema.properties;
    if (schema.required) doc["required"] = schema.required;
    const res = this.validator.validateValue(doc, value);
    return res.ok ? undefined : `${label} failed validation: ${res.errors ?? "invalid"}`;
  }

  // --- operations -----------------------------------------------------------

  private async runUiOperation(instance: Instance): Promise<ExecFailure | undefined> {
    const ui = instance.def.ui!;
    this.emit({ type: "operation.started", instanceId: instance.id, stateId: instance.stateId, op: "ui" });
    const port = this.config.interaction;
    if (!port) {
      // Run-fatal, not a state outcome: a transition could otherwise keep re-entering
      // the ui state (e.g. §7.3's blocked → human_review) and loop forever.
      const failure: ExecFailure = {
        classification: "permanent",
        reason: `state '${instance.stateId}' requires user interaction but no InteractionPort was supplied`,
      };
      this.emit({ type: "operation.failed", instanceId: instance.id, stateId: instance.stateId, op: "ui", failure });
      this.fatal = failure;
      this.rootAbort?.abort();
      return failure;
    }
    let result: unknown;
    try {
      result = await port.request({
        stateId: instance.stateId,
        component: ui.component,
        inputs: { config: ui, inputs: shallowRedactArtifacts(instance.inputs, true) },
      });
    } catch (e) {
      const failure: ExecFailure = { classification: "permanent", reason: `interaction rejected: ${(e as Error).message}` };
      this.emit({ type: "operation.failed", instanceId: instance.id, stateId: instance.stateId, op: "ui", failure });
      return failure;
    }
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles
    const failure = this.acceptOpOutputs(instance, "ui", result);
    if (failure) return failure;
    if (result !== null && typeof result === "object") Object.assign(instance.uiResults, result);
    this.emit({ type: "operation.completed", instanceId: instance.id, stateId: instance.stateId, op: "ui" });
    return undefined;
  }

  private async runAgentOperation(instance: Instance, agent: AgentConfig): Promise<ExecFailure | undefined> {
    return this.runExecutorOperation(instance, "agent", {
      provider: agent.provider,
      template: agent.prompt?.template ?? "",
      config: agent.config ?? {},
      conversationMode: agent.conversation?.mode ?? "full_history",
      conversationArtifacts: agent.conversation?.artifacts,
    });
  }

  private async runSkillOperation(instance: Instance): Promise<ExecFailure | undefined> {
    const skillCfg = instance.def.skill!;
    const skill = this.config.skills?.get(skillCfg.name);
    if (!skill) {
      const failure: ExecFailure = {
        classification: "permanent",
        reason: `skill '${skillCfg.name}' is not registered`,
      };
      this.emit({ type: "operation.failed", instanceId: instance.id, stateId: instance.stateId, op: "skill", failure });
      return failure;
    }
    return this.runExecutorOperation(instance, "skill", {
      provider: skill.provider,
      template: skill.template,
      config: { ...skill.config, ...skillCfg.params },
      conversationMode: "fresh",
    });
  }

  private async runExecutorOperation(
    instance: Instance,
    op: "agent" | "skill",
    req: {
      provider: string;
      template: string;
      config: Record<string, unknown>;
      conversationMode: ConversationMode;
      conversationArtifacts?: string[];
    },
  ): Promise<ExecFailure | undefined> {
    this.emit({ type: "operation.started", instanceId: instance.id, stateId: instance.stateId, op });
    const fail = (failure: ExecFailure): ExecFailure => {
      this.emit({ type: "operation.failed", instanceId: instance.id, stateId: instance.stateId, op, failure });
      return failure;
    };

    const binding = this.config.providers[req.provider];
    if (!binding) return fail({ classification: "permanent", reason: `no provider binding for '${req.provider}'` });
    const executor = this.config.registry.get(binding.kind);
    if (!executor) return fail({ classification: "permanent", reason: `no executor registered for kind '${binding.kind}'` });

    const rendered = this.renderTemplate(req.template, instance);
    const preamble = this.conversationPreamble(req.conversationMode, req.conversationArtifacts);
    const prompt = preamble ? `${preamble}\n\n${rendered}` : rendered;

    const produced = this.producedOutputFields(instance.def);
    const outputSchema = buildOutputSchema(produced);
    const definition = binding.definition({ prompt, config: req.config });
    const spec: ExecutionSpec = {
      kind: binding.kind,
      definition,
      definitionHash: hashCanonical(definition ?? null),
      inputs: {},
      outputSchema,
      limits: instance.def.limits?.timeout !== undefined ? { timeoutMs: instance.def.limits.timeout * 1000 } : undefined,
      abortSignal: instance.abort.signal,
      repairTurns: this.config.repairTurns,
    };

    const services: ExecServices = {
      ...this.config.services,
      registry: this.config.registry,
      validator: this.validator,
    };

    let outcome;
    try {
      outcome = await executor.start(spec, services).outcome;
    } catch (e) {
      // Executors must not reject for unit failures; a rejection is an executor bug —
      // normalized here so the workflow still degrades per SPEC §3.3.
      return fail({ classification: "permanent", reason: `executor rejected: ${(e as Error).message}` });
    }
    this.childCalls += 1 + (outcome.metrics.childCalls ?? 0);
    this.childCost += outcome.metrics.cost ?? 0;
    if (instance.abort.signal.aborted || instance.timedOut) return undefined; // loop top handles

    // Conversation artifact (SPEC §4.7): every agent operation appends its exchange.
    this.conversation.push({ role: "user", content: prompt });
    this.conversation.push({ role: "assistant", content: outcome.rawText ?? JSON.stringify(outcome.value ?? null) });

    if (outcome.error) {
      return fail(outcome.error);
    }
    const failure = this.acceptOpOutputs(instance, op, outcome.value);
    if (failure) return fail(failure);
    this.emit({ type: "operation.completed", instanceId: instance.id, stateId: instance.stateId, op, metrics: outcome.metrics });
    return undefined;
  }

  /**
   * Merge an operation's structured result into the instance outputs (SPEC §3.3 step 3:
   * "its outputs are validated"). Provided fields are type-checked; artifact-typed
   * fields arrive as inline content and are registered as artifacts. Once the last
   * output-producing operation of the state has run, required produced fields must all
   * be present.
   */
  private acceptOpOutputs(instance: Instance, op: "ui" | "agent" | "skill", value: unknown): ExecFailure | undefined {
    const produced = this.producedOutputFields(instance.def);
    const record = value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
    for (const [name, schema] of Object.entries(produced)) {
      let v = record[name];
      if (v === undefined) continue;
      if (schema.type === "artifact" && typeof v === "string") {
        const ref: ArtifactRef = {
          artifact: true,
          name: `${instance.stateId.replace(/\//g, ".")}#${instance.id}.${name}`,
          format: schema.format,
          content: v,
        };
        this.artifacts.push(ref);
        v = ref;
      } else {
        const err = this.validateFieldValue(name, schema, v);
        if (err) return { classification: "api-retriable", reason: `operation output ${err}` };
      }
      instance.outputs[name] = v;
    }
    const remaining =
      (instance.def.ui && !instance.opsRun.ui) ||
      (instance.def.agent && !instance.opsRun.agent) ||
      (instance.def.skill && !instance.opsRun.skill);
    if (!remaining) {
      for (const [name, schema] of Object.entries(produced)) {
        if (!schema.optional && instance.outputs[name] === undefined) {
          return { classification: "api-retriable", reason: `${op} operations did not produce required output '${name}'` };
        }
      }
    }
    return undefined;
  }

  /** Output fields produced by operations: everything not derived via `from` and not passthrough. */
  private producedOutputFields(def: StateDef): Record<string, OutputFieldSchema> {
    const out: Record<string, OutputFieldSchema> = {};
    for (const [name, schema] of Object.entries(def.outputs ?? {})) {
      if (schema.from === undefined && schema.type !== "passthrough") out[name] = schema;
    }
    return out;
  }

  // --- prompts & conversation -----------------------------------------------

  /** `{{path.to.value}}` interpolation against the instance context. Artifact refs
   *  render as their content; arrays/objects as JSON. */
  private renderTemplate(template: string, instance: Instance): string {
    const ctx = this.exprContext(instance);
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

  private conversationPreamble(mode: ConversationMode, artifactNames?: string[]): string {
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
        if (this.conversation.length === 0) return "";
        const lines = this.conversation.map((m) => `${m.role}: ${m.content}`);
        return `<conversation-history>\n${lines.join("\n")}\n</conversation-history>`;
      }
    }
  }
}

// --- helpers -----------------------------------------------------------------

function defaultsOf(fields: Record<string, FieldSchema> | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [name, schema] of Object.entries(fields ?? {})) {
    if (schema.default !== undefined) out[name] = schema.default;
  }
  return out;
}

/** The state failed while flagged canceled/timed-out? Loop top decides; here we always report error. */
function failureOutcome(instance: Instance): TerminationOutcome {
  if (instance.timedOut) return "timeout";
  if (instance.abort.signal.aborted) return "canceled";
  return "error";
}

/** Failure payload for an author-directed `terminate.error` transition. */
function errorOf(instance: Instance, target: string): ExecFailure {
  return { classification: "permanent", reason: `state '${instance.stateId}' transitioned to ${target}` };
}

function buildOutputSchema(fields: Record<string, OutputFieldSchema>): Record<string, unknown> | undefined {
  const names = Object.keys(fields);
  if (names.length === 0) return undefined;
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const [name, schema] of Object.entries(fields)) {
    if (schema.type === "artifact") {
      properties[name] = {
        type: "string",
        description: `Artifact content${schema.format ? ` (${schema.format})` : ""}.`,
      };
    } else {
      const doc: Record<string, unknown> = { type: schema.type };
      if (schema.enum) doc["enum"] = schema.enum;
      if (schema.items) doc["items"] = schema.items;
      if (schema.properties) doc["properties"] = schema.properties;
      if (schema.description) doc["description"] = schema.description;
      properties[name] = doc;
    }
    if (!schema.optional) required.push(name);
  }
  return { type: "object", properties, required, additionalProperties: true };
}

/** Keep event payloads readable: inline artifact contents are elided (or kept, for UI display). */
function shallowRedactArtifacts(values: Record<string, unknown>, keepContent = false): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(values)) {
    out[k] = isArtifactRef(v) && !keepContent ? { artifact: true, name: v.name, format: v.format } : v;
  }
  return out;
}
