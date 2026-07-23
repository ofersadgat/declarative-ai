/**
 * A hierarchical workflow as an `Executor` (DESIGN §7, §3.1).
 *
 * With `ExecutionSpec` and `UnitKind` gone, a workflow run is started by a `FunctionOp` whose bound
 * inputs are the workflow's declared inputs; the BUNDLE is held at construction, which is what it
 * always was in practice — a workflow's identity is its snapshot, not a payload the caller re-supplies
 * per run. Because this is an ordinary `Executor`, the generic wrappers compose around it unchanged:
 * `compose(workflow).with(withMemoize({ cache, identify: workflowIdentify(definition) }))`.
 */
import {
  EventQueue,
  failureOf,
  hashOperation,
  memoKey,
  resolveLiteralInputs,
  type Capabilities,
  type CapabilityRegistry,
  type Failure,
  type ResolvedValue,
  type ExecHandle,
  type ExecServices,
  type Executor,
  type InlineFamily,
  type JsonValue,
  type Operation,
  type ExecResult,
} from "@declarative-ai/exec";
import { WorkflowEngine } from "./engine";
import type { StateDef } from "./format";
import { isByteStream, materialize, MaterializeError } from "./materialize";
import { loadBundle, snapshotHash } from "./loader";
import type { Persistence, WorkflowMetrics } from "./ports";
import { mergeWorkflowMetrics } from "./ports";
import { validateBundle } from "./validate";

/** The workflow definition: raw state files + the root id (as stored/authored). */
export interface HierarchicalWorkflowDefinition {
  rootId: string;
  states: Record<string, StateDef>;
}

export interface WorkflowExecutorOptions {
  /** The authored bundle this executor runs. */
  definition: HierarchicalWorkflowDefinition;
  /** The typed capability registry — `functions` (host code, sub-workflows, and delegated agents alike,
   *  as ONE map of discriminated entries), `skills` (named prompt templates), `tools`. */
  registry: CapabilityRegistry<WorkflowMetrics>;
  /** The executor a `PromptOp` inside the workflow dispatches to (`@declarative-ai/promptop`). Typed
   *  as a plain `Executor`, so hw never learns that a prompt op HAS an llm lowering — which is what
   *  keeps the AI SDK out of this package's dependency graph. */
  prompt?: Executor<ExecServices, WorkflowMetrics>;
  persistence?: Persistence;
}

const CAPABILITIES: Capabilities = {
  structuredOutput: true,
  sessionResume: false, // v1: a canceled workflow is re-run (DESIGN §7)
  streaming: true,
  interactive: true, // supported via interactive `function` states (registry.functions)
  readOnly: false,
  mutatesWorkspace: false, // becomes true per-definition once process units exist
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

/**
 * The content identity of a workflow execution: the bundle's SNAPSHOT hash folded with the op's own
 * hash (which carries the run's resolved inputs). This is the `identify` seam `withMemoize` takes, and
 * it is why `memoize` never has to brute-force-canonicalize an opaque bundle.
 */
export function workflowIdentify(definition: HierarchicalWorkflowDefinition): (op: Operation<InlineFamily>) => string {
  const snapshot = snapshotHash(loadBundle(definition.states, definition.rootId));
  return (op) => `${snapshot}:${hashOperation(op)}`;
}

/** The memo key for a workflow execution, for a caller keying its own cache. */
export function workflowMemoKey(definition: HierarchicalWorkflowDefinition, op: Operation<InlineFamily>, workspaceTreeHash?: string): string {
  return memoKey({ operationHash: workflowIdentify(definition)(op), ...(workspaceTreeHash !== undefined ? { workspaceTreeHash } : {}) });
}

export class WorkflowExecutor implements Executor<ExecServices, WorkflowMetrics> {
  readonly metrics = { merge: mergeWorkflowMetrics };
  readonly capabilities = CAPABILITIES;

  constructor(private readonly options: WorkflowExecutorOptions) {}

  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, WorkflowMetrics> {
    const events = new EventQueue();
    const abort = new AbortController();
    const result = this.execute(op, ctx, events, abort).finally(() => events.close());
    return {
      events: events.iterate(),
      result,
      cancel: async () => {
        abort.abort();
        await result;
      },
    };
  }

  private async execute(
    op: Operation<InlineFamily>,
    ctx: ExecServices,
    events: EventQueue,
    abort: AbortController,
  ): Promise<ExecResult<ResolvedValue, WorkflowMetrics>> {
    const startMs = Date.now();
    const fail = (classification: Failure["classification"], reason: string, metrics?: Partial<WorkflowMetrics>): ExecResult<ResolvedValue, WorkflowMetrics> => ({
      metrics: { durationMs: Date.now() - startMs, startMs, costUsd: 0, costSource: "unknown", ...metrics },
      error: { classification, reason },
    });

    // --- Inputs -----------------------------------------------------------
    const resolved = resolveLiteralInputs(op);
    if ("error" in resolved) return fail("permanent", resolved.error);
    // A blob INPUT that arrived as a live stream must be drained BEFORE the op is hashed for a memo key
    // (§7.3, rule 1): `hashOperation` cannot hash a stream and throws, by design, exactly so this drain
    // happens first. The drain upgrades the op's binding IN PLACE, so the op the outer `withMemoize`
    // hashes carries bytes — and re-running the same (already-drained) op is idempotent. Only runtime
    // inputs are ever streams; an authored document is JSON, so the snapshot hash never sees one.
    try {
      await materializeOpInputs(op, resolved.values, ctx.abortSignal);
    } catch (e) {
      return fail("permanent", e instanceof MaterializeError ? e.message : `input materialization failed: ${(e as Error).message}`);
    }

    // --- Definition intake ------------------------------------------------
    const def = this.options.definition;
    let bundle;
    try {
      bundle = loadBundle(def.states, def.rootId);
    } catch (e) {
      return fail("permanent", `definition failed to load: ${(e as Error).message}`);
    }
    // Validation is a function of *(document, registry)* (§2): with the registry in hand, a
    // `functionRef` naming nothing registered is an authoring error caught before the run rather than
    // a run-fatal surprise partway through it.
    const report = validateBundle(bundle, { functions: this.options.registry.functions });
    if (report.errors.length > 0) {
      const first = report.errors
        .slice(0, 5)
        .map((e) => `${e.stateId} ${e.path}: ${e.message}`)
        .join("; ");
      return fail("permanent", `workflow validation failed (${report.errors.length} errors): ${first}`);
    }

    // --- Abort / timeout wiring -------------------------------------------
    let timedOutByLimit = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (ctx.timeoutMs !== undefined) {
      timer = setTimeout(() => {
        timedOutByLimit = true;
        abort.abort();
      }, ctx.timeoutMs);
    }
    const onCtxAbort = (): void => abort.abort();
    if (ctx.abortSignal?.aborted) abort.abort();
    else ctx.abortSignal?.addEventListener("abort", onCtxAbort, { once: true });

    // --- Engine run --------------------------------------------------------
    const engine = new WorkflowEngine({
      bundle,
      registry: this.options.registry,
      prompt: this.options.prompt,
      validator: ctx.validator,
      persistence: this.options.persistence,
      services: ctx,
      clock: ctx.clock,
      onEvent: (event) => {
        // Normalized event surface (DESIGN §3.2): completions carry metrics for budget/cost observers;
        // the rest stream as progress.
        if (event.type === "operation.completed" && event.metrics) {
          events.push({ type: "child_result", ref: { label: event.stateId }, metrics: event.metrics });
        } else if (event.type === "transition.taken") {
          events.push({ type: "progress", message: `${event.stateId} → ${event.to} (iteration ${event.iteration})` });
        } else if (event.type === "instance.entered") {
          events.push({ type: "progress", message: `entered ${event.stateId}` });
        } else if (event.type === "instance.terminated") {
          events.push({ type: "progress", message: `${event.stateId} terminated: ${event.outcome}` });
        }
      },
    });

    let result;
    try {
      result = await engine.run({ inputs: resolved.values, abortSignal: abort.signal });
    } catch (e) {
      // A crashed engine may already have run children that SPENT MONEY, so the spend accumulated up to
      // the crash is reported rather than zeroed — `costUsd: 0` here would silently forgive real charges.
      const spent = engine.spentSoFar();
      return {
        error: failureOf(e, "engine crashed"),
        metrics: {
          durationMs: Date.now() - startMs,
          startMs,
          costUsd: spent.childCost,
          costSource: spent.childCost > 0 ? "table" : "unknown",
          childLlmCalls: spent.childLlmCalls,
          childCostUsd: spent.childCost,
        },
      };
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      ctx.abortSignal?.removeEventListener("abort", onCtxAbort);
    }

    const metrics: WorkflowMetrics = {
      durationMs: Date.now() - startMs,
      startMs,
      // A composite's cost INCLUDES its children's (the exec contract).
      costUsd: result.metrics.childCost,
      costSource: "table",
      childLlmCalls: result.metrics.childLlmCalls,
      childCostUsd: result.metrics.childCost,
    };

    if (result.outcome === "timeout" || (result.outcome === "canceled" && timedOutByLimit)) {
      return { metrics, error: { classification: "deadline", reason: "workflow exceeded its time limit" } };
    }
    if (result.outcome === "canceled") {
      return { metrics, error: { classification: "canceled", reason: "workflow canceled" } };
    }
    if (result.outcome === "error" || result.outputs === undefined) {
      return { metrics, error: result.failure ?? { classification: "permanent", reason: "workflow terminated with error" } };
    }

    // The run's RESULT is stored/returned to the caller (§7.3, rule 3), so a blob output that reached
    // this boundary as a live STREAM — a single-consumer pipe that ran all the way to the top — is
    // drained here. This is the one place that knows the value is the terminal result rather than an
    // intermediate to pipe onward. A drain failure is a NON-RETRIABLE result failure, context-named.
    let outputs: Record<string, ResolvedValue>;
    try {
      outputs = await materializeResultOutputs(result.outputs, abort.signal);
    } catch (e) {
      return { metrics, error: { classification: "permanent", reason: e instanceof MaterializeError ? e.message : `output materialization failed: ${(e as Error).message}` } };
    }

    // The op-level output contract (in addition to per-state validation the engine did).
    const outputSchema = op.output.schema;
    if (outputSchema && ctx.validator) {
      const res = ctx.validator.validateValue(outputSchema, (outputs ?? null) as JsonValue);
      if (!res.ok) {
        return {
          value: outputs as JsonValue | undefined,
          metrics,
          error: { classification: "api-retriable", reason: `workflow outputs failed the operation's contract: ${res.errors}` },
        };
      }
    }

    return { value: (outputs ?? null) as ResolvedValue, metrics };
  }
}

export function createWorkflowExecutor(options: WorkflowExecutorOptions): WorkflowExecutor {
  return new WorkflowExecutor(options);
}

/**
 * Drain any blob input bound to a live stream, upgrading the op's binding IN PLACE and mirroring the
 * bytes into the already-resolved `values` the engine will run with (§7.3, rule 1). In place because
 * the op object is the very one `withMemoize` hashes: replacing the stream on its binding is what lets
 * `hashOperation` see bytes instead of throwing. A `Uint8Array` binding is left untouched.
 */
async function materializeOpInputs(op: Operation<InlineFamily>, values: Record<string, ResolvedValue>, signal: AbortSignal | undefined): Promise<void> {
  for (const [name, param] of Object.entries(op.input)) {
    const binding = param.binding;
    if (binding === undefined || !("blob" in binding) || !isByteStream(binding.blob)) continue;
    const bytes = await materialize(binding.blob, signal, `operation input '${name}'`);
    binding.blob = bytes;
    values[name] = bytes;
  }
}

/**
 * Drain any blob RESULT output that reached the run boundary as a live stream (§7.3, rule 3). Copies
 * lazily — only when a stream is actually present — so the common all-bytes result is returned as-is.
 */
async function materializeResultOutputs(outputs: Record<string, ResolvedValue>, signal: AbortSignal | undefined): Promise<Record<string, ResolvedValue>> {
  let copy: Record<string, ResolvedValue> | undefined;
  for (const [name, value] of Object.entries(outputs)) {
    if (!isByteStream(value)) continue;
    copy ??= { ...outputs };
    copy[name] = await materialize(value, signal, `workflow output '${name}'`);
  }
  return copy ?? outputs;
}
