/**
 * The `hierarchical-workflow` executor (DESIGN §7): exposes one workflow run as an
 * ai-exec execution unit. The definition is a state-file bundle; its content IDENTITY is the
 * snapshot hash (SPEC §12), which `workflowDefinitionHash` computes from the bundle. That is a
 * memoization concern — pass it to `withMemoize(cache, { identify: workflowDefinitionHash })` so a
 * workflow run is memoizable under the standard key (DESIGN §3.4) without the caller carrying a
 * `definitionHash` on the generic spec.
 */
import {
  memoKey,
  type ExecEvent,
  type ExecFailure,
  type ExecHandle,
  type ExecMetrics,
  type ExecutionSpec,
  type Executor,
  type ExecutorCapabilities,
  type ExecServices,
  type Outcome,
  type ProducedArtifact,
  type UnitKind,
} from "@declarative-ai/core";
import { WorkflowEngine } from "./engine";
import type { StateDef } from "./format";
import { loadBundle, snapshotHash } from "./loader";
import type { Persistence, ProviderBinding, SkillResolver } from "./ports";
import { validateBundle } from "./validate";

/** The unit definition: raw state files + the root id (as stored/authored). */
export interface HierarchicalWorkflowDefinition {
  rootId: string;
  states: Record<string, StateDef | Record<string, unknown>>;
}

export interface HwExecutorOptions {
  /** `agent.provider` name → executor binding (DESIGN §7). */
  providers: Record<string, ProviderBinding>;
  skills?: SkillResolver;
  persistence?: Persistence;
  /**
   * When a definition contains interactive (ui) states and no `spec.interaction`
   * port is supplied: `"lazy"` (default) fails only if such a state is actually
   * reached — right for interactive apps where e.g. a review state may never fire;
   * `"eager"` fails before spending anything — right for search/batch contexts
   * (DESIGN §3.3), where an unreachable-in-practice guarantee isn't worth money.
   */
  interactionPolicy?: "eager" | "lazy";
}

const CAPABILITIES: ExecutorCapabilities = {
  structuredOutput: true,
  sessionResume: false, // v1: a canceled workflow is re-run (DESIGN §7)
  streaming: true,
  interactive: true, // UI states supported when spec.interaction is supplied
  mutatesWorkspace: false, // becomes true per-definition once process units exist
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

/** The content identity of a workflow spec's definition — its snapshot hash (SPEC §12). This is the
 *  memo-key identity component `withMemoize` needs: `withMemoize(cache, { identify: workflowDefinitionHash })`. */
export function workflowDefinitionHash(spec: ExecutionSpec): string {
  const def = spec.definition as HierarchicalWorkflowDefinition;
  return snapshotHash(loadBundle(def.states as Record<string, unknown>, def.rootId));
}

/** Compute the memo key for a workflow execution (DESIGN §3.4). */
export function workflowMemoKey(spec: ExecutionSpec): string {
  return memoKey({
    kind: spec.kind,
    definitionHash: workflowDefinitionHash(spec),
    inputs: spec.inputs,
    workspaceTreeHash: spec.workspace?.treeHash,
  });
}

class EventQueue {
  private buffer: ExecEvent[] = [];
  private waiters: Array<(v: IteratorResult<ExecEvent>) => void> = [];
  private closed = false;

  push(event: ExecEvent): void {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter({ value: event, done: false });
    else this.buffer.push(event);
  }

  close(): void {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) waiter({ value: undefined as never, done: true });
  }

  iterate(): AsyncIterable<ExecEvent> {
    const self = this;
    return {
      [Symbol.asyncIterator](): AsyncIterator<ExecEvent> {
        return {
          next(): Promise<IteratorResult<ExecEvent>> {
            const buffered = self.buffer.shift();
            if (buffered !== undefined) return Promise.resolve({ value: buffered, done: false });
            if (self.closed) return Promise.resolve({ value: undefined as never, done: true });
            return new Promise((resolve) => self.waiters.push(resolve));
          },
        };
      },
    };
  }
}

export class HierarchicalWorkflowExecutor implements Executor {
  readonly kind: UnitKind = "hierarchical-workflow";
  readonly capabilities = CAPABILITIES;

  constructor(private readonly options: HwExecutorOptions) {}

  start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
    const events = new EventQueue();
    const abort = new AbortController();
    const outcome = this.execute(spec, ctx, events, abort).finally(() => events.close());
    return {
      events: events.iterate(),
      outcome,
      cancel: async () => {
        abort.abort();
        await outcome;
      },
    };
  }

  private async execute(
    spec: ExecutionSpec,
    ctx: ExecServices,
    events: EventQueue,
    abort: AbortController,
  ): Promise<Outcome> {
    const startMs = Date.now();
    const fail = (classification: ExecFailure["classification"], reason: string, metrics?: Partial<ExecMetrics>): Outcome => ({
      metrics: { durationMs: Date.now() - startMs, startMs, ...metrics },
      error: { classification, reason },
    });

    // --- Definition intake -----------------------------------------------------
    const def = spec.definition as HierarchicalWorkflowDefinition | undefined;
    if (!def || typeof def.rootId !== "string" || def.states === null || typeof def.states !== "object") {
      return fail("permanent", "definition must be { rootId, states }");
    }
    let bundle;
    try {
      bundle = loadBundle(def.states as Record<string, unknown>, def.rootId);
    } catch (e) {
      return fail("permanent", `definition failed to load: ${(e as Error).message}`);
    }
    // No caller-supplied `definitionHash` to reconcile: identity is derived from the bundle itself
    // where it's needed (memoization), so there is no stale-hash footgun to guard against here.
    const report = validateBundle(bundle);
    if (report.errors.length > 0) {
      const first = report.errors
        .slice(0, 5)
        .map((e) => `${e.stateId} ${e.path}: ${e.message}`)
        .join("; ");
      return fail("permanent", `workflow validation failed (${report.errors.length} errors): ${first}`);
    }

    // Interactive definitions: eager contexts (search) refuse before any money is
    // spent; lazy contexts fail only if a ui state is actually reached (the engine
    // produces that failure).
    if (this.options.interactionPolicy === "eager" && spec.interaction === undefined) {
      const interactiveStates = interactiveStatesOf(bundle.states);
      if (interactiveStates.length > 0) {
        return fail(
          "permanent",
          `definition contains interactive states (${interactiveStates.join(", ")}) but no interaction port was supplied`,
        );
      }
    }

    if (!ctx.registry) {
      return fail("permanent", "hierarchical-workflow execution requires ctx.registry for child operations");
    }

    // --- Abort / timeout wiring ------------------------------------------------
    let timedOutByLimit = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (spec.limits?.timeoutMs !== undefined) {
      timer = setTimeout(() => {
        timedOutByLimit = true;
        abort.abort();
      }, spec.limits.timeoutMs);
    }
    const onSpecAbort = (): void => abort.abort();
    if (spec.abortSignal?.aborted) abort.abort();
    else spec.abortSignal?.addEventListener("abort", onSpecAbort, { once: true });

    // --- Engine run ------------------------------------------------------------
    const engine = new WorkflowEngine({
      bundle,
      providers: this.options.providers,
      registry: ctx.registry,
      validator: ctx.validator,
      interaction: spec.interaction,
      skills: this.options.skills,
      persistence: this.options.persistence,
      services: ctx,
      clock: ctx.clock,
      onEvent: (event) => {
        // Normalized event surface (DESIGN §3.2): completions carry metrics for
        // budget/cost observers; the rest stream as progress.
        if (event.type === "operation.completed" && event.metrics) {
          events.push({ type: "child_outcome", ref: { kind: this.kind, label: event.stateId }, metrics: event.metrics });
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
      result = await engine.run({ inputs: spec.inputs, abortSignal: abort.signal });
    } catch (e) {
      return fail("permanent", `engine crashed: ${(e as Error).message}`);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      spec.abortSignal?.removeEventListener("abort", onSpecAbort);
    }

    const artifacts: ProducedArtifact[] = result.artifacts.map((a) => ({
      name: a.name,
      content: a.content,
      path: a.path,
      format: a.format,
    }));
    const metrics: ExecMetrics = {
      durationMs: Date.now() - startMs,
      startMs,
      cost: result.metrics.childCost, // composite cost INCLUDES child cost (core contract)
      childCalls: result.metrics.childCalls,
      childCost: result.metrics.childCost,
    };

    if (result.outcome === "timeout" || (result.outcome === "canceled" && timedOutByLimit)) {
      return {
        artifacts,
        metrics,
        error: { classification: "deadline", reason: "workflow exceeded its time limit" },
      };
    }
    if (result.outcome === "canceled") {
      return { artifacts, metrics, error: { classification: "canceled", reason: "workflow canceled" } };
    }
    if (result.outcome === "error" || result.outputs === undefined) {
      return {
        artifacts,
        metrics,
        error: result.failure ?? { classification: "permanent", reason: "workflow terminated with error" },
      };
    }

    // Spec-level output contract (in addition to per-state validation the engine did).
    if (spec.outputSchema && ctx.validator) {
      const res = ctx.validator.validateValue(spec.outputSchema, result.outputs);
      if (!res.ok) {
        return {
          value: result.outputs,
          artifacts,
          metrics,
          error: { classification: "api-retriable", reason: `workflow outputs failed the spec contract: ${res.errors}` },
        };
      }
    }

    return { value: result.outputs, artifacts, metrics };
  }
}

export function createHierarchicalWorkflowExecutor(options: HwExecutorOptions): HierarchicalWorkflowExecutor {
  return new HierarchicalWorkflowExecutor(options);
}

/** State IDs of interactive (ui) states in a bundle's state map. */
export function interactiveStatesOf(states: Record<string, StateDef>): string[] {
  return Object.entries(states)
    .filter(([, s]) => s.ui !== undefined)
    .map(([id]) => id);
}
