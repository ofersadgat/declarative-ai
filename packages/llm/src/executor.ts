/**
 * The @declarative-ai/core `Executor` for kind "llm-call" — the MINIMAL core unit: ONE structured LLM
 * call mapped onto the contract's `Outcome` (spec/outcome mapping + caller cancel), and nothing else.
 * Cross-cutting concerns — repair, rate limiting, deadline fail-fast, memoization — are NOT here; they are
 * composable `ExecutorWrapper`s (see ./wrappers) stacked around this core via `composeExecutors`. Keeping
 * the unit small is the point: it delivers exactly its value and lets the wrappers deliver theirs.
 *
 * The actual call is an injectable `CallRunner` (params + deps → never-throwing `CallOutcome`), defaulting
 * to `executeStructuredCall`. That seam makes the mapping/cancel control flow testable with no provider.
 */
import type {
  ExecEvent,
  ExecHandle,
  ExecServices,
  ExecutionSpec,
  Executor,
  ExecutorCapabilities,
  Outcome,
} from "@declarative-ai/core";
import { systemClock } from "@declarative-ai/services";
import type { CallOutcome } from "./generate";
import { executeStructuredCall, type CallDeps, type LlmCallDefinition, type LlmCallEnvironment, type StructuredCallParams } from "./llmStep";
import { createRouter, type ProviderRouter } from "./router";

// The serializable `spec.definition` for kind "llm-call" is `@declarative-ai/llm`'s `LlmCallDefinition`
// (defined with `StructuredCallParams` in ./llmStep). Re-exported here because this is the executor that
// consumes it as `spec.definition`.
export type { LlmCallDefinition } from "./llmStep";

/** Runner deps: the call environment with the router optional (a fake runner needs no provider) — which
 *  is exactly {@link LlmCallEnvironment} (`CallDeps` is the environment with `providers` required). */
export type CallRunnerDeps = LlmCallEnvironment;

/** The injectable call seam: one structured call, params → never-throwing `CallOutcome`. */
export type CallRunner = (params: StructuredCallParams, deps: CallRunnerDeps) => Promise<CallOutcome>;

const defaultRunner: CallRunner = (params, deps) => {
  if (!deps.providers) throw new Error("llm-call executor: no ProviderRouter available (ctx.providers or options.router)");
  return executeStructuredCall(params, deps as CallDeps);
};

/** Default per-call timeout when neither the definition nor `spec.limits` names one (10 min). */
const DEFAULT_TIMEOUT_MS = 10 * 60_000;

export interface LlmCallExecutorOptions {
  /** Explicit router; else `ctx.providers` (when it looks like a router), else a lazy env-key router. */
  router?: ProviderRouter;
  /** The call seam; defaults to the real `executeStructuredCall` pipeline. */
  runner?: CallRunner;
}

const CAPABILITIES: ExecutorCapabilities = {
  structuredOutput: true,
  sessionResume: false,
  streaming: true,
  interactive: false,
  mutatesWorkspace: false,
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

/** Duck-type check that `ctx.providers` (typed `unknown` in core) is a `ProviderRouter`. */
function asRouter(candidate: unknown): ProviderRouter | undefined {
  if (candidate && typeof candidate === "object" && typeof (candidate as ProviderRouter).resolveModel === "function") {
    return candidate as ProviderRouter;
  }
  return undefined;
}

/** An empty, already-completed event stream (v1 emits no events; the seam stays for later). Shared with the
 *  wrappers, which build handles too. */
export function emptyEvents(): AsyncIterable<ExecEvent> {
  return {
    async *[Symbol.asyncIterator]() {
      /* no events in v1 */
    },
  };
}

export class LlmCallExecutor implements Executor {
  readonly kind = "llm-call" as const;
  readonly capabilities = CAPABILITIES;

  private envRouter: ProviderRouter | undefined;

  constructor(private readonly options: LlmCallExecutorOptions = {}) {}

  start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
    const internal = new AbortController();
    let cancelRequested = false;

    const outcome = this.run(spec, ctx, internal.signal, () => cancelRequested);

    return {
      events: emptyEvents(),
      outcome,
      cancel: async () => {
        cancelRequested = true;
        internal.abort();
        await outcome;
      },
    };
  }

  private resolveRouter(ctx: ExecServices): ProviderRouter | undefined {
    if (this.options.router) return this.options.router;
    const injected = asRouter(ctx.providers);
    if (injected) return injected;
    // A custom runner needs no router at all — never force env keys on it.
    if (this.options.runner) return undefined;
    this.envRouter ??= createRouter();
    return this.envRouter;
  }

  private async run(
    spec: ExecutionSpec,
    ctx: ExecServices,
    signal: AbortSignal,
    wasCanceled: () => boolean,
  ): Promise<Outcome> {
    const startMs = (ctx.clock ?? systemClock).now();
    const def = spec.definition as LlmCallDefinition;
    const runner = this.options.runner ?? defaultRunner;
    const refuse = (reason: string): Outcome => ({
      rawText: "",
      finishReason: "error",
      metrics: { startMs, durationMs: 0 },
      error: { classification: "permanent", reason },
    });

    // LOUD-FAILURE contract: fields this bare core does NOT implement must never arrive silently. Each
    // wrapper CONSUMES (and strips) its own field, so anything still present here means the caller relied
    // on behavior that is not composed into the stack — refuse instead of silently degrading.
    if (ctx.deadline !== undefined) {
      return refuse(
        "ctx.deadline is set, but the bare llm-call core does not implement deadline handling — compose withDeadline() around it (the wrapper consumes ctx.deadline/ctx.stepStartMs)",
      );
    }
    if (def.sessionId !== undefined || def.providerSessionId !== undefined) {
      return refuse(
        "the declaration carries sessionId/providerSessionId, but no session layer consumed it — compose withSession(...) around the core (the wrapper resolves and strips the session fields)",
      );
    }

    // Per-call time budget: the definition's own `timeoutMs`, else the spec limit (which may exceed the
    // default — a caller-declared budget is honored, not clamped), else the default. A definition budget
    // ABOVE the spec limit is a conflict the caller must resolve — refused, never silently clamped.
    // (`withDeadline` lowers BOTH the limit and the definition budget to the remaining window, so a
    // deadline clamp never manufactures this conflict.)
    if (def.timeoutMs !== undefined && spec.limits?.timeoutMs !== undefined && def.timeoutMs > spec.limits.timeoutMs) {
      return refuse(
        `definition timeoutMs (${def.timeoutMs}ms) exceeds spec.limits.timeoutMs (${spec.limits.timeoutMs}ms) — lower the definition's budget or raise the limit`,
      );
    }
    const timeoutMs = def.timeoutMs ?? spec.limits?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

    // Combined abort: internal cancel + the caller's signal (the per-call timeout is applied in llmStep).
    const abortSignal = spec.abortSignal ? AbortSignal.any([signal, spec.abortSignal]) : signal;

    const deps: CallRunnerDeps = {
      providers: this.resolveRouter(ctx),
      validator: ctx.validator,
      abortSignal,
      blobs: ctx.blobs,
    };

    // Spread the whole definition (sampling-XOR-reasoning config + prompt) so every serializable field
    // carries through without re-listing; layer on the output schema (from spec, v1) + resolved timeout.
    const params: StructuredCallParams = { ...def, schema: spec.outputSchema, timeoutMs };

    let last: CallOutcome;
    try {
      last = await runner(params, deps);
    } catch (err) {
      // The runner contract is never-throw; a throw here is a wiring error (missing router). Normalize it.
      return {
        rawText: "",
        finishReason: "error",
        metrics: { startMs, durationMs: 0 },
        error: { classification: "permanent", reason: err instanceof Error ? err.message : String(err) },
      };
    }

    const canceled = wasCanceled() || spec.abortSignal?.aborted === true;
    const error = last.error ? (canceled ? { ...last.error, classification: "canceled" as const } : last.error) : undefined;

    return {
      value: last.value,
      rawText: last.rawText,
      thinking: last.thinking,
      toolCalls: last.toolCalls,
      toolResults: last.toolResults,
      artifacts: last.artifacts,
      finishReason: last.finishReason,
      metrics: { ...last.metrics, startMs },
      error,
    };
  }
}

/** Convenience factory mirroring the class constructor — the BARE core (no wrappers). Compose the
 *  cross-cutting behaviors you want with `composeExecutors(core, withRateLimit(...), withRepair(), ...)`. */
export function createLlmCallExecutor(options: LlmCallExecutorOptions = {}): Executor {
  return new LlmCallExecutor(options);
}
