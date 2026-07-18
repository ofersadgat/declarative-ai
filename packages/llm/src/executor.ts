/**
 * The @ai-exec/core `Executor` for kind "llm-call" — one structured LLM call as a
 * uniform execution unit (DESIGN §3). Wraps the findmyprompt call pipeline
 * (`executeStructuredCall` → `generateStructured`) behind the contract seams:
 * spec/outcome mapping, injected rate limiting, deadline fail-fast, caller cancel,
 * and the opt-in bounded output-repair loop.
 *
 * The actual call is an injectable `CallRunner` (params + deps → `CallOutcome`),
 * defaulting to `executeStructuredCall`. That seam is what makes the executor's
 * control flow (mapping, repair, cancel, deadline, rate-limiter pass-through)
 * testable against a fake runner with no provider or network in play.
 */
import type {
  ExecEvent,
  ExecHandle,
  ExecMetrics,
  ExecServices,
  ExecutionSpec,
  Executor,
  ExecutorCapabilities,
  Outcome,
  ReasoningSpec,
} from "@ai-exec/core";
import { deadlineDecision, estimateCallTokens, systemClock, DEADLINE_FLOOR_REASON } from "@ai-exec/services";
import type { CallMetrics, CallOutcome } from "./generate";
import { executeStructuredCall, type CallDeps, type StructuredCallParams } from "./llmStep";
import { createRouter, type ProviderRouter } from "./router";

/**
 * The serializable definition of one LLM call (`spec.definition` for kind "llm-call"):
 * `StructuredCallParams` minus `schema`/`schemaId` — the output schema comes from
 * `spec.outputSchema`. `spec.inputs` are ignored in v1 (callers pre-render prompts).
 */
export interface LlmCallDefinition {
  modelId: string;
  prompt: string;
  system?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  stopSequences?: string[];
  /** Provider-neutral reasoning request (effort/budget), adapted at the call boundary. */
  reasoning?: ReasoningSpec;
  /** Per-call wall-clock budget (ms). Falls back to `spec.limits.timeoutMs`, then the default. */
  timeoutMs?: number;
}

/** Runner deps: `CallDeps` with the router optional (a fake runner needs no provider). */
export type CallRunnerDeps = Omit<CallDeps, "providers"> & { providers?: ProviderRouter };

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

/** Numeric-summing accumulation of one attempt's metrics into the running total (repair loop). */
function accumulateMetrics(total: ExecMetrics | undefined, m: CallMetrics): ExecMetrics {
  if (!total) return { ...m };
  const sum = (a: number | undefined, b: number | undefined): number | undefined =>
    a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0);
  return {
    inputTokens: sum(total.inputTokens, m.inputTokens),
    outputTokens: sum(total.outputTokens, m.outputTokens),
    noCacheTokens: sum(total.noCacheTokens, m.noCacheTokens),
    cacheReadTokens: sum(total.cacheReadTokens, m.cacheReadTokens),
    cacheWriteTokens: sum(total.cacheWriteTokens, m.cacheWriteTokens),
    cacheWrite1hTokens: sum(total.cacheWrite1hTokens, m.cacheWrite1hTokens),
    reasoningTokens: sum(total.reasoningTokens, m.reasoningTokens),
    totalTokens: sum(total.totalTokens, m.totalTokens),
    cost: sum(total.cost, m.cost),
    costSource: m.costSource ?? total.costSource,
    rawUsage: m.rawUsage ?? total.rawUsage,
    durationMs: (total.durationMs ?? 0) + m.durationMs,
  };
}

/** A validation failure eligible for the repair loop: retriable AND validation-caused. */
function isValidationFailure(outcome: CallOutcome): boolean {
  return outcome.error !== undefined && outcome.error.classification === "api-retriable" && /validation/i.test(outcome.error.reason);
}

/** The repair suffix appended to the prompt after a validation failure. */
function repairSuffix(errors: string): string {
  return `\n\nYour previous output failed schema validation: ${errors}. Return ONLY corrected JSON matching the schema.`;
}

/** An empty, already-completed event stream (v1 emits no events; the seam stays for later). */
function emptyEvents(): AsyncIterable<ExecEvent> {
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

    // Deadline fail-fast (§time-vs-money): below the floor, don't start the call at all.
    let deadlineRemainingMs: number | undefined;
    if (ctx.deadline && ctx.stepStartMs !== undefined) {
      const decision = deadlineDecision(ctx.stepStartMs, ctx.deadline, (ctx.clock ?? systemClock).now());
      if (!decision.proceed) {
        return {
          rawText: "",
          finishReason: "error",
          metrics: { startMs, durationMs: 0 },
          error: {
            classification: "deadline",
            reason: `${DEADLINE_FLOOR_REASON}: ${decision.remainingMs}ms remaining is below the start floor`,
          },
        };
      }
      deadlineRemainingMs = decision.remainingMs;
    }

    // Per-call time budget: definition → spec.limits → default, clamped by the window deadline.
    let timeoutMs = def.timeoutMs ?? spec.limits?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (deadlineRemainingMs !== undefined) timeoutMs = Math.min(timeoutMs, deadlineRemainingMs);

    // Combined abort: internal cancel + the caller's signal (timeout is applied per-call in llmStep).
    const abortSignal = spec.abortSignal ? AbortSignal.any([signal, spec.abortSignal]) : signal;

    const deps: CallRunnerDeps = {
      providers: this.resolveRouter(ctx),
      validator: ctx.validator,
      abortSignal,
    };

    const paramsFor = (prompt: string): StructuredCallParams => ({
      modelId: def.modelId,
      prompt,
      system: def.system,
      schema: spec.outputSchema,
      temperature: def.temperature,
      topP: def.topP,
      topK: def.topK,
      maxOutputTokens: def.maxOutputTokens,
      stopSequences: def.stopSequences,
      reasoning: def.reasoning,
      timeoutMs,
    });

    const invoke = async (prompt: string): Promise<CallOutcome> => {
      const params = paramsFor(prompt);
      const call = (): Promise<CallOutcome> => runner(params, deps);
      let result: CallOutcome;
      if (ctx.rateLimiter) {
        const est = estimateCallTokens(params.prompt, params.system, params.maxOutputTokens);
        result = await ctx.rateLimiter.schedule({ ...est, modelId: def.modelId }, call);
        ctx.rateLimiter.reportOutcome({ rateLimited: result.error?.rateLimited, modelId: def.modelId });
      } else {
        result = await call();
      }
      return result;
    };

    const repairTurns = spec.repairTurns ?? 0;
    let metrics: ExecMetrics | undefined;
    let last: CallOutcome;
    let prompt = def.prompt;
    let turn = 0;
    // Total calls = 1 + at most repairTurns repairs, each triggered only by a validation failure.
    for (;;) {
      try {
        last = await invoke(prompt);
      } catch (err) {
        // The runner contract is never-throw; a throw here is an executor wiring error
        // (missing router, rate-limiter fault). Normalize it into a permanent failure.
        return {
          rawText: "",
          finishReason: "error",
          metrics: { ...accumulateMetrics(metrics, { durationMs: 0 }), startMs },
          error: { classification: "permanent", reason: err instanceof Error ? err.message : String(err) },
        };
      }
      metrics = accumulateMetrics(metrics, last.metrics);
      if (wasCanceled() || abortSignal.aborted) break;
      if (!isValidationFailure(last) || turn >= repairTurns) break;
      turn++;
      prompt = def.prompt + repairSuffix(last.error!.reason);
    }

    const canceled = wasCanceled() || spec.abortSignal?.aborted === true;
    const error = last.error
      ? canceled
        ? { ...last.error, classification: "canceled" as const }
        : last.error
      : undefined;

    return {
      value: last.value,
      rawText: last.rawText,
      thinking: last.thinking,
      finishReason: last.finishReason,
      metrics: { ...metrics!, startMs },
      error,
    };
  }
}

/** Convenience factory mirroring the class constructor. */
export function createLlmCallExecutor(options: LlmCallExecutorOptions = {}): Executor {
  return new LlmCallExecutor(options);
}
