/**
 * Dispatch a `PromptOp` by its model's ROUTE PREFIX (DESIGN §3.1, §4.4).
 *
 * The second dispatcher, and deliberately the same shape as the first: `OperationExecutor` reads
 * `op.kind` and hands the op to one of the two executors it holds; this reads the `{route}/…` prefix
 * off `op.config.model` and hands it to one of the prompt executors it holds. Neither is a subclass of
 * what it dispatches to — each delegates before any call machinery runs, so inheriting that machinery
 * would give it a member that overrides everything it inherits.
 *
 * **This is not a second copy of `ModelRouter`.** Provider routing already lives inside
 * `resolveModel`, with lazy client construction, per-server caching and managed-server supervision,
 * and re-dispatching `anthropic`/`openrouter`/`local`/`embedded` above it would be strictly worse.
 * What `ModelRouter` cannot express is a route whose EXECUTOR differs rather than whose provider
 * client differs: `resolveModel` must return an AI-SDK `LanguageModel`, and a coding-agent subprocess
 * is not one. That gap is the whole of this file.
 *
 * Which is also why the split happens ABOVE `llm`: `MODEL_ROUTES` stays a closed set and
 * `parseModelRoute` keeps throwing on anything it does not know, rather than growing agent transports
 * it could never return a model handle for. A prefix this executor recognizes is consumed here; every
 * other id is passed DOWN UNTOUCHED, so the provider path still sees the id its own parser expects.
 */
import type { Capabilities, InlineFamily, MetricsAlgebra, Operation, ResolvedValue } from "@declarative-ai/exec";
import { finishedHandle, type ExecHandle, type ExecServices, type Executor } from "@declarative-ai/exec";
import { emptyLlmMetrics, mergeLlmMetrics, type LlmMetrics } from "@declarative-ai/llm";

export interface PromptRouterExecutorOptions {
  /**
   * Prefix → executor. A key is the part before the FIRST `/` of a model id (`claude-cli`,
   * `anthropic`), without the slash.
   */
  routes: Record<string, Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, ResolvedValue>>;
  /**
   * Where an id whose prefix names no route goes — in practice the provider executor, which owns
   * every prefix `MODEL_ROUTES` knows and produces the authoritative error for one it does not.
   *
   * Absent ⇒ an unmatched prefix is refused here instead, naming the routes that ARE wired. That is
   * the honest answer for a stack with no provider path at all: silently trying one would report a
   * missing API key for a model the caller never meant to send anywhere.
   */
  fallback?: Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, ResolvedValue>;
  /** Override the advertised capabilities. Defaults to the fallback's, else the first route's. */
  capabilities?: Capabilities;
}

/** The `{route}` prefix of a model id, or `undefined` when the id names none. */
export function routePrefixOf(model: unknown): string | undefined {
  if (typeof model !== "string") return undefined;
  const slash = model.indexOf("/");
  return slash > 0 ? model.slice(0, slash) : undefined;
}

/** The model id an op asks for, read off its inline config. */
function modelOf(op: Operation<InlineFamily>): unknown {
  if (op.kind !== "prompt") return undefined;
  const config = op.config;
  return config !== null && typeof config === "object" && !Array.isArray(config) ? (config as Record<string, unknown>)["model"] : undefined;
}

export class PromptRouterExecutor implements Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, ResolvedValue> {
  /** The hierarchy's serializable discriminant — see `FunctionExecutor.kind` in `exec`. */
  static readonly kind: string = "prompt-router";

  readonly capabilities: Capabilities;
  readonly metrics: MetricsAlgebra<LlmMetrics> = { merge: mergeLlmMetrics, empty: emptyLlmMetrics };

  constructor(private readonly options: PromptRouterExecutorOptions) {
    const first = options.fallback ?? Object.values(options.routes)[0];
    if (first === undefined && options.capabilities === undefined) {
      throw new Error("PromptRouterExecutor needs at least one route or a fallback to advertise capabilities from");
    }
    this.capabilities = options.capabilities ?? first!.capabilities;
  }

  /**
   * The capabilities of the executor that would ACTUALLY serve this op.
   *
   * Load-bearing rather than tidy: a wrapper gating on `structuredOutput` or `policyEnforcement` reads
   * this, and one static record standing in for a table of transports is exactly the degradation
   * `capabilitiesFor` exists to prevent — an agent-served call and a provider-served call disagree on
   * every one of those fields.
   */
  capabilitiesFor(op: Operation<InlineFamily>): Capabilities {
    const target = this.route(op);
    if (target === undefined) return this.capabilities;
    return target.capabilitiesFor?.(op) ?? target.capabilities;
  }

  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, LlmMetrics> {
    const target = this.route(op);
    if (target === undefined) {
      const model = modelOf(op);
      const known = Object.keys(this.options.routes).sort();
      return finishedHandle<ResolvedValue, LlmMetrics>({
        error: {
          classification: "permanent",
          reason:
            model === undefined
              ? `no model is configured for this prompt operation, and no fallback executor is wired in (routes: ${known.join(", ") || "none"})`
              : `model "${String(model)}" names no wired route — known routes are ${known.join(", ") || "none"}`,
        },
        metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" },
      });
    }
    // The op travels UNCHANGED, prefix included. Stripping it would hand the provider path an id its
    // own parser rejects, and would make the route invisible to anything downstream that reads the
    // model — a memo key, a price table, a diagnostic.
    return target.start(op, ctx);
  }

  private route(op: Operation<InlineFamily>): Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, ResolvedValue> | undefined {
    const prefix = routePrefixOf(modelOf(op));
    return (prefix !== undefined ? this.options.routes[prefix] : undefined) ?? this.options.fallback;
  }
}

/** Convenience factory mirroring the class constructor. */
export function createPromptRouterExecutor(options: PromptRouterExecutorOptions): PromptRouterExecutor {
  return new PromptRouterExecutor(options);
}
