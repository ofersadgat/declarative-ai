/**
 * The PROMPT `Executor` (DESIGN §4.1) — the MINIMAL core: lower a `PromptOp` to an
 * `LlmCallDefinition`, run it, map the result onto `Outcome`, and nothing else.
 *
 * Cross-cutting concerns — retry, rate limiting, deadline fail-fast, sessions, budget, memoization —
 * are NOT here; they are composable `ExecutorWrapper`s stacked around this core. Keeping the unit small
 * is the point: it delivers exactly its value and lets the wrappers deliver theirs.
 *
 * On the layering: conceptually it is `llm ← promptop ← exec`, but in DEPENDENCY terms `promptop`
 * depends on `exec`, because this class IMPLEMENTS the `Executor` interface `exec` defines. Both
 * readings hold — `exec` owns the generic machinery (low), `promptop` owns the LLM-specific
 * implementation (high). Nothing in `exec` knows `PromptOp` exists.
 */
import type {
  Capabilities,
  ExecHandle,
  ExecResult,
  ExecServices,
  Executor,
  InlineFamily,
  MetricsAlgebra,
  Operation,
  PromptOp,
  ResolvedSession,
  ResolvedValue,
  Tool,
} from "@declarative-ai/exec";
import { emptyEvents, finishedHandle, isOk, systemClock } from "@declarative-ai/exec";
import {
  createModelRouter,
  executeLlmCall,
  mergeLlmMetrics,
  type CallDeps,
  type LlmCallResult,
  type LlmMetrics,
  type LlmOutput,
  type LlmCallDefinition,
  type LlmCallEnvironment,
  type ModelMessage,
  type ModelRouter,
  type ToolExecutor,
} from "@declarative-ai/llm";
import { lowerPromptOp, type LoweringOptions } from "./lowering";

// `modelRouter` is llm's seam, so llm's own type is what names it — and this is the package that can
// (§1.2). `exec` therefore never declares an opaque `ModelHandle` it cannot describe.
declare module "@declarative-ai/exec" {
  interface ExecServices {
    /** Provider routing for prompt ops: a route-prefixed model id → a provider model handle. */
    modelRouter?: ModelRouter;
  }
}

/**
 * Execute ONE `PromptOp` at the llm layer — lowering + `executeLlmCall`, returning the FULL
 * `LlmCallResult` (value, `thinking`, `finishReason`, metrics). This is the op-level call for a
 * consumer that PERSISTS what the model produced (an `OperationRecord`'s `R` is the payload, so the
 * projection would lose exactly what it stores); the {@link PromptExecutor} below is the same
 * pipeline behind the `Executor` seam, PROJECTING the payload down to the op's output value for the
 * execution stack. Lowering faults resolve as a `permanent` failure — the seam never throws.
 */
export async function executePromptOp(
  op: PromptOp<InlineFamily>,
  env: CallDeps,
  options: LoweringOptions & { runner?: CallRunner } = {},
): Promise<LlmCallResult> {
  let def: LlmCallDefinition;
  try {
    def = lowerPromptOp(op, options);
  } catch (e) {
    return {
      error: { classification: "permanent", reason: e instanceof Error ? e.message : String(e) },
      value: { finishReason: "error" },
      metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" },
    };
  }
  return (options.runner ?? defaultRunner)(def, env, def.timeoutMs);
}

/** The runtime environment this executor builds, re-exported so a custom {@link CallRunner} can name
 *  what it receives. */
export type { CallDeps, LlmCallResult, LlmCallDefinition };

/** The injectable call seam: one structured call, declaration + environment → a never-throwing
 *  `LlmCallResult`. It makes the mapping/cancel control flow testable with no provider. */
export type CallRunner = (def: LlmCallDefinition, env: CallDeps, timeoutMs?: number) => Promise<LlmCallResult>;

const defaultRunner: CallRunner = (def, env, timeoutMs) => executeLlmCall(def, env, timeoutMs);

export interface PromptExecutorOptions extends LoweringOptions {
  /** Explicit router; else the typed `ctx.modelRouter`; else a lazy env-key router. */
  router?: ModelRouter;
  /** The call seam; defaults to the real `executeLlmCall` pipeline. */
  runner?: CallRunner;
  /**
   * RECORD mode: the execution value is the FULL `LlmOutput` payload — value, `thinking`,
   * `finishReason`, tool trace — instead of the projection down to the op's output value. The mode is
   * a TYPE-LEVEL fact carried on the executor's `Out` parameter, so a wrapper stack composed around a
   * record-mode core yields `LlmCallResult`-shaped results outward (`ExecResult<LlmOutput, LlmMetrics>`
   * ≡ `LlmCallResult`) — the interface and the pipeline are unchanged; only what the value IS differs.
   *
   * For consumers that PERSIST what the model produced (an `OperationRecord`'s `R` is the payload).
   * NB `withSession` composes over VALUE-mode executors only — its transcript fold reads the op's
   * output value, which in record mode is buried inside the payload.
   */
  record?: boolean;
}

const CAPABILITIES: Capabilities = {
  structuredOutput: true,
  sessionResume: false,
  streaming: true,
  interactive: false,
  readOnly: true,
  mutatesWorkspace: false,
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

/**
 * The turns a lowered call sends, as messages — its `messages` preamble, or its `prompt` read as the
 * single user turn the lowering would have made of it.
 *
 * This is the request half of a session delta. The response half comes back on the output, because
 * only the provider knows what it actually appended.
 */
function requestTurns(def: LlmCallDefinition): ModelMessage[] {
  if (def.messages !== undefined) return [...def.messages];
  const prompt = def.prompt;
  if (prompt === undefined) return [];
  return typeof prompt === "string" ? [{ role: "user", content: prompt }] : [...prompt];
}

/**
 * THE PROJECTION (DESIGN §3.1): an `LlmOutput` narrowed to the op's output-PARAMETER value.
 *
 * It exists so everything above the executor speaks one vocabulary regardless of op kind — a state's
 * `outputs.answer` is the answer, not an envelope, and hw never learns what `thinking` is.
 *
 * Exported because WHERE it happens matters. Applied to the whole result, it destroys the payload that
 * a session needs to record, and the layer that wanted it then has to smuggle it back down through a
 * side channel. Applied by whichever layer is the last to need the payload, it narrows only the value
 * slot — which is what `withSession` does, projecting on its way out over a record-mode core.
 *
 * A generated FILE lands in a `blob`-kind output parameter — that is §7.1's "a produced artifact is an
 * output parameter, not a parallel channel". A json/text parameter ignores it.
 */
export function projectLlmOutput(op: PromptOp<InlineFamily>, output: LlmOutput | undefined): ResolvedValue | undefined {
  return op.output.kind === "blob" && output?.files && output.files.length > 0
    ? (output.files[0]!.bytes as ResolvedValue)
    : (output?.value as ResolvedValue | undefined);
}

/** Adapt core {@link Tool}s (`run(input, ctx)`) into llm {@link ToolExecutor}s (`(input, options)`),
 *  closing over the call ctx. The tool's `run` IS its `execute`; the SDK's per-call `options` are
 *  dropped (a v1 tool needs only its input + the shared services). */
function adaptTools(tools: Record<string, Tool> | undefined, ctx: ExecServices): Record<string, ToolExecutor> | undefined {
  if (!tools) return undefined;
  const entries = Object.entries(tools);
  if (entries.length === 0) return undefined;
  const out: Record<string, ToolExecutor> = {};
  for (const [name, tool] of entries) out[name] = (input) => tool.run(input, ctx);
  return out;
}

export class PromptExecutor<Out = ResolvedValue> implements Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, Out> {
  readonly capabilities = CAPABILITIES;
  /** How two of THIS executor's measurements combine — tokens and money add, the start is the first
   *  observation. exec calls this to fold retry attempts without knowing what a token is. */
  readonly metrics: MetricsAlgebra<LlmMetrics> = { merge: mergeLlmMetrics };

  private envRouter: ModelRouter | undefined;

  constructor(private readonly options: PromptExecutorOptions = {}) {}

  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<Out, LlmMetrics> {
    if (op.kind !== "prompt") {
      return finishedHandle<Out, LlmMetrics>({
        error: { classification: "permanent", reason: `the prompt executor was handed a ${op.kind} operation` },
        metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" },
      });
    }
    const internal = new AbortController();
    let cancelRequested = false;
    // The body speaks `ResolvedValue` throughout; record mode changes WHAT the value is at the one
    // projection site, and the class's `Out` parameter is the type-level record of that choice.
    const result = this.run(op, ctx, internal.signal, () => cancelRequested) as Promise<ExecResult<Out, LlmMetrics>>;
    return {
      events: emptyEvents(),
      result,
      cancel: async () => {
        cancelRequested = true;
        internal.abort();
        await result;
      },
    };
  }

  private resolveRouter(ctx: ExecServices): ModelRouter | undefined {
    if (this.options.router) return this.options.router;
    if (ctx.modelRouter) return ctx.modelRouter;
    // A custom runner needs no router at all — never force env keys on it.
    if (this.options.runner) return undefined;
    this.envRouter ??= createModelRouter();
    return this.envRouter;
  }

  private async run(
    op: PromptOp<InlineFamily>,
    ctx: ExecServices,
    signal: AbortSignal,
    wasCanceled: () => boolean,
  ): Promise<ExecResult<ResolvedValue, LlmMetrics>> {
    const startMs = (ctx.clock ?? systemClock).now();
    /** A refusal made BEFORE any provider call. `costUsd: 0` is a real claim here — nothing was sent,
     *  so nothing was billed — which is why this is NOT the shape used when a call may have happened. */
    const refuse = (reason: string): ExecResult<ResolvedValue, LlmMetrics> => ({
      error: { classification: "permanent", reason },
      metrics: { startMs, durationMs: 0, costUsd: 0, costSource: "table" },
    });

    /**
     * A failure where a call MAY have been made and billed but its metrics were lost — a runner that
     * threw instead of resolving. A FAILED CALL STILL COSTS MONEY: a truncated generation bills the
     * tokens it emitted, a 5xx after the model started bills, and a validation failure is a call the
     * provider completed and charged for. So `costUsd: 0` here is not "free", it is "unmeasured", and
     * `costSource: "unknown"` is what says so — a budget settling this reserve is under-charging and
     * the provenance flag is the only signal it has.
     */
    const lostMetrics = (reason: string): ExecResult<ResolvedValue, LlmMetrics> => ({
      error: { classification: "permanent", reason },
      metrics: { startMs, durationMs: (ctx.clock ?? systemClock).now() - startMs, costUsd: 0, costSource: "unknown" },
    });

    // LOUD-FAILURE contract: fields this bare core does NOT implement must never arrive silently. Each
    // wrapper CONSUMES (and strips) its own field, so anything still present here means the caller
    // relied on behavior that is not composed into the stack.
    if (ctx.deadline !== undefined) {
      return refuse(
        "ctx.deadline is set, but the bare prompt executor does not implement deadline handling — compose withDeadline() around it (the wrapper consumes ctx.deadline/ctx.stepStartMs)",
      );
    }

    // ONE tool source for BOTH halves of the declaration/environment split. Reading `ctx.tools` for the
    // executors while the lowering read `ctx.tools ?? this.options.tools` for the DECLARATIONS meant a
    // construction-time `createPromptExecutor({ tools })` told the model a tool existed and then supplied
    // nothing that could run it — `call.ts`'s `executable` check goes false, `stopWhen` is never set, and
    // the tool LOOP silently degrades to a single turn that returns an unexecuted tool call.
    // `PromptExecutorOptions extends LoweringOptions`, so that is a documented public path.
    const tools = ctx.tools ?? this.options.tools;

    let definition: LlmCallDefinition;
    try {
      definition = lowerPromptOp(op, { ...this.options, tools });
    } catch (e) {
      return refuse(`invalid llm config: ${(e as Error).message}`);
    }
    if (definition.sessionId !== undefined || definition.providerSessionId !== undefined) {
      return refuse(
        "the declaration carries sessionId/providerSessionId, but no session layer consumed it — compose withSession(...) around the core (the wrapper resolves and strips the session fields)",
      );
    }

    /**
     * THE SESSION, resolved to a position and reserved by the wrapper (DESIGN.md §1.6).
     *
     * This is the executor's half of the split: the wrapper decided WHETHER this appends or forks;
     * shaping the request for that decision is the executor's business, and it is deliberately not
     * something a rewritten op config could express.
     *
     * Replay is the strategy here — the Messages API and everything reached through the AI SDK are
     * stateless, so history goes on the wire every call and a fork costs nothing but a different key
     * on the way out. `messages()` is the lazy accessor; a native-fork adapter would read zero
     * messages and pass a handle instead.
     *
     * The handle is threaded only on an APPEND. A fork must never inherit its parent's handle unless
     * the adapter declares native fork, or two branches write into one remote session.
     */
    const session = ctx.session as ResolvedSession<ModelMessage> | undefined;
    /** The request turns this call ADDS beyond the stream it started from — its half of the delta. */
    let sent: ModelMessage[] = [];
    if (session !== undefined) {
      // Resolved from `id` through the store when the accessor is missing: non-enumerable properties
      // do not survive a spread or a structured clone, and `{ ...session, fork: true }` is a thing
      // people write. Losing the accessor must cost a store read, never correctness.
      const prior =
        typeof session.messages === "function"
          ? await session.messages()
          : (((await ctx.sessions?.messages(session.id)) ?? []) as ModelMessage[]);
      // Whatever the lowering produced IS the request beyond the stream: the wrapper no longer
      // injects history into the config, so nothing here is already in `prior`.
      sent = requestTurns(definition);
      definition = { ...definition, messages: [...prior, ...sent] };
      delete (definition as { prompt?: unknown }).prompt; // the SDK rejects both
      if (session.mode === "append" && session.providerSessionId !== undefined) {
        definition = { ...definition, providerSessionId: session.providerSessionId };
      }
    }

    const runner = this.options.runner;
    const router = this.resolveRouter(ctx);
    // Only the DEFAULT runner needs a provider: a custom runner (a test fake, a recorded transport)
    // resolves the call itself, so forcing env keys on it would be gratuitous.
    if (!router && !runner) return refuse("no ModelRouter available (ctx.modelRouter or options.router)");

    // Combined abort: internal cancel + the caller's signal (the per-call timeout is applied in `llm`).
    const abortSignal = ctx.abortSignal ? AbortSignal.any([signal, ctx.abortSignal]) : signal;
    const env: LlmCallEnvironment & { modelRouter: ModelRouter } = {
      modelRouter: router as ModelRouter,
      validator: ctx.validator,
      abortSignal,
      toolExecutors: adaptTools(tools, ctx),
    };

    let call: LlmCallResult;
    try {
      call = await (runner ?? defaultRunner)(definition, env, ctx.timeoutMs);
    } catch (err) {
      // A throw means we do not know what the provider saw, so nothing is reported: an invented delta
      // is worse than an absent one, because the mirror would then disagree with the remote silently.
      // The wrapper still releases its reservation.
      return lostMetrics(err instanceof Error ? err.message : String(err));
    }

    // NOTHING to report. The delta is already on `call.value` — an `LlmOutput` carrying `messages`
    // verbatim — and `withRecord` writes that payload into the record occupying this session position.
    // A session is the records sharing a `session.id`, so appending the turn and recording the call are
    // one write; there is no separate transcript to fold and no channel to fold it through.
    //
    // That covers the awkward cases without special-casing them. Repair turns are in the payload
    // because the provider genuinely produced them. A call that appended and THEN failed is recorded
    // too, because `withRecord` fills its stub either way — which matters, since those turns exist
    // remotely whether or not we kept them.
    //
    // What the executor DOES owe the delta is the request half: `sent` is prepended below so the
    // stored payload is the whole exchange rather than only what came back.
    if (session !== undefined && call.value !== undefined && sent.length > 0) {
      call = { ...call, value: { ...call.value, messages: [...sent, ...(call.value.messages ?? [])] } };
    }

    // THE PROJECTION (DESIGN §3.1). An `LlmOutput` — output value, thinking, tool calls, finish
    // reason — is what the PROVIDER produced. What an EXECUTION returns is the value of
    // the op's output parameter, and nothing else. So this is the boundary where the model payload
    // stops: everything past here sees a `ResolvedValue`, which is why `exec` and `hw` no longer name
    // `thinking` at all.
    const output = call.value;
    // RECORD mode: no projection — the execution value IS the payload (see
    // {@link PromptExecutorOptions.record}); the class's `Out` parameter carries that outward.
    if (this.options.record) {
      const payload = (output ?? { finishReason: "error" }) as unknown as ResolvedValue;
      const recordMetrics: LlmMetrics = { ...call.metrics, startMs };
      if (isOk(call)) return { value: payload, metrics: recordMetrics };
      const canceledCall = wasCanceled() || ctx.abortSignal?.aborted === true;
      const recordError = canceledCall ? { ...call.error, classification: "canceled" as const } : call.error;
      return { error: recordError, value: payload, metrics: recordMetrics };
    }
    const value = projectLlmOutput(op, output);

    const metrics: LlmMetrics = { ...call.metrics, startMs };
    if (isOk(call)) return { value: value as ResolvedValue, metrics };

    // A cancel that raced the call re-classifies the provider's failure without discarding it.
    const canceled = wasCanceled() || ctx.abortSignal?.aborted === true;
    const error = canceled ? { ...call.error, classification: "canceled" as const } : call.error;
    return { error, ...(value !== undefined ? { value } : {}), metrics };
  }
}

/** Convenience factory mirroring the class constructor — the BARE core (no wrappers). Compose the
 *  cross-cutting behaviors you want with `compose(core).with(withRateLimit(...)).with(withRetry(...))`. */
export function createPromptExecutor(options?: PromptExecutorOptions & { record?: false }): Executor<ExecServices, LlmMetrics>;
export function createPromptExecutor(
  options: PromptExecutorOptions & { record: true },
): Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, LlmOutput>;
export function createPromptExecutor(options: PromptExecutorOptions = {}): Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, never> {
  // `never` is assignable to BOTH overloads' Out; the constructed instance's true Out is the flag's.
  return new PromptExecutor(options) as PromptExecutor<never>;
}
