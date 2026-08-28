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
 *
 * It is also the BASE of the family that answers a `PromptOp`: `AgentExecutor` (DESIGN §4.4) is this
 * class with a different `invoke`. That is why `run` below is split into `protected` phases rather
 * than written as one method — the phases are exactly where the family differs, and there is only one.
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
  emptyLlmMetrics, mergeLlmMetrics, entriesOfMessages, providerOf,
  type CallDeps,
  type LlmCallResult,
  type LlmMetrics,
  type LlmOutput,
  type LlmCallDefinition,
  type LlmCallEnvironment,
  type ModelMessage,
  type ModelRouter,
  type RawMessage,
  type ToolExecutor,
} from "@declarative-ai/llm";
import { lowerPromptOp, type LoweringOptions } from "./lowering.js";

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

/**
 * The PROMPT `Executor` — and the base of the executor family that answers a `PromptOp`.
 *
 * `run` is split into `protected` phases rather than written as one method, because the phases are
 * where the family actually differs. A delegated agent lowers the SAME `LlmCallDefinition` (a
 * serializable description of one call: system, turns, model, tools, a step budget — which is
 * precisely what a coding-agent CLI is configured with) and resolves its session the SAME way; it
 * differs only in {@link PromptExecutor.invoke}, where the call is made. Naming the seams is what
 * lets that be a subclass rather than a second copy of the whole pipeline — and the copy is not
 * hypothetical, it is what `agents-api` used to be.
 *
 * The phases, in order: {@link lower} → {@link applySession} → {@link invoke} → {@link project}.
 * Each is behaviour-identical to the single method it came out of.
 */
export class PromptExecutor<Out = ResolvedValue> implements Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, Out> {
  /** The hierarchy's serializable discriminant — see `FunctionExecutor.kind` in `exec`. Typed as
   *  `string` rather than as its own literal so a subclass can narrow it to its own name. */
  static readonly kind: string = "prompt";

  readonly capabilities: Capabilities = CAPABILITIES;
  /** How two of THIS executor's measurements combine — tokens and money add, the start is the first
   *  observation. exec calls this to fold retry attempts without knowing what a token is. */
  readonly metrics: MetricsAlgebra<LlmMetrics> = { merge: mergeLlmMetrics, empty: emptyLlmMetrics };

  private envRouter: ModelRouter | undefined;

  constructor(protected readonly options: PromptExecutorOptions = {}) {}

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

  protected resolveRouter(ctx: ExecServices): ModelRouter | undefined {
    if (this.options.router) return this.options.router;
    if (ctx.modelRouter) return ctx.modelRouter;
    // A custom runner needs no router at all — never force env keys on it.
    if (this.options.runner) return undefined;
    this.envRouter ??= createModelRouter();
    return this.envRouter;
  }

  /**
   * Does a missing router make this call impossible?
   *
   * True for the provider path with no custom runner: there is nothing to send the call to. A
   * subclass whose {@link invoke} reaches somewhere else entirely — a subprocess, an SDK — answers
   * false, which is how it opts out of the provider requirement without pretending to have a router.
   */
  protected requiresRouter(): boolean {
    return this.options.runner === undefined;
  }

  /**
   * The turns a lowered call sends. Overridable so a transport that speaks a different message shape
   * can say what its delta is.
   */
  protected requestTurns(definition: LlmCallDefinition): ModelMessage[] {
    return requestTurns(definition);
  }

  /** Adapt the resolved `Tool`s into the executable form this transport hands to its call. */
  protected adaptTools(tools: Record<string, Tool> | undefined, ctx: ExecServices): Record<string, ToolExecutor> | undefined {
    return adaptTools(tools, ctx);
  }

  /**
   * PHASE 1 — the op becomes a call declaration.
   *
   * ONE tool source for BOTH halves of the declaration/environment split. Reading `ctx.tools` for the
   * executors while the lowering read `ctx.tools ?? this.options.tools` for the DECLARATIONS meant a
   * construction-time `createPromptExecutor({ tools })` told the model a tool existed and then supplied
   * nothing that could run it — `call.ts`'s `executable` check goes false, `stopWhen` is never set, and
   * the tool LOOP silently degrades to a single turn that returns an unexecuted tool call.
   * `PromptExecutorOptions extends LoweringOptions`, so that is a documented public path.
   *
   * Throws on a malformed config; the caller turns that into a `permanent` refusal, honoring the
   * never-throws contract at the seam.
   */
  protected lower(op: PromptOp<InlineFamily>, tools: Record<string, Tool> | undefined): LlmCallDefinition {
    return lowerPromptOp(op, { ...this.options, tools });
  }

  /**
   * PHASE 2 — THE SESSION, resolved to a position and reserved by the wrapper (DESIGN.md §1.6).
   *
   * This is the executor's half of the split: the wrapper decided WHETHER this appends or forks;
   * shaping the request for that decision is the executor's business, and it is deliberately not
   * something a rewritten op config could express.
   *
   * The strategy branches on ONE declared fact — `capabilities.sessionResume` — rather than on which
   * class is running, which is what makes this method shared rather than duplicated:
   *
   *  - **No native resume ⇒ REPLAY.** The Messages API and everything reached through the AI SDK are
   *    stateless, so history goes on the wire every call and a fork costs nothing but a different key
   *    on the way out.
   *  - **Native resume ⇒ HANDLE.** The remote already holds the conversation, so zero prior messages
   *    are read and the provider handle is threaded instead. Reading and resending them would pay for
   *    the whole transcript on every turn to tell the agent what it already knows.
   *
   * The handle is threaded only on an APPEND. A fork must never inherit its parent's handle unless
   * the transport declares native fork, or two branches write into one remote session.
   */
  protected async applySession(
    definition: LlmCallDefinition,
    ctx: ExecServices,
  ): Promise<{ definition: LlmCallDefinition; sent: ModelMessage[] }> {
    const session = ctx.session as ResolvedSession<ModelMessage> | undefined;
    if (session === undefined) return { definition, sent: [] };

    // Whatever the lowering produced IS the request beyond the stream: the wrapper no longer
    // injects history into the config, so nothing here is already in `prior`.
    const sent = this.requestTurns(definition);
    // Can this transport BRANCH a conversation server-side? Read off its own declared capabilities, so
    // there is one answer and it is the one the engine also reads. Absent means "the same as resume",
    // which is what every adapter predating the split meant by `sessionResume: true`.
    const nativeFork = this.capabilities.sessionFork ?? this.capabilities.sessionResume;
    // The FORK-WITHOUT-A-FORK-PRIMITIVE case (SESSIONS.md §6, "Strategies") folded into the general
    // rule: replay when the remote holds nothing for us (no native resume at all), or when it holds a
    // conversation we are not allowed to branch. `messages()` is a LAZY accessor and this is the only
    // path that pays for it — the cheap paths read zero messages.
    const replaying = !this.capabilities.sessionResume || (session.mode === "fork" && !nativeFork);

    if (replaying) {
      // Resolved from `id` through the store when the accessor is missing: non-enumerable properties
      // do not survive a spread or a structured clone, and `{ ...session, fork: true }` is a thing
      // people write. Losing the accessor must cost a store read, never correctness.
      const prior = await priorMessages(session);
      const replayed: LlmCallDefinition = { ...definition, messages: [...prior, ...sent] };
      delete (replayed as { prompt?: unknown }).prompt; // the SDK rejects both
      definition = replayed;
    }
    // The handle is threaded ONLY where it is safe to: an append always, a fork solely when this
    // transport can branch server-side. A fork that carried its parent's handle would put two branches
    // into one remote session — silent and unrecoverable, which is why the `mode` test is kept as the
    // belt to `MapSessionStore`'s braces (it already withholds the handle on a fork).
    //
    // For a transport with no native resume this reduces to "append only", which is exactly the rule
    // the provider path had before the two were one method: `nativeFork` is false there, so the
    // disjunction collapses.
    if (session.providerSessionId !== undefined && (session.mode !== "fork" || nativeFork)) {
      definition = { ...definition, providerSessionId: session.providerSessionId };
    }
    return { definition, sent };
  }

  /**
   * PHASE 3 — THE CALL. The one line the family disagrees about.
   *
   * Everything above and below this is shared: the same op became the same declaration under the
   * same session. A subclass overrides only this to reach a subprocess or an SDK instead of a
   * provider endpoint, and inherits the rest unchanged.
   */
  protected async invoke(definition: LlmCallDefinition, env: CallDeps, ctx: ExecServices): Promise<LlmCallResult> {
    // No per-call timeout ARGUMENT. A window reaches the call as cancellation now — `withDeadline`
    // and `hw` fold `AbortSignal.timeout(...)` into `ctx.abortSignal`, which `env.abortSignal` already
    // carries — so passing a number as well would be a second spelling of the same bound. The llm
    // layer keeps its own `def.timeoutMs ?? DEFAULT_TIMEOUT_MS` floor, so a call is still never
    // unbounded.
    return (this.options.runner ?? defaultRunner)(definition, env);
  }

  /**
   * PHASE 4 — THE PROJECTION (DESIGN §3.1). An `LlmOutput` — output value, thinking, tool calls,
   * finish reason — is what the PROVIDER produced. What an EXECUTION returns is the value of the op's
   * output parameter, and nothing else. So this is the boundary where the model payload stops:
   * everything past here sees a `ResolvedValue`, which is why `exec` and `hw` no longer name
   * `thinking` at all.
   */
  protected project(op: PromptOp<InlineFamily>, output: LlmOutput | undefined): ResolvedValue | undefined {
    return projectLlmOutput(op, output);
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

    // The LOUD-FAILURE check that used to stand here — refusing a `ctx.deadline` this bare core cannot
    // honour — is gone because the misconfiguration it caught is now unrepresentable. A window is a
    // construction option of `withDeadline` and no longer a field on `ExecServices`, so "set a
    // deadline without composing the wrapper" does not typecheck. A compile error beats a runtime
    // refusal, and neither beats a shape that cannot express the mistake.

    const tools = ctx.tools ?? this.options.tools;

    let definition: LlmCallDefinition;
    try {
      definition = this.lower(op, tools);
    } catch (e) {
      return refuse(`invalid llm config: ${(e as Error).message}`);
    }
    if (definition.sessionId !== undefined || definition.providerSessionId !== undefined) {
      return refuse(
        "the declaration carries sessionId/providerSessionId, but no session layer consumed it — compose withSession(...) around the core (the wrapper resolves and strips the session fields)",
      );
    }

    const session = ctx.session as ResolvedSession<ModelMessage> | undefined;
    /** The request turns this call ADDS beyond the stream it started from — its half of the delta. */
    let sent: ModelMessage[];
    ({ definition, sent } = await this.applySession(definition, ctx));

    const router = this.resolveRouter(ctx);
    // Only the DEFAULT provider path needs a router: a custom runner (a test fake, a recorded
    // transport) or a subclass reaching a subprocess resolves the call itself, so forcing env keys on
    // it would be gratuitous.
    if (!router && this.requiresRouter()) return refuse("no ModelRouter available (ctx.modelRouter or options.router)");

    // Combined abort: internal cancel + the caller's signal (the per-call timeout is applied in `llm`).
    const abortSignal = ctx.abortSignal ? AbortSignal.any([signal, ctx.abortSignal]) : signal;
    const env: LlmCallEnvironment & { modelRouter: ModelRouter } = {
      modelRouter: router as ModelRouter,
      validator: ctx.validator,
      abortSignal,
      toolExecutors: this.adaptTools(tools, ctx),
    };

    let call: LlmCallResult;
    try {
      call = await this.invoke(definition, env, ctx);
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
      // Onto ENTRIES, which is where the conversation lives now — the payload holds one array and
      // the wire history is projected from it. Prepending to a `messages` field the call no longer
      // writes would store the request half where nothing reads it, and the stored exchange would be
      // the answer with no question in front of it.
      const asked = entriesOfMessages(sent as RawMessage[], { provider: providerOf(call.value.model), at: new Date().toISOString() });
      call = { ...call, value: { ...call.value, entries: [...asked, ...(call.value.entries ?? [])] } };
    }

    const output = call.value;
    // THE PAYLOAD, alongside the value — when the recorder asked for it.
    //
    // This replaced a `record` CONSTRUCTION option that swapped what the execution's value IS. That
    // could not be typed: `Out` is a static parameter, `AgentExecutor` drops it, and every agent class
    // therefore claimed to return a projection while returning a payload. Reported on its own channel
    // the value never changes, so there is no type to be wrong about — and the decision moves to the
    // layer that actually needs the payload, per call, instead of being fixed when the executor is built.
    const record = ctx.returnRecord === true ? { record: (output ?? { finishReason: "error" }) as unknown } : {};
    const value = this.project(op, output);

    const metrics: LlmMetrics = { ...call.metrics, startMs };
    // THE DELTA, on the declared channel — because in VALUE mode the payload does not survive this
    // line and the record would otherwise hold a position with no conversation in it.
    //
    // "The response IS the delta" (SESSIONS.md §7) holds only in RECORD mode, where the payload IS the
    // execution value and `withRecord` stores it whole. In value mode the projection above is the last
    // thing that sees `messages`, so a host composing `withRecord` around a value-mode core — which is
    // every host that uses exec's `withSessionPosition` rather than this package's `withSession` —
    // recorded the answer and lost the conversation. Silently: nothing failed, transcripts were just
    // always empty, and only a scripted fake (which reports here) ever produced one.
    //
    // Gated on `ctx.session` rather than on an option: a resolved position means a position layer is
    // composed, and a position layer is only ever composed with `withRecord` beneath it. So this is
    // exactly "somebody is recording", already known, needing no new seam. The cost when true is a
    // reference to an array that already exists.
    const reported = sessionOutcomeOfCall(ctx, output);

    if (isOk(call)) return { value: value as ResolvedValue, metrics, ...reported, ...record };

    // A cancel that raced the call re-classifies the provider's failure without discarding it.
    const canceled = wasCanceled() || ctx.abortSignal?.aborted === true;
    const error = canceled ? { ...call.error, classification: "canceled" as const } : call.error;
    // Reported on failure TOO: turns the provider appended before it failed exist remotely whether or
    // not we kept them, and not recording them is divergence on the very next call.
    return { error, ...(value !== undefined ? { value } : {}), metrics, ...reported, ...record };
  }
}

/**
 * The conversation at a position, for a transport that has to REPLAY it.
 *
 * `ResolvedSession.messages` is non-enumerable, so it does not survive an object spread, a structured
 * clone, or a JSON round-trip. This used to fall back to `ctx.sessions.messages(id)` when it went
 * missing — which meant every executor had to be handed the whole store for a case that should not
 * happen, and which silently replayed NOTHING when the store was absent too.
 *
 * REFUSING is the honest answer. A replay that quietly sends no history is a call the provider answers
 * from an empty conversation while the workflow believes it continued one — the exact silent-divergence
 * failure the session model exists to prevent. Losing the accessor must cost an error, not correctness.
 */
async function priorMessages(session: ResolvedSession<ModelMessage>): Promise<ModelMessage[]> {
  if (typeof session.messages !== "function") {
    throw new Error(
      `session ${session.id} arrived without its messages accessor — it is non-enumerable and does not survive a spread or a clone, so pass the resolved session through untouched rather than copying it`,
    );
  }
  return await session.messages();
}

/**
 * What a value-mode call added to its conversation, as a `SessionOutcome` — absent when it added
 * nothing, or when no conversation was in play.
 *
 * Absent rather than empty is load-bearing: `withRecord` passes a reported outcome to `records.close`,
 * and a store reads `messages` there to decide whether the stored payload IS the conversation. An empty
 * report would claim a call that produced nothing, which is a different fact from a call that reported
 * nothing — a lowering refusal, say, where no provider was ever reached.
 */
function sessionOutcomeOfCall(
  ctx: ExecServices,
  output: LlmOutput | undefined,
): { session?: { messages?: readonly ModelMessage[]; providerSessionId?: string } } {
  if (output === undefined) return {};
  const { messages, providerSessionId } = output;
  if ((messages === undefined || messages.length === 0) && providerSessionId === undefined) return {};
  // Reported when a conversation is in play, OR whenever there is a PROVIDER HANDLE at all. The second
  // clause is not redundant: a handle is the one thing a caller cannot recover by any other means — it
  // is minted remotely and never appears in the projected value — so withholding it because no local
  // position was resolved would discard the only record that a remote conversation exists.
  if (ctx.session === undefined && providerSessionId === undefined) return {};
  return {
    session: {
      ...(messages !== undefined && messages.length > 0 ? { messages } : {}),
      ...(providerSessionId !== undefined ? { providerSessionId } : {}),
    },
  };
}

/** Convenience factory mirroring the class constructor — the BARE core (no wrappers). Compose the
 *  cross-cutting behaviors you want with `compose(core).with(withRateLimit(...)).with(withRetry(...))`. */
export function createPromptExecutor(options: PromptExecutorOptions = {}): Executor<ExecServices, LlmMetrics> {
  // `never` is assignable to BOTH overloads' Out; the constructed instance's true Out is the flag's.
  return new PromptExecutor(options) as PromptExecutor<never>;
}
