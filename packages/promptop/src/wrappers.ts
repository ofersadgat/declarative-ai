/**
 * The llm-AWARE executor wrappers. Each needs something only this layer knows: a token estimate off a
 * prompt (`withRateLimit`), model pricing (`withBudget`), or how to fold an outcome into a transcript
 * (`withSession`). The generic ones — `withDeadline`, `withRetry`, `withMemoize` — live in
 * `@declarative-ai/exec` and apply to function ops too.
 *
 * All three read and rewrite the op's `config` slot, which IS the `LlmConfiguration` surface. That is
 * what makes them work against the single `start(op, ctx)` seam without a spec: the rewritten OP is
 * what gets sent, so an inner memoize (keyed on the op hash) keys on exactly what was sent, with no
 * separate hash to keep in sync.
 *
 * The two PRICING wrappers read that slot through {@link pricedCall}, i.e. through the same
 * `lowerPromptOp` resolution the core runs — so they meter the call that will actually be made
 * (`defaults` ← preset ← inline, transcript included) rather than the op's unresolved inline fragment.
 * That means they take the same `defaults`/`configs` the core executor is constructed with.
 */
import type {
  BudgetMeter,
  ExecHandle,
  Capabilities,
  ExecServices,
  Executor,
  ExecutorWrapper,
  InlineFamily,
  JsonValue,
  Operation,
  ExecResult,
  ExecMetrics,
  BudgetMetrics,
  ResolvedValue,
  PromptOp,
  RateLimiter,
  SessionStore,
  CallEstimate,
} from "@declarative-ai/exec";
import { canceledFailure, curryOrApply, forwardCapabilitiesFor, isExecutor, isOk, wrapHandle } from "@declarative-ai/exec";
import {
  DEFAULT_HOLD_OUTPUT_MULTIPLIER,
  MIN_USEFUL_OUTPUT_TOKENS,
  ModelInfo,
  estimateCallTokens,
  estimateInputTokens,
  estimateOutputTokens,
  noteOutputTokens,
  promptText,
  type OutputTokenStats,
} from "@declarative-ai/llm";
import type { ModelMessage } from "@declarative-ai/llm";
import { lowerPromptOp, type LoweringOptions } from "./lowering";

/**
 * What `withBudget` reads off a measurement: money (its job), plus the observed output size it prices
 * the NEXT reserve from. The token field is declared here rather than on `BudgetMetrics` because it is
 * not money — it is this wrapper's estimator wanting feedback, and saying so in the constraint is how
 * an executor that cannot supply it fails to compile instead of silently mis-estimating.
 */
type BudgetReadable = ExecMetrics & BudgetMetrics & { outputTokens?: number };

/** The ctx seam `withSession` consumes. */
type SessionSeams = { sessions: SessionStore };

/** A prompt op's `config` slot read as a plain record — the `LlmConfiguration` surface. */
function configOf(op: PromptOp<InlineFamily>): Record<string, JsonValue> {
  const c = op.config;
  return c !== null && typeof c === "object" && !Array.isArray(c) ? c : {};
}

/** Rewrite a prompt op's config. The op is the payload, so a wrapper's adjustment IS an op edit. */
function withConfig(op: PromptOp<InlineFamily>, patch: Record<string, JsonValue | undefined>): PromptOp<InlineFamily> {
  const next: Record<string, JsonValue> = { ...configOf(op) };
  for (const [k, v] of Object.entries(patch)) {
    if (v === undefined) delete next[k];
    else next[k] = v;
  }
  return { ...op, config: next };
}

const stringField = (cfg: Record<string, JsonValue>, key: string): string | undefined =>
  typeof cfg[key] === "string" ? (cfg[key] as string) : undefined;
const numberField = (cfg: Record<string, JsonValue>, key: string): number | undefined =>
  typeof cfg[key] === "number" ? (cfg[key] as number) : undefined;

/**
 * The config-resolution inputs a PRICING wrapper needs. They are the same `defaults`/`configs` the core
 * executor is constructed with, and they must be given to the wrappers too: money and rate headroom are
 * spent on the call the CORE will make, which is the RESOLVED one.
 */
export type ResolutionOptions = Pick<LoweringOptions, "defaults" | "configs">;

/** What a wrapper prices a call on — read off the definition the core will actually send. */
interface PricedCall {
  /** The resolved model id, or `undefined` when no layer named one (an unpriceable call). */
  model: string | undefined;
  /** The resolved output ceiling — a real cap, not the estimator's 512 fallback. */
  maxOutputTokens: number | undefined;
  /** EVERY text fragment that goes on the wire: system + prompt + the full `messages` transcript. */
  text: string;
}

/**
 * Price against the RESOLVED config and the FULL message set.
 *
 * Reading `op.config` and `op.system + op.user` directly is what made these wrappers blind in three
 * proved ways: a model supplied via `options.defaults` was invisible, so `withBudget` returned early
 * and the call ran entirely UNMETERED and `withRateLimit` degraded its per-model AIMD to a global one;
 * a `defaults`-supplied `maxOutputTokens` was replaced by the 512 fallback; and a transcript threaded in
 * as config-layer `messages` by `withSession` contributed nothing at all (20k chars sent, 7 tokens
 * declared). So budgets under-reserved and limiters under-declared by orders of magnitude on any
 * multi-turn conversation.
 *
 * `lowerPromptOp` is the SAME resolution the core runs, which is the point — there is no second merge to
 * keep in sync. It THROWS on a malformed config; that failure belongs to the core (which turns it into a
 * `permanent` outcome), so here it degrades to the op's own unresolved view rather than pre-empting the
 * real error message. `promptText` is llm's own extractor, so a multimodal turn contributes its text
 * parts and nothing else — exactly what a token proxy should count.
 */
function pricedCall(op: PromptOp<InlineFamily>, options: ResolutionOptions | undefined): PricedCall {
  try {
    const def = lowerPromptOp(op, options ?? {});
    return { model: def.model, maxOutputTokens: def.maxOutputTokens, text: promptText(def) };
  } catch {
    const cfg = configOf(op);
    return {
      model: stringField(cfg, "model"),
      maxOutputTokens: numberField(cfg, "maxOutputTokens"),
      text: op.system !== undefined ? `${op.system}\n${op.user}` : op.user,
    };
  }
}

/** A non-prompt op passes straight through: these wrappers have nothing to say about a function op,
 *  and pretending otherwise would silently mis-price or mis-key it. */
function isPrompt(op: Operation<InlineFamily>): op is PromptOp<InlineFamily> {
  return op.kind === "prompt";
}

// --- Rate limiting -------------------------------------------------------------

/** Options for {@link withRateLimit}: the limiter, plus the config-resolution inputs the ESTIMATE is
 *  priced against (see {@link pricedCall}). */
export type RateLimitOptions = { limiter: RateLimiter } & ResolutionOptions;

/**
 * Rate limiting: admit the inner call through the injected `RateLimiter` (concurrency slot + rate
 * headroom) using a token estimate off the prompt text, and feed the outcome back (`reportOutcome`
 * drives AIMD). A cancel that lands while the call is still QUEUED prevents it from ever starting
 * (returns a `canceled` outcome); a limiter fault is normalized into a permanent failure.
 */
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics>(config: RateLimitOptions): ExecutorWrapper<R, R, M>;
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics>(config: RateLimitOptions, inner: Executor<R, M>): Executor<R, M>;
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics>(config: RateLimitOptions, inner?: Executor<R, M>): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const { limiter } = config;
  /** Per-op token-estimate cache: the estimate is derived from the full prompt text (potentially a long
   *  transcript), and the SAME op object is re-submitted per repair attempt / retry. Scoped to THIS
   *  wrapper because the estimate now depends on its `defaults`/`configs` too — a module-level cache
   *  would hand one stack's resolution to another's. */
  const estimateCache = new WeakMap<object, CallEstimate>();
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    // Forward per-dispatched-entry capabilities so a `withMemoize` ABOVE this wrapper still gates on the
    // real entry's `memoizable`/`mutatesWorkspace` rather than the static record. Rate-limiting changes
    // nothing about an op's capabilities, so a straight passthrough is correct.
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      if (!isPrompt(op)) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        let est = estimateCache.get(op);
        if (!est) {
          // Resolving is what makes the estimate honest, and it is also the expensive part — so it
          // happens on the cache MISS, not once per `start`.
          const priced = pricedCall(op, config);
          est = { ...estimateCallTokens(priced.text, undefined, priced.maxOutputTokens), modelId: priced.model };
          estimateCache.set(op, est);
        }
        const modelId = est.modelId;
        let ran = false;
        const result = await limiter.schedule(est, () => {
          if (ctl.canceled()) return Promise.resolve(canceledFailure("canceled while queued for rate-limit admission"));
          ran = true;
          return ctl.started(innerExec.start(op, ctx)).result;
        });
        if (ran) limiter.reportOutcome({ rateLimited: isOk(result) ? undefined : result.error.rateLimited, modelId });
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

// --- Budget --------------------------------------------------------------------

/** Options for {@link withBudget}. All optional — with no `meter` (here or on `ctx.meter`) the wrapper
 *  is a pure passthrough (an absent service is a no-op, like the rest of the stack). */
export interface BudgetOptions extends ResolutionOptions {
  /** The metered wallet. Defaults to `ctx.meter`; supplied here it drops the ctx dependency. */
  meter?: BudgetMeter;
  /** Runtime-tunable output-token headroom for the pre-call reserve estimate (default 2×). */
  headroomMultiplier?: number;
  /** Per-model observed output-token stats (RUN-scoped, mutable): read to price the reserve, and folded
   *  on settle so later reserves in the same run are better estimated. */
  stats?: Map<string, OutputTokenStats>;
  /** Cost-model override (test / consumer seam). Defaults to catalog pricing. */
  pricing?: {
    estimateCostUsd(modelId: string, inputTokens: number, outputTokens: number): number;
    affordableOutputTokens(modelId: string, inputTokens: number, availableUsd: number): number;
  };
}

/**
 * Per-call budget RESERVATION (the reserve→debit wallet lifecycle): before the call, hold its ESTIMATED
 * cost against the injected {@link BudgetMeter}; after it returns, settle the reserve to the ACTUAL
 * cost. When the estimate doesn't fit the balance, FLIP the relationship — compute the AFFORDABLE
 * output ceiling from the remaining headroom and rewrite the op's `maxOutputTokens` to it, so the
 * reserve becomes provider-ENFORCED instead of a guess; still short ⇒ an `out-of-credits` outcome, no
 * call made. A failed call still settles (its cost, usually $0, is real spend and the hold must not
 * linger). With no meter available the wrapper runs the inner call untouched.
 */
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable>(
  config?: BudgetOptions,
): ExecutorWrapper<R, R, M>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable>(
  config: BudgetOptions,
  inner: Executor<R, M>,
): Executor<R, M>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable>(
  inner: Executor<R, M>,
): Executor<R, M>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable>(
  configOrInner?: BudgetOptions | Executor<R, M>,
  maybeInner?: Executor<R, M>,
): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as BudgetOptions | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M> | undefined;
  // Default pricing = the catalog. Un-priced models estimate $0 (they cost the platform nothing; a
  // meter's balance>0 admission floor still applies).
  const estCost = config?.pricing?.estimateCostUsd ?? ((m: string, i: number, o: number) => ModelInfo.instance.computeCostUsd(m, i, o) ?? 0);
  const affordOutput =
    config?.pricing?.affordableOutputTokens ?? ((m: string, i: number, avail: number) => ModelInfo.instance.affordableOutputTokens(m, i, avail));
  const headroom = config?.headroomMultiplier ?? DEFAULT_HOLD_OUTPUT_MULTIPLIER;
  const wrap = ((innerExec: Executor<ExecServices, BudgetReadable>): Executor<ExecServices, BudgetReadable> => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    // Budgeting does not change an op's capabilities, so forward the per-entry accessor unchanged (see
    // {@link withRateLimit}).
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, BudgetReadable> {
      const meter = config?.meter ?? ctx.meter;
      if (!meter || !isPrompt(op)) return innerExec.start(op, ctx);
      // The RESOLVED model: a `defaults`-supplied one used to be invisible here, so this early return
      // fired and the whole call ran unmetered — a complete no-op wrapper, silently.
      const priced = pricedCall(op, config);
      const model = priced.model;
      if (model === undefined) return innerExec.start(op, ctx); // nothing to price
      const declaredMax = priced.maxOutputTokens;
      const stats = config?.stats;
      return wrapHandle<BudgetReadable>(async (ctl): Promise<ExecResult<ResolvedValue, BudgetReadable>> => {
        const inputTokens = estimateInputTokens(priced.text);
        const estOut = estimateOutputTokens(model, inputTokens, declaredMax, stats, headroom);
        let hold = await meter.reserve(estCost(model, inputTokens, estOut));
        let sentOp: Operation<InlineFamily> = op;
        if (!hold) {
          // Tight wallet: clamp the output ceiling to what the remaining balance buys, then retry the
          // reserve ONCE with that cap made real — a truncated answer beats no answer, and the wallet
          // can no longer be overshot.
          const afford = affordOutput(model, inputTokens, await meter.availableCostUsd());
          const clamped = Math.min(afford, declaredMax ?? Number.POSITIVE_INFINITY);
          if (clamped >= MIN_USEFUL_OUTPUT_TOKENS && Number.isFinite(clamped)) {
            hold = await meter.reserve(estCost(model, inputTokens, clamped));
            if (hold) sentOp = withConfig(op, { maxOutputTokens: clamped });
          }
        }
        if (!hold) {
          return {
            error: { classification: "out-of-credits" as const, reason: "the wallet cannot cover this call's estimated cost" },
            metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" as const },
          };
        }
        if (ctl.canceled()) {
          await hold.settle(0);
          return { error: { classification: "canceled" as const, reason: "canceled before the call started" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" as const } };
        }
        let result: ExecResult<ResolvedValue, BudgetReadable>;
        try {
          result = await ctl.started(innerExec.start(sentOp, ctx)).result;
        } catch (err) {
          await hold.settle(0).catch(() => undefined); // no result → no real spend; free the reserve
          throw err;
        }
        // reserve → debit: correct the hold to the ACTUAL cost (a failed call still settles), and feed
        // the observed output tokens back so the next reserve in this run is better priced.
        await hold.settle(result.metrics.costUsd);
        if (stats) noteOutputTokens(stats, model, result.metrics.outputTokens);
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

// --- Sessions ------------------------------------------------------------------

/**
 * Fold a successful result into an assistant turn for the transcript.
 *
 * Reads the op's OUTPUT VALUE. It used to prefer `rawText`, but the model's raw text is `LlmOutput`
 * payload and stops at the prompt executor — and for a text-output op the output value IS that text,
 * so the two agree wherever it mattered.
 */
function foldResultToAssistant(result: ExecResult<ResolvedValue, ExecMetrics>): ModelMessage[] {
  if (!isOk(result) || result.value === undefined) return [];
  const text = typeof result.value === "string" ? result.value : JSON.stringify(result.value);
  return text.length > 0 ? [{ role: "assistant", content: text }] : [];
}

/**
 * Client-managed conversations: resolve the op's LOGICAL `sessionId` against the injected
 * {@link SessionStore}, PREPEND the stored transcript to this call's turn, run the inner call, then
 * FOLD the outcome back into the transcript (only on success — other stored fields are preserved,
 * never clobbered). The session fields are CONSUMED (stripped from the op sent inward — the bare core
 * refuses leftovers); the sent op carries the full transcript as config-layer `messages`, so an inner
 * memoize keys on the real content, and the outcome carries `session.id` (the logical id).
 *
 * LOUD failures instead of silent degradation: a `sessionId` with NO store available is an error, and
 * `providerSessionId` is refused entirely — no current executor can thread a provider-side handle, and
 * resuming "by handle" through the transcript store would silently do the wrong thing. Sits OUTSIDE
 * `withMemoize` (which refuses to wrap it).
 */
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics>(inner: Executor<R, M>): Executor<R & SessionSeams, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<SessionSeams> = {}>(
  config?: P,
): ExecutorWrapper<R, R & Omit<SessionSeams, keyof P>, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<SessionSeams> = {}>(
  config: P,
  inner: Executor<R, M>,
): Executor<R & Omit<SessionSeams, keyof P>, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics>(
  configOrInner?: Partial<SessionSeams> | Executor<R, M>,
  maybeInner?: Executor<R, M>,
): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<SessionSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M> | undefined;
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: { ...innerExec.capabilities, sessionResume: true },
    metrics: innerExec.metrics,
    // A session layer resumes state, so a `withMemoize` above it must refuse to cache. Unlike the other
    // wrappers we cannot forward the inner accessor verbatim: the per-op record has to carry
    // `sessionResume: true` (mirroring `capabilities` above), or a memoize checking per-op caps would see
    // the inner entry's record — which knows nothing about the session — and wrongly cache the call.
    capabilitiesFor: (op: Operation<InlineFamily>): Capabilities => ({
      ...(innerExec.capabilitiesFor?.(op) ?? innerExec.capabilities),
      sessionResume: true,
    }),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      if (!isPrompt(op)) return innerExec.start(op, ctx);
      // Store from construction (standalone) OR the run-scoped `ctx.sessions` (e.g. a workflow run).
      // One transcript store is shared across consumers that pin different message shapes (hw stores
      // `Turn`s, promptop `ModelMessage`s), so `ExecServices` declares it at the JSON base and this is
      // the llm-side view of it — the messages this wrapper reads are exactly the ones it wrote.
      const sessions = (config?.sessions ?? ctx.sessions) as SessionStore<ModelMessage> | undefined;
      const cfg = configOf(op);
      if (cfg.providerSessionId !== undefined) {
        return finished(
          "providerSessionId is set, but provider-side session resume is not supported yet (no executor threads a provider handle) — use a logical sessionId with a SessionStore",
        );
      }
      const key = stringField(cfg, "sessionId");
      if (key === undefined) return innerExec.start(op, ctx);
      if (sessions === undefined) {
        return finished(
          `the declaration carries sessionId "${key}" but no SessionStore is available — provide it via withSession({ sessions }) or ctx.sessions`,
        );
      }
      return wrapHandle(async (ctl) => {
        const priorState = await sessions.get(key);
        const prior = priorState?.messages ?? [];
        // Rewrite the call to carry the history as config-layer `messages`; the lowering appends the
        // op's own `user` text as the final turn, which is exactly the preamble contract.
        const sentOp = withConfig(op, {
          sessionId: undefined,
          providerSessionId: undefined,
          messages: prior as unknown as JsonValue,
        });
        if (ctl.canceled()) return canceledFailure("canceled before the call started");
        const result = await ctl.started(innerExec.start(sentOp, ctx)).result;
        if (isOk(result)) {
          const current: ModelMessage[] = [{ role: "user", content: op.user }];
          await sessions.put(key, { ...priorState, messages: [...prior, ...current, ...foldResultToAssistant(result)] });
        }
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

/** A completed handle carrying a permanent refusal. */
function finished(reason: string): ExecHandle<ResolvedValue> {
  return {
    events: {
      // eslint-disable-next-line require-yield
      async *[Symbol.asyncIterator]() {
        /* no events */
      },
    },
    result: Promise.resolve({ error: { classification: "permanent" as const, reason }, metrics: { durationMs: 0 } }),
    cancel: async () => {},
  };
}
