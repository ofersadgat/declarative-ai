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
  ResolvedSession,

  SessionStore,
  CallEstimate,
} from "@declarative-ai/exec";
import { canceledFailure, curryOrApply, forwardCapabilitiesFor, isExecutor, isOk, isPositionTaken, systemClock, wrapHandle,
  withSessionPosition,
} from "@declarative-ai/exec";
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
import type { LlmOutput, ModelMessage, ResidencyLease, ResidencyManager } from "@declarative-ai/llm";
import { lowerPromptOp, type LoweringOptions } from "./lowering.js";
import { projectLlmOutput } from "./executor.js";

/**
 * What `withBudget` reads off a measurement: money (its job), plus the observed output size it prices
 * the NEXT reserve from. The token field is declared here rather than on `BudgetMetrics` because it is
 * not money — it is this wrapper's estimator wanting feedback, and saying so in the constraint is how
 * an executor that cannot supply it fails to compile instead of silently mis-estimating.
 */
type BudgetReadable = ExecMetrics & BudgetMetrics & { outputTokens?: number };

/** The ctx seam `withSession` consumes. */
type SessionSeams = { sessions: SessionStore };

/** Construction-time knobs, distinct from the ctx seam above. */
interface SessionOptions {
  /**
   * A STABLE discriminator for any stream a call mints, derived from the op.
   *
   * Stable rather than random because lineage names change between runs otherwise, and observability
   * is the reason durable sessions exist. Absent ⇒ the store picks, which is fine for an in-memory
   * one and not for a durable one.
   */
  seedFor?: (op: PromptOp<InlineFamily>) => string;
  /** Which adapter is about to serve the call, so the store hands back THAT provider's handle. The
   *  same stream replayed against two providers has two unrelated handles and both are worth keeping. */
  provider?: string;
}

/**
 * The text a resolved session will put on the wire, for the two PRICING wrappers.
 *
 * They price the call the core will actually make, and the transcript stopped being visible in the
 * op's config the moment `withSession` stopped inlining it there. Without this a 20k-char conversation
 * declares ~7 input tokens again — budgets under-reserve and the limiter under-declares by orders of
 * magnitude on any multi-turn call, which is the exact regression the inlining was covering up.
 *
 * It materializes, and for a replay strategy that costs nothing extra: the executor is about to
 * materialize the same messages. For a native-fork adapter it is a read that the call itself would
 * skip — and pricing a branch by the history the provider will process is still the right answer.
 */
async function sessionText(ctx: ExecServices): Promise<string> {
  const session = ctx.session as ResolvedSession<ModelMessage> | undefined;
  if (session === undefined || typeof session.messages !== "function") return "";
  try {
    return promptText({ messages: await session.messages() } as never);
  } catch {
    // Pricing must never be the thing that fails a call. An unreadable transcript prices as empty,
    // which under-reserves — the same failure mode as before, but no worse, and it does not throw.
    return "";
  }
}

/**
 * Narrow a record-mode result to the op's output-parameter value.
 *
 * A result whose value is not an `LlmOutput` passes through untouched, so composing this over a
 * VALUE-mode core is a no-op rather than a corruption — which matters, because the composition order
 * is the caller's to get right and a silent mangling would be the worst way to find out.
 */
function project<M extends ExecMetrics>(op: PromptOp<InlineFamily>, result: ExecResult<ResolvedValue, M>): ExecResult<ResolvedValue, M> {
  const payload = result.value as LlmOutput | undefined;
  if (payload === undefined || typeof payload !== "object" || !("finishReason" in payload)) return result;
  const value = projectLlmOutput(op, payload);
  return "error" in result && result.error !== undefined
    ? { ...result, ...(value !== undefined ? { value } : { value: undefined }) }
    : ({ ...result, value: value as ResolvedValue } as ExecResult<ResolvedValue, M>);
}

/** Record the EFFECTIVE session position on the outcome — see {@link ExecMetrics.sessionRef}. */
function withSessionOutcome<M extends ExecMetrics>(result: ExecResult<ResolvedValue, M>, sessionRef: string): ExecResult<ResolvedValue, M> {
  return { ...result, metrics: { ...result.metrics, sessionRef } };
}

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

/**
 * Does this wrapper's scope predicate claim this model?
 *
 * The predicates callers pass are llm's own route readers (`isEmbeddedModel`, `isRemoteModel`), and
 * those THROW on an id whose prefix is not a serving route — deliberately, so a typo'd provider id is
 * caught rather than silently mis-routed. That strictness became REACHABLE FROM HERE once a prompt op
 * could be served by something other than a provider: an agent-served call carries an id like
 * `claude-cli/sonnet`, which names an EXECUTOR rather than a serving route, and letting the throw out
 * of `start` would break the never-throws contract at the seam — synchronously, before any handle
 * exists to carry the failure.
 *
 * A predicate that cannot parse an id has, by definition, no opinion about it, so the honest reading
 * of the throw is "not mine". The scope narrows; nothing is silently governed by the wrong limiter,
 * and an agent call does not queue behind a residency manager for weights it will never load.
 */
function governs(appliesTo: ((modelId: string | undefined) => boolean) | undefined, modelId: string | undefined): boolean {
  if (appliesTo === undefined) return true;
  try {
    return appliesTo(modelId);
  } catch {
    return false;
  }
}

// --- Rate limiting -------------------------------------------------------------

/** Options for {@link withRateLimit}: the limiter, the models it GOVERNS, plus the config-resolution
 *  inputs the ESTIMATE is priced against (see {@link pricedCall}). */
export type RateLimitOptions = {
  limiter: RateLimiter;
  /**
   * Which models this limiter governs, by resolved model id. Absent ⇒ ALL of them (the historical
   * behavior, and the right default for a stack whose calls all go to one provider fleet).
   *
   * It exists because rate headroom and MEMORY residency are two different scarcities gating two
   * different sets of models. A remote call waits for provider quota; a local one waits for weights to
   * be resident. Splitting them by model id lets each layer govern the set it actually knows about —
   * `withRateLimit({ appliesTo: isRemote })` outside a residency manager scoped to the local ones.
   *
   * That split is not merely tidy, it is what keeps the composition SAFE. Both layers are bounded
   * resources acquired in nested order, so if one model is gated by both there is a lock ordering:
   * call A holds a residency lease and waits for a concurrency slot, while call B holds the slot and
   * waits for a lease that cannot be granted until A finishes. Disjoint sets make that cycle
   * unconstructible.
   *
   * A call whose model no layer resolved arrives as `undefined` — the predicate decides, because
   * "governed by default" and "ungoverned by default" are both defensible and only the caller knows
   * which fleet an unnamed model would land in.
   */
  appliesTo?: (modelId: string | undefined) => boolean;
} & ResolutionOptions;

/**
 * Rate limiting: admit the inner call through the injected `RateLimiter` (concurrency slot + rate
 * headroom) using a token estimate off the prompt text, and feed the outcome back (`reportOutcome`
 * drives AIMD). A cancel that lands while the call is still QUEUED prevents it from ever starting
 * (returns a `canceled` outcome); a limiter fault is normalized into a permanent failure.
 *
 * Scoped to the models {@link RateLimitOptions.appliesTo} names; anything else passes through to the
 * inner executor untouched, exactly as a non-prompt op does.
 */
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: RateLimitOptions,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: RateLimitOptions,
  inner: Executor<R, M, Operation<InlineFamily>, Out>,
): Executor<R, M, Operation<InlineFamily>, Out>;
export function withRateLimit<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: RateLimitOptions,
  inner?: Executor<R, M, Operation<InlineFamily>, Out>,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out> | Executor<R, M, Operation<InlineFamily>, Out> {
  const { limiter, appliesTo } = config;
  /** Per-op token-estimate cache: the estimate is derived from the full prompt text (potentially a long
   *  transcript), and the SAME op object is re-submitted per repair attempt / retry. Scoped to THIS
   *  wrapper because the estimate now depends on its `defaults`/`configs` too — a module-level cache
   *  would hand one stack's resolution to another's. */
  const estimateCache = new WeakMap<object, CallEstimate>();
  /** The RESOLUTION, cached separately from the estimate because `appliesTo` needs only the model id
   *  and must decide BEFORE the estimate is computed — tokenizing a 20k-char transcript to price a call
   *  this limiter then declines to govern is pure waste. Same key and same lifetime as the estimate
   *  cache, so the two never disagree about which call they describe. */
  const pricedCache = new WeakMap<object, PricedCall>();
  const pricedFor = (op: PromptOp<InlineFamily>): PricedCall => {
    let priced = pricedCache.get(op);
    if (priced === undefined) {
      priced = pricedCall(op, config);
      pricedCache.set(op, priced);
    }
    return priced;
  };
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    // Forward per-dispatched-entry capabilities so a `withMemoize` ABOVE this wrapper still gates on the
    // real entry's `memoizable`/`mutatesWorkspace` rather than the static record. Rate-limiting changes
    // nothing about an op's capabilities, so a straight passthrough is correct.
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      if (!isPrompt(op)) return innerExec.start(op, ctx);
      const priced = pricedFor(op);
      // An UNGOVERNED model is handed straight to the inner executor — the same untouched passthrough a
      // non-prompt op gets, rather than a wrapped handle that would only forward. Deciding here, before
      // `wrapHandle`, is what keeps the inner handle's event stream identical for a call this limiter
      // has nothing to say about.
      if (!governs(appliesTo, priced.model)) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        const cached = estimateCache.get(op);
        // Tokenizing the resolved text is the expensive half, so it happens on the cache MISS — the
        // resolution itself is already shared with the `appliesTo` check above.
        const base = cached ?? { ...estimateCallTokens(priced.text, undefined, priced.maxOutputTokens), modelId: priced.model };
        if (cached === undefined) estimateCache.set(op, base);
        // Added OUTSIDE the cache: the op is the same object across repair attempts and retries, but
        // the session it runs against has moved on, so a cached transcript size would be stale.
        const prior = estimateInputTokens(await sessionText(ctx));
        const est = prior > 0 ? { ...base, inputTokens: base.inputTokens + prior } : base;
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
  })) as unknown as ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
  return curryOrApply(wrap, inner);
}

// --- Model residency -----------------------------------------------------------

/** Options for {@link withModelManager}: the arbiter, the models it governs, and the config-resolution
 *  inputs used to learn which model a call wants. */
export type ModelManagerOptions = {
  manager: ResidencyManager;
  /**
   * Which models this manager governs. Absent ⇒ ALL of them.
   *
   * The correct pairing is `withModelManager({ appliesTo: isEmbeddedModel })` outside
   * `withRateLimit({ appliesTo: (id) => !isEmbeddedModel(id) })` — total and disjoint.
   *
   * Note it is `isEmbeddedModel`, NOT `isLocalModel`. A `local/` server runs on your hardware but its
   * memory belongs to another process that swaps on its own schedule, so managing residency for it
   * would serialize work on bookkeeping we cannot enforce. It still has finite capacity and is worth
   * rate-limiting, which is why the limiter takes the complement rather than `isRemoteModel`.
   *
   * Disjointness is not merely tidy: both layers are bounded resources acquired in nested order, so an
   * overlap admits a lock ordering — see {@link RateLimitOptions.appliesTo}.
   */
  appliesTo?: (modelId: string | undefined) => boolean;
} & ResolutionOptions;

/**
 * MODEL RESIDENCY: hold a lease on the model's weights for the duration of the call.
 *
 * Everything interesting is in the {@link ResidencyManager}; this wrapper's whole job is to learn which
 * model the call wants, take a lease before dispatching, and give it back afterwards. Two details are
 * what make it correct rather than merely plausible:
 *
 *  - **The lease is released on EVERY path** — success, failure, cancel-while-queued, and a throw. A
 *    lease never returned pins a model resident forever, and since eviction requires zero leases, one
 *    leak permanently poisons a slot on a machine that may only have one.
 *  - **A cancel that lands while QUEUED never loads the model.** Waiting for residency can take minutes
 *    (an eviction, then a multi-gigabyte load), so a caller that gave up must not cause the work.
 *
 * Compose it OUTSIDE `withDeadline`: queueing and loading are not the call's own latency, and charging
 * a 60-second model load to the call's window turns "the machine was busy" into a deadline failure that
 * reads like provider slowness.
 */
export function withModelManager<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: ModelManagerOptions,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
export function withModelManager<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: ModelManagerOptions,
  inner: Executor<R, M, Operation<InlineFamily>, Out>,
): Executor<R, M, Operation<InlineFamily>, Out>;
export function withModelManager<R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  config: ModelManagerOptions,
  inner?: Executor<R, M, Operation<InlineFamily>, Out>,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out> | Executor<R, M, Operation<InlineFamily>, Out> {
  const { manager, appliesTo } = config;
  const modelCache = new WeakMap<object, string | undefined>();
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    ...forwardCapabilitiesFor(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      if (!isPrompt(op)) return innerExec.start(op, ctx);
      let modelId = modelCache.get(op);
      if (!modelCache.has(op)) {
        modelId = pricedCall(op, config).model;
        modelCache.set(op, modelId);
      }
      // An ungoverned model is handed straight through, untouched — the same passthrough a non-prompt
      // op gets. A model with no residency to manage must not queue behind one that has.
      if (!governs(appliesTo, modelId)) return innerExec.start(op, ctx);
      if (modelId === undefined) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        let lease: ResidencyLease;
        const waitFrom = (ctx.clock ?? systemClock).now();
        try {
          lease = await manager.acquire(modelId, ctl.signal);
        } catch (err) {
          // A REFUSED placement is permanent — the machine cannot run this model and retrying changes
          // nothing — while a cancel is a cancel. Anything else (a probe or policy fault) is reported
          // as permanent too rather than being swallowed into a silent admission.
          if (ctl.canceled()) return canceledFailure("canceled while queued for model residency");
          return {
            error: { classification: "permanent" as const, reason: err instanceof Error ? err.message : String(err) },
            metrics: { durationMs: 0 },
          };
        }
        // Measured across the WHOLE wait — queueing behind other calls, an eviction, and the load that
        // followed it — because from the caller's side those are one indivisible "not started yet".
        const queuedMs = (ctx.clock ?? systemClock).now() - waitFrom;
        try {
          if (ctl.canceled()) return canceledFailure("canceled while queued for model residency");
          const result = await ctl.started(innerExec.start(op, ctx)).result;
          // Added rather than overwritten: an inner layer that also queued (a rate limiter) already
          // recorded its own wait, and the call really did spend both.
          return { ...result, metrics: { ...result.metrics, queuedMs: (result.metrics.queuedMs ?? 0) + queuedMs } };
        } finally {
          // On EVERY path. A lease never returned pins the model resident forever, and eviction needs
          // zero leases — so one leak permanently costs a slot on a machine that may only have one.
          lease.release();
        }
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
  return curryOrApply(wrap, inner);
}

// --- Budget --------------------------------------------------------------------

/** Options for {@link withBudget}. All optional — with no `meter` the wrapper
 *  is a pure passthrough (an absent service is a no-op, like the rest of the stack). */
export interface BudgetOptions extends ResolutionOptions {
  /** The metered wallet. Absent ⇒ the wrapper is a passthrough, as every absent seam is. */
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
  /**
   * Selects POST-CHARGE mode: this layer's cost is COMPUTED FROM THE RESULT instead of reserved
   * against a pre-call estimate. The mode is implied rather than a separate flag because it follows
   * from the data dependency — a computed charge prices a result that does not exist before the call,
   * so there is nothing to reserve.
   *
   * The wrapper runs the inner executor, then bills `computeCost(op, result)` when it is > 0: via
   * {@link BudgetMeter.debit} (the channel for spend that cannot be refused), falling back to an
   * immediate reserve→settle for a meter without one — which may under-collect on an empty balance,
   * since a post-hoc charge cannot un-serve the result. The amount is folded into the result's
   * `costUsd` so the layers above see the true spend.
   *
   * This is what an OUTER budget instance stacked ABOVE `withMemoize` uses to bill memo REUSE: the
   * cache implementation annotates a hit's metrics with what the original run cost and who already
   * owns it, and `computeCost` turns that into this principal's charge — 0 on a miss (no annotation),
   * so real calls are billed once, by the INNER reserve-mode instance. Unlike reserve mode this
   * applies to EVERY op kind, not just prompts: any op's cached record can carry a reuse charge.
   */
  computeCost?: (op: Operation<InlineFamily>, result: ExecResult<ResolvedValue, BudgetReadable>) => number;
}

/**
 * The ONE billing wrapper, in one of two modes selected by {@link BudgetOptions.computeCost}:
 *
 * **Reserve mode** (no `computeCost`) — per-call budget RESERVATION (the reserve→debit wallet
 * lifecycle): before the call, hold its ESTIMATED cost against the injected {@link BudgetMeter}; after
 * it returns, settle the reserve to the ACTUAL cost. When the estimate doesn't fit the balance, FLIP
 * the relationship — compute the AFFORDABLE output ceiling from the remaining headroom and rewrite the
 * op's `maxOutputTokens` to it, so the reserve becomes provider-ENFORCED instead of a guess; still
 * short ⇒ an `out-of-credits` outcome, no call made. A failed call still settles (its cost, usually
 * $0, is real spend and the hold must not linger).
 *
 * **Post-charge mode** (`computeCost` present) — run the inner executor, then debit what the RESULT
 * says this layer owes (see {@link BudgetOptions.computeCost}). An outer post-charge instance above
 * `withMemoize` and an inner reserve instance below it is the standard composition: hits are billed by
 * the outer layer from the record's reuse annotation, real calls by the inner reserve.
 *
 * With no meter available the wrapper runs the inner call untouched, in either mode.
 */
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable, Out = ResolvedValue>(
  config?: BudgetOptions,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable, Out = ResolvedValue>(
  config: BudgetOptions,
  inner: Executor<R, M, Operation<InlineFamily>, Out>,
): Executor<R, M, Operation<InlineFamily>, Out>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable, Out = ResolvedValue>(
  inner: Executor<R, M, Operation<InlineFamily>, Out>,
): Executor<R, M, Operation<InlineFamily>, Out>;
export function withBudget<R = ExecServices, M extends BudgetReadable = BudgetReadable, Out = ResolvedValue>(
  configOrInner?: BudgetOptions | Executor<R, M, Operation<InlineFamily>, Out>,
  maybeInner?: Executor<R, M, Operation<InlineFamily>, Out>,
): ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out> | Executor<R, M, Operation<InlineFamily>, Out> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as BudgetOptions | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M, Operation<InlineFamily>, Out> | undefined;
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
      // CONSTRUCTION only. A wallet is what this wrapper spends from; a caller composing it has one in
      // hand. It was never something the executor needed to make a call — and while it was reachable
      // from ctx, an executor did reach for it, and charged money the wrapper was already charging.
      const meter = config?.meter;
      const computeCost = config?.computeCost;
      if (computeCost) {
        if (!meter) return innerExec.start(op, ctx);
        return wrapHandle<BudgetReadable>(async (ctl): Promise<ExecResult<ResolvedValue, BudgetReadable>> => {
          const result = await ctl.started(innerExec.start(op, ctx)).result;
          const amount = computeCost(op, result);
          if (!(amount > 0)) return result;
          if (meter.debit) {
            await meter.debit(amount);
          } else {
            // No debit channel: an immediate reserve→settle is the closest a refusable meter gets. A
            // refusal is swallowed — the result was already served, so refusing cannot un-spend it —
            // which is exactly the under-collection `BudgetMeter.debit`'s doc warns about.
            const hold = await meter.reserve(amount);
            if (hold) await hold.settle(amount);
          }
          // The charge is real spend of THIS execution: fold it into the reported cost so run totals
          // and outer layers see it. `costSource` is untouched — the amount came from the caller's own
          // cost model, and relabeling the underlying measurement would destroy its provenance.
          return { ...result, metrics: { ...result.metrics, costUsd: result.metrics.costUsd + amount } };
        });
      }
      if (!meter || !isPrompt(op)) return innerExec.start(op, ctx);
      // The RESOLVED model: a `defaults`-supplied one used to be invisible here, so this early return
      // fired and the whole call ran unmetered — a complete no-op wrapper, silently.
      const priced = pricedCall(op, config);
      const model = priced.model;
      if (model === undefined) return innerExec.start(op, ctx); // nothing to price
      const declaredMax = priced.maxOutputTokens;
      const stats = config?.stats;
      return wrapHandle<BudgetReadable>(async (ctl): Promise<ExecResult<ResolvedValue, BudgetReadable>> => {
        // The transcript counts. It is no longer inlined into the op's config, so a reserve priced on
        // `priced.text` alone would be blind to every prior turn the call is about to resend.
        const inputTokens = estimateInputTokens(priced.text) + estimateInputTokens(await sessionText(ctx));
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
  })) as unknown as ExecutorWrapper<R, R, M, Operation<InlineFamily>, Out>;
  return curryOrApply(wrap, inner);
}

// --- Sessions ------------------------------------------------------------------

/**
 * Sessions: RESERVE a position, run the call with the resolved session on the services bundle, fold
 * what the executor says it appended, release.
 *
 * ## What changed, and why each half of it is load-bearing
 *
 * **The wrapper no longer rewrites the op's config.** It used to do `withConfig(op, { messages })`,
 * which hardcoded replay — that is precisely why `providerSessionId` had to be refused outright:
 * there was no way for a stateful adapter to be told "resume this handle" rather than "here is the
 * whole transcript again". The resolved session now rides on `ExecServices`, and the executor shapes
 * its own request. The wrapper keeps the POLICY (resolve, reserve, decide, fold, release); the
 * executor owns the MECHANISM (build the request, perform the fork, report what it appended).
 *
 * **The wrapper no longer synthesizes the transcript.** It only ever saw `op.user` and a final result
 * value, so it wrote one user turn and one stringified assistant turn and threw away every tool call,
 * tool result and reasoning part in between. The executor reports the real delta instead.
 *
 * **The fold runs on failure too.** If the provider appended turns and the call then failed, those
 * entries exist remotely. Not recording them means the next append-by-handle meets a remote head we
 * do not mirror — which is divergence on the very next call.
 *
 * **The outcome carries the EFFECTIVE id.** A store that had to fork reports a different id than the
 * one it was asked for, and echoing the input key back would hand the caller a position its own call
 * did not end at.
 *
 * ## Reserve, don't observe
 *
 * The reservation is taken BEFORE the call and released unconditionally — success, failure, cancel
 * and throw. A peek would leave a window the length of the whole model call, during which two calls
 * both read head == 14 and both decide "linear append". That is survivable if both replay, since the
 * loser forks on the way out; it is NOT survivable if the winner took a resume-by-handle fast path,
 * because by the time it loses it has already appended remotely.
 *
 * A leaked reservation is the other failure worth naming: it pins the stream forever and silently
 * forks everything downstream, which looks like the system working.
 *
 * Still sits OUTSIDE `withMemoize`, which refuses to wrap it.
 */
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics>(inner: Executor<R, M>): Executor<R & SessionSeams, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<SessionSeams> & SessionOptions = {}>(
  config?: P,
): ExecutorWrapper<R, R & Omit<SessionSeams, keyof P>, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<SessionSeams> & SessionOptions = {}>(
  config: P,
  inner: Executor<R, M>,
): Executor<R & Omit<SessionSeams, keyof P>, M>;
export function withSession<R = ExecServices, M extends ExecMetrics = ExecMetrics>(
  configOrInner?: (Partial<SessionSeams> & SessionOptions) | Executor<R, M>,
  maybeInner?: Executor<R, M>,
): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as (Partial<SessionSeams> & SessionOptions) | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M> | undefined;
  const wrap = ((innerExec: Executor): Executor => {
    // ONE implementation of the session policy, in `exec`. This wrapper used to carry a second copy —
    // its own resolve, its own fork-on-`PositionTaken`, its own position stamping — which meant two
    // places to keep the fork rules right and only one of them (the `exec` one) reachable from `hw`.
    // What is genuinely THIS layer's is the FRONT-END: reading a session out of an op's `config`,
    // which is llm vocabulary. So that is all that is left here; the policy is delegated.
    const positioned = withSessionPosition(
      config?.sessions !== undefined ? { sessions: config.sessions } : {},
      innerExec as never,
    ) as unknown as Executor;
    return {
      capabilities: positioned.capabilities,
      metrics: innerExec.metrics,
      capabilitiesFor: (op: Operation<InlineFamily>): Capabilities =>
        positioned.capabilitiesFor?.(op) ?? positioned.capabilities,
      start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
        if (!isPrompt(op)) return innerExec.start(op, ctx);
        const sessions = config?.sessions as SessionStore<ModelMessage> | undefined;
        const cfg = configOf(op);
        const ref = stringField(cfg, "sessionId");
        const providerHandle = stringField(cfg, "providerSessionId");
        const fork = cfg.fork === true;
        // No session named anywhere ⇒ nothing to do. `fork` alone is meaningless — it says how to
        // consume a position, and there is no position.
        if (ref === undefined && providerHandle === undefined) return innerExec.start(op, ctx);
        if (sessions === undefined) {
          return finished(
            `the declaration carries session "${ref ?? providerHandle}" but no SessionStore is available — provide it via withSession({ sessions })`,
          );
        }
        // The session fields are CONSUMED — the bare core refuses leftovers, which is what stops a
        // declaration quietly relying on a layer that is not composed in.
        const sentOp = withConfig(op, { sessionId: undefined, providerSessionId: undefined, fork: undefined });
        // RESOLVE, then delegate. The declaration names a conversation; the policy layer below wants a
        // position. Looking it up here is this wrapper's whole job — reading a session out of
        // `op.config` is llm vocabulary, and everything after the lookup (claim, fork on conflict,
        // detect divergence) is the one implementation in `exec`.
        return wrapHandle(async (ctl) => {
          if (ctl.canceled()) return canceledFailure("canceled before the call started");
          const resolved = await sessions.resolve({
            ...(ref !== undefined ? { ref } : {}),
            ...(fork ? { fork: true } : {}),
            ...(config?.seedFor !== undefined ? { seed: config.seedFor(op) } : {}),
            ...(config?.provider !== undefined ? { provider: config.provider } : {}),
          });
          return await ctl.started(positioned.start(sentOp, { ...ctx, session: resolved as never })).result;
        });
      },
    };
  }) as unknown as ExecutorWrapper<R, R, M>;
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
