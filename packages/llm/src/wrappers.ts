/**
 * Composable cross-cutting concerns for the `llm-call` core (see ./executor), each an `ExecutorWrapper`
 * (`Executor → Executor`) stacked via `composeExecutors`. Keeping these OUT of the core is the separation:
 * the core is one call; you opt into repair / rate-limiting / deadline fail-fast / sessions / memoization
 * by wrapping.
 *
 * Deps model: SERVICES (a rate limiter, a memo cache, a session store) and per-deployment POLICY (repair
 * turns) are captured at CONSTRUCTION — explicit and composable. Each wrapper CONSUMES its own trigger
 * (`withDeadline` strips `ctx.deadline`, `withSession` strips the declaration's session ids); the bare
 * core REFUSES anything left unconsumed, so a mis-composed stack fails loudly on first use instead of
 * silently degrading.
 *
 * Handle semantics (via `wrapHandle`): inner event streams are FORWARDED (concatenated) to the outer
 * handle — never swallowed; `cancel()` covers the window BEFORE the inner call starts (e.g. queued in a
 * rate limiter), so a canceled call never starts and a started one is aborted; a wrapper-body throw (a
 * faulty limiter, a broken store) is normalized into a permanent-failure `Outcome`, never a rejected
 * `outcome` promise.
 */
import type {
  BudgetMeter,
  DeadlineConfig,
  ExecHandle,
  ExecMetrics,
  ExecServices,
  ExecutionSpec,
  Executor,
  ExecutorWrapper,
  MemoCache,
  MemoizeOptions,
  Outcome,
  RateLimiter,
  SessionStore,
  CallEstimate,
} from "@declarative-ai/core";
import { hashCanonical, memoKey } from "@declarative-ai/core";
import { deadlineDecision, estimateCallTokens, systemClock, DEADLINE_FLOOR_REASON } from "@declarative-ai/services";
import type { ModelMessage } from "ai";
import { emptyEvents } from "./executor";
import { promptAsMessages, promptText } from "./generate";
import type { LlmCallDefinition, StructuredCallParams } from "./llmStep";
import {
  estimateInputTokens,
  estimateOutputTokens,
  noteOutputTokens,
  DEFAULT_HOLD_OUTPUT_MULTIPLIER,
  MIN_USEFUL_OUTPUT_TOKENS,
  type OutputTokenStats,
} from "./costEstimate";
import { ModelInfo } from "./model-catalog";

/** The prompt-carrying fields of a call, threaded through the repair loop (string OR message prompt). */
type PromptFields = Pick<StructuredCallParams, "system" | "prompt" | "messages">;

/** A completed handle wrapping a ready outcome (for wrappers that short-circuit, e.g. deadline fail-fast). */
function finishedHandle(outcome: Outcome): ExecHandle {
  return { events: emptyEvents(), outcome: Promise.resolve(outcome), cancel: async () => {} };
}

/** A zero-cost permanent-failure outcome (wrapper-level refusals and normalized wiring faults). */
function permanentOutcome(reason: string): Outcome {
  return { rawText: "", finishReason: "error", metrics: { durationMs: 0 }, error: { classification: "permanent", reason } };
}

/** A zero-cost canceled outcome (cancel landed before any inner call started). */
function canceledOutcome(reason: string): Outcome {
  return { rawText: "", finishReason: "error", metrics: { durationMs: 0 }, error: { classification: "canceled", reason } };
}

/** The control surface {@link wrapHandle} hands a wrapper body: check whether cancel already landed
 *  (BEFORE starting an inner call), and register each inner handle as it starts — registration forwards
 *  its events and makes it the cancel target. */
interface WrapControl {
  canceled(): boolean;
  started(h: ExecHandle): ExecHandle;
}

/**
 * The shared handle scaffold for wrappers whose body starts inner handles asynchronously. Every inner
 * handle registered via `ctl.started` has its events forwarded (concatenated, in registration order) to
 * the outer `events` stream. `cancel()` flips the canceled flag FIRST — so a body that hasn't started its
 * inner call yet can observe it and short-circuit (the pre-start window a plain `innerHandle?.cancel()`
 * misses) — then aborts the current inner handle and awaits completion. A body throw/rejection is
 * normalized into a permanent-failure outcome (the `outcome` promise NEVER rejects, per the contract).
 */
function wrapHandle(body: (ctl: WrapControl) => Promise<Outcome>): ExecHandle {
  const registered: ExecHandle[] = [];
  let current: ExecHandle | undefined;
  let canceled = false;
  let done = false;
  let notify: (() => void) | undefined;
  const wake = (): void => {
    const n = notify;
    notify = undefined;
    n?.();
  };
  const ctl: WrapControl = {
    canceled: () => canceled,
    started(h) {
      current = h;
      registered.push(h);
      if (canceled) void h.cancel();
      wake();
      return h;
    },
  };
  const outcome = body(ctl)
    .catch((err) => permanentOutcome(err instanceof Error ? err.message : String(err)))
    .finally(() => {
      done = true;
      wake();
    });
  return {
    events: {
      async *[Symbol.asyncIterator]() {
        let i = 0;
        for (;;) {
          while (i < registered.length) yield* registered[i++]!.events;
          if (done) return;
          await new Promise<void>((resolve) => {
            notify = resolve;
          });
        }
      },
    },
    outcome,
    cancel: async () => {
      canceled = true;
      await current?.cancel();
      await outcome.catch(() => undefined);
    },
  };
}

const sum = (a: number | undefined, b: number | undefined): number | undefined =>
  a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0);

/** Numeric-summing accumulation of one attempt's metrics into the running total (repair loop). `startMs`
 *  stays the FIRST attempt's; `durationMs` sums; provenance/rawUsage take the latest. */
function accumulateMetrics(total: ExecMetrics | undefined, m: ExecMetrics): ExecMetrics {
  if (!total) return { ...m };
  return {
    startMs: total.startMs ?? m.startMs,
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
    durationMs: (total.durationMs ?? 0) + (m.durationMs ?? 0),
    childCalls: sum(total.childCalls, m.childCalls),
    childCost: sum(total.childCost, m.childCost),
  };
}

/** A validation failure eligible for repair: retriable AND validation-caused. */
function isValidationFailure(outcome: Outcome): boolean {
  return outcome.error !== undefined && outcome.error.classification === "api-retriable" && /validation/i.test(outcome.error.reason);
}

/** The repair suffix appended to the prompt after a validation failure. */
function repairSuffix(errors: string): string {
  return `\n\nYour previous output failed schema validation: ${errors}. Return ONLY corrected JSON matching the schema.`;
}

/** Re-prompt after a validation failure with the concrete errors appended. A string prompt gets the suffix
 *  concatenated; a message-based prompt (`messages` or a `ModelMessage[]` prompt) gets the errors as an
 *  appended user turn (normalized to `messages`), so repair works regardless of prompt shape. */
function withRepairHint(base: PromptFields, errors: string): PromptFields {
  if (!base.messages && !Array.isArray(base.prompt)) {
    return { ...base, prompt: `${typeof base.prompt === "string" ? base.prompt : ""}${repairSuffix(errors)}` };
  }
  const hint: ModelMessage = { role: "user", content: repairSuffix(errors) };
  return { ...base, prompt: undefined, messages: [...promptAsMessages(base), hint] };
}

/** Dual-mode dispatch: a wrapper called WITHOUT an inner executor returns the curried `ExecutorWrapper`
 *  (for the `compose(...).with(...)` builder / `composeExecutors`); called WITH one it applies immediately,
 *  so form-1 direct nesting reads inside-out: `withMemoize({cache}, withDeadline(withRateLimit({limiter}, core)))`. */
function curryOrApply<RIn, ROut>(
  wrap: ExecutorWrapper<RIn, ROut>,
  inner?: Executor<RIn>,
): ExecutorWrapper<RIn, ROut> | Executor<ROut> {
  return inner ? wrap(inner) : wrap;
}

/** The ctx SEAMS each wrapper consumes — its `config` object mirrors these (see the per-wrapper docs). A seam
 *  PROVIDED at construction drops out of what `.start` requires; an omitted one stays required (`Omit`-tracked). */
type DeadlineSeams = { deadline: DeadlineConfig; stepStartMs: number };
type SessionSeams = { sessions: SessionStore };

/** True iff a value is an `Executor` (has `.start`) — disambiguates a wrapper's optional trailing `inner`
 *  from its optional config/store argument (both structurally disjoint from `Executor`). */
function isExecutor(x: unknown): x is Executor {
  return typeof x === "object" && x !== null && typeof (x as Executor).start === "function";
}

/**
 * Deadline fail-fast + timeout clamp (§time-vs-money; see DESIGN §3.5 for how `timeoutMs` vs `deadline`
 * vs this wrapper relate). Its `config` mirrors the ctx seams it reads — `{ deadline, stepStartMs }` — and
 * whatever you supply at CONSTRUCTION drops out of what `.start` requires (the rest is read from ctx): so
 * `withDeadline({ deadline })` needs only `stepStartMs` at start, `withDeadline({ deadline, stepStartMs })`
 * needs neither, and `withDeadline()` needs both. Below the start floor it short-circuits with a `deadline`
 * failure and NEVER starts the inner call; otherwise it lowers `spec.limits.timeoutMs` (and the definition's
 * own `timeoutMs`, when larger) to the remaining window. Pass an inner executor as the last argument to apply
 * it directly (form 1); omit it for the curried wrapper (builder / composeExecutors).
 */
export function withDeadline<R = ExecServices>(inner: Executor<R>): Executor<R & DeadlineSeams>;
export function withDeadline<R = ExecServices, P extends Partial<DeadlineSeams> = {}>(config?: P): ExecutorWrapper<R, R & Omit<DeadlineSeams, keyof P>>;
export function withDeadline<R = ExecServices, P extends Partial<DeadlineSeams> = {}>(config: P, inner: Executor<R>): Executor<R & Omit<DeadlineSeams, keyof P>>;
export function withDeadline<R = ExecServices>(
  configOrInner?: Partial<DeadlineSeams> | Executor<R>,
  maybeInner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<DeadlineSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R> | undefined;
  // Body typed against `ExecServices` (unchanged shape); the generic signature ADDS the seam(s) the config
  // does NOT provide to what `start` requires — the cast bridges the two (the wrapper genuinely reads them).
  const wrap = ((innerExec: Executor): Executor => ({
    kind: innerExec.kind,
    capabilities: innerExec.capabilities,
    start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
      const { deadline: ctxDeadline, stepStartMs: ctxStep, ...restCtx } = ctx;
      const deadline = config?.deadline ?? ctxDeadline; // construction config wins; else read from ctx
      const stepStartMs = config?.stepStartMs ?? ctxStep;
      if (deadline) {
        if (stepStartMs === undefined) {
          return finishedHandle(permanentOutcome("the deadline is set without a stepStartMs — deadline arithmetic needs the step-start origin"));
        }
        const now = (ctx.clock ?? systemClock).now();
        const decision = deadlineDecision(stepStartMs, deadline, now);
        if (!decision.proceed) {
          return finishedHandle({
            rawText: "",
            finishReason: "error",
            metrics: { startMs: now, durationMs: 0 },
            error: {
              classification: "deadline",
              reason: `${DEADLINE_FLOOR_REASON}: ${decision.remainingMs}ms remaining is below the start floor`,
            },
          });
        }
        const ceiling = Math.min(spec.limits?.timeoutMs ?? Number.POSITIVE_INFINITY, decision.remainingMs);
        const def = spec.definition as LlmCallDefinition | undefined;
        const definition = def?.timeoutMs !== undefined && def.timeoutMs > ceiling ? { ...def, timeoutMs: ceiling } : spec.definition;
        spec = { ...spec, limits: { ...spec.limits, timeoutMs: ceiling }, definition };
      }
      return innerExec.start(spec, restCtx);
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/** Per-definition token-estimate cache: the estimate is derived from the full prompt text (potentially a
 *  long transcript), and the SAME definition object is re-submitted per repair attempt / retry. */
const estimateCache = new WeakMap<object, CallEstimate>();

/**
 * Rate limiting: admit the inner call through the injected `RateLimiter` (concurrency slot + rate headroom)
 * using a token estimate off the prompt text, and feed the outcome back (`reportOutcome` drives AIMD).
 * A cancel that lands while the call is still QUEUED prevents it from ever starting (returns a `canceled`
 * outcome); a limiter fault (schedule/reportOutcome throwing) is normalized into a permanent failure.
 */
export function withRateLimit<R = ExecServices>(config: { limiter: RateLimiter }): ExecutorWrapper<R, R>;
export function withRateLimit<R = ExecServices>(config: { limiter: RateLimiter }, inner: Executor<R>): Executor<R>;
export function withRateLimit<R = ExecServices>(config: { limiter: RateLimiter }, inner?: Executor<R>): ExecutorWrapper<R, R> | Executor<R> {
  const { limiter } = config;
  const wrap = ((innerExec: Executor): Executor => ({
    kind: innerExec.kind,
    capabilities: innerExec.capabilities,
    start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
      const def = spec.definition as LlmCallDefinition;
      return wrapHandle(async (ctl) => {
        let est = estimateCache.get(def);
        if (!est) {
          est = { ...estimateCallTokens(promptText(def), undefined, def.maxOutputTokens), modelId: def.model };
          estimateCache.set(def, est);
        }
        let ran = false;
        const result = await limiter.schedule(est, () => {
          if (ctl.canceled()) return Promise.resolve(canceledOutcome("canceled while queued for rate-limit admission"));
          ran = true;
          return ctl.started(innerExec.start(spec, ctx)).outcome;
        });
        if (ran) limiter.reportOutcome({ rateLimited: result.error?.rateLimited, modelId: def.model });
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/**
 * The opt-in bounded output-repair loop: on a schema-VALIDATION failure, re-invoke the inner executor with
 * the concrete errors appended to the prompt, up to `turns` extra turns (policy fixed at construction),
 * accumulating metrics across attempts. Any non-validation failure (or a success) stops immediately. The
 * augmented definition itself carries the repair hint, so an inner memoize (which derives its key by
 * hashing the definition) keys on exactly what is sent — no separate hash for this wrapper to keep in sync.
 */
export function withRepair<R = ExecServices>(config: { turns: number }): ExecutorWrapper<R, R>;
export function withRepair<R = ExecServices>(config: { turns: number }, inner: Executor<R>): Executor<R>;
export function withRepair<R = ExecServices>(config: { turns: number }, inner?: Executor<R>): ExecutorWrapper<R, R> | Executor<R> {
  const { turns } = config;
  const wrap = ((innerExec: Executor): Executor => ({
    kind: innerExec.kind,
    capabilities: innerExec.capabilities,
    start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
      return wrapHandle(async (ctl) => {
        const def = spec.definition as LlmCallDefinition;
        const base: PromptFields = { system: def.system, prompt: def.prompt, messages: def.messages };
        let accumulated: ExecMetrics | undefined;
        let last: Outcome | undefined;
        let currentSpec = spec;
        let turn = 0;
        while (!ctl.canceled()) {
          last = await ctl.started(innerExec.start(currentSpec, ctx)).outcome;
          accumulated = accumulateMetrics(accumulated, last.metrics);
          if (ctl.canceled() || !isValidationFailure(last) || turn >= turns) break;
          turn++;
          const augmented: LlmCallDefinition = { ...def, ...withRepairHint(base, last.error!.reason) };
          currentSpec = { ...spec, definition: augmented };
        }
        if (!last) return canceledOutcome("canceled before the call started");
        return { ...last, metrics: accumulated! };
      });
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/** Fold a successful outcome into an assistant turn for the transcript. Phase 4 handles text + structured
 *  value (the JSON the model "said"); tool-call and file folding are later refinements. */
function foldOutcomeToAssistant(outcome: Outcome): ModelMessage[] {
  const text = outcome.rawText && outcome.rawText.length > 0 ? outcome.rawText : outcome.value !== undefined ? JSON.stringify(outcome.value) : "";
  return text.length > 0 ? [{ role: "assistant", content: text }] : [];
}

/**
 * Client-managed conversations: resolve the declaration's LOGICAL `sessionId` against the injected
 * {@link SessionStore}, PREPEND the stored transcript to this call's turn, run the inner call, then FOLD
 * the outcome back into the transcript (only on success — other stored fields are preserved, never
 * clobbered). The session fields are CONSUMED (stripped from the definition sent inward — the bare core
 * refuses leftovers); the sent definition carries the full transcript, so an inner memoize (which derives
 * its key by hashing the definition) keys on the real content, and the outcome carries `session.id`
 * (the logical id — the client-side continuation token).
 *
 * LOUD failures instead of silent degradation: a `sessionId` with NO store available is an error, and
 * `providerSessionId` is refused entirely — no current executor can thread a provider-side handle (that
 * arrives with the agent-sdk executor); resuming "by handle" through the transcript store would silently
 * do the wrong thing. Sits OUTSIDE the pure core and OUTSIDE `withMemoize` (which refuses to wrap it).
 */
export function withSession<R = ExecServices>(inner: Executor<R>): Executor<R & SessionSeams>;
export function withSession<R = ExecServices, P extends Partial<SessionSeams> = {}>(config?: P): ExecutorWrapper<R, R & Omit<SessionSeams, keyof P>>;
export function withSession<R = ExecServices, P extends Partial<SessionSeams> = {}>(config: P, inner: Executor<R>): Executor<R & Omit<SessionSeams, keyof P>>;
export function withSession<R = ExecServices>(
  configOrInner?: Partial<SessionSeams> | Executor<R>,
  maybeInner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  // `sessions` store at construction → no added requirement; omitted → the composed `start` must supply it.
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<SessionSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R> | undefined;
  const wrap = ((innerExec: Executor): Executor => ({
    kind: innerExec.kind,
    capabilities: { ...innerExec.capabilities, sessionResume: true },
    start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
      // Store from construction (standalone) OR the run-scoped `ctx.sessions` (e.g. a workflow run).
      const sessions = config?.sessions ?? ctx.sessions;
      const def = spec.definition as LlmCallDefinition;
      if (def.providerSessionId !== undefined) {
        return finishedHandle(
          permanentOutcome(
            "providerSessionId is set, but provider-side session resume is not supported yet (no executor threads a provider handle) — use a logical sessionId with a SessionStore",
          ),
        );
      }
      if (def.sessionId === undefined) return innerExec.start(spec, ctx);
      if (sessions === undefined) {
        return finishedHandle(
          permanentOutcome(
            `the declaration carries sessionId "${def.sessionId}" but no SessionStore is available — provide it via withSession({ sessions }) or ctx.sessions`,
          ),
        );
      }
      const key = def.sessionId;
      return wrapHandle(async (ctl) => {
        const priorState = await sessions.get(key);
        const prior = (priorState?.messages as ModelMessage[] | undefined) ?? [];
        const current = promptAsMessages(def);
        // Rewrite the call to carry the full history as `messages` (system + config knobs preserved).
        const sentDef: LlmCallDefinition = {
          ...def,
          sessionId: undefined,
          providerSessionId: undefined,
          prompt: undefined,
          messages: [...prior, ...current],
        };
        if (ctl.canceled()) return canceledOutcome("canceled before the call started");
        const result = await ctl.started(innerExec.start({ ...spec, definition: sentDef }, ctx)).outcome;
        if (!result.error) {
          await sessions.put(key, { ...priorState, messages: [...prior, ...current, ...foldOutcomeToAssistant(result)] });
        }
        return { ...result, session: { id: key } };
      });
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/** Options for {@link withBudget}. All optional — with no `meter` (here or on `ctx.meter`) the wrapper is a
 *  pure passthrough (an absent service is a no-op, like the rest of the stack). */
export interface BudgetOptions {
  /** The metered wallet. Defaults to `ctx.meter`; supplied here it drops the ctx dependency. */
  meter?: BudgetMeter;
  /** Runtime-tunable output-token headroom for the pre-call reserve estimate (default 2×; settle usually
   *  corrects DOWNWARD and an over-reserve lives only for the call's duration). */
  headroomMultiplier?: number;
  /** Per-model observed output-token stats (RUN-scoped, mutable): read to price the reserve, and folded on
   *  settle so later reserves in the same run are better estimated. */
  stats?: Map<string, OutputTokenStats>;
  /** Cost-model override (test / consumer seam). Defaults to catalog pricing (`ModelInfo.instance`). */
  pricing?: {
    estimateCostUsd(modelId: string, inputTokens: number, outputTokens: number): number;
    affordableOutputTokens(modelId: string, inputTokens: number, availableUsd: number): number;
  };
}

/**
 * Per-call budget RESERVATION (the reserve→debit wallet lifecycle): before the call, hold its ESTIMATED
 * cost against the injected {@link BudgetMeter} (`ctx.meter` or `config.meter`); after it returns, settle
 * the reserve to the ACTUAL cost. When the estimate doesn't fit the balance, FLIP the relationship —
 * compute the AFFORDABLE output ceiling from the remaining headroom and rewrite the call's
 * `maxOutputTokens` to it (the same definition-rewrite pattern {@link withDeadline} uses for `timeoutMs`),
 * so the reserve becomes provider-ENFORCED instead of a guess; still short ⇒ an `out-of-credits` outcome,
 * no call made. A failed call still settles (its cost, usually $0, is real spend and the hold must not
 * linger). With no meter available the wrapper runs the inner call untouched.
 *
 * The wrapper owns only the generic lifecycle; the concrete metered wallet (ledger / Stripe / credits) is
 * the consumer's `BudgetMeter` implementation, and cross-call accounting (reuse billing, round-boundary
 * reconciliation) stays in the consumer — this is one call's reserve and settle.
 */
export function withBudget<R = ExecServices>(config?: BudgetOptions): ExecutorWrapper<R, R>;
export function withBudget<R = ExecServices>(config: BudgetOptions, inner: Executor<R>): Executor<R>;
export function withBudget<R = ExecServices>(inner: Executor<R>): Executor<R>;
export function withBudget<R = ExecServices>(
  configOrInner?: BudgetOptions | Executor<R>,
  maybeInner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as BudgetOptions | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R> | undefined;
  // Default pricing = the catalog. Un-priced models estimate $0 (they cost the platform nothing; a
  // meter's balance>0 admission floor still applies). `affordableOutputTokens` is catalog math too.
  const estCost = config?.pricing?.estimateCostUsd ?? ((m: string, i: number, o: number) => ModelInfo.instance.computeCostUsd(m, i, o) ?? 0);
  const affordOutput = config?.pricing?.affordableOutputTokens ?? ((m: string, i: number, avail: number) => ModelInfo.instance.affordableOutputTokens(m, i, avail));
  const headroom = config?.headroomMultiplier ?? DEFAULT_HOLD_OUTPUT_MULTIPLIER;
  const wrap = ((innerExec: Executor): Executor => ({
    kind: innerExec.kind,
    capabilities: innerExec.capabilities,
    start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
      const meter = config?.meter ?? ctx.meter;
      const def = spec.definition as LlmCallDefinition | undefined;
      // Unmetered, or a definition with no model to price → run untouched.
      if (!meter || !def || typeof def.model !== "string") return innerExec.start(spec, ctx);
      const model = def.model;
      const stats = config?.stats;
      return wrapHandle(async (ctl) => {
        const inputTokens = estimateInputTokens(promptText(def));
        const estOut = estimateOutputTokens(model, inputTokens, def.maxOutputTokens, stats, headroom);
        let hold = await meter.reserve(estCost(model, inputTokens, estOut));
        let sentSpec = spec;
        if (!hold) {
          // Tight wallet: clamp the output ceiling to what the remaining balance buys, then retry the
          // reserve ONCE with that cap made real — a truncated answer is preferable to no answer, and the
          // wallet can no longer be overshot.
          const afford = affordOutput(model, inputTokens, await meter.availableCostUsd());
          const clamped = Math.min(afford, def.maxOutputTokens ?? Number.POSITIVE_INFINITY);
          if (clamped >= MIN_USEFUL_OUTPUT_TOKENS && Number.isFinite(clamped)) {
            hold = await meter.reserve(estCost(model, inputTokens, clamped));
            if (hold) sentSpec = { ...spec, definition: { ...def, maxOutputTokens: clamped } };
          }
        }
        if (!hold) {
          return {
            rawText: "",
            finishReason: "error",
            metrics: { durationMs: 0 },
            error: { classification: "out-of-credits", reason: "the wallet cannot cover this call's estimated cost" },
          };
        }
        if (ctl.canceled()) {
          await hold.settle(0);
          return canceledOutcome("canceled before the call started");
        }
        let result: Outcome;
        try {
          result = await ctl.started(innerExec.start(sentSpec, ctx)).outcome;
        } catch (err) {
          await hold.settle(0).catch(() => undefined); // no result → no real spend; free the reserve
          throw err;
        }
        // reserve → debit: correct the hold to the ACTUAL cost (a failed call still settles), and feed the
        // observed output tokens back so the next reserve in this run is better priced.
        await hold.settle(result.metrics.cost ?? 0);
        if (stats) noteOutputTokens(stats, model, result.metrics.outputTokens);
        return hold.ledgerId ? { ...result, metrics: { ...result.metrics, ledgerId: hold.ledgerId } } : result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}

/**
 * Memoization: key the call by its §3.4 memo key (content hash of kind + definitionHash + inputs
 * [+ workspaceTreeHash]); on a hit return the cached outcome without executing; on a miss execute and cache
 * the result — ONLY on success (failures are never cached). This is the "memoized-llm-call" as a wrapper,
 * not a separate unit kind. Placed OUTERMOST so it caches the final (post-repair) result — but it REFUSES
 * (throws at composition time) to wrap a session layer: session state is not in the memo key, so a hit
 * would replay a stale answer and silently skip the transcript update. Compose `withSession` OUTSIDE
 * `withMemoize` instead — sound, because `withSession` recomputes the sent definition's hash from the full
 * transcript, so the memo key inside sees the real content identity.
 */
export function withMemoize<R = ExecServices>(config: { cache: MemoCache } & MemoizeOptions): ExecutorWrapper<R, R>;
export function withMemoize<R = ExecServices>(config: { cache: MemoCache } & MemoizeOptions, inner: Executor<R>): Executor<R>;
export function withMemoize<R = ExecServices>(
  config: { cache: MemoCache } & MemoizeOptions,
  inner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  const { cache, identify } = config;
  const wrap = ((innerExec: Executor): Executor => {
    if (innerExec.capabilities.sessionResume) {
      throw new Error(
        "withMemoize must not wrap a session layer: session state is not part of the memo key, so a hit would replay a stale answer and skip the transcript update — compose withSession OUTSIDE withMemoize",
      );
    }
    return {
      kind: innerExec.kind,
      capabilities: innerExec.capabilities,
      start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
        // Identity is a memo concern: derive the definition's content hash here (a unit-supplied
        // `identify` — e.g. the hw snapshot hash — else canonical-hash the definition), so the
        // generic spec carries no `definitionHash` the caller must compute and keep in sync.
        const definitionHash = identify ? identify(spec) : hashCanonical(spec.definition);
        const key = memoKey({
          kind: innerExec.kind,
          definitionHash,
          inputs: spec.inputs,
          ...(spec.workspace?.treeHash !== undefined ? { workspaceTreeHash: spec.workspace.treeHash } : {}),
        });
        return wrapHandle(async (ctl) => {
          const hit = await cache.get(key);
          if (hit) return hit;
          if (ctl.canceled()) return canceledOutcome("canceled before the call started");
          const result = await ctl.started(innerExec.start(spec, ctx)).outcome;
          if (!result.error) await cache.set(key, result);
          return result;
        });
      },
    };
  }) as unknown as ExecutorWrapper<R, R>;
  return curryOrApply(wrap, inner);
}
