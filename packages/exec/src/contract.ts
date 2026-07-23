/**
 * The ONE execution seam (DESIGN §3.1/§3.2). An `Executor` takes an `Operation` and returns an
 * `ExecHandle`; that is the whole contract.
 *
 * Before this, there were three seams — `Executor.start(spec, ctx)`, `PromptOpRunner.run(op, ctx, env)`,
 * and a bare `AsyncFunctionImpl(inputs, ctx)`. The second was an op→spec lowering wrapped around the
 * first; the third had no error channel at all, so the engine GUESSED a classification from `err.name`
 * and every non-`AbortError` became `permanent` — a 429 inside a function impl was never retried, with
 * the retry machinery sitting right there. All three collapse here.
 *
 * Dispatch is by op kind: `"prompt"` → the prompt executor, `"function"` → a registry lookup by
 * `functionRef` (see {@link OperationExecutor}). Wrapper composition therefore applies UNIFORMLY to
 * prompt and function ops alike.
 *
 * **What this package does NOT know.** An execution returns the value of the op's output PARAMETER —
 * a `ResolvedValue`, which is ops vocabulary, because executing ops is this package's job. It never
 * learns what a token is, what a model produced, or what anything costs: `thinking`,
 * `toolCalls`, and `finishReason` used to ride on the result and are now `llm`'s `LlmOutput`, which
 * stops at `promptop`. Money and tokens are quarantined in `budget.ts` and `ratelimit.ts`, which
 * nothing else here imports.
 */
import type {
  Capabilities,
  Failure,
  FunctionInputs,
  FunctionRegistry,
  InlineFamily,
  JsonSchema,
  JsonValue,
  MetricsAlgebra,
  Operation,
  OutputValidator,
  ResolvedValue,
  ResultWithMetrics,
} from "@declarative-ai/ops";

// The op vocabulary is what flows through this contract — re-exported so a consumer that speaks
// execution imports one name set.
export * from "@declarative-ai/ops";
export * from "./budget";
export * from "./ratelimit";

// --- Metrics ------------------------------------------------------------------

/**
 * What EXECUTION measures: how long the work took and how many child LLM calls it fanned out to. No
 * money and no tokens — those belong to whatever ran (see `budget.ts` for why).
 *
 * This is the CONSTRAINT the machinery here is written against, not a record anyone must use verbatim.
 * A producer's own flat record satisfies it structurally and adds whatever else it measured:
 * `LlmMetrics` adds tokens and cost, an agent adapter adds its billed spend.
 */
export interface ExecMetrics {
  /** Wall-clock duration of the execution, ms. */
  durationMs: number;
  /** When it started (ms epoch). */
  startMs?: number;
  /** LLM calls made by children, rolled up by a composite. A prompt op IS one such call; a non-LLM
   *  function (a pure helper, a sub-workflow that made none) contributes zero. */
  childLlmCalls?: number;
}

/** Merge two executions' timing/counts: duration sums, the start is the FIRST observation, child LLM
 *  calls sum. A richer `M` builds its algebra on top of this rather than restating it. */
export function mergeExecMetrics<M extends ExecMetrics>(a: M, b: M): M {
  const startMs = a.startMs ?? b.startMs;
  return {
    ...a,
    ...b,
    durationMs: a.durationMs + b.durationMs,
    ...(startMs !== undefined ? { startMs } : {}),
    ...(a.childLlmCalls !== undefined || b.childLlmCalls !== undefined ? { childLlmCalls: (a.childLlmCalls ?? 0) + (b.childLlmCalls ?? 0) } : {}),
  };
}

/** The algebra for a bare {@link ExecMetrics} — the default an executor uses when its `M` adds nothing. */
export const EXEC_METRICS_ALGEBRA: MetricsAlgebra<ExecMetrics> = { merge: mergeExecMetrics };

// --- Result -------------------------------------------------------------------

/**
 * What execution RETURNS. NEVER thrown for a unit failure — always returned, and the failure branch may
 * still carry the partial value, so a failure is diagnosable rather than empty.
 *
 * `O` is the value of the op's output parameter; `M` is whatever the producer measured. This is `json`'s
 * `ResultWithMetrics` with `E` pinned to the shared classified {@link Failure} — the layer customizing
 * the base, which is all a layer-prefixed name should ever mean.
 */
export type ExecResult<O, M extends ExecMetrics = ExecMetrics> = ResultWithMetrics<O, Failure, M>;

// --- Events -------------------------------------------------------------------

export type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: JsonValue } // transcript stream
  | { type: "child_result"; ref: { label?: string }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: JsonValue } // process units
  | { type: "command_result"; decision: "allowed" | "blocked" | "approved" | "denied" }
  | { type: "output_partial"; text: string };

// --- Executor -----------------------------------------------------------------

export interface ExecHandle<O, M extends ExecMetrics = ExecMetrics> {
  /**
   * The operation's event stream — **SINGLE-CONSUMER**. Events are DELIVERED (each to exactly one
   * iterator), not broadcast, so a second `for await` over the same handle would steal events from the
   * first; attaching twice throws rather than silently splitting or hanging the stream. A caller that
   * needs several observers drains once and fans out itself.
   */
  events: AsyncIterable<ExecEvent>;
  /** Resolves when done; NEVER rejects for a unit failure (see `ExecResult.error`). */
  result: Promise<ExecResult<O, M>>;
  /**
   * Stop the operation. Settles `result` — with a `canceled` failure unless the work had already
   * finished — and returns once it HAS settled, bounded by this handle rather than by whatever the
   * operation is parked on. Equivalent to aborting `ctx.abortSignal`: both are the same event.
   */
  cancel(): Promise<void>;
}

/**
 * An executable. Generic in `R` — the environment it still REQUIRES at `start` — in `M`, what it
 * measures, and in `Op`, the operation PAYLOAD it accepts. Composition NARROWS `R`: a wrapper that
 * reads a ctx seam (e.g. `withDeadline` → `deadline`/`stepStartMs`) ADDS it to `R`, so a stack's
 * `start` demands exactly the fields its wrappers consume — a missing one is a compile error (see
 * {@link compose}).
 *
 * `Op` defaults to the RESOLVED inline op — the only thing a leaf can run — and every wrapper that
 * reads op CONTENT (a prompt to price, a user text to repair) is pinned there. What generalizes is the
 * layers that need only the op's IDENTITY: `withMemoize` keys any serializable op, and
 * `withHydration` is the family-transition wrapper that turns a stack over inline ops into a stack
 * over some other family's ops (e.g. content-id ops whose leaves are cheap to hash and expensive to
 * load — hydration then happens only below the memo, on a miss).
 *
 * `metrics` is how the executor's measurements COMBINE — across retry attempts, or a child into a
 * parent. It is registered by the producer because only the producer knows which of its fields sum,
 * which take the latest, and which are the first observation; every consumer of a merge calls it
 * without learning what the fields mean.
 */
export interface Executor<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>> {
  readonly capabilities: Capabilities;
  readonly metrics: MetricsAlgebra<M>;
  /**
   * The capabilities of the entry THIS op dispatches to, when the executor is a dispatcher and they
   * differ per op. Absent ⇒ `capabilities` is total for every op (a leaf executor), which is what a
   * wrapper falls back to.
   *
   * A dispatcher's static `capabilities` is one record for a whole REGISTRY: DESIGN §3.2 makes each entry's
   * record required and total, and without this seam a gate that reads `executor.capabilities` consults
   * a record belonging to no particular entry — so `withMemoize` memoized an entry declaring
   * `memoizable: false`, cached a `mutatesWorkspace` entry under an "any workspace" key, and let one
   * session-capable variant make every op in the registry un-memoizable. That per-variant record is the
   * registry redesign's payoff; this is how a wrapper claims it.
   *
   * A wrapper MUST forward it (`capabilitiesFor: (o) => inner.capabilitiesFor!(o)`), or the stack
   * silently degrades to the static record.
   */
  capabilitiesFor?(op: Op): Capabilities;
  start(op: Op, ctx: R): ExecHandle<ResolvedValue, M>;
}

// --- The named facets a workflow's operations reference ------------------------

/** A named prompt template a prompt op can reference (a skill = name → prompt, `{{...}}` parameters). */
export type SkillTemplate = string;

/**
 * A tool a runtime (agent) may invoke mid-loop: an impl PLUS the call-metadata a model needs to decide
 * to call it — a `description` and an `inputSchema`. The same impl can be surfaced as a graph
 * `function` op or an agent tool.
 *
 * A tool's `run` returns its value and MAY throw: a tool failure travels back to the MODEL as a result
 * it reads and reacts to, so it is not the classified-failure channel that `FunctionResult` is for
 * (see DESIGN §5.1, "Functions and tools").
 */
export interface Tool<I = FunctionInputs, O = JsonValue> {
  /** What the tool does — shown to the model. */
  readonly description?: string;
  /** JSON Schema for the input the model must produce for a call. */
  readonly inputSchema: JsonSchema<I>;
  /** Does not mutate the workspace/world — what the `read-only`/`plan` profiles gate on. */
  readonly readOnly: boolean;
  run(input: I, ctx: ExecServices): O | Promise<O>;
}

/**
 * A per-runtime redirect to a DELEGATED agent's built-in tool of the given native name
 * (DESIGN §5.1, "Tool renames are just overlay bindings"). Unlike a {@link Tool} we cannot execute it ourselves — it names
 * the black-box agent's own tool, handed to the adapter as an alias/allowlist entry.
 */
export interface NativeToolRef {
  readonly native: string;
}

/**
 * The named things an operation can reference: registered `functions` (host code including interactive
 * UI, sub-workflows, AND delegated runtime adapters — one map of discriminated entries), `skills`
 * (named prompt templates), and agent `tools`.
 *
 * All three are plain `Map`s. They were a `Registry<T>` interface plus a `MapRegistry<T>` class whose
 * entire content was `get` and a `register` that did what `Map.set` does — including returning itself
 * for chaining. There was never a second implementation.
 *
 * There is no `prompt` facet: a `PromptOp` is dispatched to an `Executor` like everything else (DESIGN §3.1),
 * which is what removed the "the llm runtime is a facet, every other runtime is a registry entry"
 * asymmetry.
 */
export interface CapabilityRegistry<M extends ExecMetrics = ExecMetrics> {
  functions: FunctionRegistry<ExecServices, M>;
  skills: Map<string, SkillTemplate>;
  tools: Map<string, Tool>;
}

/** An empty {@link CapabilityRegistry} — three empty maps. */
export function newCapabilityRegistry<M extends ExecMetrics = ExecMetrics>(
  functions: FunctionRegistry<ExecServices, M> = new Map(),
): CapabilityRegistry<M> {
  return { functions, skills: new Map(), tools: new Map() };
}

// --- Injected services --------------------------------------------------------

export interface Clock {
  now(): number;
  /** The clock's OWN delay: resolves after `ms` of this clock's time, or early when `signal` aborts.
   *  Optional — absent means "use a real `setTimeout`". Injected alongside `now` so a virtual clock
   *  enforces time windows (e.g. a deadline in flight) in the same units it reports `now()` in, rather
   *  than mixing computed virtual-time budgets with wall-clock `setTimeout` enforcement. */
  wait?(ms: number, signal: AbortSignal): Promise<void>;
}

export interface DeadlineConfig {
  maxDurationMs: number;
  safetyMarginMs?: number;
  floorMs?: number;
}

// The validation seam is `json`'s minimal structural interface (`validateValue`), declared once so
// exec, llm, and hw all consume the SAME three lines and none of them learns about ajv (DESIGN §2).
export type { OutputValidator } from "@declarative-ai/ops";

/**
 * A working directory an operation's tools act within (DESIGN §5.1, "Sessions: the run-scoped resource bundle") — a
 * Session-owned resource: ops sharing a session share it; a fan-out may isolate each branch in its own.
 *
 * Two plain fields, no filesystem: `root` is what every consumer needs (hw threads it, a delegated
 * agent uses it as `cwd`), and `treeHash` is what MEMOIZATION needs (a side-effecting run is only
 * memoizable against a pinned snapshot). The fs-backed tools that actually read the directory live in
 * `@declarative-ai/tools`, which is what keeps `exec` free of `node:*`.
 */
export interface Workspace {
  /** Absolute path a workspace tool resolves its inputs against, and may not escape (SPEC §7.2). */
  root: string;
  /** Snapshot identity (e.g. a git tree sha). REQUIRED for memoizing a `mutatesWorkspace` op. */
  treeHash?: string;
}

/**
 * The injected seam bundle an executor runs with. All fields optional: an absent service is a no-op
 * (unthrottled, unmetered, unvalidated).
 *
 * This interface is AUGMENTABLE (DESIGN §3.2). Splitting packages does not by itself stop `exec` from NAMING
 * every optional capability, so each optional package declares its own seam by declaration merging:
 *
 * ```ts
 * declare module "@declarative-ai/exec" {
 *   interface ExecServices { policy?: ExecPolicy }
 * }
 * ```
 *
 * `exec` then does not know that permissions, model routing, or workspaces-with-filesystems exist. The
 * cost is that augmentation is GLOBAL — two packages cannot declare conflicting seams, and
 * go-to-definition lands in the owning package.
 */
export interface ExecServices {
  /** The metered wallet, when one is wired in. Declared by `budget.ts` and read ONLY by the layer whose
   *  job is money — `exec` itself never touches it. */
  meter?: import("./budget").BudgetMeter;
  /** Boundary schema validation. */
  validator?: OutputValidator;
  clock?: Clock;
  deadline?: DeadlineConfig;
  /** Step-start origin for deadline arithmetic (ms epoch). */
  stepStartMs?: number;
  /** Composite ops execute children through this. */
  executor?: Executor;
  /** Executable tools the current operation may call mid-loop, keyed by name. */
  tools?: Record<string, Tool>;
  /** Mutable, logical-id-keyed session store — e.g. a workflow run injects a RUN-SCOPED one so ops
   *  sharing a `sessionId` continue the same conversation. Absent ⇒ sessions unavailable. */
  sessions?: SessionStore;
  /** The workspace the current operation acts within — a Session-owned resource. */
  workspace?: Workspace;
  /** Per-call wall-clock budget (ms). Was `PromptOpEnvironment.timeoutMs`. */
  timeoutMs?: number;
  /** Per-call cost ceiling (USD). */
  maxCostUsd?: number;
  /** Cancellation for the operation in flight. */
  abortSignal?: AbortSignal;
}

// --- Sessions -----------------------------------------------------------------

/**
 * The state a session accumulates, keyed by a LOGICAL session id. Client-managed conversations store
 * the `messages` transcript; a provider-side (stateful) executor instead stores the opaque
 * `providerSessionId` handle it resumes. A logical id NEVER carries the provider handle in the
 * portable declaration — it lives here, mapped from the logical id.
 */
export interface SessionState<Msg = JsonValue> {
  /** Client-managed conversation transcript (prior turns). Generic in the message shape so each
   *  consumer pins it (promptop: the AI-SDK `ModelMessage`; hw: a `Turn`); the default is plain JSON,
   *  since a stored transcript is serializable by construction. */
  messages?: Msg[];
  /** Provider-assigned session handle to resume (for a stateful executor). */
  providerSessionId?: string;
}

/** A mutable, logical-id-keyed session store. Both methods may be sync or async. */
export interface SessionStore<Msg = JsonValue> {
  get(logicalId: string): SessionState<Msg> | undefined | Promise<SessionState<Msg> | undefined>;
  put(logicalId: string, state: SessionState<Msg>): void | Promise<void>;
}

/** A plain in-memory session store. */
export class MapSessionStore<Msg = JsonValue> implements SessionStore<Msg> {
  private readonly map = new Map<string, SessionState<Msg>>();
  get(logicalId: string): SessionState<Msg> | undefined {
    return this.map.get(logicalId);
  }
  put(logicalId: string, state: SessionState<Msg>): void {
    this.map.set(logicalId, state);
  }
}

// --- Composition --------------------------------------------------------------

/**
 * A composable behavior wrapped around an executor — memoize / retry / rate-limit / deadline / budget /
 * session. It maps an executor requiring `RIn` to one requiring `ROut`: a construction-injected wrapper
 * leaves the requirement unchanged (`ExecutorWrapper<R, R>`); a ctx-reading one ADDS its seam
 * (`withDeadline(): ExecutorWrapper<R, R & { deadline; stepStartMs }>`). The stacking ORDER encodes
 * semantics — see the two forms below.
 */
export type ExecutorWrapper<RIn = ExecServices, ROut = RIn, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>> = (
  inner: Executor<RIn, M, Op>,
) => Executor<ROut, M, Op>;

/**
 * Forward a dispatcher's per-op capability lookup through a wrapper — spread into the wrapper's executor
 * literal, e.g. `{ capabilities: inner.capabilities, ...forwardCapabilitiesFor(inner), start(...) }`.
 *
 * It forwards the ABSENCE too. "No per-op record" is itself information — it means the static record is
 * the whole truth for every op — and `withMemoize` reads it to decide whether its session refusal can
 * fire at composition time or has to wait for an op. A wrapper that always defined the method would
 * erase that distinction for every layer above it.
 */
export function forwardCapabilitiesFor<R, M extends ExecMetrics, Op = Operation<InlineFamily>>(
  inner: Executor<R, M, Op>,
): { capabilitiesFor?: (op: Op) => Capabilities } {
  const perOp = inner.capabilitiesFor;
  return perOp ? { capabilitiesFor: (op): Capabilities => perOp.call(inner, op) } : {};
}

/**
 * There are TWO ways to stack wrappers; pick whichever reads clearer. Both nest identically — each
 * wrapper becomes an OUTER layer around the previous — and the ORDER is meaningful: `memoize` outermost
 * caches the final (post-repair) result; per-attempt concerns (`rateLimit`/`deadline`) sit inner so
 * they apply to each attempt; `memoize` must not sit outside a `session` layer (it throws if it does).
 *
 * 1. Function application — `withMemoize(c)(withDeadline()(core))` — reads INNER→OUTER (core first).
 * 2. Inside-out builder — {@link compose} — reads core-first then each added layer, and
 *    TYPE-ACCUMULATES the requirements each wrapper adds, so the final `.start` demands exactly them.
 *
 * {@link composeExecutors} is the loose variadic convenience (flat list, no requirement tracking).
 */
export function composeExecutors<M extends ExecMetrics = ExecMetrics>(
  core: Executor<ExecServices, M>,
  ...wrappers: ExecutorWrapper<ExecServices, ExecServices, M>[]
): Executor<ExecServices, M> {
  return wrappers.reduce<Executor<ExecServices, M>>((inner, wrap) => wrap(inner), core);
}

/**
 * The inside-out builder (form 2): `compose(core).with(a).with(b)` = `b(a(core))`, read core-first with
 * each `.with` adding an OUTER layer. Unlike {@link composeExecutors} it tracks requirements in the
 * type: each wrapper that adds a ctx seam narrows `R`, so the final {@link ComposableExecutor.start}
 * requires exactly the union of what the stack consumes — forgetting one (e.g. `stepStartMs` after
 * `withDeadline`) is a compile error, and it IS an {@link Executor} so it drops into a registry
 * unchanged.
 */
export class ComposableExecutor<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>> implements Executor<R, M, Op> {
  /** Forwarded so the per-op capability lookup survives the builder — and forwarded CONDITIONALLY, so
   *  that "this executor has no per-op record" (which `withMemoize` reads as "the static record IS the
   *  whole truth") survives too. See {@link Executor.capabilitiesFor}. */
  readonly capabilitiesFor?: (op: Op) => Capabilities;
  constructor(private readonly inner: Executor<R, M, Op>) {
    const perOp = inner.capabilitiesFor;
    if (perOp) this.capabilitiesFor = (op): Capabilities => perOp.call(inner, op);
  }
  get capabilities(): Capabilities {
    return this.inner.capabilities;
  }
  get metrics(): MetricsAlgebra<M> {
    return this.inner.metrics;
  }
  /**
   * Add an OUTER layer. The parameter shape subsumes both an {@link ExecutorWrapper} (op type
   * unchanged) and a FAMILY-TRANSITION adapter like `withHydration`, which changes what the stack
   * above it accepts: `compose(leaf).with(withBudget(...)).with(withHydration(resolve)).with(withMemoize(...))`
   * prices inline ops below the transition and memoizes id ops above it.
   */
  with<ROut, OpOut = Op>(wrap: (inner: Executor<R, M, Op>) => Executor<ROut, M, OpOut>): ComposableExecutor<ROut, M, OpOut> {
    return new ComposableExecutor(wrap(this.inner));
  }
  start(op: Op, ctx: R): ExecHandle<ResolvedValue, M> {
    return this.inner.start(op, ctx);
  }
}

/** Start the inside-out builder around a core executor — see {@link ComposableExecutor}. */
export function compose<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>>(
  core: Executor<R, M, Op>,
): ComposableExecutor<R, M, Op> {
  return new ComposableExecutor(core);
}
