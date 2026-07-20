import type { ErrorClass } from "./classification";
import type { BlobStore, SessionStore } from "./stores";

/**
 * The ai-exec execution contract (DESIGN §3): one uniform way to execute an "AI unit" —
 * an LLM call, a hierarchical workflow, or (later) a process-based agent — with a
 * normalized observable event stream, never-throws outcomes, and shared metric shapes.
 *
 * Extracted/generalized from findmyprompt's `CallOutcome`/`CallMetrics`/`RunCtx`
 * (`src/engine/providers/generate.ts`, `src/engine/execution/context.ts`) and JaiRA's
 * `RunnerAdapter` design (JaiRA DESIGN.md §8).
 */

// --- Unit kinds --------------------------------------------------------------

export type UnitKind =
  | "llm-call" // one structured LLM call (@declarative-ai/llm)
  | "hierarchical-workflow" // a hierarchical state-machine run (@declarative-ai/hw)
  | "agent-sdk" // process units (@declarative-ai/agents, deferred implementations)
  | "claude-cli"
  | "generic-cli";

// --- Metrics -----------------------------------------------------------------

/**
 * Token counts pulled off a provider usage object. `inputTokens`/`outputTokens` are the
 * cache-INCLUSIVE totals; the optional breakdown is what makes cost billing-accurate
 * under prompt caching. Absent fields = the provider didn't report them.
 */
export interface TokenCounts {
  /** Total input tokens, INCLUDING cache reads + writes (matches the provider's billed input). */
  inputTokens?: number;
  /** Total output tokens, including reasoning tokens. */
  outputTokens?: number;
  /** Uncached (fresh) input tokens — billed at the base input rate. */
  noCacheTokens?: number;
  /** Cached input tokens read on a cache hit — billed at the discounted read rate (~0.1x). */
  cacheReadTokens?: number;
  /** Input tokens written into the prompt cache — billed at the write rate (~1.25x). */
  cacheWriteTokens?: number;
  /** 1-hour-TTL subset of `cacheWriteTokens` — billed at the 2x tier. */
  cacheWrite1hTokens?: number;
  /** Reasoning/thinking output tokens (subset of `outputTokens`), for diagnostics. */
  reasoningTokens?: number;
  /** Provider-reported grand total, when present. */
  totalTokens?: number;
}

export interface ExecMetrics extends TokenCounts {
  /** USD price: the provider's reported charge when available, else a price-table estimate. */
  cost?: number;
  /** Provenance of `cost`: `"provider"` = the real charge (authoritative); `"table"` = estimate. */
  costSource?: "provider" | "table";
  /** Provider's exact usage object (ground truth) — kept so `cost` is always recomputable. */
  rawUsage?: unknown;
  startMs?: number;
  durationMs: number;
  /**
   * Composite units (hierarchical workflows, agents): aggregate of child executions,
   * so budget gates and cost folds see through nesting without needing child records.
   * `cost` on a composite outcome already INCLUDES `childCost`.
   */
  childCalls?: number;
  childCost?: number;
  /** Audit link to the wallet ledger row a budget reservation opened for this call — stamped by the
   *  `withBudget` wrapper from `BudgetReservation.ledgerId`, for GR→ledger reconciliation. */
  ledgerId?: string;
}

// --- Outcome -----------------------------------------------------------------

export interface ExecFailure {
  classification: ErrorClass;
  /** The real underlying cause, human-readable. */
  reason: string;
  /** Server-advised wait before the next attempt (`retry-after`), ms. */
  retryAfterMs?: number;
  /** True iff this was a 429 rate-limit — feeds AIMD's multiplicative decrease. */
  rateLimited?: boolean;
}

/** One reasoning/tool segment of a run's thinking trace, positioned against the output. */
export interface ReasoningSegment {
  type: "reasoning" | "tool-call";
  text: string;
  /** Length of the accumulated OUTPUT text when this segment began. */
  textOffset: number;
  /** Tool name, for `type:"tool-call"` segments. */
  toolName?: string;
  /** Native provider metadata (e.g. an Anthropic thinking-block signature), preserved intact. */
  providerMetadata?: Record<string, unknown>;
}

export interface ProducedArtifact {
  name: string;
  /** Path relative to the workspace root (workspace units) — absent for pure units. */
  path?: string;
  /** Inline content for pure units that produce artifacts without a filesystem. */
  content?: string;
  format?: string;
  contentHash?: string;
}

/** A tool invocation the model requested. For a single-turn (unexecuted) tool op these ARE the primary
 *  output the caller acts on; in an executed loop they are the intermediate calls. */
export interface ToolCall {
  toolCallId?: string;
  toolName: string;
  /** The parsed tool input the model produced (validated against the tool's input schema). */
  input: unknown;
}

/** The result of an EXECUTED tool call, fed back to the model during a tool loop. */
export interface ToolResult {
  toolCallId?: string;
  toolName?: string;
  output: unknown;
}

/**
 * The result of one unit execution. NEVER thrown for a unit failure — always returned,
 * best-effort populated: on failure, `value`/`rawText`/`artifacts`/`thinking` carry
 * whatever was produced, so a failure is diagnosable, not empty.
 */
export interface Outcome {
  /** Parsed structured value when the unit produced usable output (kept even when a
   *  later validation fails); undefined only when nothing parseable was produced. */
  value?: unknown;
  /** Raw output text — the partial on failure. Present for text-producing units. */
  rawText?: string;
  artifacts?: ProducedArtifact[];
  thinking?: ReasoningSegment[];
  /** Tool calls the model requested (the primary output for a single-turn tool op; intermediate calls
   *  for an executed loop). */
  toolCalls?: ToolCall[];
  /** Results of tool calls executed during the run. */
  toolResults?: ToolResult[];
  finishReason?: string;
  metrics: ExecMetrics;
  /** Continuation token for resume, when the executor supports `sessionResume`. */
  session?: { id?: string };
  /** Present iff the execution failed; other fields are then best-effort partials. */
  error?: ExecFailure;
}

// --- Spec --------------------------------------------------------------------

/** Workspace binding for units that read/mutate files. */
export interface WorkspaceRef {
  rootDir: string;
  /** Snapshot identity (e.g. a git tree sha). REQUIRED for memoizing a
   *  `mutatesWorkspace` unit (DESIGN §3.4). */
  treeHash?: string;
}

export interface ArtifactTarget {
  name: string;
  /** Workspace-relative target path the unit is instructed to write. */
  path: string;
  format?: string;
}

export interface ExecLimits {
  timeoutMs?: number;
  maxCostUsd?: number;
}

/**
 * The human-in-the-loop seam (DESIGN §3.3). Callers with a real user supply a UI-backed
 * port; search/batch callers supply a fixture-scripted port or one that rejects
 * (making interactive states a `permanent` failure). The port is caller-supplied and
 * never readable by code running inside the unit, which is what makes approval gates
 * user-controlled by construction.
 */
export interface InteractionPort {
  request(req: { stateId: string; component: string; inputs: unknown }): Promise<unknown>;
}

export interface ExecutionSpec {
  kind: UnitKind;
  /** Unit definition. For `llm-call`: the call params. For composite/process units:
   *  a content-addressed definition bundle. The definition's IDENTITY (its content hash) is a
   *  MEMOIZATION concern, not a generic spec field: `withMemoize` derives it from `definition`
   *  (default `hashCanonical(definition)`, or a unit-supplied `identify` — e.g. the hw snapshot
   *  hash), so no caller has to compute a hash it may never use or keep in sync with `definition`. */
  definition: unknown;
  /** Named input values, resolved by the caller. Schemas live in the definition. */
  inputs: Record<string, unknown>;
  workspace?: WorkspaceRef;
  /** Session/conversation continuity (provider session id, transcript ref). */
  session?: { id?: string; transcript?: unknown };
  /** JSON Schema the data payload must satisfy. Validation is executor-performed
   *  (with the opt-in repair loop) and always caller-observable via the outcome. */
  outputSchema?: Record<string, unknown>;
  /** Engine-assigned artifact target paths the unit is instructed to write. */
  artifactTargets?: ArtifactTarget[];
  limits?: ExecLimits;
  /** Compiled safety policy — opaque here; enforced per executor capability. */
  policy?: unknown;
  /** Required iff the executor is `interactive` and the definition contains
   *  interactive states. */
  interaction?: InteractionPort;
  abortSignal?: AbortSignal;
}

// --- Events ------------------------------------------------------------------

export type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: unknown } // transcript stream
  | { type: "child_outcome"; ref: { kind: UnitKind; label?: string }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: unknown } // process units
  | { type: "command_result"; decision: "allowed" | "blocked" | "approved" | "denied" }
  | { type: "interaction_request"; stateId: string; component: string; payload: unknown }
  | { type: "output_partial"; text: string };

// --- Executor ----------------------------------------------------------------

export interface ExecutorCapabilities {
  /** Native schema-constrained output support. */
  structuredOutput: boolean;
  sessionResume: boolean;
  streaming: boolean;
  /** May emit `interaction_request` events; needs `spec.interaction` for interactive definitions. */
  interactive: boolean;
  /** Requires `spec.workspace`; memo keys must include the workspace tree hash. */
  mutatesWorkspace: boolean;
  policyEnforcement: "callback" | "config" | "none";
  /** Sound to memoize under DESIGN §3.4 keying. */
  memoizable: boolean;
  runtime: "edge-safe" | "node";
}

export interface ExecHandle {
  events: AsyncIterable<ExecEvent>;
  /** Resolves when done; NEVER rejects for a unit failure (see `Outcome.error`). */
  outcome: Promise<Outcome>;
  cancel(): Promise<void>;
}

/**
 * An executable unit. Generic in `R` — the environment it still REQUIRES at `start`. The bare core and
 * registry-dispatched executors use the default `R = ExecServices` (every seam optional). Composition
 * NARROWS `R`: a wrapper that reads a ctx seam (e.g. `withDeadline` → `deadline`/`stepStartMs`) ADDS it to
 * `R`, so a stack's `start` demands exactly the fields its wrappers consume — a missing one is a compile
 * error (see {@link compose}), not just the bare core's runtime refusal.
 */
export interface Executor<R = ExecServices> {
  readonly kind: UnitKind;
  readonly capabilities: ExecutorCapabilities;
  start(spec: ExecutionSpec, ctx: R): ExecHandle;
}

export interface ExecutorRegistry {
  get(kind: UnitKind): Executor | undefined;
}

/** A plain-map registry — sufficient for both consumers. */
export class MapExecutorRegistry implements ExecutorRegistry {
  private readonly map = new Map<UnitKind, Executor>();
  register(executor: Executor): this {
    this.map.set(executor.kind, executor);
    return this;
  }
  get(kind: UnitKind): Executor | undefined {
    return this.map.get(kind);
  }
}

// --- Composition -------------------------------------------------------------

/**
 * A composable behavior wrapped around an executor — memoize / repair / rate-limit / deadline / budget /
 * session. It maps an executor requiring `RIn` to one requiring `ROut`: a construction-injected wrapper
 * leaves the requirement unchanged (`ExecutorWrapper<R, R>`); a ctx-reading one ADDS its seam
 * (`withDeadline(): ExecutorWrapper<R, R & { deadline; stepStartMs }>`). The stacking ORDER encodes
 * semantics — see the two forms below.
 */
export type ExecutorWrapper<RIn = ExecServices, ROut = RIn> = (inner: Executor<RIn>) => Executor<ROut>;

/**
 * There are TWO ways to stack wrappers; pick whichever reads clearer. Both nest identically — each wrapper
 * becomes an OUTER layer around the previous — and the ORDER is meaningful: `memoize` outermost caches the
 * final (post-repair) result; per-attempt concerns (`rateLimit`/`deadline`) sit inner so they apply to each
 * attempt; `memoize` must not sit outside a `session` layer (it throws if it does).
 *
 * 1. Function application — `withMemoize(c)(withDeadline()(core))` — reads INNER→OUTER (core first).
 * 2. Inside-out builder — {@link compose} — `compose(core).with(withDeadline()).with(withMemoize(c))` —
 *    reads core-first then each added layer, and TYPE-ACCUMULATES the requirements each wrapper adds, so the
 *    final `.start` demands exactly them.
 *
 * {@link composeExecutors} is the loose variadic convenience (flat list, no requirement tracking) — handy,
 * but the two forms above are clearer about ordering and are compile-time-checked.
 */
export function composeExecutors(core: Executor, ...wrappers: ExecutorWrapper[]): Executor {
  return wrappers.reduce<Executor>((inner, wrap) => wrap(inner), core);
}

/**
 * The inside-out builder (form 2): `compose(core).with(a).with(b)` = `b(a(core))`, read core-first with each
 * `.with` adding an OUTER layer. Unlike {@link composeExecutors} it tracks requirements in the type: each
 * wrapper that adds a ctx seam narrows `R`, so the final {@link ComposableExecutor.start} requires exactly
 * the union of what the stack consumes — forgetting one (e.g. `stepStartMs` after `withDeadline`) is a
 * compile error, and it IS an {@link Executor} so it drops into a registry unchanged.
 */
export class ComposableExecutor<R = ExecServices> implements Executor<R> {
  constructor(private readonly inner: Executor<R>) {}
  get kind(): UnitKind {
    return this.inner.kind;
  }
  get capabilities(): ExecutorCapabilities {
    return this.inner.capabilities;
  }
  with<ROut>(wrap: ExecutorWrapper<R, ROut>): ComposableExecutor<ROut> {
    return new ComposableExecutor(wrap(this.inner));
  }
  start(spec: ExecutionSpec, ctx: R): ExecHandle {
    return this.inner.start(spec, ctx);
  }
}

/** Start the inside-out builder around a core executor — see {@link ComposableExecutor}. */
export function compose<R = ExecServices>(core: Executor<R>): ComposableExecutor<R> {
  return new ComposableExecutor(core);
}

/**
 * A memoization cache keyed by the §3.4 memo key (content hash of kind + the definition's content
 * hash + inputs [+ workspaceTreeHash]). The definition hash is derived BY the `memoize` wrapper (it
 * owns the memo key), not carried on the spec. Only SUCCESSFUL outcomes should be cached. Injected
 * into a `memoize` wrapper; both methods may be sync or async so an in-memory map or a durable store fit.
 */
export interface MemoCache {
  get(key: string): Promise<Outcome | undefined> | Outcome | undefined;
  set(key: string, outcome: Outcome): Promise<void> | void;
}

/** Options for the `memoize` wrapper. */
export interface MemoizeOptions {
  /**
   * Derive the definition's content hash (the memo key's identity component) from the spec. Defaults
   * to `hashCanonical(spec.definition)`. A unit with a cheaper/canonical identity supplies its own —
   * e.g. `hierarchical-workflow` passes its snapshot hash so `memoize` never brute-force-canonicalizes
   * an opaque bundle. This is the seam that lets `definitionHash` stay OUT of the generic `ExecutionSpec`.
   */
  identify?(spec: ExecutionSpec): string;
}

// --- Injected services -------------------------------------------------------

/** One call's estimated token footprint, input/output split. */
export interface CallTokenEstimate {
  inputTokens: number;
  outputTokens: number;
}

export interface CallEstimate extends CallTokenEstimate {
  modelId?: string;
}

export interface RateLimiter {
  /** Admit one call: take a concurrency slot, wait for rate headroom, run it. */
  schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T>;
  /** Feed the call's outcome back (a 429 halves concurrency; success grows it). */
  reportOutcome(outcome: { rateLimited?: boolean; modelId?: string }): void;
}

/**
 * Per-call budget RESERVATION — the engine-side interface to a metered wallet
 * (findmyprompt `WalletMeter`, `src/engine/execution/context.ts`). One call's
 * lifecycle: `reserve(−estimate)` BEFORE the call (atomic conditional decrement;
 * refused when the balance can't cover it), then `settle(−actual)` when it returns.
 */
export interface BudgetReservation {
  /** Backing ledger row id, for audit references. */
  ledgerId?: string;
  /** Correct the reserved estimate to the call's ACTUAL cost. Idempotent. */
  settle(actualCostUsd: number): Promise<void>;
}

export interface BudgetMeter {
  /** `null` ⇒ the balance can't cover the estimate — clamp output or refuse the call. */
  reserve(estCostUsd: number): Promise<BudgetReservation | null>;
  /** Current USD headroom (balance minus outstanding reserves). */
  availableCostUsd(): Promise<number>;
}

export interface Clock {
  now(): number;
}

export interface DeadlineConfig {
  maxDurationMs: number;
  safetyMarginMs?: number;
  floorMs?: number;
}

/**
 * The injected seam bundle an executor runs with — the shared-library descendant of
 * findmyprompt's `RunCtx`, reduced to what execution (not search) needs. All fields
 * optional: an absent service is a no-op (unthrottled, unmetered, unvalidated).
 */
export interface ExecServices {
  meter?: BudgetMeter;
  /** Boundary schema validation (an `@declarative-ai/services` SchemaValidator or compatible). */
  validator?: OutputValidator;
  clock?: Clock;
  deadline?: DeadlineConfig;
  /** Step-start origin for deadline arithmetic (ms epoch). */
  stepStartMs?: number;
  /** Composite units execute children through this. */
  registry?: ExecutorRegistry;
  /** Optional provider router for llm-backed executors (typed in @declarative-ai/llm). */
  providers?: unknown;
  /** Mutable, logical-id-keyed session store — e.g. a workflow run injects a RUN-SCOPED one so states
   *  sharing a `sessionId` continue the same conversation. Absent ⇒ sessions unavailable. */
  sessions?: SessionStore;
  /** Content-addressed blob store for file/media I/O (Phase 5). Absent ⇒ blob refs can't be resolved. */
  blobs?: BlobStore;
}

/** Minimal validation seam (implemented by `@declarative-ai/services` SchemaValidator). */
export interface OutputValidator {
  validateValue(schema: Record<string, unknown>, value: unknown): { ok: boolean; errors?: string };
}
