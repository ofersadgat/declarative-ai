import type { ErrorClass } from "./classification";
import type { Approver } from "./permissions";
import type { BlobStore, SessionStore, Workspace } from "./stores";

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
  abortSignal?: AbortSignal;
}

// --- Events ------------------------------------------------------------------

export type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: unknown } // transcript stream
  | { type: "child_outcome"; ref: { kind: UnitKind; label?: string }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: unknown } // process units
  | { type: "command_result"; decision: "allowed" | "blocked" | "approved" | "denied" }
  | { type: "output_partial"; text: string };

// --- Executor ----------------------------------------------------------------

export interface ExecutorCapabilities {
  /** Native schema-constrained output support. */
  structuredOutput: boolean;
  sessionResume: boolean;
  streaming: boolean;
  /** Supports interactive states — those whose `function` operation needs a human/renderer
   *  (a registered `HostFunction` with `capabilities.interactive`). */
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

// --- Capability registry (typed facets) --------------------------------------

/**
 * The named things a state operation can reference, split into TYPED facets so each facet's native
 * interface matches how it is invoked (no `definition: unknown` impedance-mismatch, no per-name binding
 * table). Behavior facets — `runtimes` (agent operations) and `functions` (host code, incl. interactive
 * UI). Content facet — `skills` (named prompt templates a runtime op's prompt reads from). Agent-tool
 * facet — `tools` (functions + call-metadata a runtime may invoke mid-loop, referenced by logical name;
 * see RUNTIMES-AND-PERMISSIONS.md §2–3). See HW-REDESIGN.md. In-bundle nested states are NOT registry
 * entries; cross-bundle sub-workflow composition is deferred (a black-box sub-workflow is itself a
 * `HostFunction`).
 */
export interface CapabilityRegistry {
  runtimes: Registry<Runtime>;
  functions: Registry<HostFunction>;
  skills: Registry<SkillTemplate>;
  tools: Registry<Tool>;
}

/** A name-keyed lookup facet. `register` returns `this` for chaining. */
export interface Registry<T> {
  get(name: string): T | undefined;
  register(name: string, value: T): this;
}

/**
 * The normalized operation a {@link Runtime} runs — hw renders this from a `runtime` state operation:
 * the prompt (inline template OR a named skill, with any conversation preamble prepended), the merged
 * config surface (binding defaults ← `configRef` preset ← inline), and the produced output schema. The
 * Runtime resolves `config` into its own definition shape (e.g. the llm runtime → an `LlmCallDefinition`).
 */
export interface RuntimeOp {
  prompt: string;
  system?: string;
  config: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  /** Executable tools the runtime (agent) may call mid-loop, keyed by logical name — the engine resolved
   *  the state's `runtime.tools` names through `registry.tools`. A composed runtime (llm) runs these
   *  in-loop; a delegated runtime declares them to its agent. Absent/empty ⇒ a plain single call. */
  tools?: Record<string, Tool>;
  timeoutMs?: number;
  abortSignal?: AbortSignal;
}

/**
 * A runtime adapter — executes an agent operation (llm, claude-code, codex, …). Keyed by name in
 * `registry.runtimes`; a state's `runtime.name` selects it. `llm` is simply the simplest runtime
 * (uniform interface, capabilities distinguish it from a file-editing agent).
 */
export interface Runtime {
  readonly capabilities: ExecutorCapabilities;
  run(op: RuntimeOp, ctx: ExecServices): ExecHandle;
}

/**
 * A registered host function — inputs → structured output, sync or async. A UI component is just an
 * interactive function (it renders and awaits human input); an async function's promise is treated like
 * any other async value in the dataflow (its outputs are PENDING until it resolves). Interactive
 * functions close over their renderer at host-registration time.
 */
export interface HostFunction {
  /** `interactive`: needs a human/renderer (was a `ui` op) — refused up-front by search callers.
   *  `pure`: deterministic and memoizable. `readOnly` (tools): does not mutate the workspace/world — the
   *  distinction the `read-only`/`plan` permission profiles gate on (RUNTIMES-AND-PERMISSIONS.md §4). */
  readonly capabilities?: { interactive?: boolean; pure?: boolean; readOnly?: boolean };
  run(inputs: Record<string, unknown>, ctx: ExecServices): unknown | Promise<unknown>;
}

/** A named prompt template a runtime op's prompt can reference (a skill = name → prompt, `{{...}}` slots). */
export type SkillTemplate = string;

/**
 * A tool a runtime (agent) may invoke mid-loop. A tool IS a {@link HostFunction} (`run(input, ctx)`) PLUS
 * the call-metadata a model needs to decide to call it — a `description` and an `inputSchema`. So
 * `Tool ⊂ HostFunction`: the same impl can be surfaced as a graph `function` op or an agent tool. A tool's
 * body may itself invoke another runtime/sub-workflow — tools can be anything, like functions. See
 * RUNTIMES-AND-PERMISSIONS.md §2.
 */
export interface Tool extends HostFunction {
  /** What the tool does — shown to the model. */
  readonly description?: string;
  /** JSON Schema for the input the model must produce for a call. */
  readonly inputSchema: Record<string, unknown>;
}

/**
 * A per-runtime redirect to a DELEGATED agent's built-in tool of the given native name (RUNTIMES-AND-
 * PERMISSIONS.md §3). Unlike a {@link Tool} we cannot execute it ourselves — it names the black-box agent's
 * own tool, handed to the adapter as an alias/allowlist entry. A tool rename binding is therefore
 * `Tool | NativeToolRef`; `native` is a delegated-runtime concern (the `llm` runtime is always our-impl).
 */
export interface NativeToolRef {
  readonly native: string;
}

/** A plain-map facet. */
class MapRegistry<T> implements Registry<T> {
  private readonly map = new Map<string, T>();
  get(name: string): T | undefined {
    return this.map.get(name);
  }
  register(name: string, value: T): this {
    this.map.set(name, value);
    return this;
  }
}

/** A plain-map {@link CapabilityRegistry} — sufficient for both consumers. */
export class MapCapabilityRegistry implements CapabilityRegistry {
  readonly runtimes: Registry<Runtime> = new MapRegistry<Runtime>();
  readonly functions: Registry<HostFunction> = new MapRegistry<HostFunction>();
  readonly skills: Registry<SkillTemplate> = new MapRegistry<SkillTemplate>();
  readonly tools: Registry<Tool> = new MapRegistry<Tool>();
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
  /** Optional model router for llm-backed executors (typed as ModelRouter in @declarative-ai/llm). */
  modelRouter?: unknown;
  /** Executable tools the current agent operation may call mid-loop, keyed by name (RUNTIMES-AND-
   *  PERMISSIONS.md §2). A runtime forwards these here; an llm-backed executor adapts each into its tool
   *  loop. Absent ⇒ no host tools available for this call. */
  tools?: Record<string, Tool>;
  /** Mutable, logical-id-keyed session store — e.g. a workflow run injects a RUN-SCOPED one so states
   *  sharing a `sessionId` continue the same conversation. Absent ⇒ sessions unavailable. */
  sessions?: SessionStore;
  /** Content-addressed blob store for file/media I/O (Phase 5). Absent ⇒ blob refs can't be resolved. */
  blobs?: BlobStore;
  /** The working directory the current operation's workspace tools act within — a Session-owned resource
   *  (RUNTIMES-AND-PERMISSIONS.md §3). Absent ⇒ workspace-backed tools error. */
  workspace?: Workspace;
  /** The human tool-call approver (RUNTIMES-AND-PERMISSIONS.md §4). The engine wraps a COMPOSED runtime's
   *  tools itself, but a DELEGATED runtime (claude-code/opencode) that drives its own loop reads this to
   *  route its native permission callback back through our approval UI. Absent ⇒ no interactive gate. */
  approve?: Approver;
}

/** Minimal validation seam (implemented by `@declarative-ai/services` SchemaValidator). */
export interface OutputValidator {
  validateValue(schema: Record<string, unknown>, value: unknown): { ok: boolean; errors?: string };
}
