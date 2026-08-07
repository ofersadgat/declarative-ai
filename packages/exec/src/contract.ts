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
 * Dispatch is by op kind: `"prompt"` → the prompt executor, `"function"` → the function executor, a
 * registry lookup by `functionRef` (see {@link OperationExecutor}, which holds one of each and is
 * neither). Wrapper composition therefore applies UNIFORMLY to prompt and function ops alike.
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
export * from "./budget.js";
export * from "./ratelimit.js";

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
  /**
   * Time spent WAITING TO START, ms — queued for admission behind a bounded resource, and any loading
   * that admission had to do first.
   *
   * It exists because otherwise that time is indistinguishable from the work. A model that had to be
   * evicted and reloaded can add a minute before a single token is generated, and folding that into
   * `durationMs` reports the call as slow when the machine was busy — two very different findings, and
   * the one you act on differently. Kept as time rather than anything provider-specific, which is why
   * it can live here alongside `durationMs` without this package learning what a model is.
   */
  queuedMs?: number;
  /** LLM calls made by children, rolled up by a composite. A prompt op IS one such call; a non-LLM
   *  function (a pure helper, a sub-workflow that made none) contributes zero. */
  childLlmCalls?: number;
  /**
   * The session position this execution ENDED at, when a session was in play. Opaque.
   *
   * It rides the measurement record because that is the one channel every layer already forwards
   * unchanged — and because it has to be the EFFECTIVE position, which only the session layer knows:
   * a store that had to fork ended the call somewhere the caller cannot otherwise learn. Folding two
   * attempts keeps the later one, which is correct — the last attempt is the one that appended.
   */
  sessionRef?: string;
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
    // Sums like duration: two retried attempts that each queued for a model swap really did spend both
    // waits. Absent on both sides stays absent, so nothing gains a spurious zero.
    ...(a.queuedMs !== undefined || b.queuedMs !== undefined ? { queuedMs: (a.queuedMs ?? 0) + (b.queuedMs ?? 0) } : {}),
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
  | { type: "output_partial"; text: string }
  /**
   * A provider's own event, forwarded OPAQUELY.
   *
   * A delegated agent narrates a great deal that has no neutral home and should not be given one:
   * session init, compaction boundaries, hook lifecycle, task progress, API retries, rate-limit
   * windows, permission denials. Naming each in this union would teach `exec` one provider's
   * vocabulary for the sake of events it cannot act on — and the list is open, so the next version
   * would add to it.
   *
   * So the payload is JSON and nothing here interprets it. A host that wants to render a compaction
   * boundary can; a host that does not can ignore the whole variant. Same precedent as `rawUsage` and
   * `providerMetadata`: open by nature, JSON by construction (§2.2).
   */
  | { type: "provider_event"; payload: JsonValue }
  /**
   * Events were DROPPED because nothing was draining the stream — `count` of them, oldest first.
   *
   * A bounded queue has to shed something, and shedding it silently is the failure this variant
   * exists to prevent: a late consumer would otherwise receive a truncated stream that looks
   * complete. See `EventQueue`.
   */
  | { type: "events_dropped"; count: number };

// --- Executor -----------------------------------------------------------------

/**
 * Steering a RUNNING operation — the channel that is neither watching it nor stopping it.
 *
 * Every method is OPTIONAL and none of them throws when absent: absent MEANS unsupported, `if
 * (handle.control?.interrupt)` is the runtime check, and `capabilities.sessionSteering` is how a caller
 * knows before the call rather than by trying. A throwing stub would make "this transport cannot do
 * that" indistinguishable from "that failed", which is the distinction the whole record exists for.
 *
 * ⚠️ **`interrupt` is NOT `cancel`.** `handles.ts` deliberately unifies `cancel()` and `ctx.abortSignal`
 * into one event that settles the handle with a `canceled` failure. Interrupt is a third thing: the turn
 * ends EARLY, the operation still produces its answer, and the call SUCCEEDS. Wiring it to the abort
 * controller would throw away an answer that was actually produced — which is the opposite of what a
 * user pressing Stop on a long agent run is asking for. They want it to stop and tell them what it
 * found.
 */
export interface ExecControl {
  /** End the current turn early and let it settle NORMALLY. Idempotent: a call racing the operation's
   *  own completion is a no-op, never a second settle. */
  interrupt?(): Promise<void>;
  /** Add input to a run already under way. */
  send?(text: string): Promise<void>;
  /** Change the permission posture for the rest of the run. */
  setPermissionMode?(mode: string): Promise<void>;
  /** Change the model for subsequent responses. */
  setModel?(model: string): Promise<void>;
}

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
  /**
   * Steer the run, when the executor underneath offers it (see {@link ExecControl}).
   *
   * Absent ⇒ nothing to steer, which is the case for every ordinary operation. Through a wrapper stack
   * this follows the CURRENT inner handle — the same path `cancel()` takes — so it targets the attempt
   * actually running and a retry re-points it without the caller re-reading the handle.
   */
  readonly control?: ExecControl;
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
export interface Executor<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>, Out = ResolvedValue> {
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
  start(op: Op, ctx: R): ExecHandle<Out, M>;
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
 *
 * A delegated agent can now ANSWER a prompt op too (`AgentExecutor` is a `PromptExecutor`, DESIGN §4.4)
 * and that does NOT bring the asymmetry back: it is reached by being installed in the dispatcher's
 * prompt slot, not by a facet and not by a third op kind. The same agent is still a `functions` entry
 * when a `FunctionOp` names it — one executor, two ways in.
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
  meter?: import("./budget.js").BudgetMeter;
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
  /** Conversation lineage and resolution — a workflow run injects one so ops naming the same
   *  conversation continue it. Absent ⇒ sessions unavailable. */
  sessions?: SessionStore;
  /** Where executions are RECORDED (`withRecord`). Also where a session's messages come from, since a
   *  session is the records sharing a `session.id`. Absent ⇒ nothing is recorded. */
  records?: import("./record.js").RecordStore;
  /**
   * The session this call runs in, already resolved to a position and RESERVED (see
   * {@link SessionLease}).
   *
   * It arrives on the services bundle rather than by rewriting the op's config, and that is the whole
   * layering change: the wrapper owns the POLICY (resolve, reserve, decide append-vs-fork, fold,
   * release) and the executor owns the MECHANISM (shape the request, perform the fork, report what it
   * appended). Rewriting the config hardcoded replay, which is why a provider handle could not be
   * threaded at all.
   *
   * Declared at the JSON base, like {@link ExecServices.sessions} itself, and narrowed by whichever
   * consumer pins the message shape — promptop reads it as `ResolvedSession<ModelMessage>`, which are
   * exactly the messages it wrote.
   */
  session?: ResolvedSession;
  /**
   * The conversation the CALLER wants this call to run in — a request, not a resolution.
   *
   * The two are deliberately separate seams. A requester (`hw`, say) knows which conversation an
   * operation was authored against and whether the author asked to branch; it does NOT know where
   * that conversation currently is, and should not, because only the store does. `withSessionPosition`
   * turns this into {@link ExecServices.session}.
   *
   * That split is also what lets `hw` participate at all: it can state a request using nothing but
   * `exec`, where resolving one would drag in the layer it is not allowed to depend on.
   */
  sessionRequest?: SessionRequest;
  /**
   * Reads a conversation back FROM the provider, for re-syncing after divergence (DESIGN.md §1.6).
   *
   * Per-adapter and optional, because the capability genuinely is: Claude Code has
   * `getSessionMessages()`, Managed Agents has `events.list`, and the Messages API has neither — and
   * needs neither, being stateless and therefore unable to diverge. Absent ⇒ a resync starts EMPTY,
   * which the edge records rather than passing off as a conversation that happened to be empty.
   */
  sessionReader?: { read(providerSessionId: string): Promise<readonly unknown[]> };
  /**
   * Told when the remote moved underneath us, before anything is done about it.
   *
   * §11 says to log and then resync, in that order and both: the resync keeps the run going, and the
   * log is what stops a silently-diverging provider looking like normal operation.
   */
  onDivergence?: (event: { session: string; resumed: string; reported: string; reason: string }) => void;
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
 * A session is an APPEND-ONLY conversation, and a session ref names one AT a position — which is what
 * makes "continue from here" and "branch from here" the same primitive.
 *
 * **There is no message store.** A session IS the {@link OperationRecord}s sharing a `session.id`,
 * ordered by `seq`; a record already holds what its call produced, and for a prompt op that payload
 * carries the messages verbatim. Appending a turn and recording a call are one write (see
 * `withRecord`), which is also why the position reservation is just uniqueness on `(session, seq)`.
 *
 * `id` is OPAQUE. Nothing outside the store parses it — not this package, not the wrapper, not any
 * executor — which is what lets the spelling change without touching a consumer. It is also the ONLY
 * enumerable property, so `JSON.stringify`, an events journal and any serialized inputs/outputs see
 * `{ id }` and nothing else.
 */
export interface SessionRef {
  readonly id: string;
}

/** What a caller asks for when it opens a session for one call. */
export interface SessionRequest {
  /** The position to continue or branch from. Absent ⇒ a new conversation. */
  ref?: string;
  /** Always branch, rather than continuing when the position is still the head. */
  fork?: boolean;
  /**
   * A stable discriminator for any conversation this MINTS — a state id, a child key plus iteration.
   *
   * Stable rather than random on purpose: a fan-out that mints random ids produces different lineage
   * every run, which degrades exactly the observability durable sessions exist for.
   */
  seed?: string;
  /**
   * The provider about to serve the call.
   *
   * A conversation is LOCKED to the provider it was used with, so asking for one under a different
   * provider is a fork — enforced by resolution rather than by a rule anyone has to remember. The
   * provider a conversation belongs to is read off its latest record, not stored separately.
   */
  provider?: string;
}

/**
 * A session resolved to the concrete position a call will claim.
 *
 * Everything but `id` is NON-ENUMERABLE — the same technique as the resolved-definition snapshot —
 * so the value flowing through the data plane stays `{ id }`.
 *
 * That has a consequence worth stating plainly: **`messages` is a cache, never the source of truth.**
 * Non-enumerable properties are dropped by object spread, by `JSON.parse(JSON.stringify(x))`, by
 * deep-clone helpers, and across a structured-clone IPC boundary. Since forking is expressed at the
 * CONSUMPTION site, somebody will eventually write `{ ...session, fork: true }`. An executor must
 * therefore be able to resolve messages from `id` alone and use the accessor only when it is there:
 * losing it must cost a store read, never correctness.
 */
export interface ResolvedSession<Msg = JsonValue> extends SessionRef {
  /** Whether this call continues the conversation or branched off it. Decided BEFORE the call,
   *  because "is this a fork" and "how do I shape the request" are the same question — a fork must
   *  replay and must NOT pass a resume handle, or it appends to the wrong remote conversation. */
  readonly mode: "append" | "fork";
  /** The record slot this call claims. `withRecord` stamps its stub here; the store's uniqueness on
   *  this pair is the reservation. */
  readonly at: { id: string; seq: number };
  /** The provider handle to resume from, when the adapter can and the mode allows it. */
  readonly providerSessionId?: string;
  /** The conversation's contents at this position. LAZY because the cheap path never needs them: an
   *  adapter that branches server-side reads zero messages. Only replay strategies materialize. */
  messages(): Promise<Msg[]>;
}

/**
 * Attach the non-enumerable half of a {@link ResolvedSession} to a bare ref.
 *
 * One helper so the non-enumerability is stated once. Declaring these as ordinary properties is the
 * mistake it exists to prevent — it would put a function and a position into every journal entry and
 * every `inputs_json`.
 */
export function resolveSessionRef<Msg = JsonValue>(id: string, rest: Omit<ResolvedSession<Msg>, "id">): ResolvedSession<Msg> {
  const session = { id } as ResolvedSession<Msg>;
  for (const [key, value] of Object.entries(rest)) {
    Object.defineProperty(session, key, { value, enumerable: false, writable: false, configurable: true });
  }
  return session;
}

/**
 * Conversation lineage and resolution. The RECORDS are written through `RecordStore`; this store owns
 * which conversation a ref names, where it sits, and how a new one comes into being.
 */
export interface SessionStore<Msg = JsonValue> {
  /** Resolve a request to the position a call should claim. */
  resolve(request: SessionRequest): ResolvedSession<Msg> | Promise<ResolvedSession<Msg>>;
  /**
   * Branch `ref`, returning a ref to the new conversation at the same content.
   *
   * What a caller does when a position turns out to be taken. Not a retry at the next slot: appending
   * at 15 instead of 14 would continue a conversation containing a turn this call never saw.
   */
  fork(ref: string, seed?: string): string | Promise<string>;
  /** The conversation's contents at a position. */
  messages(ref: string): Msg[] | Promise<Msg[]>;
  /**
   * A NEW conversation whose older messages are replaced by a summary.
   *
   * Not a fork: a fork's prefix is byte-identical to its origin's, which is the whole claim a position
   * makes, whereas a compacted conversation opens with a summary appearing nowhere in the origin.
   * Rewriting in place would be worse still — it silently changes what every existing ref refers to,
   * and invalidates the provider's prompt cache (a strict prefix match) on every compaction.
   */
  compact?(ref: string, messages: readonly Msg[]): string | Promise<string>;
  /** A NEW conversation re-read from the provider after the remote diverged from our mirror. Where an
   *  adapter has no read API the caller passes nothing and it starts EMPTY — visible on the edge
   *  rather than silent, which is the point of it being a distinct kind. */
  resync?(ref: string, messages: readonly Msg[]): string | Promise<string>;
}

/**
 * Raised when a stub cannot claim its position because something already holds it.
 *
 * A distinct class on purpose: "someone got there first" is routine and has a correct answer, whereas
 * a broken database does not. The correct answer is NOT the one findmyprompt's `appendDraw` takes for
 * a draw list — it recomputes `MAX(index)` and retries at the next slot, which is right there because
 * a draw's order commits to nothing. A session must FORK instead: appending at 15 rather than 14 would
 * continue a conversation containing a turn this call never saw.
 */
export class PositionTaken extends Error {
  constructor(
    readonly session: string,
    readonly seq: number,
  ) {
    super(`session ${session}: position ${seq} is already claimed`);
    this.name = "PositionTaken";
  }
}

/** How a stored record yields conversation messages. Injected because only the llm layer knows that a
 *  prompt op's payload is an `LlmOutput` with a `messages` field; the store stays generic. */
export type MessagesOf<Msg> = (record: { result?: { value?: unknown } }) => readonly Msg[];

/** The default: a payload carrying `messages`, which is what a prompt op's `LlmOutput` does. */
export const defaultMessagesOf = <Msg>(record: { result?: { value?: unknown } }): readonly Msg[] => {
  const value = record.result?.value as { messages?: readonly Msg[] } | undefined;
  return value?.messages ?? [];
};

/**
 * The in-memory default: conversation lineage plus the records themselves.
 *
 * One class rather than two because the two are inseparable in the small — a conversation's messages
 * ARE its records — and a durable implementation splits them across tables in the same database
 * anyway. It implements the real semantics rather than approximating them: a claimed position is held
 * by the record occupying it, a second claim throws {@link PositionTaken}, and a fork shares its
 * origin's prefix by lineage rather than by copying.
 */
export class MapSessionStore<Msg = JsonValue> implements SessionStore<Msg> {
  /** Lineage only. A branch's own records live in `rows`; its prefix is its parent's. */
  private readonly branches = new Map<string, { parent?: string; cursor: number }>();
  private readonly rows = new Map<string, Map<number, { id: string; result?: { value?: unknown }; externalId?: string }>>();
  private minted = 0;

  constructor(private readonly messagesOf: MessagesOf<Msg> = defaultMessagesOf) {}

  resolve(request: SessionRequest): ResolvedSession<Msg> {
    const asked = request.ref !== undefined ? split(request.ref) : undefined;
    let id = asked?.[0] ?? this.mint(request.seed);
    if (asked === undefined) this.branches.set(id, { cursor: 0 });
    let seq = asked?.[1] ?? this.head(id);
    let mode: "append" | "fork" = "append";
    // `fork: true` skips any check — the answer is already known. Everything else is decided by the
    // record write itself, which is the only place that can decide it without a race.
    if (request.fork === true) {
      id = this.branchFrom(id, seq, request.seed);
      seq = this.head(id);
      mode = "fork";
    }
    // The provider handle this conversation currently sits on, read off its LATEST record — there is
    // no handle map, because a conversation is locked to the provider it was used with. A FORK gets
    // none: inheriting the parent's handle would put two branches into one remote session.
    const handle = mode === "append" ? this.handleAt(id, seq) : undefined;
    return resolveSessionRef<Msg>(join(id, seq), {
      mode,
      at: { id, seq },
      ...(handle !== undefined ? { providerSessionId: handle } : {}),
      messages: async () => this.materialize(id, seq),
    });
  }

  fork(ref: string, seed?: string): string {
    const [id, seq] = split(ref);
    const forked = this.branchFrom(id, seq ?? this.head(id), seed);
    return join(forked, this.head(forked));
  }

  messages(ref: string): Msg[] {
    const [id, seq] = split(ref);
    return this.materialize(id, seq ?? this.head(id));
  }

  compact(ref: string, messages: readonly Msg[]): string {
    return this.derive(ref, "compact", messages);
  }

  resync(ref: string, messages: readonly Msg[]): string {
    return this.derive(ref, "resync", messages);
  }

  // --- The record half ----------------------------------------------------------

  open(stub: { id: string; session?: { id: string; seq: number } }): void {
    const at = stub.session;
    if (at === undefined) return; // a record outside any conversation is not this store's business
    const rows = this.rows.get(at.id) ?? new Map();
    if (rows.has(at.seq)) throw new PositionTaken(at.id, at.seq);
    rows.set(at.seq, { id: stub.id });
    this.rows.set(at.id, rows);
  }

  close(id: string, settled: { result?: { value?: unknown }; sessionOutcome?: { messages?: readonly unknown[]; providerSessionId?: string } }): void {
    for (const rows of this.rows.values()) {
      for (const row of rows.values()) {
        if (row.id === id) {
          // An executor whose payload IS a conversation needs nothing further; one whose payload is
          // not — a delegated agent, a fake — reports its delta on the session channel, and that is
          // what the conversation is made of. Preferring it keeps `messagesOf` reading one shape.
          const reported = settled.sessionOutcome?.messages;
          row.result = reported !== undefined ? { value: { messages: reported } } : settled.result;
          // The handle the call ENDED in. Kept per record rather than per conversation so that reading
          // it at an earlier position reports what was true THEN — which is what makes a mismatch on
          // the next call detectable as divergence rather than invisible.
          const handle = settled.sessionOutcome?.providerSessionId;
          if (handle !== undefined) row.externalId = handle;
          return;
        }
      }
    }
  }

  bySession(session: string, upTo?: number): Array<{ id: string; result?: { value?: unknown } }> {
    const rows = this.rows.get(session) ?? new Map();
    return [...rows.entries()]
      .filter(([seq]) => upTo === undefined || seq < upTo)
      .sort(([a], [b]) => a - b)
      .map(([, row]) => row);
  }

  // --- Internals ----------------------------------------------------------------

  /** Walk the lineage, taking each ancestor's records below the cursor its child took. Copy-on-write:
   *  a branch stores only what it appended, so cost is proportional to divergence. */
  private materialize(id: string, upTo: number): Msg[] {
    const out: Msg[] = [];
    let at: string | undefined = id;
    let limit = upTo;
    const chain: Array<[string, number]> = [];
    while (at !== undefined) {
      chain.unshift([at, limit]);
      const branch: { parent?: string; cursor: number } | undefined = this.branches.get(at);
      if (branch?.parent === undefined) break;
      limit = branch.cursor;
      at = branch.parent;
    }
    for (const [branch, bound] of chain) {
      const start = this.branches.get(branch)?.cursor ?? 0;
      for (const [seq, row] of [...(this.rows.get(branch) ?? new Map()).entries()].sort(([a], [b]) => a - b)) {
        if (seq >= start && seq < bound) out.push(...this.messagesOf(row));
      }
    }
    return out;
  }

  /** The latest handle at or before a position, walking the lineage as materializing does. */
  private handleAt(id: string, upTo: number): string | undefined {
    let at: string | undefined = id;
    let bound = upTo;
    while (at !== undefined) {
      const rows = [...(this.rows.get(at) ?? new Map()).entries()].sort(([a], [b]) => b - a);
      for (const [seq, row] of rows) if (seq < bound && row.externalId !== undefined) return row.externalId;
      const branch: { parent?: string; cursor: number } | undefined = this.branches.get(at);
      if (branch?.parent === undefined) return undefined;
      bound = branch.cursor;
      at = branch.parent;
    }
    return undefined;
  }

  /** The next position an append would occupy. A branch's own records begin at its cursor. */
  private head(id: string): number {
    const rows = this.rows.get(id);
    const max = rows === undefined || rows.size === 0 ? undefined : Math.max(...rows.keys());
    return max === undefined ? (this.branches.get(id)?.cursor ?? 0) : max + 1;
  }

  private branchFrom(id: string, cursor: number, seed?: string): string {
    const forked = this.mint(seed !== undefined ? `${seed}@${id}:${cursor}` : undefined);
    // `seq` CONTINUES from the cursor, so a position is one integer across a whole lineage.
    this.branches.set(forked, { parent: id, cursor });
    return forked;
  }

  private derive(ref: string, word: string, messages: readonly Msg[]): string {
    const [id] = split(ref);
    // A distinct conversation, so the origin keeps meaning exactly what every ref into it meant.
    const derived = `${id}~${word}${++this.minted}`;
    this.branches.set(derived, { cursor: 0 });
    const rows = new Map<number, { id: string; result?: { value?: unknown } }>();
    rows.set(0, { id: `${derived}:0`, result: { value: { messages } } });
    this.rows.set(derived, rows);
    return join(derived, 1);
  }

  private mint(seed: string | undefined): string {
    return seed !== undefined ? `s_${seed}` : `s_${++this.minted}`;
  }
}

const join = (id: string, seq: number): string => `${id}@${seq}`;

/** `<id>@<position>`, or a bare id — which names the conversation at whatever its head currently is. */
function split(ref: string): [string, number | undefined] {
  const at = ref.lastIndexOf("@");
  if (at <= 0) return [ref, undefined];
  const seq = Number(ref.slice(at + 1));
  return Number.isInteger(seq) && seq >= 0 ? [ref.slice(0, at), seq] : [ref, undefined];
}

// --- Composition --------------------------------------------------------------

/**
 * A composable behavior wrapped around an executor — memoize / retry / rate-limit / deadline / budget /
 * session. It maps an executor requiring `RIn` to one requiring `ROut`: a construction-injected wrapper
 * leaves the requirement unchanged (`ExecutorWrapper<R, R>`); a ctx-reading one ADDS its seam
 * (`withDeadline(): ExecutorWrapper<R, R & { deadline; stepStartMs }>`). The stacking ORDER encodes
 * semantics — see the two forms below.
 */
export type ExecutorWrapper<RIn = ExecServices, ROut = RIn, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>, Out = ResolvedValue> = (
  inner: Executor<RIn, M, Op, Out>,
) => Executor<ROut, M, Op, Out>;

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
export class ComposableExecutor<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>, Out = ResolvedValue> implements Executor<R, M, Op, Out> {
  /** Forwarded so the per-op capability lookup survives the builder — and forwarded CONDITIONALLY, so
   *  that "this executor has no per-op record" (which `withMemoize` reads as "the static record IS the
   *  whole truth") survives too. See {@link Executor.capabilitiesFor}. */
  readonly capabilitiesFor?: (op: Op) => Capabilities;
  constructor(private readonly inner: Executor<R, M, Op, Out>) {
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
  with<ROut, OpOut = Op>(wrap: (inner: Executor<R, M, Op, Out>) => Executor<ROut, M, OpOut, Out>): ComposableExecutor<ROut, M, OpOut, Out> {
    return new ComposableExecutor(wrap(this.inner));
  }
  start(op: Op, ctx: R): ExecHandle<Out, M> {
    return this.inner.start(op, ctx);
  }
}

/** Start the inside-out builder around a core executor — see {@link ComposableExecutor}. */
export function compose<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>, Out = ResolvedValue>(
  core: Executor<R, M, Op, Out>,
): ComposableExecutor<R, M, Op, Out> {
  return new ComposableExecutor(core);
}
