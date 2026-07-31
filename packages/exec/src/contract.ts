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
  /** The append-only session store — a workflow run injects one so ops naming the same stream
   *  continue one conversation. Absent ⇒ sessions unavailable. */
  sessions?: SessionStore;
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
 * A session is an APPEND-ONLY stream of messages, and a session ref names one AT a position — which
 * is what makes "continue from here" and "branch from here" the same primitive.
 *
 * `id` is OPAQUE. Nothing outside the store parses it: not this package, not the wrapper, not any
 * executor. That is what lets the spelling change — including the human-readable lineage label a
 * store may keep alongside — without touching a consumer.
 *
 * It is also the ONLY enumerable property, so `JSON.stringify`, an events journal, and any serialized
 * inputs/outputs see `{ id }` and nothing else.
 */
export interface SessionRef {
  readonly id: string;
}

/** One entry in a stream: a message verbatim, plus what the provider called it. */
export interface SessionMessage<Msg = JsonValue> {
  message: Msg;
  /** The provider's own id for this entry, when it has one. */
  providerRef?: string;
}

/**
 * What an executor REPORTS it appended.
 *
 * The wrapper cannot synthesize this. It sees only the op's prompt text and a final result value,
 * which is exactly the lossy behaviour this replaces: tool calls, tool results and reasoning parts
 * are all discarded by a wrapper-side fold. The executor is the only layer that knows what actually
 * went over the wire, so it is the layer that says so.
 *
 * Reported on FAILURE as well as success. If the provider appended turns and the call then failed,
 * those entries exist remotely; not recording them means the next append-by-handle meets a remote
 * head we do not mirror, which is divergence on the very next call.
 */
export interface SessionDelta<Msg = JsonValue> {
  /** Every entry the call added, in order — the request turns AND everything that came back. */
  messages: readonly SessionMessage<Msg>[];
  /** The provider's own session handle, when the provider is stateful. */
  providerSessionId?: string;
}

/**
 * The session an executor was handed, resolved to a concrete position.
 *
 * Everything but `id` is NON-ENUMERABLE — the same technique as the resolved-definition snapshot —
 * so the value that flows through the data plane stays `{ id }`.
 *
 * That has a consequence worth stating plainly: **`messages` is a cache, never the source of truth.**
 * Non-enumerable properties are dropped by object spread, by `JSON.parse(JSON.stringify(x))`, by
 * deep-clone helpers, and across a structured-clone IPC boundary. Since forking is expressed at the
 * CONSUMPTION site, somebody will eventually write `{ ...session, fork: true }`. An executor must
 * therefore be able to resolve messages from `id` alone and use the accessor only when it is there:
 * losing it must cost a store read, never correctness.
 */
export interface ResolvedSession<Msg = JsonValue> extends SessionRef {
  /** Whether this call continues the stream or branched off it. Decided BEFORE the call, because
   *  "is this a fork" and "how do I shape the request" are the same question — a fork must replay and
   *  must NOT pass a resume handle, or it appends to the wrong remote stream. */
  readonly mode: "append" | "fork";
  /** The provider handle to resume from, when the adapter can and the mode allows it. */
  readonly providerSessionId?: string;
  /** The stream's contents at this position. LAZY because the cheap path never needs them: an adapter
   *  that branches server-side reads zero messages. Only replay strategies materialize. */
  messages(): Promise<Msg[]>;
  /**
   * Report what this call actually appended.
   *
   * The channel exists here rather than on the execution result because a result is shared by every
   * op kind, and because a session is the one thing that is already present exactly when there is a
   * delta to report. It also makes append-on-error fall out for free: an executor reports before it
   * returns, whichever way it returns.
   */
  report(delta: SessionDelta<Msg>): void;
}

/**
 * Attach the non-enumerable half of a {@link ResolvedSession} to a bare ref.
 *
 * One helper so the non-enumerability is stated once. Defining these as ordinary properties is the
 * mistake this exists to prevent — it would put a function and a mode flag into every journal entry
 * and every `inputs_json`.
 */
export function resolveSessionRef<Msg = JsonValue>(
  id: string,
  rest: Omit<ResolvedSession<Msg>, "id">,
): ResolvedSession<Msg> {
  const session = { id } as ResolvedSession<Msg>;
  for (const [key, value] of Object.entries(rest)) {
    Object.defineProperty(session, key, { value, enumerable: false, writable: false, configurable: true });
  }
  return session;
}

/** What a caller asks for when it opens a session for one call. */
export interface SessionRequest {
  /** The position to continue or branch from. Absent ⇒ a new stream. */
  ref?: string;
  /** Always branch, rather than continuing when the position is still the head. */
  fork?: boolean;
  /**
   * A stable discriminator for any stream this call MINTS — a state id, a child key plus iteration.
   *
   * Stable rather than random on purpose: a fan-out that mints random ids produces different lineage
   * on every run, which degrades exactly the observability durable sessions exist for.
   */
  seed?: string;
  /** Which provider is about to be used, so the store can hand back that adapter's handle and only
   *  that one. The same stream replayed against two providers has two unrelated handles. */
  provider?: string;
}

/**
 * A RESERVED position, held for the duration of one call.
 *
 * Reserving rather than observing is the whole point. A peek leaves a window the length of the entire
 * model call: two calls both see head == 14, both conclude "linear append", and one clobbers the
 * other. That is survivable if both replay — the loser forks on the way out — and NOT survivable if
 * the winner took a resume-by-handle fast path, because by the time it loses it has already appended
 * remotely and there is nothing left to retroactively fork.
 */
export interface SessionLease<Msg = JsonValue> {
  /**
   * The EFFECTIVE id to write under, which may not be the one that was asked for: a store that had
   * to fork says so here. Without this channel a store can decide a fork and has no way to report it.
   */
  readonly session: ResolvedSession<Msg>;
  /**
   * Fold the reported delta, drop the reservation, and return the position the call ENDED at.
   *
   * The END position, because that is the only one that can exist by the time anyone reads it: you
   * append AT a position but do not know where you finished until the provider resolves. It is also
   * what consumers want — "append after me" and "fork after me" both mean *after*.
   *
   * Must run on success, on failure, on cancel and on throw; a leaked reservation pins the stream
   * forever and silently forks everything downstream. IDEMPOTENT, so a caller can release on the
   * happy path to read the end position and still release unconditionally in a `finally`.
   */
  release(delta?: SessionDelta<Msg>): string | Promise<string>;
}

/**
 * An append-only session store.
 *
 * `begin` RESERVES; `release` folds and frees. There is deliberately no `get`/`put` pair: an
 * observe-then-write API cannot express the reservation above, and a store that decides a fork has
 * nowhere to say so.
 */
export interface SessionStore<Msg = JsonValue> {
  begin(request: SessionRequest): SessionLease<Msg> | Promise<SessionLease<Msg>>;
  /** The stream's contents at a position, for a consumer that only wants to READ one. */
  read?(ref: string): Msg[] | Promise<Msg[]>;
  /**
   * Replace a stream's older entries with a summary, as a NEW stream, and return its head.
   *
   * A new stream rather than a rewrite, and NOT a fork. A fork's prefix is byte-identical to its
   * origin's — that is the whole claim a position makes — whereas a compacted stream begins with a
   * summary that appears nowhere in the origin. Rewriting in place would be worse still: it would
   * silently change what every existing ref refers to, and it invalidates the provider's prompt cache
   * (a strict prefix match) on every compaction.
   *
   * Optional, because not every store can express lineage. A store without it simply never compacts.
   */
  compact?(originRef: string, entries: readonly SessionMessage<Msg>[]): string | Promise<string>;
}

/**
 * A plain in-memory append-only store — the default when no durable one is injected.
 *
 * Small, but it implements the real semantics rather than approximating them: a reserved position is
 * held, a second reservation at the same position forks, and a fork's prefix is copied at the cursor
 * so the origin is never mutated.
 */
export class MapSessionStore<Msg = JsonValue> implements SessionStore<Msg> {
  private readonly streams = new Map<string, SessionMessage<Msg>[]>();
  private readonly handles = new Map<string, string>();
  private readonly held = new Set<string>();
  private minted = 0;

  read(ref: string): Msg[] {
    const [id, position] = split(ref);
    const stream = this.streams.get(id) ?? [];
    return stream.slice(0, position ?? stream.length).map((entry) => entry.message);
  }

  begin(request: SessionRequest): SessionLease<Msg> {
    const asked = request.ref !== undefined ? split(request.ref) : undefined;
    let [id, at] = asked ?? [this.mint(request.seed), undefined];
    if (asked === undefined) this.streams.set(id, []);
    const head = (): number => (this.streams.get(id) ?? []).length;
    // A bare id — no position — names the stream AT ITS HEAD, which is what "continue this
    // conversation" means when you hold a name rather than a position. Reading it as 0 instead would
    // make every second call fork, and the first turn would be replayed forever as the only history.
    let position = at ?? head();

    // `fork: true` skips the reservation entirely — the answer is already known. Otherwise a position
    // that is still the head and not held is an append, and anything else forks.
    let mode: "append" | "fork" = "append";
    if (request.fork === true || position !== head() || this.held.has(key(id, position))) {
      mode = "fork";
      const forked = this.mint(request.seed);
      this.streams.set(forked, (this.streams.get(id) ?? []).slice(0, position));
      id = forked;
      position = this.streams.get(forked)!.length;
    }
    this.held.add(key(id, position));

    const handle = mode === "append" && request.provider !== undefined ? this.handles.get(key(id, request.provider)) : undefined;
    const session = resolveSessionRef<Msg>(join(id, position), {
      mode,
      ...(handle !== undefined ? { providerSessionId: handle } : {}),
      messages: async () => (this.streams.get(id) ?? []).slice(0, position).map((entry) => entry.message),
      report: () => {
        /* the lease folds on release; nothing to buffer for an in-memory store */
      },
    });

    const heldKey = key(id, position);
    let released = false;
    return {
      session,
      release: (delta) => {
        // Idempotent: the caller releases on the happy path to read the end position, and again in a
        // `finally` that cannot know whether it already ran.
        if (released) return join(id, this.streams.get(id)?.length ?? position);
        released = true;
        this.held.delete(heldKey);
        if (delta !== undefined) {
          const stream = this.streams.get(id) ?? [];
          stream.push(...delta.messages);
          this.streams.set(id, stream);
          if (delta.providerSessionId !== undefined && request.provider !== undefined) {
            this.handles.set(key(id, request.provider), delta.providerSessionId);
          }
        }
        return join(id, this.streams.get(id)?.length ?? position);
      },
    };
  }

  compact(originRef: string, entries: readonly SessionMessage<Msg>[]): string {
    const [origin] = split(originRef);
    // A distinct stream, so the origin keeps meaning exactly what every ref into it already meant.
    const id = `${origin}~compact${++this.compactions}`;
    this.streams.set(id, [...entries]);
    return join(id, entries.length);
  }

  private compactions = 0;

  private mint(seed: string | undefined): string {
    // Seeded ids stay stable across a replayed run; an unseeded one only has to be unique.
    return seed !== undefined ? `s_${seed}` : `s_${++this.minted}`;
  }
}

const key = (a: string, b: string | number): string => `${a} ${b}`;
const join = (id: string, position: number): string => `${id}@${position}`;

/** `<id>@<position>`, or a bare id — which means the stream at whatever its head currently is. */
function split(ref: string): [string, number | undefined] {
  const at = ref.lastIndexOf("@");
  if (at <= 0) return [ref, undefined];
  const position = Number(ref.slice(at + 1));
  return Number.isInteger(position) && position >= 0 ? [ref.slice(0, at), position] : [ref, undefined];
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
