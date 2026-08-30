/**
 * The function registry (API.md, "The function registry") — findmyprompt's `FunctionRegistry` made GENERIC in the
 * async context, with the runtime facet FULLY merged in. A `functionRef` on a `FunctionOp` names an
 * entry here.
 *
 * The registry IS a `Map`. It was an interface plus a class whose whole content was `get`/`has`/`refs`
 * (already `Map.get`/`has`/`keys`) and three `registerX` helpers that only built a discriminated entry —
 * so the entry constructors became free functions and the container became the map it always was.
 * `Map.set` returns the map, so even the chaining the class existed to provide is unchanged:
 *
 *     functions.set("summarize", pureFunction(impl)).set("ask", hostFunction(impl, HOST_CAPABILITIES))
 *
 * An entry is a DISCRIMINATED UNION over how it runs — `pure` | `host` | `runtime` — and each variant's
 * capability record is REQUIRED and TOTAL for that variant. There is no "registered but uncharacterized"
 * state, so permission gating and search refusal read a definite value instead of falling through an
 * `undefined`.
 *
 * Consequence for validation: hw's static validator wants to reject "an interactive function in a
 * search-only workflow" at authoring time, which means the checker reads the registry. Validation is
 * therefore a function of *(document, registry)*, not of the document alone. That is deliberate — see
 * {@link FunctionCapabilities}, the read-only view it needs.
 *
 * Errors are DATA (§4.2): an impl RESOLVES with a `Result` carrying value-or-failure rather than
 * throwing, so a 429 raised inside a function impl carries its classification to the retry machinery
 * instead of being reconstructed from `err.name`. `catch` remains a fallback for impls that throw
 * anyway — see `liftThrowing`.
 */
import type { Failure, JsonValue, Result } from "@declarative-ai/json";
import { classifyError, describeError, isRateLimit, retryAfterMs } from "@declarative-ai/json";
import type { InlineFamily, ResolvedValue, Signature } from "./model.js";

/** The dynamic base for function inputs (§2.2); refined per-def by `FunctionDef<I, O>`. */
export type FunctionInputs = Record<string, ResolvedValue>;

/**
 * What an impl RESOLVES with: the value, or a classified failure. Errors travel as data through the
 * promise, not as exceptions — which is what lets a retriable failure inside a function impl actually
 * be retried.
 *
 * `metrics` is the impl's optional REPORT of what the work cost, and `M` is a parameter because the
 * report belongs to the impl, not to the registry. Most impls are pure glue and have nothing to say;
 * the ones that do are the expensive ones — a delegated agent spends real money inside its own loop and
 * is the only thing that knows how much. Without this channel that spend has nowhere to go, so an
 * agent's cost simply vanished once dispatch moved to the function registry.
 */
export type FunctionResult<O, M> = Result<O, Failure> & { metrics?: M; session?: ReportedSession };

/**
 * What an impl reports about the CONVERSATION it ran in — `exec`'s `SessionOutcome`, restated
 * structurally because `exec` depends on this package and not the other way round.
 *
 * It is declared rather than left to structural leniency, and that is not a formality: this type is a
 * union, so an undeclared extra field typechecks and is then silently dropped by anything that
 * REBUILDS the result — which the operation dispatcher does on every call. A delegated agent's
 * provider session id travelled exactly that path and vanished, so every resume started a fresh
 * conversation and no failure was raised anywhere.
 */
export interface ReportedSession {
  /** The provider's own session id as of this run. A CHANGE from what we resumed is divergence. */
  providerSessionId?: string;
  /** What the run added, for an executor whose payload is not already a conversation. */
  messages?: readonly unknown[];
}

/** A pure, sync function implementation: deterministic glue (parse, combine, select) — no model, no
 *  stage, no cost. It still resolves value-or-error, because "this input is malformed" is a real,
 *  classifiable outcome. */
export type FunctionImpl<I, O, M> = (inputs: I) => FunctionResult<O, M>;

/** An async, ctx-bearing function implementation — it orchestrates (runs inner operations through
 *  `Ctx`, spends budget, is long-running and stochastic). */
export type AsyncFunctionImpl<I, O, M, Ctx> = (inputs: I, ctx: Ctx) => Promise<FunctionResult<O, M>>;

/**
 * The `catch` FALLBACK (§4.2): wrap an impl that throws anyway so its exception becomes a classified
 * failure. The classification comes from the error itself (`classifyError` reads AI-SDK retryable
 * flags, HTTP status, abort, and network codes), so a 429 thrown by an impl is `network-retriable`
 * rather than the blanket `permanent` an `err.name` guess produced.
 */
export function liftThrowing<I, O, Ctx>(
  impl: (inputs: I, ctx: Ctx) => O | Promise<O>,
  context?: string,
): (inputs: I, ctx: Ctx) => Promise<Result<O, Failure>> {
  return async (inputs, ctx) => {
    try {
      return { value: await impl(inputs, ctx) };
    } catch (e) {
      return { error: failureOf(e, context) };
    }
  };
}

/** Classify a thrown value into a {@link Failure}, prefixing the context that raised it. */
export function failureOf(e: unknown, context?: string): Failure {
  const name = e !== null && typeof e === "object" ? (e as { name?: unknown }).name : undefined;
  const classification = name === "AbortError" ? "canceled" : classifyError(e);
  const reason = describeError(e);
  const wait = retryAfterMs(e);
  return {
    classification,
    reason: context ? `${context}: ${reason}` : reason,
    ...(wait !== undefined ? { retryAfterMs: wait } : {}),
    ...(isRateLimit(e) ? { rateLimited: true } : {}),
  };
}

// --- Capabilities -------------------------------------------------------------

/** How an executable enforces the compiled safety policy. */
export type PolicyEnforcement = "callback" | "config" | "none";

/** A `pure` entry: deterministic glue. The one axis that varies is whether re-running it is sound to
 *  skip — a pure-but-clock/random-reading helper says `memoizable: false`. */
export interface PureCapabilities {
  memoizable: boolean;
}

/** A `host` entry: host code, including interactive UI. */
export interface HostCapabilities {
  /** Needs a human/renderer (was a `ui` op) — refused up-front by search callers. */
  interactive: boolean;
  /** Does not mutate the workspace/world — what the `read-only`/`plan` profiles gate on. */
  readOnly: boolean;
  /** Sound to memoize under the standard memo key. */
  memoizable: boolean;
  /**
   * The call does not RUN so much as WAIT: it registers interest in something outside the run and
   * settles when that thing happens — a person answering, an event arriving, a deadline passing.
   *
   * Optional, and absent means `false`, which is what every function written before this field
   * meant. Declaring it changes only WHO WAITS. An ordinary embedded call is awaited by the
   * scheduling round that needed it, which is right for a computation that will finish on its own
   * and wrong for one that may never finish: a guard calling it would hold the round open for as
   * long as nobody acts, and a round holding open is a state that cannot report what it is waiting
   * for. A deferred call is instead STARTED by the round and read by a later one — the same
   * `PENDING`-then-value protocol a running child already resolves through — so the engine stays
   * live, the wait is visible, and cancelling the run cancels the wait.
   *
   * It says nothing about purity or side effects; `readOnly` and `memoizable` still answer those. A
   * deferred function is very often `memoizable: false`, because what it returns is an event and not
   * a computation over its arguments.
   */
  deferred?: boolean;
}

/**
 * A `runtime` entry: a delegated agent adapter, or any executable driving its own loop. This is also
 * the record an `Executor` advertises (see {@link Capabilities}) — `ExecutorCapabilities` merged in
 * here rather than kept alongside.
 */
export interface RuntimeCapabilities extends HostCapabilities {
  /** Native schema-constrained output support. */
  structuredOutput: boolean;
  /** Requires a workspace; memo keys must fold its tree hash. */
  mutatesWorkspace: boolean;
  /** How the adapter enforces the compiled safety policy. */
  policyEnforcement: PolicyEnforcement;
  /** Supports session continuation. */
  sessionResume: boolean;
  /**
   * Supports BRANCHING a conversation server-side, as distinct from continuing one.
   *
   * Optional, and absent means "the same as {@link RuntimeCapabilities.sessionResume}" — which is what
   * every adapter written before this field meant by declaring resume: Claude Code's `resume` +
   * `forkSession` gives both at once, and the Messages API replays, so both halves are free.
   *
   * Codex is what splits them: `codex exec resume <id>` appends natively, and there is no fork
   * primitive at all. An adapter in that position declares `{ sessionResume: true, sessionFork: false }`
   * and REPLAYS a fork instead — which is a different request shape, not a slower path to the same one.
   *
   * The rule this exists to make checkable: a fork must never inherit the parent's provider handle
   * unless the adapter can really branch, because two branches writing into one remote session is
   * silent and unrecoverable (SESSIONS.md §6).
   */
  sessionFork?: boolean;
  /**
   * Supports branching a conversation at an ARBITRARY point, not only from where it now stands.
   *
   * Claude Code takes `--resume-session-at <message id>` — "only messages up to and including the
   * assistant message with <message.id>" — so a branch cut behind the remote's tip is still one
   * server-side copy. Everything else copies the tip or nothing, and a branch behind it has to be
   * replayed: copying would hand it turns it never had, which in the automatic-fork case is the very
   * turn that took its position.
   *
   * Absent means NO. This is the narrower capability, and an adapter that has it says so.
   */
  sessionForkAt?: boolean;
  /**
   * A RUNNING call can be steered — interrupted, redirected, given more input mid-turn.
   *
   * Separate from `streaming`, which is about watching, and from `interactive`, which is about being
   * asked. This is about talking back: `ExecHandle.control`.
   *
   * It is declared rather than discovered because the alternative is a caller finding out by calling.
   * A UI decides whether to render a Stop button BEFORE the run starts, and "the method was absent so
   * nothing happened" is the worst possible answer to a user pressing it. Absent ⇒ no steering, which
   * is the honest default for a transport whose only mid-run signal is a kill (codex is the case:
   * SIGINT to a subprocess ends the process, not the turn).
   */
  sessionSteering?: boolean;
  /** Emits incremental output. */
  streaming: boolean;
  /** Where it can run. */
  runtime: "edge-safe" | "node";
}

/** The capability record an `Executor` advertises — the same total record a `runtime` entry carries,
 *  because an executor IS the thing a runtime entry delegates to. */
export type Capabilities = RuntimeCapabilities;

/** Sensible totals for hand-registering an entry without spelling out every field. */
export const PURE_CAPABILITIES: PureCapabilities = { memoizable: true };

export const HOST_CAPABILITIES: HostCapabilities = { interactive: false, readOnly: true, memoizable: true };

export const RUNTIME_CAPABILITIES: RuntimeCapabilities = {
  interactive: false,
  readOnly: false,
  memoizable: false,
  structuredOutput: false,
  mutatesWorkspace: false,
  policyEnforcement: "none",
  sessionResume: false,
  streaming: false,
  runtime: "node",
};

// --- Entries ------------------------------------------------------------------

/** Per-registration options. `stream` opts into two-phase in-flight visibility (a partial record at
 *  open, filled on completion) + spawn tracking — meaningful only for spawning async functions.
 *  Progressive output rides THIS, never the return value, which is why §4.2's move to a resolved
 *  `Result` costs nothing in streaming expressiveness. */
export interface AsyncFunctionOptions {
  stream?: boolean;
}

/**
 * One registry entry, discriminated by HOW it runs. `capabilities` is required and total per variant.
 */
/**
 * The I/O contract an entry implements, when it declares one.
 *
 * Optional because a registry is populated by host code that may not care — but a caller that DOES
 * declare it gets something no name lookup can give: a checker can verify that whatever names this
 * entry agrees with it. That matters most for a callee named from an expression, where the argument
 * types are decided by a DOCUMENT and the implementation is over here; without a signature the two
 * can disagree and nothing notices until the call runs.
 *
 * Since `Signature.input` became a map of NAMED slots, this is also how an entry declares how it is
 * CALLED and not merely what it accepts. `index` on a slot is the positional order an expression
 * binds against, so `review(doc, "terse")` reaches a host function with nothing written in any
 * document — which is what removed the need for a host to ship an operation document beside the
 * implementation just to have somewhere to say that.
 */
export interface EntrySignature {
  signature?: Signature<InlineFamily>;
}

export interface PureFunction<M> extends EntrySignature {
  kind: "pure";
  impl: FunctionImpl<FunctionInputs, ResolvedValue, M>;
  capabilities: PureCapabilities;
  stream?: false;
  description?: string;
}

export interface HostFunction<Ctx, M> extends EntrySignature {
  kind: "host";
  impl: AsyncFunctionImpl<FunctionInputs, ResolvedValue, M, Ctx>;
  capabilities: HostCapabilities;
  stream?: boolean;
  description?: string;
}

export interface RuntimeFunction<Ctx, M> extends EntrySignature {
  kind: "runtime";
  impl: AsyncFunctionImpl<FunctionInputs, ResolvedValue, M, Ctx>;
  capabilities: RuntimeCapabilities;
  stream?: boolean;
  description?: string;
}

/** A `pure` entry takes no ctx, so it fits ANY registry — which is why it is named separately rather
 *  than pinned to the union's `Ctx`. */
export type RegisteredFunction<Ctx, M> = PureFunction<M> | HostFunction<Ctx, M> | RuntimeFunction<Ctx, M>;

/**
 * The read-only VIEW of an entry that a checker needs: how it runs, and what it may do. Validation
 * never invokes an impl, so it must not have to name `Ctx` or `M` to ask whether a function is
 * interactive — this is what `Pick<FunctionRegistry<never>, "get" | "has">` was reaching for.
 *
 * A real registry is assignable to `ReadonlyMap<string, FunctionCapabilities>`, and the discriminant
 * survives: `entry.kind !== "pure"` still narrows to the variants that HAVE `interactive`.
 */
export type FunctionCapabilities =
  | ({ kind: "pure"; capabilities: PureCapabilities } & EntrySignature)
  | ({ kind: "host"; capabilities: HostCapabilities } & EntrySignature)
  | ({ kind: "runtime"; capabilities: RuntimeCapabilities } & EntrySignature);

/**
 * The registry: a plain map from `functionRef` to entry. `refs()` was `[...map.keys()]`; `has`/`get`
 * are the map's own.
 */
export type FunctionRegistry<Ctx, M> = Map<string, RegisteredFunction<Ctx, M>>;

/** Build a deterministic, sync entry. */
export function pureFunction<M>(
  impl: FunctionImpl<FunctionInputs, ResolvedValue, M>,
  capabilities: PureCapabilities = PURE_CAPABILITIES,
  signature?: Signature<InlineFamily>,
): PureFunction<M> {
  return { kind: "pure", impl, capabilities, ...(signature !== undefined ? { signature } : {}) };
}

/** Build a host-code entry (including interactive UI). */
export function hostFunction<Ctx, M>(
  impl: AsyncFunctionImpl<FunctionInputs, ResolvedValue, M, Ctx>,
  capabilities: HostCapabilities,
  opts?: AsyncFunctionOptions & EntrySignature,
): HostFunction<Ctx, M> {
  return {
    kind: "host",
    impl,
    capabilities,
    ...(opts?.stream !== undefined ? { stream: opts.stream } : {}),
    ...(opts?.signature !== undefined ? { signature: opts.signature } : {}),
  };
}

/** Build a delegated-runtime-adapter entry. */
export function runtimeFunction<Ctx, M>(
  impl: AsyncFunctionImpl<FunctionInputs, ResolvedValue, M, Ctx>,
  capabilities: RuntimeCapabilities,
  opts?: AsyncFunctionOptions & EntrySignature,
): RuntimeFunction<Ctx, M> {
  return {
    kind: "runtime",
    impl,
    capabilities,
    ...(opts?.stream !== undefined ? { stream: opts.stream } : {}),
    ...(opts?.signature !== undefined ? { signature: opts.signature } : {}),
  };
}

/**
 * Run an entry uniformly: a `pure` impl takes no ctx, the other two do. The caller never branches on
 * the variant just to invoke it — only to GATE it.
 *
 * NEVER THROWS. The impl contract is that errors resolve as data (§4.2), but nothing at registration
 * forces an impl through {@link liftThrowing}. So the `catch` fallback lives HERE, at the one place
 * every dispatch path goes through, rather than being a rule each caller has to remember: a throwing
 * impl becomes a CLASSIFIED failure instead of a rejection that escapes the caller's error handling.
 */
export async function runFunction<Ctx, M>(entry: RegisteredFunction<Ctx, M>, inputs: FunctionInputs, ctx: Ctx): Promise<FunctionResult<ResolvedValue, M>> {
  try {
    return await (entry.kind === "pure" ? entry.impl(inputs) : entry.impl(inputs, ctx));
  } catch (e) {
    return { error: failureOf(e) };
  }
}

/** True when an entry is registered as streaming (two-phase open + spawn tracking). */
export function isStreaming<Ctx, M>(entry: RegisteredFunction<Ctx, M> | undefined): boolean {
  return entry?.stream === true;
}
