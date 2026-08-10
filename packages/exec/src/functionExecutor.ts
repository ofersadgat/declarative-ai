/**
 * The FUNCTION `Executor` (DESIGN §3.1) — a registry lookup, and the call that follows it.
 *
 * This is one half of what {@link OperationExecutor} used to be. That class was simultaneously the
 * DISPATCHER (which of the two kinds is this?) and the function IMPLEMENTATION (find the entry,
 * resolve the literals, run it under a linked abort controller, fold the metrics). Those are two
 * domains, and fusing them had a visible cost: the prompt side was a first-class object that could be
 * subclassed, wrapped and swapped, while the function side was a private method nobody could reach.
 *
 * Split out, the asymmetry goes away — dispatch holds a reference to each half, and each half is an
 * ordinary `Executor` that composition reaches on equal terms.
 *
 * The input contract is unchanged and worth restating: this executor takes a RESOLVED op. Every input
 * `Parameter` is either free (filled by name from `inputs`) or bound to a LITERAL value ref. Walking a
 * producer edge is the ref family's business, so one arriving here is a wiring bug and is reported as
 * one. Embedded operations are settled one layer up, in `OperationExecutor.start`, because running
 * them needs an executor that can dispatch BOTH kinds.
 */
import type {
  Capabilities,
  FunctionInputs,
  FunctionRegistry,
  InlineFamily,
  MetricsAlgebra,
  Operation,
  RegisteredFunction,
  ResolvedValue,
  FunctionResult,
} from "@declarative-ai/ops";
import { RUNTIME_CAPABILITIES, failureOf, isOk, runFunction } from "@declarative-ai/ops";
import type { ExecHandle, ExecMetrics, ExecServices, Executor, ExecResult } from "./contract.js";
import { EXEC_METRICS_ALGEBRA } from "./contract.js";
import { resolveLiteralInputs } from "./inputs.js";
import { isResolvedCall } from "./resolvedOperation.js";
import { EventQueue, canceledFailure, failure, finishedHandle, linkAbort, raceWork } from "./handles.js";
import { systemClock } from "./deadline.js";

export interface FunctionExecutorOptions<M extends ExecMetrics = ExecMetrics> {
  /** The one registry of discriminated entries (§2) — host code, sub-workflows, and delegated runtime
   *  adapters alike. */
  functions: FunctionRegistry<ExecServices, M>;
  /** Override the advertised capabilities (defaults to the conservative function-only record). */
  capabilities?: Capabilities;
  /** How this executor's measurements combine. Defaults to timing/counts. */
  metrics?: MetricsAlgebra<M>;
}

/** The capability floor for a registry-dispatched function op, and the fallback when a `functionRef`
 *  resolves to nothing (there is no entry whose record could be read). */
export const FUNCTION_CAPABILITIES: Capabilities = { ...RUNTIME_CAPABILITIES, memoizable: true, runtime: "node" };

/**
 * A registry entry's OWN capability record, widened to the total {@link Capabilities} a wrapper gate
 * reads. `pure` and `host` variants declare a strict subset (§2 keeps each variant's record total FOR
 * THAT VARIANT), so the fields they do not name take the conservative runtime defaults — a pure function
 * neither mutates a workspace nor resumes a session, which is exactly what those defaults say.
 */
export function entryCapabilities(entry: RegisteredFunction<ExecServices, ExecMetrics>): Capabilities {
  return { ...RUNTIME_CAPABILITIES, ...entry.capabilities };
}

export class FunctionExecutor<M extends ExecMetrics = ExecMetrics> implements Executor<ExecServices, M> {
  /**
   * The hierarchy's own discriminant.
   *
   * A serializable tag on the class rather than a union maintained beside it: a consumer across a
   * process boundary cannot `instanceof` anything (structured clone drops prototypes), so it needs a
   * name for the same taxonomy — and a name derived from the class cannot drift from it. In-process,
   * `instanceof` remains the right check and this is not used.
   */
  static readonly kind = "function";

  readonly capabilities: Capabilities;
  readonly metrics: MetricsAlgebra<M>;

  constructor(private readonly options: FunctionExecutorOptions<M>) {
    this.capabilities = options.capabilities ?? FUNCTION_CAPABILITIES;
    // The floor is the honest default for a registry of impls: a function op measures time, not money.
    // A caller whose `M` is richer supplies its own algebra, which is what keeps the dispatcher above
    // from flattening what its prompt half measured.
    this.metrics = options.metrics ?? (EXEC_METRICS_ALGEBRA as MetricsAlgebra<M>);
  }

  /** The registry this executor dispatches against, for a caller that must resolve an entry without
   *  running it (the dispatcher's `capabilitiesFor`). */
  get functions(): FunctionRegistry<ExecServices, M> {
    return this.options.functions;
  }

  /**
   * The DISPATCHED entry's capabilities (§2: required and total per variant), not this executor's
   * static record — one record standing in for a whole registry is what `capabilitiesFor` exists to
   * avoid (see {@link Executor.capabilitiesFor}).
   */
  capabilitiesFor(op: Operation<InlineFamily>): Capabilities {
    if (op.kind === "prompt") return this.capabilities;
    const fn = isResolvedCall(op) ? op.call : this.options.functions.get(op.functionRef);
    return fn ? entryCapabilities(fn) : this.capabilities;
  }

  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, M> {
    // Cancellation is checked BEFORE any work starts, on every path: an already-aborted caller must not
    // get one more call out of a dispatch that happened to be in flight.
    if (ctx.abortSignal?.aborted) return finishedHandle(canceledFailure("canceled before the operation started"));

    if (op.kind === "prompt") {
      // Reachable only by wiring this executor somewhere the dispatcher should have been. Named as the
      // wiring fault it is, rather than silently refusing as "no function registered".
      return finishedHandle(
        failure("permanent", "a prompt operation reached the function executor — dispatch by kind is OperationExecutor's job"),
      );
    }

    // A caller that resolved this op ahead of time carries the entry on it; anyone else gets the
    // lookup. Both paths exist on purpose — resolution is an optimization a holder may have done, so
    // the executor can always do it itself and never requires anyone to have done it first.
    // The PRE-RESOLVED entry (`op.call`) is typed at the floor, because a resolved call travels through
    // layers that never learn what `M` is. The registry lookup is already `M`; this asserts the other
    // path back to it rather than widening the whole executor to the floor, which is what pinned this
    // class before.
    const fn = (isResolvedCall(op) ? op.call : this.options.functions.get(op.functionRef)) as
      | RegisteredFunction<ExecServices, M>
      | undefined;
    if (!fn) {
      return finishedHandle(failure("permanent", `no function '${op.functionRef}' is registered`, this.metrics.empty()));
    }
    const resolved = resolveLiteralInputs(op);
    if ("error" in resolved) {
      return finishedHandle(failure("permanent", `function '${op.functionRef}': ${resolved.error}`));
    }

    const events = new EventQueue();
    const startMs = (ctx.clock ?? systemClock).now();
    // One controller for BOTH cancellation paths — `handle.cancel()` and the caller's `ctx.abortSignal`
    // are the same event — and the impl runs against it, so a well-behaved impl (an agent adapter, a
    // fetch) actually stops rather than being merely abandoned. `unlink` runs on settle: the caller's
    // signal is run-scoped and would otherwise accumulate one listener per operation.
    const abort = new AbortController();
    const unlink = linkAbort(abort, ctx.abortSignal);
    const work = this.runFunction(fn, resolved.values, { ...ctx, abortSignal: abort.signal }, startMs);
    const result = raceWork(work, undefined, abort.signal)
      .then((outcome) =>
        outcome.status === "done"
          ? outcome.value
          : // Abandoned, not awaited: an impl that ignores its signal must not be able to hold the
            // handle (and its caller's `cancel()`) open for as long as it feels like running.
            (canceledFailure("the operation was canceled", this.metrics.empty()) as ExecResult<ResolvedValue, M>),
      )
      .catch((e: unknown) => ({ metrics: { ...this.metrics.empty(), startMs, durationMs: 0 }, error: failureOf(e) }) as ExecResult<ResolvedValue, M>)
      .finally(() => {
        unlink();
        events.close();
      });
    return {
      events: events.iterate(),
      result,
      cancel: async () => {
        abort.abort();
        await result;
      },
    };
  }

  private async runFunction(
    fn: RegisteredFunction<ExecServices, M>,
    inputs: FunctionInputs,
    ctx: ExecServices,
    startMs: number,
  ): Promise<ExecResult<ResolvedValue, M>> {
    const clock = ctx.clock ?? systemClock;
    let produced: FunctionResult<ResolvedValue, M>;
    try {
      produced = await runFunction(fn, inputs, ctx);
    } catch (e) {
      // Belt and braces: `runFunction` already classifies a throwing impl (§4.2), so this is unreachable
      // for a registered entry. Kept so the never-rejects contract of `outcome` holds structurally,
      // not by trusting a guarantee one package away.
      return { metrics: { ...this.metrics.empty(), startMs, durationMs: clock.now() - startMs }, error: failureOf(e) };
    }
    // The impl's own report (an agent's spend) wins over our timing frame, which only knows the wall
    // clock; `startMs`/`durationMs` stay ours so they measure the dispatch, not the impl's opinion.
    const metrics: M = { ...this.metrics.empty(), ...produced.metrics, startMs, durationMs: clock.now() - startMs };
    // The CONVERSATION report rides along, on the failure path too. This result is REBUILT rather than
    // spread, so a field not named here is dropped — and the field that was being dropped is the
    // provider session id a delegated agent ended in. Losing it is silent and expensive: `withRecord`
    // stores no outcome, the next resume finds no handle, and every call opens a new remote
    // conversation while the workflow reads as though they were one.
    //
    // On failure too, because a call that failed AFTER the provider appended turns still moved the
    // remote — and not recording that is divergence on the very next call.
    const session = produced.session !== undefined ? { session: produced.session } : {};
    return isOk(produced) ? { value: produced.value, metrics, ...session } : { error: produced.error, metrics, ...session };
  }
}

/** Convenience factory mirroring the class constructor. */
export function createFunctionExecutor(options: FunctionExecutorOptions): FunctionExecutor {
  return new FunctionExecutor(options);
}
