/**
 * The dispatching `Executor` (DESIGN §3.1). With `Operation` as the payload, dispatch is
 * exactly two cases:
 *
 *   op.kind === "prompt"    → the prompt executor (`@declarative-ai/promptop`, injected)
 *   op.kind === "function"  → a registry lookup by `functionRef`
 *
 * That is the whole reason `UnitKind` could be deleted: it was a third taxonomy overlapping both op
 * kinds and registry entries. And because this is an ordinary `Executor`, wrapper composition reaches
 * FUNCTION ops too — memoize, retry, and deadline previously stopped at the registry boundary.
 *
 * This executor takes a RESOLVED op: every input `Parameter` is either free (filled by name from
 * `inputs`) or bound to a LITERAL value ref. Resolving producer edges is the family's business — hw's
 * engine walks its own scope, PENDING joins and all — so a producer edge reaching here is a wiring bug
 * and is reported as one rather than silently skipped.
 */
import type {
  Capabilities,
  FunctionInputs,
  FunctionRegistry,
  InlineFamily,
  JsonValue,
  Operation,
  Parameter,
  Ref,
  RegisteredFunction,
  ResolvedValue,
  FunctionResult,
  MetricsAlgebra,
} from "@declarative-ai/ops";
import { RUNTIME_CAPABILITIES, failureOf, isOk, runFunction } from "@declarative-ai/ops";
import type { ExecHandle, ExecMetrics, ExecServices, Executor, ExecResult } from "./contract";
import { EXEC_METRICS_ALGEBRA } from "./contract";
import { carryCall, isResolvedCall } from "./resolvedOperation";
import { EventQueue, canceledFailure, failure, finishedHandle, linkAbort, raceWork, withMetrics, wrapHandle } from "./handles";
import { systemClock } from "./deadline";

export interface OperationExecutorOptions {
  /** The one registry of discriminated entries (§2) — host code, sub-workflows, and delegated runtime
   *  adapters alike. */
  functions: FunctionRegistry<ExecServices, ExecMetrics>;
  /** The executor `PromptOp`s dispatch to. Absent ⇒ a prompt op fails permanently with that reason,
   *  which is the honest answer for a graph that has no LLM wired in. Typed as a plain `Executor`, so
   *  this package never learns that `PromptOp` HAS a lowering. */
  prompt?: Executor;
  /** Override the advertised capabilities (defaults to the prompt executor's, else a conservative
   *  function-only record). */
  capabilities?: Capabilities;
  /** How this executor's measurements combine. Defaults to timing/counts. */
  metrics?: MetricsAlgebra<ExecMetrics>;
}

/**
 * Read a resolved op's inputs. Free slots are absent (the caller fills them by name); a bound slot must
 * carry a value that is ALREADY here by the time it reaches an executor. Four binding forms are:
 *
 *  - `{text}` / `{json}` / `{blob}` — a literal leaf; the value is the binding's payload.
 *  - `{op}` with `param.kind` in `{prompt, function}` — the op DEFINITION itself is the value
 *    (higher-order: the consumer receives an op to apply, it does not run it). `model.ts` states this as
 *    the meaning of the two non-data kinds, so refusing it here made the documented higher-order form
 *    undispatchable. An embedded `Operation` only: a local child NAME is a family-scope lookup, not a
 *    definition we hold.
 *  - `{result}` — an already-RESOLVED `OperationRecord`, whose value is right there. Nothing needs to
 *    run; a failed record is an error, because there is no value to pass.
 *
 * `{op}` with a DATA kind (run the producer, use its output) and `{refs}` (resolve a tree) both stay
 * errors: walking a producer edge is the family's business — hw's engine does it against its own scope,
 * PENDING joins and all — so one arriving here is a wiring bug and is reported as one.
 */
export function resolveLiteralInputs(op: Operation<InlineFamily>): { values: FunctionInputs } | { error: string } {
  const values: FunctionInputs = {};
  for (const [name, param] of Object.entries(op.input)) {
    const binding = param.binding;
    if (binding === undefined) continue; // free slot
    const resolved = resolveBinding(name, param, binding);
    if ("error" in resolved) return resolved;
    values[name] = resolved.value;
  }
  return { values };
}

function resolveBinding(name: string, param: Parameter<InlineFamily>, binding: Ref<InlineFamily>): { value: ResolvedValue } | { error: string } {
  if ("text" in binding) return { value: binding.text };
  if ("json" in binding) return { value: binding.json };
  if ("blob" in binding) return { value: binding.blob };
  if ("result" in binding) {
    const record = binding.result;
    if (!isOk(record.result)) {
      return { error: `input '${name}' is bound to a result record that FAILED (${record.result.error.reason}) — there is no value to pass` };
    }
    return { value: record.result.value };
  }
  if ("op" in binding) {
    // The kind decides what a producer edge MEANS (model.ts, `Parameter`): a data kind runs it, the two
    // op kinds pass the definition through untouched.
    if (param.kind !== "prompt" && param.kind !== "function") {
      return {
        // Unreachable for an EMBEDDED operation — `start` runs those and substitutes their outputs
        // before this walk. What is left is the case it cannot run: a producer edge naming a declared
        // CHILD, which is a lookup in the enclosing family's scope.
        error: `input '${name}' still carries an unresolved binding — a producer edge on a '${param.kind}' parameter must be RUN and its output substituted before dispatching the operation`,
      };
    }
    if (typeof binding.op === "string") {
      return {
        error: `input '${name}' names a declared child '${binding.op}' rather than embedding its definition — resolve the local name against the enclosing scope before dispatching the operation`,
      };
    }
    // Higher-order: the value IS the op document. Structurally JSON-shaped, but `Operation` is not
    // declared as one (its `blob`/`schema` members are not `JsonValue`), so the cast is where "an op
    // definition is data" is asserted once.
    return { value: binding.op as unknown as ResolvedValue };
  }
  return {
    error: `input '${name}' is bound to a ref TREE ({refs}) — resolving a tree is the ref family's job (hw's engine walks its own scope); dispatch the operation with the tree already resolved to a value`,
  };
}

/**
 * True when any input binds a producer edge carrying a whole operation on a DATA kind — the edges
 * `start` resolves by running them. An edge on a `prompt`/`function` kind is higher-order (the
 * definition itself is the value) and a string names a declared child, so neither counts.
 *
 * The check keeps the common case free: an operation whose inputs are already literals takes the
 * same path it always did, straight to `dispatch`.
 */
export function hasEmbeddedOperation(op: Operation<InlineFamily>): boolean {
  return Object.values(op.input).some((p) => {
    const b = p.binding;
    return b !== undefined && "op" in b && typeof b.op !== "string" && p.kind !== "prompt" && p.kind !== "function";
  });
}

/** The capability floor for a registry-dispatched function op, and the fallback when a `functionRef`
 *  resolves to nothing (there is no entry whose record could be read). */
const FUNCTION_CAPABILITIES: Capabilities = { ...RUNTIME_CAPABILITIES, memoizable: true, runtime: "node" };

/**
 * A registry entry's OWN capability record, widened to the total {@link Capabilities} a wrapper gate
 * reads. `pure` and `host` variants declare a strict subset (§2 keeps each variant's record total FOR
 * THAT VARIANT), so the fields they do not name take the conservative runtime defaults — a pure function
 * neither mutates a workspace nor resumes a session, which is exactly what those defaults say.
 */
export function entryCapabilities(entry: RegisteredFunction<ExecServices, ExecMetrics>): Capabilities {
  return { ...RUNTIME_CAPABILITIES, ...entry.capabilities };
}

export class OperationExecutor implements Executor {
  readonly capabilities: Capabilities;
  readonly metrics: MetricsAlgebra<ExecMetrics>;

  constructor(private readonly options: OperationExecutorOptions) {
    this.capabilities = options.capabilities ?? options.prompt?.capabilities ?? FUNCTION_CAPABILITIES;
    this.metrics = options.metrics ?? options.prompt?.metrics ?? EXEC_METRICS_ALGEBRA;
  }

  /**
   * The DISPATCHED entry's capabilities (§2: required and total per variant) — a prompt op's from the
   * prompt executor, a function op's from its registry entry. Without this a wrapper gating on
   * capabilities reads {@link OperationExecutor.capabilities}, one static record standing in for a whole
   * registry; see {@link Executor.capabilitiesFor}.
   */
  capabilitiesFor(op: Operation<InlineFamily>): Capabilities {
    if (op.kind === "prompt") {
      const prompt = this.options.prompt;
      return prompt ? (prompt.capabilitiesFor?.(op) ?? prompt.capabilities) : this.capabilities;
    }
    const fn = isResolvedCall(op) ? op.call : this.options.functions.get(op.functionRef);
    return fn ? entryCapabilities(fn) : this.capabilities;
  }

  /**
   * Run every operation EMBEDDED in this one's inputs, then dispatch it with their outputs substituted.
   *
   * A nested operation is the one unresolved binding dispatch can settle by itself: it is a whole
   * operation document, so running it needs nothing but an executor — and the executor to use is
   * THIS one, recursively, so the nested op reaches the right place by its own kind. A prompt callee
   * goes to the prompt executor and a function callee to the registry, decided in one spot rather
   * than by each caller guessing.
   *
   * Everything else that can be unresolved belongs to the ref family and must arrive already
   * substituted — a `{refs}` tree, or a producer edge naming a declared CHILD, both of which are
   * lookups against a scope this layer cannot see. That is the whole contract at this boundary: an
   * operation handed to an executor is READY TO RUN, meaning literals and embedded operations only.
   */
  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
    if (!hasEmbeddedOperation(op)) return this.dispatch(op, ctx);
    return wrapHandle(
      async (ctl): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
        if (ctl.canceled()) return canceledFailure("canceled before the operation started");
        const input: Record<string, Parameter<InlineFamily>> = { ...op.input };
        // Nested metrics roll up rather than being discarded: an embedded operation can be the
        // expensive half of the work, and a caller reading only the outer frame would never see it.
        let nested: ExecMetrics | undefined;
        for (const [name, param] of Object.entries(op.input)) {
          const binding = param.binding;
          if (binding === undefined || !("op" in binding) || typeof binding.op === "string") continue;
          if (param.kind === "prompt" || param.kind === "function") continue; // higher-order: the definition IS the value
          const outcome = await ctl.started(this.start(binding.op, ctx)).result;
          nested = nested === undefined ? outcome.metrics : this.metrics.merge(nested, outcome.metrics);
          if (!isOk(outcome)) {
            // The nested failure travels WHOLE. Restating it as "an input failed" would drop the
            // classification, so a retriable provider error inside a nested call would arrive
            // indistinguishable from a wiring mistake.
            return { error: outcome.error, metrics: nested };
          }
          input[name] = { ...param, binding: { json: outcome.value as JsonValue } };
        }
        // `carryCall` because the spread would otherwise drop a pre-resolved entry right before the
        // dispatch that would have used it.
        const resolved = carryCall(op, { ...op, input } as Operation<InlineFamily>);
        const result = await ctl.started(this.dispatch(resolved, ctx)).result;
        return nested === undefined ? result : withMetrics(result, this.metrics.merge(nested, result.metrics));
      },
      { signal: ctx.abortSignal, canceledReason: "canceled while an embedded operation was in flight" },
    );
  }

  /** Dispatch a READY operation by kind — the one place `prompt` and `function` part ways. */
  private dispatch(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
    // Cancellation is checked BEFORE any work starts, on every path: an already-aborted caller must not
    // get one more provider call out of a dispatch that happened to be in flight.
    if (ctx.abortSignal?.aborted) return finishedHandle(canceledFailure("canceled before the operation started"));

    if (op.kind === "prompt") {
      const prompt = this.options.prompt;
      if (!prompt) {
        return finishedHandle(
          failure("permanent", "this graph contains a prompt operation but no prompt executor is wired in (OperationExecutor.prompt)"),
        );
      }
      return prompt.start(op, ctx);
    }

    // A caller that resolved this op ahead of time carries the entry on it; anyone else gets the
    // lookup. Both paths exist on purpose — resolution is an optimization a holder may have done, so
    // the executor can always do it itself and never requires anyone to have done it first.
    const fn = isResolvedCall(op) ? op.call : this.options.functions.get(op.functionRef);
    if (!fn) {
      return finishedHandle(failure("permanent", `no function '${op.functionRef}' is registered`));
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
            (canceledFailure("the operation was canceled") as ExecResult<ResolvedValue>),
      )
      .catch((e: unknown) => ({ metrics: { startMs, durationMs: 0 }, error: failureOf(e) }) as ExecResult<ResolvedValue>)
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
    fn: RegisteredFunction<ExecServices, ExecMetrics>,
    inputs: FunctionInputs,
    ctx: ExecServices,
    startMs: number,
  ): Promise<ExecResult<ResolvedValue>> {
    const clock = ctx.clock ?? systemClock;
    let produced: FunctionResult<ResolvedValue, ExecMetrics>;
    try {
      produced = await runFunction(fn, inputs, ctx);
    } catch (e) {
      // Belt and braces: `runFunction` already classifies a throwing impl (§4.2), so this is unreachable
      // for a registered entry. Kept so the never-rejects contract of `outcome` holds structurally,
      // not by trusting a guarantee one package away.
      return { metrics: { startMs, durationMs: clock.now() - startMs }, error: failureOf(e) };
    }
    // The impl's own report (an agent's spend) wins over our timing frame, which only knows the wall
    // clock; `startMs`/`durationMs` stay ours so they measure the dispatch, not the impl's opinion.
    const metrics: ExecMetrics = { ...produced.metrics, startMs, durationMs: clock.now() - startMs };
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
export function createOperationExecutor(options: OperationExecutorOptions): Executor {
  return new OperationExecutor(options);
}
