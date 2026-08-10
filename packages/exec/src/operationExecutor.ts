/**
 * The DISPATCHING `Executor` (DESIGN §3.1). With `Operation` as the payload, dispatch is
 * exactly two cases:
 *
 *   op.kind === "prompt"    → the prompt executor (`@declarative-ai/promptop`, injected)
 *   op.kind === "function"  → the function executor (a registry lookup by `functionRef`)
 *
 * That is the whole reason `UnitKind` could be deleted: it was a third taxonomy overlapping both op
 * kinds and registry entries. And because this is an ordinary `Executor`, wrapper composition reaches
 * FUNCTION ops too — memoize, retry, and deadline previously stopped at the registry boundary.
 *
 * What this class does NOT do any more is BE either half. It used to hold the registry, resolve the
 * literals, run the entry under a linked abort controller and fold the metrics — the function
 * implementation, inlined into the dispatcher, while the prompt side was an injected object. The two
 * halves are now symmetric: {@link FunctionExecutor} and the prompt executor are both ordinary
 * `Executor`s this one holds a reference to, so either can be wrapped, subclassed or swapped, and
 * neither is privileged by the shape of the dispatcher.
 *
 * Both slots stay OPTIONAL-ish in the same way they always were: an absent prompt executor is the
 * honest answer for a graph with no LLM wired in, and it fails permanently with that reason rather
 * than pretending.
 */
import type {
  Capabilities,
  InlineFamily,
  JsonValue,
  MetricsAlgebra,
  Operation,
  Parameter,
  ResolvedValue,
  FunctionRegistry,
} from "@declarative-ai/ops";
import { isOk } from "@declarative-ai/ops";
import type { ExecHandle, ExecMetrics, ExecServices, Executor, ExecResult } from "./contract.js";
import { EXEC_METRICS_ALGEBRA } from "./contract.js";
import { FUNCTION_CAPABILITIES, FunctionExecutor } from "./functionExecutor.js";
import { hasEmbeddedOperation } from "./inputs.js";
import { carryCall } from "./resolvedOperation.js";
import { canceledFailure, failure, finishedHandle, withMetrics, wrapHandle } from "./handles.js";

export interface OperationExecutorOptions<M extends ExecMetrics = ExecMetrics> {
  /**
   * Where a `FunctionOp` goes.
   *
   * Either the registry itself — the spelling every existing caller uses, and the one this executor
   * wraps in a {@link FunctionExecutor} for you — or an already-built function executor, for a caller
   * that wants to wrap or subclass that half.
   */
  functions: FunctionRegistry<ExecServices, M> | FunctionExecutor<M>;
  /**
   * The executor `PromptOp`s dispatch to. Absent ⇒ a prompt op fails permanently with that reason,
   * which is the honest answer for a graph that has no LLM wired in. Typed as a plain `Executor` in
   * every respect but `M`, so this package never learns that `PromptOp` HAS a lowering.
   *
   * `M` is threaded so the DISPATCHER inherits what its prompt half measures. It is the half that
   * measures money — the function registry reports timing — so a dispatcher fixed at `ExecMetrics`
   * flattened every cost its own leaf produced, and its caller then had to cast the whole executor
   * back to the type it never stopped being.
   */
  prompt?: Executor<ExecServices, M>;
  /** Override the advertised capabilities (defaults to the prompt executor's, else a conservative
   *  function-only record). */
  capabilities?: Capabilities;
  /** How this executor's measurements combine. Defaults to the prompt half's, else timing/counts. */
  metrics?: MetricsAlgebra<M>;
}

export class OperationExecutor<M extends ExecMetrics = ExecMetrics> implements Executor<ExecServices, M> {
  /** The hierarchy's serializable discriminant — see {@link FunctionExecutor.kind}. */
  static readonly kind = "operation";

  readonly capabilities: Capabilities;
  readonly metrics: MetricsAlgebra<M>;
  /** The function half, built from a bare registry when the caller passed one. */
  private readonly functions: FunctionExecutor<M>;

  constructor(private readonly options: OperationExecutorOptions<M>) {
    this.functions =
      options.functions instanceof FunctionExecutor ? options.functions : new FunctionExecutor({ functions: options.functions });
    this.capabilities = options.capabilities ?? options.prompt?.capabilities ?? FUNCTION_CAPABILITIES;
    // The PROMPT half's algebra by default: it is the half that measures money, and a dispatcher that
    // reported timing-only would flatten every metric its own leaf produced.
    this.metrics = options.metrics ?? (options.prompt?.metrics as MetricsAlgebra<M> | undefined) ?? (EXEC_METRICS_ALGEBRA as MetricsAlgebra<M>);
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
    return this.functions.capabilitiesFor?.(op) ?? this.functions.capabilities;
  }

  /**
   * Run every operation EMBEDDED in this one's inputs, then dispatch it with their outputs substituted.
   *
   * A nested operation is the one unresolved binding dispatch can settle by itself: it is a whole
   * operation document, so running it needs nothing but an executor — and the executor to use is
   * THIS one, recursively, so the nested op reaches the right place by its own kind. A prompt callee
   * goes to the prompt executor and a function callee to the function executor, decided in one spot
   * rather than by each caller guessing.
   *
   * Everything else that can be unresolved belongs to the ref family and must arrive already
   * substituted — a `{refs}` tree, or a producer edge naming a declared CHILD, both of which are
   * lookups against a scope this layer cannot see. That is the whole contract at this boundary: an
   * operation handed to an executor is READY TO RUN, meaning literals and embedded operations only.
   */
  start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, M> {
    if (!hasEmbeddedOperation(op)) return this.dispatch(op, ctx);
    return wrapHandle(
      async (ctl): Promise<ExecResult<ResolvedValue, M>> => {
        if (ctl.canceled()) return canceledFailure("canceled before the operation started", this.metrics.empty());
        const input: Record<string, Parameter<InlineFamily>> = { ...op.input };
        // Nested metrics roll up rather than being discarded: an embedded operation can be the
        // expensive half of the work, and a caller reading only the outer frame would never see it.
        let nested: M | undefined;
        for (const [name, param] of Object.entries(op.input)) {
          const binding = param.binding;
          if (binding === undefined || !("op" in binding) || typeof binding.op === "string") continue;
          if (param.kind === "prompt" || param.kind === "function") continue; // higher-order: the definition IS the value
          const outcome = await ctl.started(this.start(binding.op, ctx)).result;
          nested = nested === undefined ? (outcome.metrics as M) : this.metrics.merge(nested, outcome.metrics as M);
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
        return nested === undefined ? (result as ExecResult<ResolvedValue, M>) : withMetrics(result, this.metrics.merge(nested, result.metrics as M));
      },
      { signal: ctx.abortSignal, canceledReason: "canceled while an embedded operation was in flight" },
    );
  }

  /** Dispatch a READY operation by kind — the one place `prompt` and `function` part ways. */
  /**
   * Dispatch in the CALLER'S metric algebra.
   *
   * The two halves measure differently — a prompt leaf reports money, the function registry reports
   * timing — and this frames whichever answered. The assertion is on the half that cannot be proven
   * statically: a registry entry is looked up by name, so its algebra is not known here. `metrics` is
   * the prompt half's by default precisely so that the common path is honest.
   */
  private dispatch(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, M> {
    // Cancellation is checked BEFORE any work starts, on every path: an already-aborted caller must not
    // get one more provider call out of a dispatch that happened to be in flight. Both halves check it
    // again on entry; this one covers the branch that refuses before reaching either.
    if (ctx.abortSignal?.aborted) return finishedHandle(canceledFailure("canceled before the operation started", this.metrics.empty()));

    if (op.kind === "prompt") {
      const prompt = this.options.prompt;
      if (!prompt) {
        return finishedHandle(
          failure("permanent", "this graph contains a prompt operation but no prompt executor is wired in (OperationExecutor.prompt)"),
        );
      }
      return prompt.start(op, ctx) as ExecHandle<ResolvedValue, M>;
    }
    return this.functions.start(op, ctx) as ExecHandle<ResolvedValue, M>;
  }
}

/** Convenience factory mirroring the class constructor. */
export function createOperationExecutor<M extends ExecMetrics = ExecMetrics>(options: OperationExecutorOptions<M>): Executor<ExecServices, M> {
  return new OperationExecutor(options);
}
