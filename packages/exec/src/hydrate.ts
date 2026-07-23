/**
 * The FAMILY-TRANSITION wrapper: run a stack of inline-op wrappers against ops of another family.
 *
 * A ref family with id leaves has the opposite cost profile from the inline family: IDENTITY is cheap
 * (the op is small — its leaves are content ids — and may even carry its own hash) while CONTENT is
 * expensive (every leaf is a store read). The layers that need only identity — memoization above all —
 * should therefore run BEFORE the content exists, and the layers that read content (pricing a prompt,
 * repairing a user text, the leaf itself) after. This wrapper is that boundary:
 *
 *     compose(leaf)                       // inline ops: reads content
 *       .with(withRateLimit(...))         // inline: estimates off the prompt text
 *       .with(withBudget(...))            // inline: prices the resolved call
 *       .with(withRetry(...))             // inline: may rewrite the user text
 *       .with(withHydration(resolve))     // ← id ops above, inline ops below
 *       .with(withMemoize({ cache, identify: (op) => op.id }))   // id: keys on the content id, free
 *       .with(withBudget({ meter, computeCost }))                // id: bills memo reuse
 *
 * On a memo HIT nothing below the transition runs — the whole point: the expensive store reads happen
 * only on a miss. `resolve` is the family's hydrator (DESIGN: "hydration is the family's business"):
 * it loads leaf artifacts, folds structural sharing, runs producer edges — whatever turns its op into
 * the RESOLVED inline op the dispatcher demands.
 *
 * A hydration failure is a classified failure like any other (a missing artifact is `permanent`, a
 * store timeout `network-retriable` — `failureOf` reads it off the error), returned through the
 * never-throws handle with floor metrics.
 */
import type { Capabilities, InlineFamily, Operation, ResolvedValue } from "@declarative-ai/ops";
import { failureOf } from "@declarative-ai/ops";
import type { ExecHandle, ExecMetrics, ExecServices, Executor } from "./contract";
import { systemClock } from "./deadline";
import { wrapHandle } from "./handles";

/** How a family turns ITS op into the resolved inline op an executor runs — store reads, `$base`
 *  folding, producer edges. Receives the ctx so a hydrator that needs services (a session's workspace,
 *  an abort signal for its reads) has them. */
export type Hydrator<Op, R = ExecServices> = (op: Op, ctx: R) => Operation<InlineFamily> | Promise<Operation<InlineFamily>>;

export interface HydrationOptions<Op> {
  /**
   * The per-op capability record for the FAMILY's ops, so a gate above the transition (a memoize
   * deciding memoizability, a policy layer) still reads the dispatched entry's record without
   * hydrating. Absent ⇒ the inner executor's static record stands in for every op — sound but coarse,
   * exactly the degradation {@link Executor.capabilitiesFor} documents.
   */
  capabilitiesFor?: (op: Op) => Capabilities;
}

/**
 * Build the transition. NOT an `ExecutorWrapper` — it changes the op type the stack accepts, which is
 * the entire point — so it composes via the same `.with(...)` as everything else but yields a stack
 * whose `start` takes the FAMILY's ops.
 */
export function withHydration<Op, R = ExecServices, M extends ExecMetrics = ExecMetrics, Out = ResolvedValue>(
  resolve: Hydrator<Op, R>,
  options: HydrationOptions<Op> = {},
): (inner: Executor<R, M, Operation<InlineFamily>, Out>) => Executor<R, M, Op, Out> {
  return (inner) => ({
    capabilities: inner.capabilities,
    metrics: inner.metrics,
    ...(options.capabilitiesFor ? { capabilitiesFor: options.capabilitiesFor } : {}),
    start(op: Op, ctx: R): ExecHandle<Out, M> {
      const services = ctx as ExecServices;
      const clock = services.clock ?? systemClock;
      // The body speaks `ResolvedValue` (wrapHandle's floor) and passes the inner value through
      // untouched, so the outward `Out` is asserted once here rather than threaded through the
      // cancellation scaffolding.
      return wrapHandle<M>(
        async (ctl) => {
          const startMs = clock.now();
          let inline: Operation<InlineFamily>;
          try {
            inline = await resolve(op, ctx);
          } catch (e) {
            // Floor metrics only: hydration spent no model tokens and no money, and a generic wrapper
            // cannot fabricate a richer `M`'s required fields — the same convention every wrapper-level
            // refusal in this package follows.
            return { error: failureOf(e, "hydration"), metrics: { startMs, durationMs: clock.now() - startMs } as ExecMetrics as M };
          }
          if (ctl.canceled()) {
            return { error: { classification: "canceled", reason: "canceled during hydration" }, metrics: { startMs, durationMs: clock.now() - startMs } as ExecMetrics as M };
          }
          return ctl.started(inner.start(inline, ctx) as unknown as ExecHandle<ResolvedValue, M>).result;
        },
        { signal: services.abortSignal },
      ) as unknown as ExecHandle<Out, M>;
    },
  });
}
