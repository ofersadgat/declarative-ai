/**
 * Recording what ran — the TWO-PHASE write, as its own wrapper.
 *
 * A stub row is stamped before the call so it is visible in flight, and filled in when the call
 * settles. findmyprompt has exactly this (`MemoStore.open` / `put`), but it lives inside the memo
 * store there, and that fusion is worth undoing: "has this been computed before?" and "this call is
 * happening, record it" are different questions with different keys and different lifetimes. A cache
 * has no business knowing a call is in flight.
 *
 * Splitting it out buys three things at once:
 *
 *  - **Observability.** A record exists from the moment a call starts, so an in-flight call is
 *    visible and a crashed one leaves evidence instead of nothing. A FAILED call is recorded too —
 *    that is the point, not an edge case.
 *  - **The session append.** A session is not a separate store: it IS the records sharing a
 *    `session.id`, ordered by `seq`. Appending a turn and recording a call are the same write.
 *  - **The position reservation.** The store's uniqueness on `(session, seq)` is what makes two
 *    writers at one position impossible — durably, and across processes, which neither an in-process
 *    lock nor a read-then-write conditional manages.
 */
import type { ExecHandle, ExecMetrics, ExecServices, Executor, ExecutorWrapper, Failure, InlineFamily, Operation, ResolvedValue } from "./contract";
import { PositionTaken, isOk } from "./contract";
import { wrapHandle } from "./handles";
import { curryOrApply, isExecutor } from "./wrappers";
import { hashOperation } from "./memo";

export { PositionTaken };

/** What is known about a call BEFORE it runs — everything but the answer. */
export interface RecordStub {
  /** Content id of the operation. Stable, because operations are immutable. */
  id: string;
  /**
   * The OPERATION, not a resolved definition.
   *
   * Operations are immutable over a run: a layer that adjusts one (a budget clamping
   * `maxOutputTokens`, say) produces a NEW operation rather than mutating this one, so the op named
   * here is precisely the call that ran. Resolution that depends on the EXECUTOR rather than the
   * operation — a `defaults ← preset ← inline` merge, a router turning a route-prefixed model id
   * into a provider handle — is not in the op and belongs in a resolved-operation record, written
   * once per resolution instead of copied onto every row.
   */
  source: Operation<InlineFamily>;
  /** The conversation this call appends to, when it runs in one. */
  session?: { id: string; seq: number };
  startMs: number;
}

/** A record as stored: the stub, plus the answer once there is one. */
export interface StoredRecord<R = ResolvedValue, M extends ExecMetrics = ExecMetrics> extends RecordStub {
  result?: { value: R } | { error: Failure; value?: R };
  metrics?: M;
}

/** A failure carrying the position that was already claimed — what the session layer forks on. */
export interface PositionTakenFailure extends Failure {
  positionTaken: { session: string; seq: number };
}

/** Whether a result failed because its position was taken. */
export function isPositionTaken(result: unknown): boolean {
  const error = (result as { error?: unknown } | undefined)?.error;
  return error !== null && typeof error === "object" && "positionTaken" in (error as object);
}

/** Where records are written and read. */
export interface RecordStore<R = ResolvedValue, M extends ExecMetrics = ExecMetrics> {
  /**
   * Stamp a stub, claiming its position.
   *
   * Throws {@link PositionTaken} when the position is held. findmyprompt's `appendDraw` answers the
   * same violation by recomputing `MAX(index)` and RETRYING at the next one — correct there, because
   * a draw list's order commits to nothing. A session must never do that: appending at 15 instead of
   * 14 means continuing a conversation that contains a turn this call never saw. Draws retry,
   * sessions FORK.
   */
  open(stub: RecordStub): void | Promise<void>;
  /** Fill in a stamped record — on failure as well as success. */
  close(id: string, settled: Pick<StoredRecord<R, M>, "result" | "metrics">): void | Promise<void>;
  /** A session's records in order, up to (exclusive) `upTo`. */
  bySession?(session: string, upTo?: number): StoredRecord<R, M>[] | Promise<StoredRecord<R, M>[]>;
}

/** The ctx seam {@link withRecord} consumes. */
type RecordSeams = { records: RecordStore };

/**
 * Record every execution: stub before, fill after.
 *
 * Composed INSIDE a session layer, which resolves the position; this wrapper claims it. Composed
 * inside a memoize layer too, so a cache HIT writes no record — which is right, since nothing ran.
 */
export function withRecord<R = ExecServices, M extends ExecMetrics = ExecMetrics>(inner: Executor<R, M>): Executor<R & RecordSeams, M>;
export function withRecord<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<RecordSeams> = {}>(
  config?: P,
): ExecutorWrapper<R, R & Omit<RecordSeams, keyof P>, M>;
export function withRecord<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<RecordSeams> = {}>(
  config: P,
  inner: Executor<R, M>,
): Executor<R & Omit<RecordSeams, keyof P>, M>;
export function withRecord<R = ExecServices, M extends ExecMetrics = ExecMetrics>(
  configOrInner?: Partial<RecordSeams> | Executor<R, M>,
  maybeInner?: Executor<R, M>,
): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<RecordSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M> | undefined;
  const wrap = ((innerExec: Executor): Executor => ({
    capabilities: innerExec.capabilities,
    metrics: innerExec.metrics,
    ...forward(innerExec),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      const records = config?.records ?? ctx.records;
      if (records === undefined) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        const startMs = (ctx.clock ?? { now: () => Date.now() }).now();
        const position = ctx.session?.at;
        // A session record's identity is its POSITION, not its content. The content hash is right for
        // memoization — two identical calls are one answer — and wrong here: the same prompt asked
        // twice in one conversation is two turns, and keying them alike makes the second silently
        // overwrite the first.
        const id = position !== undefined ? `${position.id}:${position.seq}` : hashOperation(op);
        const stub: RecordStub = {
          id,
          source: op,
          ...(position !== undefined ? { session: position } : {}),
          startMs,
        };
        // Claim BEFORE the call.
        try {
          await records.open(stub);
        } catch (e) {
          if (!(e instanceof PositionTaken)) throw e;
          // Reported as a FAILURE rather than rethrown, because `wrapHandle` turns a throw into a
          // permanent failure and the reason string would be all that survived. The session layer
          // needs to recognise this one and fork, so it travels as typed data on the envelope —
          // the same way `LlmFailure` carries `rawOutput`.
          return {
            error: { classification: "permanent" as const, reason: e.message, positionTaken: { session: e.session, seq: e.seq } },
            metrics: { durationMs: 0 },
          } as never;
        }
        const result = await ctl.started(innerExec.start(op, ctx)).result;
        // Filled whichever way it went. A failed call is a record: it is evidence, it cost money, and
        // for a session its turns may already exist remotely.
        const settled: StoredRecord["result"] = isOk(result)
          ? { value: result.value }
          : { error: result.error, ...(result.value !== undefined ? { value: result.value } : {}) };
        await records.close(id, { result: settled, metrics: result.metrics });
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

/** Forward a per-op capability lookup, if the inner executor has one. Recording changes nothing. */
function forward(innerExec: Executor): { capabilitiesFor?: (op: Operation<InlineFamily>) => ReturnType<NonNullable<Executor["capabilitiesFor"]>> } {
  const perOp = innerExec.capabilitiesFor;
  return perOp ? { capabilitiesFor: (op: Operation<InlineFamily>) => perOp.call(innerExec, op) } : {};
}
