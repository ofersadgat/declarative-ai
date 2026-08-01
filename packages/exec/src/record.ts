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
import type {
  ExecHandle,
  ExecMetrics,
  ExecResult,
  ExecServices,
  Executor,
  ExecutorWrapper,
  Failure,
  InlineFamily,
  Operation,
  ResolvedSession,
  ResolvedValue,
  SessionRequest,
  SessionStore,
} from "./contract";
import { PositionTaken, isOk } from "./contract";
import { canceledFailure, wrapHandle } from "./handles";
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

/**
 * What an execution reports about the conversation it ran in.
 *
 * A PROMPT op needs none of this: its payload is an `LlmOutput`, which already carries the messages,
 * so the record reads them straight off the result. A DELEGATED agent does need it — it answers with
 * text, keeps its transcript server-side, and the one thing it must hand back is the session id the
 * run ENDED in, because a native fork returns a NEW one and losing it puts two branches into a single
 * remote session.
 *
 * Declared rather than left to structural luck: `FunctionResult` is a union, so an undeclared extra
 * field typechecks by leniency and would be dropped by anything that rebuilt the result.
 */
export interface SessionOutcome {
  /** The provider's own session id as of this run. A CHANGE from what we resumed is divergence. */
  providerSessionId?: string;
  /** What the run added, for an executor whose payload is not already a conversation. */
  messages?: readonly unknown[];
}

/** Read the session outcome an execution reported, if it reported one. */
export function sessionOutcomeOf(result: unknown): SessionOutcome | undefined {
  const session = (result as { session?: unknown } | null | undefined)?.session;
  return session !== null && typeof session === "object" ? (session as SessionOutcome) : undefined;
}

/** A record as stored: the stub, plus the answer once there is one. */
export interface StoredRecord<R = ResolvedValue, M extends ExecMetrics = ExecMetrics> extends RecordStub {
  result?: { value: R } | { error: Failure; value?: R };
  metrics?: M;
  /**
   * What the execution SAID about its conversation — see {@link SessionOutcome}.
   *
   * Deliberately not called `session`: on the stub that name means the position this record CLAIMS,
   * which is decided before the call. This is what came back.
   */
  sessionOutcome?: SessionOutcome;
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
  close(id: string, settled: Pick<StoredRecord<R, M>, "result" | "metrics" | "sessionOutcome">): void | Promise<void>;
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
        // The session outcome rides along when the executor reported one. A prompt op reports none
        // and needs none — its payload IS the conversation — so this is empty on the common path.
        const session = sessionOutcomeOf(result);
        await records.close(id, { result: settled, metrics: result.metrics, ...(session !== undefined ? { sessionOutcome: session } : {}) });
        return result;
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

/** The ctx seam {@link withSessionPosition} consumes. */
type PositionSeams = { sessions: SessionStore };

/**
 * Resolve the conversation a call runs in, and fork when its position turns out to be taken.
 *
 * This is the POLICY half of the session split, and it lives in `exec` rather than in the llm layer
 * because a delegated agent needs it just as much as a prompt op does — and `hw`, which is where the
 * request comes from, cannot import `promptop`. `withSession` in promptop is this plus the two things
 * only that layer knows: reading a session out of an op's config, and projecting an `LlmOutput` down
 * to the op's output value on the way back.
 *
 * The caller states a REQUEST (`ctx.sessionRequest`) — which conversation, and whether to branch —
 * and this resolves it to a position and puts it on `ctx.session`. Resolution has to happen here
 * rather than at the requester because only the store knows where a conversation currently is, and
 * `hw` deliberately does not.
 */
export function withSessionPosition<R = ExecServices, M extends ExecMetrics = ExecMetrics>(inner: Executor<R, M>): Executor<R & PositionSeams, M>;
export function withSessionPosition<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<PositionSeams> = {}>(
  config?: P,
): ExecutorWrapper<R, R & Omit<PositionSeams, keyof P>, M>;
export function withSessionPosition<R = ExecServices, M extends ExecMetrics = ExecMetrics, P extends Partial<PositionSeams> = {}>(
  config: P,
  inner: Executor<R, M>,
): Executor<R & Omit<PositionSeams, keyof P>, M>;
export function withSessionPosition<R = ExecServices, M extends ExecMetrics = ExecMetrics>(
  configOrInner?: Partial<PositionSeams> | Executor<R, M>,
  maybeInner?: Executor<R, M>,
): ExecutorWrapper<R, R, M> | Executor<R, M> {
  const config = (isExecutor(configOrInner) ? undefined : configOrInner) as Partial<PositionSeams> | undefined;
  const inner = (isExecutor(configOrInner) ? configOrInner : maybeInner) as Executor<R, M> | undefined;
  const wrap = ((innerExec: Executor): Executor => ({
    // A session layer resumes state, so a `withMemoize` above must refuse to cache — and the per-op
    // record has to say so too, or a memoize checking per-op caps would read the inner entry's record,
    // which knows nothing about the session, and cache the call.
    capabilities: { ...innerExec.capabilities, sessionResume: true },
    metrics: innerExec.metrics,
    capabilitiesFor: (op: Operation<InlineFamily>) => ({
      ...(innerExec.capabilitiesFor?.(op) ?? innerExec.capabilities),
      sessionResume: true,
    }),
    start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
      const sessions = config?.sessions ?? ctx.sessions;
      const request = ctx.sessionRequest;
      // No conversation asked for, or nowhere to keep one ⇒ nothing to do. Silence is right here: a
      // run without sessions wired is an ordinary run, not a misconfiguration.
      if (request === undefined || sessions === undefined) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        const attempt = async (session: ResolvedSession): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
          const result = await ctl.started(innerExec.start(op, { ...ctx, session })).result;
          // The END position, and the EFFECTIVE one: a call is one record, and a call that had to fork
          // ended somewhere the caller has no other way to learn.
          return { ...result, metrics: { ...result.metrics, sessionRef: `${session.at.id}@${session.at.seq + 1}` } };
        };
        if (ctl.canceled()) return canceledFailure("canceled before the call started");
        const resolved = await sessions.resolve(request);
        const first = await attempt(resolved);
        if (isPositionTaken(first)) {
          // FORK, not retry-at-the-next-slot. Something already claimed this position, so continuing
          // here would mean continuing a conversation containing a turn this call never saw.
          const forked = await sessions.fork(resolved.id, request.seed);
          const second = await attempt(await sessions.resolve({ ...request, ref: forked, fork: false }));
          return await checkDivergence(sessions, resolved, second, ctx);
        }
        return await checkDivergence(sessions, resolved, first, ctx);
      });
    },
  })) as unknown as ExecutorWrapper<R, R, M>;
  return curryOrApply(wrap, inner);
}

/**
 * Notice that the remote moved underneath us, and answer it with a `resync` (DESIGN.md §1.6).
 *
 * The check is cheap and exact: we RESUMED a handle, the call reports the handle it actually ended
 * in, and on an append those must agree. When they do not, the provider's conversation is no longer
 * the one our mirror describes — server-side compaction did it (Managed Agents does this on its own),
 * or somebody resumed the session outside JaiRA.
 *
 * VERIFY ON APPEND rather than trusting: a stale mirror is silent, and the next call would replay a
 * digest that no longer describes what the provider will send.
 *
 * Answering it is deliberately not "carry on". The id is a content commitment, and the same reasoning
 * that makes an unresolvable id an error applies here — so this LOGS, then starts a new conversation
 * with a `resync` edge whose contents are re-read from the provider. Where the adapter has no read
 * API, the new conversation starts EMPTY, and that emptiness is visible on the edge rather than being
 * mistaken for a conversation that happened to have nothing in it.
 *
 * A FORK is exempt: a new handle is exactly what a native fork returns, and calling that divergence
 * would resync on every branch.
 */
async function checkDivergence(
  sessions: SessionStore,
  resolved: ResolvedSession,
  result: ExecResult<ResolvedValue, ExecMetrics>,
  ctx: ExecServices,
): Promise<ExecResult<ResolvedValue, ExecMetrics>> {
  const resumed = resolved.providerSessionId;
  const reported = sessionOutcomeOf(result)?.providerSessionId;
  if (resolved.mode !== "append" || resumed === undefined || reported === undefined || reported === resumed) return result;

  const reason = `session ${resolved.id} diverged: resumed provider session ${resumed}, but the call ran in ${reported}`;
  ctx.onDivergence?.({ session: resolved.id, resumed, reported, reason });

  if (sessions.resync === undefined) return result;
  // Re-read from the provider when it offers a way to. `read` is per-adapter and optional — the
  // Messages API has none and, being stateless, cannot diverge in the first place.
  let contents: readonly unknown[] = [];
  try {
    contents = (await ctx.sessionReader?.read(reported)) ?? [];
  } catch {
    // A failed re-read is still a resync, just an empty one. Losing the conversation is bad; carrying
    // on against a mirror we know is wrong is worse.
    contents = [];
  }
  const resynced = await sessions.resync(resolved.id, contents as never);
  // The outcome points at the RESYNCED conversation, so whatever continues from here continues from
  // what the provider actually has rather than from what we thought it had.
  return { ...result, metrics: { ...result.metrics, sessionRef: resynced } };
}

/** Forward a per-op capability lookup, if the inner executor has one. Recording changes nothing. */
function forward(innerExec: Executor): { capabilitiesFor?: (op: Operation<InlineFamily>) => ReturnType<NonNullable<Executor["capabilitiesFor"]>> } {
  const perOp = innerExec.capabilitiesFor;
  return perOp ? { capabilitiesFor: (op: Operation<InlineFamily>) => perOp.call(innerExec, op) } : {};
}
