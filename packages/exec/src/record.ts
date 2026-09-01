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
} from "./contract.js";
import type { OperationScope } from "./contract.js";
import { PositionTaken, isOk } from "./contract.js";
import { canceledFailure, wrapHandle } from "./handles.js";
import { curryOrApply, isExecutor } from "./wrappers.js";
import { hashOperation } from "./memo.js";
import { canonicalize, sha256Hex } from "@declarative-ai/ops";

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

/**
 * The row {@link RecordStore.append} wrote, named by its id — which IS a key now.
 *
 * There used to be an `attempt` half, because the id was a bare content hash and a loop dispatching
 * the identical operation twice collided. The id is the hash of the SCOPED request today (see
 * {@link scopedOperationId}): the dispatch site inside it never repeats, so the id alone names the
 * row, and both layers can compute it independently — no key ever has to be handed across a
 * boundary to be agreed on.
 */
export interface RecordRef {
  id: string;
}

/** A flush into a record that is still open — what a call has produced so far. */
export interface RecordPartial {
  value: unknown;
  /** The provider handle, stamped as soon as the stream carries one: a crashed call with no handle
   *  can be neither resumed nor resynced, and waiting for the settle is why it used to have none. */
  providerSessionId?: string;
}

/**
 * Where records are written and read — one row per call, across three moments of its life.
 *
 * `append` → `update`* → `finish`. The verbs are the lifecycle, and the split is not bookkeeping:
 * the row has to exist BEFORE the call so the claim can refuse a taken position without spending
 * anything, so a killed process leaves evidence rather than silence, and so a stream has somewhere
 * to land.
 */
export interface RecordStore<R = ResolvedValue, M extends ExecMetrics = ExecMetrics> {
  /**
   * Write the row and claim its position.
   *
   * Throws {@link PositionTaken} when the position is held. findmyprompt's `appendDraw` answers the
   * same violation by recomputing `MAX(index)` and RETRYING at the next one — correct there, because
   * a draw list's order commits to nothing. A session must never do that: appending at 15 instead of
   * 14 means continuing a conversation that contains a turn this call never saw. Draws retry,
   * sessions FORK.
   */
  append(stub: RecordStub): RecordRef | Promise<RecordRef>;
  /**
   * Flush what the call has produced so far into the open row.
   *
   * By REF, like the other two. It took a position at first, on the reasoning that a partial only ever
   * exists for a placed call and `(session_id, seq)` is a primary key — true, and beside the point:
   * a record's LINEAGE can change while it is still streaming. When the handle coming back says this
   * call is not in the conversation we assumed, the record moves to a branch, and every flush after
   * that would address a position that no longer holds it. A ref survives the move, because a
   * record's identity does not travel with its lineage.
   *
   * Optional, like {@link RecordStore.bySession}: absent MEANS this store cannot hold a partial, which
   * an in-memory one genuinely cannot — there is no crash for it to survive.
   */
  update?(ref: RecordRef, partial: RecordPartial): void | Promise<void>;
  /**
   * Fill in a stamped record — on failure as well as success.
   *
   * Takes the ref {@link RecordStore.append} handed back. The id names exactly one row now (the
   * scope inside it never repeats), so the "which of two identical open rows" guessing the old
   * attempt-keyed settle had to do cannot recur — but the ref stays the interface, because it is
   * the store's acknowledgement that THIS row was written, not the caller's recomputation of it.
   */
  finish(ref: RecordRef, settled: Pick<StoredRecord<R, M>, "result" | "metrics" | "sessionOutcome">): void | Promise<void>;
  /**
   * Where a record sits NOW — which may not be where it was claimed.
   *
   * A store that corrects an append it was wrong about moves the record to a branch, and the caller
   * that claimed the position is the one thing that has to be told: what it reports as the call's
   * ending position is what everything downstream continues from. Optional, like the reads above —
   * absent MEANS this store never moves a record, so the claimed position is still the answer.
   */
  positionOf?(ref: RecordRef): { id: string; seq: number } | undefined | Promise<{ id: string; seq: number } | undefined>;
  /** A session's records in order, up to (exclusive) `upTo`. */
  bySession?(session: string, upTo?: number): StoredRecord<R, M>[] | Promise<StoredRecord<R, M>[]>;
}

/**
 * The identity of one ask made at one place — the record id.
 *
 * A content hash alone repeats whenever the same operation is dispatched twice; the scope inside
 * this fold never does, because a dispatch site `(instanceId, sequence)` belongs to exactly one
 * instance and instances are named once. Derived from unique parts IS unique, which is what lets
 * the id be a primary key with no `attempt` beside it.
 *
 * A FOLD over the op's hash rather than a hash of an op-with-scope-inside, deliberately: the op's
 * own hash is a SHARED identity — the memo key, the call cache key, "would someone else making this
 * identical call reuse the answer?" — and it must keep answering that question unscoped. The two
 * identities share the content half and differ by the scope, and this function is the only place
 * that relationship is written down.
 */
export function scopedOperationId(operationHash: string, scope: OperationScope): string {
  return sha256Hex(canonicalize({ operationHash, instanceId: scope.instanceId, sequence: scope.sequence }));
}

/**
 * The id a record is keyed by — {@link hashOperation} folded with the dispatch site, made TOTAL.
 *
 * `hashOperation` deliberately throws on an op carrying a LIVE byte stream (a single-consumer blob
 * kept un-materialized for piping, DESIGN §10.1): such an op has no stable content identity, and a
 * memo must refuse it. A RECORD must not — recording is unconditional, and a throw here would fail
 * the very call it was meant to witness. An unhashable op with a scope is still uniquely named BY
 * the scope, which is the honest identity of a stream: this ask, made here.
 *
 * A dispatch with NO scope gets a process-local ordinal as its site. Uniqueness is the invariant
 * the whole schema leans on — the id is a primary key, and two identical prompts appended to one
 * conversation used to be exactly the collision `attempt` papered over — and a caller that supplies
 * no scope is a caller that never re-computes the id either, so nothing is lost by the ordinal
 * being local. Every real dispatcher (the engine, a chat host) supplies its own.
 */
let localSite = 0;
function contentIdOf(op: Operation<InlineFamily>, scope: OperationScope | undefined): string {
  const at = scope ?? { instanceId: "local", sequence: ++localSite };
  try {
    return scopedOperationId(hashOperation(op), at);
  } catch {
    return scopedOperationId("unhashable", at);
  }
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
      // CONSTRUCTION only. It used to fall back to `ctx.records`, which was a second way to say the
      // same thing: a caller composing this wrapper necessarily has the store in hand, and nothing
      // else in any package ever read the ctx field. Two channels for one dependency is how a bundle
      // meant to carry SERVICES turns into a bag of whatever a wrapper felt like passing sideways.
      const records = config?.records;
      if (records === undefined) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        const startMs = (ctx.clock ?? { now: () => Date.now() }).now();
        const position = ctx.session?.at;
        /**
         * The SCOPED hash, whether or not this call is PLACED.
         *
         * It used to be `<sessionId>:<seq>` for a placed record, which spelled the position into the
         * id — and migration 8 pointed out what that cost: "for a PLACED record `record_id` is
         * literally `session_id:seq`, so those rows were carrying a rowid that duplicated the very
         * pair the position table already keys on". The claim, the ordering and the conflict all live
         * on the position table; the id was never what detected anything.
         *
         * The old objection — that two identical prompts dispatched twice would collide — used to be
         * answered by `attempt`. It is answered by the SCOPE now: the dispatch site inside the fold
         * never repeats, so the id alone is the key. See {@link scopedOperationId}.
         */
        const id = contentIdOf(op, ctx.scope);
        const stub: RecordStub = {
          id,
          source: op,
          ...(position !== undefined ? { session: position } : {}),
          startMs,
        };
        // Claim BEFORE the call, and keep the ref it hands back — that, not the stub's id, is what
        // names the row this call owns for the rest of its life.
        let ref: RecordRef;
        try {
          ref = await records.append(stub);
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
        // THE DISPATCH SEAM, in its load-bearing order: insert the row (above), invoke the callback,
        // make the provider call (below). A journal written from here can never name a row that does
        // not exist, and a crash between the insert and the call leaves an open row for the recovery
        // sweep rather than an event pointing at nothing. Fired at THIS layer — the innermost, past
        // every wrapper that could still change the op — so completeness does not depend on how the
        // executor stack happens to be composed.
        ctx.onDispatch?.({ id: ref.id });
        // ASK for the payload. This is the point of the flag: the recorder is the layer that needs
        // what the executor would otherwise project away inside the call, and the only layer that
        // knows a record is about to be written. Requesting it per call beats an executor built in a
        // mode that answers differently forever — which is what `record: true` was, and which could
        // not be typed, since it swapped an `Out` that every agent subclass had already pinned.
        const result = await ctl.started(innerExec.start(op, { ...ctx, returnRecord: true })).result;
        // The PAYLOAD when one was reported, else the value. A record says what the call PRODUCED, and
        // for a prompt op that is the whole `LlmOutput` — messages, reasoning, tool trace — not the one
        // field the op happened to declare as its output.
        const produced = (result as { record?: unknown }).record;
        const recorded = (produced !== undefined ? produced : result.value) as ResolvedValue | undefined;
        // Filled whichever way it went. A failed call is a record: it is evidence, it cost money, and
        // for a session its turns may already exist remotely.
        const settled: StoredRecord["result"] = isOk(result)
          ? { value: recorded as ResolvedValue }
          : { error: result.error, ...(recorded !== undefined ? { value: recorded } : {}) };
        // The session outcome rides along when the executor reported one — the handle and the delta,
        // which every recorded call needs, as against the payload above, which only a persisting
        // caller does.
        const session = sessionOutcomeOf(result);
        await records.finish(ref, { result: settled, metrics: result.metrics, ...(session !== undefined ? { sessionOutcome: session } : {}) });
        // WHERE IT LANDED, when that is not where it was claimed. A store settling this record may
        // have found that the call did not run in the conversation the position belonged to — a remote
        // that compacted itself, an adapter that branched, a different provider — and moved it onto a
        // branch. The claimed position is then the wrong answer to "where does this conversation
        // continue", and this is the only layer holding the ref needed to ask.
        const landed = await records.positionOf?.(ref);
        if (landed !== undefined && (landed.id !== position?.id || landed.seq !== position.seq)) {
          return { ...result, metrics: { ...result.metrics, sessionRef: `${landed.id}@${landed.seq + 1}` } } as typeof result;
        }
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
      // CONSTRUCTION only: the store is what this wrapper FORKS with, and a caller composing it has
      // the store in hand. It was never something the executor below needed to make a call.
      const sessions = config?.sessions;
      // ALREADY RESOLVED — this layer no longer looks a conversation up, it only enforces what happens
      // to one. Whoever knows which conversation an operation belongs to resolves it and hands the
      // POSITION over: `hw` in `servicesFor`, `withSession` from an op's declaration. That is why
      // `sessionRequest` is gone from `ExecServices` — a request is not something a prompt call
      // consumes, and a bundle of services is for what the executor needs to make the call.
      const resolved = ctx.session;
      // No conversation in play, or nowhere to fork into ⇒ nothing to do. Silence is right here: a run
      // without sessions wired is an ordinary run, not a misconfiguration.
      if (resolved === undefined || sessions === undefined) return innerExec.start(op, ctx);
      return wrapHandle(async (ctl) => {
        const attempt = async (session: ResolvedSession): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
          const result = await ctl.started(innerExec.start(op, { ...ctx, session })).result;
          // The END position, and the EFFECTIVE one: a call is one record, and a call that had to fork
          // ended somewhere the caller has no other way to learn.
          // The position it was CLAIMED at — unless something below already said where it actually
          // landed, which a store that corrects a mistaken append does.
          const reported = (result.metrics as { sessionRef?: string } | undefined)?.sessionRef;
          return { ...result, metrics: { ...result.metrics, sessionRef: reported ?? `${session.at.id}@${session.at.seq + 1}` } };
        };
        if (ctl.canceled()) return canceledFailure("canceled before the call started");
        const first = await attempt(resolved);
        if (isPositionTaken(first)) {
          // FORK, not retry-at-the-next-slot. Something already claimed this position, so continuing
          // here would mean continuing a conversation containing a turn this call never saw.
          // Forked from the RESOLUTION, not from the request. The seed rides on `ResolvedSession` now,
          // which is what lets a caller hand this layer a position it already resolved — the request
          // no longer has to survive alongside the resolution just so a fork can name itself.
          const forked = await sessions.fork(resolved.id, resolved.seed);
          const second = await attempt(await sessions.resolve({ ref: forked, ...(resolved.seed !== undefined ? { seed: resolved.seed } : {}) }));
          return second;
        }
        return first;
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
