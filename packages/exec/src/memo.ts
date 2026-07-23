/**
 * Memoization (DESIGN §3.4). With `canonicalize`/`sha256Hex` at the bottom of the
 * graph in `@declarative-ai/json`, this is a few dozen dependency-free lines — which is why it folds
 * into `exec` rather than needing its own package.
 *
 * The identity component is the OPERATION's content hash. That is the whole simplification the single
 * execution seam buys: an op's `input` Parameters carry their bindings, so a RESOLVED op already
 * embeds the values a run was given. There is no `definition` + `inputs` pair to hash separately, no
 * `DefinitionOf<K>` union to discharge, and no `hashDefinition` asserting serializability on the
 * caller's behalf.
 */
import type { InlineFamily, JsonValue, Operation, ResolvedValue, Serializable } from "@declarative-ai/ops";
import { canonicalize, hashCanonical, sha256Hex } from "@declarative-ai/ops";
import type { ExecHandle, ExecMetrics, Executor, ExecutorWrapper, ExecServices, ExecResult } from "./contract";
import { forwardCapabilitiesFor } from "./contract";
import { systemClock } from "./deadline";
import type { WrapControl } from "./handles";
import { canceledFailure, finishedHandle, permanentFailure, raceWork, withMetrics, wrapHandle } from "./handles";
import { isOk } from "@declarative-ai/ops";

/**
 * The content hash of an operation. Blob leaves are replaced by the hash of their BYTES, so a memo key
 * is stable across two runs that were handed the same image by different means.
 *
 * A live STREAM cannot be hashed without draining it, so this throws with the remedy rather than
 * silently keying on object identity: materialization is the caller's decision and it is an idempotent,
 * in-place upgrade of the runtime value (DESIGN §10.1, a known limit — not yet implemented). Note this only ever applies to runtime inputs — an
 * authored document is JSON, so a workflow snapshot hash never sees a stream.
 */
export function hashOperation(op: Operation<InlineFamily>): string {
  return hashCanonical(hashableValue(op) as Serializable);
}

function hashableValue(node: unknown): unknown {
  if (node instanceof Uint8Array) return { blobHash: sha256Hex(bytesToBase64(node)) };
  if (isByteStream(node)) {
    throw new Error(
      "hashOperation: a blob input is still a live stream — materialize it to a Uint8Array before memoizing (DESIGN §10.1)",
    );
  }
  if (Array.isArray(node)) return node.map(hashableValue);
  if (node !== null && typeof node === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node)) out[k] = hashableValue(v);
    return out;
  }
  return node;
}

function isByteStream(node: unknown): boolean {
  return node !== null && typeof node === "object" && typeof (node as { getReader?: unknown }).getReader === "function";
}

function bytesToBase64(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

/**
 * The canonical memo key for one execution:
 *
 *   memoKey = sha256(canonicalize({ operationHash, workspaceTreeHash?, executorId? }))
 *
 * `workspaceTreeHash` is folded in whenever the run HAS a workspace — reading one is enough to make the
 * answer snapshot-dependent — and is REQUIRED for an executor declaring `mutatesWorkspace` (a
 * side-effecting run is only memoizable against a pinned snapshot; `withMemoize` refuses without it).
 *
 * `executorId` names WHO produced the value. The op hash says what was asked, not who answered, so two
 * executors sharing one cache — different model routing, different registries, a real one and a stub —
 * collided on identical ops and served each other's results. It is the caller's identity string;
 * `withMemoize` supplies one automatically (see {@link MemoizeOptions.namespace}).
 *
 * Nondeterminism (draw indices, retry scopes) is the caller's concern via unhashed scope tokens — this
 * key deliberately has no place for them.
 */
export function memoKey(params: { operationHash: string; workspaceTreeHash?: string; executorId?: string }): string {
  const { operationHash, workspaceTreeHash, executorId } = params;
  return sha256Hex(
    canonicalize({
      operationHash,
      ...(workspaceTreeHash !== undefined ? { workspaceTreeHash } : {}),
      ...(executorId !== undefined ? { executorId } : {}),
    }),
  );
}

/**
 * A memoization cache keyed by {@link memoKey}. Only SUCCESSFUL outcomes should be cached. Both methods
 * may be sync or async so an in-memory map or a durable store fit.
 */
export interface MemoCache {
  get(key: string): Promise<ExecResult<ResolvedValue> | undefined> | ExecResult<ResolvedValue> | undefined;
  set(key: string, outcome: ExecResult<ResolvedValue>): Promise<void> | void;
}

/**
 * A plain in-memory {@link MemoCache}.
 *
 * UNBOUNDED by default: entries hold whole result payloads and nothing evicts them, so a long-lived
 * process memoizing many distinct ops grows without limit. `maxEntries` opts into an LRU bound (a hit
 * refreshes recency; the oldest entry is dropped on overflow). It is opt-in rather than defaulted
 * because silently forgetting a result is a correctness-visible change of behavior for a caller who
 * expected a cache to be a cache — a durable store is the other answer.
 */
export class MapMemoCache implements MemoCache {
  private readonly map = new Map<string, ExecResult<ResolvedValue>>();
  constructor(private readonly maxEntries?: number) {}
  get(key: string): ExecResult<ResolvedValue> | undefined {
    const hit = this.map.get(key);
    // Re-insert so Map's insertion order IS recency order — that is what makes the eviction below LRU
    // rather than FIFO, at the cost of one delete+set per hit.
    if (hit !== undefined && this.maxEntries !== undefined) {
      this.map.delete(key);
      this.map.set(key, hit);
    }
    return hit;
  }
  set(key: string, outcome: ExecResult<ResolvedValue>): void {
    this.map.delete(key);
    this.map.set(key, outcome);
    if (this.maxEntries === undefined) return;
    while (this.map.size > this.maxEntries) {
      const oldest = this.map.keys().next();
      if (oldest.done === true) break;
      this.map.delete(oldest.value);
    }
  }
}

export interface MemoizeOptions {
  /**
   * Derive the operation's content hash. Defaults to {@link hashOperation}. An op with a cheaper or
   * more canonical identity supplies its own — e.g. a hierarchical workflow passes its snapshot hash so
   * `memoize` never brute-force-canonicalizes an opaque bundle.
   */
  identify?(op: Operation<InlineFamily>): string;
  /**
   * Who is answering — the `executorId` component of {@link memoKey}. Defaults to a per-inner-executor
   * token generated here, which is what makes a shared cache SAFE by default: two stacks over different
   * executors never collide, even on byte-identical ops.
   *
   * Supply one explicitly for a DURABLE cache. The generated token is process-local, so without it a
   * store that outlives the process can never hit — safe (recompute), but not what a durable cache is
   * for. The string should name what makes this executor's answers different: its registry, its model
   * routing, its version.
   */
  namespace?: string;
  /**
   * Whether a failing `cache.set` should FAIL the operation. Default `false`: a durable cache's write
   * rejecting (a transient store hiccup, a network blip) must NOT turn an already-successful run into a
   * permanent failure with the computed value lost — persistence is best-effort, and the caller asked
   * for the result, not for a guarantee that it was written. Set `true` only when the write landing is
   * itself part of the contract (e.g. the cache is the operation's sole durable output), so a write that
   * did not persist should surface as a hard error.
   */
  strictCacheWrites?: boolean;
}

/** Per-executor namespace tokens for the default {@link MemoizeOptions.namespace}. Keyed weakly so an
 *  executor that goes away takes its token with it; the counter only ever has to be unique in-process. */
const AUTO_NAMESPACE = new WeakMap<object, string>();
let autoNamespaceSeq = 0;
function autoNamespace(exec: Executor): string {
  let token = AUTO_NAMESPACE.get(exec);
  if (token === undefined) {
    token = `executor#${++autoNamespaceSeq}`;
    AUTO_NAMESPACE.set(exec, token);
  }
  return token;
}

/**
 * The metrics a cache HIT reports: none.
 *
 * A hit returned the stored result verbatim, so the 5 seconds and the $0.004 of the run that filled the
 * cache were re-reported for work that did not happen — and an outer retry or budget layer SUMS them
 * (that is what `MetricsAlgebra.merge` is for), inflating a run's measured cost with every hit. Every
 * numeric field is zeroed, not just the two `ExecMetrics` names: a number in a metrics record IS a
 * measurement of work, whoever declared it, and this wrapper is generic over `M` precisely because it
 * cannot know that `costUsd` means money. Non-numeric fields (a `costSource` tag) are left alone, and
 * `startMs` is re-stamped to when the hit was served.
 */
function cacheHitMetrics(metrics: ExecMetrics, nowMs: number): ExecMetrics {
  const zeroed: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(metrics)) zeroed[k] = typeof v === "number" ? 0 : v;
  return { ...(zeroed as unknown as ExecMetrics), startMs: nowMs, durationMs: 0 };
}

/**
 * Memoization: key the execution by its {@link memoKey}; on a hit return the cached outcome without
 * executing; on a miss execute and cache the result — ONLY on success (failures are never cached).
 * This is the "memoized call" as a wrapper, not a separate unit kind.
 *
 * Placed OUTERMOST so it caches the final (post-repair) result — but it REFUSES (throws at composition
 * time) to wrap a session layer: session state is not in the memo key, so a hit would replay a stale
 * answer and silently skip the transcript update. Compose the session layer OUTSIDE `withMemoize`
 * instead — sound, because that layer recomputes the sent op from the full transcript, so the memo key
 * inside sees the real content identity.
 */
export function withMemoize<R = ExecServices>(config: { cache: MemoCache } & MemoizeOptions): ExecutorWrapper<R, R>;
export function withMemoize<R = ExecServices>(config: { cache: MemoCache } & MemoizeOptions, inner: Executor<R>): Executor<R>;
export function withMemoize<R = ExecServices>(
  config: { cache: MemoCache } & MemoizeOptions,
  inner?: Executor<R>,
): ExecutorWrapper<R, R> | Executor<R> {
  const { cache, identify } = config;
  const strictCacheWrites = config.strictCacheWrites ?? false;
  const wrap = ((innerExec: Executor): Executor => {
    // The composition-time refusal survives only for an executor whose static record IS the whole
    // truth. A DISPATCHER's is not: one session-capable prompt executor behind it would otherwise make
    // every FUNCTION op in the same registry un-memoizable, refused before a single op was even looked
    // at. With a per-op record available the same refusal moves into `start`, where it can name the op
    // it actually applies to.
    if (!innerExec.capabilitiesFor && innerExec.capabilities.sessionResume) throw new Error(SESSION_REFUSAL);
    const namespace = config.namespace ?? autoNamespace(innerExec);
    /** Executions started here and not yet settled, by memo key — the dedup window between "started"
     *  and "written to the cache", which is where a fan-out of identical calls all lands. */
    const inFlight = new Map<string, Promise<ExecResult<ResolvedValue, ExecMetrics>>>();
    return {
      capabilities: innerExec.capabilities,
      metrics: innerExec.metrics,
      ...forwardCapabilitiesFor(innerExec),
      start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue> {
        // THE DISPATCHED ENTRY's record, not the dispatcher's (§2 makes each entry's capabilities
        // required and total — this is where that becomes worth something). Reading
        // `innerExec.capabilities` consulted one static record for a whole registry, so all three gates
        // below answered for an entry that was not running.
        const caps = innerExec.capabilitiesFor?.(op) ?? innerExec.capabilities;
        // Not memoizable ⇒ not memoized. A `pure` helper that reads the clock, a runtime adapter whose
        // answers are not reproducible: the entry says so, and the wrapper becomes a passthrough rather
        // than caching a value the entry told us not to reuse.
        if (!caps.memoizable) return innerExec.start(op, ctx);
        if (caps.sessionResume) return finishedHandle(permanentFailure(SESSION_REFUSAL));
        // A workspace snapshot is part of the identity whenever there IS one — not only when the
        // executor mutates it. An op that merely READS the workspace (grep, read-file, a workflow over
        // a checkout) produces a different answer per tree, so keying without the hash replays a stale
        // result after the tree changes.
        const treeHash = ctx.workspace?.treeHash;
        if (caps.mutatesWorkspace && treeHash === undefined) {
          // The one case that cannot be keyed soundly: a side-effecting run is only memoizable against
          // a PINNED snapshot, and there is none. Refuse rather than cache under a key that silently
          // means "any workspace".
          return finishedHandle(
            permanentFailure(
              "withMemoize: this operation's entry declares mutatesWorkspace, but ctx.workspace.treeHash is absent — a side-effecting run is only memoizable against a pinned workspace snapshot",
            ),
          );
        }
        const clock = ctx.clock ?? systemClock;
        return wrapHandle(
          async (ctl): Promise<ExecResult<ResolvedValue, ExecMetrics>> => {
            // INSIDE the body, not in the synchronous part of `start`. `hashOperation` throws by design
            // on a live-stream blob and `identify` is caller-supplied, so computing the key out there
            // threw straight out of the caller's `start(...)` call — breaking the never-throws seam that
            // all of `handles.ts` exists to hold, through a documented (§7.3) input shape.
            const key = memoKey({
              operationHash: identify ? identify(op) : hashOperation(op),
              ...(treeHash !== undefined ? { workspaceTreeHash: treeHash } : {}),
              executorId: namespace,
            });
            const hit = await cache.get(key);
            if (hit) return withMetrics(hit, cacheHitMetrics(hit.metrics, clock.now()));
            // IN-FLIGHT dedup. The cache is written on COMPLETION, so without this the fan-out case
            // memoization exists for — N identical calls issued together — misses N times and executes
            // N times, every one of them paid for. Followers share the leader's promise and, like a
            // cache hit, report no work: the leader is the one that did it.
            //
            // Cancellation stays per-caller in one direction only: a follower that cancels stops WAITING
            // and leaves the leader running for everyone else, but a follower cannot outlive a leader
            // that is canceled — it receives that canceled result, uncached, and its own next call
            // re-runs. Deduping means sharing the execution, and a shared execution has one fate.
            const pending = inFlight.get(key);
            if (pending) {
              const shared = await raceCancellation(pending, ctl);
              if (shared === undefined) return canceledFailure("canceled while waiting on an identical in-flight call");
              return isOk(shared) ? withMetrics(shared, cacheHitMetrics(shared.metrics, clock.now())) : shared;
            }
            if (ctl.canceled()) return canceledFailure("canceled before the call started");
            const run = ctl.started(innerExec.start(op, ctx)).result;
            inFlight.set(key, run);
            try {
              const result = await run;
              if (isOk(result)) {
                if (strictCacheWrites) {
                  await cache.set(key, result);
                } else {
                  // Best-effort persistence: a rejecting durable `set` must not discard an
                  // already-successful result. Swallow it — `strictCacheWrites` opts into surfacing it.
                  try {
                    await cache.set(key, result);
                  } catch {
                    /* cache write failed; the value is still returned to the caller */
                  }
                }
              }
              return result;
            } finally {
              inFlight.delete(key);
            }
          },
          { signal: ctx.abortSignal },
        );
      },
    };
  }) as unknown as ExecutorWrapper<R, R>;
  return inner ? wrap(inner) : wrap;
}

/** Why `withMemoize` will not sit over a session layer — one string, raised at composition time for a
 *  leaf executor and per-op for a dispatcher. */
const SESSION_REFUSAL =
  "withMemoize must not wrap a session layer: session state is not part of the memo key, so a hit would replay a stale answer and skip the transcript update — compose withSession OUTSIDE withMemoize";

/** Await a shared in-flight execution, but give up the moment THIS caller cancels. A follower must not
 *  be held by the leader's call — its cancel is its own — so `undefined` means "canceled, stop waiting",
 *  and the leader keeps running for everyone else. */
async function raceCancellation(
  pending: Promise<ExecResult<ResolvedValue, ExecMetrics>>,
  ctl: WrapControl,
): Promise<ExecResult<ResolvedValue, ExecMetrics> | undefined> {
  const raced = await raceWork(pending, undefined, ctl.signal);
  return raced.status === "done" ? raced.value : undefined;
}

/** Re-exported so a consumer keying its own cache does not have to reach past `exec`. */
export type { JsonValue };
