/**
 * Divergence and resync (DESIGN.md §1.6).
 *
 * The remote can move without us — Managed Agents compacts server-side on its own, a Claude Code
 * session can be resumed outside JaiRA. Then our mirror describes a conversation the provider no
 * longer has, and the digest no longer describes what it will send.
 *
 * The check is exact rather than heuristic: we RESUMED a handle, the call reports the one it actually
 * ran in, and on an append those must agree.
 */
import { describe, expect, it } from "vitest";
import { createOperationExecutor, MapSessionStore, newCapabilityRegistry, runtimeFunction, RUNTIME_CAPABILITIES, withRecord, withSessionPosition , type DivergenceOptions } from "../src/index.js";
import type { ExecServices, Executor, ExecResult, ResolvedValue } from "../src/index.js";
import { EXEC_METRICS_ALGEBRA, wrapHandle } from "../src/index.js";

/** An executor that answers with a fixed conversation delta and a fixed provider handle. */
function agent(handle: string | undefined, text = "ok"): Executor<ExecServices> {
  return {
    capabilities: { ...RUNTIME_CAPABILITIES },
    metrics: EXEC_METRICS_ALGEBRA,
    start: () =>
      wrapHandle(async () => ({
        value: text as ResolvedValue,
        metrics: { durationMs: 1 },
        ...(handle !== undefined ? { session: { providerSessionId: handle, messages: [{ role: "assistant", content: text }] } } : {}),
      }) as unknown as ExecResult<ResolvedValue>),
  };
}

/**
 * Divergence deps ride on the WRAPPER now, not on `ExecServices`.
 *
 * Both were read by one private function in `record.ts` and written by whoever composed this wrapper —
 * a handoff between two adjacent layers, travelling through a bundle meant for services an executor
 * needs at arbitrary depth. Passing them here is the same information, stated where a reader can see
 * which layer consumes it.
 */
const stack = (store: MapSessionStore, core: Executor<ExecServices>, divergence: DivergenceOptions = {}): Executor =>
  withSessionPosition(
    { sessions: store, ...divergence },
    withRecord({ records: store as never }, core as never),
  ) as never;

const op = { kind: "function", functionRef: "x", input: {}, output: { kind: "json" } } as never;

/**
 * A resolved position, the way a HOST supplies one — `hw` does exactly this in `servicesFor`.
 *
 * This layer no longer looks a conversation up; it enforces what happens to one. Whoever knows which
 * conversation an operation belongs to resolves it and hands the position over.
 */
async function positionIn(store: MapSessionStore, ref: string, fork = false): Promise<ExecServices> {
  return { session: await store.resolve({ ref, ...(fork ? { fork: true } : {}) }) };
}


describe("divergence detection", () => {
  it("says nothing while the provider stays where we left it", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const onDivergence = (e: unknown): void => void seen.push(e);
    await stack(store, agent("sess-a"), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    await stack(store, agent("sess-a"), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    expect(seen).toEqual([]);
  });

  it("REPORTS a handle that changed under an append, with both ids", async () => {
    // Server-side compaction, or somebody resuming the session outside JaiRA.
    const store = new MapSessionStore();
    const seen: Array<{ resumed: string; reported: string }> = [];
    const onDivergence = (e: unknown): void => void seen.push(e as never);
    await stack(store, agent("sess-a"), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    await stack(store, agent("sess-MOVED"), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ resumed: "sess-a", reported: "sess-MOVED" });
  });

  it("RESYNCS onto a new conversation rather than carrying on", async () => {
    // Carrying on would keep appending against a mirror we know is wrong, and the id is a content
    // commitment — the same reasoning that makes an unresolvable id an error applies here.
    const store = new MapSessionStore();
    const first = await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const second = await stack(store, agent("sess-MOVED")).start(op, await positionIn(store, "chat")).result;
    expect(second.metrics.sessionRef).not.toBe(first.metrics.sessionRef);
    expect(second.metrics.sessionRef).toContain("~resync");
  });

  it("re-reads from the provider when the adapter offers a way to", async () => {
    const store = new MapSessionStore();
    const readSession = { read: async (): Promise<readonly unknown[]> => [{ role: "user", content: "what the provider actually has" }] };
    await stack(store, agent("sess-a"), { readSession }).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-MOVED"), { readSession }).start(op, await positionIn(store, "chat")).result;
    expect(store.messages(out.metrics.sessionRef!)).toEqual([{ role: "user", content: "what the provider actually has" }]);
  });

  it("starts the resync EMPTY when there is no read API, visibly on the edge", async () => {
    // The Messages API has none — and being stateless, cannot diverge in the first place. An adapter
    // that can diverge but cannot be read leaves an empty conversation, which the edge records rather
    // than passing off as one that happened to have nothing in it.
    const store = new MapSessionStore();
    await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-MOVED")).start(op, await positionIn(store, "chat")).result;
    expect(store.messages(out.metrics.sessionRef!)).toEqual([]);
    expect(out.metrics.sessionRef).toContain("~resync");
  });

  it("survives a read API that throws — an empty resync beats a wrong mirror", async () => {
    const store = new MapSessionStore();
    const readSession = {
      read: async (): Promise<readonly unknown[]> => {
        throw new Error("provider unreachable");
      },
    };
    await stack(store, agent("sess-a"), { readSession }).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-MOVED"), { readSession }).start(op, await positionIn(store, "chat")).result;
    expect(out.metrics.sessionRef).toContain("~resync");
  });

  it("does NOT call a FORK divergence — a new handle is what a native fork returns", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const onDivergence = (e: unknown): void => void seen.push(e);
    await stack(store, agent("sess-a"), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    await stack(store, agent("sess-forked"), { onDivergence }).start(op, await positionIn(store, "chat", true)).result;
    expect(seen).toEqual([]);
  });

  it("says nothing when no handle was resumed — a stateless provider cannot diverge", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const onDivergence = (e: unknown): void => void seen.push(e);
    await stack(store, agent(undefined), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    await stack(store, agent(undefined), { onDivergence }).start(op, await positionIn(store, "chat")).result;
    expect(seen).toEqual([]);
  });
});

/**
 * The same channel, through the DISPATCHER — which is how a delegated agent actually reaches it.
 *
 * `FunctionResult` is a union, so an undeclared field on it typechecks by leniency and is dropped by
 * anything that REBUILDS the result. The dispatcher rebuilds every function result, so an agent's
 * provider session id was being discarded between the adapter and the record layer: no outcome
 * stored, no handle to resume, a new remote conversation on every call, and not one error anywhere.
 */
describe("a delegated agent's session outcome survives dispatch", () => {
  const dispatcherFor = (handle: string) => {
    const registry = newCapabilityRegistry();
    // Registered under the name the shared `op` dispatches to.
    registry.functions.set(
      "x",
      runtimeFunction(
        (async () => ({
          value: "ok" as ResolvedValue,
          metrics: { durationMs: 1 },
          session: { providerSessionId: handle, messages: [{ role: "assistant", content: "ok" }] },
        })) as never,
        { ...RUNTIME_CAPABILITIES, sessionResume: true },
      ) as never,
    );
    return createOperationExecutor({ functions: registry.functions as never });
  };

  it("reaches the record layer, so the NEXT call can resume the handle", async () => {
    const store = new MapSessionStore();
    await stack(store, dispatcherFor("sess-a") as never).start(op, await positionIn(store, "chat")).result;

    let resumed: string | undefined;
    const observing: Executor<ExecServices> = {
      capabilities: { ...RUNTIME_CAPABILITIES },
      metrics: EXEC_METRICS_ALGEBRA,
      start: (_o, c) =>
        wrapHandle(async () => {
          resumed = c.session?.providerSessionId;
          return { value: "ok" as ResolvedValue, metrics: { durationMs: 1 } };
        }),
    };
    await stack(store, observing).start(op, await positionIn(store, "chat")).result;
    expect(resumed).toBe("sess-a");
  });

  it("still reports divergence when the dispatched call ran somewhere else", async () => {
    const store = new MapSessionStore();
    const seen: Array<{ resumed: string; reported: string }> = [];
    const onDivergence = (e: unknown): void => void seen.push(e as never);
    await stack(store, dispatcherFor("sess-a") as never, { onDivergence }).start(op, await positionIn(store, "chat")).result;
    await stack(store, dispatcherFor("sess-MOVED") as never, { onDivergence }).start(op, await positionIn(store, "chat")).result;
    expect(seen).toHaveLength(1);
  });
});
