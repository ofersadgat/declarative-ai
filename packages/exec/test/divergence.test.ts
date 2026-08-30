/**
 * Divergence (DESIGN.md §1.6) — asserted on the LINEAGE, which is the only place it shows.
 *
 * The remote can move without us: Managed Agents compacts server-side on its own, a Claude Code
 * session can be resumed outside JaiRA. Then our mirror describes a conversation the provider no
 * longer has. The check is exact rather than heuristic — we RESUMED a handle, the call reports the one
 * it actually ran in, and on an append those must agree.
 *
 * These used to assert through an `onDivergence` callback that the tests supplied and NOTHING in
 * production did, so they passed on a channel no run ever used: `checkDivergence` computed the
 * mismatch, called a hook nobody had registered, and returned. The detection and the response both
 * live in the store now — `correctLineage` moves the record onto a branch — so these assert what the
 * lineage looks like afterwards, which is what a run actually gets.
 */
import { describe, expect, it } from "vitest";
import { createOperationExecutor, MapSessionStore, newCapabilityRegistry, runtimeFunction, RUNTIME_CAPABILITIES, withRecord, withSessionPosition } from "../src/index.js";
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

/** The session stack a host composes — the same one `wiring.ts` builds, with nothing extra. */
const stack = (store: MapSessionStore, core: Executor<ExecServices>): Executor =>
  withSessionPosition({ sessions: store }, withRecord({ records: store as never }, core as never)) as never;

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
  it("stays in one conversation while the provider stays where we left it", async () => {
    const store = new MapSessionStore();
    await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    expect(out.metrics.sessionRef).toBe("chat@2");
    expect(store.messages("chat")).toHaveLength(2);
  });

  it("branches when the handle changed under an append — server-side compaction, or an outside resume", async () => {
    const store = new MapSessionStore();
    await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-MOVED")).start(op, await positionIn(store, "chat")).result;
    expect(out.metrics.sessionRef).not.toBe("chat@2");
    expect(out.metrics.sessionRef).toContain("diverged");
  });

  /**
   * ONE answer, and it is the store's.
   *
   * This used to `resync` — mint a separate conversation re-read from the provider — and report that
   * on `sessionRef`. Meanwhile the store settling the record had already moved it onto a branch whose
   * handle is the remote actually used. Two answers to one mismatch: the record on one lineage, every
   * later reader pointed at the other, and with no read API wired the other was EMPTY.
   *
   * So the store decides where the record belongs and this reports where it landed.
   */
  it("carries on in the branch the store moved the record to, not a conversation of its own", async () => {
    const store = new MapSessionStore();
    const first = await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const second = await stack(store, agent("sess-MOVED")).start(op, await positionIn(store, "chat")).result;

    expect(second.metrics.sessionRef).not.toBe(first.metrics.sessionRef);
    // Not a rival conversation…
    expect(second.metrics.sessionRef).not.toContain("~resync");
    // …but a branch of the one it was claimed in, and the turn is IN it rather than beside it.
    expect(second.metrics.sessionRef).toContain("diverged");
    expect(store.messages(second.metrics.sessionRef!)).toHaveLength(2);
  });

  it("leaves the trunk meaning what every ref into it meant", async () => {
    const store = new MapSessionStore();
    await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    await stack(store, agent("sess-MOVED")).start(op, await positionIn(store, "chat")).result;
    // The trunk keeps only the turn that really happened in `sess-a`.
    expect(store.messages("chat")).toHaveLength(1);
  });

  it("does NOT branch a FORK again — a new handle is exactly what a fork returns", async () => {
    const store = new MapSessionStore();
    await stack(store, agent("sess-a")).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent("sess-forked")).start(op, await positionIn(store, "chat", true)).result;
    // One branch — the fork the caller asked for — and not a second one on top of it.
    expect(out.metrics.sessionRef).not.toContain("diverged");
  });

  it("stays in one conversation when no handle was resumed — a stateless provider cannot diverge", async () => {
    const store = new MapSessionStore();
    await stack(store, agent(undefined)).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, agent(undefined)).start(op, await positionIn(store, "chat")).result;
    expect(out.metrics.sessionRef).toBe("chat@2");
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

  it("still branches when the DISPATCHED call ran somewhere else", async () => {
    // The same correction, reached through the dispatcher — which is how a delegated agent gets here,
    // and the path where the outcome used to be projected away before anything could read it.
    const store = new MapSessionStore();
    await stack(store, dispatcherFor("sess-a") as never).start(op, await positionIn(store, "chat")).result;
    const out = await stack(store, dispatcherFor("sess-MOVED") as never).start(op, await positionIn(store, "chat")).result;
    expect(out.metrics.sessionRef).toContain("diverged");
  });
});
