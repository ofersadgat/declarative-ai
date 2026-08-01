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
import { MapSessionStore, RUNTIME_CAPABILITIES, withRecord, withSessionPosition } from "../src";
import type { ExecServices, Executor, ExecResult, ResolvedValue } from "../src";
import { EXEC_METRICS_ALGEBRA, wrapHandle } from "../src";

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

const stack = (store: MapSessionStore, core: Executor<ExecServices>): Executor =>
  withSessionPosition({ sessions: store }, withRecord({ records: store as never }, core as never)) as never;

const op = { kind: "function", functionRef: "x", input: {}, output: { kind: "json" } } as never;

describe("divergence detection", () => {
  it("says nothing while the provider stays where we left it", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const ctx = { sessionRequest: { ref: "chat" }, onDivergence: (e: unknown) => seen.push(e) } as ExecServices;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    expect(seen).toEqual([]);
  });

  it("REPORTS a handle that changed under an append, with both ids", async () => {
    // Server-side compaction, or somebody resuming the session outside JaiRA.
    const store = new MapSessionStore();
    const seen: Array<{ resumed: string; reported: string }> = [];
    const ctx = { sessionRequest: { ref: "chat" }, onDivergence: (e: unknown) => seen.push(e as never) } as ExecServices;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    await stack(store, agent("sess-MOVED")).start(op, ctx).result;
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ resumed: "sess-a", reported: "sess-MOVED" });
  });

  it("RESYNCS onto a new conversation rather than carrying on", async () => {
    // Carrying on would keep appending against a mirror we know is wrong, and the id is a content
    // commitment — the same reasoning that makes an unresolvable id an error applies here.
    const store = new MapSessionStore();
    const ctx = { sessionRequest: { ref: "chat" } } as ExecServices;
    const first = await stack(store, agent("sess-a")).start(op, ctx).result;
    const second = await stack(store, agent("sess-MOVED")).start(op, ctx).result;
    expect(second.metrics.sessionRef).not.toBe(first.metrics.sessionRef);
    expect(second.metrics.sessionRef).toContain("~resync");
  });

  it("re-reads from the provider when the adapter offers a way to", async () => {
    const store = new MapSessionStore();
    const ctx = {
      sessionRequest: { ref: "chat" },
      sessionReader: { read: async () => [{ role: "user", content: "what the provider actually has" }] },
    } as ExecServices;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    const out = await stack(store, agent("sess-MOVED")).start(op, ctx).result;
    expect(store.messages(out.metrics.sessionRef!)).toEqual([{ role: "user", content: "what the provider actually has" }]);
  });

  it("starts the resync EMPTY when there is no read API, visibly on the edge", async () => {
    // The Messages API has none — and being stateless, cannot diverge in the first place. An adapter
    // that can diverge but cannot be read leaves an empty conversation, which the edge records rather
    // than passing off as one that happened to have nothing in it.
    const store = new MapSessionStore();
    const ctx = { sessionRequest: { ref: "chat" } } as ExecServices;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    const out = await stack(store, agent("sess-MOVED")).start(op, ctx).result;
    expect(store.messages(out.metrics.sessionRef!)).toEqual([]);
    expect(out.metrics.sessionRef).toContain("~resync");
  });

  it("survives a read API that throws — an empty resync beats a wrong mirror", async () => {
    const store = new MapSessionStore();
    const ctx = {
      sessionRequest: { ref: "chat" },
      sessionReader: {
        read: async () => {
          throw new Error("provider unreachable");
        },
      },
    } as ExecServices;
    await stack(store, agent("sess-a")).start(op, ctx).result;
    const out = await stack(store, agent("sess-MOVED")).start(op, ctx).result;
    expect(out.metrics.sessionRef).toContain("~resync");
  });

  it("does NOT call a FORK divergence — a new handle is what a native fork returns", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const onDivergence = (e: unknown): void => void seen.push(e);
    await stack(store, agent("sess-a")).start(op, { sessionRequest: { ref: "chat" }, onDivergence } as ExecServices).result;
    await stack(store, agent("sess-forked")).start(op, { sessionRequest: { ref: "chat", fork: true }, onDivergence } as ExecServices).result;
    expect(seen).toEqual([]);
  });

  it("says nothing when no handle was resumed — a stateless provider cannot diverge", async () => {
    const store = new MapSessionStore();
    const seen: unknown[] = [];
    const ctx = { sessionRequest: { ref: "chat" }, onDivergence: (e: unknown) => seen.push(e) } as ExecServices;
    await stack(store, agent(undefined)).start(op, ctx).result;
    await stack(store, agent(undefined)).start(op, ctx).result;
    expect(seen).toEqual([]);
  });
});
