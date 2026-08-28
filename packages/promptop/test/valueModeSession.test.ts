/**
 * A VALUE-MODE core keeps the conversation (SESSIONS.md §7).
 *
 * ## The gap this closes
 *
 * Every session test in this package composes {@link withSession}, which projects an `LlmOutput` down
 * to the op's output value on its way out over a RECORD-MODE core. That path was well covered and it
 * works.
 *
 * It is also not the path anything actually runs. `withSession` reads its request from `op.config`,
 * and `hw` resolves the conversation itself, handing the executor a POSITION. So a workflow host
 * composes exec's `withSessionPosition` + `withRecord` around a core left in its DEFAULT value mode,
 * and in that composition the projection happens inside the call: `withRecord` stored the answer and
 * the messages were gone.
 *
 * The result was silent and total. Real runs recorded a position holding `{"answer":"4"}` and no
 * conversation; `defaultMessagesOf` read `result.value.messages`, found nothing, and every transcript
 * came back empty — while the suite stayed green, because a scripted fake reports its delta on the
 * session channel and a `withSession` test projects from a payload that still had one.
 *
 * So these tests compose the stack a HOST composes, and assert on what the store ended up holding.
 */
import { describe, expect, it } from "vitest";
import {
  MapSessionStore,
  defaultMessagesOf,
  withRecord,
  withSessionPosition,
  type ExecServices,
  type Executor,
  type SessionStore,
} from "@declarative-ai/exec";
import type { ModelMessage } from "ai";
import { thinkingOfEntries } from "@declarative-ai/llm";
import { PromptExecutor } from "../src/executor.js";
import { fakeRunner, okOutcome, promptOp } from "./fakes.js";

/**
 * A resolved position, the way a HOST supplies one — `hw` does exactly this in `servicesFor`.
 *
 * The engine used to publish a REQUEST and let a layer below look it up. Resolving immediately before
 * dispatch hands the executor the two session facts a prompt call actually consumes — the provider
 * handle and the append/fork decision — instead of a request it has no use for.
 */
async function positionIn(store: MapSessionStore<ModelMessage>, ref: string, fork = false): Promise<ExecServices> {
  return { session: (await store.resolve({ ref, ...(fork ? { fork: true } : {}) })) as never };
}

/** The composition a workflow host builds: request on ctx, position resolved, record beneath. */
function hostStack(store: MapSessionStore<ModelMessage>, runner: ReturnType<typeof fakeRunner>["runner"]) {
  const seam = store as unknown as SessionStore;
  // NO `record: true`. That is the whole point — this is the default core, as a host gets it.
  const core = new PromptExecutor({ runner });
  return withSessionPosition(
    { sessions: seam },
    withRecord({ records: seam as never }, core as never),
  ) as unknown as Executor<ExecServices, { durationMs: number }>;
}

describe("a value-mode core under withSessionPosition/withRecord", () => {
  it("records the conversation, not just the answer", async () => {
    const store = new MapSessionStore<ModelMessage>();
    const { runner } = fakeRunner([okOutcome()]);

    const result = await hostStack(store, runner).start(promptOp(), await positionIn(store, "review")).result;

    // The caller still gets the op's OUTPUT VALUE. A transcript bought by breaking the answer would
    // be no fix at all — value mode has to stay value mode from above.
    expect(result.value).toEqual({ answer: "4" });

    // …and the conversation is in the record the position points at.
    // BOTH halves of the exchange. The core prepends the request turns it sent — the delta is what
    // this call added to the conversation, and the question is as much part of that as the answer.
    const records = store.bySession("review");
    expect(records).toHaveLength(1);
    const messages = defaultMessagesOf(records[0]!) as ModelMessage[];
    expect(messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(messages[1]).toEqual({ role: "assistant", content: [{ type: "text", text: '{"answer":"4"}' }] });
  });

  it("carries the provider handle, so the next call can resume rather than start afresh", async () => {
    const store = new MapSessionStore<ModelMessage>();
    const { runner } = fakeRunner([okOutcome({ providerSessionId: "sess-abc" })]);

    await hostStack(store, runner).start(promptOp(), await positionIn(store, "review")).result;

    // Read back through `resolve`, which is what an appending call actually consults — a handle stored
    // but not resolvable is the same as no handle at all.
    const resolved = await store.resolve({ ref: "review@1" });
    expect(resolved.providerSessionId).toBe("sess-abc");
  });

  it("records the turns of a call that appended and THEN failed", async () => {
    // Those turns exist remotely whether or not we kept them, so dropping them means the next append
    // meets a head our mirror does not describe — divergence on the very next call.
    const store = new MapSessionStore<ModelMessage>();
    const { runner } = fakeRunner([
      okOutcome({ error: { classification: "api-retriable", reason: "overloaded" } }),
    ]);

    await hostStack(store, runner).start(promptOp(), await positionIn(store, "review")).result;

    expect(defaultMessagesOf(store.bySession("review")[0]!)).toHaveLength(2);
  });

  it("reports nothing when no conversation is in play, rather than an empty one", async () => {
    // No position ⇒ no conversation ⇒ nothing recorded. "Reported nothing" and "produced nothing" are
    // different claims, and only the first is true of a call that never ran in a conversation.
    const store = new MapSessionStore<ModelMessage>();
    const { runner } = fakeRunner([okOutcome()]);

    const result = await hostStack(store, runner).start(promptOp(), {}).result;

    expect(result.value).toEqual({ answer: "4" });
    expect((result as { session?: unknown }).session).toBeUndefined();
  });

  it("leaves a RECORD-mode payload whole rather than replacing it with the report", async () => {
    // Both channels are populated now, so `close` has to choose — and the payload is the richer one:
    // its entries carry the reasoning and the tool trace, which a `{ messages }` report does not.
    // Preferring the report unconditionally was safe only while a record-mode core reported nothing.
    const store = new MapSessionStore<ModelMessage>();
    const { runner } = fakeRunner([okOutcome({ entries: [{ kind: "message", role: "assistant", provider: "test", timestamp: "t", content: [{ type: "thinking", thinking: "considering" }] }] } as never)]);
    const seam = store as unknown as SessionStore;
    const core = new PromptExecutor({ runner });
    const stack = withSessionPosition(
      { sessions: seam },
      withRecord({ records: seam as never }, core as never),
    ) as unknown as Executor<ExecServices, { durationMs: number }>;

    await stack.start(promptOp(), await positionIn(store, "review")).result;

    const stored = store.bySession("review")[0]!.result?.value as { entries?: never[] };
    // The wire history is DERIVED from those entries — one array, projected on read.
    // Two: the request half the executor prepends, and the answer the runner reported.
    expect(defaultMessagesOf(store.bySession("review")[0]!)).toHaveLength(2);
    expect(thinkingOfEntries(stored.entries ?? [])).toHaveLength(1);
  });
});
