import { composeExecutors, MapSessionStore } from "@declarative-ai/core";
import type { ModelMessage } from "ai";
import { describe, expect, it } from "vitest";
import { LlmCallExecutor } from "../src/executor";
import { withSession } from "../src/wrappers";
import { fakeRunner, okOutcome, specOf } from "./fakes";

const sessionExec = (store?: MapSessionStore) => {
  const { runner, calls } = fakeRunner([okOutcome()]);
  // withSession({sessions}) is self-satisfied; store-less withSession() requires a run-scoped ctx.sessions.
  const wrap = store ? withSession({ sessions: store }) : withSession();
  return { exec: composeExecutors(new LlmCallExecutor({ runner }), wrap), calls };
};

describe("withSession — client-managed conversation", () => {
  it("no session id → passthrough (prompt unchanged, store untouched)", async () => {
    const store = new MapSessionStore();
    const { exec, calls } = sessionExec(store);
    await exec.start(specOf({}, { model: "m", prompt: "hi", timeoutMs: 1000 }), {}).outcome;
    expect(calls[0]!.params.prompt).toBe("hi");
    expect(calls[0]!.params.messages).toBeUndefined();
    expect(store.get("anything")).toBeUndefined();
  });

  it("a fresh logical session seeds the transcript with the turn + the assistant reply", async () => {
    const store = new MapSessionStore();
    const { exec, calls } = sessionExec(store);
    const outcome = await exec.start(specOf({}, { model: "m", prompt: "hi", sessionId: "s1", timeoutMs: 1000 }), {}).outcome;

    // The inner call received only the current turn (prior was empty); prompt was folded into messages,
    // and the session fields were CONSUMED (stripped) so the bare core never refuses them.
    expect(calls[0]!.params.prompt).toBeUndefined();
    expect(calls[0]!.params.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(calls[0]!.params.sessionId).toBeUndefined();

    // The transcript now holds the user turn + the assistant reply, and the outcome names the
    // continuation token (the logical id).
    expect((store.get("s1") as { messages: ModelMessage[] }).messages).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: '{"answer":"4"}' },
    ]);
    expect(outcome.session?.id).toBe("s1");
  });

  it("resuming a session prepends the stored transcript to the new turn", async () => {
    const store = new MapSessionStore();
    store.put("s1", {
      messages: [
        { role: "user", content: "q1" },
        { role: "assistant", content: "a1" },
      ] satisfies ModelMessage[],
    });
    const { exec, calls } = sessionExec(store);
    await exec.start(specOf({}, { model: "m", prompt: "q2", sessionId: "s1", timeoutMs: 1000 }), {}).outcome;

    expect(calls[0]!.params.messages).toEqual([
      { role: "user", content: "q1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "q2" },
    ]);
    expect((store.get("s1") as { messages: ModelMessage[] }).messages).toHaveLength(4); // + new assistant turn
  });

  it("the transcript fold PRESERVES other stored SessionState fields (never clobbers the record)", async () => {
    const store = new MapSessionStore();
    store.put("s1", { messages: [], providerSessionId: "srv-abc" });
    const { exec } = sessionExec(store);
    await exec.start(specOf({}, { model: "m", prompt: "hi", sessionId: "s1", timeoutMs: 1000 }), {}).outcome;
    const state = store.get("s1")!;
    expect(state.providerSessionId).toBe("srv-abc");
    expect(state.messages).toHaveLength(2);
  });

  it("REFUSES providerSessionId — provider-side session resume is not supported yet", async () => {
    const store = new MapSessionStore();
    const { exec, calls } = sessionExec(store);
    const outcome = await exec.start(specOf({}, { model: "m", prompt: "hi", providerSessionId: "p1", timeoutMs: 1000 }), {}).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/provider-side session resume is not supported/);
    expect(store.get("p1")).toBeUndefined();
  });

  it("REFUSES a sessionId with no SessionStore available (construction or ctx.sessions)", async () => {
    const { exec, calls } = sessionExec(undefined);
    const outcome = await exec.start(specOf({}, { model: "m", prompt: "hi", sessionId: "s1", timeoutMs: 1000 }), {}).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/no SessionStore is available/);
  });

  it("falls back to the run-scoped ctx.sessions store when none was constructed", async () => {
    const store = new MapSessionStore();
    const { exec } = sessionExec(undefined);
    await exec.start(specOf({}, { model: "m", prompt: "hi", sessionId: "s1", timeoutMs: 1000 }), { sessions: store }).outcome;
    expect(store.get("s1")).toBeDefined();
  });

  it("declares sessionResume capability", () => {
    const { exec } = sessionExec(new MapSessionStore());
    expect(exec.capabilities.sessionResume).toBe(true);
  });
});
