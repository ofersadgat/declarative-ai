/**
 * The agent as an `Executor` (DESIGN §4.4) — and the proof that the session rules are now ONE
 * implementation rather than two.
 *
 * The last suite here is the point of the whole split. It runs the SAME session expectations against
 * `PromptExecutor` and `AgentExecutor` and asserts they differ in exactly the way their declared
 * capabilities say they should: a transport with no native resume replays the transcript, one that
 * resumes natively carries a handle and reads zero messages, and one that resumes but cannot BRANCH
 * replays a fork while continuing an append. Before the refactor those rules lived twice — in
 * `promptop`'s executor in provider vocabulary and in `agents-api`'s function impl in agent
 * vocabulary — so a test like this could not be written at all.
 */
import { describe, expect, it } from "vitest";
import { isOk, promptOp, resolveSessionRef, type ExecServices, type ResolvedSession } from "@declarative-ai/exec";
import type { ModelMessage } from "@declarative-ai/llm";
import { PromptExecutor } from "@declarative-ai/promptop";
import { AgentApiExecutor, AgentExecutor, DELEGATED_CAPS } from "../src/index.js";
import type { AgentQuery, AgentQueryOptions } from "../src/index.js";

const op = (user = "do it", config: Record<string, unknown> = {}) =>
  promptOp({ user, config: config as never, output: { name: "answer", schema: { type: "string" } } });

/** A query that records what it was handed and answers with a fixed result. */
function capturing(result: { text?: string; costUsd?: number; sessionId?: string } = {}) {
  let seen: AgentQueryOptions | undefined;
  const query: AgentQuery = async function* (opts) {
    seen = opts;
    yield { type: "assistant" };
    yield { type: "result", result: { text: result.text ?? "done", ...result } };
  };
  return { query, seen: () => seen };
}

/** A resolved session with the non-enumerable half attached, exactly as a store hands one over. */
/** Typed as the ctx slot is (`JsonValue`), since that is where it gets assigned; the executor casts
 *  it back to `ModelMessage` exactly as the production code does. */
function session(over: { mode?: "append" | "fork"; providerSessionId?: string; messages?: ModelMessage[] }): ResolvedSession {
  return resolveSessionRef<ModelMessage>("conv:1", {
    mode: over.mode ?? "append",
    at: { id: "conv", seq: 1 },
    ...(over.providerSessionId !== undefined ? { providerSessionId: over.providerSessionId } : {}),
    messages: async () => over.messages ?? [{ role: "user", content: "earlier" }],
  }) as unknown as ResolvedSession;
}

describe("AgentExecutor — a delegated agent answering a PROMPT op", () => {
  it("answers a prompt op, so a prompt state can be served by an agent at all", async () => {
    const { query } = capturing({ text: "the answer" });
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(isOk(result) && result.value).toBe("the answer");
  });

  it("carries the agent's OWN spend, which is the only cost channel a delegated agent has", async () => {
    const { query } = capturing({ costUsd: 0.02 });
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(result.metrics).toMatchObject({ costUsd: 0.02, childCostUsd: 0.02, childLlmCalls: 1, costSource: "provider" });
  });

  it("reports 0 with costSource 'unknown' when the agent said nothing about money", async () => {
    // Not "free" — UNMEASURED. A budget settling this reserve is under-charging, and the provenance
    // flag is the only signal it has.
    const { query } = capturing({});
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(result.metrics).toMatchObject({ costUsd: 0, costSource: "unknown" });
  });

  it("declares the delegated capability record, not the provider one", () => {
    const agent = new AgentExecutor();
    expect(agent.capabilities).toEqual(DELEGATED_CAPS);
    expect(agent.capabilities.structuredOutput).toBe(false);
    expect(agent.capabilities.memoizable).toBe(false);
  });

  it("needs no ModelRouter — there is no provider endpoint in the picture", async () => {
    // The provider path refuses without one. An agent that inherited that check would be unusable
    // exactly where it is most useful: a machine with a CLI subscription and no API key.
    const { query } = capturing();
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(isOk(result)).toBe(true);
  });

  it("renders system + user into the ONE prompt a delegated agent takes", async () => {
    const { query, seen } = capturing();
    const withSystem = promptOp({ system: "be terse", user: "do it", output: { name: "a", schema: { type: "string" } } });
    await new AgentExecutor({ query }).start(withSystem, {}).result;
    expect(seen()?.prompt).toBe("be terse\n\ndo it");
  });

  it("routes the agent's tool approvals to ctx.approve", async () => {
    const { query, seen } = capturing();
    const asked: string[] = [];
    await new AgentExecutor({ query }).start(op(), {
      approve: async (req) => {
        asked.push(req.tool);
        return { decision: "deny" };
      },
    } as ExecServices).result;
    const decision = await seen()?.canUseTool?.({ toolName: "bash", input: {} }, { signal: new AbortController().signal });
    expect(asked).toEqual(["bash"]);
    expect(decision).toEqual({ allow: false, reason: "denied by permission policy" });
  });

  it("classifies an abort as canceled rather than as a permanent failure", async () => {
    const aborted = AbortSignal.abort();
    const query: AgentQuery = async function* () {
      throw new Error("stopped");
    };
    const result = await new AgentExecutor({ query }).start(op(), { abortSignal: aborted }).result;
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.classification).toBe("canceled");
  });

  it("surfaces a run-fatal agent error as data, never as a throw", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "other", error: "the model refused" };
    };
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(!isOk(result) && result.error.reason).toContain("the model refused");
  });

  it("AgentApiExecutor is the SDK transport under its own name", () => {
    expect(new AgentApiExecutor() instanceof AgentExecutor).toBe(true);
    expect(AgentApiExecutor.kind).toBe("agent-api");
  });
});

describe("one session implementation, two transports (the point of the split)", () => {
  it("NO native resume ⇒ REPLAY: the provider path puts the transcript back on the wire", async () => {
    let sent: ModelMessage[] | undefined;
    const prompt = new PromptExecutor({
      runner: async (def) => {
        sent = def.messages as ModelMessage[];
        return { value: { value: "ok", finishReason: "stop" }, metrics: { durationMs: 1, costUsd: 0, costSource: "table" } };
      },
    });
    // The provider path needs a real model id — routing is what it is FOR. That an agent does not is
    // the asymmetry this whole hierarchy exists to express.
    await prompt.start(op("do it", { model: "anthropic/claude-sonnet-5" }), {
      session: session({ mode: "append", providerSessionId: "sess-abc" }),
    }).result;
    expect(sent?.map((m) => m.content)).toEqual(["earlier", "do it"]);
  });

  it("NATIVE resume ⇒ HANDLE: the agent path reads ZERO messages and carries the id instead", async () => {
    // The saving is the whole reason `sessionResume` is a declared capability: replaying here would
    // pay for the entire transcript on every turn to tell the agent what it already knows.
    let materialized = 0;
    const conv = session({ mode: "append", providerSessionId: "sess-abc" });
    const counting = Object.create(conv, { messages: { value: async () => (materialized++, []), enumerable: false } }) as ResolvedSession;
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), { session: counting }).result;
    expect(seen()?.resume).toBe("sess-abc");
    expect(seen()?.messages).toBeUndefined();
    expect(materialized).toBe(0);
  });

  it("native resume WITHOUT a fork primitive ⇒ a fork replays, while an append still resumes", async () => {
    // Codex is the case: `resume` continues server-side and there is no fork primitive at all. The
    // branch is read off `sessionFork`, so stating the capability is the whole of the wiring.
    const caps = { ...DELEGATED_CAPS, sessionFork: false };
    const forked = capturing();
    await new AgentExecutor({ query: forked.query, capabilities: caps }).start(op(), {
      session: session({ mode: "fork", providerSessionId: "sess-abc" }),
    }).result;
    expect(forked.seen()?.messages).toBeDefined();

    const appended = capturing();
    await new AgentExecutor({ query: appended.query, capabilities: caps }).start(op(), {
      session: session({ mode: "append", providerSessionId: "sess-abc" }),
    }).result;
    expect(appended.seen()?.resume).toBe("sess-abc");
    expect(appended.seen()?.messages).toBeUndefined();
  });

  it("a native FORK carries the handle plus the branch flag", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), { session: session({ mode: "fork", providerSessionId: "sess-abc" }) }).result;
    expect(seen()?.resume).toBe("sess-abc");
    expect(seen()?.forkSession).toBe(true);
  });

  it("reports the handle the run ENDED in, so a fork's new id is not lost", async () => {
    // Losing it would put two branches into one remote session — silent and unrecoverable. In RECORD
    // mode the payload IS the call output, which is where the handle rides.
    const { query } = capturing({ sessionId: "sess-after" });
    const agent = new AgentExecutor({ query, record: true });
    const result = await agent.start(op(), { session: session({ mode: "fork", providerSessionId: "sess-abc" }) }).result;
    expect((result.value as { providerSessionId?: string }).providerSessionId).toBe("sess-after");
  });

  it("passes nothing session-shaped when no conversation is in play", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), {}).result;
    expect(seen()?.resume).toBeUndefined();
    expect(seen()?.messages).toBeUndefined();
  });
});

describe("the model reaches the transport", () => {
  it("strips the route prefix — the binary knows 'sonnet', not 'claude-cli/sonnet'", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op("do it", { model: "claude-cli/sonnet" }), {}).result;
    expect(seen()?.model).toBe("sonnet");
  });

  it("keeps a multi-segment id after the first slash", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op("do it", { model: "agent/anthropic/claude-opus-4" }), {}).result;
    expect(seen()?.model).toBe("anthropic/claude-opus-4");
  });

  it("asks for NOTHING when the call named no model", async () => {
    // The placeholder this class supplies must not become a request for a model literally named
    // `default` — every one of these binaries would refuse it, and the zero-configuration case is the
    // one that has to work.
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), {}).result;
    expect(seen()?.model).toBeUndefined();
  });
});
