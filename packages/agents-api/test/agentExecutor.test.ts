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
import { thinkingOfEntries, toolResultsOfEntries, toolUsesOfEntries } from "@declarative-ai/llm";
import { isOk, sessionOutcomeOf, promptOp, resolveSessionRef, type ExecEvent, type ExecServices, type ResolvedSession, type Tool } from "@declarative-ai/exec";
import type { LlmOutput, ModelMessage } from "@declarative-ai/llm";
import { PromptExecutor } from "@declarative-ai/promptop";
import { createToolGate, PermissionLedger, type Approver, type PermissionBaseline, type PermissionMode, type SmartApprover } from "@declarative-ai/permissions";
import { AgentApiExecutor, AgentExecutor, DELEGATED_CAPS } from "../src/index.js";
import type { AgentQuery, AgentQueryOptions } from "../src/index.js";

const op = (user = "do it", config: Record<string, unknown> = {}) =>
  promptOp({ user, config: config as never, output: { name: "answer", schema: { type: "string" } } });

/** A query that records what it was handed and answers with a fixed result. */
function capturing(result: { text?: string; costUsd?: number; sessionId?: string; structured?: unknown } = {}) {
  let seen: AgentQueryOptions | undefined;
  const query: AgentQuery = async function* (opts) {
    seen = opts;
    yield { type: "assistant" };
    yield { type: "result", result: { text: result.text ?? "done", ...result } as never };
  };
  return { query, seen: () => seen };
}

/** A prompt op declaring an OBJECT output — the shape a state with declared outputs lowers to. */
const objectOp = () =>
  promptOp({
    user: "extract them",
    output: {
      name: "answer",
      schema: { type: "object", properties: { items: { type: "array", items: { type: "string" } } }, required: ["items"] },
    },
  });

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
    // Structured output is NATIVE here: both transports carry the schema and retry inside their own
    // loop until the value validates. What stays false is memoizability — an agent runs its own
    // non-deterministic loop, and two runs of it are not the same call.
    expect(agent.capabilities.structuredOutput).toBe(true);
    expect(agent.capabilities.memoizable).toBe(false);
  });

  /**
   * THE OUTPUT SCHEMA, which is what lets a delegated agent serve a state that declares outputs.
   *
   * It was read off the lowered declaration by every other transport and dropped by this one, so the
   * agent answered in prose and the engine found none of the slots the state declared — reported as
   * "prompt operation did not produce required output 'x'", which names the state and not the reason.
   */
  it("hands the output schema to the transport, which is what constrains the answer", async () => {
    const { query, seen } = capturing({ structured: { items: ["a", "b"] } });
    await new AgentExecutor({ query }).start(objectOp(), {}).result;
    expect(seen()?.schema).toMatchObject({ type: "object", required: ["items"] });
  });

  it("answers with the STRUCTURED value, not with the prose beside it", async () => {
    // Both arrive on the terminal message and they are not the same thing: `result` summarizes the
    // work. Answering with it where an object was declared fills none of the state's slots.
    const { query } = capturing({ text: "I found two items.", structured: { items: ["a", "b"] } });
    const result = await new AgentExecutor({ query }).start(objectOp(), {}).result;
    expect(isOk(result) && result.value).toEqual({ items: ["a", "b"] });
  });

  it("says nothing about a schema when the op declares no object output", async () => {
    const { query, seen } = capturing({ text: "the answer" });
    await new AgentExecutor({ query }).start(op(), {}).result;
    // A `{type: "string"}` output is a TEXT call — there is nothing to constrain, and sending a schema
    // would make the agent wrap a plain answer in JSON nobody asked for.
    expect(seen()?.schema).toBeUndefined();
  });

  it("fails, retriably, when a schema was asked for and no structured value came back", async () => {
    // NOT a fall back to the prose. The caller cannot tell that apart from an agent that ignored the
    // request, and the engine would reject it one layer further from the transport that knows why.
    const { query } = capturing({ text: "I found two items." });
    const result = await new AgentExecutor({ query }).start(objectOp(), {}).result;
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error).toMatchObject({ classification: "api-retriable" });
    expect(!isOk(result) && result.error.reason).toMatch(/no structured output/);
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
    // Losing it would put two branches into one remote session — silent and unrecoverable. It rides on
    // the SESSION channel, which is where `withRecord` reads it and therefore where it has to be:
    // every recorded call needs the handle, where only a persisting caller needs the reasoning and the
    // tool trace, so the two are reported separately.
    const { query } = capturing({ sessionId: "sess-after" });
    const agent = new AgentExecutor({ query });
    const result = await agent.start(op(), { session: session({ mode: "fork", providerSessionId: "sess-abc" }) }).result;
    expect(sessionOutcomeOf(result)?.providerSessionId).toBe("sess-after");
  });

  it("passes nothing session-shaped when no conversation is in play", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), {}).result;
    expect(seen()?.resume).toBeUndefined();
    expect(seen()?.messages).toBeUndefined();
  });
});

/**
 * The losslessness claim, end to end.
 *
 * `llmConfig.ts` states the goal on the way IN — a stored config "transforms losslessly into a real
 * call" — and `output.ts` implements it on the way OUT. The agent boundary honoured neither: every
 * non-`result` message collapsed to `{type: "other"}`, `invoke` fabricated `finishReason: "stop"` and
 * synthesized a one-turn message log, and not one token was reported. The information was all there.
 */
describe("lossless output — what the agent produced reaches the caller", () => {
  /** A fake stream carrying everything a real turn carries. */
  const fullTurn: AgentQuery = async function* () {
    yield { type: "provider_event", event: { type: "system", subtype: "init", model: "claude-opus-4-7" } };
    yield { type: "partial", delta: "The " };
    yield { type: "partial", delta: "answer" };
    yield {
      type: "assistant",
      message: { role: "assistant", content: [{ type: "thinking", thinking: "check the file" }] },
      thinking: [{ text: "check the file", providerMetadata: { anthropic: { signature: "sig-1" } } }],
    };
    yield {
      type: "assistant",
      message: { role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "Read", input: { path: "a.txt" } }] },
      toolCalls: [{ toolCallId: "toolu_1", toolName: "Read", input: { path: "a.txt" } }],
    };
    yield {
      type: "user",
      message: { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ZEPHYR" }] },
      toolResults: [{ toolCallId: "toolu_1", output: "ZEPHYR" }],
    };
    yield { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "The answer" }] }, text: "The answer" };
    yield {
      type: "result",
      result: {
        text: "The answer",
        costUsd: 0.19,
        sessionId: "sess-1",
        finishReason: "length",
        usage: { inputTokens: 30283, outputTokens: 8, noCacheTokens: 6, cacheWriteTokens: 30277, cacheWrite1hTokens: 30277, totalTokens: 30291 },
        rawUsage: { usage: { input_tokens: 6 } },
      },
    };
  };

  /**
   * The full `LlmOutput`, ASKED FOR on the record channel — which is what a session persists.
   *
   * This used to construct the executor with `record: true` and read the payload off the execution
   * VALUE. That mode swapped `Out`, and `AgentExecutor` drops `PromptExecutor`'s `Out` parameter, so
   * the class typechecked as returning a projection while returning a payload. Asking per call leaves
   * the value alone and puts the payload beside it, where no type has to lie.
   */
  const payloadOf = async (query: AgentQuery, ctx: ExecServices = {}) =>
    (await new AgentExecutor({ query }).start(op(), { ...ctx, returnRecord: true }).result as { record?: LlmOutput }).record!;

  it("reports the agent's OWN finish reason, so a truncated run does not read as a clean one", async () => {
    expect((await payloadOf(fullTurn)).finishReason).toBe("length");
  });

  it("falls back to `unknown` rather than to a fabricated `stop` when the transport said nothing", async () => {
    const { query } = capturing({ text: "done" });
    expect((await payloadOf(query)).finishReason).toBe("unknown");
  });

  it("hands back the agent's OWN log, verbatim — not one synthesized assistant turn", async () => {
    const entries = (await payloadOf(fullTurn)).entries ?? [];
    expect(entries).toHaveLength(4);
    // The blocks are NORMALIZED — one vocabulary whichever transport produced them — but nothing is
    // dropped and nothing is invented: a signature that must go back byte-identical still does.
    expect(entries[0]).toMatchObject({
      kind: "message",
      role: "assistant",
      content: [{ type: "thinking", thinking: "check the file", signature: "sig-1" }],
    });
    // The provider's own spelling, kept: `tool_use_id`/`content` rather than the neutral names the
    // read-side projections use. Storing the translation instead is what would break a replay.
    expect(entries[2]).toMatchObject({
      kind: "message",
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ZEPHYR" }],
    });
  });

  it("keeps thinking with its signature, where the block that carried it sits", async () => {
    // Derived rather than stored beside the log: `thinking` was a projection of exactly these
    // messages, and the record used to hold both.
    const payload = await payloadOf(fullTurn);
    expect(thinkingOfEntries(payload.entries ?? [])).toEqual([
      { type: "thinking", thinking: "check the file", signature: "sig-1" },
    ]);
  });

  it("keeps the tool calls and the results that answered them", async () => {
    const payload = await payloadOf(fullTurn);
    expect(toolUsesOfEntries(payload.entries ?? [])).toEqual([
      { type: "tool_use", id: "toolu_1", name: "Read", input: { path: "a.txt" } },
    ]);
    expect(toolResultsOfEntries(payload.entries ?? [])).toEqual([
      { type: "tool_result", toolUseId: "toolu_1", text: "ZEPHYR" },
    ]);
  });

  it("carries every token count plus rawUsage, so costUsd stays recomputable", async () => {
    const result = await new AgentExecutor({ query: fullTurn }).start(op(), {}).result;
    expect(result.metrics).toMatchObject({
      costUsd: 0.19,
      costSource: "provider",
      inputTokens: 30283,
      outputTokens: 8,
      noCacheTokens: 6,
      cacheWriteTokens: 30277,
      cacheWrite1hTokens: 30277,
      totalTokens: 30291,
      rawUsage: { usage: { input_tokens: 6 } },
    });
  });

  it("delivers output deltas and provider events to a `for await` over the handle", async () => {
    // `DELEGATED_CAPS.streaming: true` was aspirational: the base returns `emptyEvents()`, so nothing
    // could be watched at all. For a run that takes minutes that is indistinguishable from a hang.
    const handle = new AgentExecutor({ query: fullTurn }).start(op(), {});
    const seen: ExecEvent[] = [];
    for await (const event of handle.events) seen.push(event);
    await handle.result;
    expect(seen.filter((e) => e.type === "output_partial").map((e) => (e as { text: string }).text)).toEqual(["The ", "answer"]);
    // Opaque passthrough: `exec` never learns what an `init` message is.
    expect(seen.filter((e) => e.type === "provider_event")).toEqual([
      { type: "provider_event", payload: { type: "system", subtype: "init", model: "claude-opus-4-7" } },
    ]);
  });

  /** A run that spawns a subagent: the sidechain's turns arrive on the same stream, tagged. */
  const withSubagent: AgentQuery = async function* () {
    yield {
      type: "assistant",
      message: { role: "assistant", content: [{ type: "tool_use", id: "toolu_task", name: "Task", input: { prompt: "explore" } }] },
      toolCalls: [{ toolCallId: "toolu_task", toolName: "Task", input: { prompt: "explore" } }],
    };
    yield {
      type: "assistant",
      parentToolUseId: "toolu_task",
      message: { role: "assistant", content: [{ type: "text", text: "I am the subagent" }] },
      text: "I am the subagent",
    };
    yield {
      type: "user",
      parentToolUseId: "toolu_task",
      message: { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_sub", content: "sub data" }] },
      toolResults: [{ toolCallId: "toolu_sub", output: "sub data" }],
    };
    yield {
      type: "user",
      message: { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_task", content: "the report" }] },
      toolResults: [{ toolCallId: "toolu_task", output: "the report" }],
    };
    yield { type: "provider_event", event: { type: "system", subtype: "compact_boundary" } };
    yield { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "Done." }] }, text: "Done." };
    yield { type: "result", result: { text: "Done." } };
  };

  it("keeps a subagent's turns OUT of the main log and in a sidechain keyed by the spawning call", async () => {
    const payload = await payloadOf(withSubagent);
    // The main thread: Task call, its report, the final answer — and nothing the subagent said.
    const main = (payload.entries ?? []).filter((e) => e.sidechain === undefined);
    expect(JSON.stringify(main)).not.toContain("I am the subagent");
    expect(main).toHaveLength(3);
    // One array, and the subagent's turns are IN it — marked, not filed under a second key space.
    const sub = (payload.entries ?? []).filter((e) => e.sidechain?.id === "toolu_task");
    expect(sub).toMatchObject([
      { kind: "message", role: "assistant", content: [{ type: "text", text: "I am the subagent" }] },
      { kind: "message", role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_sub", content: "sub data" }] },
    ]);
    // The projections are the MAIN thread's: the subagent's tool results are not the agent's own.
    expect(toolResultsOfEntries(payload.entries ?? [])).toEqual([
      { type: "tool_result", toolUseId: "toolu_task", text: "the report" },
    ]);
    // The subagent's answer text never leaks into the main answer.
    expect(payload.value).toBe("Done.");
  });

  it("records provider events pinned to their place among the turns", async () => {
    const payload = await payloadOf(withSubagent);
    // After the third main-thread message, before the fourth — where it happened.
    expect(payload.providerEvents).toEqual([{ index: 2, event: { type: "system", subtype: "compact_boundary" } }]);
  });

  it("tags a subagent's live `message` events with the spawning call", async () => {
    const handle = new AgentExecutor({ query: withSubagent }).start(op(), {});
    const seen: ExecEvent[] = [];
    for await (const event of handle.events) seen.push(event);
    await handle.result;
    const messages = seen.filter((e) => e.type === "message") as Array<{ parentToolUseId?: string }>;
    expect(messages.map((m) => m.parentToolUseId)).toEqual([undefined, "toolu_task", "toolu_task", undefined, undefined]);
  });

  it("delivers reasoning deltas as `thinking_partial`, never on the answer's channel", async () => {
    const thinking: AgentQuery = async function* () {
      yield { type: "thinking-partial", delta: "check " };
      yield { type: "thinking-partial", delta: "the file" };
      yield { type: "partial", delta: "The answer" };
      yield { type: "result", result: { text: "The answer" } };
    };
    const handle = new AgentExecutor({ query: thinking }).start(op(), {});
    const seen: ExecEvent[] = [];
    for await (const event of handle.events) seen.push(event);
    await handle.result;
    expect(seen.filter((e) => e.type === "thinking_partial").map((e) => (e as { text: string }).text)).toEqual(["check ", "the file"]);
    expect(seen.filter((e) => e.type === "output_partial").map((e) => (e as { text: string }).text)).toEqual(["The answer"]);
  });

  it("delivers each finished turn as a `message` event, tool calls and results included", async () => {
    // The variant the contract declared and nothing emitted: without it a live viewer learns about
    // an agent's tool calls only when the record closes — for an hour-long run, that reads as an
    // agent doing nothing, and then as a transcript that appeared out of nowhere.
    const handle = new AgentExecutor({ query: fullTurn }).start(op(), {});
    const seen: ExecEvent[] = [];
    for await (const event of handle.events) seen.push(event);
    await handle.result;
    const messages = seen.filter((e) => e.type === "message") as Array<{ role: string; content: unknown }>;
    expect(messages.map((m) => m.role)).toEqual(["assistant", "assistant", "user", "assistant"]);
    // Verbatim turn objects, in stream order — the tool_use and the tool_result that answers it.
    expect(messages[1]?.content).toEqual({ role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "Read", input: { path: "a.txt" } }] });
    expect(messages[2]?.content).toEqual({ role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ZEPHYR" }] });
  });

  it("CLOSES the event stream when the run settles, so a consumer is never left parked", async () => {
    const handle = new AgentExecutor({ query: capturing().query }).start(op(), {});
    const drained: ExecEvent[] = [];
    for await (const event of handle.events) drained.push(event);
    expect(drained).toEqual([]);
  });
});

/**
 * The losslessness claim on the way IN.
 *
 * `resolveConfig`, `PromptExecutorOptions.defaults` and `op.config` all merge into one declaration, so a
 * field that reached the executor was ASKED FOR. `runAgent` used to read three of them — model,
 * providerSessionId, messages — and drop the rest, which meant a state authored with a reasoning level
 * and a step budget ran at the agent's own defaults and said nothing about it.
 */
describe("lossless input — what the caller configured reaches the transport", () => {
  it("forwards a reasoning request, xhigh included", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op("do it", { reasoning: { effort: "xhigh" } }), {}).result;
    expect(seen()?.reasoning).toEqual({ effort: "xhigh" });
  });

  it("forwards a step budget and a tool choice", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op("do it", { maxSteps: 8, toolChoice: "none" }), {}).result;
    expect(seen()?.maxSteps).toBe(8);
    expect(seen()?.toolChoice).toBe("none");
  });

  it("hands the transport ONLY its own providerOptions bag", async () => {
    // Keyed by provider precisely so one config can carry settings for several transports and each
    // takes what is addressed to it.
    const { query, seen } = capturing();
    const config = { providerOptions: { claudeCode: { fastMode: true }, openrouter: { reasoning: { effort: "high" } } } };
    await new AgentExecutor({ query }).start(op("do it", config), {}).result;
    expect(seen()?.providerOptions).toEqual({ fastMode: true });
  });

  it("REFUSES a decoding knob rather than dropping it — the agent issues its own model calls", async () => {
    const { query } = capturing();
    const result = await new AgentExecutor({ query }).start(op("do it", { temperature: 0.2 }), {}).result;
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/temperature/);
  });

  it("REFUSES maxOutputTokens and stopSequences, naming what to use instead", async () => {
    const tokens = await new AgentExecutor({ query: capturing().query }).start(op("do it", { maxOutputTokens: 500 }), {}).result;
    expect(!isOk(tokens) && tokens.error.reason).toMatch(/maxOutputTokens.*cannot be honoured/s);
    const stops = await new AgentExecutor({ query: capturing().query }).start(op("do it", { stopSequences: ["END"] }), {}).result;
    expect(!isOk(stops) && stops.error.reason).toMatch(/stopSequences/);
  });

  it("REFUSES a toolChoice that constrains ONE turn, since an agent runs a whole loop", async () => {
    const result = await new AgentExecutor({ query: capturing().query }).start(op("do it", { toolChoice: "required" }), {}).result;
    expect(!isOk(result) && result.error.reason).toMatch(/whole loop/);
  });

  it("classifies a refusal, it does not throw — the Result envelope never throws for a unit failure", async () => {
    const handle = new AgentExecutor({ query: capturing().query }).start(op("do it", { seed: 7 }), {});
    await expect(handle.result).resolves.toBeDefined();
  });
});

/**
 * Tool injection, and the two things it could not do.
 *
 * ✅ OBSERVED (claude 2.1.142) through a real loopback bridge: with `read_file` injected and `Read`
 * still available, the agent used `Read` — every time, because its system prompt steers it there. Deny
 * `Read` and the SAME run reaches for `mcp__dai__read_file`, our impl executes, and the call goes
 * through the approver. Injection on its own was adding a second set of tools the model ignored.
 */
describe("injectTools — displacing the natives, and adding to them", () => {
  const tool = (name: string): Tool => ({ description: name, readOnly: true, inputSchema: { type: "object" } as never, run: () => name });
  const ctxWith = (tools: Record<string, Tool>): ExecServices => ({ tools });

  it("DISPLACES the built-in an injected tool stands in for, so the substitution is real", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, replacesNative: { read_file: "Read" } }).start(op(), ctxWith({ read_file: tool("read_file") })).result;
    expect(seen()?.mcpTools).toHaveProperty("read_file");
    expect(seen()?.disallowedTools).toEqual(["Read"]);
  });

  it("takes a list, because one logical tool can stand in for several built-ins", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, replacesNative: { search: ["Grep", "Glob"] } }).start(op(), ctxWith({ search: tool("search") })).result;
    expect(seen()?.disallowedTools).toEqual(["Grep", "Glob"]);
  });

  it("displaces NOTHING when the tool is routed natively — that built-in was just requested", async () => {
    const { query, seen } = capturing();
    const agent = new AgentExecutor({ query, nativeTools: { read_file: { native: "Read" } }, replacesNative: { read_file: "Read" } });
    await agent.start(op(), ctxWith({ read_file: tool("read_file") })).result;
    expect(seen()?.allowedTools).toEqual(["Read"]);
    expect(seen()?.disallowedTools).toBeUndefined();
  });

  it("displaces nothing when the caller never said what a tool stands in for", async () => {
    // Which built-in a logical tool replaces is a fact about the agent being driven, and this executor
    // drives more than one — so it is stated, never guessed.
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), ctxWith({ read_file: tool("read_file") })).result;
    expect(seen()?.disallowedTools).toBeUndefined();
  });

  it("adds EXTRA tools without giving up the agent's built-ins", async () => {
    // The case the single switch could not express: a host exposing its own capability — a preview
    // pane, a build runner — to an otherwise stock agent. `false` routed everything native; `true`
    // replaced the lot.
    const { query, seen } = capturing();
    const agent = new AgentExecutor({ query, injectTools: false, extraTools: { preview: tool("preview") } });
    await agent.start(op(), ctxWith({ read_file: tool("read_file") })).result;
    expect(seen()?.allowedTools).toEqual(["read_file"]); // the agent's own, by name
    expect(Object.keys(seen()?.mcpTools ?? {})).toEqual(["preview"]); // plus ours
  });

  it("injects extras even when the run declares no ctx.tools at all", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, extraTools: { preview: tool("preview") } }).start(op(), {}).result;
    expect(Object.keys(seen()?.mcpTools ?? {})).toEqual(["preview"]);
  });

  it("keeps the deny floor over an extra tool — an extra tool is a tool", async () => {
    const { query, seen } = capturing();
    const agent = new AgentExecutor({ query, extraTools: { preview: tool("preview") } });
    await agent.start(op(), { policy: { baseline: { tools: { preview: "deny" } } } } as unknown as ExecServices).result;
    expect(seen()?.mcpTools).toBeUndefined();
  });

  it("pre-approves NOTHING with an empty allow-list, which is what the binary reads it as", async () => {
    // Checked on a live run: `--allowedTools ""` still let the agent use its native `Read`. So an empty
    // list is "pre-approve nothing", not "allow nothing", and the two transports agree about it.
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), ctxWith({ read_file: tool("read_file") })).result;
    expect(seen()?.allowedTools).toEqual([]);
  });
});

/**
 * Steering a live turn.
 *
 * The seam was a one-shot created and consumed inside a private method, so nothing outside could touch
 * a running agent — and the thing a caller most wants from a five-minute run is to stop it and keep
 * what it found.
 *
 * ⚠️ The distinction the whole design turns on: `interrupt` is NOT `cancel`. Cancellation settles the
 * handle with a `canceled` failure and discards the work. An interrupted agent turn ends early, still
 * emits its `result`, and SUCCEEDS.
 */
/**
 * A binary that is PRESENT but cannot run — and the difference between the kinds.
 *
 * ✅ CAPTURED from `claude 2.1.142` driven with an empty `CLAUDE_CONFIG_DIR`. Note what the run looks
 * like from outside: exit code 0, `subtype: "success"`, `terminal_reason: "completed"`. Only `is_error`
 * says anything is wrong, and the sentence explaining it sits where the answer would be — so reading
 * the discriminator reports "Not logged in · Please run /login" as a review that found nothing.
 */
describe("present but not usable — detection and classification", () => {
  /** The real assistant turn: `<synthetic>` model, the failure text, and the machine-readable code. */
  const authFailure = (code = "authentication_failed", text = "Not logged in · Please run /login"): AgentQuery =>
    () => ({
      [Symbol.asyncIterator]: async function* () {
        yield {
          type: "assistant",
          message: { role: "assistant", model: "<synthetic>", content: [{ type: "text", text }] },
          text,
          errorCode: code,
        } as const;
        yield { type: "other", error: text } as const;
      },
    });

  it("DETECTS a not-logged-in run, which otherwise looks like a successful empty answer", async () => {
    const result = await new AgentExecutor({ query: authFailure() }).start(op(), {}).result;
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/Not logged in/);
  });

  it("classifies it PERMANENT, so no retry wrapper burns its budget on it", async () => {
    // Logging in is not something a retry can accomplish.
    const result = await new AgentExecutor({ query: authFailure() }).start(op(), {}).result;
    expect(!isOk(result) && result.error.classification).toBe("permanent");
  });

  it("classifies a transient overload as RETRIABLE, which is what the code is for", async () => {
    // Without the code every delegated failure was `permanent`: an `AgentError` carries no status and
    // no retryable flag, so a rate limit inside the agent's own loop looked like a broken workflow and
    // defeated every retry wrapper above — the exact thing `invoke` promises not to do.
    const limited = await new AgentExecutor({ query: authFailure("rate_limit", "rate limited") }).start(op(), {}).result;
    expect(!isOk(limited) && limited.error.classification).toBe("network-retriable");
    expect(!isOk(limited) && limited.error.rateLimited).toBe(true);
    const overloaded = await new AgentExecutor({ query: authFailure("overloaded", "overloaded") }).start(op(), {}).result;
    expect(!isOk(overloaded) && overloaded.error.classification).toBe("network-retriable");
  });

  it("defaults an UNRECOGNISED code to permanent rather than retrying an unknown condition", async () => {
    const result = await new AgentExecutor({ query: authFailure("invented_next_release", "something") }).start(op(), {}).result;
    expect(!isOk(result) && result.error.classification).toBe("permanent");
  });

  it("names WHICH transport produced it, so three wired agents are tellable apart", async () => {
    const result = await new AgentExecutor({ query: authFailure(), label: "claude-cli" }).start(op(), {}).result;
    expect(!isOk(result) && result.error.reason).toMatch(/^claude-cli:/);
  });
});

describe("the control channel", () => {
  /** A run that streams until interrupted, then answers with what it had. */
  function interruptible() {
    let stop!: () => void;
    const stopped = new Promise<void>((resolve) => (stop = resolve));
    let interrupts = 0;
    const query: AgentQuery = () => ({
      [Symbol.asyncIterator]: async function* () {
        yield { type: "partial", delta: "half an " } as const;
        await stopped;
        yield { type: "result", result: { text: "half an answer", finishReason: "stop" } } as const;
      },
      interrupt: async () => {
        interrupts++;
        stop();
      },
    });
    return { query, interrupts: () => interrupts };
  }

  it("ends the turn and settles as a SUCCESS carrying the partial answer", async () => {
    const { query } = interruptible();
    const handle = new AgentExecutor({ query }).start(op(), {});
    await handle.control!.interrupt!();
    const result = await handle.result;
    // NOT a cancellation. Routing interrupt through the abort controller would settle this handle with
    // a `canceled` failure and throw away an answer the agent actually produced.
    expect(isOk(result)).toBe(true);
    expect(isOk(result) && result.value).toBe("half an answer");
  });

  it("answers with what the TURNS carried when an aborted run's terminal message has no text", async () => {
    // ✅ OBSERVED: an interrupted run comes back with `terminal_reason: "aborted_streaming"` and an
    // EMPTY result. The partial answer exists only in the assistant turns already streamed, so
    // reporting `result.text` verbatim would answer "stop and tell me what you found" with nothing.
    const query: AgentQuery = () => ({
      [Symbol.asyncIterator]: async function* () {
        yield { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "1\n2\n3" }] }, text: "1\n2\n3" } as const;
        yield { type: "result", result: { text: "", finishReason: "aborted" } } as const;
      },
    });
    const result = await new AgentExecutor({ query }).start(op(), {}).result;
    expect(isOk(result)).toBe(true);
    // The op's OUTPUT VALUE directly — projection is the default, so there is no payload to
    // reach through unless a caller asked for one.
    expect(result.value).toBe("1\n2\n3");
    expect(((await new AgentExecutor({ query }).start(op(), { returnRecord: true }).result) as { record?: LlmOutput }).record?.finishReason).toBe("aborted");
  });

  it("is IDEMPOTENT, so a Stop that races the answer landing is a no-op", async () => {
    const { query, interrupts } = interruptible();
    const handle = new AgentExecutor({ query }).start(op(), {});
    await handle.control!.interrupt!();
    await handle.result;
    await handle.control!.interrupt!();
    await handle.control!.interrupt!();
    expect(interrupts()).toBe(1);
  });

  it("reaches the run one tick in, before a single message has been read", async () => {
    // A caller pressing Stop at the start of a five-minute turn must reach something.
    const { query, interrupts } = interruptible();
    const handle = new AgentExecutor({ query }).start(op(), {});
    await handle.control!.interrupt!();
    await handle.result;
    expect(interrupts()).toBe(1);
  });

  it("forwards send / setPermissionMode / setModel to the transport", async () => {
    const seen: string[] = [];
    const query: AgentQuery = () => ({
      [Symbol.asyncIterator]: async function* () {
        yield { type: "result", result: { text: "ok" } } as const;
      },
      send: async (t) => void seen.push(`send:${t}`),
      setPermissionMode: async (m) => void seen.push(`mode:${m}`),
      setModel: async (m) => void seen.push(`model:${m}`),
    });
    const handle = new AgentExecutor({ query }).start(op(), {});
    await handle.control!.send!("also check the tests");
    await handle.control!.setPermissionMode!("plan");
    await handle.control!.setModel!("opus");
    await handle.result;
    expect(seen).toEqual(["send:also check the tests", "mode:plan", "model:opus"]);
  });

  it("does NOTHING rather than throwing when the transport offers no interrupt", async () => {
    // Absent MEANS unsupported. A throwing stub would make "this transport cannot" indistinguishable
    // from "that failed".
    const { query } = capturing({ text: "done" });
    const handle = new AgentExecutor({ query }).start(op(), {});
    await expect(handle.control!.interrupt!()).resolves.toBeUndefined();
    expect(isOk(await handle.result)).toBe(true);
  });

  it("declares the capability, so a caller decides BEFORE the call whether to offer a Stop button", () => {
    expect(DELEGATED_CAPS.sessionSteering).toBe(true);
    // And a transport that cannot steer says so, rather than exposing a control surface that does
    // nothing: killing a subprocess is not a graceful turn end.
    const unsteerable = new AgentExecutor({ query: capturing().query, capabilities: { ...DELEGATED_CAPS, sessionSteering: false } });
    expect(unsteerable.start(op(), {}).control).toBeUndefined();
  });

  it("DROPS a queued request when the run never started, rather than replaying it at nothing", async () => {
    // A refused call has no turn to interrupt. Holding the request would leave it pending on a channel
    // that is already closed; replaying it would reach a transport that was never built.
    const handle = new AgentExecutor({ query: capturing().query }).start(op("do it", { temperature: 0.2 }), {});
    await handle.control!.interrupt!();
    const result = await handle.result;
    expect(isOk(result)).toBe(false);
    // And a request made AFTER the refusal settles is a no-op, not a throw.
    await expect(handle.control!.interrupt!()).resolves.toBeUndefined();
  });

  it("leaves an ordinary handle with no control at all", () => {
    // `control` is optional on `ExecHandle` precisely so the ordinary operation carries nothing.
    const prompt = new PromptExecutor({ runner: async () => ({ value: { value: "x", finishReason: "stop" }, metrics: { durationMs: 0, costUsd: 0, costSource: "table" } }) });
    expect(prompt.start(op("do it", { model: "anthropic/claude-sonnet-5" }), {}).control).toBeUndefined();
  });
});

describe("which binary, and under what environment", () => {
  it("carries a pinned binaryPath to the transport on every call", async () => {
    // "An agent answered" and "the agent we pinned answered" are different statements, and only one of
    // them is reproducible.
    const { query, seen } = capturing();
    await new AgentExecutor({ query, binaryPath: "D:\\builds\\claude.exe" }).start(op(), {}).result;
    expect(seen()?.binaryPath).toBe("D:\\builds\\claude.exe");
  });

  it("carries the environment verbatim, interpreting nothing about it", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, env: { CLAUDE_CONFIG_DIR: "/tmp/acct-a" } }).start(op(), {}).result;
    expect(seen()?.env).toEqual({ CLAUDE_CONFIG_DIR: "/tmp/acct-a" });
  });

  it("says nothing about either when the caller pinned neither, so each transport keeps its default", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), {}).result;
    expect(seen()?.binaryPath).toBeUndefined();
    expect(seen()?.env).toBeUndefined();
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

/**
 * The permission MODES on the delegated path.
 *
 * A delegated adapter gets RAW tools — it runs its own loop and calls its own built-ins, so wrapping
 * them here would double-gate what it never routes through us anyway. What it gets instead is
 * `ctx.gate`, and these pin the three places the modes are actually consumed.
 *
 * Every one of them was silently broken before the gate existed. The adapter read
 * `ctx.policy.baseline.tools` for its deny floor (missing anything authored on the STATE),
 * pre-approved every declared tool regardless of mode, and answered its permission callback by
 * calling the human approver directly — so `smart` never ran its policy and `allow` asked anyway.
 */
describe("permission modes reach a delegated agent", () => {
  const tool = (name: string, readOnly = true): Tool => ({
    description: name,
    readOnly,
    inputSchema: { type: "object" } as never,
    run: () => name,
  });

  /** A ctx carrying the real gate over the real ledger — not a stub, so precedence is exercised. */
  const gated = (opts: {
    tools: Record<string, Tool>;
    authored?: { default?: PermissionMode; tools?: Record<string, PermissionMode> };
    baseline?: PermissionBaseline;
    smart?: Record<string, SmartApprover>;
    approve?: Approver;
  }) => {
    const asked: string[] = [];
    const approve: Approver =
      opts.approve ??
      ((req) => {
        asked.push(req.tool);
        return { decision: "allow", scope: "once" };
      });
    const ledger = new PermissionLedger({ ...(opts.baseline !== undefined ? { baseline: opts.baseline } : {}) });
    const gate = createToolGate({
      ledger,
      sessionId: "s1",
      approve,
      tools: Object.fromEntries(Object.entries(opts.tools).map(([n, t]) => [n, { readOnly: t.readOnly }])),
      ...(opts.authored !== undefined ? { authored: opts.authored } : {}),
      ...(opts.smart !== undefined ? { smart: opts.smart } : {}),
    });
    return { asked, ctx: { tools: opts.tools, gate, approve } as unknown as ExecServices };
  };

  it("pre-approves ONLY the tools whose mode is `allow`", async () => {
    const { query, seen } = capturing();
    const { ctx } = gated({
      tools: { open: tool("open"), look: tool("look"), think: tool("think") },
      authored: { tools: { open: "allow", look: "ask", think: "smart" } },
    });
    await new AgentExecutor({
      query,
      injectTools: false, // route them natively, so they are eligible for the allow-list at all
    }).start(op(), ctx).result;
    // `ask` must still be asked and `smart` must still run its policy — pre-approving either is what
    // decides a call before the thing that decides it has run.
    expect(seen()?.allowedTools).toEqual(["open"]);
  });

  it("applies a deny authored on the STATE, which the baseline never carried", async () => {
    const { query, seen } = capturing();
    const { ctx } = gated({ tools: { open: tool("open") }, authored: { tools: { open: "deny" } } });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    // Never offered — not injected, not allowed. A floor needs no human.
    expect(seen()?.mcpTools).toBeUndefined();
    expect(seen()?.allowedTools).toEqual([]);
  });

  it("applies the PROFILE as a deny, without anybody naming the tool", async () => {
    const { query, seen } = capturing();
    const { ctx } = gated({
      tools: { look: tool("look", true), write: tool("write", false) },
      baseline: { profile: "read-only" },
      authored: { default: "allow" },
    });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    expect(Object.keys(seen()?.mcpTools ?? {})).toEqual(["look"]);
  });

  it("answers the native callback through the gate, so `smart` runs instead of a human", async () => {
    const { query, seen } = capturing();
    const { ctx, asked } = gated({
      tools: { open: tool("open") },
      authored: { tools: { open: "smart" } },
      smart: { open: ({ input }) => ((input as { ok?: boolean }).ok === true ? "allow" : "deny") },
    });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const ask = (input: unknown) =>
      seen()!.canUseTool!({ toolName: "mcp__dai__open", input: input as never }, { signal: new AbortController().signal });

    expect(await ask({ ok: true })).toEqual({ allow: true });
    expect(await ask({ ok: false })).toMatchObject({ allow: false });
    // The human was never involved — that is what `smart` means.
    expect(asked).toEqual([]);
  });

  it("resolves the callback's name back to the LOGICAL one the policy is written against", async () => {
    // Three vocabularies meet at that callback: the agent addresses an injected tool as
    // `mcp__dai__open` and an aliased one as `Read`, while a mode is authored against `open`.
    const { query, seen } = capturing();
    const { ctx, asked } = gated({ tools: { open: tool("open") }, authored: { tools: { open: "allow" } } });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const decision = await seen()!.canUseTool!(
      { toolName: "mcp__dai__open", input: {} },
      { signal: new AbortController().signal },
    );
    expect(decision).toEqual({ allow: true });
    expect(asked).toEqual([]); // an `allow` does not interrupt
  });

  it("escalates a tool the gate cannot classify rather than guessing", async () => {
    // The agent's own `Bash` is not a tool we registered, so there is no `readOnly` to judge it by.
    const { query, seen } = capturing();
    const { ctx, asked } = gated({ tools: {}, baseline: { profile: "read-only", default: "allow" } });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const decision = await seen()!.canUseTool!({ toolName: "Bash", input: {} }, { signal: new AbortController().signal });
    expect(decision).toEqual({ allow: true });
    expect(asked).toEqual(["Bash"]); // asked, not assumed — in either direction
  });

  it("falls back to the bare approver for a host that publishes no gate", async () => {
    // The prior behaviour, kept: an un-gated host is not silently un-gated.
    const { query, seen } = capturing();
    const asked: string[] = [];
    const approve: Approver = (req) => {
      asked.push(req.tool);
      return { decision: "deny", scope: "once" };
    };
    await new AgentExecutor({ query }).start(op(), { tools: { open: tool("open") }, approve } as unknown as ExecServices).result;
    expect(await seen()!.canUseTool!({ toolName: "open", input: {} }, { signal: new AbortController().signal })).toEqual({
      allow: false,
      reason: "denied by permission policy",
    });
    expect(asked).toEqual(["open"]);
  });
});

/**
 * `AskUserQuestion` is a QUESTION, not a permission — see {@link ASK_USER_TOOL}. Before the routing
 * existed it hit the gate as an unclassifiable native tool, so the human was asked to APPROVE being
 * asked a question, and an allow echoed the input unchanged — the agent's question resolved with no
 * answers at all.
 */
describe("AskUserQuestion routes to ctx.askUser, never to the approval gate", () => {
  const input = {
    questions: [
      {
        question: "Which library?",
        header: "Library",
        options: [
          { label: "date-fns", description: "small" },
          { label: "luxon", description: "batteries" },
        ],
      },
    ],
  };
  const signal = () => ({ signal: new AbortController().signal });

  /** A ctx whose approver RECORDS — proof the gate and the human were never involved. */
  const attended = (answers?: Record<string, string | readonly string[]>) => {
    const asked: string[] = [];
    const questioned: string[] = [];
    const approve: Approver = (req) => {
      asked.push(req.tool);
      return { decision: "allow", scope: "once" };
    };
    const ctx = {
      approve,
      askUser: async (req: { questions: { question: string }[] }) => {
        questioned.push(...req.questions.map((q) => q.question));
        return answers;
      },
    } as unknown as ExecServices;
    return { ctx, asked, questioned };
  };

  it("puts the questions to the user and carries the answers back on updatedInput", async () => {
    const { query, seen } = capturing();
    const { ctx, asked, questioned } = attended({ "Which library?": "luxon" });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const decision = await seen()!.canUseTool!({ toolName: "AskUserQuestion", input: input as never }, signal());
    expect(decision).toEqual({
      allow: true,
      updatedInput: { ...input, answers: { "Which library?": "luxon" } },
    });
    expect(questioned).toEqual(["Which library?"]);
    // The approval gate never saw it — that is the whole point of the routing.
    expect(asked).toEqual([]);
  });

  it("tells the agent to use its own judgment when the question is dismissed", async () => {
    const { query, seen } = capturing();
    const { ctx, asked } = attended(undefined);
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const decision = await seen()!.canUseTool!({ toolName: "AskUserQuestion", input: input as never }, signal());
    expect(decision).toMatchObject({ allow: false, reason: expect.stringMatching(/own best judgment/) });
    expect(asked).toEqual([]);
  });

  it("answers the same way on a run with no question surface at all", async () => {
    // Unattended is not an error: the agent should proceed, not park — and the approval gate must
    // still not be consulted, because "may you ask?" is not a question anyone should answer.
    const { query, seen } = capturing();
    const asked: string[] = [];
    const approve: Approver = (req) => {
      asked.push(req.tool);
      return { decision: "allow", scope: "once" };
    };
    await new AgentExecutor({ query }).start(op(), { approve } as unknown as ExecServices).result;
    const decision = await seen()!.canUseTool!({ toolName: "AskUserQuestion", input: input as never }, signal());
    expect(decision).toMatchObject({ allow: false, reason: expect.stringMatching(/own best judgment/) });
    expect(asked).toEqual([]);
  });

  it("treats a malformed question batch as unanswerable rather than throwing", async () => {
    const { query, seen } = capturing();
    const { ctx, questioned } = attended({ never: "asked" });
    await new AgentExecutor({ query }).start(op(), ctx).result;
    const decision = await seen()!.canUseTool!({ toolName: "AskUserQuestion", input: { questions: "?" } as never }, signal());
    expect(decision).toMatchObject({ allow: false });
    expect(questioned).toEqual([]); // nothing readable was ever put to the user
  });
});

describe("the session profile reaches a transport that has no per-tool gate", () => {
  /**
   * `codex exec` enforces entirely up front: no permission callback, no allow-list, no deny-list —
   * just `--sandbox`, chosen from the permission mode. So the profile is the only policy input it can
   * act on, and until the gate carried it the sandbox came from a statically-configured adapter
   * option: a state authoring `plan` ran under whatever that option happened to be.
   */
  const gateWithProfile = (profile: string): ExecServices =>
    ({
      approve: () => ({ decision: "allow", scope: "once" }),
      gate: createToolGate({
        ledger: new PermissionLedger({ baseline: { profile } }),
        sessionId: "s1",
        approve: () => ({ decision: "allow", scope: "once" }),
      }),
    }) as unknown as ExecServices;

  it("derives `plan` from a plan profile — the one exact correspondence", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), gateWithProfile("plan")).result;
    expect(seen()?.permissionMode).toBe("plan");
  });

  it("derives NOTHING from read-only, which has no counterpart in this vocabulary", async () => {
    // Borrowing `plan` here would tell the agent to stop acting and start planning, which is a
    // different instruction from "you may read". The per-tool gate enforces read-only instead.
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), gateWithProfile("read-only")).result;
    expect(seen()?.permissionMode).toBeUndefined();
  });

  it("lets an explicitly configured mode win — it is the more specific statement", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, permissionMode: "acceptEdits" }).start(op(), gateWithProfile("plan")).result;
    expect(seen()?.permissionMode).toBe("acceptEdits");
  });

  it("maps read-only through `readOnlyProfileMode` where a transport declared the mapping exact", async () => {
    // Codex's arrangement: its `plan` carries no behaviour — it is nothing but `--sandbox read-only`.
    const { query, seen } = capturing();
    await new AgentExecutor({ query, readOnlyProfileMode: "plan" }).start(op(), gateWithProfile("read-only")).result;
    expect(seen()?.permissionMode).toBe("plan");
  });
});

/**
 * A `read-only` profile as CONFIGURATION, not only as a callback.
 *
 * The gate escalates what it cannot classify — but "may this agent run `Bash`?" under a profile that
 * means "no writes" is not a human question, and before this the whole restriction hung on somebody
 * answering it correctly every time. The write-capable built-ins are a known fact about the
 * transport, so they are denied up front; the callback stays the floor for everything the list
 * cannot name.
 */
describe("a narrowing profile reaches the agent up front", () => {
  const gateWithProfile = (profile: string): ExecServices =>
    ({
      approve: () => ({ decision: "allow", scope: "once" }),
      gate: createToolGate({
        ledger: new PermissionLedger({ baseline: { profile } }),
        sessionId: "s1",
        approve: () => ({ decision: "allow", scope: "once" }),
      }),
    }) as unknown as ExecServices;

  it("denies claude's write-capable built-ins under `read-only`", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), gateWithProfile("read-only")).result;
    expect(seen()?.disallowedTools).toEqual(expect.arrayContaining(["Bash", "Edit", "Write", "Task"]));
  });

  it("denies NOTHING under `plan` — `--permission-mode plan` is the exact channel, and it still allows read-only use of these tools", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query }).start(op(), gateWithProfile("plan")).result;
    expect(seen()?.permissionMode).toBe("plan");
    expect(seen()?.disallowedTools).toBeUndefined();
  });

  it("denies nothing under `full`, and nothing with no gate at all", async () => {
    const full = capturing();
    await new AgentExecutor({ query: full.query }).start(op(), gateWithProfile("full")).result;
    expect(full.seen()?.disallowedTools).toBeUndefined();
    const bare = capturing();
    await new AgentExecutor({ query: bare.query }).start(op(), {}).result;
    expect(bare.seen()?.disallowedTools).toBeUndefined();
  });

  it("lets a transport state its own list — codex passes [] and answers with its sandbox instead", async () => {
    const { query, seen } = capturing();
    await new AgentExecutor({ query, mutatingNativeTools: [] }).start(op(), gateWithProfile("read-only")).result;
    expect(seen()?.disallowedTools).toBeUndefined();
  });

  it("refuses a narrowing profile on a transport that enforces nothing, rather than running unheld", async () => {
    const { query } = capturing();
    const none = new AgentExecutor({ query, capabilities: { ...DELEGATED_CAPS, policyEnforcement: "none" } });
    const result = await none.start(op(), gateWithProfile("read-only")).result;
    expect(!isOk(result) && result.error.reason).toMatch(/read-only.*enforces no policy/s);
    // `full` excludes nothing, so there is nothing to fail to enforce — the same transport runs.
    const runs = await none.start(op(), gateWithProfile("full")).result;
    expect(isOk(runs)).toBe(true);
  });
});
