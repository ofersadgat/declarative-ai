/**
 * Reading an agent's stream.
 *
 * Every message here is a real line off a `claude 2.1.142` run — the same lines the Agent SDK hands
 * through, because it drives that binary as a subprocess. This is the mapping that used to collapse
 * everything but the terminal `result` to `{type: "other"}`, which is where the trace, the tool log, the
 * token counts and the whole conversation went.
 */
import { describe, expect, it } from "vitest";
import { agentFinishReason, readAgentMessage, readAgentUsage } from "../src/streamMessages.js";

describe("readAgentUsage — what the run consumed, priced-ably", () => {
  it("treats input_tokens as the FRESH input and sums the billed total", () => {
    // The reading that is easy to get backwards and expensive when you do: Anthropic reports cache
    // reads and writes ALONGSIDE `input_tokens`, not inside it. The live run behind this billed 6 fresh
    // tokens and 30,277 cache-creation tokens — reading `input_tokens` as the total under-reports it by
    // three orders of magnitude.
    expect(
      readAgentUsage({
        input_tokens: 6,
        cache_creation_input_tokens: 30277,
        cache_read_input_tokens: 0,
        output_tokens: 8,
        cache_creation: { ephemeral_1h_input_tokens: 30277, ephemeral_5m_input_tokens: 0 },
      }),
    ).toEqual({
      noCacheTokens: 6,
      cacheWriteTokens: 30277,
      cacheReadTokens: 0,
      cacheWrite1hTokens: 30277,
      inputTokens: 30283,
      outputTokens: 8,
      totalTokens: 30291,
    });
  });

  it("keeps the TTL split, because the two tiers are priced differently", () => {
    // A 1-hour cache write is roughly twice the base rate where a 5-minute one is roughly 1.25×, so a
    // single `cacheWriteTokens` figure cannot be priced.
    const usage = readAgentUsage({ input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 100, cache_creation: { ephemeral_1h_input_tokens: 40 } });
    expect(usage).toMatchObject({ cacheWriteTokens: 100, cacheWrite1hTokens: 40 });
  });

  it("reports nothing rather than zero when the provider said nothing", () => {
    // Absent and zero are different claims, and a budget reading the second would believe a free call.
    expect(readAgentUsage(undefined)).toBeUndefined();
    expect(readAgentUsage({})).toBeUndefined();
  });
});

describe("agentFinishReason — why the run ended, in the vocabulary a provider call uses", () => {
  it("reads an ordinary finish off stop_reason", () => {
    expect(agentFinishReason({ subtype: "success", stop_reason: "end_turn" })).toBe("stop");
  });

  it("maps a truncated turn to `length`, not to `stop`", () => {
    expect(agentFinishReason({ subtype: "success", stop_reason: "max_tokens" })).toBe("length");
  });

  it("lets the agent's OWN loop limits win over the last turn's stop_reason", () => {
    // A run that exhausted its turn cap reports a perfectly ordinary `stop_reason`, so reading only
    // that would report a truncated run as a clean finish — which is what hardcoding "stop" did for
    // every single run.
    expect(agentFinishReason({ subtype: "error_max_turns", stop_reason: "end_turn" })).toBe("max-turns");
    expect(agentFinishReason({ subtype: "error_max_budget_usd", stop_reason: "end_turn" })).toBe("max-budget");
    expect(agentFinishReason({ subtype: "error_during_execution", stop_reason: null })).toBe("error");
  });

  it("falls back to `stop` for a success that said nothing more specific", () => {
    expect(agentFinishReason({ subtype: "success", stop_reason: null })).toBe("stop");
  });

  it("reads a DELIBERATE end off terminal_reason, over the subtype that accompanies it", () => {
    // `error_during_execution` describes how the loop exited; `aborted_streaming` says the exit was
    // requested. The second is the more specific claim and wins.
    expect(agentFinishReason({ subtype: "error_during_execution", terminal_reason: "aborted_streaming", stop_reason: null })).toBe("aborted");
    expect(agentFinishReason({ subtype: "success", terminal_reason: "max_turns" })).toBe("max-turns");
    expect(agentFinishReason({ subtype: "success", terminal_reason: "budget_exhausted" })).toBe("max-budget");
  });

  it("says NOTHING when it has nothing to go on, rather than guessing", () => {
    expect(agentFinishReason({})).toBeUndefined();
  });
});

describe("readAgentMessage — one message, normalized", () => {
  it("normalizes the terminal result, cost, session and all", () => {
    const msg = readAgentMessage({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ZEPHYR",
      stop_reason: "end_turn",
      session_id: "51caeb77",
      total_cost_usd: 0.189,
      usage: { input_tokens: 6, output_tokens: 8 },
      modelUsage: { "claude-opus-4-7": { costUSD: 0.189 } },
    });
    expect(msg.type).toBe("result");
    expect(msg.result).toMatchObject({ text: "ZEPHYR", costUsd: 0.189, sessionId: "51caeb77", finishReason: "stop" });
    expect(msg.result?.usage).toMatchObject({ inputTokens: 6, outputTokens: 8 });
    // Both usage objects are kept: `usage` is the main loop only, `modelUsage` covers subagents and
    // compaction. Keeping the pair is what makes `costUsd` recomputable if our reading is incomplete.
    expect(msg.result?.rawUsage).toEqual({ usage: { input_tokens: 6, output_tokens: 8 }, modelUsage: { "claude-opus-4-7": { costUSD: 0.189 } } });
  });

  it("reads the constrained value off `structured_output`, beside the prose rather than instead of it", () => {
    // ✅ OBSERVED (claude 2.1.142) on a `--json-schema` run: `result` came back as the prose
    // "Red, yellow, and blue." while the value rode on `structured_output`. A caller that parsed
    // `result` would be parsing the summary of the work rather than the answer.
    const msg = readAgentMessage({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "Red, yellow, and blue.",
      structured_output: { colors: ["red", "yellow", "blue"] },
    });
    expect(msg.result?.structured).toEqual({ colors: ["red", "yellow", "blue"] });
    expect(msg.result?.text).toBe("Red, yellow, and blue.");
  });

  it("leaves it absent when the run was not asked for a shape", () => {
    expect(readAgentMessage({ type: "result", subtype: "success", is_error: false, result: "hi" }).result?.structured).toBeUndefined();
  });

  it("treats a run reported as FAILED as an error, not as the agent's answer", () => {
    // `is_error` is independent of `subtype` AND of the exit code, so reading only the discriminator
    // reported "Not logged in · Please run /login" as a successful answer.
    expect(readAgentMessage({ type: "result", subtype: "success", is_error: true, result: "Not logged in · Please run /login" })).toEqual({
      type: "other",
      error: "Not logged in · Please run /login",
    });
  });

  it("treats an INTERRUPTED run as a result, not as a failure, whatever `is_error` says", () => {
    // ✅ OBSERVED via a live `Query.interrupt()`: the turn comes back as `is_error: true`,
    // `subtype: "error_during_execution"`, `terminal_reason: "aborted_streaming"`, and NO result text.
    // Read at face value that settles the handle with an error and discards an answer the agent had
    // already written — the exact opposite of what someone pressing Stop is asking for. The partial
    // answer lives on the assistant turns; the terminal message only has to not lie about how it ended.
    const msg = readAgentMessage({
      type: "result",
      subtype: "error_during_execution",
      is_error: true,
      stop_reason: null,
      terminal_reason: "aborted_streaming",
      session_id: "s1",
      total_cost_usd: 0.0006,
    });
    expect(msg.type).toBe("result");
    expect(msg.result).toMatchObject({ finishReason: "aborted", sessionId: "s1", costUsd: 0.0006 });
  });

  it("still reports a genuinely failed run as an error", () => {
    // The distinction has to cut both ways, or the aborted case becomes a hole that swallows real
    // failures.
    expect(readAgentMessage({ type: "result", subtype: "error_during_execution", is_error: true, terminal_reason: "api_error", result: "upstream 500" })).toEqual({
      type: "other",
      error: "upstream 500",
    });
  });

  it("names the failure even when nothing explained it", () => {
    expect(readAgentMessage({ type: "result", is_error: true })).toEqual({ type: "other", error: expect.stringMatching(/reported a failed run/) });
  });

  it("keeps a thinking block's SIGNATURE, which the next turn cannot do without", () => {
    // A signed thinking block must come back byte-identical or the provider rejects the conversation —
    // so dropping the signature does not lose a trace, it breaks the next call.
    const msg = readAgentMessage({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "thinking", thinking: "let me check", signature: "abc123" }] },
    });
    expect(msg.type).toBe("assistant");
    expect(msg.thinking).toEqual([{ text: "let me check", providerMetadata: { anthropic: { signature: "abc123" } } }]);
  });

  it("keeps a REDACTED thinking block, whose whole content is the opaque data", () => {
    const msg = readAgentMessage({ type: "assistant", message: { role: "assistant", content: [{ type: "redacted_thinking", data: "EncryptedBlob" }] } });
    expect(msg.thinking).toEqual([{ text: "", providerMetadata: { anthropic: { data: "EncryptedBlob" } } }]);
  });

  it("reads tool calls off an assistant turn and tool results off the user turn that answers it", () => {
    const call = readAgentMessage({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "Read", input: { file_path: "a.txt" } }] },
    });
    expect(call.toolCalls).toEqual([{ toolCallId: "toolu_1", toolName: "Read", input: { file_path: "a.txt" } }]);

    const result = readAgentMessage({
      type: "user",
      message: { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ZEPHYR" }] },
    });
    expect(result.type).toBe("user");
    expect(result.toolResults).toEqual([{ toolCallId: "toolu_1", output: "ZEPHYR" }]);
  });

  it("carries the provider's own message VERBATIM alongside the projections", () => {
    // `LlmOutput.messages` is documented as the provider's log rather than a reconstruction, and a
    // reconstruction is exactly what the synthesized one-turn log was.
    const message = { role: "assistant", content: [{ type: "text", text: "Hi!" }], id: "msg_1", model: "claude-opus-4-7" };
    expect(readAgentMessage({ type: "assistant", message }).message).toEqual(message);
  });

  it("separates the answer TEXT from the reasoning, so an offset means something", () => {
    const msg = readAgentMessage({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "thinking", thinking: "hmm" }, { type: "text", text: "Hi!" }] },
    });
    expect(msg.text).toBe("Hi!");
  });

  it("turns a text delta into a `partial`, and a thinking delta into a `thinking-partial`", () => {
    expect(readAgentMessage({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "ZEP" } } })).toEqual({
      type: "partial",
      delta: "ZEP",
    });
    // Reasoning never rides the answer's stream — the one thing §5.1 says must never happen — but a
    // consumer showing a minutes-long think as it happens needs the delta on its own channel.
    expect(
      readAgentMessage({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "hmm" } } }),
    ).toEqual({ type: "thinking-partial", delta: "hmm" });
  });

  it("carries the envelope's sidechain tag, so a subagent's turn is attributable", () => {
    const msg = readAgentMessage({
      type: "assistant",
      parent_tool_use_id: "toolu_task_1",
      message: { role: "assistant", content: [{ type: "text", text: "subagent speaking" }] },
    });
    expect(msg.parentToolUseId).toBe("toolu_task_1");
    // And its absence means the main thread — never an empty string.
    expect(readAgentMessage({ type: "assistant", message: { role: "assistant", content: [] } }).parentToolUseId).toBeUndefined();
  });

  it("keeps a delta it cannot read as an opaque provider event", () => {
    // `input_json_delta` (tool arguments assembling), signature deltas, block starts — bookkeeping,
    // whose content arrives again on the finished turn.
    expect(
      readAgentMessage({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: "{\"pa" } } }).type,
    ).toBe("provider_event");
  });

  it("forwards everything with no neutral home OPAQUELY rather than discarding it", () => {
    // The system subtypes are an open list — init, compact_boundary, hook_started, task_progress,
    // api_retry, permission_denied — and naming each here would teach the neutral layer one provider's
    // vocabulary for events it cannot act on.
    const init = { type: "system", subtype: "init", tools: ["Read"], model: "claude-opus-4-7" };
    expect(readAgentMessage(init)).toEqual({ type: "provider_event", event: init });
    const rate = { type: "rate_limit_event", rate_limit_info: { status: "allowed" } };
    expect(readAgentMessage(rate)).toEqual({ type: "provider_event", event: rate });
    const future = { type: "something_invented_next_release" };
    expect(readAgentMessage(future)).toEqual({ type: "provider_event", event: future });
  });
});
