/**
 * Usage readings through the agent executor — how full the context is after a call, and what the
 * account has left — from the provider's vocabulary to the neutral readings, the event stream, the
 * record's entries and the limits board.
 */
import { describe, expect, it } from "vitest";
import { promptOp, type ExecEvent, type ExecServices, type LimitReading } from "@declarative-ai/exec";
import type { LlmOutput } from "@declarative-ai/llm";
import {
  AgentExecutor,
  compactionOf,
  contextDetailOfClaude,
  contextTokensOfUsage,
  limitReadingOfClaudeUsage,
  limitReadingOfRateLimitEvent,
  USAGE_LIMIT_CODE,
  type AgentQuery,
} from "../src/index.js";

const op = () => promptOp({ user: "go", output: { name: "answer", schema: { type: "string" } } });

// VERBATIM from a stored claude-cli record (2026-09-14).
const RATE_LIMIT_EVENT = {
  type: "rate_limit_event",
  rate_limit_info: { status: "allowed", resetsAt: 1789359000, rateLimitType: "five_hour", overageStatus: "rejected", overageDisabledReason: "org_level_disabled", isUsingOverage: false },
  uuid: "44fcb9b0-637e-4acd-9464-14e3302c80ce",
  session_id: "ec034024-2327-408c-a2a0-d22207f3ccef",
};

describe("claude's vocabulary, normalized", () => {
  it("reads a stream rate_limit_event as a SPARSE reading: the state and the reset, no percent", () => {
    const r = limitReadingOfRateLimitEvent(RATE_LIMIT_EVENT, "claude-cli", "2026-09-14T00:00:00.000Z")!;
    expect(r).toMatchObject({ route: "claude-cli", status: "ok", source: "stream" });
    expect(r.complete).toBeUndefined();
    expect(r.windows).toEqual([
      { id: "five_hour", label: "5-hour", minutes: 300, usedPercent: null, resetsAt: new Date(1789359000 * 1000).toISOString(), status: "ok" },
    ]);
  });

  it("reads a rejected event as exhausted, at 100 when no figure was given", () => {
    const r = limitReadingOfRateLimitEvent({ ...RATE_LIMIT_EVENT, rate_limit_info: { ...RATE_LIMIT_EVENT.rate_limit_info, status: "rejected" } }, "claude-cli")!;
    expect(r.status).toBe("exhausted");
    expect(r.windows[0]).toMatchObject({ usedPercent: 100, status: "exhausted" });
  });

  it("reads get_usage as a COMPLETE reading with every window and the plan", () => {
    const r = limitReadingOfClaudeUsage(
      {
        subscription_type: "max",
        rate_limits_available: true,
        rate_limits: {
          five_hour: { utilization: 62, resets_at: "2026-09-24T14:05:00Z" },
          seven_day: { utilization: 33, resets_at: "2026-09-28T09:00:00Z" },
          seven_day_opus: { utilization: 71, resets_at: "2026-09-28T09:00:00Z" },
          seven_day_sonnet: null,
        },
      },
      "claude-cli",
    )!;
    expect(r).toMatchObject({ plan: "max", complete: true, source: "query", status: "ok" });
    expect(r.windows.map((w) => [w.id, w.usedPercent, w.model])).toEqual([
      ["five_hour", 62, undefined],
      ["seven_day", 33, undefined],
      ["seven_day_opus", 71, "opus"],
    ]);
  });

  it("gives nothing for a signed-out answer (rate_limits: null)", () => {
    expect(limitReadingOfClaudeUsage({ subscription_type: null, rate_limits_available: false, rate_limits: null }, "claude-cli")).toBeUndefined();
  });

  it("counts what a response holds: input, both caches, and output", () => {
    expect(contextTokensOfUsage({ input_tokens: 6, cache_creation_input_tokens: 1000, cache_read_input_tokens: 75000, output_tokens: 204 })).toBe(76210);
    expect(contextTokensOfUsage({})).toBeUndefined();
  });

  it("reads get_context_usage into categories with their items, dropping free space and the buffer", () => {
    const d = contextDetailOfClaude({
      categories: [
        { name: "System prompt", tokens: 9800 },
        { name: "System tools", tokens: 11500 },
        { name: "MCP tools", tokens: 2800 },
        { name: "Messages", tokens: 49010 },
        { name: "Autocompact buffer", tokens: 33000 },
        { name: "Free space", tokens: 120000 },
      ],
      totalTokens: 73110,
      maxTokens: 200000,
      autoCompactThreshold: 166000,
      isAutoCompactEnabled: true,
      model: "claude-sonnet-5",
      systemTools: [{ name: "Bash", tokens: 2900 }],
      mcpTools: [
        { name: "mcp__dai__approve", serverName: "dai", tokens: 600, isLoaded: true },
        { name: "mcp__github__x", serverName: "github", tokens: 0, isLoaded: false },
      ],
      memoryFiles: [],
      agents: [],
      messageBreakdown: { toolCallTokens: 3100, toolResultTokens: 30200, attachmentTokens: 3700, assistantMessageTokens: 9800, userMessageTokens: 2200, redirectedContextTokens: 0, unattributedTokens: 0, toolCallsByType: [{ name: "Read", callTokens: 900, resultTokens: 21600 }, { name: "Bash", callTokens: 700, resultTokens: 6200 }], attachmentsByType: [] },
      gridRows: [],
      apiUsage: null,
    })!;
    expect(d).toMatchObject({ window: 200000, autoCompactAt: 166000, used: 73110, model: "claude-sonnet-5" });
    expect(d.breakdown!.map((c) => c.name)).toEqual(["System prompt", "System tools", "MCP tools", "Messages"]);
    expect(d.breakdown![1]!.parts).toEqual([{ name: "Bash", tokens: 2900 }]);
    expect(d.breakdown![2]!.parts).toEqual([
      { name: "dai", tokens: 600, parts: [{ name: "approve", tokens: 600 }] },
      { name: "1 more", tokens: 0, note: "deferred — not in context yet" },
    ]);
    const messages = d.breakdown![3]!.parts!;
    expect(messages[0]).toEqual({ name: "Tool results", tokens: 30200, parts: [{ name: "Read", tokens: 21600 }, { name: "Bash", tokens: 6200 }] });
    expect(messages.map((m) => m.name)).toEqual(["Tool results", "Tool calls", "Your messages", "Answers", "Attachments"]);
  });

  it("recognizes the compaction boundary and what it says", () => {
    expect(compactionOf({ type: "system", subtype: "compact_boundary", compact_metadata: { trigger: "auto", pre_tokens: 168400, post_tokens: 42000, duration_ms: 18000 } })).toEqual({
      trigger: "auto",
      before: 168400,
      after: 42000,
      durationMs: 18000,
    });
    expect(compactionOf({ type: "system", subtype: "init" })).toBeUndefined();
  });
});

describe("the executor carries the readings", () => {
  const turn = (extra: unknown[] = []): AgentQuery =>
    async function* () {
      yield { type: "provider_event", event: { type: "system", subtype: "init", model: "claude-sonnet-5" } };
      yield { type: "provider_event", event: RATE_LIMIT_EVENT };
      yield {
        type: "assistant",
        message: { role: "assistant", model: "claude-sonnet-5", content: [{ type: "text", text: "done" }], usage: { input_tokens: 6, cache_creation_input_tokens: 1000, cache_read_input_tokens: 75000, output_tokens: 204 } },
        text: "done",
      };
      for (const m of extra) yield m as never;
      yield { type: "result", result: { text: "done", rawUsage: { modelUsage: { "claude-sonnet-5": { contextWindow: 200000 } } } } };
    };

  async function run(query: AgentQuery, ctx: ExecServices = {}) {
    const handle = new AgentExecutor({ query, label: "claude-cli" }).start(op(), { ...ctx, returnRecord: true });
    const events: ExecEvent[] = [];
    const drain = (async () => {
      for await (const e of handle.events) events.push(e);
    })();
    const result = await handle.result;
    await drain;
    return { result, events, record: (result as { record?: LlmOutput }).record };
  }

  it("streams a context reading and keeps it on the call's last assistant entry", async () => {
    const { events, record } = await run(turn([{ type: "reading", reading: { context: { breakdown: [{ name: "Messages", tokens: 49010 }], autoCompactAt: 166000 } } }]));
    const context = events.find((e): e is Extract<ExecEvent, { type: "context" }> => e.type === "context")!.reading;
    expect(context).toMatchObject({ used: 76210, window: 200000, model: "claude-sonnet-5", autoCompactAt: 166000, breakdown: [{ name: "Messages", tokens: 49010 }] });
    const last = [...(record!.entries ?? [])].reverse().find((e) => e.kind === "message" && e.role === "assistant");
    expect(last).toMatchObject({ context: { used: 76210, window: 200000 } });
  });

  it("streams each limit reading and reports it, and the send, to ctx.usage", async () => {
    const sent: string[] = [];
    const heard: LimitReading[] = [];
    const { events } = await run(turn(), { usage: { sent: (r) => void sent.push(r), limits: (r) => void heard.push(r) } });
    expect(sent).toEqual(["claude-cli"]);
    expect(heard).toHaveLength(1);
    expect(heard[0]).toMatchObject({ route: "claude-cli", windows: [{ id: "five_hour" }] });
    expect(events.some((e) => e.type === "limits")).toBe(true);
  });

  it("fails a call refused on a spent window as usage_limit, with when it resets — never as a retriable throttle", async () => {
    const resets = Math.floor(Date.now() / 1000) + 3600;
    const refused: AgentQuery = async function* () {
      yield { type: "provider_event", event: { type: "rate_limit_event", rate_limit_info: { status: "rejected", resetsAt: resets, rateLimitType: "five_hour", isUsingOverage: false } } };
      yield { type: "assistant", errorCode: "rate_limit", message: { role: "assistant", content: [{ type: "text", text: "Claude usage limit reached" }] } };
      yield { type: "other", error: "Claude usage limit reached" };
    };
    const { result } = await run(refused);
    expect("error" in result && result.error).toMatchObject({ classification: "out-of-credits", code: USAGE_LIMIT_CODE });
    const error = (result as { error: { retryAfterMs?: number; detail?: { resetsAt?: string } } }).error;
    expect(error.detail?.resetsAt).toBe(new Date(resets * 1000).toISOString());
    expect(error.retryAfterMs).toBeGreaterThan(3_500_000);
  });

  it("does not call a refusal usage_limit while extra usage is being drawn", async () => {
    const refused: AgentQuery = async function* () {
      yield { type: "provider_event", event: { type: "rate_limit_event", rate_limit_info: { status: "rejected", resetsAt: 1, rateLimitType: "five_hour", isUsingOverage: true } } };
      yield { type: "other", error: "something else", errorCode: "server_error" };
    };
    const { result } = await run(refused);
    expect("error" in result && result.error.code).not.toBe(USAGE_LIMIT_CODE);
  });
});

describe("Compact now: a slash command reaches the agent as a command", () => {
  it("recognizes a slash command and not a path", async () => {
    const { isSlashCommand } = await import("../src/index.js");
    expect(isSlashCommand("/compact")).toBe(true);
    expect(isSlashCommand("/compact keep the inbox decision")).toBe(true);
    expect(isSlashCommand("/etc/hosts is broken")).toBe(false);
    expect(isSlashCommand("please /compact")).toBe(false);
  });

  it("sends it bare, without the system text in front", async () => {
    let prompt: string | undefined;
    const query: AgentQuery = async function* (opts) {
      prompt = opts.prompt;
      yield { type: "result", result: { text: "compacted" } };
    };
    await new AgentExecutor({ query }).start(promptOp({ user: "/compact keep X", system: "You are the writer.", output: { name: "answer", schema: { type: "string" } } } as never), {}).result;
    expect(prompt).toBe("/compact keep X");
  });
});
