/** Usage readings on the subprocess transports: claude's end-of-turn question, the no-turn probe, and codex's session file. */
import { describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createCliAgentQuery } from "../src/cliQuery.js";
import { probeClaudeUsage } from "../src/claudeUsageProbe.js";
import { readCodexLimits, readCodexThreadReadings, readingsOfCodexTokenCount } from "../src/codexUsage.js";
import type { AgentProcess, SpawnProcess } from "../src/process.js";

/** A process whose stdout answers what is written to it: `reply(line)` returns the lines to emit. */
function conversing(first: string[], reply: (written: string) => string[]): { spawn: SpawnProcess; written: string[]; argv: string[][] } {
  const written: string[] = [];
  const argv: string[][] = [];
  const spawn: SpawnProcess = (a) => {
    argv.push(a);
    const queue: string[] = [...first];
    let ended = false;
    let wake: (() => void) | undefined;
    const proc: AgentProcess = {
      lines: (async function* () {
        for (;;) {
          while (queue.length > 0) yield queue.shift()!;
          if (ended) return;
          await new Promise<void>((r) => (wake = r));
        }
      })(),
      kill: () => {
        ended = true;
        wake?.();
      },
      exit: Promise.resolve(0),
      write: (line) => {
        written.push(line);
        queue.push(...reply(line));
        wake?.();
      },
      endInput: () => {
        ended = true;
        wake?.();
      },
    };
    return proc;
  };
  return { spawn, written, argv };
}

const RESULT = '{"type":"result","subtype":"success","is_error":false,"result":"done","session_id":"s1"}';

describe("claude CLI: how full the context is, asked as the turn ends", () => {
  it("asks get_context_usage after the result and yields the reading BEFORE the result", async () => {
    const { spawn, written } = conversing([RESULT], (line) => {
      const msg = JSON.parse(line) as { request_id?: string; request?: { subtype?: string } };
      if (msg.request?.subtype !== "get_context_usage") return [];
      return [JSON.stringify({ type: "control_response", response: { subtype: "success", request_id: msg.request_id, response: { categories: [{ name: "Messages", tokens: 900 }], maxTokens: 200000, totalTokens: 900, autoCompactThreshold: 166000 } } })];
    });
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "go" })) seen.push(m);
    expect(JSON.parse(written[0]!)).toMatchObject({ type: "control_request", request: { subtype: "get_context_usage" } });
    expect(seen.map((m) => m.type)).toEqual(["reading", "result"]);
    expect(seen[0]!.reading!.context).toMatchObject({ window: 200000, autoCompactAt: 166000, breakdown: [{ name: "Messages", tokens: 900 }] });
  });

  it("settles without a reading when the process never answers", async () => {
    const { spawn } = conversing([RESULT], () => []);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn, contextUsageTimeoutMs: 20 })({ prompt: "go" })) seen.push(m);
    expect(seen.map((m) => m.type)).toEqual(["result"]);
  });
});

describe("claude: the allowance, asked without spending a turn", () => {
  it("initializes, asks get_usage, writes no prompt, and reads the windows", async () => {
    const { spawn, written, argv } = conversing([], (line) => {
      const msg = JSON.parse(line) as { request_id?: string; request?: { subtype?: string } };
      if (msg.request?.subtype !== "get_usage") return [];
      return [JSON.stringify({ type: "control_response", response: { subtype: "success", request_id: msg.request_id, response: { subscription_type: "max", rate_limits: { five_hour: { utilization: 62, resets_at: "2026-09-24T14:05:00Z" }, seven_day: { utilization: 33, resets_at: "2026-09-28T09:00:00Z" } } } } })];
    });
    const answer = await probeClaudeUsage({ spawn, command: "claude" });
    expect(argv[0]).toEqual(["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]);
    expect(written.map((w) => JSON.parse(w).type)).toEqual(["control_request", "control_request"]);
    expect(answer.reading).toMatchObject({ plan: "max", complete: true, windows: [{ id: "five_hour", usedPercent: 62 }, { id: "seven_day", usedPercent: 33 }] });
  });

  it("reports an older claude that refuses the question as unavailable", async () => {
    const { spawn } = conversing([], (line) => {
      const msg = JSON.parse(line) as { request_id?: string; request?: { subtype?: string } };
      return msg.request?.subtype === "get_usage" ? [JSON.stringify({ type: "control_response", response: { subtype: "error", request_id: msg.request_id, error: "Unsupported control request subtype: get_usage" } })] : [];
    });
    expect(await probeClaudeUsage({ spawn })).toEqual({ unavailable: "Unsupported control request subtype: get_usage" });
  });
});

// VERBATIM from codex-cli 0.147.0's session file (2026-09-24).
const TOKEN_COUNT = {
  type: "token_count",
  info: {
    total_token_usage: { input_tokens: 76161, cached_input_tokens: 65280, output_tokens: 198, total_tokens: 76359 },
    last_token_usage: { input_tokens: 15403, cached_input_tokens: 15104, output_tokens: 13, total_tokens: 15416 },
    model_context_window: 258400,
  },
  rate_limits: {
    limit_id: "codex",
    primary: { used_percent: 1.0, window_minutes: 300, resets_at: 1790245145 },
    secondary: { used_percent: 0.0, window_minutes: 10080, resets_at: 1790728949 },
    plan_type: "plus",
    rate_limit_reached_type: null,
  },
};

describe("codex: the session file", () => {
  it("reads a token_count as a context reading and a complete limit reading", () => {
    const r = readingsOfCodexTokenCount(TOKEN_COUNT, "codex-cli", "2026-09-24T05:19:42.390Z");
    expect(r.context).toEqual({ used: 15416, window: 258400, at: "2026-09-24T05:19:42.390Z" });
    expect(r.limits).toMatchObject({ plan: "plus", complete: true, status: "ok", source: "file" });
    expect(r.limits!.windows.map((w) => [w.id, w.label, w.usedPercent])).toEqual([
      ["primary", "5-hour", 1],
      ["secondary", "Weekly", 0],
    ]);
  });

  it("marks the window codex says ran out as exhausted", () => {
    const r = readingsOfCodexTokenCount({ ...TOKEN_COUNT, rate_limits: { ...TOKEN_COUNT.rate_limits, rate_limit_reached_type: "primary" } }, "codex-cli", "t");
    expect(r.limits!.status).toBe("exhausted");
    expect(r.limits!.windows[0]!.status).toBe("exhausted");
  });

  it("finds a thread's rollout by its id, and the newest limits across sessions", async () => {
    const home = mkdtempSync(join(tmpdir(), "codex-home-"));
    const day = join(home, "sessions", "2026", "09", "24");
    mkdirSync(day, { recursive: true });
    const line = (payload: unknown, ts: string) => JSON.stringify({ timestamp: ts, type: "event_msg", payload });
    writeFileSync(join(day, "rollout-2026-09-24T05-19-21-thread-1.jsonl"), [line({ type: "session_meta" }, "a"), line(TOKEN_COUNT, "2026-09-24T05:19:42.390Z")].join("\n"));
    const r = await readCodexThreadReadings("thread-1", "codex-cli", home);
    expect(r.context!.used).toBe(15416);
    expect((await readCodexLimits({ home }))!.windows).toHaveLength(2);
    expect(await readCodexThreadReadings("missing", "codex-cli", home)).toEqual({});
  });
});
