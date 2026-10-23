/**
 * The CLI transports as executors.
 *
 * The label assertions look trivial and are not. Each subclass passes COMPUTED values to `super`
 * (its label, its capability record), so anything it reads back must come from what the base actually
 * holds. Redeclaring `options` as a constructor parameter property — the obvious spelling, and the one
 * this file had first — shadows the base's field, so the two diverge silently and every failure from
 * a CLI agent came out labelled `claude-code`. It surfaced only against a real binary.
 */
import { describe, expect, it } from "vitest";
import { isOk, promptOp, type ExecServices } from "@declarative-ai/exec";
import type { AgentQuery, AgentQueryOptions } from "@declarative-ai/agents-api";
import { AgentCliExecutor, AgentCodexExecutor } from "../src/cliExecutor.js";
import { CLI_CONFIG_ONLY_CAPS, CLI_DELEGATED_CAPS } from "../src/runtime.js";
import { CODEX_CAPS } from "../src/codexRuntime.js";
import type { SpawnProcess } from "../src/process.js";

const op = () => promptOp({ user: "go", output: { name: "a", schema: { type: "string" } } });
const failing: AgentQuery = async function* () {
  yield { type: "other", error: "boom" };
};

describe("AgentCliExecutor", () => {
  it("reports failures under its OWN name, not the base class's", async () => {
    const result = await new AgentCliExecutor({ query: failing }).start(op(), {}).result;
    expect(!isOk(result) && result.error.reason).toContain("claude-cli");
    expect(!isOk(result) && result.error.reason).not.toContain("claude-code");
  });

  it("declares callback enforcement with tools injected, config enforcement without", () => {
    // Injection decides what ENFORCES the policy, not merely where the tools come from.
    expect(new AgentCliExecutor().capabilities).toEqual(CLI_DELEGATED_CAPS);
    expect(new AgentCliExecutor({ injectTools: false }).capabilities).toEqual(CLI_CONFIG_ONLY_CAPS);
  });

  it("still honours an explicitly passed label", () => {
    // The default is applied BEFORE the caller's options are spread, so naming one wins.
    expect(AgentCliExecutor.kind).toBe("agent-cli");
  });

  it("launches `claude` again when it lost a sign-in refresh before doing anything, and answers", async () => {
    // The two lines claude 2.1.287 ends such a run with, in the shape its session file recorded them
    // on 2026-10-23: a `<synthetic>` turn coded `server_error`, then an `is_error` result repeating it.
    const race =
      "Failed to refresh OAuth token: another Claude Code process is refreshing it or exited mid-refresh. This is usually transient; retry in a minute, and if it persists close other Claude Code processes or sign in again";
    const lost = [
      JSON.stringify({ type: "system", subtype: "init", session_id: "s1", model: "claude-opus-5-5" }),
      JSON.stringify({
        type: "assistant",
        message: { role: "assistant", model: "<synthetic>", content: [{ type: "text", text: race }], usage: { input_tokens: 0, output_tokens: 0 } },
        parent_tool_use_id: null,
        session_id: "s1",
        error: "server_error",
      }),
      JSON.stringify({ type: "result", subtype: "success", is_error: true, result: race, session_id: "s1", total_cost_usd: 0 }),
    ];
    const won = [JSON.stringify({ type: "result", subtype: "success", is_error: false, result: "the answer", session_id: "s1" })];
    let launches = 0;
    const spawn: SpawnProcess = () => {
      const lines = ++launches === 1 ? lost : won;
      return {
        lines: (async function* () {
          yield* lines;
        })(),
        kill: () => {},
        exit: Promise.resolve(0),
        write: () => {},
        endInput: () => {},
      };
    };
    const waits: number[] = [];
    const executor = new AgentCliExecutor({ spawn, relaunch: { wait: async (ms) => void waits.push(ms) } });
    const result = await executor.start(op(), {}).result;
    expect(isOk(result) && result.value).toBe("the answer");
    expect(launches).toBe(2);
    expect(waits).toEqual([5_000]);
  });
});

describe("AgentCodexExecutor", () => {
  it("reports failures as codex", async () => {
    const result = await new AgentCodexExecutor({ query: failing }).start(op(), {}).result;
    expect(!isOk(result) && result.error.reason).toContain("codex");
  });

  it("declares config enforcement and NO fork primitive — the two facts it inherits behaviour from", () => {
    const caps = new AgentCodexExecutor().capabilities;
    expect(caps).toEqual(CODEX_CAPS);
    expect(caps.policyEnforcement).toBe("config");
    expect(caps.sessionFork).toBe(false);
  });

  it("answers a read-only profile with its sandbox, not with claude's deny list", async () => {
    // The sandbox is codex's ONLY enforcement channel, and its `plan` mode is nothing but
    // `--sandbox read-only` — so the profile maps where it deliberately does not for claude. The
    // base's default deny list must stay clear of the argv: `codexRefusal` would refuse the whole
    // run over tool names codex has no flag for.
    let seen: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      seen = opts;
      yield { type: "result", result: { text: "ok" } };
    };
    // A stub gate is enough: the executor reads only `profile` off it on this path.
    const gated = { gate: { profile: "read-only", check: async () => ({ allow: true }), modeOf: () => "ask" } } as unknown as ExecServices;
    const result = await new AgentCodexExecutor({ query }).start(op(), gated).result;
    expect(isOk(result)).toBe(true);
    expect(seen?.permissionMode).toBe("plan");
    expect(seen?.disallowedTools).toBeUndefined();
  });
});
