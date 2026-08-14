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
