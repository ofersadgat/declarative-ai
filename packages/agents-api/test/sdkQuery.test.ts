/**
 * The SDK adapter's BOUNDARY MAPPING — the part no end-to-end test can reach without a provider.
 *
 * `@anthropic-ai/claude-agent-sdk` is an optional peer dependency, so nothing here may DEPEND on it
 * being installed, and a run that reaches `query()` reaches a real model and a real bill. That gap is
 * not academic: it is exactly where this adapter silently dropped `resume`/`forkSession` while
 * declaring `sessionResume: true`, answered `canUseTool` in a vocabulary the SDK does not parse, and
 * handed JSON Schema to a factory that only takes Zod — each of them invisible because no test ever
 * called the code.
 *
 * So everything that CAN be a pure function is one, and every remaining boundary is behind an injected
 * seam: {@link McpServerDeps} for the tool server, `binaryDeps` for the executable, `InputQueue` for
 * the streaming prompt. The one test that drives `sdkAgentQuery` itself exercises a path that returns
 * BEFORE `query()` — which is the property it is there to assert.
 */
import { describe, expect, it } from "vitest";
import { readAgentMessage } from "../src/streamMessages.js";
import type { McpToolResult } from "../src/mcpTools.js";
import {
  claudeOptionsRefusal,
  InputQueue,
  MCP_SDK_MISSING,
  readSdkResult,
  sdkAgentQuery,
  sdkMcpServer,
  sdkOptions,
  sdkPermissionCallback,
  type McpServerDeps,
} from "../src/sdkQuery.js";

/** Everything this mapping emits UNCONDITIONALLY — the settings this transport DECIDES rather than
 *  inherits, plus the streaming request that makes the declared capability true. */
const ALWAYS = {
  includePartialMessages: true,
  settingSources: ["project"],
  systemPrompt: { type: "preset", preset: "claude_code" },
};

describe("sdkOptions — the request the SDK is handed", () => {
  it("passes the caller's posture through under the SDK's own names", () => {
    expect(sdkOptions({ prompt: "x", cwd: "/repo", model: "sonnet", permissionMode: "plan", allowedTools: ["Read"], disallowedTools: ["Bash"] })).toEqual({
      ...ALWAYS,
      cwd: "/repo",
      model: "sonnet",
      permissionMode: "plan",
      allowedTools: ["Read"],
      disallowedTools: ["Bash"],
    });
  });

  it("omits what the caller did not ask for, so the SDK keeps its own defaults", () => {
    expect(sdkOptions({ prompt: "x" })).toEqual(ALWAYS);
  });

  it("always asks for PARTIAL messages, which is what makes the declared streaming capability true", () => {
    // Not a caller's choice: without it the SDK yields whole assistant turns, so the first thing anyone
    // sees is the finished answer — and for a run that takes minutes, that is indistinguishable from a
    // hung process.
    expect(sdkOptions({ prompt: "x" })).toMatchObject({ includePartialMessages: true });
  });

  it("carries the session handle, so a resumed conversation continues instead of restarting", () => {
    // Without this the session layer's whole cheap path is a lie: it reads ZERO messages on the
    // strength of `sessionResume: true`, and the agent is asked to continue a conversation it was
    // never told about.
    expect(sdkOptions({ prompt: "go on", resume: "sess-abc" })).toEqual({ ...ALWAYS, resume: "sess-abc" });
  });

  it("asks for a BRANCH when the session forked, which is what `sessionFork: true` promises", () => {
    expect(sdkOptions({ prompt: "go on", resume: "sess-abc", forkSession: true })).toEqual({ ...ALWAYS, resume: "sess-abc", forkSession: true });
  });

  it("names WHICH binary answers, so a host can pin its own build", () => {
    // Absent ⇒ the SDK's bundled executable, which is the right default and the reason this went
    // unnoticed: it only bites a host that pins a build, and then it bites on Windows.
    expect(sdkOptions({ prompt: "x", binaryPath: "D:\\builds\\claude.exe" })).toEqual({
      ...ALWAYS,
      pathToClaudeCodeExecutable: "D:\\builds\\claude.exe",
    });
    expect(sdkOptions({ prompt: "x" })).not.toHaveProperty("pathToClaudeCodeExecutable");
  });

  it("takes the ALREADY-RESOLVED path when the caller passes one, not the name it started as", () => {
    // The generator resolves before it builds the options; this parameter is how the resolved value
    // gets in, so what reaches the SDK is launchable rather than merely named.
    expect(sdkOptions({ prompt: "x", binaryPath: "claude" }, "C:\\pkg\\claude.exe")).toMatchObject({
      pathToClaudeCodeExecutable: "C:\\pkg\\claude.exe",
    });
  });

  it("forwards a whole environment, and stays silent so the SDK INHERITS when there is none", () => {
    // The SDK documents an omitted `env` as "inherits process.env" — so a partial bag here would
    // silently strip PATH and the credentials off the subprocess.
    expect(sdkOptions({ prompt: "x", env: { PATH: "/usr/bin" } })).toEqual({ ...ALWAYS, env: { PATH: "/usr/bin" } });
    expect(sdkOptions({ prompt: "x" })).not.toHaveProperty("env");
  });

  it("starts fresh when there is no handle, and never asks to fork nothing", () => {
    expect(sdkOptions({ prompt: "hi" })).not.toHaveProperty("resume");
    // A fork BRANCHES a conversation, so it says nothing without one to branch — and the executor
    // only ever sets the two together.
    expect(sdkOptions({ prompt: "hi", forkSession: true })).not.toHaveProperty("forkSession");
  });
});

describe("readSdkResult — the answer read back", () => {
  it("normalizes the terminal result, cost and all", () => {
    expect(readSdkResult({ type: "result", result: "done", total_cost_usd: 0.02 })).toEqual({ type: "result", result: { text: "done", costUsd: 0.02 } });
  });

  it("records the session the run ENDED in — a new id after a fork, the resumed one otherwise", () => {
    // Recording it is not optional: a fork that kept its parent's handle would put two branches into
    // one remote session, and a run whose id is dropped leaves the next call nothing to resume.
    expect(readSdkResult({ type: "result", result: "ZEPHYR", session_id: "51caeb77" })).toEqual({
      type: "result",
      result: { text: "ZEPHYR", costUsd: undefined, sessionId: "51caeb77" },
    });
  });

  it("treats a run the SDK reported as FAILED as an error, not as the agent's answer", () => {
    // Observed on the binary this SDK drives: `is_error` is independent of `subtype` AND of the exit
    // code, so reading only the discriminator reported "Not logged in · Please run /login" as a
    // successful answer — indistinguishable from a review that found nothing.
    expect(readSdkResult({ type: "result", subtype: "success", is_error: true, result: "Not logged in · Please run /login" })).toEqual({
      type: "other",
      error: "Not logged in · Please run /login",
    });
  });

  it("names the failure even when the SDK supplied no text to explain it", () => {
    expect(readSdkResult({ type: "result", is_error: true })).toEqual({ type: "other", error: expect.stringMatching(/reported a failed run/) });
  });

  it("is the shared mapping, so the two transports cannot disagree about a field", () => {
    // `readSdkResult` is now `readAgentMessage` under its old name. Two mappings were two chances to
    // drop the same field, and both dropped every field but the terminal result.
    expect(readSdkResult).toBe(readAgentMessage);
  });
});

describe("the neutral knobs a delegated transport can carry", () => {
  it("asks for a reasoning LEVEL in the SDK's own vocabulary, xhigh included", () => {
    // `xhigh` is in `ReasoningSpec` because THIS transport has such a tier — the alternative was
    // smuggling it through `providerOptions`, which is the provider-shape leak that type exists to stop.
    expect(sdkOptions({ prompt: "x", reasoning: { effort: "xhigh" } })).toMatchObject({ effort: "xhigh" });
  });

  it("asks for a thinking BUDGET through `thinking`, not the deprecated maxThinkingTokens", () => {
    // The deprecated field collapsed to on/off on recent models, so a caller asking for 8192 would have
    // got "some thinking" rather than the budget it paid for.
    expect(sdkOptions({ prompt: "x", reasoning: { budgetTokens: 8192 } })).toMatchObject({ thinking: { type: "enabled", budgetTokens: 8192 } });
  });

  it("bounds the agent's own loop with maxTurns, which is what a step IS here", () => {
    expect(sdkOptions({ prompt: "x", maxSteps: 12 })).toMatchObject({ maxTurns: 12 });
  });

  it("disables the agent's tools for `toolChoice: none`, and says nothing for `auto`", () => {
    expect(sdkOptions({ prompt: "x", toolChoice: "none" })).toMatchObject({ tools: [] });
    expect(sdkOptions({ prompt: "x", toolChoice: "auto" })).not.toHaveProperty("tools");
  });
});

describe("what a delegated agent LOADS — decided, not inherited", () => {
  it("asks for the claude_code system prompt EXPLICITLY, because omitting it means an EMPTY one", () => {
    // Not a tidy-up. The SDK's own code reads `if (systemPrompt === undefined) prompt = ""`, so this
    // transport was running a bare model that happened to have Claude Code's tools — while the CLI
    // sibling, passing no `--system-prompt`, got the real preset. Two transports documented as
    // interchangeable were different agents.
    expect(sdkOptions({ prompt: "x" }).systemPrompt).toEqual({ type: "preset", preset: "claude_code" });
  });

  it("loads PROJECT settings only, so a run does not inherit whoever's machine it is on", () => {
    // Omitting it loads user + project + local. A personal `permissions.allow` entry PRE-APPROVES tools,
    // so a run the workflow believes is gated by `ctx.approve` silently is not.
    expect(sdkOptions({ prompt: "x" }).settingSources).toEqual(["project"]);
  });

  it("lets a host that WANTS the operator's settings say so", () => {
    expect(sdkOptions({ prompt: "x", providerOptions: { settingSources: ["user", "project", "local"] } }).settingSources).toEqual([
      "user",
      "project",
      "local",
    ]);
  });

  it("appends to the preset, or replaces it outright, as the caller asked", () => {
    expect(sdkOptions({ prompt: "x", providerOptions: { appendSystemPrompt: "Be terse." } }).systemPrompt).toEqual({
      type: "preset",
      preset: "claude_code",
      append: "Be terse.",
    });
    expect(sdkOptions({ prompt: "x", providerOptions: { systemPrompt: "You are a linter." } }).systemPrompt).toBe("You are a linter.");
  });

  it("folds the fastMode/ultracode conveniences into the settings bag they really live in", () => {
    // Both are `Settings` keys rather than `query()` options — a caller thinks of them as top-level, so
    // they are accepted there and put where they belong.
    expect(sdkOptions({ prompt: "x", providerOptions: { fastMode: true, ultracode: true } })).toMatchObject({
      settings: { fastMode: true, ultracode: true },
    });
  });

  it("carries maxBudgetUsd and extraArgs, the two escape hatches that bound and extend a run", () => {
    expect(sdkOptions({ prompt: "x", providerOptions: { maxBudgetUsd: 2.5, extraArgs: { "add-dir": "/repo" } } })).toMatchObject({
      maxBudgetUsd: 2.5,
      extraArgs: { "add-dir": "/repo" },
    });
  });

  it("REFUSES an unrecognised providerOptions key rather than ignoring it", () => {
    // The escape hatch's own failure mode: a caller writes `fasMode: true`, nothing happens, and nothing
    // says so.
    expect(claudeOptionsRefusal({ prompt: "x", providerOptions: { fasMode: true } })).toMatch(/unknown key\(s\): fasMode/);
    expect(claudeOptionsRefusal({ prompt: "x", providerOptions: { fastMode: true } })).toBeUndefined();
  });

  it("stops the real transport BEFORE it reaches a provider, not after", async () => {
    // The refusal is checked on the way in, so this drives the actual `sdkAgentQuery` and still costs
    // nothing: it returns before `query()` is ever called. Without that ordering a misconfigured run
    // would bill for a call made under settings the caller did not get.
    const seen = [];
    for await (const m of sdkAgentQuery({ prompt: "x", providerOptions: { fasMode: true } })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/unknown key\(s\): fasMode/) }]);
  });
});

describe("sdkPermissionCallback — our approver in the SDK's calling convention", () => {
  // The shape verified against 0.3.223: `(toolName, input, options) => PermissionResult`, POSITIONAL,
  // answering in `behavior` vocabulary. This file previously read `req.toolName` off a positional string
  // and answered `{allow}` — so every ask carried an empty tool name and no verdict ever parsed. Neither
  // half could fail loudly: the SDK is an optional peer dependency, so nothing ever called it.
  const signal = new AbortController().signal;

  it("reads the tool name off the POSITIONAL argument, not off a request object", async () => {
    const asked: string[] = [];
    const cb = sdkPermissionCallback(async (req) => {
      asked.push(req.toolName);
      return { allow: true };
    }, signal);
    await cb("Bash", { command: "ls" }, {});
    expect(asked).toEqual(["Bash"]);
  });

  it("answers an allow in `behavior` vocabulary, echoing the input back", async () => {
    // `updatedInput` is REQUIRED on the wire path this drives — a bare allow does not deny, it fails the
    // parse and the tool call comes back to the agent as a harness error. We never REWRITE the agent's
    // input; restating it is how the protocol spells consent.
    const cb = sdkPermissionCallback(async () => ({ allow: true }), signal);
    expect(await cb("Bash", { command: "ls" }, {})).toEqual({ behavior: "allow", updatedInput: { command: "ls" } });
  });

  it("carries a decision's OWN updatedInput on an allow — the answered-question channel", async () => {
    // An answered AskUserQuestion travels as `{...input, answers}` on the allow. Every other decision
    // leaves `updatedInput` unset and the original input is echoed (the test above).
    const cb = sdkPermissionCallback(
      async (req) => ({ allow: true, updatedInput: { ...req.input, answers: { "Which one?": "A" } } as never }),
      signal,
    );
    expect(await cb("AskUserQuestion", { questions: [] }, {})).toEqual({
      behavior: "allow",
      updatedInput: { questions: [], answers: { "Which one?": "A" } },
    });
  });

  it("answers a deny with a MESSAGE, which the wire path also requires", async () => {
    const cb = sdkPermissionCallback(async () => ({ allow: false, reason: "denied by permission policy" }), signal);
    expect(await cb("Bash", {}, {})).toEqual({ behavior: "deny", message: "denied by permission policy" });
  });

  it("never denies without saying why, even when the approver offered no reason", async () => {
    const cb = sdkPermissionCallback(async () => ({ allow: false }), signal);
    expect(await cb("Bash", {}, {})).toEqual({ behavior: "deny", message: "denied" });
  });

  it("prefers the SDK's per-ask signal, falling back to the run's", async () => {
    const perAsk = new AbortController().signal;
    let seen: AbortSignal | undefined;
    const cb = sdkPermissionCallback(async (_req, o) => {
      seen = o.signal;
      return { allow: true };
    }, signal);
    await cb("Bash", {}, { signal: perAsk });
    expect(seen).toBe(perAsk);
    await cb("Bash", {}, {});
    expect(seen).toBe(signal);
  });
});

/**
 * The input queue the SDK reads as the run's prompt.
 *
 * Streaming input is what BUYS the control channel — `interrupt` / `setPermissionMode` / `setModel` are
 * documented as "only supported when streaming input/output is used", and a `prompt: string` is not
 * that. So this queue is load-bearing for the whole of Task 4, and none of it is reachable from a fake
 * `AgentQuery`: a test that injects its own transport never touches this code at all.
 */
describe("InputQueue — the streaming prompt the control channel needs", () => {
  /** Read the next message, or `undefined` if the queue is done. */
  const next = async (q: InputQueue): Promise<unknown> => {
    const r = await q.iterate()[Symbol.asyncIterator]().next();
    return r.done ? undefined : r.value;
  };

  it("shapes a message the way the SDK's stream-json input expects", async () => {
    const q = new InputQueue();
    q.push("do the thing");
    expect(await next(q)).toEqual({ type: "user", message: { role: "user", content: "do the thing" }, parent_tool_use_id: null });
  });

  it("BUFFERS a message pushed before anything is reading — the instruction is queued before query()", async () => {
    // The initial prompt is pushed before `sdk.query()` exists, so it has to survive until the SDK
    // first reads. Losing it would start a run with no instruction.
    const q = new InputQueue();
    q.push("first");
    q.push("second");
    expect(await next(q)).toMatchObject({ message: { content: "first" } });
    expect(await next(q)).toMatchObject({ message: { content: "second" } });
  });

  it("WAKES a reader parked on an empty queue, which is what `send()` mid-run does", async () => {
    const q = new InputQueue();
    const pending = next(q); // the SDK, waiting for more input
    q.push("also check the tests");
    expect(await pending).toMatchObject({ message: { content: "also check the tests" } });
  });

  it("ENDS a parked reader on close, so a finished run does not hold the subprocess open", async () => {
    // The `finally` closes it however the run ended. Without this the SDK waits forever on a message
    // that is never coming, and the process outlives the call.
    const q = new InputQueue();
    const pending = next(q);
    q.close();
    expect(await pending).toBeUndefined();
  });

  it("reports done once closed, and accepts nothing further", async () => {
    const q = new InputQueue();
    q.close();
    q.push("too late");
    expect(await next(q)).toBeUndefined();
  });

  it("drains what was already buffered before reporting done", async () => {
    // Closing must not discard a message the SDK has not read yet.
    const q = new InputQueue();
    q.push("first");
    q.close();
    expect(await next(q)).toMatchObject({ message: { content: "first" } });
    expect(await next(q)).toBeUndefined();
  });
});

describe("sdkMcpServer — injected tools, without the Zod conversion that rejected them", () => {
  const tool = { inputSchema: { type: "object", properties: { path: { type: "string" } } }, run: () => "ok" };

  /** A recording server, so what was WIRED is assertable — the real `Server` keeps its handlers
   *  private, and the bug this code replaced was a factory that threw before the agent ever started. */
  function recording(): { deps: McpServerDeps; list: () => Promise<{ tools: unknown[] }>; call: (name: string, args: unknown) => Promise<McpToolResult> } {
    const handlers = new Map<unknown, (req: { params: { name: string; arguments?: unknown } }) => unknown>();
    const deps: McpServerDeps = {
      create: () => ({ setRequestHandler: (schema, handler) => void handlers.set(schema, handler) }),
      listSchema: "tools/list",
      callSchema: "tools/call",
    };
    return {
      deps,
      list: async () => (await handlers.get("tools/list")!({ params: { name: "" } })) as { tools: unknown[] },
      call: async (name, args) => (await handlers.get("tools/call")!({ params: { name, arguments: args } })) as McpToolResult,
    };
  }

  it("serves nothing when there is nothing to serve", async () => {
    expect(await sdkMcpServer({})).toBeUndefined();
  });

  it("ADVERTISES our tools with their JSON Schema verbatim, which is the whole point of not using tool()", async () => {
    const server = recording();
    await sdkMcpServer({ read_file: tool }, { deps: server.deps });
    expect(await server.list()).toEqual({ tools: [{ name: "read_file", inputSchema: tool.inputSchema }] });
  });

  it("routes a tools/call to OUR impl — the agent calls our implementation, not its built-in", async () => {
    const seen: unknown[] = [];
    const server = recording();
    await sdkMcpServer({ read_file: { ...tool, run: (input) => (seen.push(input), "ZEPHYR") } }, { deps: server.deps });
    expect(await server.call("read_file", { path: "a.txt" })).toEqual({ content: [{ type: "text", text: "ZEPHYR" }] });
    expect(seen).toEqual([{ path: "a.txt" }]);
  });

  it("applies the input gate on the way through, since no MCP server validates for us", async () => {
    const server = recording();
    const validator = { validateValue: () => ({ ok: false as const, errors: "path is required" }) };
    let ran = false;
    await sdkMcpServer({ read_file: { ...tool, run: () => (ran = true) as never } }, { deps: server.deps, validator });
    const result = await server.call("read_file", {});
    expect(ran).toBe(false);
    expect(result.isError).toBe(true);
  });

  it("REFUSES rather than serving nothing when the MCP SDK is missing", async () => {
    // Dropping the tools would run the agent on its own built-ins while the caller believes its impls
    // are in play — the silent degradation this whole seam exists to prevent.
    const missing: McpServerDeps = {
      create: () => {
        throw new Error(MCP_SDK_MISSING);
      },
      listSchema: "l",
      callSchema: "c",
    };
    await expect(sdkMcpServer({ read_file: tool }, { deps: missing })).rejects.toThrow(/not installed/);
  });

  it("produces the entry shape `createSdkMcpServer` produces, built the way our schemas survive", async () => {
    // `tool()` + `createSdkMcpServer` reject a JSON Schema outright (`inputSchema must be a Zod schema
    // or raw shape`), so injected tools were DEAD on this transport — every run with one threw before
    // the agent started. The SDK only ever `.connect()`s the instance, so a low-level Server is the same
    // contract without the conversion.
    const servers = (await sdkMcpServer({ read_file: tool })) as Record<string, { type: string; name: string; instance: unknown }>;
    expect(Object.keys(servers)).toEqual(["dai"]);
    expect(servers["dai"]).toMatchObject({ type: "sdk", name: "dai" });
    expect(typeof (servers["dai"]!.instance as { connect?: unknown }).connect).toBe("function");
  });
});
