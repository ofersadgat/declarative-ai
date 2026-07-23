import { describe, expect, it } from "vitest";
import { isOk, type ExecServices, type FunctionInputs } from "@declarative-ai/exec";
import { cliArgv, createCliAgentQuery, type AgentProcess, type SpawnProcess } from "../src/cliQuery";
import { CLI_CONFIG_ONLY_CAPS, CLI_DELEGATED_CAPS, createCliAgentFunction } from "../src/runtime";
import { injectedToolAllowEntries, mcpConfigJson, PERMISSION_PROMPT_TOOL } from "../src/mcpProtocol";

/** A fake process that replays scripted stdout lines and records the argv it was launched with. */
function fakeSpawn(lines: string[], exitCode = 0): { spawn: SpawnProcess; argv: string[][]; killed: () => boolean; cwds: (string | undefined)[] } {
  const argv: string[][] = [];
  const cwds: (string | undefined)[] = [];
  let wasKilled = false;
  const spawn: SpawnProcess = (a, opts) => {
    argv.push(a);
    cwds.push(opts.cwd);
    const proc: AgentProcess = {
      lines: (async function* () {
        for (const l of lines) {
          if (wasKilled) return;
          yield l;
        }
      })(),
      kill: () => {
        wasKilled = true;
      },
      exit: Promise.resolve(exitCode),
    };
    return proc;
  };
  return { spawn, argv, cwds, killed: () => wasKilled };
}

const inputs = (): FunctionInputs => ({ prompt: "do it", config: {} });

describe("cliArgv — the flags one run is configured with", () => {
  it("maps the normalized options onto CLI flags, with the prompt as a POSITIONAL after `--`", () => {
    expect(cliArgv({ prompt: "hi", permissionMode: "plan", allowedTools: ["Read", "Bash"] })).toEqual([
      "-p",
      "--output-format",
      "stream-json",
      "--verbose",
      "--permission-mode",
      "plan",
      "--allowedTools",
      "Read,Bash",
      "--",
      "hi",
    ]);
  });

  it("omits flags the caller did not ask for", () => {
    expect(cliArgv({ prompt: "hi", allowedTools: [] })).toEqual(["-p", "--output-format", "stream-json", "--verbose", "--", "hi"]);
  });

  // `-p`/`--print` is a BOOLEAN flag in the shipping CLI and the prompt is a positional argument, so
  // `["-p", prompt]` handed the prompt to the option parser: `--verbose` as the first token would have
  // been APPLIED as a flag, and `--nonsense` would have failed the run as an unknown option. Prompts are
  // rendered from workflow data, so both are reachable from content.
  it("does not let a prompt that begins with `--` be parsed as an option", () => {
    const argv = cliArgv({ prompt: "--verbose --dangerously-skip-permissions do the thing" });
    // Exactly one `--`, and everything the parser could have eaten is behind it.
    expect(argv.filter((a) => a === "--")).toHaveLength(1);
    expect(argv.at(-1)).toBe("--verbose --dangerously-skip-permissions do the thing");
    expect(argv.indexOf("--")).toBe(argv.length - 2);
  });

  it("emits the deny channel the header has always advertised", () => {
    // Without this, an ExecPolicy baseline of per-tool `deny` could not be expressed to a CLI agent at
    // all — there was no flag and no option field to carry it.
    const argv = cliArgv({ prompt: "hi", allowedTools: ["Read"], disallowedTools: ["Bash", "Write"] });
    expect(argv[argv.indexOf("--disallowedTools") + 1]).toBe("Bash,Write");
  });
});

describe("createCliAgentQuery — newline-delimited JSON on stdout", () => {
  it("yields the terminal result and passes the workspace through as cwd", async () => {
    const { spawn, argv, cwds } = fakeSpawn(['{"type":"assistant"}', '{"type":"result","result":"done","total_cost_usd":0.02}']);
    const query = createCliAgentQuery({ command: "claude", spawn });
    const seen = [];
    for await (const m of query({ prompt: "do it", cwd: "/repo" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "result", result: { text: "done", costUsd: 0.02 } });
    expect(argv[0]![0]).toBe("claude");
    expect(cwds[0]).toBe("/repo");
  });

  it("ignores non-JSON chatter rather than failing the run", async () => {
    const { spawn } = fakeSpawn(["Loading…", "", '{"type":"result","result":"ok"}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen).toEqual([{ type: "result", result: { text: "ok", costUsd: undefined } }]);
  });

  it("surfaces a non-zero exit as an error message, so the cause is named rather than 'no result'", async () => {
    const { spawn } = fakeSpawn(['{"type":"assistant"}'], 3);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "other", error: "agent CLI exited with code 3" });
  });

  it("kills the process when the caller aborts", async () => {
    const { spawn, killed } = fakeSpawn(['{"type":"result","result":"late"}']);
    const controller = new AbortController();
    controller.abort();
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x", abortSignal: controller.signal })) seen.push(m);
    expect(killed()).toBe(true);
    expect(seen).toEqual([]);
  });
});

describe("createCliAgentFunction — the same adapter, a different invocation mechanism", () => {
  it("declares policyEnforcement 'callback' — each gated tool-use IS routed back to our approver", () => {
    // The mechanism differs from the SDK adapter's (an MCP tool the CLI calls, not an in-process
    // function), but the guarantee is the same, and the guarantee is what the field describes.
    expect(CLI_DELEGATED_CAPS.policyEnforcement).toBe("callback");
    expect(createCliAgentFunction().capabilities.policyEnforcement).toBe("callback");
    // Everything else about a delegated agent is unchanged.
    expect(createCliAgentFunction().capabilities.memoizable).toBe(false);
    expect(createCliAgentFunction().capabilities.mutatesWorkspace).toBe(true);
  });

  it("declares 'config' instead when the agent uses its OWN tools — no callback ever sees those", () => {
    // With injection off, ctx.tools become NATIVE allow-list entries: the CLI pre-approves them and our
    // approver is never asked. Declaring `callback` there tells the engine a gate exists that does not,
    // and the engine skips its own `withPermission` wrapping on the strength of it.
    expect(createCliAgentFunction({ injectTools: false }).capabilities).toEqual(CLI_CONFIG_ONLY_CAPS);
    expect(createCliAgentFunction({ injectTools: false }).capabilities.policyEnforcement).toBe("config");
  });

  it("resolves the agent's text as a Result, like every other registry entry", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const ctx: ExecServices = { workspace: { root: "/repo" } };
    const result = await createCliAgentFunction({ spawn }).run(inputs(), ctx);
    expect(isOk(result) && result.value).toBe("done");
  });

  it("resolves a CLASSIFIED failure when the CLI reports an error", async () => {
    const { spawn } = fakeSpawn(['{"error":"model unavailable"}']);
    const result = await createCliAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/model unavailable/);
  });
});

describe("the bridge lifecycle", () => {
  /** A fake bridge that records whether it was started and torn down. */
  function fakeBridge() {
    const state = { started: 0, closed: 0, spec: undefined as unknown };
    const startBridge = async (spec: unknown) => {
      state.started++;
      state.spec = spec;
      return {
        url: "http://127.0.0.1:9999/mcp",
        close: async () => {
          state.closed++;
        },
      };
    };
    return { startBridge, state };
  }

  it("starts NO bridge when the run needs no callback and injects no tools", async () => {
    const { spawn, argv } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge, state } = fakeBridge();
    for await (const _ of createCliAgentQuery({ spawn, startBridge })({ prompt: "x" })) void _;
    expect(state.started).toBe(0);
    expect(argv[0]).not.toContain("--mcp-config");
  });

  it("starts one for an approver, and tears it down when the run ends", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge, state } = fakeBridge();
    for await (const _ of createCliAgentQuery({ spawn, startBridge })({ prompt: "x", canUseTool: async () => ({ allow: true }) })) void _;
    expect(state.started).toBe(1);
    // A leaked listener would go on answering permission questions for a finished run.
    expect(state.closed).toBe(1);
  });

  it("tears the bridge down even when the consumer abandons the stream early", async () => {
    const { spawn } = fakeSpawn(['{"type":"assistant"}', '{"type":"result","result":"done"}']);
    const { startBridge, state } = fakeBridge();
    const it = createCliAgentQuery({ spawn, startBridge })({ prompt: "x", canUseTool: async () => ({ allow: true }) })[Symbol.asyncIterator]();
    await it.next();
    await it.return?.(undefined as never);
    expect(state.closed).toBe(1);
  });

  // REFUSE, never downgrade. Running the agent with the approver silently dropped is the exact bug
  // this path exists to remove.
  it("refuses the run — without spawning — when the bridge cannot start", async () => {
    const { spawn, argv } = fakeSpawn(['{"type":"result","result":"done"}']);
    const startBridge = async () => {
      throw new Error("@modelcontextprotocol/sdk is not installed");
    };
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn, startBridge })({ prompt: "x", canUseTool: async () => ({ allow: true }) })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/bridge could not start.*not installed/) }]);
    expect(argv).toHaveLength(0);
  });

  it("surfaces that refusal as a classified failure through the adapter", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const startBridge = async () => {
      throw new Error("no port available");
    };
    const ctx: ExecServices = {
      tools: { read_file: { inputSchema: {}, readOnly: true, run: () => null } },
      approve: async () => ({ decision: "allow", scope: "once" }),
    };
    const result = await createCliAgentFunction({ spawn, startBridge }).run(inputs(), ctx as never);
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/bridge could not start/);
  });

  it("hands the bridge the approver and the tools it must serve", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge, state } = fakeBridge();
    const ctx: ExecServices = {
      tools: { read_file: { inputSchema: { type: "object" }, readOnly: true, run: () => ({ ok: true }) } },
      approve: async () => ({ decision: "allow", scope: "once" }),
    };
    await createCliAgentFunction({ spawn, startBridge }).run(inputs(), ctx as never);
    const spec = state.spec as { tools?: Record<string, unknown>; approve?: unknown };
    expect(Object.keys(spec.tools ?? {})).toEqual(["read_file"]);
    expect(typeof spec.approve).toBe("function");
  });

  it("hands the bridge the VALIDATOR too — otherwise nothing checks an injected tool's arguments", async () => {
    // The engine injects `ctx.validator`; the adapter must carry it to the bridge or the boundary check
    // the module documents does not exist on the real path.
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge, state } = fakeBridge();
    const validator = { validateValue: () => ({ ok: true }) };
    const ctx: ExecServices = {
      tools: { read_file: { inputSchema: { type: "object" }, readOnly: true, run: () => ({ ok: true }) } },
      validator,
    };
    await createCliAgentFunction({ spawn, startBridge }).run(inputs(), ctx as never);
    expect((state.spec as { validator?: unknown }).validator).toBe(validator);
  });

  it("always kills the child, even when the consumer stops reading early", async () => {
    const { spawn, killed } = fakeSpawn(['{"type":"assistant"}', '{"type":"result","result":"done"}']);
    const it = createCliAgentQuery({ spawn })({ prompt: "x" })[Symbol.asyncIterator]();
    await it.next(); // take one message, then abandon the generator
    await it.return?.(undefined as never);
    expect(killed()).toBe(true);
  });
});

describe("the argv a permission-gated run actually receives", () => {
  const bridge = async () => ({ url: "http://127.0.0.1:9999/mcp", close: async () => {} });
  const flagsFor = async (opts: Record<string, unknown>) => {
    const { spawn, argv } = fakeSpawn(['{"type":"result","result":"done"}']);
    for await (const _ of createCliAgentQuery({ spawn, startBridge: bridge })({ prompt: "x", ...opts } as never)) void _;
    return argv[0]!;
  };
  const valueAfter = (flags: string[], flag: string) => flags[flags.indexOf(flag) + 1];

  it("points the agent at our server AND names the permission-prompt tool", async () => {
    const flags = await flagsFor({ canUseTool: async () => ({ allow: true }) });
    expect(valueAfter(flags, "--mcp-config")).toBe(mcpConfigJson("http://127.0.0.1:9999/mcp"));
    expect(valueAfter(flags, "--permission-prompt-tool")).toBe(PERMISSION_PROMPT_TOOL);
  });

  // CHANGED DELIBERATELY. This used to assert that injected tools JOIN `--allowedTools`. That flag is
  // the CLI's PRE-APPROVAL list: a tool named there is never put to `--permission-prompt-tool`. So the
  // approver was wired, named on the command line, and never asked about the one set of tools this
  // adapter implements — while the engine, reading `policyEnforcement: "callback"`, had already skipped
  // its own `withPermission` wrapping (hw's engine.ts). Both gates open at once.
  it("does NOT pre-approve injected tools — they must reach the permission prompt", async () => {
    const flags = await flagsFor({
      canUseTool: async () => ({ allow: true }),
      allowedTools: ["Read"],
      mcpTools: { grep: { inputSchema: {}, run: () => null } },
    });
    // The caller's own allow-list is passed through untouched; nothing of ours is added to it.
    expect(valueAfter(flags, "--allowedTools")).toBe("Read");
    expect(flags.join(" ")).not.toContain("mcp__dai__grep");
    // And the gate that must be asked instead is still named.
    expect(valueAfter(flags, "--permission-prompt-tool")).toBe(PERMISSION_PROMPT_TOOL);
  });

  it("still lets a caller pre-approve them EXPLICITLY, since that is now a visible choice", async () => {
    const flags = await flagsFor({
      canUseTool: async () => ({ allow: true }),
      allowedTools: injectedToolAllowEntries({ grep: {} as never }),
      mcpTools: { grep: { inputSchema: {}, run: () => null } },
    });
    expect(valueAfter(flags, "--allowedTools")).toBe("mcp__dai__grep");
  });

  it("serves injected tools WITHOUT claiming the permission gate when there is no approver", async () => {
    // The bridge exists here to serve tools; the CLI keeps its own permission behaviour, and we must
    // not name a prompt tool we would then answer with nobody's decision.
    const flags = await flagsFor({ mcpTools: { grep: { inputSchema: {}, run: () => null } } });
    expect(flags).toContain("--mcp-config");
    expect(flags).not.toContain("--permission-prompt-tool");
  });

  it("adds neither flag when nothing needs a bridge", async () => {
    const flags = await flagsFor({ allowedTools: ["Read"] });
    expect(flags).not.toContain("--mcp-config");
    expect(flags).not.toContain("--permission-prompt-tool");
    expect(valueAfter(flags, "--allowedTools")).toBe("Read");
  });
});
