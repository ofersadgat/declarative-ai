import { describe, expect, it } from "vitest";
import { isOk, type ExecServices, type FunctionInputs } from "@declarative-ai/exec";
import { BRIDGE_UNREACHABLE, bridgeFailureCode, cliArgv, cliRefusal, createCliAgentQuery } from "../src/cliQuery.js";
import { stderrTail, type AgentProcess, type SpawnProcess } from "../src/process.js";
import { CLI_CONFIG_ONLY_CAPS, CLI_DELEGATED_CAPS, createCliAgentFunction } from "../src/runtime.js";
import { injectedToolAllowEntries, mcpConfigJson, PERMISSION_PROMPT_TOOL } from "../src/mcpProtocol.js";

/** A fake process that replays scripted stdout lines and records how it was launched. */
function fakeSpawn(
  lines: string[],
  exitCode = 0,
): {
  spawn: SpawnProcess;
  argv: string[][];
  stdins: (string | undefined)[];
  killed: () => boolean;
  cwds: (string | undefined)[];
  /** Everything written to the child AFTER launch — the steering channel. */
  written: string[];
  ended: () => boolean;
} {
  const argv: string[][] = [];
  const cwds: (string | undefined)[] = [];
  const stdins: (string | undefined)[] = [];
  const written: string[] = [];
  let wasKilled = false;
  let wasEnded = false;
  const spawn: SpawnProcess = (a, opts) => {
    argv.push(a);
    cwds.push(opts.cwd);
    stdins.push(opts.stdin);
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
      write: (line) => void written.push(line),
      endInput: () => {
        wasEnded = true;
      },
    };
    return proc;
  };
  return { spawn, argv, stdins, cwds, written, ended: () => wasEnded, killed: () => wasKilled };
}

const inputs = (): FunctionInputs => ({ prompt: "do it", config: {} });

describe("cliArgv — the flags one run is configured with", () => {
  it("maps the normalized options onto CLI flags, and puts the prompt in NONE of them", () => {
    expect(cliArgv({ prompt: "hi", permissionMode: "plan", allowedTools: ["Read", "Bash"] })).toEqual([
      "-p",
      "--output-format",
      "stream-json",
      "--input-format",
      "stream-json",
      "--verbose",
      "--include-partial-messages",
      "--permission-mode",
      "plan",
      "--allowedTools",
      "Read,Bash",
      "--setting-sources",
      "project",
    ]);
  });

  it("omits flags the caller did not ask for", () => {
    expect(cliArgv({ prompt: "hi", allowedTools: [] })).toEqual([
      "-p",
      "--output-format",
      "stream-json",
      "--input-format",
      "stream-json",
      "--verbose",
      "--include-partial-messages",
      "--setting-sources",
      "project",
    ]);
  });

  it("always asks for PARTIAL messages, the same request the SDK sibling makes under its own name", () => {
    // What makes the declared `streaming` capability true on this path. Verified on 2.1.142's `--help`:
    // "only works with --print and --output-format=stream-json", both of which are already emitted.
    expect(cliArgv({ prompt: "hi" })).toContain("--include-partial-messages");
  });

  // `-p`/`--print` is a BOOLEAN flag in the shipping CLI and the prompt is a positional argument, so
  // `["-p", prompt]` handed the prompt to the option parser: `--verbose` as the first token would have
  // been APPLIED as a flag, and `--nonsense` would have failed the run as an unknown option. Prompts are
  // rendered from workflow data, so both are reachable from content. Off argv entirely, neither is.
  it("does not let a prompt that begins with `--` reach the option parser", () => {
    const argv = cliArgv({ prompt: "--verbose --dangerously-skip-permissions do the thing" });
    expect(argv).not.toContain("--dangerously-skip-permissions");
    expect(argv.join(" ")).not.toContain("do the thing");
    // And no `--` separator either: with nothing positional to protect, it protects nothing.
    expect(argv).not.toContain("--");
  });

  it("emits `--json-schema` when the call declared an output schema", () => {
    // The CLI's own structured-output flag — the argv spelling of the SDK's `outputFormat`. Without it
    // the agent answers in prose and a state that declared outputs gets none of them filled.
    const schema = { type: "object", properties: { items: { type: "array" } }, required: ["items"] };
    const argv = cliArgv({ prompt: "go", schema });
    expect(argv[argv.indexOf("--json-schema") + 1]).toBe(JSON.stringify(schema));
  });

  it("omits it when nothing asked, so a plain text answer stays plain", () => {
    expect(cliArgv({ prompt: "go" })).not.toContain("--json-schema");
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
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/bridge could not start.*not installed/), errorCode: BRIDGE_UNREACHABLE }]);
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
    let hits = 0;
    const validator = {
      validateValue: () => {
        hits++;
        return { ok: true };
      },
    };
    const ctx: ExecServices = {
      tools: { read_file: { inputSchema: { type: "object" }, readOnly: true, run: () => ({ ok: true }) } },
      validator,
    };
    await createCliAgentFunction({ spawn, startBridge }).run(inputs(), ctx as never);
    // Wrapped by json's fail-closed `syncOnly` narrowing on the way down — the claim is DELEGATION.
    const bridged = (state.spec as { validator?: { validateValue: (s: never, v: never) => { ok: boolean } } }).validator;
    expect(bridged).toBeDefined();
    expect(bridged!.validateValue({} as never, 1 as never)).toEqual({ ok: true });
    expect(hits).toBe(1);
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

/**
 * The SESSION — the half of `sessionResume: true` that lives on the wire.
 *
 * These assertions exist because the adapter declared native resume AND native fork while emitting
 * neither flag and recording no id, which the session layer reads as licence to skip replay: every
 * call started a cold conversation and reported it as a successful resume. Both halves are tested,
 * because either one alone still produces exactly that failure.
 *
 * The flags and the `session_id` field were verified against a live `claude 2.1.142` before being
 * written here — `--resume` returns the same id and appends, `--resume --fork-session` returns a NEW
 * id and leaves the parent's transcript untouched.
 */
describe("the session flags", () => {
  it("continues a conversation with `--resume <id>`", () => {
    const argv = cliArgv({ prompt: "go on", resume: "aaaa-0001" });
    expect(argv[argv.indexOf("--resume") + 1]).toBe("aaaa-0001");
    // An APPEND, not a branch: the CLI reuses the id and the parent conversation moves on.
    expect(argv).not.toContain("--fork-session");
  });

  it("branches it with `--fork-session`, which the CLI only accepts alongside a resume", () => {
    const argv = cliArgv({ prompt: "go on", resume: "aaaa-0001", forkSession: true });
    expect(argv[argv.indexOf("--resume") + 1]).toBe("aaaa-0001");
    expect(argv).toContain("--fork-session");
  });

  it("starts FRESH when there is no handle — no session flag anywhere", () => {
    const argv = cliArgv({ prompt: "hi" });
    expect(argv).not.toContain("--resume");
    expect(argv).not.toContain("--fork-session");
  });

  // A fork BRANCHES a conversation, so it says nothing without one to branch. The CLI documents
  // `--fork-session` as "use with --resume or --continue"; emitting it alone would be asking for a
  // branch of nothing, and the executor never sets it without a handle.
  it("drops a fork that names no session, rather than asking the CLI to branch nothing", () => {
    expect(cliArgv({ prompt: "hi", forkSession: true })).not.toContain("--fork-session");
  });

  it("records the id the run ENDED in, so the next call has something to resume", async () => {
    // Verified shape: `session_id` rides on the terminal `result` message. Without capturing it, a
    // correctly-resuming transport still starts fresh every time for want of somewhere to put the id.
    const { spawn } = fakeSpawn(['{"type":"result","is_error":false,"result":"ZEPHYR","session_id":"51caeb77","total_cost_usd":0.01}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "what was the codeword?" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "result", result: { text: "ZEPHYR", costUsd: 0.01, sessionId: "51caeb77" } });
  });

  it("omits the id entirely when the CLI reported none, rather than inventing one", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","result":"done"}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "result", result: { text: "done", costUsd: undefined } });
  });

  // Mutually exclusive by construction upstream; the refusal guards a hand-built seam call, where the
  // damage is silent — the transcript is rendered into the prompt, so resuming as well duplicates it.
  it("refuses to both resume and replay, which would put the conversation in twice", async () => {
    const { spawn, argv } = fakeSpawn(['{"type":"result","result":"done"}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x", resume: "s1", messages: [{ role: "user", content: "hi" }] })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/cannot both resume/) }]);
    expect(argv).toHaveLength(0);
  });
});

/**
 * `is_error` is the CLI's own verdict on the run, and it is INDEPENDENT of both `subtype` and the exit
 * code. An unauthenticated run comes back as `{"subtype":"success","is_error":true,"result":"Not logged
 * in · Please run /login"}` on exit 0 — observed, not hypothesised.
 */
describe("a result message that reports its own failure", () => {
  it("is an error, not the agent's answer", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","subtype":"success","is_error":true,"result":"Not logged in · Please run /login"}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "other", error: "Not logged in · Please run /login" });
    expect(seen).not.toContainEqual(expect.objectContaining({ type: "result" }));
  });

  it("names the run as failed even when the CLI supplied no text to explain it", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","is_error":true}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "other", error: expect.stringMatching(/reported a failed run/) });
  });

  it("reaches the caller as a classified FAILURE, not as a successful empty review", async () => {
    // The whole point: `finishReason: "stop"` carrying "/login" is indistinguishable from an agent that
    // ran and found nothing.
    const { spawn } = fakeSpawn(['{"type":"result","subtype":"success","is_error":true,"result":"Not logged in · Please run /login"}']);
    const result = await createCliAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/Not logged in/);
  });

  it("still resolves a normal run, where `is_error` is false", async () => {
    const { spawn } = fakeSpawn(['{"type":"result","subtype":"success","is_error":false,"result":"done"}']);
    const result = await createCliAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result) && result.value).toBe("done");
  });
});

/**
 * What the caller configured, in the CLI's own flag vocabulary — and a loud refusal for the two things
 * this binary genuinely cannot do.
 */
describe("the neutral knobs, and what this transport cannot carry", () => {
  const flagValue = (argv: string[], flag: string): string | undefined => (argv.includes(flag) ? argv[argv.indexOf(flag) + 1] : undefined);

  it("asks for a reasoning LEVEL with --effort, which takes xhigh on 2.1.142", () => {
    expect(flagValue(cliArgv({ prompt: "go", reasoning: { effort: "xhigh" } }), "--effort")).toBe("xhigh");
  });

  it("REFUSES a thinking budget, because this binary has no flag for one", () => {
    // Checked against 2.1.142's own `--help`: `--effort` exists, a thinking-budget flag does not. The
    // SDK sibling carries it, so the refusal names the transport that can serve the request.
    expect(cliRefusal({ prompt: "go", reasoning: { budgetTokens: 8192 } })).toMatch(/no thinking-budget flag/);
  });

  it("REFUSES a step budget, and says what to bound the run with instead", () => {
    expect(cliRefusal({ prompt: "go", maxSteps: 8 })).toMatch(/no turn-cap flag/);
  });

  it("disables the agent's tools for `toolChoice: none`", () => {
    expect(flagValue(cliArgv({ prompt: "go", toolChoice: "none" }), "--tools")).toBe("");
    expect(cliArgv({ prompt: "go", toolChoice: "auto" })).not.toContain("--tools");
  });

  it("loads PROJECT settings only, so a run does not inherit whoever's machine it is on", () => {
    // Omitting the flag loads user + project + local, and a personal `permissions.allow` entry
    // PRE-APPROVES tools — quietly disarming the approval callback this transport promises.
    expect(flagValue(cliArgv({ prompt: "go" }), "--setting-sources")).toBe("project");
    expect(flagValue(cliArgv({ prompt: "go", providerOptions: { settingSources: ["user", "project"] } }), "--setting-sources")).toBe("user,project");
  });

  it("carries the settings bag, the budget ceiling, and the system-prompt overrides", () => {
    const argv = cliArgv({
      prompt: "go",
      providerOptions: { fastMode: true, ultracode: true, maxBudgetUsd: 2.5, appendSystemPrompt: "Be terse." },
    });
    expect(JSON.parse(flagValue(argv, "--settings")!)).toEqual({ fastMode: true, ultracode: true });
    expect(flagValue(argv, "--max-budget-usd")).toBe("2.5");
    expect(flagValue(argv, "--append-system-prompt")).toBe("Be terse.");
  });

  it("spells extraArgs exactly as the SDK does, so one config drives both transports", () => {
    const argv = cliArgv({ prompt: "go", providerOptions: { extraArgs: { "add-dir": "/repo", bare: null } } });
    expect(argv).toContain("--add-dir");
    expect(flagValue(argv, "--add-dir")).toBe("/repo");
    expect(argv).toContain("--bare");
  });

  it("REFUSES an unrecognised providerOptions key rather than ignoring it", () => {
    expect(cliRefusal({ prompt: "go", providerOptions: { fasMode: true } })).toMatch(/unknown key\(s\): fasMode/);
  });

  it("emits every generated flag, and nothing positional for them to run into", () => {
    const argv = cliArgv({
      prompt: "go",
      reasoning: { effort: "high" },
      toolChoice: "none",
      providerOptions: { maxBudgetUsd: 1, extraArgs: { "add-dir": "/repo" } },
    });
    for (const flag of ["--effort", "--tools", "--setting-sources", "--max-budget-usd", "--add-dir"]) {
      expect(argv.indexOf(flag), flag).toBeGreaterThanOrEqual(0);
    }
    expect(argv).not.toContain("go");
  });

  it("refuses the run BEFORE spawning, so nothing launches under a configuration it cannot honour", async () => {
    const { spawn, argv } = fakeSpawn(['{"type":"result","result":"done"}']);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "go", maxSteps: 4 })) seen.push(m);
    expect(argv).toHaveLength(0);
    expect(seen[0]?.error).toMatch(/no turn-cap flag/);
  });
});

describe("the model flag", () => {
  it("passes --model when the caller named one", () => {
    expect(cliArgv({ prompt: "go", model: "sonnet" })).toContain("--model");
    expect(cliArgv({ prompt: "go", model: "sonnet" }).join(" ")).toContain("--model sonnet");
  });

  it("omits it entirely when none was named, so the CLI keeps its own default", () => {
    expect(cliArgv({ prompt: "go" })).not.toContain("--model");
  });

});

/**
 * WHERE THE PROMPT GOES, which is the one thing about this argv that is not a flag.
 *
 * On stdin, which `-p` reads when nothing positional was given. It used to be a `--` operand, and that
 * failed on length: Windows caps a command line at 32,767 characters, so a prompt rendering a document
 * or a replayed transcript died as `spawn ENAMETOOLONG` — before the agent started, reported as a
 * launch failure that said nothing about the prompt. The codex sibling has always used stdin, for this
 * reason, through the same seam.
 */
describe("the prompt channel", () => {
  it("writes the prompt to stdin rather than argv, as the stream's first message", async () => {
    const { spawn, argv, stdins } = fakeSpawn(['{"type":"result","result":"done"}']);
    for await (const _ of createCliAgentQuery({ spawn })({ prompt: "summarise this" })) void _;

    // A stream-json session reads MESSAGES, so the prompt is one — the channel is unchanged and its
    // shape is what the control protocol costs.
    expect(JSON.parse(stdins[0]!)).toMatchObject({ type: "user", message: { role: "user", content: "summarise this" } });
    expect(argv[0]).not.toContain("summarise this");
  });

  it("carries a prompt far longer than a command line can hold", async () => {
    // The failing case, at the size that produced it: a 66 KB workflow description, twice over the
    // 32,767-character Windows limit. As an operand this never reached the agent at all.
    const huge = "x".repeat(66_000);
    const { spawn, argv, stdins } = fakeSpawn(['{"type":"result","result":"done"}']);
    for await (const _ of createCliAgentQuery({ spawn })({ prompt: huge })) void _;

    expect((JSON.parse(stdins[0]!) as { message: { content: string } }).message.content).toHaveLength(66_000);
    expect(argv[0]!.join(" ").length).toBeLessThan(1_000);
  });
});

describe("steering — ending a turn without ending the process", () => {
  it("sends a control_request the CLI understands, not the shape it silently ignores", async () => {
    const { spawn, written } = fakeSpawn(['{"type":"result","result":"done"}']);
    const run = createCliAgentQuery({ spawn })({ prompt: "count to 400" });
    // Pull one message so the process exists — before that there is nothing to interrupt.
    const it0 = run[Symbol.asyncIterator]();
    await it0.next();

    await run.interrupt!();

    expect(written).toHaveLength(1);
    // Measured against claude 2.1.246: this shape is acknowledged in ~1 ms, while the simpler
    // `{"type":"interrupt"}` is accepted, ignored, and the turn runs to completion — a stop that
    // reports success and changes nothing, which is the exact failure this whole path exists to end.
    expect(JSON.parse(written[0]!)).toMatchObject({ type: "control_request", request: { subtype: "interrupt" } });
  });

  it("is idempotent, and says nothing at all when there is no process to say it to", async () => {
    const { spawn, written } = fakeSpawn(['{"type":"result","result":"done"}']);
    const run = createCliAgentQuery({ spawn })({ prompt: "hi" });
    // Before the stream is pulled the generator has not run, so no child exists yet.
    await expect(run.interrupt!()).resolves.toBeUndefined();
    expect(written).toEqual([]);

    const it0 = run[Symbol.asyncIterator]();
    await it0.next();
    await run.interrupt!();
    await run.interrupt!();
    expect(written).toHaveLength(2);
  });

  it("closes the input when the turn settles, or a completed run would never exit", async () => {
    // Under streaming input the CLI waits for another message rather than exiting on `result`. Without
    // this the process outlives every run and `exit` never resolves.
    const { spawn, ended } = fakeSpawn(['{"type":"result","result":"done"}']);
    for await (const _ of createCliAgentQuery({ spawn })({ prompt: "hi" })) void _;
    expect(ended()).toBe(true);
  });
});

/**
 * The CLI stream, normalized the SAME way the SDK's is.
 *
 * The Agent SDK drives this binary as a subprocess and hands its `stream-json` lines through untouched,
 * so these are literally the same objects. Two mappings were two chances to drop the same field — and
 * both dropped every field but the terminal result.
 */
describe("stream-json arrives as the same normalized shape the SDK path produces", () => {
  const lines = [
    '{"type":"system","subtype":"init","model":"claude-opus-4-7"}',
    '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"ZEP"}}}',
    '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig-1"},{"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"a.txt"}}]}}',
    '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ZEPHYR"}]}}',
    '{"type":"result","subtype":"success","is_error":false,"result":"ZEPHYR","stop_reason":"end_turn","session_id":"s1","total_cost_usd":0.02,"usage":{"input_tokens":6,"output_tokens":8}}',
  ];

  async function collect() {
    const { spawn } = fakeSpawn(lines);
    const out = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "go" })) out.push(m);
    return out;
  }

  it("forwards a system message opaquely rather than discarding it", async () => {
    const seen = await collect();
    expect(seen[0]).toEqual({ type: "provider_event", event: { type: "system", subtype: "init", model: "claude-opus-4-7" } });
  });

  it("turns a text delta into a `partial`", async () => {
    expect((await collect())[1]).toEqual({ type: "partial", delta: "ZEP" });
  });

  it("carries thinking with its signature, the tool call, and the result that answered it", async () => {
    const seen = await collect();
    expect(seen[2]).toMatchObject({
      type: "assistant",
      thinking: [{ text: "hmm", providerMetadata: { anthropic: { signature: "sig-1" } } }],
      toolCalls: [{ toolCallId: "toolu_1", toolName: "Read", input: { path: "a.txt" } }],
    });
    expect(seen[3]).toMatchObject({ type: "user", toolResults: [{ toolCallId: "toolu_1", output: "ZEPHYR" }] });
  });

  it("reads the terminal result's finish reason and token counts, not just its text", async () => {
    const seen = await collect();
    expect(seen[4]).toMatchObject({
      type: "result",
      result: { text: "ZEPHYR", costUsd: 0.02, sessionId: "s1", finishReason: "stop", usage: { inputTokens: 6, outputTokens: 8 } },
    });
  });
});

/**
 * A binary that is PRESENT but not logged in.
 *
 * The lines below are VERBATIM from `claude 2.1.142` run with an empty `CLAUDE_CONFIG_DIR` — the exact
 * shape a first-run machine produces. Everything about it says success: exit code 0,
 * `subtype: "success"`, `terminal_reason: "completed"`, `stop_reason: "stop_sequence"`. Only `is_error`
 * disagrees, and the sentence explaining it sits where the answer would be.
 */
describe("not logged in — the failure that looks exactly like a successful empty answer", () => {
  const NOT_LOGGED_IN = [
    '{"type":"system","subtype":"init","apiKeySource":"none","claude_code_version":"2.1.142"}',
    '{"type":"assistant","message":{"id":"4ce0f2fc","model":"<synthetic>","role":"assistant","stop_reason":"stop_sequence","content":[{"type":"text","text":"Not logged in · Please run /login"}]},"parent_tool_use_id":null,"session_id":"a47138e2","error":"authentication_failed"}',
    '{"type":"result","subtype":"success","is_error":true,"result":"Not logged in · Please run /login","stop_reason":"stop_sequence","session_id":"a47138e2","total_cost_usd":0,"terminal_reason":"completed"}',
  ];

  it("reads the machine-readable code off the assistant turn that carries the prose", async () => {
    // The code appears HERE and nowhere else — the terminal result repeats the sentence but not the
    // classification, so dropping this turn's `error` loses the only part a retry decision can use.
    const { spawn } = fakeSpawn(NOT_LOGGED_IN);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "say hi" })) seen.push(m);
    expect(seen[1]).toMatchObject({ type: "assistant", errorCode: "authentication_failed" });
  });

  it("reports the run as a FAILURE despite exit 0 and subtype success", async () => {
    const { spawn } = fakeSpawn(NOT_LOGGED_IN, 0);
    const result = await createCliAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/Not logged in/);
    expect(!isOk(result) && result.error.classification).toBe("permanent");
  });

  it("does NOT charge for it — a run that never reached a model cost nothing", async () => {
    const { spawn } = fakeSpawn(NOT_LOGGED_IN, 0);
    const result = await createCliAgentFunction({ spawn }).run(inputs(), {});
    expect(result.metrics?.costUsd).toBe(0);
  });
});

/**
 * A binary that is NOT THERE — the other half of "is this agent usable".
 *
 * A spawn that never happened has no exit code, so it arrives as the sentinel `-1`. "agent CLI exited
 * with code -1" is the least useful sentence available for what is usually the commonest first-run
 * outcome, and the real error — `spawn claude ENOENT` — was being captured and thrown away.
 */
describe("not installed — named as a launch failure, not as an exit code", () => {
  /** A spawn that FAILED: no output, the sentinel exit, and the reason it could not start. */
  const failedLaunch = (message: string): SpawnProcess => () => ({
    lines: (async function* () {})(),
    kill: () => {},
    exit: Promise.resolve(-1),
    launchFailure: () => new Error(message),
  });

  it("names the binary and the reason, rather than a sentinel exit code", async () => {
    const seen = [];
    const query = createCliAgentQuery({ command: "claude", spawn: failedLaunch("spawn claude ENOENT") });
    for await (const m of query({ prompt: "hi" })) seen.push(m);
    expect(seen[0]?.error).toBe("the agent binary 'claude' could not be launched: spawn claude ENOENT");
  });

  it("still classifies as a permanent failure that cost nothing", async () => {
    const result = await createCliAgentFunction({ command: "claude", spawn: failedLaunch("spawn claude ENOENT") }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.classification).toBe("permanent");
    expect(result.metrics?.costUsd).toBe(0);
  });

  it("leaves an ordinary non-zero exit reported as an exit code", async () => {
    // The launch succeeded; the run failed. Those are different facts and must read differently.
    const { spawn } = fakeSpawn([], 2);
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "hi" })) seen.push(m);
    expect(seen[0]?.error).toBe("agent CLI exited with code 2");
  });

  it("carries what the CLI said on stderr, which is the only place a startup death is explained", async () => {
    // Seen live: a run recorded as `exited with code 1` had this sentence in the pipe nobody read,
    // and the workflow went on as though the model had merely failed to answer.
    const stderr = "Error: MCP tool mcp__dai__approve (passed via --permission-prompt-tool) not found. Available MCP tools: none\n";
    const spawn: SpawnProcess = () => ({
      lines: (async function* () {})(),
      kill: () => {},
      exit: Promise.resolve(1),
      stderrTail: () => stderr.trim(),
    });
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "hi" })) seen.push(m);
    expect(seen[0]?.error).toBe(`agent CLI exited with code 1: ${stderr.trim()}`);
    // …and NAMES it as the bridge being unreachable, which is the transient case a retry can get past.
    expect(seen[0]?.errorCode).toBe(BRIDGE_UNREACHABLE);
    const result = await createCliAgentFunction({ command: "claude", spawn }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.classification).toBe("network-retriable"); // the class `withRetry`'s transient cap re-attempts
  });

  it("leaves every other stderr death permanent — only the bridge is ours to retry past", async () => {
    const spawn: SpawnProcess = () => ({
      lines: (async function* () {})(),
      kill: () => {},
      exit: Promise.resolve(1),
      stderrTail: () => "Error: Invalid model name: claude-opus-9",
    });
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn })({ prompt: "hi" })) seen.push(m);
    expect(seen[0]?.errorCode).toBeUndefined();
    const result = await createCliAgentFunction({ command: "claude", spawn }).run(inputs(), {});
    expect(!isOk(result) && result.error.classification).toBe("permanent");
    expect(bridgeFailureCode('Failed to connect to MCP server "dai": ECONNREFUSED')).toBe(BRIDGE_UNREACHABLE);
    expect(bridgeFailureCode("MCP server 'other' failed to connect")).toBeUndefined();
  });

  it("keeps the LAST of a long stderr, where the line that names the death is", () => {
    const tail = stderrTail(16);
    expect(tail.read()).toBeUndefined();
    tail.push("   \n");
    expect(tail.read()).toBeUndefined();
    tail.push("warning: old news\n");
    tail.push("Error: the end");
    expect(tail.read()).toBe("s\nError: the end");
  });
});

/**
 * WHICH binary answers, and under what environment.
 *
 * This was a live defect rather than a gap: `cliQuery` spawned a bare `"claude"` with no way for a
 * caller to name its own build, and on Windows a bare name is not launchable without a shell at all.
 */
describe("binaryPath and env reach the subprocess", () => {
  const NPM = "C:\\Users\\me\\AppData\\Roaming\\npm";
  const PACKAGE_EXE = `${NPM}\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe`;
  /** A fake Windows filesystem holding an npm shim and the package entry it delegates to. */
  const winFs = { platform: "win32", pathDirs: [NPM], exists: (p: string) => [`${NPM}\\claude.cmd`, PACKAGE_EXE].includes(p) };

  /** A fake spawn that records the environment it was handed as well as the argv. */
  function recording(): { spawn: SpawnProcess; argv: string[][]; envs: (NodeJS.ProcessEnv | undefined)[] } {
    const argv: string[][] = [];
    const envs: (NodeJS.ProcessEnv | undefined)[] = [];
    const spawn: SpawnProcess = (a, opts) => {
      argv.push(a);
      envs.push(opts.env);
      return { lines: (async function* () { yield '{"type":"result","result":"done"}'; })(), kill: () => {}, exit: Promise.resolve(0) };
    };
    return { spawn, argv, envs };
  }

  it("RESOLVES a named binary before spawning it — an npm shim reaches the package entry", async () => {
    const { spawn, argv } = recording();
    const query = createCliAgentQuery({ spawn, binaryDeps: winFs, warn: () => {} });
    for await (const _ of query({ prompt: "go", binaryPath: "claude" })) void _;
    expect(argv[0]![0]).toBe(PACKAGE_EXE);
  });

  it("lets the CALL's binaryPath beat the transport's wired-in command", async () => {
    // One is how this adapter was constructed, the other is what this run asked for.
    const { spawn, argv } = recording();
    const query = createCliAgentQuery({ spawn, command: "claude", binaryDeps: winFs, warn: () => {} });
    for await (const _ of query({ prompt: "go", binaryPath: "D:\\builds\\claude.exe" })) void _;
    expect(argv[0]![0]).toBe("D:\\builds\\claude.exe");
  });

  it("WARNS rather than refusing when the resolution finds nothing, and runs with what it was given", async () => {
    // A resolution we could not complete is not the same claim as a binary that is definitely absent —
    // so the run proceeds and the spawn failure names the command the caller actually wrote.
    const warnings: string[] = [];
    const { spawn, argv } = recording();
    const query = createCliAgentQuery({ spawn, binaryDeps: { ...winFs, exists: () => false }, warn: (m) => warnings.push(m) });
    for await (const _ of query({ prompt: "go", binaryPath: "claude" })) void _;
    expect(argv[0]![0]).toBe("claude");
    expect(warnings[0]).toContain("not found on the PATH");
  });

  it("forwards the environment verbatim, and INHERITS when the caller named none", async () => {
    const { spawn, envs } = recording();
    const query = createCliAgentQuery({ spawn });
    for await (const _ of query({ prompt: "go", env: { PATH: "/usr/bin", CLAUDE_CONFIG_DIR: "/tmp/a" } })) void _;
    expect(envs[0]).toEqual({ PATH: "/usr/bin", CLAUDE_CONFIG_DIR: "/tmp/a" });
    // Absent means INHERIT. An empty object here would strip the child's PATH and its credentials.
    for await (const _ of query({ prompt: "go" })) void _;
    expect(envs[1]).toBeUndefined();
  });
});
