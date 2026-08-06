import { describe, expect, it } from "vitest";
import { isOk, type ExecServices, type FunctionInputs, type ResolvedSession } from "@declarative-ai/exec";
import { codexArgv, codexRefusal, createCodexAgentQuery, mcpServerOverride, readCodexEvent, replayPreamble, sandboxFor } from "../src/codexQuery.js";
import { CODEX_CAPS, createCodexAgentFunction } from "../src/codexRuntime.js";
import type { AgentProcess, SpawnOptions, SpawnProcess } from "../src/process.js";

/** A fake process that replays scripted stdout lines and records how it was launched. */
function fakeSpawn(lines: string[], exitCode = 0): {
  spawn: SpawnProcess;
  argv: string[][];
  opts: SpawnOptions[];
  killed: () => boolean;
} {
  const argv: string[][] = [];
  const opts: SpawnOptions[] = [];
  let wasKilled = false;
  const spawn: SpawnProcess = (a, o) => {
    argv.push(a);
    opts.push(o);
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
  return { spawn, argv, opts, killed: () => wasKilled };
}

const inputs = (): FunctionInputs => ({ prompt: "review the diff", config: {} });

/** The terminal event pair a healthy run produces, in the current dialect. */
const answered = (text: string, threadId = "01997"): string[] => [
  JSON.stringify({ type: "thread.started", thread_id: threadId }),
  JSON.stringify({ type: "item.completed", item: { item_type: "assistant_message", text } }),
  JSON.stringify({ type: "turn.completed", usage: { input_tokens: 10, output_tokens: 4 } }),
];

describe("codexArgv — the flags one run is configured with", () => {
  const valueAfter = (flags: string[], flag: string): string | undefined => flags[flags.indexOf(flag) + 1];

  it("runs `exec` with JSONL events and takes its prompt from stdin", () => {
    const argv = codexArgv({ prompt: "hi" });
    expect(argv[0]).toBe("exec");
    expect(argv).toContain("--json");
    // The prompt is NOT in argv: it is `-` (stdin), behind the end-of-options separator.
    expect(argv.slice(-2)).toEqual(["--", "-"]);
    expect(argv.join(" ")).not.toContain("hi");
  });

  // As a CONFIG OVERRIDE, not as `--sandbox`: that flag does not exist on `codex exec resume`, so a
  // resumed run built with it fails argument parsing. This spelling is the one both accept.
  it("states the sandbox explicitly, so blast radius never depends on ~/.codex/config.toml", () => {
    expect(codexArgv({ prompt: "hi" })).toContain('sandbox_mode="workspace-write"');
    expect(codexArgv({ prompt: "hi", permissionMode: "plan" })).toContain('sandbox_mode="read-only"');
    expect(codexArgv({ prompt: "hi", permissionMode: "bypassPermissions" })).toContain('sandbox_mode="danger-full-access"');
    // A caller may lower the default for a whole registration (a review agent that must not write).
    expect(codexArgv({ prompt: "hi" }, { sandbox: "read-only" })).toContain('sandbox_mode="read-only"');
    // The flag form must not reappear: it is what breaks a resume.
    expect(codexArgv({ prompt: "hi", resume: "s1" })).not.toContain("--sandbox");
  });

  // `codex exec` is non-interactive and there is nobody to ask: a run that stopped to ask would HANG
  // the workflow rather than fail it, which is the worse of the two failures.
  it("pins approval_policy to never", () => {
    expect(codexArgv({ prompt: "hi" })).toContain('approval_policy="never"');
  });

  it("continues a conversation with `resume <id>`, before the option list is closed", () => {
    const argv = codexArgv({ prompt: "hi", resume: "0199-abc" });
    expect(argv.slice(0, 3)).toEqual(["exec", "resume", "0199-abc"]);
    // SESSION_ID is codex's first positional, so it has to precede `--`; the prompt is the second.
    expect(argv.indexOf("resume")).toBeLessThan(argv.indexOf("--"));
  });

  it("points codex at our bridge only when there is a bridge", () => {
    expect(codexArgv({ prompt: "hi" }).join(" ")).not.toContain("mcp_servers");
    const argv = codexArgv({ prompt: "hi" }, {}, "http://127.0.0.1:9/mcp/tok");
    expect(argv).toContain(mcpServerOverride("http://127.0.0.1:9/mcp/tok"));
    // A TOML inline table, with the URL quoted — the secret rides in it, so it must survive verbatim.
    expect(mcpServerOverride("http://127.0.0.1:9/mcp/tok")).toBe('mcp_servers.dai={url="http://127.0.0.1:9/mcp/tok"}');
  });

  it("falls back to the caller's default sandbox for an unknown mode", () => {
    expect(sandboxFor(undefined)).toBe("workspace-write");
    expect(sandboxFor(undefined, "read-only")).toBe("read-only");
  });
});

describe("codexRefusal — what this transport will not pretend to do", () => {
  // Each of these would otherwise be a guarantee the caller believes in and nothing enforces.
  it("refuses an approver, because codex exec has no permission-prompt channel", () => {
    expect(codexRefusal({ prompt: "x", canUseTool: async () => ({ allow: true }) })).toMatch(/no mid-run permission callback/);
  });

  it("refuses a native fork, because codex has no fork primitive", () => {
    expect(codexRefusal({ prompt: "x", forkSession: true })).toMatch(/no server-side fork/);
  });

  it("refuses resume + replay together — that would duplicate the conversation", () => {
    expect(codexRefusal({ prompt: "x", resume: "s1", messages: [{ role: "user", content: "hi" }] })).toMatch(/cannot both resume/);
  });

  it("refuses a deny list it cannot express", () => {
    expect(codexRefusal({ prompt: "x", disallowedTools: ["bash"] })).toMatch(/no per-tool deny list/);
  });

  it("refuses a native allow-list it cannot express", () => {
    expect(codexRefusal({ prompt: "x", allowedTools: ["Read"] })).toMatch(/no native tool allow-list/);
  });

  it("passes an ordinary run through", () => {
    expect(codexRefusal({ prompt: "x", allowedTools: [], disallowedTools: [] })).toBeUndefined();
  });
});

describe("readCodexEvent — tolerant of both dialects the shipping binary carries", () => {
  it("reads the thread id and the assistant message (thread/turn/item)", () => {
    let run = readCodexEvent({ type: "thread.started", thread_id: "t1" }, {});
    run = readCodexEvent({ type: "item.completed", item: { item_type: "assistant_message", text: "first" } }, run);
    // The LAST assistant message is the answer: codex narrates as it works.
    run = readCodexEvent({ type: "item.completed", item: { item_type: "assistant_message", text: "final" } }, run);
    expect(run).toEqual({ sessionId: "t1", text: "final" });
  });

  it("ignores items that are not the agent's answer", () => {
    const run = readCodexEvent({ type: "item.completed", item: { item_type: "command_execution", command: "ls" } }, { text: "kept" });
    expect(run.text).toBe("kept");
  });

  it("reads the older msg-wrapped dialect too", () => {
    let run = readCodexEvent({ msg: { type: "session_configured", session_id: "s9" } }, {});
    run = readCodexEvent({ msg: { type: "agent_message", message: "hello" } }, run);
    expect(run).toEqual({ sessionId: "s9", text: "hello" });
  });

  it("surfaces a failed turn as an error, however the message is nested", () => {
    expect(readCodexEvent({ type: "turn.failed", error: { message: "rate limited" } }, {}).error).toBe("rate limited");
    expect(readCodexEvent({ type: "error", message: "bad auth" }, {}).error).toBe("bad auth");
    expect(readCodexEvent({ msg: { type: "error", message: "boom" } }, {}).error).toBe("boom");
  });
});

describe("createCodexAgentQuery — a subprocess speaking JSONL", () => {
  it("yields the terminal result with the thread id, and writes the prompt to stdin", async () => {
    const { spawn, argv, opts } = fakeSpawn(answered("looks good"));
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "review it", cwd: "/repo" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "result", result: { text: "looks good", sessionId: "01997" } });
    expect(argv[0]![0]).toBe("codex");
    expect(opts[0]!.stdin).toBe("review it");
    // The working root is the process CWD, never codex's `-C`: only the spawn seam knows how to
    // translate a directory for the environment it launches into (a WSL project's agent runs in the
    // distro, where a host path means nothing).
    expect(opts[0]!.cwd).toBe("/repo");
    expect(argv[0]!).not.toContain("-C");
  });

  it("reports NO cost — codex counts tokens, and inventing money corrupts the roll-up", async () => {
    const { spawn } = fakeSpawn(answered("done"));
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    const result = seen.at(-1) as { result: { costUsd?: number } };
    expect(result.result.costUsd).toBeUndefined();
  });

  it("ignores non-JSON chatter rather than failing the run", async () => {
    const { spawn } = fakeSpawn(["Reading config…", "", ...answered("ok")]);
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "result", result: { text: "ok", sessionId: "01997" } });
  });

  it("surfaces a non-zero exit as an error, so the cause is named rather than 'no result'", async () => {
    const { spawn } = fakeSpawn([], 2);
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "other", error: "codex exited with code 2" });
  });

  it("says so when codex answered nothing, rather than yielding empty text", async () => {
    const { spawn } = fakeSpawn([JSON.stringify({ type: "thread.started", thread_id: "t" })]);
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x" })) seen.push(m);
    expect(seen.at(-1)).toEqual({ type: "other", error: "codex produced no assistant message" });
  });

  it("refuses BEFORE spawning anything it cannot honour", async () => {
    const { spawn, argv } = fakeSpawn(answered("done"));
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x", canUseTool: async () => ({ allow: true }) })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/no mid-run permission callback/) }]);
    expect(argv).toHaveLength(0);
  });

  it("kills the process when the caller aborts", async () => {
    const { spawn, killed } = fakeSpawn(answered("late"));
    const controller = new AbortController();
    controller.abort();
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn })({ prompt: "x", abortSignal: controller.signal })) seen.push(m);
    expect(killed()).toBe(true);
  });

  it("always kills the child, even when the consumer stops reading early", async () => {
    const { spawn, killed } = fakeSpawn(answered("done"));
    const it = createCodexAgentQuery({ spawn })({ prompt: "x" })[Symbol.asyncIterator]();
    await it.next();
    await it.return?.(undefined as never);
    expect(killed()).toBe(true);
  });
});

describe("the tool bridge — reachable, and refused all the same", () => {
  function fakeBridge() {
    const state = { started: 0, closed: 0 };
    const startBridge = async () => {
      state.started++;
      return {
        url: "http://127.0.0.1:9999/mcp/tok",
        close: async () => {
          state.closed++;
        },
      };
    };
    return { startBridge, state };
  }

  /**
   * ⚠️ The refusal is EVIDENCE-BASED, not caution. Against codex-cli 0.145.0 the bridge is reached and
   * the tool is offered — and codex then auto-denies the call and hands the agent the string
   * `user cancelled MCP tool call`, which it reports as its answer. A run like that SUCCEEDS while
   * having done nothing, and a workflow cannot tell it from a review that found no problems.
   */
  it("refuses injected tools — without spawning, and without standing a bridge up", async () => {
    const { spawn, argv } = fakeSpawn(answered("done"));
    const { startBridge, state } = fakeBridge();
    const seen = [];
    for await (const m of createCodexAgentQuery({ spawn, startBridge })({
      prompt: "x",
      mcpTools: { grep: { inputSchema: {}, run: () => null } },
    })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/auto-denies the call/) }]);
    expect(state.started).toBe(0);
    expect(argv).toHaveLength(0);
  });

  it("names the tools that would have been silently skipped", () => {
    expect(codexRefusal({ prompt: "x", mcpTools: { grep: {} as never, write_file: {} as never } })).toMatch(/grep, write_file/);
  });

  it("runs an ordinary tool-less state with no bridge at all", async () => {
    const { spawn, argv } = fakeSpawn(answered("done"));
    const { startBridge, state } = fakeBridge();
    for await (const _ of createCodexAgentQuery({ spawn, startBridge })({ prompt: "x" })) void _;
    expect(state.started).toBe(0);
    expect(argv[0]!.join(" ")).not.toContain("mcp_servers");
  });

  // The declaration the bridge WOULD carry, kept tested so the day the approval key is found the
  // change is one refusal away. `approval_mode` and its variants are verified: `--strict-config`
  // rejects an unknown field, and accepts this one.
  it("declares each served tool auto-approved, since our own gate runs in-process", () => {
    expect(mcpServerOverride("http://h/x", ["grep"])).toBe('mcp_servers.dai={url="http://h/x",tools={"grep"={approval_mode="auto"}}}');
    expect(mcpServerOverride("http://h/x")).toBe('mcp_servers.dai={url="http://h/x"}');
  });
});

describe("createCodexAgentFunction — the entry the engine reads", () => {
  it("declares 'config' enforcement: no tool-use reaches our approver mid-run", () => {
    expect(CODEX_CAPS.policyEnforcement).toBe("config");
    expect(createCodexAgentFunction().capabilities.policyEnforcement).toBe("config");
  });

  it("declares native resume WITHOUT native fork — the split codex forces", () => {
    expect(CODEX_CAPS.sessionResume).toBe(true);
    expect(CODEX_CAPS.sessionFork).toBe(false);
  });

  // Structural, not incidental: the adapter is constructed with `approvalCallback: false`, so an
  // approver on the ctx is never turned into a callback the query would have to refuse. The two facts
  // — "declares config" and "builds no callback" — are one invariant.
  it("never builds an approval callback, even when ctx.approve is wired", async () => {
    const { spawn } = fakeSpawn(answered("reviewed"));
    const ctx: ExecServices = { workspace: { root: "/repo" }, approve: async () => ({ decision: "allow", scope: "once" }) };
    const result = await createCodexAgentFunction({ spawn }).run(inputs(), ctx as never);
    expect(isOk(result) && result.value).toBe("reviewed");
  });

  it("resolves a CLASSIFIED failure when codex reports an error", async () => {
    const { spawn } = fakeSpawn([JSON.stringify({ type: "turn.failed", error: { message: "model unavailable" } })]);
    const result = await createCodexAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toMatch(/model unavailable/);
  });

  it("records the thread id as the session it ended in, so the next call can resume it", async () => {
    const { spawn } = fakeSpawn(answered("done", "th-77"));
    const result = await createCodexAgentFunction({ spawn }).run(inputs(), {});
    expect(isOk(result) && (result as { session?: { providerSessionId?: string } }).session?.providerSessionId).toBe("th-77");
  });
});

describe("sessions — native append, replayed fork (SESSIONS.md §6)", () => {
  /** A resolved session at a position, shaped as `withSessionPosition` hands one over. */
  const session = (mode: "append" | "fork", messages: unknown[] = []): ResolvedSession =>
    ({
      id: "planning",
      mode,
      at: { id: "planning", seq: 3 },
      providerSessionId: "th-1",
      messages: async () => messages,
    }) as unknown as ResolvedSession;

  it("APPENDS by resuming the handle — no transcript on the wire", async () => {
    const { spawn, argv, opts } = fakeSpawn(answered("continued"));
    await createCodexAgentFunction({ spawn }).run(inputs(), { session: session("append") } as never);
    expect(argv[0]!.slice(1, 4)).toEqual(["exec", "resume", "th-1"]);
    expect(opts[0]!.stdin).toBe("review the diff");
  });

  it("FORKS by replaying the conversation — and must NOT carry the parent's handle", async () => {
    const { spawn, argv, opts } = fakeSpawn(answered("branched"));
    const messages = [
      { role: "user", content: "plan the work" },
      { role: "assistant", content: "here is the plan" },
    ];
    await createCodexAgentFunction({ spawn }).run(inputs(), { session: session("fork", messages) } as never);
    // Two branches writing into one remote session is silent and unrecoverable, so this is the rule
    // that matters most: no `resume` anywhere in the argv.
    expect(argv[0]!).not.toContain("resume");
    expect(opts[0]!.stdin).toContain("plan the work");
    expect(opts[0]!.stdin).toContain("here is the plan");
    // …and the instruction still arrives, after the replayed outline.
    expect(opts[0]!.stdin!.endsWith("review the diff")).toBe(true);
  });

  it("renders a transcript as a readable outline, naming what it cannot show", () => {
    const rendered = replayPreamble([
      { role: "user", content: [{ type: "text", text: "look at this" }] },
      { role: "assistant", content: [{ type: "tool-call", toolName: "bash" }] },
      { role: "system", content: "" },
    ]);
    expect(rendered).toContain("[user]\nlook at this");
    // A tool call the model made is NAMED rather than dropped: it must be able to tell that something
    // happened it is not being shown.
    expect(rendered).toContain("[tool-call content omitted]");
    // An empty turn contributes nothing.
    expect(rendered).not.toContain("[system]");
  });

  it("renders nothing for an empty conversation, so a fresh branch reads as fresh", () => {
    expect(replayPreamble([])).toBe("");
  });
});

describe("the model override", () => {
  it("travels as a CONFIG key, not as -m", () => {
    // `-m` exists on `codex exec` and NOT on `codex exec resume`, so a resumed run built with the flag
    // fails argument parsing — and a resumed run is the common case once a conversation is under way.
    const argv = codexArgv({ prompt: "go", model: "gpt-5" });
    expect(argv).not.toContain("-m");
    expect(argv.join(" ")).toContain('model="gpt-5"');
  });

  it("survives a RESUME, which is the case the flag form would have broken", () => {
    const argv = codexArgv({ prompt: "go", model: "gpt-5", resume: "sess-1" });
    expect(argv.join(" ")).toContain('model="gpt-5"');
    expect(argv.slice(0, 3)).toEqual(["exec", "resume", "sess-1"]);
  });

  it("omits it when none was named", () => {
    expect(codexArgv({ prompt: "go" }).join(" ")).not.toContain("model=");
  });
});
