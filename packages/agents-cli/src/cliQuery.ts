/**
 * A CLI-DRIVEN {@link AgentQuery} — the sibling of the SDK-driven one in `@declarative-ai/agents-api`.
 * The split is by INVOCATION MECHANISM (DESIGN §4.4): both drive the same normalized seam
 * and both produce the same `runtime` registry entry, so a workflow authored against one runs against
 * the other unchanged. What differs is only how the agent is reached — an in-process SDK call, or a
 * subprocess speaking newline-delimited JSON on stdout.
 *
 * ⚠️ UNVERIFIED against a live CLI: the flag names and the stream-json message shape below reflect the
 * documented interface and MUST be confirmed against the installed CLI version — adjust
 * {@link CliAgentOptions} and the two mapping spots if they differ. All adapter LOGIC is tested through
 * the injectable {@link SpawnProcess} seam; only this boundary mapping is untested here.
 *
 * Permission posture reaches a CLI agent on TWO channels, and which one is load-bearing depends on the
 * run:
 *
 *  - UP FRONT, as configuration: `--permission-mode`, `--allowedTools` (a PRE-APPROVAL list — a tool
 *    named there is never asked about), and `--disallowedTools` (the deny floor, checked first).
 *  - MID-RUN, as a callback: `--permission-prompt-tool mcp__dai__approve`, backed by the MCP bridge this
 *    package stands up and passes with `--mcp-config` — the same server that serves host-implemented
 *    tools. That is what makes `policyEnforcement: "callback"` true here rather than aspirational, and
 *    it is why injected tools must stay OFF the pre-approval list (see {@link cliArgv}).
 *
 * Anything this transport cannot honour is REFUSED LOUDLY rather than dropped: silently discarding an
 * approver or a tool set is how an agent ends up running with its own defaults while the workflow
 * believes it is gated.
 */
import type { AgentQuery, AgentQueryOptions, AgentStreamMessage } from "@declarative-ai/agents-api";
import { defaultStartMcpBridge, type McpBridge, type StartMcpBridge } from "./mcpBridge";
import { mcpConfigJson, PERMISSION_PROMPT_TOOL } from "./mcpProtocol";

/** One line of the agent's stdout, already parsed. */
export type CliMessage = Record<string, unknown>;

/** A running child process, reduced to what the adapter needs. */
export interface AgentProcess {
  /** Newline-delimited JSON messages the agent wrote to stdout. */
  lines: AsyncIterable<string>;
  /** Terminate the process (wired to the caller's abort signal). */
  kill(): void;
  /** Resolves with the exit code once the process ends. */
  exit: Promise<number>;
}

/** The injectable process seam: launch the agent CLI with these argv and return its stream. */
export type SpawnProcess = (argv: string[], opts: { cwd?: string }) => AgentProcess;

export interface CliAgentOptions {
  /** The executable to run. Default `"claude"`. */
  command?: string;
  /** Extra argv appended after the generated flags. */
  args?: string[];
  /** The process seam. Default: a `node:child_process` spawn, loaded lazily so this module stays
   *  importable (and testable) in a runtime with no child processes. */
  spawn?: SpawnProcess;
  /** The MCP-bridge seam — how the agent reaches back for permission decisions and host tools.
   *  Default: {@link defaultStartMcpBridge}, a loopback HTTP server. Tests inject a fake. */
  startBridge?: StartMcpBridge;
}

/** Does this run need the agent to call BACK into us — for an approval, or for a host-implemented
 *  tool? Only then is a bridge worth standing up. */
export function needsBridge(opts: AgentQueryOptions): boolean {
  return opts.canUseTool !== undefined || Object.keys(opts.mcpTools ?? {}).length > 0;
}

/**
 * Build the argv for one run. Kept separate from the spawn so it is directly assertable.
 *
 * `bridgeUrl` is present when a bridge is serving this run: it adds `--mcp-config` (so the agent can
 * reach our server, and so the bridge's per-run secret reaches the CLI, since the secret is in that URL)
 * and `--permission-prompt-tool` (so the CLI ASKS before each gated tool-use rather than deciding on its
 * own).
 *
 * Two things this deliberately does NOT do:
 *
 *  - **Injected tools do not join `--allowedTools`.** That flag is the CLI's PRE-APPROVAL list: a tool
 *    named there is never put to `--permission-prompt-tool`. Adding our own bridge-served tools to it
 *    opened both gates at once — the CLI never asked, and the engine skips its own `withPermission`
 *    wrapping for an adapter declaring `policyEnforcement: "callback"` precisely because the callback is
 *    supposed to be the gate. A caller that really means to pre-approve them can pass
 *    `injectedToolAllowEntries(tools)` in `allowedTools` itself, which is at least visible.
 *  - **The prompt is not passed as an option value.** `-p`/`--print` is a BOOLEAN flag in the shipping
 *    CLI and the prompt is a positional argument (`claude [options] [command] [prompt]`), so a prompt
 *    whose first token starts with `-` was parsed as a flag: an exact match silently APPLIED, anything
 *    else failed the run as an unknown option. Prompts are rendered from workflow data, so that is
 *    reachable from content. The `--` end-of-options separator puts the prompt beyond the parser.
 */
export function cliArgv(opts: AgentQueryOptions, config: CliAgentOptions = {}, bridgeUrl?: string): string[] {
  const allowed = opts.allowedTools ?? [];
  const denied = opts.disallowedTools ?? [];
  return [
    "-p",
    "--output-format",
    "stream-json",
    // `stream-json` output requires `--verbose` in non-interactive mode.
    "--verbose",
    ...(bridgeUrl !== undefined ? ["--mcp-config", mcpConfigJson(bridgeUrl)] : []),
    // Only when there is an approver to ask. With tools injected but no approver, the bridge exists to
    // SERVE those tools and the CLI keeps its own permission behaviour.
    ...(bridgeUrl !== undefined && opts.canUseTool !== undefined ? ["--permission-prompt-tool", PERMISSION_PROMPT_TOOL] : []),
    ...(opts.permissionMode !== undefined ? ["--permission-mode", opts.permissionMode] : []),
    ...(allowed.length > 0 ? ["--allowedTools", allowed.join(",")] : []),
    // The deny channel the header has always claimed: a `deny` in the authored baseline reaches the CLI
    // here. It is checked BEFORE the allow list by the CLI, so it is a real floor, not a hint.
    ...(denied.length > 0 ? ["--disallowedTools", denied.join(",")] : []),
    ...(config.args ?? []),
    // Everything after this is an OPERAND, whatever it looks like.
    "--",
    opts.prompt,
  ];
}

/** Build a CLI-driven agent query. */
export function createCliAgentQuery(config: CliAgentOptions = {}): AgentQuery {
  return async function* cliAgentQuery(opts: AgentQueryOptions): AsyncIterable<AgentStreamMessage> {
    // Stand the bridge up BEFORE spawning, and refuse the run if it cannot start. Running the agent
    // anyway would leave it under its own defaults while the caller believes its approver is in force —
    // silence is the failure mode this whole path exists to remove.
    let bridge: McpBridge | undefined;
    if (needsBridge(opts)) {
      const start = config.startBridge ?? defaultStartMcpBridge;
      try {
        bridge = await start({
          ...(opts.mcpTools !== undefined ? { tools: opts.mcpTools } : {}),
          // The boundary check for injected tool ARGUMENTS. Injected, not built in: the MCP server hands
          // the impl whatever arrived on the wire, and `json`'s `OutputValidator` is the seam that turns
          // "an arbitrary payload reaches the impl" into "a malformed call fails".
          ...(opts.validator !== undefined ? { validator: opts.validator } : {}),
          ...(opts.canUseTool !== undefined
            ? { approve: (req) => opts.canUseTool!(req, { signal: opts.abortSignal ?? new AbortController().signal }) }
            : {}),
        });
      } catch (e) {
        yield { type: "other", error: `the agent's permission/tool bridge could not start: ${e instanceof Error ? e.message : String(e)}` };
        return;
      }
    }

    // Spawn INSIDE the try so a throw from `defaultSpawn()` (a failed dynamic import on an edge runtime)
    // or a synchronous `spawn()` failure still tears the bridge down in `finally`. An already-started
    // bridge left open keeps its loopback listener bound and answering asks for a run that never launched.
    let child: AgentProcess | undefined;
    let onAbort: (() => void) | undefined;
    try {
      const spawn = config.spawn ?? (await defaultSpawn());
      const c = spawn([config.command ?? "claude", ...cliArgv(opts, config, bridge?.url)], { cwd: opts.cwd });
      child = c;

      onAbort = (): void => c.kill();
      if (opts.abortSignal?.aborted) c.kill();
      else opts.abortSignal?.addEventListener("abort", onAbort, { once: true });

      let sawResult = false;
      for await (const line of c.lines) {
        if (line.trim().length === 0) continue;
        let msg: CliMessage;
        try {
          msg = JSON.parse(line) as CliMessage;
        } catch {
          continue; // a non-JSON line is CLI chatter, never a message
        }
        // ⚠️ VERIFY: the terminal message discriminator + text/cost field names.
        if (msg["type"] === "result") {
          const text = typeof msg["result"] === "string" ? msg["result"] : typeof msg["text"] === "string" ? msg["text"] : "";
          const costUsd = typeof msg["total_cost_usd"] === "number" ? msg["total_cost_usd"] : undefined;
          sawResult = true;
          yield { type: "result", result: { text, costUsd } };
        } else if (typeof msg["error"] === "string") {
          yield { type: "other", error: msg["error"] };
        } else {
          yield { type: "other" };
        }
      }
      const code = await c.exit;
      // A non-zero exit with NO `result` yet seen is the CLI's way of failing; surface it so the adapter
      // classifies it. If a result was already streamed, a nonzero exit is post-result cleanup noise and
      // must not throw away the answer the caller already has.
      if (code !== 0 && !sawResult) yield { type: "other", error: `agent CLI exited with code ${code}` };
    } finally {
      if (onAbort) opts.abortSignal?.removeEventListener("abort", onAbort);
      // ALWAYS kill, and always tear the bridge down. The consumer can finalize this generator early
      // (the adapter throws out of its `for await` on the first error message), and without this the
      // subprocess outlives the run — with the abort listener just removed, nothing can reach it any
      // more — while a leaked listener would keep answering permission questions for a run that ended.
      child?.kill();
      await bridge?.close();
    }
  };
}

/** The default `node:child_process` spawn, imported lazily so the module stays edge-importable. */
async function defaultSpawn(): Promise<SpawnProcess> {
  const { spawn } = await import("node:child_process");
  const readline = await import("node:readline");
  return (argv, opts) => {
    const [command, ...args] = argv;
    // stderr is IGNORED, not piped: an unread pipe fills at ~64 KB and the child then blocks forever on
    // write, so stdout stops and `exit` never settles. The CLI's diagnostics are not our channel — the
    // exit code and the `result` message are.
    const child = spawn(command!, args, { cwd: opts.cwd, stdio: ["ignore", "pipe", "ignore"] });
    const lines = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });

    // A ChildProcess `'error'` event with no listener THROWS, which would take down the host process —
    // and ENOENT on a missing `claude` binary is the likeliest first-run outcome. Capture it and let it
    // surface through the exit code instead.
    let spawnError: Error | undefined;
    child.on("error", (e: Error) => {
      spawnError = e;
      lines.close();
    });

    return {
      lines,
      kill: () => void child.kill(),
      exit: new Promise<number>((resolve) => {
        child.on("error", () => resolve(-1));
        child.on("close", (code) => resolve(spawnError ? -1 : (code ?? 0)));
      }),
    };
  };
}
