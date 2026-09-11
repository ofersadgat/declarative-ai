/**
 * A CLI-DRIVEN {@link AgentQuery} — the sibling of the SDK-driven one in `@declarative-ai/agents-api`.
 * The split is by INVOCATION MECHANISM (DESIGN §4.4): both drive the same normalized seam
 * and both produce the same `runtime` registry entry, so a workflow authored against one runs against
 * the other unchanged. What differs is only how the agent is reached — an in-process SDK call, or a
 * subprocess speaking newline-delimited JSON on stdout.
 *
 * ✅ VERIFIED against `claude 2.1.142` by running it. Every flag {@link cliArgv} emits was checked
 * against that binary's own `--help` and exercised on a live run through a real loopback MCP bridge:
 * `-p`, `--output-format stream-json`, `--verbose`, `--resume`, `--fork-session`, `--mcp-config`,
 * `--model`, `--permission-mode`, `--allowedTools`, `--disallowedTools`, and the terminal `result`
 * message's `session_id` / `is_error` / `total_cost_usd` / `stop_reason` fields.
 *
 * ⚠️ One flag needs watching: **`--permission-prompt-tool` is no longer listed in `claude --help`** on
 * 2.1.142. It is still accepted and still drives the callback — the runs behind `mcpProtocol.ts`'s
 * observations go through it — but an undocumented flag is one release from disappearing, and the way
 * it would fail is silent: the agent runs under its own permission posture and nothing ever reaches
 * `ctx.approve`. A run where the approver is never asked about a tool that should have been gated is
 * the symptom.
 *
 * All adapter LOGIC is tested through the injectable {@link SpawnProcess} seam; only this boundary
 * mapping is untested here.
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
import type { AgentQuery, AgentQueryOptions, AgentRun, AgentStreamMessage, BinaryDeps } from "@declarative-ai/agents-api";
import { claudeOptionsRefusal, DEFAULT_SETTING_SOURCES, defaultBinaryDeps, readAgentMessage, resolveAgentBinary } from "@declarative-ai/agents-api";
import { defaultStartMcpBridge, type McpBridge, type StartMcpBridge } from "./mcpBridge.js";
import { mcpConfigJson, PERMISSION_PROMPT_TOOL } from "./mcpProtocol.js";
import { defaultSpawn, exitMessage, type AgentProcess, type SpawnProcess } from "./process.js";

/** One line of the agent's stdout, already parsed. */
export type CliMessage = Record<string, unknown>;

export interface CliAgentOptions {
  /** The executable to run when the CALL names none. `AgentQueryOptions.binaryPath` wins over this: one
   *  is how the transport was wired, the other is what this run asked for. Default `"claude"`. */
  command?: string;
  /** Extra argv appended after the generated flags. */
  args?: string[];
  /** The process seam. Default: a `node:child_process` spawn, loaded lazily so this module stays
   *  importable (and testable) in a runtime with no child processes. */
  spawn?: SpawnProcess;
  /** The MCP-bridge seam — how the agent reaches back for permission decisions and host tools.
   *  Default: {@link defaultStartMcpBridge}, a loopback HTTP server. Tests inject a fake. */
  startBridge?: StartMcpBridge;
  /** The filesystem/environment facts a named binary is resolved against. Default:
   *  `defaultBinaryDeps()`, read off the real process. Tests inject a fake filesystem. */
  binaryDeps?: BinaryDeps;
  /** Where a binary-resolution warning goes. Default: `console.warn`. */
  warn?: (message: string) => void;
}

/**
 * The failure text for a process that could not be LAUNCHED, or `undefined` when it launched fine.
 *
 * Names the command, because that is the fact a reader needs: on a machine where the agent is simply
 * not installed — the commonest first-run outcome — an unadorned `ENOENT` and an exit code of `-1` are
 * equally uninformative about which binary was missing.
 */
export function launchError(child: Pick<AgentProcess, "launchFailure">, command: string): string | undefined {
  const failure = child.launchFailure?.();
  return failure === undefined ? undefined : `the agent binary '${command}' could not be launched: ${failure.message}`;
}

/**
 * The failure code for a run whose agent could not reach THIS package's bridge — a code of our own,
 * beside the SDK's vocabulary, and one `RETRIABLE_AGENT_ERROR_CODES` names as transient.
 *
 * Seen live: fourteen launches through the same code, one of which died at startup with
 * `MCP tool mcp__dai__approve (passed via --permission-prompt-tool) not found` — the CLI had loaded
 * every other MCP server it knew and simply never connected to the loopback bridge stood up 54 ms
 * after the previous run's was torn down. Nothing about the workflow, the prompt or the account was
 * wrong, and the next launch worked. Classifying that as `permanent` is what let a retry step do
 * nothing and let the engine record a dead draft as the model's answer.
 */
export const BRIDGE_UNREACHABLE = "bridge_unreachable";

/**
 * Whether stderr says the agent ran but could not reach the bridge, as the code above — or nothing.
 *
 * Two spellings, both the CLI's own: the permission tool it was told to route through is "not
 * found" (the server never connected, so its tools never registered), or it names the server —
 * `mcpConfigJson` calls it `dai` — as one it failed to connect to. Anything else on stderr is some
 * other death and stays unclassified, i.e. permanent.
 */
export function bridgeFailureCode(stderr: string | undefined): string | undefined {
  if (stderr === undefined) return undefined;
  if (stderr.includes(PERMISSION_PROMPT_TOOL) && /not found/i.test(stderr)) return BRIDGE_UNREACHABLE;
  if (/(failed to connect|connection (refused|failed|error)|could not connect)[^\n]*\bdai\b/i.test(stderr)) return BRIDGE_UNREACHABLE;
  return undefined;
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
 * ✅ OBSERVED (claude 2.1.142), and the reason the session flags below are here at all: this function
 * used to emit NEITHER, so a transport declaring `sessionResume: true` — which is what tells the session
 * layer to skip replay and read zero messages — started a brand-new conversation on every call and
 * reported it as a successful resume. What a live run confirms:
 *
 *  - `--resume <id>` continues the conversation in place and comes back under the SAME `session_id`;
 *  - `--resume <id> --fork-session` mints a NEW `session_id`, seeded with a copy of the parent's history,
 *    and leaves the parent's transcript untouched — the branch is real, and it costs no replay.
 *
 * `--fork-session` is nested inside the resume branch because the CLI documents it as "use with
 * `--resume` or `--continue`": a fork of nothing is not a request this transport can express, and the
 * executor only ever sets it alongside a handle.
 *
 * Two things this deliberately does NOT do:
 *
 *  - **Injected tools do not join `--allowedTools`.** That flag is the CLI's PRE-APPROVAL list: a tool
 *    named there is never put to `--permission-prompt-tool`. Adding our own bridge-served tools to it
 *    opened both gates at once — the CLI never asked, and the engine skips its own `withPermission`
 *    wrapping for an adapter declaring `policyEnforcement: "callback"` precisely because the callback is
 *    supposed to be the gate. A caller that really means to pre-approve them can pass
 *    `injectedToolAllowEntries(tools)` in `allowedTools` itself, which is at least visible.
 *  - **The prompt is not in argv AT ALL.** It goes on STDIN, which `-p` reads when no positional prompt
 *    is given (`--input-format text`, the default). It was a `--` operand, and that failed two ways.
 *    The fatal one is length: Windows caps a command line at 32,767 characters, and a prompt rendering
 *    a document or a replayed transcript passes that easily — `spawn ENAMETOOLONG`, before the agent
 *    starts, reported as a launch failure with nothing in it about the prompt. The other is parsing:
 *    `-p`/`--print` is a BOOLEAN flag and the prompt is positional (`claude [options] [command]
 *    [prompt]`), so a prompt whose first token starts with `-` was read as a flag — an exact match
 *    silently APPLIED, anything else failed the run as an unknown option. Prompts are rendered from
 *    workflow data, so both are reachable from content. Stdin answers both, and it is the channel the
 *    codex sibling has always used for the same reason.
 */
/**
 * What this run asks for that the CLI cannot honour — the reason to refuse, or `undefined` to proceed.
 *
 * The two gaps are real rather than unimplemented, and both were checked against `claude 2.1.142`'s own
 * `--help`: it has `--effort` and `--max-budget-usd`, and it has NO thinking-budget flag and NO
 * turn-cap flag. The SDK sibling carries both (`thinking`, `maxTurns`) because it speaks the control
 * protocol rather than argv. So a caller asking for either is told which transport can serve it,
 * instead of getting a cheaper run than the one it configured.
 */
export function cliRefusal(opts: AgentQueryOptions): string | undefined {
  const shared = claudeOptionsRefusal(opts);
  if (shared !== undefined) return shared;
  if (opts.reasoning?.budgetTokens !== undefined) {
    return (
      "the claude CLI has no thinking-budget flag, so `reasoning.budgetTokens` cannot be honoured. " +
      "Ask for a level instead (`reasoning.effort`), or drive the agent through the SDK transport, which carries a budget"
    );
  }
  if (opts.maxSteps !== undefined) {
    return (
      "the claude CLI has no turn-cap flag, so `maxSteps` cannot be honoured. " +
      "Bound the run with `providerOptions.claudeCode.maxBudgetUsd`, or drive it through the SDK transport, which carries `maxTurns`"
    );
  }
  return undefined;
}

/** A `providerOptions.claudeCode` value as a flag argument, when it is a string. */
function optionString(opts: AgentQueryOptions, key: string): string | undefined {
  const value = opts.providerOptions?.[key];
  return typeof value === "string" ? value : undefined;
}

/**
 * `--setting-sources`, DECIDED rather than inherited.
 *
 * Omitting the flag loads user, project and local — the operator's personal `~/.claude/settings.json`
 * included, whose `permissions.allow` entries PRE-APPROVE tools and therefore quietly disarm the
 * approval callback this transport's `policyEnforcement: "callback"` promises. `project` alone keeps
 * what belongs to the repository (including CLAUDE.md) and drops what belongs to a person. See
 * `DEFAULT_SETTING_SOURCES`.
 */
function settingSources(opts: AgentQueryOptions): string {
  const declared = opts.providerOptions?.["settingSources"];
  const sources = Array.isArray(declared) ? declared.filter((s): s is string => typeof s === "string") : [...DEFAULT_SETTING_SOURCES];
  return sources.join(",");
}

/** The `--settings` document: the explicit bag plus the two conveniences the escape hatch names at top
 *  level (`fastMode`/`ultracode` are `Settings` keys, not flags). Absent when nothing asked. */
function settingsJson(opts: AgentQueryOptions): string | undefined {
  const explicit = opts.providerOptions?.["settings"];
  const settings: Record<string, unknown> = { ...(explicit !== null && typeof explicit === "object" && !Array.isArray(explicit) ? explicit : {}) };
  for (const key of ["fastMode", "ultracode"] as const) {
    if (opts.providerOptions?.[key] !== undefined) settings[key] = opts.providerOptions[key];
  }
  return Object.keys(settings).length > 0 ? JSON.stringify(settings) : undefined;
}

export function cliArgv(opts: AgentQueryOptions, config: CliAgentOptions = {}, bridgeUrl?: string): string[] {
  const allowed = opts.allowedTools ?? [];
  const denied = opts.disallowedTools ?? [];
  const custom = optionString(opts, "systemPrompt");
  const append = optionString(opts, "appendSystemPrompt");
  const settings = settingsJson(opts);
  const budget = opts.providerOptions?.["maxBudgetUsd"];
  const extraArgs = opts.providerOptions?.["extraArgs"];
  return [
    "-p",
    "--output-format",
    "stream-json",
    // STREAMING INPUT — the flag the control protocol lives behind.
    //
    // Under `--input-format text` (the default) the prompt is written once and stdin is closed, so the
    // only mid-run signal this adapter had was `kill()`: the PROCESS ends, the turn does not, and the
    // partial answer dies with it. As a stream, stdin stays open and carries `control_request`, which
    // is how `interrupt()` ends a turn and still gets a `result` back.
    //
    // Measured against claude 2.1.246: the interrupt is acknowledged in ~1 ms and the turn settles as
    // `subtype: "error_during_execution"`. The simpler `{"type":"interrupt"}` some clients send is
    // SILENTLY IGNORED — no response, no error, and the turn runs to completion — so it is not used.
    "--input-format",
    "stream-json",
    // `stream-json` output requires `--verbose` in non-interactive mode.
    "--verbose",
    // STREAMING. The SDK sibling passes `includePartialMessages: true`; this is the same request under
    // the CLI's own name, and it is what makes the declared `streaming` capability true on this path
    // too. `--help` on 2.1.142: "Include partial message chunks as they arrive (only works with --print
    // and --output-format=stream-json)" — both of which are already above.
    "--include-partial-messages",
    // `--resume-session-at` cuts the copy: "only messages up to and including the assistant message
    // with <message.id>". Nested inside the fork branch because a cut without a fork would truncate
    // the conversation being CONTINUED rather than the copy being made.
    ...(opts.resume !== undefined
      ? [
          "--resume",
          opts.resume,
          ...(opts.forkSession === true
            ? ["--fork-session", ...(opts.resumeSessionAt !== undefined ? ["--resume-session-at", opts.resumeSessionAt] : [])]
            : []),
        ]
      : []),
    ...(bridgeUrl !== undefined ? ["--mcp-config", mcpConfigJson(bridgeUrl)] : []),
    // Only when there is an approver to ask. With tools injected but no approver, the bridge exists to
    // SERVE those tools and the CLI keeps its own permission behaviour.
    ...(bridgeUrl !== undefined && opts.canUseTool !== undefined ? ["--permission-prompt-tool", PERMISSION_PROMPT_TOOL] : []),
    // The model, when the caller named one. Absent ⇒ the CLI's own configured default, which is the
    // ordinary case: a transport that needs no API key generally needs no model id either.
    ...(opts.model !== undefined ? ["--model", opts.model] : []),
    ...(opts.permissionMode !== undefined ? ["--permission-mode", opts.permissionMode] : []),
    ...(allowed.length > 0 ? ["--allowedTools", allowed.join(",")] : []),
    // The deny channel the header has always claimed: a `deny` in the authored baseline reaches the CLI
    // here. It is checked BEFORE the allow list by the CLI, so it is a real floor, not a hint.
    ...(denied.length > 0 ? ["--disallowedTools", denied.join(",")] : []),
    // How hard to think. `--effort` takes `low|medium|high|xhigh|max` on 2.1.142, which is why `xhigh`
    // is in the neutral `ReasoningSpec` rather than smuggled through `providerOptions`. A BUDGET has no
    // flag here and is refused (see {@link cliRefusal}) rather than dropped.
    ...(opts.reasoning?.effort !== undefined ? ["--effort", opts.reasoning.effort] : []),
    // `toolChoice: "none"` — answer from what you already know. `--tools ""` is the CLI's documented
    // "disable all tools"; `auto` is the default and says nothing.
    ...(opts.toolChoice === "none" ? ["--tools", ""] : []),
    // The answer's shape. `--json-schema` is the CLI's own structured-output flag (the argv spelling of
    // the SDK's `outputFormat: {type: "json_schema"}`), and the agent retries inside its own loop until
    // the value validates. The value comes back on the terminal message's `structured_output`, NOT on
    // `result` — which stays a prose summary of the work either way.
    ...(opts.schema !== undefined ? ["--json-schema", JSON.stringify(opts.schema)] : []),
    // WHAT THE AGENT LOADS, decided rather than inherited — see {@link settingSources}.
    "--setting-sources",
    settingSources(opts),
    // A caller's own prompt REPLACES the preset; an append rides ON it. Neither ⇒ nothing is passed,
    // which for the CLI already means the preset (unlike the SDK, where omitting it meant no system
    // prompt at all — the divergence that made these two transports quietly different agents).
    ...(custom !== undefined ? ["--system-prompt", custom] : []),
    ...(custom === undefined && append !== undefined ? ["--append-system-prompt", append] : []),
    ...(settings !== undefined ? ["--settings", settings] : []),
    ...(typeof budget === "number" ? ["--max-budget-usd", String(budget)] : []),
    // The escape hatch's escape hatch: `{flag: "value"}` becomes `--flag value`, `{flag: null}` a bare
    // `--flag`, matching the SDK's `extraArgs` exactly so one config drives both transports.
    ...Object.entries(extraArgs !== null && typeof extraArgs === "object" && !Array.isArray(extraArgs) ? extraArgs : {}).flatMap(([flag, value]) =>
      value === null ? [`--${flag}`] : [`--${flag}`, String(value)],
    ),
    ...(config.args ?? []),
  ];
}

/** Build a CLI-driven agent query. */
export function createCliAgentQuery(config: CliAgentOptions = {}): AgentQuery {
  return (opts: AgentQueryOptions): AgentRun => {
    /**
     * The live child, once there is one — what {@link AgentRun.interrupt} writes to.
     *
     * Held here rather than inside the generator because the two have different lifetimes: the caller
     * holds the run and may interrupt it at any moment, while the generator is a body that has not
     * necessarily started. Before it starts there is no process and nothing to interrupt, which is
     * the same answer as "already finished".
     */
    let live: AgentProcess | undefined;
    let controls = 0;

    const messages = async function* cliAgentQuery(): AsyncIterable<AgentStreamMessage> {
    // REFUSE BEFORE SPAWNING, exactly as the codex sibling does. `applySession` makes these two
    // mutually exclusive by construction — a handle is threaded only when the transport is NOT
    // replaying — so this is unreachable from the executor and guards a hand-built seam call. It stays
    // because the alternative is silent: the replayed transcript is rendered into the prompt, so
    // resuming as well would put the whole conversation into the session TWICE.
    if (opts.resume !== undefined && opts.messages !== undefined) {
      yield { type: "other", error: "the claude CLI cannot both resume a session and replay a transcript — that would duplicate the conversation" };
      return;
    }
    // Everything else this transport cannot honour, refused for the same reason and in the same place.
    const refusal = cliRefusal(opts);
    if (refusal !== undefined) {
      yield { type: "other", error: refusal };
      return;
    }

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
        // TRANSIENT: a loopback listener that could not bind is a fact about this moment's ports, not
        // about the workflow — see `BRIDGE_UNREACHABLE`.
        yield {
          type: "other",
          error: `the agent's permission/tool bridge could not start: ${e instanceof Error ? e.message : String(e)}`,
          errorCode: BRIDGE_UNREACHABLE,
        };
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
      // WHICH binary. The call's `binaryPath` beats the transport's wired-in `command`: one is how this
      // adapter was constructed, the other is what this run asked for.
      //
      // Only a CALL-named binary is resolved here, and the asymmetry is deliberate. `config.command`
      // reaches the spawn seam, whose `resolveProgram` already makes a Windows command launchable —
      // following an npm `.cmd` shim to `node <entry>`, which is the resolution a subprocess wants.
      // `binaryPath` is the SEAM's field, shared with the SDK transport, which spawns the path itself
      // and can be handed only ONE string; resolving it here is what makes the same value mean the same
      // thing on both. Resolving `config.command` as well would just do the spawn seam's work twice.
      let command = opts.binaryPath ?? config.command ?? "claude";
      if (opts.binaryPath !== undefined) {
        const resolved = resolveAgentBinary(opts.binaryPath, config.binaryDeps ?? (await defaultBinaryDeps()));
        command = resolved.path;
        if (resolved.warning !== undefined) (config.warn ?? ((m: string) => console.warn(m)))(resolved.warning);
      }
      const c = spawn([command, ...cliArgv(opts, config, bridge?.url)], {
        cwd: opts.cwd,
        ...(opts.env !== undefined ? { env: opts.env } : {}),
        // The prompt as the stream's first message — still the channel with no length limit (see
        // {@link cliArgv}), now in the shape a stream-json session reads. The CLI emits NOTHING until
        // it has input, so this goes in immediately rather than after any handshake.
        stdin: `${JSON.stringify({ type: "user", message: { role: "user", content: opts.prompt ?? "" }, parent_tool_use_id: null })}
`,
        // Held open, because closing it is what ENDS the session — see `endInput` below.
        keepInputOpen: true,
      });
      child = c;
      live = c;

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
        // ONE mapping, shared with the SDK sibling (`readAgentMessage`): the Agent SDK drives THIS
        // binary as a subprocess and hands these same lines through untouched, so a second mapping here
        // would be a second chance to drop the same field — and both used to drop every field, keeping
        // only the terminal result.
        //
        // ✅ OBSERVED (claude 2.1.142): `{"type":"result","subtype":"success","is_error":false,
        // "result":"…","stop_reason":"end_turn","session_id":"…","total_cost_usd":0.029,"usage":{…}}`.
        // `is_error` is the CLI's own verdict on the run and is INDEPENDENT of `subtype` and of the exit
        // code: a run that fails to authenticate comes back as `{"subtype":"success","is_error":true,
        // "result":"Not logged in · Please run /login"}` with exit 0 — which the mapping turns into an
        // error rather than into the agent's answer.
        const normalized = readAgentMessage(msg);
        if (normalized.type === "result") {
          sawResult = true;
          // CLOSE THE INPUT. Under streaming input the CLI waits for another message rather than
          // exiting when a turn finishes, so without this a completed run never settles and `c.exit`
          // never resolves. Measured: closing stdin exits 0.
          c.endInput?.();
        }
        // A CLI-level `{"error": "..."}` line has no `type` we recognise, so it would otherwise pass
        // through opaquely. It is run-fatal and has to reach the adapter as such.
        if (normalized.type === "provider_event" && typeof msg["error"] === "string") {
          yield { type: "other", error: msg["error"] };
          continue;
        }
        yield normalized;
        if (normalized.error !== undefined) return;
      }
      const code = await c.exit;
      // A non-zero exit with NO `result` yet seen is the CLI's way of failing; surface it so the adapter
      // classifies it. If a result was already streamed, a nonzero exit is post-result cleanup noise and
      // must not throw away the answer the caller already has.
      //
      // A failed LAUNCH is named as one. It has no exit code, so it arrives as the sentinel `-1`, and
      // "exited with code -1" is the least useful sentence available for what is usually the commonest
      // first-run outcome — the binary is not installed, or is not where it was said to be.
      //
      // An ordinary nonzero exit carries what STDERR said. The code alone names nothing — `exited
      // with code 1` covered a permission tool the CLI could not find on the bridge, and the sentence
      // saying so was in the pipe (`AgentProcess.stderrTail`).
      if (code !== 0 && !sawResult) {
        const tail = c.stderrTail?.();
        const bridgeCode = bridgeFailureCode(tail);
        yield {
          type: "other",
          error: launchError(c, command) ?? exitMessage("agent CLI", code, tail),
          ...(bridgeCode !== undefined ? { errorCode: bridgeCode } : {}),
        };
      }
    } finally {
      if (onAbort) opts.abortSignal?.removeEventListener("abort", onAbort);
      // ALWAYS kill, and always tear the bridge down. The consumer can finalize this generator early
      // (the adapter throws out of its `for await` on the first error message), and without this the
      // subprocess outlives the run — with the abort listener just removed, nothing can reach it any
      // more — while a leaked listener would keep answering permission questions for a run that ended.
      child?.kill();
      live = undefined;
      await bridge?.close();
    }
    };

    const stream = messages();
    return {
      [Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](),
      /**
       * End the current TURN, not the process — the whole point of the streaming channel above.
       *
       * NOT cancellation. The turn stops, a `result` still arrives (with `subtype`
       * `"error_during_execution"`), and the call settles with whatever the agent had produced. That
       * is what lets "stop and tell me what you found" be answered with what was found, where
       * `kill()` answers it by throwing the answer away.
       *
       * Fire-and-forget by design: the control response is an acknowledgement, and the observable
       * outcome is the run settling, which the caller is already awaiting. Idempotent — a second
       * interrupt is another request the CLI answers the same way, and one sent after the process has
       * gone writes to nothing.
       */
      interrupt: async (): Promise<void> => {
        const write = live?.write;
        // No process, or a spawn seam that cannot write: there is nothing to interrupt. A fake spawn
        // in a test is the ordinary case, and failing loudly there would make every such test assert
        // an error it does not care about.
        if (write === undefined) return;
        write(`${JSON.stringify({ type: "control_request", request_id: `req_${++controls}`, request: { subtype: "interrupt" } })}
`);
      },
    };
  };
}
