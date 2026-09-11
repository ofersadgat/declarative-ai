/**
 * A CODEX-DRIVEN {@link AgentQuery} — the third delegated-agent transport, alongside the Agent SDK
 * (`agents-api`) and the `claude` CLI (`./cliQuery`).
 *
 * It lives in this package because the split upstream is by INVOCATION MECHANISM (DESIGN §4.4) and
 * this one is a subprocess speaking newline-delimited JSON, exactly as `claude` is. What it is NOT is
 * a variant of `cliQuery`: `codex` shares neither the flag vocabulary nor the message schema, and the
 * one thing it genuinely does share — the MCP bridge — it reaches through a different channel.
 *
 * ## What was verified, and against what
 *
 * Flags and subcommand shapes below were read off `codex-cli 0.145.0`'s own `--help` output, not from
 * documentation:
 *
 *  - `codex exec [OPTIONS] [PROMPT]`, and `codex exec resume [OPTIONS] [SESSION_ID] [PROMPT]`;
 *  - `--json` prints events to stdout as JSONL; `-s/--sandbox` takes
 *    `read-only|workspace-write|danger-full-access`; `-C/--cd` sets the working root;
 *  - `-c key=value` overrides any config key, the value parsed as TOML;
 *  - a prompt of `-` is read from stdin;
 *  - `codex mcp add --url` documents **streamable HTTP** MCP servers, which is precisely what
 *    {@link defaultStartMcpBridge} serves — so host tools reach codex over the same bridge the
 *    `claude` adapter uses, declared here as a `-c mcp_servers.<name>` override.
 *
 * ⚠️ What is NOT verified is the EVENT SCHEMA. The `thread.started` / `turn.*` / `item.*` vocabulary
 * and the older `msg.type` one are both present in the shipping binary; the field names below are
 * inferred and are the first thing to check if a real run yields no text. {@link readCodexEvent} is
 * deliberately tolerant of both dialects and of either spelling of a field, and it is a pure function
 * so a captured transcript can be replayed against it directly.
 *
 * ## What this transport CANNOT honour
 *
 * `codex exec` has no `--permission-prompt-tool` analogue: there is no mid-run channel to ask a human
 * about a tool call, so this adapter's enforcement is `config` (see `./codexRuntime`), and anything
 * that would need a callback is REFUSED rather than dropped — the same rule `cliQuery` follows.
 * Refusing is not a downgrade of safety here: an adapter declaring `config` gets its injected tools
 * policy-WRAPPED by the engine, so those calls are still gated; it is codex's own built-ins that the
 * sandbox mode has to answer for.
 */
import type { AgentQuery, AgentQueryOptions, AgentResult, AgentStreamMessage } from "@declarative-ai/agents-api";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import type { JsonValue } from "./deps.js";
import { launchError } from "./cliQuery.js";
import { defaultStartMcpBridge, type McpBridge, type StartMcpBridge } from "./mcpBridge.js";
import { MCP_SERVER_NAME } from "./mcpProtocol.js";
import { defaultSpawn, exitMessage, type SpawnProcess } from "./process.js";

/** The executable this adapter drives. */
export const CODEX_COMMAND = "codex";

/** Codex's sandbox policy for model-generated commands — its ONLY up-front enforcement channel. */
export type CodexSandbox = "read-only" | "workspace-write" | "danger-full-access";

export interface CodexAgentOptions {
  /** The executable to run. Default {@link CODEX_COMMAND}. */
  command?: string;
  /** Extra argv inserted after the generated flags (e.g. `--skip-git-repo-check`, `-m <model>`). */
  args?: string[];
  /** The process seam. Default: {@link defaultSpawn}. Tests inject a fake. */
  spawn?: SpawnProcess;
  /** The MCP-bridge seam — how codex reaches host-implemented tools. Tests inject a fake. */
  startBridge?: StartMcpBridge;
  /**
   * The sandbox used when the caller states no permission mode.
   *
   * `workspace-write`, matching SPEC §7.2 ("agents may modify files within the project directory"),
   * and stated EXPLICITLY on every run rather than inherited: codex's own default is a config
   * question, and a workflow's blast radius must not depend on the contents of `~/.codex/config.toml`.
   */
  sandbox?: CodexSandbox;
}

/**
 * Map the normalized permission mode onto a sandbox.
 *
 * The mapping is coarse because the two vocabularies are: codex gates by BLAST RADIUS where the
 * normalized modes gate by how much is asked of a human. `plan` is the one exact correspondence —
 * a planning turn must not write — and `bypassPermissions` is the one place the caller has explicitly
 * accepted an ungated agent.
 */
export function sandboxFor(mode: AgentQueryOptions["permissionMode"], fallback: CodexSandbox = "workspace-write"): CodexSandbox {
  switch (mode) {
    case "plan":
      return "read-only";
    case "bypassPermissions":
      return "danger-full-access";
    case "default":
    case "acceptEdits":
      return "workspace-write";
    default:
      return fallback;
  }
}

/**
 * The `-c` override that points codex at our bridge. TOML inline table; the URL carries the run's
 * secret, so this is also how the secret reaches the CLI — there is no second channel to keep in sync.
 *
 * ✅ OBSERVED, and the detail that makes injection work at all: each tool is declared
 * `approval_mode = "auto"` (the variants are `auto | prompt | writes | approve`). Without it codex
 * asks before calling an MCP tool, and a non-interactive `codex exec` has nobody to ask — so the call
 * came back as `user cancelled MCP tool call` and the agent answered with that instead of doing the
 * work.
 *
 * Auto-approving is not a hole, it is where the gate MOVES to. These are OUR implementations, and an
 * adapter declaring `policyEnforcement: "config"` gets its tools policy-wrapped by the engine — so a
 * call reaches the host's approver in-process. Codex asking as well would be a second gate that
 * nothing can answer.
 */
export function mcpServerOverride(url: string, tools: readonly string[] = [], server: string = MCP_SERVER_NAME): string {
  const declarations = tools.map((tool) => `${JSON.stringify(tool)}={approval_mode="auto"}`).join(",");
  const toolTable = declarations.length > 0 ? `,tools={${declarations}}` : "";
  return `mcp_servers.${server}={url=${JSON.stringify(url)}${toolTable}}`;
}

/**
 * What this run asks for that codex cannot do — the reason to refuse, or `undefined` to proceed.
 *
 * Every entry here is something whose SILENT loss would leave the caller believing in a guarantee
 * that is not in force. That is the whole rule: a transport may lack a capability, but it may never
 * pretend to have one.
 */
export function codexRefusal(opts: AgentQueryOptions): string | undefined {
  if (opts.canUseTool !== undefined) {
    return (
      "codex exec has no mid-run permission callback (no --permission-prompt-tool analogue), so an approver cannot be honoured. " +
      "This adapter declares policyEnforcement: 'config' and must be constructed with approvalCallback: false"
    );
  }
  if (opts.reasoning !== undefined) {
    return (
      "this adapter has no verified reasoning channel for codex, so `reasoning` cannot be honoured. " +
      "Codex exposes model behaviour through `-c` config overrides, which is `providerOptions.codex.args` territory — " +
      "state it there once the key is confirmed against the installed binary, or run the state on a claude transport"
    );
  }
  if (opts.maxSteps !== undefined) {
    return "`codex exec` has no turn-cap option, so `maxSteps` cannot be honoured — run the state on a transport that carries one";
  }
  if (opts.toolChoice === "none") {
    return "`codex exec` has no way to run with its tools disabled, so `toolChoice: \"none\"` cannot be honoured";
  }
  if (opts.providerOptions !== undefined && Object.keys(opts.providerOptions).length > 0) {
    return (
      `providerOptions.codex carries [${Object.keys(opts.providerOptions).join(", ")}], and this adapter maps none of them. ` +
      "Pass codex settings as `args` on the transport instead, where they reach argv verbatim"
    );
  }
  if (opts.resumeSessionAt !== undefined) {
    return "codex cannot cut a copy at a message — a branch behind the tip must be replayed, so the caller must pass `messages` instead of `resumeSessionAt`";
  }
  if (opts.forkSession === true) {
    return "codex has no server-side fork — a fork must be replayed, so the caller must pass `messages` instead of `forkSession`";
  }
  if (opts.resume !== undefined && opts.messages !== undefined) {
    return "codex cannot both resume a session and replay a transcript — that would duplicate the conversation";
  }
  if (opts.disallowedTools !== undefined && opts.disallowedTools.length > 0) {
    return (
      `codex has no per-tool deny list, so [${opts.disallowedTools.join(", ")}] cannot be denied to it. ` +
      "Express the floor as a sandbox mode, or drive the agent through an adapter whose transport carries a deny channel"
    );
  }
  if (opts.allowedTools !== undefined && opts.allowedTools.length > 0) {
    return (
      `codex has no native tool allow-list, so [${opts.allowedTools.join(", ")}] cannot be pre-approved or aliased to its built-ins. ` +
      "Run the state with no tools — codex uses its own — or drive it through an adapter whose transport carries an allow-list"
    );
  }
  if (Object.keys(opts.mcpTools ?? {}).length > 0) {
    // ⚠️ OBSERVED, and the reason this is a refusal rather than a wiring bug left for later. The
    // transport WORKS: codex connects to the bridge, sees the tool, and asks to call it. The call is
    // then auto-DENIED, and the denial comes back as the literal text `user cancelled MCP tool call`
    // — which the agent reports as its answer. Tried and ruled out: `approval_policy` (`never`, and
    // unset), both sandbox modes, and `mcp_servers.<server>.tools.<tool>.approval_mode = "auto"`
    // under every tool-name spelling (bare, `dai__x`, `mcp__dai__x`). The field and its variants
    // (`auto|prompt|writes|approve`) are real — `--strict-config` accepts them — so what is missing
    // is which key codex matches a streamable-HTTP server's tools on.
    //
    // Shipping it anyway would mean an agent that answers "user cancelled MCP tool call" INSTEAD of
    // doing the work, and reports SUCCESS. A workflow cannot tell that from a review that found
    // nothing. Refusing names the problem at the state that asked for it.
    return (
      `codex reaches our tool bridge but auto-denies the call, so [${Object.keys(opts.mcpTools ?? {}).join(", ")}] would never run. ` +
      "Declare no tools on a codex state — it uses its own — or run that state on an adapter that can serve ours"
    );
  }
  return undefined;
}

/**
 * Build the argv for one run.
 *
 * Kept separate from the spawn so it is directly assertable, exactly as `cliArgv` is. Two shapes are
 * deliberate:
 *
 *  - **The prompt goes on STDIN**, named by the `-` positional. `codex exec` accepts a prompt as an
 *    argument, but a replayed conversation is rendered INTO that prompt and Windows caps a command
 *    line at ~32 KB — so argv is not a channel this adapter can rely on. It also sidesteps the
 *    subcommand ambiguity: a prompt whose first word is `resume` or `review` is a positional here and
 *    could never be read as a command.
 *  - **`approval_policy` is pinned to `never`.** `codex exec` is non-interactive and there is nobody
 *    to ask; a run that stopped to ask would hang the workflow rather than fail it.
 *
 * ⚠️ **`codex exec resume` accepts a strict SUBSET of `codex exec`'s options.** Verified against
 * 0.145.0: it takes `--json`, `-c`, `-m`, `-i`, `-o` and the `--dangerously-*` pair, and it takes
 * neither `--sandbox`, nor `-C/--cd`, nor `--color`. Any of those on a resumed run is an argument
 * PARSE failure — the run dies with exit 2 before the model is reached, which reads like a broken
 * adapter rather than a wrong flag. So this builds ONE argv from the intersection: everything here is
 * accepted by both forms. Add a flag only after checking `codex exec resume --help`.
 *
 * The working root is the process's CWD, deliberately, rather than codex's `-C` flag: a host may run
 * the agent somewhere its own path vocabulary does not reach — JaiRA drives a WSL project's agents
 * inside the distro — and the spawn seam is the one layer that knows how to translate a directory for
 * the environment it is launching into. A host path in `-C` would arrive inside the distro unmapped
 * and land the agent somewhere that does not exist.
 */
export function codexArgv(
  opts: AgentQueryOptions,
  config: CodexAgentOptions = {},
  bridgeUrl?: string,
  schemaFile?: string,
): string[] {
  return [
    "exec",
    // `resume <id>` continues the conversation server-side — codex's native append (DESIGN.md §1.6).
    // SESSION_ID is the first positional, so it must precede the `--` that closes the option list.
    ...(opts.resume !== undefined ? ["resume", opts.resume] : []),
    "--json",
    // ✅ OBSERVED: the sandbox travels as a CONFIG OVERRIDE, not as `--sandbox`. The flag exists on
    // `codex exec` and does NOT exist on `codex exec resume` — which accepts no `-s`, no `-C`, and no
    // `-p` — so a resumed run built with the flag fails argument parsing outright. The config key is
    // the one spelling both accept, and it is verified rather than assumed: `--strict-config` rejects
    // an unknown key, and `sandbox_mode` passes it.
    "-c",
    `sandbox_mode="${sandboxFor(opts.permissionMode, config.sandbox)}"`,
    // Non-interactive: there is nobody to ask, and a run that stopped to ask would HANG the workflow
    // rather than fail it.
    "-c",
    'approval_policy="never"',
    ...(bridgeUrl !== undefined ? ["-c", mcpServerOverride(bridgeUrl, Object.keys(opts.mcpTools ?? {}))] : []),
    // The model, when the caller named one. As a CONFIG OVERRIDE rather than `-m`, for the same reason
    // the sandbox is: `-m` exists on `codex exec` and NOT on `codex exec resume`, so a resumed run
    // built with the flag would fail argument parsing — and a resumed run is the common case once a
    // conversation is under way. `model` is a documented key, and `--strict-config` would reject it if
    // it were not.
    ...(opts.model !== undefined ? ["-c", `model="${opts.model}"`] : []),
    // The answer's SHAPE. Codex takes a FILE where the `claude` sibling takes the schema inline, so the
    // caller writes one and passes its path — which keeps this builder pure and directly assertable.
    //
    // ✅ `codex exec --help`: `--output-schema <FILE>  Path to a JSON Schema file describing the
    // model's final response shape`. The constrained value arrives as the final `agent_message`, which
    // is why the reader parses that text rather than looking for a field of its own.
    ...(schemaFile !== undefined ? ["--output-schema", schemaFile] : []),
    ...(config.args ?? []),
    // Everything after this is an OPERAND, whatever it looks like; `-` is "read the prompt from stdin".
    "--",
    "-",
  ];
}

/**
 * Render a conversation into prompt text — the REPLAY half of codex's session story (SESSIONS.md §6).
 *
 * Codex appends natively (`exec resume`) but cannot branch, so a fork has to be replayed. Replay
 * against a delegated agent is lossy in a way it is not against a message-based provider: there is no
 * message array to send, only one prompt, and what the transcript holds for a delegated agent is
 * already thin — one assistant turn per call, because the agent keeps its real log server-side.
 *
 * So this renders an OUTLINE, and the honest thing is that the caller records it as one. It is
 * deliberately plain text with explicit role markers rather than JSON: the reader is a model, and the
 * failure mode to avoid is a model treating a serialized transcript as data to analyse rather than as
 * the conversation it is continuing.
 */
export function replayPreamble(messages: readonly JsonValue[]): string {
  const turns = messages.map(renderTurn).filter((t) => t.length > 0);
  if (turns.length === 0) return "";
  return [
    "The following is the conversation so far, which you are continuing. It was carried over from",
    "another session, so you are seeing a summary of it rather than your own memory of it.",
    "",
    ...turns,
    "",
    "Continue from there. The next instruction follows.",
    "",
  ].join("\n");
}

/** One transcript entry as text. Tolerant by necessity: a stored message is provider-shaped. */
function renderTurn(message: JsonValue): string {
  if (message === null || typeof message !== "object" || Array.isArray(message)) return "";
  const bag = message as Record<string, JsonValue>;
  const role = typeof bag["role"] === "string" ? bag["role"] : "unknown";
  const content = bag["content"];
  const text =
    typeof content === "string"
      ? content
      : Array.isArray(content)
        ? content
            .map((part) => {
              if (typeof part === "string") return part;
              if (part !== null && typeof part === "object" && !Array.isArray(part) && typeof (part as Record<string, JsonValue>)["text"] === "string") {
                return (part as Record<string, string>)["text"]!;
              }
              // Tool calls, tool results, reasoning parts: named rather than dropped, so the model can
              // see that something happened it is not being shown.
              const kind = part !== null && typeof part === "object" && !Array.isArray(part) ? (part as Record<string, JsonValue>)["type"] : undefined;
              return `[${typeof kind === "string" ? kind : "non-text"} content omitted]`;
            })
            .join("\n")
        : "";
  return text.length > 0 ? `[${role}]\n${text}` : "";
}

/** What one run accumulates from the event stream, before it becomes an {@link AgentResult}. */
export interface CodexRun {
  text?: string;
  sessionId?: string;
  /** The model codex named when it configured the session — see {@link AgentResult.model}. */
  model?: string;
  error?: string;
}

/**
 * Fold one parsed event into the run.
 *
 * Pure, and tolerant of BOTH event dialects the shipping binary carries — the `thread`/`turn`/`item`
 * vocabulary and the older `{ msg: { type } }` one — because which is emitted is a version question
 * and getting it wrong looks like an agent that answered nothing.
 */
export function readCodexEvent(event: Record<string, unknown>, run: CodexRun): CodexRun {
  const inner = event["msg"];
  if (inner !== null && typeof inner === "object" && !Array.isArray(inner)) {
    // The older dialect: the discriminator and payload sit under `msg`.
    const bag = inner as Record<string, unknown>;
    switch (bag["type"]) {
      case "agent_message":
        return { ...run, text: stringOf(bag, "message", "text") ?? run.text };
      case "session_configured":
        // The MODEL comes with the session here, where the `claude` adapters announce it in an init
        // event. Read tolerantly — absent it simply stays unknown — because what it feeds is the
        // settle's refusal to record a `default` placeholder as the model that ran.
        return {
          ...run,
          sessionId: stringOf(bag, "session_id", "sessionId") ?? run.sessionId,
          model: stringOf(bag, "model") ?? run.model,
        };
      case "error":
        return { ...run, error: stringOf(bag, "message", "error") ?? "codex reported an error" };
      default:
        return run;
    }
  }
  switch (event["type"]) {
    case "thread.started":
      return {
        ...run,
        sessionId: stringOf(event, "thread_id", "threadId", "session_id") ?? run.sessionId,
        model: stringOf(event, "model") ?? run.model,
      };
    case "item.completed": {
      const item = event["item"];
      if (item === null || typeof item !== "object" || Array.isArray(item)) return run;
      const bag = item as Record<string, unknown>;
      // The LAST agent message is the answer: codex narrates as it works, and only the final one is
      // what a caller asked for.
      //
      // ✅ OBSERVED (codex-cli 0.145.0): `{"type":"item.completed","item":{"id":"item_0",
      // "type":"agent_message","text":"…"}}`. The name is `agent_message`, NOT `assistant_message` —
      // which this originally guessed, and which would have made every codex run report "produced no
      // assistant message". `assistant_message` is kept as an accepted spelling because it is what the
      // adjacent vocabularies use and a rename would otherwise be a silent, total failure.
      const kind = stringOf(bag, "item_type", "type");
      return kind === "agent_message" || kind === "assistant_message"
        ? { ...run, text: stringOf(bag, "text", "message") ?? run.text }
        : run;
    }
    case "turn.failed":
      return { ...run, error: errorTextOf(event["error"]) ?? "codex turn failed" };
    case "thread.error":
    case "error":
      return { ...run, error: stringOf(event, "message", "error") ?? errorTextOf(event["error"]) ?? "codex reported an error" };
    default:
      return run;
  }
}

function stringOf(bag: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = bag[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  return undefined;
}

function errorTextOf(error: unknown): string | undefined {
  if (typeof error === "string") return error;
  if (error !== null && typeof error === "object" && !Array.isArray(error)) return stringOf(error as Record<string, unknown>, "message", "reason");
  return undefined;
}

/** Build a codex-driven agent query. */
export function createCodexAgentQuery(config: CodexAgentOptions = {}): AgentQuery {
  return async function* codexAgentQuery(opts: AgentQueryOptions): AsyncIterable<AgentStreamMessage> {
    // REFUSE BEFORE SPAWNING. Anything this transport cannot honour has to stop the run here: an
    // agent launched with the guarantee quietly dropped is the failure this whole path exists to
    // remove, and it is invisible from the outside.
    const refusal = codexRefusal(opts);
    if (refusal !== undefined) {
      yield { type: "other", error: refusal };
      return;
    }

    // A bridge is worth standing up only when there is something for codex to call back FOR. Unlike
    // the `claude` adapter, an approver is never such a reason — codex has no callback channel — so
    // this is host-implemented tools alone.
    let bridge: McpBridge | undefined;
    if (Object.keys(opts.mcpTools ?? {}).length > 0) {
      const start = config.startBridge ?? defaultStartMcpBridge;
      try {
        bridge = await start({
          ...(opts.mcpTools !== undefined ? { tools: opts.mcpTools } : {}),
          ...(opts.validator !== undefined ? { validator: opts.validator } : {}),
        });
      } catch (e) {
        yield { type: "other", error: `the agent's tool bridge could not start: ${e instanceof Error ? e.message : String(e)}` };
        return;
      }
    }

    let child: ReturnType<SpawnProcess> | undefined;
    let onAbort: (() => void) | undefined;
    let schemaFile: string | undefined;
    try {
      const spawn = config.spawn ?? (await defaultSpawn());
      const preamble = opts.messages !== undefined ? replayPreamble(opts.messages) : "";
      // The call's `binaryPath` beats the transport's wired-in `command`, exactly as it does for the
      // `claude` sibling. No Windows resolution here: `codex` is spawned through the process seam, whose
      // `resolveProgram` already follows an npm `.cmd` shim to its JS entry — the resolution this
      // transport needs, and the one it has had since it was written.
      // The schema, on disk, because that is the only way codex takes one. Written beside the OS temp
      // dir and removed in the `finally` below — a run must not leave one behind, and a crash leaves at
      // most one small file where the OS already collects them.
      if (opts.schema !== undefined) {
        schemaFile = join(await mkdtemp(join(tmpdir(), "codex-schema-")), "schema.json");
        await writeFile(schemaFile, JSON.stringify(opts.schema), "utf8");
      }
      const c = spawn([opts.binaryPath ?? config.command ?? CODEX_COMMAND, ...codexArgv(opts, config, bridge?.url, schemaFile)], {
        ...(opts.cwd !== undefined ? { cwd: opts.cwd } : {}),
        ...(opts.env !== undefined ? { env: opts.env } : {}),
        stdin: preamble.length > 0 ? `${preamble}\n${opts.prompt}` : opts.prompt,
      });
      child = c;

      onAbort = (): void => c.kill();
      if (opts.abortSignal?.aborted) c.kill();
      else opts.abortSignal?.addEventListener("abort", onAbort, { once: true });

      let run: CodexRun = {};
      for await (const line of c.lines) {
        if (line.trim().length === 0) continue;
        let event: Record<string, unknown>;
        try {
          event = JSON.parse(line) as Record<string, unknown>;
        } catch {
          continue; // a non-JSON line is CLI chatter, never an event
        }
        const next = readCodexEvent(event, run);
        // An error is terminal for the run and is reported the moment it is seen — the adapter throws
        // out of its `for await` on it, and the `finally` below kills the process.
        if (next.error !== undefined && next.error !== run.error) {
          run = next;
          yield { type: "other", error: next.error };
          return;
        }
        run = next;
        yield { type: "other" };
      }
      const code = await c.exit;
      // The terminal message is assembled at the END rather than on a `result` event, because codex
      // has no single event that carries the answer: the text arrives as an item and the process
      // ending is what makes it final.
      if (code !== 0) {
        // A failed LAUNCH is named as one, for the same reason the `claude` sibling does it: a spawn
        // that never happened has no exit code, so it arrives as `-1`, which says nothing about the
        // binary being absent.
        yield { type: "other", error: launchError(c, config.command ?? CODEX_COMMAND) ?? exitMessage("codex", code, c.stderrTail?.()) };
        return;
      }
      if (run.text === undefined) {
        yield { type: "other", error: "codex produced no assistant message" };
        return;
      }
      // NO `costUsd`: codex reports token counts, not money. Inventing a number from a price table
      // this package has no access to would corrupt the run's roll-up — the adapter records
      // `costSource: "unknown"` instead, which is the truth.
      // With `--output-schema`, the final agent message IS the constrained value — codex has no field
      // of its own for it, unlike the `claude` sibling's `structured_output`. So it is parsed here, and
      // a schema that was asked for and came back unparseable is reported as no structured answer at
      // all rather than as a string that happens to look like one.
      const structured = opts.schema === undefined ? undefined : parseStructured(run.text);
      const result: AgentResult = {
        text: run.text,
        ...(structured !== undefined ? { structured } : {}),
        ...(run.sessionId !== undefined ? { sessionId: run.sessionId } : {}),
        ...(run.model !== undefined ? { model: run.model } : {}),
      };
      yield { type: "result", result };
    } finally {
      if (onAbort) opts.abortSignal?.removeEventListener("abort", onAbort);
      // ALWAYS kill, and always tear the bridge down: the consumer can finalize this generator early,
      // and without this the subprocess outlives the run while a leaked bridge goes on serving tools
      // for a run that ended.
      child?.kill();
      await bridge?.close();
    }
  };
}

/**
 * The final agent message as the value a schema asked for.
 *
 * Tolerant of the two things a model does to JSON it was told to produce: wrapping it in a fenced
 * block, and saying a sentence first. Neither is a reason to discard a correct answer — but an answer
 * that will not parse at all is reported as ABSENT rather than as text, because the caller asked for a
 * shape and a string is not one.
 */
function parseStructured(text: string): JsonValue | undefined {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(text);
  const candidates = [text.trim(), fenced?.[1]?.trim()].filter((c): c is string => c !== undefined && c.length > 0);
  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate) as JsonValue;
    } catch {
      // Try the next shape.
    }
  }
  return undefined;
}
