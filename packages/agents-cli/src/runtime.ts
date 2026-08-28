/**
 * The CLI-driven delegated agent as a `runtime` REGISTRY ENTRY. It is the same adapter logic as the
 * SDK-driven one — a delegated agent runs ITS OWN loop, so we configure it and read back its answer —
 * differing only in the two things the invocation mechanism actually decides:
 *
 *  - how the agent is reached ({@link createCliAgentQuery}, a subprocess rather than an SDK call);
 *  - how the safety policy is enforced: `callback` when the agent calls OUR tool impls back over the MCP
 *    bridge (the default), `config` when it uses its own built-ins and the up-front flags are the only
 *    thing standing between it and them.
 *
 * That second field is REQUIRED and total on the entry (DESIGN §3.3), which is exactly why the
 * difference is legible to the permission layer instead of defaulting silently.
 */
import { createClaudeCodeFunction, DELEGATED_CAPS, type ClaudeCodeFunctionOptions } from "@declarative-ai/agents-api";
import type { RuntimeCapabilities } from "@declarative-ai/exec";
import { createCliAgentQuery, type CliAgentOptions } from "./cliQuery.js";

/**
 * A CLI agent routes each gated tool-use back to our approver — over the MCP bridge, via
 * `--permission-prompt-tool` — so its enforcement model is `callback`, the same as the SDK adapter's.
 * The mechanism differs (an MCP tool the CLI calls, rather than an in-process function); the guarantee
 * does not, and the guarantee is what this field describes.
 *
 * This is only honest because `cliArgv` keeps injected tools OFF `--allowedTools`. That flag pre-approves,
 * so listing them there meant the prompt tool was never asked about the very tools this adapter
 * implements — while the engine, reading `callback` here, had already skipped its own `withPermission`
 * wrapping. Both gates open. The two facts are one invariant: declare `callback` only where a call really
 * does reach `ctx.approve`.
 */
export const CLI_DELEGATED_CAPS: RuntimeCapabilities = { ...DELEGATED_CAPS, policyEnforcement: "callback", sessionSteering: true };

/**
 * The honest record when the agent uses its OWN built-in tools (`injectTools: false`, or a `nativeTools`
 * alias): those never reach our approver — they are governed by the up-front posture the CLI is
 * configured with (`--permission-mode`, `--allowedTools`, `--disallowedTools`), which is what `config`
 * names. Declaring `callback` there would tell the engine a callback protects tools that no callback
 * ever sees.
 *
 * What this CANNOT express is the run with no approver configured at all: capabilities are fixed at
 * registration and `ctx.approve` is a per-run fact. In that case the engine hands raw tools to a `config`
 * adapter just as it does to a `callback` one, so the declaration makes no difference — a pre-existing
 * gap in the engine, not one this record can close.
 */
export const CLI_CONFIG_ONLY_CAPS: RuntimeCapabilities = { ...DELEGATED_CAPS, policyEnforcement: "config", sessionSteering: true };

/**
 * The claude CLI adapter STEERS; the codex one does not.
 *
 * `interrupt` is what `sessionSteering` promises here, and it is real: the adapter drives the CLI
 * over `--input-format stream-json`, so a `control_request` reaches a running turn. The turn ends
 * early, a `result` still arrives, and the call settles with the partial answer — which is the
 * difference between "stop and tell me what you found" and `kill()`, which answers that by throwing
 * the answer away. Measured against claude 2.1.246: acknowledged in ~1 ms, and the process stays
 * alive afterwards, which is what lets a PAUSE hold a warm session instead of releasing it.
 *
 * `send` / `setPermissionMode` / `setModel` are still absent. The channel now exists for all three,
 * but absent means unsupported and a caller reads these to decide what to offer — so they arrive when
 * they are implemented, not when they become possible.
 *
 * codex keeps `sessionSteering: false`: SIGINT to that subprocess is a kill, not a graceful turn end,
 * and offering `interrupt` on top of it would be the confusion this capability exists to prevent.
 */

export interface CliAgentFunctionOptions extends Omit<ClaudeCodeFunctionOptions, "query">, CliAgentOptions {}

/**
 * Build the CLI-driven adapter. Register it with
 * `registry.functions.registerRuntime("claude-cli", fn.run, fn.capabilities)`.
 */
export function createCliAgentFunction(options: CliAgentFunctionOptions = {}): ReturnType<typeof createClaudeCodeFunction> {
  const { command, args, spawn, startBridge, ...rest } = options;
  return createClaudeCodeFunction({
    label: "claude-cli",
    ...rest,
    // Tool injection now means the same thing it does for the SDK adapter — the agent calls OUR impls,
    // reached over the bridge — so the default matches, and a workflow authored against one adapter
    // behaves the same against the other. Turning injection OFF changes what enforces the policy, not
    // just where the tools come from, so the declared capability follows it.
    capabilities: options.capabilities ?? (options.injectTools === false ? CLI_CONFIG_ONLY_CAPS : CLI_DELEGATED_CAPS),
    query: createCliAgentQuery({ command, args, spawn, startBridge }),
  });
}
