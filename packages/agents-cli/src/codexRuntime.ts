/**
 * The CODEX delegated agent as a `runtime` REGISTRY ENTRY.
 *
 * Same adapter logic as its two siblings — a delegated agent runs ITS OWN loop, so we configure it and
 * read back its answer — differing in the two things the invocation mechanism actually decides: how the
 * agent is reached ({@link createCodexAgentQuery}), and what its capability record can honestly claim.
 *
 * Both differences are recorded on the entry rather than discovered at run time, because the engine
 * reads the entry to decide how to treat the call (DESIGN §3.3).
 */
import { createClaudeCodeFunction, DELEGATED_CAPS, type ClaudeCodeFunctionOptions } from "@declarative-ai/agents-api";
import type { RuntimeCapabilities } from "@declarative-ai/exec";
import { createCodexAgentQuery, type CodexAgentOptions } from "./codexQuery.js";

/**
 * What codex can honestly claim.
 *
 * Two fields carry the whole difference from `claude-cli`, and both are load-bearing:
 *
 *  - **`policyEnforcement: "config"`.** `codex exec` has no `--permission-prompt-tool` analogue, so no
 *    tool-use reaches `ctx.approve` mid-run. What DOES enforce the policy is the up-front posture:
 *    `--sandbox`, and the fact that the only tools we hand it are ones we implement. Declaring
 *    `callback` here would tell the engine a gate exists that nothing implements — and the engine
 *    answers `callback` by handing over RAW tools, so the claim would open both gates at once. Under
 *    `config` it wraps them with `withPermission` instead, which is why refusing the approval callback
 *    moves the gate rather than removing it.
 *  - **`sessionFork: false`.** `codex exec resume <id>` appends server-side, so resume is real; there
 *    is no fork primitive anywhere in the CLI, so a branch is REPLAYED (SESSIONS.md §6). Splitting the
 *    two is what stops a fork inheriting the parent's handle and putting two branches in one session.
 *
 * `interactive: false` follows from the first: a codex run in a workflow has no channel to ask a human
 * anything, so nothing should route it work that needs one.
 */
export const CODEX_CAPS: RuntimeCapabilities = {
  ...DELEGATED_CAPS,
  policyEnforcement: "config",
  interactive: false,
  sessionResume: true,
  sessionFork: false,
  // Codex streams JSONL events, but this adapter surfaces only the terminal answer — no partial text
  // reaches a caller, and `streaming: true` is a promise about what a consumer can observe.
  streaming: false,
  // No graceful turn end. The only mid-run signal a `codex exec` subprocess has is SIGINT, which is a
  // KILL: the answer it had produced is lost rather than returned. Offering `interrupt` on top of that
  // would answer "stop and tell me what you found" by discarding what it found.
  sessionSteering: false,
};

export interface CodexAgentFunctionOptions extends Omit<ClaudeCodeFunctionOptions, "query" | "approvalCallback">, CodexAgentOptions {}

/**
 * Build the codex adapter. Register it with
 * `registry.functions.registerRuntime("codex-cli", fn.run, fn.capabilities)`.
 *
 * `approvalCallback: false` is fixed, not an option: it is the same fact as `policyEnforcement:
 * "config"` said at the other end, and letting a caller set one without the other would produce an
 * entry whose declaration and behaviour disagree.
 */
export function createCodexAgentFunction(options: CodexAgentFunctionOptions = {}): ReturnType<typeof createClaudeCodeFunction> {
  const { command, args, spawn, startBridge, sandbox, ...rest } = options;
  return createClaudeCodeFunction({
    label: "codex",
    ...rest,
    capabilities: options.capabilities ?? CODEX_CAPS,
    approvalCallback: false,
    query: createCodexAgentQuery({
      ...(command !== undefined ? { command } : {}),
      ...(args !== undefined ? { args } : {}),
      ...(spawn !== undefined ? { spawn } : {}),
      ...(startBridge !== undefined ? { startBridge } : {}),
      ...(sandbox !== undefined ? { sandbox } : {}),
    }),
  });
}
