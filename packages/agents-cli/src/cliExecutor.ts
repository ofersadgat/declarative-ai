/**
 * The CLI-driven delegated agent as an `Executor` — the sibling of the SDK-driven
 * {@link AgentApiExecutor}, split by INVOCATION MECHANISM (DESIGN §4.4).
 *
 * Everything that makes an agent an agent is inherited: the same `PromptOp` is lowered to the same
 * call declaration, the session is resolved by the same rule, tools are split native-vs-injected the
 * same way, and approvals route to the same `ctx.approve`. What this class supplies is a transport —
 * a subprocess speaking newline-delimited JSON instead of an in-process SDK call — and the capability
 * record that says what that transport can actually enforce.
 *
 * "CLI is a transport, not a category" (DESIGN §4.4) is the claim; this file is two `query()` methods.
 */
import { AgentExecutor, type AgentExecutorOptions } from "@declarative-ai/agents-api";
import type { AgentQuery } from "@declarative-ai/agents-api";
import { createCliAgentQuery, type CliAgentOptions } from "./cliQuery.js";
import { createCodexAgentQuery, type CodexAgentOptions } from "./codexQuery.js";
import { CLI_CONFIG_ONLY_CAPS, CLI_DELEGATED_CAPS } from "./runtime.js";
import { CODEX_CAPS } from "./codexRuntime.js";

export interface AgentCliExecutorOptions extends AgentExecutorOptions, CliAgentOptions {}

/**
 * `claude` as a subprocess.
 *
 * The declared capability follows tool injection, because injection decides what ENFORCES the policy
 * rather than merely where the tools come from: with tools injected the agent calls our impls back
 * over the MCP bridge and every gated call reaches `ctx.approve` (`callback`); with `injectTools:
 * false` it uses its own built-ins under nothing but the up-front flags (`config`). Declaring
 * `callback` in the second case would tell the engine a callback protects tools no callback ever sees.
 */
export class AgentCliExecutor extends AgentExecutor {
  static override readonly kind: string = "agent-cli";

  /**
   * The transport settings, read back off the options the BASE holds.
   *
   * Not a constructor parameter property, which is the shape this class had first and which was
   * quietly wrong: a redeclared `options` shadows the base's, so the object the base was constructed
   * with (label and capabilities filled in) and the object the subclass read (the caller's raw one)
   * were two different things. Every failure from this executor was reported as `claude-code`.
   */
  private get cli(): AgentCliExecutorOptions {
    return this.agent as AgentCliExecutorOptions;
  }

  constructor(options: AgentCliExecutorOptions = {}) {
    super({
      // Before the spread, so an explicit label still wins; capabilities after, because they are
      // COMPUTED from what the caller passed and must not be overwritten by an absent key.
      label: "claude-cli",
      ...options,
      capabilities: options.capabilities ?? (options.injectTools === false ? CLI_CONFIG_ONLY_CAPS : CLI_DELEGATED_CAPS),
    });
  }

  protected override query(): AgentQuery {
    if (this.cli.query !== undefined) return this.cli.query;
    const { command, args, spawn, startBridge } = this.cli;
    return createCliAgentQuery({ command, args, spawn, startBridge });
  }
}

export interface AgentCodexExecutorOptions extends AgentExecutorOptions, CodexAgentOptions {}

/**
 * `codex exec` as a subprocess.
 *
 * Two capability differences from `claude`, both real rather than conservative defaults: codex offers
 * nothing like `--permission-prompt-tool`, so there is no mid-run approval channel and its enforcement
 * is `config` — its sandbox, set up front; and it has `resume` but no fork primitive, so
 * `sessionFork: false`. That second flag is what makes the inherited `applySession` replay a fork here
 * while letting an append continue server-side — the branch is read off the capability, so this class
 * states the fact and inherits the behaviour.
 */
export class AgentCodexExecutor extends AgentExecutor {
  static override readonly kind: string = "agent-codex";

  /** See {@link AgentCliExecutor} — read off the base's options, never a shadowing redeclaration. */
  private get cli(): AgentCodexExecutorOptions {
    return this.agent as AgentCodexExecutorOptions;
  }

  constructor(options: AgentCodexExecutorOptions = {}) {
    super({
      label: "codex",
      // Codex answers a narrowing profile with its SANDBOX, not with a deny list — it has no such
      // flag, and `codexRefusal` would refuse the run over the claude names the base's default list
      // carries. The mapping is exact where it is not for claude: `sandboxFor("plan")` is nothing but
      // `--sandbox read-only`, with no planning behaviour behind the word, so a `read-only` profile
      // may borrow it. The base maps only `plan` on its own precisely because claude's `plan` IS a
      // different instruction.
      mutatingNativeTools: [],
      readOnlyProfileMode: "plan",
      ...options,
      capabilities: options.capabilities ?? CODEX_CAPS,
      // `codex exec` cannot ask, so an approver must not be handed to it as though it could.
      approvalCallback: options.approvalCallback ?? false,
    });
  }

  /** Codex takes its OWN bag out of `providerOptions`, not claude's — that keying is the whole reason
   *  `providerOptions` is per-provider, so one config can carry settings for several transports. */
  protected override providerOptionsKey(): string {
    return "codex";
  }

  protected override query(): AgentQuery {
    if (this.cli.query !== undefined) return this.cli.query;
    const { command, args, spawn, sandbox } = this.cli;
    return createCodexAgentQuery({ command, args, spawn, sandbox });
  }
}
