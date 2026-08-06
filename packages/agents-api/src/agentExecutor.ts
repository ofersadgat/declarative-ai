/**
 * The DELEGATED-AGENT `Executor` (DESIGN §4.4) — a {@link PromptExecutor} whose call reaches an agent.
 *
 * The claim DESIGN §4.4 already makes is that "a prompt op with reasoning, tools, and a step budget
 * already IS an agent, because the bounded tool loop is the same machinery". This class is that
 * sentence made structural. An agent answers the same `PromptOp`, lowered to the same
 * `LlmCallDefinition` — which is not a stretch: a call declaration is a system prompt, a turn list, a
 * model, a tool set and a step budget, and that is exactly what a coding-agent CLI is configured
 * with. What differs is one phase, {@link AgentExecutor.invoke}, where the call is made.
 *
 * That matters most for SESSIONS. Request shaping used to exist twice — once here in agent
 * vocabulary (`resume`/`forkSession`/`messages`) and once in `promptop` in provider vocabulary
 * (`messages`/`providerSessionId`) — with no way for either to reuse the other, so the fork rules had
 * to be stated and maintained in two places. They are now one inherited method that branches on ONE
 * declared fact, `capabilities.sessionResume`: a transport that resumes natively reads zero messages
 * and carries a handle; one that cannot, replays. Neither branch is agent-specific or provider-
 * specific — it is a property of the transport, declared where the engine already reads it.
 *
 * What stays below the seam is everything genuinely about agents: the native/injected tool split, the
 * deny floor, and routing the agent's own approval callback to `ctx.approve`.
 */
import {
  syncOnly,
  type ExecServices,
  type InlineFamily,
  type JsonSchema,
  type PromptOp,
  type ResolvedSession,
  type Tool,
} from "@declarative-ai/exec";
import type { CallDeps, LlmCallDefinition, LlmCallResult, LlmMetrics, LlmOutput, ModelMessage } from "@declarative-ai/llm";
import { PromptExecutor, type PromptExecutorOptions } from "@declarative-ai/promptop";
import { failureOf, type Capabilities, type RuntimeCapabilities } from "@declarative-ai/ops";
import type { BudgetMeter, BudgetMetrics, ExecMetrics } from "@declarative-ai/exec";
// Imported for its MODULE AUGMENTATION as much as for the type: `permissions` is what puts `approve`
// and `policy` on `ExecServices`, and this executor reads both. Without the import they are absent
// from the type in every package that compiles this one — which is how a whole approval path can
// typecheck as missing while the tests, running on the merged runtime shape, still pass.
import type { Approver } from "@declarative-ai/permissions";
import { sdkAgentQuery } from "./sdkQuery.js";
import type { AgentPermissionMode, AgentQuery, AgentQueryOptions, AgentResult, AgentSessionReader, InjectedTool } from "./seam.js";

/** Delegated agents: schema-constrained output isn't guaranteed (they answer in text), they mutate the
 *  workspace, run their own non-deterministic loop (not memoizable), gate tools via a callback, and are
 *  interactive (tool approvals route to our UI). Carried on the REGISTRY ENTRY, per §3.1 — and on the
 *  EXECUTOR, which is the same record because an entry is what an executor delegates to (DESIGN §3.2). */
export const DELEGATED_CAPS: RuntimeCapabilities = {
  interactive: true,
  readOnly: false,
  mutatesWorkspace: true,
  memoizable: false,
  structuredOutput: false,
  policyEnforcement: "callback",
  // NATIVE session resume, and native FORK with it (DESIGN.md §1.6). Declaring it is what tells the
  // session layer not to reach for the replay strategy: this transport branches server-side and reads
  // zero messages, where replay would resend the whole conversation for the same result.
  sessionResume: true,
  // Stated rather than left to the default, now that resume and fork are separable: this transport has
  // BOTH, and `forkSession` is the primitive that makes the second one true.
  sessionFork: true,
  streaming: true,
  runtime: "node",
};

/** What a delegated agent measures: execution timing/counts plus the spend it billed itself. */
export interface AgentMetrics extends ExecMetrics, BudgetMetrics {}

/**
 * Record money the agent has ALREADY spent. `reserve` returns `null` when the balance cannot cover the
 * amount — but this spend is a past FACT, not a request, so treating `null` as "nothing to do" leaves
 * the wallet reporting headroom it does not have and admits the next call against a phantom balance.
 * `debit` is the honest path when the meter offers one; without it we fall back to reserve/settle and
 * the overspend stays unrecorded on the ledger.
 *
 * Never throws: the money is gone either way, and failing the operation here would discard the agent's
 * result over a bookkeeping problem. The cost reaches the caller regardless, on `Result.metrics`.
 */
export async function debitSpentCost(meter: BudgetMeter, costUsd: number): Promise<void> {
  if (meter.debit) return meter.debit(costUsd);
  const reservation = await meter.reserve(costUsd);
  await reservation?.settle(costUsd);
}

/** Thrown when the delegated agent fails or is canceled. The invoking executor classifies it (a
 *  cancellation carries `name: "AbortError"`, which `classifyError` maps to `canceled`). */
export class AgentError extends Error {
  constructor(message: string, readonly canceled = false) {
    super(message);
    this.name = canceled ? "AbortError" : "AgentError";
  }
}

const PERMISSION_MODES: readonly AgentPermissionMode[] = ["default", "plan", "acceptEdits", "bypassPermissions"];

/**
 * The model id used when a call names none — a PLACEHOLDER, never a routing decision.
 *
 * Route-prefixed like every other id so it cannot be mistaken for a provider-native one, and so
 * anything that parses ids (a price table, a memo key, a diagnostic) reads it as the non-provider
 * route it is. What actually answers the call is whatever the agent binary is configured to use.
 */
export const AGENT_DEFAULT_MODEL = "agent/default";

export interface AgentExecutorOptions extends PromptExecutorOptions {
  /** The agent-query seam. Default: {@link sdkAgentQuery} (lazily loads `@anthropic-ai/claude-agent-sdk`). */
  query?: AgentQuery;
  /** Override the advertised capabilities (e.g. a variant that is workspace-read-only, or codex's
   *  `policyEnforcement: "config"` / `sessionFork: false` record). */
  capabilities?: RuntimeCapabilities;
  /** Inject `ctx.tools` into the agent over MCP so it calls OUR impls. Default `true`. Set `false` to
   *  instead pass every tool name as a NATIVE allow-list (the agent uses its own built-ins by name). */
  injectTools?: boolean;
  /** Per-logical-name overrides: a tool listed here resolves to the agent's NATIVE built-in (aliased)
   *  instead of being MCP-injected. Ignored tools default to injection. */
  nativeTools?: Record<string, { native: string }>;
  /**
   * Route the agent's tool approvals to `ctx.approve`. Default `true`.
   *
   * `false` states that this transport HAS no mid-run approval channel — codex is the case. Making it
   * an option rather than letting the query silently ignore an approver is the point: an executor
   * constructed this way declares `policyEnforcement: "config"`, and the engine answers that by
   * policy-WRAPPING its injected tools instead of handing them over raw, so the gate moves rather than
   * disappearing.
   */
  approvalCallback?: boolean;
  /** Reads a provider-side conversation back, for re-syncing after divergence (DESIGN.md §1.6). */
  readSession?: AgentSessionReader;
  /** The agent's NATIVE permission profile, when the caller pins one. */
  permissionMode?: AgentPermissionMode;
  /** Approval scope key for `ctx.approve`. Defaults to `"delegated"`. */
  approvalScope?: string;
  /**
   * What this transport is CALLED in a failure reason and in its own error messages.
   *
   * A failure that says only "the agent errored" is unactionable when three of them are wired: the
   * first question is always which binary or SDK produced it. Defaults to `claude-code`, which is what
   * the base class actually drives.
   */
  label?: string;
}

export class AgentExecutor extends PromptExecutor {
  /** The hierarchy's serializable discriminant — see `FunctionExecutor.kind` in `exec`. */
  static override readonly kind: string = "agent";

  override readonly capabilities: Capabilities;

  /**
   * The agent settings, read back off the options the base holds.
   *
   * A getter rather than a redeclared field. A subclass that redeclares `options` SHADOWS the base's,
   * so the object the base was constructed with and the object the subclass reads become two
   * different things the moment a subclass passes anything computed to `super` — which is exactly how
   * every `AgentCliExecutor` failure came out labelled `claude-code`.
   */
  protected get agent(): AgentExecutorOptions {
    return this.options as AgentExecutorOptions;
  }

  constructor(options: AgentExecutorOptions = {}) {
    super(options);
    this.capabilities = options.capabilities ?? DELEGATED_CAPS;
  }

  /** Present only when this transport can actually read a conversation back. The distinction is
   *  load-bearing: an absent reader means a resync starts EMPTY, and §11 requires that to be visible
   *  rather than mistaken for a conversation that happened to have nothing in it. */
  get sessionReader(): { read(providerSessionId: string): Promise<readonly unknown[]> } | undefined {
    const read = this.agent.readSession;
    return read !== undefined ? { read: (id: string) => read(id) } : undefined;
  }

  /** No provider endpoint is involved, so a missing router is not a reason to refuse. */
  protected override requiresRouter(): boolean {
    return false;
  }

  /**
   * Lower the op — tolerating a call that names no model.
   *
   * `LlmConfiguration.model` is required because a provider call cannot be ROUTED without one. A
   * delegated agent has no such problem: the binary picks its own model, from its own configuration
   * and its own subscription, and that is the whole reason an agent needs no API key. Refusing here
   * would make the zero-configuration case — the one an agent is best at — the one it cannot serve.
   *
   * So a placeholder is supplied when nothing else names one. It is inert: this executor never routes
   * on `definition.model`, and a real id (whatever selected this transport, e.g. `claude-cli/sonnet`)
   * passes through untouched for a transport that knows how to forward it.
   */
  protected override lower(op: PromptOp<InlineFamily>, tools: Record<string, Tool> | undefined): LlmCallDefinition {
    const inline = op.config !== null && typeof op.config === "object" && !Array.isArray(op.config) ? (op.config as Record<string, unknown>) : {};
    const named = typeof inline["model"] === "string" || typeof (this.options.defaults as { model?: unknown } | undefined)?.model === "string";
    return super.lower(named ? op : { ...op, config: { ...inline, model: AGENT_DEFAULT_MODEL } as never }, tools);
  }

  /** What this transport is called in a failure reason — see {@link AgentExecutorOptions.label}. */
  protected label(): string {
    return this.agent.label ?? "claude-code";
  }

  /** The transport this executor drives. Subclasses supply a subprocess; the default is the SDK. */
  protected query(): AgentQuery {
    return this.agent.query ?? sdkAgentQuery;
  }

  /**
   * THE CALL — configure the agent, run its own loop, read back its answer.
   *
   * A delegated agent runs ITS OWN loop, so this does not stream turns: it hands over a configured
   * request and waits for the terminal `result` message.
   */
  protected override async invoke(definition: LlmCallDefinition, env: CallDeps, ctx: ExecServices): Promise<LlmCallResult> {
    const startMs = Date.now();
    try {
      const result = await this.runAgent(definition, ctx);
      // A delegated agent spends real money inside its own loop, so the charge lands after the fact:
      // settle it against the wallet when one is injected. Absent meter ⇒ unmetered, as everywhere else.
      if (result.costUsd !== undefined && ctx.meter) await debitSpentCost(ctx.meter, result.costUsd);
      const output: LlmOutput = {
        value: result.text,
        finishReason: "stop",
        // The payload shape a session records: the agent's answer as one assistant turn. A delegated
        // agent does not hand back its internal log, so that turn is what a later replay against
        // another transport has to work from — saying so beats leaving the conversation empty.
        messages: [{ role: "assistant", content: result.text }] as ModelMessage[],
        ...(result.sessionId !== undefined ? { providerSessionId: result.sessionId } : {}),
      };
      return { value: output, metrics: this.agentMetrics(startMs, result.costUsd) };
    } catch (e) {
      // CLASSIFIED, not flattened. An agent SDK is an exception-shaped world, and the exception still
      // carries what happened: a 429 raised inside the agent's own loop is retriable, an abort is a
      // cancellation, a `retry-after` is a wait. Reporting all of it as `permanent` would make a
      // transient rate limit look like a broken workflow and defeat every retry wrapper above.
      return {
        error: failureOf(e, this.label()),
        value: { finishReason: "error" },
        metrics: this.agentMetrics(startMs, undefined),
      };
    }
  }

  /**
   * What a delegated agent measures.
   *
   * A delegated agent is the clearest case for cost NOT being an llm concern: it bills inside its own
   * loop and is the only thing that knows what it spent. `costUsd` is required, so an agent that
   * reported nothing says 0 with `costSource: "unknown"` rather than leaving it absent. One delegated
   * agent is one child call from the graph's point of view, which is how a budget gate sees through
   * the delegation without child records.
   */
  protected agentMetrics(startMs: number, costUsd: number | undefined): LlmMetrics {
    return {
      startMs,
      durationMs: Date.now() - startMs,
      childLlmCalls: 1,
      costUsd: costUsd ?? 0,
      costSource: costUsd !== undefined ? "provider" : "unknown",
      ...(costUsd !== undefined ? { childCostUsd: costUsd } : {}),
    } as LlmMetrics;
  }

  /** Build the query options from the lowered call, run the agent, and return its terminal result. */
  private async runAgent(definition: LlmCallDefinition, ctx: ExecServices): Promise<AgentResult> {
    const inject = this.agent.injectTools ?? true;
    const nativeMap = this.agent.nativeTools ?? {};
    const wantsApprovalCallback = this.agent.approvalCallback ?? true;
    const approve = ctx.approve;
    // The APPROVAL SCOPE — a resource-bundle key, not a conversation. The two used to be one string
    // and cannot be: a conversation moves on every call, so an approval scoped to it would cover
    // exactly one tool call (DESIGN.md §5.1). The conversation is `ctx.session`.
    const scope = this.agent.approvalScope ?? "delegated";
    const session = ctx.session as ResolvedSession<ModelMessage> | undefined;

    // The run is driven by the caller's abort signal directly. A run that completes without aborting
    // must not leave a listener attached to a possibly long-lived, shared `ctx.abortSignal`.
    const signal = ctx.abortSignal ?? new AbortController().signal;

    // A per-tool `deny` in the authored baseline needs no human, so it must reach the agent as
    // CONFIGURATION rather than waiting for an approval that will never be asked for. Native names are
    // what the agent addresses, so an aliased tool is denied under its `native` name.
    const denied = Object.entries(ctx.policy?.baseline?.tools ?? {})
      .filter(([, mode]) => mode === "deny")
      .map(([name]) => nativeMap[name]?.native ?? name);
    const denySet = new Set(denied);

    // Resolve each logical tool to NATIVE (the agent's built-in, aliased) or MCP-INJECTED (our impl,
    // ctx-bound). The engine hands a delegated runtime RAW tools, and authorization flows through
    // `canUseTool` → `ctx.approve`, so injected tools are not double-gated.
    const tools = (ctx.tools ?? this.options.tools) as Record<string, Tool> | undefined;
    let allowedTools: string[] | undefined;
    let mcpTools: Record<string, InjectedTool> | undefined;
    if (tools) {
      const native: string[] = [];
      const injected: Record<string, InjectedTool> = {};
      for (const [name, tool] of Object.entries(tools)) {
        const ref = nativeMap[name];
        // A `deny` is an unconditional floor: the tool is never OFFERED, native or injected. An injected
        // tool is addressed as `mcp__dai__<name>`, which no logical-name deny entry matches, so leaving
        // it injected would route around the floor entirely — drop it here.
        if (denySet.has(ref ? ref.native : name)) continue;
        if (!inject || ref) native.push(ref ? ref.native : name);
        else injected[name] = { description: tool.description, inputSchema: tool.inputSchema as JsonSchema, run: (input) => tool.run(input, ctx) };
      }
      // `allowedTools` PRE-APPROVES; denied tools are already excluded above, native and injected alike.
      allowedTools = native;
      if (Object.keys(injected).length > 0) mcpTools = injected;
    }

    // The session decision was already made by `applySession`, which is the point of the refactor:
    // `providerSessionId` is present exactly when this transport may resume, and `messages` carries a
    // replayed transcript exactly when it may not. All that is left here is spelling those two facts
    // in the agent's own vocabulary.
    const resume = definition.providerSessionId;
    const replayed = resume === undefined && session !== undefined ? definition.messages : undefined;

    const queryOptions: AgentQueryOptions = {
      prompt: this.renderPrompt(definition),
      ...(this.agentModel(definition) !== undefined ? { model: this.agentModel(definition)! } : {}),
      cwd: ctx.workspace?.root,
      allowedTools,
      ...(denied.length > 0 ? { disallowedTools: denied } : {}),
      mcpTools,
      // The injected-tool input gate is sync (`seam.ts`); the ctx seam is maybe-async — narrow
      // FAIL-CLOSED (json's `syncOnly`) rather than let an async validator read as a pass.
      ...(ctx.validator !== undefined ? { validator: syncOnly(ctx.validator) } : {}),
      permissionMode: this.permissionMode(),
      ...(resume !== undefined ? { resume, ...(session?.mode === "fork" ? { forkSession: true } : {}) } : {}),
      ...(replayed !== undefined ? { messages: replayed as never } : {}),
      // Route the agent's native tool-approval callback through our approver (DESIGN §5.1,
      // "Delegated approval fidelity").
      canUseTool:
        approve && wantsApprovalCallback
          ? async (req) => {
              const decision = await approve({ tool: req.toolName, input: req.input, sessionId: scope });
              return decision.decision === "allow" ? { allow: true } : { allow: false, reason: `denied by permission policy` };
            }
          : undefined,
      abortSignal: signal,
    };

    let result: AgentResult | undefined;
    try {
      for await (const msg of this.query()(queryOptions)) {
        if (msg.error) throw new AgentError(`${this.label()} agent error: ${msg.error}`);
        if (msg.type === "result" && msg.result) result = msg.result;
      }
    } catch (e) {
      if (signal.aborted) throw new AgentError("aborted", true);
      if (e instanceof AgentError) throw e;
      throw new AgentError(`${this.label()} query threw: ${(e as Error).message}`);
    }
    if (signal.aborted) throw new AgentError("aborted", true);
    if (!result) throw new AgentError(`${this.label()} produced no result message`);
    return result;
  }

  /**
   * The model to ask the agent for — provider-native, or `undefined` for "use your own default".
   *
   * The route prefix is STRIPPED, because it named this transport and means nothing to the binary:
   * `claude-cli/sonnet` reaches `claude` as `sonnet`, which is a model it knows, rather than as
   * `claude-cli/sonnet`, which it would reject.
   *
   * {@link AGENT_DEFAULT_MODEL} maps to `undefined` rather than to the literal `default`. It is the
   * placeholder this class supplies when a call names no model at all, so forwarding it would turn
   * "whatever you normally use" into a request for a model named `default` — an argument every one of
   * these binaries would refuse, and the zero-configuration case is precisely the one that must work.
   */
  protected agentModel(definition: LlmCallDefinition): string | undefined {
    const id = definition.model;
    if (typeof id !== "string" || id === AGENT_DEFAULT_MODEL) return undefined;
    const slash = id.indexOf("/");
    const native = slash > 0 ? id.slice(slash + 1) : id;
    return native.length > 0 ? native : undefined;
  }

  /** An author-supplied permission mode, ignoring an unknown value. */
  protected permissionMode(): AgentPermissionMode | undefined {
    const m = this.agent.permissionMode;
    return m !== undefined && PERMISSION_MODES.includes(m) ? m : undefined;
  }

  /**
   * The instruction text, from a call declaration that may carry either shape.
   *
   * A delegated agent takes ONE prompt — there is no message array on the wire — so a replayed
   * transcript is rendered into text. That lossiness is why SESSIONS.md records such a branch as
   * summary-seeded rather than as a native fork, and it is why the cheap paths avoid replay entirely.
   */
  protected renderPrompt(definition: LlmCallDefinition): string {
    const system = definition.system !== undefined ? `${definition.system}\n\n` : "";
    if (typeof definition.prompt === "string") return `${system}${definition.prompt}`;
    const turns = definition.messages ?? definition.prompt ?? [];
    const rendered = (turns as ModelMessage[])
      .map((m) => `${String(m.role).toUpperCase()}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`)
      .join("\n\n");
    return `${system}${rendered}`;
  }
}

/**
 * The agent reached through an in-process SDK.
 *
 * A NAME rather than behaviour: {@link AgentExecutor}'s default transport already is the SDK, and the
 * subclass exists so the two invocation mechanisms are equally visible in the hierarchy — the
 * alternative reads as though the CLI were a special case of "agent" rather than one of two peers.
 * Its sibling is `AgentCliExecutor` in `@declarative-ai/agents-cli`.
 */
export class AgentApiExecutor extends AgentExecutor {
  static override readonly kind: string = "agent-api";
}
