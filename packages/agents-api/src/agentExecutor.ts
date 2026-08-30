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
  EventQueue,
  syncOnly,
  type ExecControl,
  type ExecHandle,
  type ExecServices,
  type FunctionInputs,
  type InlineFamily,
  type JsonSchema,
  type JsonValue,
  type Operation,
  type PromptOp,
  type ResolvedSession,
  type ResolvedValue,
  type Tool,
} from "@declarative-ai/exec";
import type {
  CallDeps,
  LlmCallDefinition,
  LlmCallResult,
  LlmMetrics,
  LlmOutput,
  Entry,
  ModelMessage,
  RawMessage,
  ReasoningSegment,
  ToolCall,
  ToolResult,
} from "@declarative-ai/llm";
import { entriesOfMessages } from "@declarative-ai/llm";

/**
 * Whose vocabulary a delegated agent's `providerData` and block types belong to.
 *
 * The transport IS the provider here: a delegated agent hands back its own log in its own shape, and
 * a reader has to know which one to interpret an unrecognized block by.
 */
const AGENT_PROVIDER = "anthropic";
import { PromptExecutor, type PromptExecutorOptions } from "@declarative-ai/promptop";
import { failureOf, type Capabilities, type RuntimeCapabilities } from "@declarative-ai/ops";
import type { BudgetMeter, BudgetMetrics, ExecMetrics } from "@declarative-ai/exec";
// Imported for its MODULE AUGMENTATION as much as for the type: `permissions` is what puts `approve`
// and `policy` on `ExecServices`, and this executor reads both. Without the import they are absent
// from the type in every package that compiles this one — which is how a whole approval path can
// typecheck as missing while the tests, running on the merged runtime shape, still pass.
import type { Approver, PermissionMode, UserAnswers, UserQuestion } from "@declarative-ai/permissions";
import { mcpToolName } from "./mcpTools.js";
import { sdkAgentQuery } from "./sdkQuery.js";
import { isRetriableAgentError } from "./streamMessages.js";
import type { AgentPermissionMode, AgentQuery, AgentQueryOptions, AgentResult, AgentRun, AgentSessionReader, InjectedTool } from "./seam.js";

/**
 * The per-call EVENT SINK, threaded on a shallow copy of `ctx`.
 *
 * `start` is the phase that owns the handle and `invoke` is the phase that produces events, and the
 * base class hands nothing between them — so the queue has to travel with the call. It travels on
 * `ctx` rather than on the instance because one executor serves many concurrent calls, and an instance
 * field would deliver one call's partial output onto another's stream.
 *
 * A SYMBOL rather than a declared `ExecServices` field, deliberately: module augmentation is global and
 * would put a slot on the public services bundle that no caller should ever set. This is an internal
 * channel between two phases of one class, and it is spelled like one.
 */
const EVENT_SINK = Symbol("declarative-ai.agent.events");

/** The per-call channel between `start` (which owns the handle) and `invoke` (which produces the run). */
interface AgentChannel {
  events: EventQueue;
  /** The live run, once `invoke` has created one. Written once per call; read live by the handle's
   *  `control` getter, so a caller holding the handle before the run started still steers it after. */
  run?: AgentRun;
  /** The stream is over. Every control method becomes a no-op — an interrupt racing the run's own end
   *  is the ORDINARY case (a user presses Stop as the answer lands) and must not become an error. */
  finished?: boolean;
  /**
   * Control requests made BEFORE the run existed, replayed the moment it does.
   *
   * The window is real and not small: `start` returns its handle synchronously, while lowering and
   * session resolution are awaits that happen before the transport is reached. A caller pressing Stop
   * in that window is asking for the run to stop, and dropping the request because the object was not
   * built yet would answer by letting it run to completion — the worst possible response to Stop.
   */
  pending: Array<(run: AgentRun) => Promise<void>>;
}

/** `ctx` carrying this executor's per-call channel. */
type WithEventSink = { [EVENT_SINK]?: AgentChannel };

/** Everything ONE agent turn produced, accumulated off its stream. */
interface AgentTurn {
  result: AgentResult;
  /**
   * The OUTPUT text the assistant turns carried, accumulated.
   *
   * Normally the same thing the terminal `result` says, and then unused. It earns its place on an
   * INTERRUPTED run, where the terminal message carries no text at all — the partial answer exists only
   * in the turns already streamed, and this is what stops "stop and tell me what you found" from
   * answering with nothing.
   */
  text: string;
  /** The agent's own log, verbatim — assistant and user turns in the order it produced them. */
  messages: ModelMessage[];
  thinking: ReasoningSegment[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
  /** Subagent conversations, keyed by the spawning tool call — kept OUT of `messages` (see LlmOutput). */
  sidechains: Map<string, ModelMessage[]>;
  /** Opaque provider events, each pinned to how many main-thread messages preceded it. */
  providerEvents: Array<{ index: number; event: JsonValue }>;
}

/** Delegated agents: they mutate the workspace, run their own non-deterministic loop (not memoizable),
 *  gate tools via a callback, and are interactive (tool approvals route to our UI). Carried on the
 *  REGISTRY ENTRY, per §3.1 — and on the EXECUTOR, which is the same record because an entry is what an
 *  executor delegates to (DESIGN §3.2). */
export const DELEGATED_CAPS: RuntimeCapabilities = {
  interactive: true,
  readOnly: false,
  mutatesWorkspace: true,
  memoizable: false,
  /**
   * TRUE, and natively so. Both `claude` transports carry the output schema — `--json-schema` on argv,
   * `outputFormat: {type: "json_schema"}` through the SDK — and retry inside their own loop until the
   * value validates, answering on the terminal message's `structured_output`.
   *
   * It read `false` while this was unimplemented, with the note "they answer in text". That was true of
   * the adapter rather than of the transports, and the cost of leaving it there was a whole class of
   * workflow: any state declaring outputs failed on an agent route with "did not produce required
   * output", which names the state and not the reason.
   */
  structuredOutput: true,
  policyEnforcement: "callback",
  // NATIVE session resume, and native FORK with it (DESIGN.md §1.6). Declaring it is what tells the
  // session layer not to reach for the replay strategy: this transport branches server-side and reads
  // zero messages, where replay would resend the whole conversation for the same result.
  sessionResume: true,
  // Stated rather than left to the default, now that resume and fork are separable: this transport has
  // BOTH, and `forkSession` is the primitive that makes the second one true.
  sessionFork: true,
  // `--resume-session-at` / `resumeSessionAt`: a copy cut at a named message, so a branch behind the
  // remote's tip is still one server-side operation rather than a full replay.
  sessionForkAt: true,
  streaming: true,
  // The run can be STEERED while it runs — interrupted, redirected, given more input. True for this
  // transport because the SDK's `Query` carries the control requests; a caller reads it to decide
  // whether to offer a Stop button BEFORE the call, rather than by pressing one and finding out.
  sessionSteering: true,
  runtime: "node",
};

/** What a delegated agent measures: execution timing/counts plus the spend it billed itself. */
export interface AgentMetrics extends ExecMetrics, BudgetMetrics {}


/** Thrown when the delegated agent fails or is canceled. The invoking executor classifies it (a
 *  cancellation carries `name: "AbortError"`, which `classifyError` maps to `canceled`). */
export class AgentError extends Error {
  /**
   * Whether a RETRY could plausibly get past this — read by `classifyError`, which looks for exactly
   * this field. Absent ⇒ it falls through to `permanent`, which is the right default for a failure
   * nothing told us about.
   */
  readonly retryable?: boolean;
  /** 429 for a rate limit, so `isRateLimit` sets `rateLimited` on the failure and a limiter upstream
   *  sees it. The shared classifier reads transport vocabulary; this speaks it. */
  readonly status?: number;

  /**
   * @param code the agent's own failure code (`rate_limit`, `authentication_failed`, …), when it gave
   * one. Without it every delegated failure classified as `permanent` — including a transient overload
   * inside the agent's own loop, which is precisely the case {@link AgentExecutor.invoke} promises to
   * classify rather than flatten.
   */
  constructor(message: string, readonly canceled = false, readonly code?: string) {
    super(message);
    this.name = canceled ? "AbortError" : "AgentError";
    if (code !== undefined) {
      this.retryable = isRetriableAgentError(code);
      if (code === "rate_limit") this.status = 429;
    }
  }
}

const PERMISSION_MODES: readonly AgentPermissionMode[] = ["default", "plan", "acceptEdits", "bypassPermissions"];

/**
 * The agent's own "put a question to the human" tool.
 *
 * It arrives on the SAME callback as every permission ask — the CLI routes it there even when an
 * allow rule matches, because the call cannot be pre-approved: the call IS the question, and its
 * answer travels back as input (`updatedInput.answers`, the documented contract). So it must be
 * picked off BEFORE the gate. Left to the gate it is an unclassifiable native tool, which escalates
 * to the approver — a human asked to APPROVE being asked a question they are then never shown, and
 * an allow that resolves the question with no answers at all.
 */
export const ASK_USER_TOOL = "AskUserQuestion";

/**
 * The questions off an `AskUserQuestion` input, read defensively — the input crossed a process
 * boundary as untyped JSON, and a malformed batch must become "nothing to ask" rather than a throw
 * inside the permission callback (which the transports surface as a harness error the agent then
 * works around).
 */
export function questionsOf(input: FunctionInputs): UserQuestion[] {
  const raw = (input as { questions?: unknown }).questions;
  if (!Array.isArray(raw)) return [];
  const out: UserQuestion[] = [];
  for (const entry of raw) {
    if (entry === null || typeof entry !== "object" || Array.isArray(entry)) continue;
    const q = entry as Record<string, unknown>;
    if (typeof q["question"] !== "string" || q["question"].length === 0) continue;
    const options = Array.isArray(q["options"])
      ? (q["options"] as unknown[])
          .filter((o): o is Record<string, unknown> => o !== null && typeof o === "object" && !Array.isArray(o))
          .filter((o) => typeof o["label"] === "string")
          .map((o) => ({
            label: o["label"] as string,
            ...(typeof o["description"] === "string" ? { description: o["description"] } : {}),
          }))
      : [];
    out.push({
      question: q["question"],
      ...(typeof q["header"] === "string" ? { header: q["header"] } : {}),
      options,
      ...(q["multiSelect"] === true ? { multiSelect: true } : {}),
    });
  }
  return out;
}

/**
 * `claude`'s own built-ins that can mutate — what a `read-only` profile denies up-front.
 *
 * A blocklist, not an enumeration of the agent's whole tool set, and the distinction is what keeps it
 * honest: an unlisted read-only built-in (`Read`, `Glob`, `Grep`) is exactly what the profile permits,
 * and an unlisted WRITER a future release adds still hits the permission callback, where the gate
 * escalates anything it cannot classify. `Task` and `Agent` are here because a sub-agent inherits
 * tools this deny list never sees; `SlashCommand` because a command file can instruct anything.
 */
export const CLAUDE_MUTATING_BUILTINS: readonly string[] = [
  "Bash",
  "Edit",
  "Write",
  "MultiEdit",
  "NotebookEdit",
  "Task",
  "Agent",
  "SlashCommand",
];

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
   *  instead pass every tool name as a NATIVE allow-list (the agent uses its own built-ins by name).
   *  This governs `ctx.tools` ONLY — {@link AgentExecutorOptions.extraTools} is the separate question. */
  injectTools?: boolean;
  /**
   * Tools to ADD to whatever the agent already has, always injected, whatever `injectTools` says.
   *
   * The gap this closes: `injectTools` is one switch over `ctx.tools`, so "the agent's own tools for
   * everything, plus these extras" was not expressible at all — `false` routed everything native and
   * `true` replaced the lot. That is precisely what a host exposing its OWN capability to an otherwise
   * stock agent needs: a preview pane, a build runner, a ticket lookup. Nothing about those wants the
   * agent to stop using its own `Read`.
   *
   * Separating them is the point: `injectTools` answers "replace `ctx.tools`", this answers "add
   * these", and the two compose.
   */
  extraTools?: Record<string, Tool>;
  /** Per-logical-name overrides: a tool listed here resolves to the agent's NATIVE built-in (aliased)
   *  instead of being MCP-injected. Ignored tools default to injection. */
  nativeTools?: Record<string, { native: string }>;
  /**
   * Which of the agent's OWN built-ins an injected tool DISPLACES, keyed by logical name.
   *
   * ✅ OBSERVED (claude 2.1.142). Injection alone does not displace anything. A live run with
   * `read_file` injected and no denies gave the agent both `Read` and `mcp__dai__read_file`, and it used
   * `Read` — every time, because its system prompt steers it there. Disallow `Read` and the SAME run
   * reaches for `mcp__dai__read_file`, our impl executes, and the call goes through `ctx.approve`.
   *
   * So injection without this is not the portable-vocabulary story it exists for (DESIGN §5.1) — it is
   * a second set of tools the model ignores. Naming the displaced built-ins puts them on
   * `disallowedTools`, which the agent checks BEFORE its allow-list, so the substitution is real.
   *
   * It is the caller's to state rather than a table here: which built-in a logical tool stands in for is
   * a fact about the agent being driven, and this executor drives more than one.
   */
  replacesNative?: Record<string, string | readonly string[]>;
  /**
   * The agent's OWN built-ins that can MUTATE — write files, run commands, spawn sub-agents — denied
   * up-front when the session's profile is `read-only`.
   *
   * The gap this closes is the one `ToolGate.modeOf` documents from the other side: a `read-only`
   * profile is enforced per-tool by the gate, but the gate cannot CLASSIFY an agent's own built-in
   * (no `readOnly` we know), so every such call escalates to a human — including the ones the profile
   * plainly forbids, which a distracted click then allows. The built-ins whose write-capability is a
   * known fact about the transport need no escalation: they are denied as CONFIGURATION, before the
   * model can reach for them. The callback stays the floor for everything this list cannot name.
   *
   * Defaults to {@link CLAUDE_MUTATING_BUILTINS}, because the base class drives `claude`. A transport
   * with a different vocabulary states its own list — codex passes `[]` and answers the profile with
   * its sandbox instead, which is the channel it actually has.
   */
  mutatingNativeTools?: readonly string[];
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
  /**
   * WHICH binary answers — an absolute path, or a name the transport resolves.
   *
   * Absent ⇒ each transport's own default. Present, it is a claim about which build ran, which is a
   * thing a workflow needs to be able to make: "an agent answered" and "the agent we pinned answered"
   * are different statements, and only one of them is reproducible.
   */
  binaryPath?: string;
  /**
   * The environment the agent runs under. Absent ⇒ it inherits this process's.
   *
   * Forwarded verbatim and interpreted by nothing here. Multi-account isolation — a per-instance
   * `CLAUDE_CONFIG_DIR`, continuation-group keys — is a policy question for the caller that owns the
   * accounts; this is only the channel it would travel on.
   */
  env?: NodeJS.ProcessEnv;
  /** The agent's NATIVE permission profile, when the caller pins one. */
  permissionMode?: AgentPermissionMode;
  /**
   * The native permission mode a `read-only` session profile maps onto, for a transport where that
   * mapping is EXACT. Absent ⇒ unmapped, which is claude's truth: it has no read-only mode, and
   * borrowing `plan` would tell the agent to stop acting and start planning — a different
   * instruction, not a narrower one. Codex sets `"plan"`, because for it the word carries no
   * behaviour at all: `sandboxFor("plan")` is nothing but `--sandbox read-only`.
   */
  readOnlyProfileMode?: AgentPermissionMode;
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
   * The handle, with a REAL event stream behind it.
   *
   * The base returns `emptyEvents()`, which was honest for a provider call resolved in one await and
   * dishonest here: a delegated agent runs for minutes, narrating the whole way, and
   * `DELEGATED_CAPS.streaming: true` promised a caller could watch. So this wraps the inherited handle
   * with a queue the {@link AgentExecutor.invoke} phase pushes into — output deltas as the answer is
   * written, and the agent's own events forwarded opaquely.
   *
   * The queue is CLOSED when the result settles, however it settles. A stream left open after the
   * operation finished leaves a `for await` parked forever on a run that is already over.
   */
  override start(op: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, LlmMetrics> {
    const channel: AgentChannel = { events: new EventQueue(), pending: [] };
    const inner = super.start(op, { ...ctx, [EVENT_SINK]: channel } as ExecServices & WithEventSink) as ExecHandle<ResolvedValue, LlmMetrics>;
    const close = (): void => {
      channel.finished = true;
      // A request queued for a run that never started is dropped here rather than replayed at nothing —
      // a refused call has no turn to interrupt.
      channel.pending.length = 0;
      channel.events.close();
    };
    // `result` never rejects for a unit failure, but a wiring fault is not a unit failure — and a
    // stream that stays open because of one is a hang rather than an error.
    const result = inner.result.then(
      (r) => (close(), r),
      (e: unknown) => {
        close();
        throw e;
      },
    );
    /**
     * STEERING, and deliberately NOT cancellation.
     *
     * `interrupt` is forwarded to the transport's own interrupt and never to the abort controller.
     * `handles.ts` unifies `cancel()` and `ctx.abortSignal` into one event that settles the handle with
     * a `canceled` failure — and an interrupted agent turn is the opposite of that: the turn ends
     * early, the `result` message still arrives, and the call SUCCEEDS with the partial answer. Wiring
     * the two together would throw away an answer the agent actually produced, which is precisely what
     * a user pressing Stop on a long run does NOT want.
     *
     * Every method reads `channel.run` live and is a no-op once the run is over, so a control request
     * racing the stream's end is nothing rather than a second settle.
     */
    const steer = async (apply: (run: AgentRun) => Promise<void> | undefined): Promise<void> => {
      if (channel.finished) return;
      // Already running ⇒ straight through. Not yet ⇒ QUEUED, and this resolves now: the method is a
      // request, and blocking a caller's Stop until the transport happens to exist would be a worse
      // answer than acknowledging it.
      if (channel.run) await apply(channel.run);
      else channel.pending.push(async (run) => void (await apply(run)));
    };
    const control: ExecControl = {
      interrupt: () => steer((run) => run.interrupt?.()),
      send: (text) => steer((run) => run.send?.(text)),
      setPermissionMode: (mode) => steer((run) => run.setPermissionMode?.(mode as AgentPermissionMode)),
      setModel: (model) => steer((run) => run.setModel?.(model)),
    };
    return {
      events: channel.events.iterate(),
      result,
      // Present only when this transport actually steers. `capabilities.sessionSteering` is the
      // BEFORE-the-call answer; this is the runtime one, and the two must agree — a handle offering
      // `control` on a transport that declares no steering would be a stub that silently did nothing.
      ...(this.capabilities.sessionSteering === true ? { control } : {}),
      cancel: async () => {
        await inner.cancel();
        close();
      },
    };
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
      const turn = await this.runAgent(definition, ctx);
      const result = turn.result;
      // NOT charged here. A delegated agent spends real money inside its own loop, and this used to
      // reach for `ctx.meter` and debit it — an executor doing a wrapper's job, and a DOUBLE CHARGE
      // whenever the wrapper was actually composed: `withBudget` settles its reservation against
      // `result.metrics.costUsd`, which is the same money, so the wallet was hit twice for one call.
      //
      // The cost is reported on the measurement, which is where a metering layer reads it. A run with
      // no budget wrapper composed is unmetered — the same answer every other cross-cutting concern
      // gives when its wrapper is absent.
      // A run that was given a schema is ANSWERED by the constrained value, not by the prose beside it.
      // The two are both present and they are not the same thing: `result` summarizes the work, and a
      // caller handed that where it declared an object output gets a string that fills none of its
      // slots. A schema asked for and not produced is a FAILURE — falling back to the prose would hand
      // the engine an answer it must then reject, one layer further from the transport that knows why.
      if (definition.schema !== undefined && result.structured === undefined) {
        return {
          error: {
            classification: "api-retriable",
            reason: "the agent produced no structured output for a call that declared an output schema",
          },
          metrics: this.agentMetrics(startMs, result),
        };
      }
      // Stamped with ONE time for the whole call: the stream reports no per-message clock, and an
      // entry with no timestamp cannot be merged with a captured one later.
      const at = new Date().toISOString();
      const said = entriesOfMessages(turn.messages as RawMessage[], { provider: AGENT_PROVIDER, at });
      // The events, spliced back among the turns they arrived between.
      //
      // They were pinned by `index` — how many main turns preceded each — precisely so a reader could
      // interleave them, which meant every reader had to do the interleaving and could get it wrong.
      // Doing it HERE, once, is what lets them be entries rather than a second array: the pin becomes
      // the position, and `kind: "event"` is the format's own word for a session fact that is not a
      // message (RECORDS.md §2.2).
      const main: Entry[] = [];
      let pending = 0;
      for (let k = 0; k <= said.length; k++) {
        while (pending < turn.providerEvents.length && turn.providerEvents[pending]!.index === k) {
          const raw = turn.providerEvents[pending]!.event;
          const type = (raw as { type?: unknown } | null)?.type;
          main.push({
            kind: "event",
            provider: AGENT_PROVIDER,
            timestamp: at,
            // VERBATIM in `data`, opaque by the same rule the live stream's `provider_event` follows.
            // `type` is lifted out so a reader can name the row without interpreting the payload.
            event: { type: typeof type === "string" ? type : "provider_event", data: raw },
          });
          pending += 1;
        }
        if (k < said.length) main.push(said[k]!);
      }
      const entries: Entry[] = [
        ...main,
        // A subagent's turns carry the call that spawned them, so the main thread never comes to
        // claim it said what a subagent said — which is why they were a separate field before.
        ...[...turn.sidechains].flatMap(([parentToolUseId, messages]) =>
          entriesOfMessages(messages as RawMessage[], {
            provider: AGENT_PROVIDER,
            at,
            sidechain: { id: parentToolUseId, parentToolUseId },
          }),
        ),
      ];
      const output: LlmOutput = {
        // The terminal message's answer, falling back to what the turns themselves carried. An
        // INTERRUPTED run is the case: its terminal message has no text, and the partial answer exists
        // only in the assistant turns already streamed. Reporting an empty string there would answer
        // "stop and tell me what you found" with nothing.
        value: result.structured ?? (result.text.length > 0 ? result.text : turn.text),
        // The agent's OWN verdict on how its run ended. Hardcoding `"stop"` here reported a run that
        // exhausted its turn cap or its budget ceiling — a PARTIAL answer — as a clean finish, which is
        // how a truncated review reads as a complete one.
        finishReason: result.finishReason ?? "unknown",
        // The agent's own log, verbatim. It used to be one synthesized assistant turn carrying the
        // final text, under a comment claiming a delegated agent hands back nothing else. It does hand
        // it back — every turn, on the same wire — and we were discarding it.
        // ONE array. `thinking`/`toolCalls`/`toolResults` were projections OF these messages and
        // are computed from the entries now; `sidechains` was the same conversation under a second
        // key space; `providerEvents` was the rest of it under a third, pinned by an index every
        // reader had to re-splice. Measured on one record, the two tool indexes alone were 178 KB.
        ...(entries.length > 0 ? { entries } : {}),
        ...(result.sessionId !== undefined ? { providerSessionId: result.sessionId } : {}),
      };
      return { value: output, metrics: this.agentMetrics(startMs, result) };
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

  /** The per-call event sink, when this executor's own {@link AgentExecutor.start} supplied one. Absent
   *  for a subclass or test that drove `invoke` directly, in which case events go nowhere — which is
   *  the same thing that happened before there was a stream at all. */
  private sink(ctx: ExecServices): AgentChannel | undefined {
    return (ctx as ExecServices & WithEventSink)[EVENT_SINK];
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
  protected agentMetrics(startMs: number, result: AgentResult | undefined): LlmMetrics {
    const costUsd = result?.costUsd;
    return {
      startMs,
      durationMs: Date.now() - startMs,
      childLlmCalls: 1,
      costUsd: costUsd ?? 0,
      costSource: costUsd !== undefined ? "provider" : "unknown",
      ...(costUsd !== undefined ? { childCostUsd: costUsd } : {}),
      // What the run CONSUMED, split the way it is billed — a cache read costs a tenth of the base
      // rate and a 1-hour cache write roughly twice it, so a single input figure cannot be priced.
      // These used to be absent entirely, which made a delegated call the one kind of call whose spend
      // could not be checked against anything.
      ...(result?.usage ?? {}),
      // The provider's exact object, so `costUsd` stays recomputable if our reading of the fields is
      // wrong or incomplete — the same reason the provider path keeps it.
      ...(result?.rawUsage !== undefined ? { rawUsage: result.rawUsage } : {}),
    } as LlmMetrics;
  }

  /** Build the query options from the lowered call, run the agent, and accumulate everything its
   *  stream produced. */
  private async runAgent(definition: LlmCallDefinition, ctx: ExecServices): Promise<AgentTurn> {
    // REFUSE BEFORE CONFIGURING. `resolveConfig`, `defaults` and `op.config` all merge into this
    // declaration, so a field that reached here was ASKED FOR — and reading three of them off it while
    // dropping the rest is how a call that paid for a reasoning budget quietly got none.
    const refusal = this.agentRefusal(definition);
    if (refusal !== undefined) throw new AgentError(`${this.label()}: ${refusal}`);

    /**
     * A NARROWING profile on a transport that enforces nothing is refused, not run.
     *
     * The profile is the one restriction an author states about a whole delegated run ("this state
     * must not write"), and `policyEnforcement: "none"` is this executor's own declaration that no
     * channel exists to hold the agent to it — no callback, no deny flag, no sandbox. Running anyway
     * would be the exact failure the capability record exists to prevent: the workflow reads as
     * restricted while the agent runs under nothing but its own defaults.
     */
    const profile = ctx.gate?.profile;
    if (profile !== undefined && profile !== "full" && this.capabilities.policyEnforcement === "none") {
      throw new AgentError(
        `${this.label()}: this state runs under the '${profile}' permission profile, and this transport ` +
          `enforces no policy (policyEnforcement: "none") — nothing could hold the agent to it. ` +
          `Run the state on claude-code, claude-cli or codex-cli, or drop the profile`,
      );
    }

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

    /**
     * The resolved mode for one tool, WITHOUT its input — what up-front configuration may assume.
     *
     * `ctx.gate` is the real answer: the session profile, THIS state's authored block, the run's
     * ledger overlays and the workflow-wide baseline, resolved by the one implementation that knows
     * their precedence. Reading `ctx.policy.baseline.tools` directly — as this used to — saw the last
     * of those and nothing else, so a mode authored on the state was invisible here. That is most of
     * what a caller actually sets.
     *
     * The fallback keeps a host that publishes no gate working exactly as before.
     */
    const modeOf = (name: string, readOnly?: boolean): PermissionMode | undefined =>
      ctx.gate?.modeOf({ name, ...(readOnly !== undefined ? { readOnly } : {}) }) ??
      (Object.hasOwn(ctx.policy?.baseline?.tools ?? {}, name) ? ctx.policy?.baseline?.tools?.[name] : undefined);

    // A per-tool `deny` needs no human, so it must reach the agent as CONFIGURATION rather than
    // waiting for an approval that will never be asked for. Native names are what the agent
    // addresses, so an aliased tool is denied under its `native` name.
    const denied = Object.keys(ctx.policy?.baseline?.tools ?? {})
      .filter((name) => modeOf(name) === "deny")
      .map((name) => nativeMap[name]?.native ?? name);
    const denySet = new Set(denied);

    // Resolve each logical tool to NATIVE (the agent's built-in, aliased) or MCP-INJECTED (our impl,
    // ctx-bound). The engine hands a delegated runtime RAW tools, and authorization flows through
    // `canUseTool` → `ctx.approve`, so injected tools are not double-gated.
    const tools = (ctx.tools ?? this.options.tools) as Record<string, Tool> | undefined;
    const extraTools = this.agent.extraTools;
    const bind = (tool: Tool): InjectedTool => ({
      description: tool.description,
      inputSchema: tool.inputSchema as JsonSchema,
      run: (input) => tool.run(input, ctx),
    });
    let allowedTools: string[] | undefined;
    let mcpTools: Record<string, InjectedTool> | undefined;
    const native: string[] = [];
    const injected: Record<string, InjectedTool> = {};
    /** Tools resolving to `allow` — the only ones it is honest to pre-approve. See below. */
    const preApproved: string[] = [];
    if (tools) {
      for (const [name, tool] of Object.entries(tools)) {
        const ref = nativeMap[name];
        const mode = modeOf(name, tool.readOnly);
        // A `deny` is an unconditional floor: the tool is never OFFERED, native or injected. An injected
        // tool is addressed as `mcp__dai__<name>`, which no logical-name deny entry matches, so leaving
        // it injected would route around the floor entirely — drop it here.
        //
        // `modeOf` rather than the deny SET, because the set is built from the workflow-wide baseline
        // and this tool's `deny` may have been authored on the state — which is where a caller writes
        // one. It also covers a tool the PROFILE excludes: under `read-only`, a writer resolves to
        // `deny` without anybody naming it.
        if (denySet.has(ref ? ref.native : name) || mode === "deny") continue;
        if (!inject || ref) native.push(ref ? ref.native : name);
        else injected[name] = bind(tool);
        // ONLY an explicit `allow`. This list is a PRE-APPROVAL — a tool named here is never put to
        // the permission callback — so carrying every tool regardless of mode, as it used to, meant an
        // authored `ask` never asked and a `smart` policy never ran. `smart` is the sharper of the two:
        // it INSPECTS the input, so pre-approving it decides the call before the thing that decides it
        // has run.
        if (mode === "allow") preApproved.push(ref ? ref.native : name);
      }
      // ✅ An EMPTY list means "pre-approve nothing", NOT "allow nothing" — checked on a live run, where
      // `--allowedTools ""` still let the agent use its native `Read`. So this and `cliArgv`'s
      // omit-when-empty are the same request, and neither silently disarms the agent. Which is what
      // makes narrowing this safe: what is not pre-approved is ASKED about, not refused.
      //
      // With no gate there are no modes to read, so the prior behaviour stands: pre-approve what was
      // declared. A host that publishes a gate gets the mode honoured.
      allowedTools = ctx.gate !== undefined ? preApproved : native;
    }
    // EXTRA tools ride on top, whatever `injectTools` decided about `ctx.tools`. This is the "natives
    // plus extras" case: a host exposing its own capability to an otherwise stock agent, which the one
    // switch could not express. The deny floor still applies — an extra tool is a tool.
    for (const [name, tool] of Object.entries(extraTools ?? {})) {
      if (denySet.has(nativeMap[name]?.native ?? name)) continue;
      injected[name] = bind(tool);
    }
    if (Object.keys(injected).length > 0) mcpTools = injected;

    /**
     * The tool a permission callback is asking about, named the way the POLICY is written.
     *
     * Three vocabularies meet at that callback and only one of them is the policy's. The agent
     * addresses an injected tool as `mcp__dai__read_file` and an aliased one by its built-in name
     * (`Read`), while an authored mode is written against the LOGICAL name (`read_file`) — so asking
     * the gate about the string the agent used would miss the mode every time an alias or an
     * injection was in play, and silently fall through to the default.
     *
     * A name that is neither — the agent's own `Bash`, which we never registered — passes through
     * with no `readOnly`, which is exactly the unclassifiable case `ToolGate.modeOf` documents.
     */
    const mcpPrefix = mcpToolName("");
    const logicalByNative = new Map(Object.entries(nativeMap).map(([logical, ref]) => [ref.native, logical]));
    const gateSubject = (addressed: string): { name: string; readOnly?: boolean } => {
      const bare = addressed.startsWith(mcpPrefix) ? addressed.slice(mcpPrefix.length) : addressed;
      const logical = logicalByNative.get(bare) ?? bare;
      const known = tools?.[logical];
      return { name: logical, ...(known !== undefined ? { readOnly: known.readOnly } : {}) };
    };

    // DISPLACE what an injected tool stands in for. Without this, injection adds a second set of tools
    // the model ignores: a live run with `read_file` injected and `Read` still available used `Read`
    // every time. Denying the built-in is what makes the substitution real — and it is denied under the
    // agent's own name, since that is what it addresses.
    //
    // A built-in that some OTHER logical tool is aliased to (`nativeTools`) is exempt: the caller asked
    // for that one natively, and disallowing it would refuse a tool it just requested.
    const aliased = new Set(Object.values(nativeMap).map((ref) => ref.native));
    for (const name of Object.keys(injected)) {
      const replaced = this.agent.replacesNative?.[name];
      for (const builtin of replaced === undefined ? [] : typeof replaced === "string" ? [replaced] : replaced) {
        if (!aliased.has(builtin) && !denySet.has(builtin)) {
          denied.push(builtin);
          denySet.add(builtin);
        }
      }
    }

    // A `read-only` profile reaches the agent as CONFIGURATION, not only as a callback. The gate
    // already refuses what it can classify and escalates what it cannot — but escalation is a human
    // question, and "may this agent run `Bash`?" under a profile that means "no writes" is not one:
    // the write-capable built-ins are a known fact about the transport, so they are denied up-front
    // (see {@link AgentExecutorOptions.mutatingNativeTools}). NOT applied under `plan`, which has an
    // exact native counterpart (`--permission-mode plan`, set below) that still allows read-only use
    // of these same tools; and not under a custom profile, whose predicate this executor cannot read
    // — there the callback remains the whole answer. An alias is not exempt: the profile is the
    // narrower statement, so a declared `bash` aliased to `Bash` is still denied under `read-only`.
    if (profile === "read-only") {
      for (const builtin of this.agent.mutatingNativeTools ?? CLAUDE_MUTATING_BUILTINS) {
        if (!denySet.has(builtin)) {
          denied.push(builtin);
          denySet.add(builtin);
        }
      }
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
      // WHICH binary, and under what environment. Both are transport-level facts the caller pinned at
      // construction, so they travel on every call this executor makes rather than being re-stated.
      ...(this.agent.binaryPath !== undefined ? { binaryPath: this.agent.binaryPath } : {}),
      ...(this.agent.env !== undefined ? { env: this.agent.env } : {}),
      allowedTools,
      ...(denied.length > 0 ? { disallowedTools: denied } : {}),
      mcpTools,
      // The injected-tool input gate is sync (`seam.ts`); the ctx seam is maybe-async — narrow
      // FAIL-CLOSED (json's `syncOnly`) rather than let an async validator read as a pass.
      ...(ctx.validator !== undefined ? { validator: syncOnly(ctx.validator) } : {}),
      permissionMode: this.permissionMode(ctx),
      // THE NEUTRAL KNOBS a delegated transport can actually carry. This used to read three fields off
      // the lowered declaration and drop everything else — so a state authored with `reasoning: {effort:
      // "xhigh"}` and a step budget ran at the agent's own defaults, silently, and cost what the deeper
      // run would have cost only if you were unlucky.
      ...("reasoning" in definition && definition.reasoning !== undefined ? { reasoning: definition.reasoning } : {}),
      ...(definition.maxSteps !== undefined ? { maxSteps: definition.maxSteps } : {}),
      ...(definition.toolChoice === "none" || definition.toolChoice === "auto" ? { toolChoice: definition.toolChoice } : {}),
      // The output SCHEMA, which is what makes a delegated agent able to serve a state that declares
      // outputs. It was read off the lowered declaration by every other transport and dropped by this
      // one: the agent answered in prose, the engine found none of the declared slots, and the run
      // failed with "did not produce required output" — naming the state rather than the omission.
      ...(definition.schema !== undefined ? { schema: definition.schema as JsonValue } : {}),
      // Only THIS transport's bag: `providerOptions` is keyed by provider so one config can carry
      // settings for several, and each takes only its own.
      ...(definition.providerOptions?.[this.providerOptionsKey()] !== undefined
        ? { providerOptions: definition.providerOptions[this.providerOptionsKey()]! }
        : {}),
      // FORK exactly when the handle we were given is the fork source, not the append target — the two
      // travel on different fields (`ResolvedSession.forkFrom`), so this reads the difference rather
      // than inferring it from a mode that an automatic branch does not set.
      ...(resume !== undefined
        ? {
            resume,
            ...(resume === session?.forkFrom?.handle
              ? {
                  forkSession: true,
                  // WHERE to cut the copy, when the branch does not start at the remote's tip. The
                  // CLI takes it as `--resume-session-at`; without it a fork copies the session as it
                  // now stands, which for a branch cut behind that is turns it never had.
                  ...(session.forkFrom.at !== undefined ? { resumeSessionAt: session.forkFrom.at } : {}),
                }
              : {}),
          }
        : {}),
      ...(replayed !== undefined ? { messages: replayed as never } : {}),
      /**
       * Route the agent's native tool-approval callback through our GATE (DESIGN §5.1, "Delegated
       * approval fidelity").
       *
       * The gate, not the approver. `approve` is only the LAST of the four things a permission
       * decision does — and calling it directly, as this used to, silently collapsed the other three:
       *
       *  - `smart` never ran its policy. Its whole point is to inspect the call and decide without a
       *    human, and every `smart` tool was put to one instead.
       *  - `allow` asked anyway, so a mode whose meaning is "do not interrupt me" interrupted.
       *  - the session PROFILE was never consulted, so a `read-only` state could be talked into a
       *    write by one distracted click.
       *
       * None of it failed loudly: the modes were configured, shown in the UI, and ignored.
       *
       * `approve` remains the fallback for a host that publishes no gate, and it is what the gate
       * itself escalates to when a decision really does need a human.
       */
      canUseTool:
        (ctx.gate ?? approve) && wantsApprovalCallback
          ? async (req) => {
              // A QUESTION, not a permission — picked off before the gate (see {@link ASK_USER_TOOL}).
              // The human's answers ride back on the allow; a dismissal or an unattended run answers
              // "use your own judgment", which the agent handles by proceeding. Only a run with no
              // question surface at all falls through to that deny.
              if (req.toolName === ASK_USER_TOOL) {
                const questions = questionsOf(req.input);
                let answers: UserAnswers | undefined;
                if (ctx.askUser !== undefined && questions.length > 0) {
                  answers = await ctx.askUser({ questions, sessionId: scope });
                }
                return answers !== undefined
                  ? { allow: true, updatedInput: { ...req.input, answers: answers as never } }
                  : { allow: false, reason: "the user is not available to answer — use your own best judgment and continue" };
              }
              if (ctx.gate) {
                const verdict = await ctx.gate.check(gateSubject(req.toolName), req.input);
                return verdict.allow ? { allow: true } : { allow: false, reason: verdict.reason };
              }
              const decision = await approve!({ tool: req.toolName, input: req.input, sessionId: scope });
              return decision.decision === "allow" ? { allow: true } : { allow: false, reason: "denied by permission policy" };
            }
          : undefined,
      abortSignal: signal,
    };

    // THE ACCUMULATION. Everything below used to be `if (msg.type === "result") result = msg.result`,
    // with every other message discarded — which is where the trace, the tool log, the token counts and
    // the conversation went. None of it was missing at the source.
    const channel = this.sink(ctx);
    const events = channel?.events;
    const turn: AgentTurn = { result: { text: "" }, text: "", messages: [], thinking: [], toolCalls: [], toolResults: [], sidechains: new Map(), providerEvents: [] };
    let result: AgentResult | undefined;
    // The last machine-readable failure code the agent gave. It arrives on the ASSISTANT turn that
    // carries the failure text, one message BEFORE the terminal result that repeats the prose — so it
    // has to be remembered to still be in hand when the run is failed below.
    let errorCode: string | undefined;
    // The LIVE run, published to the handle before a single message is read — a caller pressing Stop
    // one tick into a five-minute turn must reach something.
    const run = this.query()(queryOptions);
    if (channel) {
      channel.run = run;
      // Replay whatever was asked for while the run was still being built — a Stop pressed one tick
      // into a five-minute turn lands in that window, and dropping it would answer by letting the run
      // finish. Fire-and-forget: a control request is a request, and the stream below is what reports
      // what came of it.
      for (const queued of channel.pending.splice(0)) void queued(run);
    }
    try {
      for await (const msg of run) {
        if (msg.errorCode !== undefined) errorCode = msg.errorCode;
        if (msg.error) throw new AgentError(`${this.label()} agent error: ${msg.error}`, false, msg.errorCode ?? errorCode);
        switch (msg.type) {
          case "result":
            if (msg.result) result = msg.result;
            break;
          case "partial":
            // The answer as it is written. This is what makes the declared `streaming` capability true.
            if (msg.delta !== undefined && msg.delta.length > 0) events?.push({ type: "output_partial", text: msg.delta });
            break;
          case "thinking-partial":
            // The reasoning as it happens, on its own channel — never folded into the answer.
            if (msg.delta !== undefined && msg.delta.length > 0) events?.push({ type: "thinking_partial", text: msg.delta });
            break;
          case "provider_event":
            // Forwarded OPAQUELY. `exec` must not learn this agent's vocabulary, and a host that wants
            // to render a compaction boundary must not be stopped because we had no neutral name for it.
            if (msg.event !== undefined) {
              events?.push({ type: "provider_event", payload: msg.event });
              // …and RECORDED, pinned to its place among the turns. The live view already showed it;
              // a replay that shows less than the person watching saw is a record telling a smaller
              // story than the run. Except delta bookkeeping: a `stream_event` is a fragment whose
              // content arrives again on the finished turn, and hundreds of them per turn would
              // swamp the record with what it already holds.
              if ((msg.event as { type?: unknown }).type !== "stream_event") {
                turn.providerEvents.push({ index: turn.messages.length, event: msg.event });
              }
            }
            break;
          case "assistant":
          case "user":
            // The whole turn, live — the contract's declared `message` variant, which nothing emitted
            // until now. Deltas carry only the answer's text; the tool calls, their results and the
            // thinking blocks all ride on the finished turn objects, so a viewer that gets no
            // `message` events learns about an agent's tools only when the record closes — for a
            // run that takes an hour, that is indistinguishable from an agent doing nothing.
            if (msg.message !== undefined) {
              events?.push({
                type: "message",
                role: msg.type,
                content: msg.message as JsonValue,
                ...(msg.parentToolUseId !== undefined ? { parentToolUseId: msg.parentToolUseId } : {}),
              });
            }
            break;
          default:
            break;
        }
        // A SUBAGENT's turn goes to its own chain and nowhere else. Its text is not the answer, its
        // thinking is not the main thread's reasoning, its tool calls are not the agent's own — and
        // before the tag existed they were all folded in, which made the record claim the main
        // thread said things a subagent said.
        if (msg.parentToolUseId !== undefined) {
          if (msg.message !== undefined) {
            const chain = turn.sidechains.get(msg.parentToolUseId);
            if (chain === undefined) turn.sidechains.set(msg.parentToolUseId, [msg.message as unknown as ModelMessage]);
            else chain.push(msg.message as unknown as ModelMessage);
          }
          continue;
        }
        // The provider's own turn objects, kept whole. `LlmOutput.messages` is documented as the
        // provider's log rather than a reconstruction; an Anthropic message and a `ModelMessage` are
        // both `{role, content}` with provider-shaped parts, so this is a cast and not a translation —
        // and a translation is precisely what would break the signed thinking blocks inside it.
        // The SIGNATURE arrives beside the message rather than inside it, and Anthropic requires it
        // back byte-identical. Stamping it onto the thinking block here — the one place the two are
        // known to belong together — is what keeps the entry lossless; correlating them later, off a
        // record that had already dropped one side, is the reconstruction this format exists to stop.
        if (msg.message !== undefined) {
          signThinking(msg.message as unknown as { content?: unknown }, msg.thinking);
          turn.messages.push(msg.message as unknown as ModelMessage);
        }
        for (const block of msg.thinking ?? []) {
          turn.thinking.push({
            type: "reasoning",
            text: block.text,
            // Positioned against the OUTPUT text accumulated so far — `ReasoningSegment` is placed
            // against the answer, which is what lets a consumer show thinking where it happened rather
            // than in a heap at the end.
            textOffset: turn.text.length,
            ...(block.providerMetadata !== undefined ? { providerMetadata: block.providerMetadata } : {}),
          });
        }
        for (const call of msg.toolCalls ?? []) turn.toolCalls.push(call as ToolCall);
        for (const toolResult of msg.toolResults ?? []) turn.toolResults.push(toolResult as ToolResult);
        // Accumulated from the finished turns rather than from the deltas: a `partial` and the
        // `assistant` message that follows it carry the SAME text, so adding both would double it.
        if (msg.type === "assistant") turn.text += msg.text ?? "";
      }
    } catch (e) {
      if (signal.aborted) throw new AgentError("aborted", true);
      if (e instanceof AgentError) throw e;
      throw new AgentError(`${this.label()} query threw: ${(e as Error).message}`);
    }
    if (signal.aborted) throw new AgentError("aborted", true);
    if (!result) throw new AgentError(`${this.label()} produced no result message`);
    turn.result = result;
    return turn;
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

  /**
   * The `providerOptions` key whose bag reaches THIS transport.
   *
   * `providerOptions` is keyed by provider precisely so one config can carry settings for several and
   * each takes only its own. The codex sibling overrides this with `"codex"`; nothing else about the
   * forwarding differs, which is why it is one string rather than a second code path.
   */
  protected providerOptionsKey(): string {
    return "claudeCode";
  }

  /**
   * What this call asks for that NO delegated transport can honour — the reason to refuse, or
   * `undefined` to proceed.
   *
   * The rule the seam already states for `model` and `disallowedTools`, applied to the rest of the
   * declaration: a transport may lack a capability, but it may never pretend to have one. These are the
   * knobs that are meaningless to an agent rather than merely unimplemented — a decoding parameter has
   * no channel because the agent, not us, issues the model calls.
   *
   * Per-transport gaps are refused by the transport (see `codexRefusal`, `cliRefusal`), not here: this
   * is the floor they share.
   */
  protected agentRefusal(definition: LlmCallDefinition): string | undefined {
    const sampling = definition as unknown as Record<string, unknown>;
    const knobs = ["temperature", "topP", "topK", "presencePenalty", "frequencyPenalty", "seed"].filter((k) => sampling[k] !== undefined);
    if (knobs.length > 0) {
      return (
        `a delegated agent issues its own model calls, so the decoding knobs [${knobs.join(", ")}] never reach one. ` +
        `Express how hard it should think as \`reasoning\`, or run this state on a provider transport`
      );
    }
    if (definition.maxOutputTokens !== undefined) {
      return (
        "a delegated agent runs a multi-turn loop with no per-response token cap, so `maxOutputTokens` cannot be honoured. " +
        "Bound the run with `maxSteps`, or with `providerOptions.claudeCode.maxBudgetUsd`"
      );
    }
    if (definition.stopSequences !== undefined) {
      return "a delegated agent has no stop-sequence channel — `stopSequences` cannot be honoured on this transport";
    }
    if (definition.outputModalities !== undefined) {
      return "a delegated agent answers in text; `outputModalities` cannot be honoured on this transport";
    }
    const choice = definition.toolChoice;
    if (choice !== undefined && choice !== "auto" && choice !== "none") {
      const named = typeof choice === "object" ? ` (${choice.toolName})` : "";
      return (
        `\`toolChoice: ${typeof choice === "object" ? "{ type: 'tool' }" : choice}\`${named} constrains ONE model turn, and a delegated agent runs a whole loop — ` +
        `it cannot be honoured. Use "auto" or "none"`
      );
    }
    return undefined;
  }

  /**
   * The permission mode this run is configured with — the adapter's own, else the session PROFILE's.
   *
   * The fallback exists for the transport that has nothing else. `codex exec` has no per-tool gate:
   * its whole enforcement is the up-front `--sandbox`, chosen from this. So without a profile arm a
   * state authoring `plan` ran codex with whatever sandbox the adapter happened to be constructed
   * with — the profile said "must not write" and nothing carried it to the one place that could act.
   *
   * ONLY `plan` maps unconditionally, and deliberately. It is the one exact correspondence between
   * the two vocabularies (a planning turn must not write, which is what `--sandbox read-only` and
   * claude's `--permission-mode plan` both mean). `read-only` maps only where a transport DECLARED
   * the mapping exact ({@link AgentExecutorOptions.readOnlyProfileMode} — codex, whose `plan` is
   * nothing but its sandbox): claude has no read-only mode, and borrowing `plan` for it would tell
   * the agent to stop acting and start planning, which is a different instruction from "you may
   * read". For claude that profile is enforced by the up-front deny of its write-capable built-ins
   * ({@link AgentExecutorOptions.mutatingNativeTools}) plus the gate's per-call answer.
   *
   * An explicitly configured mode always wins: it is the more specific statement.
   */
  protected permissionMode(ctx?: ExecServices): AgentPermissionMode | undefined {
    const m = this.agent.permissionMode;
    if (m !== undefined && PERMISSION_MODES.includes(m)) return m;
    if (ctx?.gate?.profile === "plan") return "plan";
    // Only where a transport declared the mapping exact — see {@link AgentExecutorOptions.readOnlyProfileMode}.
    if (ctx?.gate?.profile === "read-only") return this.agent.readOnlyProfileMode;
    return undefined;
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

/**
 * Copy a streamed thinking block's SIGNATURE onto the message block it belongs to, in place.
 *
 * The transport reports the two on one event but in two places: `message.content` holds the text and
 * `thinking[]` holds the provider metadata. Anthropic requires the signature back byte-identical, so
 * a record keeping only the message would replay a reasoning block the provider then refuses.
 *
 * Paired by ORDER within the event — the arrays are the provider's own, emitted together, and there
 * is no id on either side to join. A mismatch drops the signature rather than guessing: an unsigned
 * block is a block that cannot be replayed, which is visible, where a wrongly-signed one is not.
 */
function signThinking(message: { content?: unknown }, thinking: readonly { providerMetadata?: unknown }[] | undefined): void {
  if (thinking === undefined || thinking.length === 0 || !Array.isArray(message.content)) return;
  let seen = 0;
  for (const block of message.content as Array<Record<string, unknown>>) {
    if (block === null || typeof block !== "object" || block["type"] !== "thinking") continue;
    const meta = thinking[seen++]?.providerMetadata as Record<string, { signature?: unknown }> | undefined;
    const signature = meta === undefined ? undefined : Object.values(meta).find((v) => typeof v?.signature === "string")?.signature;
    if (typeof signature === "string" && block["signature"] === undefined) block["signature"] = signature;
  }
}
