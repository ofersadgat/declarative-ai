/**
 * The SDK-driven delegated agent as a `runtime` REGISTRY ENTRY (DESIGN §4.4).
 *
 * There is no `Runtime` interface and no normalized runtime-op payload: a runtime invocation is a
 * **plain `FunctionOp`** naming a registered function, so this factory produces a registry entry
 * carrying the delegated-agent capabilities (`mutatesWorkspace`, `memoizable: false`,
 * `policyEnforcement: "callback"`) — required and total, per §2, so the permission gate reads a
 * definite value instead of falling through an `undefined`. Permission gating and search refusal read
 * that resolved entry; the op shape carries no runtime marker at all.
 *
 * What changed is where the WORK lives. This file used to hold the whole adapter: the tool split, the
 * deny floor, the approval bridge, and its own copy of the session request-shaping — a copy that could
 * not be shared with `promptop`'s, so the fork rules had to be maintained in two vocabularies. All of
 * that now belongs to {@link AgentExecutor}, and this is the ADAPTER that presents it as a function
 * entry: one shape, two ways in.
 *
 * Keeping the entry is not backwards-compatibility theatre. It is what `operation: {kind: "function",
 * function: "claude-code"}` resolves to in every existing workflow, and it is the subject DESIGN §5.1's
 * delegated-approval rule needs — the engine decides whether to wrap tools by reading a registry
 * ENTRY's capabilities. A prompt op reaches the same executor directly; a function op reaches it
 * through here.
 *
 * Register it as `registry.functions.registerRuntime("claude-code", fn.run, fn.capabilities)`, then
 * author a call with the `runtimeOp` builder — which lowers to exactly `{ kind: "function", functionRef:
 * "claude-code", input: { prompt, config } }`.
 */
import {
  isOk,
  promptOp,
  type FunctionResult,
  type ExecServices,
  type FunctionInputs,
  type JsonValue,
  type NativeToolRef,
  type RuntimeCapabilities,
  type Tool,
} from "@declarative-ai/exec";
import type { LlmOutput } from "@declarative-ai/llm";
import { AgentExecutor, DELEGATED_CAPS, type AgentExecutorOptions, type AgentMetrics } from "./agentExecutor.js";
import type { AgentPermissionMode, AgentQuery, AgentSessionReader } from "./seam.js";

export { DELEGATED_CAPS, debitSpentCost, type AgentMetrics } from "./agentExecutor.js";

const PERMISSION_MODES: readonly AgentPermissionMode[] = ["default", "plan", "acceptEdits", "bypassPermissions"];

/** The authored runtime surface, bound as the op's `config` input (§3.1) — never part of the op shape. */
export interface ClaudeCodeConfig {
  /** The agent's NATIVE permission profile control. */
  permissionMode?: AgentPermissionMode;
  /** Approval scope key for `ctx.approve` (defaults to `"delegated"`). */
  sessionId?: string;
}

export interface ClaudeCodeFunctionOptions {
  /** The agent-query seam. Default: `sdkAgentQuery` (lazily loads `@anthropic-ai/claude-agent-sdk`). */
  query?: AgentQuery;
  /** Override the advertised entry capabilities (e.g. a variant that is workspace-read-only). */
  capabilities?: RuntimeCapabilities;
  /** Inject `ctx.tools` into the agent over MCP so it calls OUR impls (identical behavior to a prompt op
   *  with executable tools). Default `true`. Set `false` to instead pass every tool name as a NATIVE
   *  allow-list (the agent uses its own built-ins by that name). */
  injectTools?: boolean;
  /** Tools to ADD to whatever the agent already has, always injected, whatever `injectTools` says —
   *  the "natives plus extras" case a single switch could not express. See
   *  {@link AgentExecutorOptions.extraTools}. */
  extraTools?: Record<string, Tool>;
  /** Per-logical-name overrides (DESIGN §5.1, "Tool renames are just overlay bindings"): a tool listed here resolves to the agent's
   *  NATIVE built-in `ref.native` (aliased) instead of being MCP-injected — so a run can use the agent's own
   *  `Read` for `read_file` while still injecting our `bash`. Ignored tools default to injection. */
  nativeTools?: Record<string, NativeToolRef>;
  /** Which of the agent's OWN built-ins an injected tool DISPLACES. Without it, injection adds a second
   *  set of tools the model ignores — see {@link AgentExecutorOptions.replacesNative}. */
  replacesNative?: Record<string, string | readonly string[]>;
  /**
   * Route the agent's tool approvals to `ctx.approve`. Default `true`.
   *
   * `false` states that this transport HAS no mid-run approval channel — codex is the case: `codex
   * exec` offers nothing like `--permission-prompt-tool`. Making it an option rather than letting the
   * query silently ignore an approver is the point: an adapter constructed this way declares
   * `policyEnforcement: "config"`, and the engine answers that by policy-WRAPPING its injected tools
   * instead of handing them over raw — so the gate moves, rather than disappearing.
   *
   * Setting this `false` on a transport that CAN ask is a safety regression, not a tidy-up: the agent
   * would then reach its native tools under nothing but the up-front posture.
   */
  approvalCallback?: boolean;
  /**
   * Reads a provider-side conversation back, for re-syncing after divergence (DESIGN.md §1.6).
   *
   * Injected rather than reached for directly, exactly as {@link ClaudeCodeFunctionOptions.query} is:
   * it is a second call into the SDK, and a test has to stand in for it without a provider. Absent ⇒
   * this adapter offers no read capability, and a resync starts empty.
   */
  readSession?: AgentSessionReader;
  /** WHICH binary answers — an absolute path, or a name the transport resolves. Absent ⇒ the
   *  transport's own default. See {@link AgentExecutorOptions.binaryPath}. */
  binaryPath?: string;
  /** The environment the agent runs under. Absent ⇒ it inherits this process's. Forwarded verbatim. */
  env?: NodeJS.ProcessEnv;
  /** What this transport is CALLED in a failure reason. Defaults to `claude-code`, which is what this
   *  factory drives; a CLI sibling passes its own binary's name so a failure says which one produced it. */
  label?: string;
}

/** Thrown when the delegated agent fails or is canceled. The invoking executor classifies it (a
 *  cancellation carries `name: "AbortError"`, which `classifyError` maps to `canceled`). */
export class ClaudeCodeError extends Error {
  constructor(message: string, readonly canceled = false) {
    super(message);
    this.name = canceled ? "AbortError" : "ClaudeCodeError";
  }
}

/** Read an author-supplied permission mode from the bound `config` input, ignoring an unknown value. */
function permissionModeOf(config: Record<string, JsonValue>): AgentPermissionMode | undefined {
  const m = config["permissionMode"];
  return typeof m === "string" && (PERMISSION_MODES as readonly string[]).includes(m) ? (m as AgentPermissionMode) : undefined;
}

/** Read the op's bound `config` input as a plain record (absent/non-object/bytes ⇒ empty). */
function configOf(inputs: FunctionInputs): Record<string, JsonValue> {
  const c = inputs.config;
  return c !== null && typeof c === "object" && !Array.isArray(c) && !(c instanceof Uint8Array) && !("getReader" in c)
    ? (c as Record<string, JsonValue>)
    : {};
}

/**
 * Present the agent executor as a `runtime` registry entry.
 *
 * `build` is the seam a CLI-driven sibling overrides: everything about mapping a `FunctionOp`'s
 * inputs onto an executor is identical across transports, and only which executor gets built differs.
 */
export function agentRuntimeEntry(
  build: (perCall: AgentExecutorOptions) => AgentExecutor,
  options: ClaudeCodeFunctionOptions = {},
): {
  capabilities: RuntimeCapabilities;
  run: (inputs: FunctionInputs, ctx: ExecServices) => Promise<FunctionResult<string, AgentMetrics>>;
  /** The provider read seam a host wires into `ctx.sessionReader`, when this transport has one. */
  sessionReader?: { read(providerSessionId: string): Promise<readonly unknown[]> };
} {
  const readSession = options.readSession;
  // Built once purely to read the capability record the per-call executors will carry, so the ENTRY
  // and the EXECUTOR cannot disagree about what this transport enforces (DESIGN §3.2).
  const capabilities = options.capabilities ?? build({}).capabilities;
  return {
    capabilities: capabilities as RuntimeCapabilities,
    // Present only when this transport can actually read a conversation back. The distinction is
    // load-bearing: an absent reader means a resync starts EMPTY, and §11 requires that to be visible
    // rather than mistaken for a conversation that happened to have nothing in it.
    ...(readSession !== undefined ? { sessionReader: { read: (id: string) => readSession(id) } } : {}),
    run: async (inputs: FunctionInputs, ctx: ExecServices): Promise<FunctionResult<string, AgentMetrics>> => {
      const config = configOf(inputs);
      const prompt = typeof inputs.prompt === "string" ? inputs.prompt : String(inputs.prompt ?? "");
      // A fresh executor per call, because `permissionMode` and the approval scope are per-CALL facts
      // that arrive on the op's `config` input while the executor takes them at construction. It is an
      // options bag and a closure — no I/O, no connection, nothing worth pooling.
      const executor = build({
        ...(options.query !== undefined ? { query: options.query } : {}),
        ...(options.capabilities !== undefined ? { capabilities: options.capabilities } : {}),
        ...(options.injectTools !== undefined ? { injectTools: options.injectTools } : {}),
        ...(options.extraTools !== undefined ? { extraTools: options.extraTools } : {}),
        ...(options.nativeTools !== undefined ? { nativeTools: options.nativeTools } : {}),
        ...(options.replacesNative !== undefined ? { replacesNative: options.replacesNative } : {}),
        ...(options.approvalCallback !== undefined ? { approvalCallback: options.approvalCallback } : {}),
        ...(options.readSession !== undefined ? { readSession: options.readSession } : {}),
        ...(options.binaryPath !== undefined ? { binaryPath: options.binaryPath } : {}),
        ...(options.env !== undefined ? { env: options.env } : {}),
        ...(options.label !== undefined ? { label: options.label } : {}),
        ...(permissionModeOf(config) !== undefined ? { permissionMode: permissionModeOf(config) } : {}),
        ...(typeof config["sessionId"] === "string" ? { approvalScope: config["sessionId"] } : {}),
        // RECORD mode: the value is the whole call payload rather than the projection, which is what
        // carries the provider handle and the conversation delta this entry has to report back.
        record: true,
      });
      const op = promptOp({ user: prompt, output: { name: "result", schema: { type: "string" } } });
      const result = await executor.start(op, ctx).result;
      const payload = result.value as LlmOutput | undefined;
      const metrics = result.metrics as AgentMetrics;
      if (!isOk(result)) return { error: result.error, metrics };
      // The payload shape a session records: the agent's answer, plus the handle the run ACTUALLY ended
      // in. Reported only when there IS a handle — an entry that claimed one unconditionally would make
      // every stateless call look resumable.
      const providerSessionId = payload?.providerSessionId;
      return {
        value: (payload?.value ?? "") as string,
        metrics,
        ...(providerSessionId !== undefined
          ? { session: { providerSessionId, messages: (payload?.messages ?? []) as readonly JsonValue[] } }
          : {}),
      };
    },
  };
}

/**
 * Build the `claude-code` adapter as a `runtime` registry entry. Its `prompt` input is the agent
 * instruction and its `config` input the authored runtime surface; `ctx.workspace` is the cwd,
 * `ctx.tools` the resolved tool set, and `ctx.approve` gates the agent's tool calls.
 */
export function createClaudeCodeFunction(options: ClaudeCodeFunctionOptions = {}): ReturnType<typeof agentRuntimeEntry> {
  return agentRuntimeEntry((perCall) => new AgentExecutor({ capabilities: DELEGATED_CAPS, ...perCall }), options);
}
