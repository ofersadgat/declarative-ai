/**
 * The SDK-DRIVEN delegated agent adapter (DESIGN §4.4).
 *
 * There is no `Runtime` interface and no normalized runtime-op payload: a runtime invocation is a
 * **plain `FunctionOp`** naming a registered function, so this factory produces a `runtime` REGISTRY
 * ENTRY carrying the delegated-agent capabilities (`mutatesWorkspace`, `memoizable: false`,
 * `policyEnforcement: "callback"`) — required and total, per §2, so the permission gate reads a definite
 * value instead of falling through an `undefined`. Permission gating and search refusal read that
 * resolved entry; the op shape carries no runtime marker at all.
 *
 * Behavior is unchanged: a delegated agent runs ITS OWN loop, so we configure it (a prompt, a workspace
 * `cwd`, an allowed-tools list, a permission mode) and route its native tool-approval callback back
 * through OUR approver (`ctx.approve`), keeping the human-gate UX uniform across runtimes. The agent is
 * reached through the injectable {@link AgentQuery} seam (default: the real SDK; tests: a fake).
 *
 * Register it as `registry.functions.registerFunction("claude-code", createClaudeCodeFunction())`, then
 * author a call with the `runtimeOp` builder — which lowers to exactly `{ kind: "function", functionRef:
 * "claude-code", input: { prompt, config } }`.
 */
import {
  failureOf,
  syncOnly,
  type BudgetMeter,
  type ExecMetrics,
  type BudgetMetrics,
  type FunctionResult,
  type ExecServices,
  type FunctionInputs,
  type JsonSchema,
  type JsonValue,
  type NativeToolRef,
  type Result,
  type RuntimeCapabilities,
} from "@declarative-ai/exec";
import type { Approver } from "@declarative-ai/permissions";
import { sdkAgentQuery } from "./sdkQuery";
import type { AgentPermissionMode, AgentQuery, AgentQueryOptions, InjectedTool } from "./seam";

/** Delegated agents: schema-constrained output isn't guaranteed (they answer in text), they mutate the
 *  workspace, run their own non-deterministic loop (not memoizable), gate tools via a callback, and are
 *  interactive (tool approvals route to our UI). Carried on the REGISTRY ENTRY, per §3.1. */
export const DELEGATED_CAPS: RuntimeCapabilities = {
  interactive: true,
  readOnly: false,
  mutatesWorkspace: true,
  memoizable: false,
  structuredOutput: false,
  policyEnforcement: "callback",
  sessionResume: false,
  streaming: true,
  runtime: "node",
};

const PERMISSION_MODES: readonly AgentPermissionMode[] = ["default", "plan", "acceptEdits", "bypassPermissions"];

/** The authored runtime surface, bound as the op's `config` input (§3.1) — never part of the op shape. */
export interface ClaudeCodeConfig {
  /** The agent's NATIVE permission profile control. */
  permissionMode?: AgentPermissionMode;
  /** Approval scope key for `ctx.approve` (defaults to `"delegated"`). */
  sessionId?: string;
}

export interface ClaudeCodeFunctionOptions {
  /** The agent-query seam. Default: {@link sdkAgentQuery} (lazily loads `@anthropic-ai/claude-agent-sdk`). */
  query?: AgentQuery;
  /** Override the advertised entry capabilities (e.g. a variant that is workspace-read-only). */
  capabilities?: RuntimeCapabilities;
  /** Inject `ctx.tools` into the agent over MCP so it calls OUR impls (identical behavior to the `llm`
   *  runtime). Default `true`. Set `false` to instead pass every tool name as a NATIVE allow-list (the agent
   *  uses its own built-ins by that name). */
  injectTools?: boolean;
  /** Per-logical-name overrides (DESIGN §5.1, "Tool renames are just overlay bindings"): a tool listed here resolves to the agent's
   *  NATIVE built-in `ref.native` (aliased) instead of being MCP-injected — so a run can use the agent's own
   *  `Read` for `read_file` while still injecting our `bash`. Ignored tools default to injection. */
  nativeTools?: Record<string, NativeToolRef>;
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
 * Build the `claude-code` adapter as a `runtime` registry entry. Its `prompt` input is the agent
 * instruction and its `config` input the authored runtime surface; `ctx.workspace` is the cwd,
 * `ctx.tools` the resolved tool set, and `ctx.approve` gates the agent's tool calls.
 *
 * Register it with `registry.functions.registerRuntime("claude-code", fn.run, fn.capabilities)`, then
 * author a call with the `runtimeOp` builder — which lowers to exactly
 * `{ kind: "function", functionRef: "claude-code", input: { prompt, config } }`.
 */
/** What a delegated agent measures: execution timing/counts plus the spend it billed itself. */
export interface AgentMetrics extends ExecMetrics, BudgetMetrics {}

export function createClaudeCodeFunction(options: ClaudeCodeFunctionOptions = {}): {
  capabilities: RuntimeCapabilities;
  run: (inputs: FunctionInputs, ctx: ExecServices) => Promise<FunctionResult<string, AgentMetrics>>;
} {
  const query = options.query ?? sdkAgentQuery;
  const inject = options.injectTools ?? true;
  const nativeMap = options.nativeTools ?? {};
  return {
    capabilities: options.capabilities ?? DELEGATED_CAPS,
    run: (inputs: FunctionInputs, ctx: ExecServices): Promise<FunctionResult<string, AgentMetrics>> => {
      // Errors are DATA (§4.2). The adapter still THROWS internally — an agent SDK is an exception-shaped
      // world — so `liftThrowing`'s classification is applied at the seam, which is what turns a 429 or an
      // abort inside the agent's own loop into a retriable/canceled outcome rather than a blanket permanent.
      const startMs = Date.now();
      const metricsOf = (costUsd?: number): AgentMetrics => ({
        startMs,
        durationMs: Date.now() - startMs,
        // One delegated agent is one child call from the graph's point of view, and its spend is a
        // child cost — that is how a budget gate sees through the delegation without child records.
        childLlmCalls: 1,
        // A delegated agent is the clearest case for cost NOT being an llm concern: it bills inside its
        // own loop and is the only thing that knows what it spent. `costUsd` is required, so an agent
        // that reported nothing says 0 with `costSource: "unknown"` rather than leaving it absent.
        costUsd: costUsd ?? 0,
        costSource: costUsd !== undefined ? "provider" : "unknown",
        ...(costUsd !== undefined ? { childCostUsd: costUsd } : {}),
      });
      return run(inputs, ctx).then(
        (r) => ({ value: r.text, metrics: metricsOf(r.costUsd) }),
        (e: unknown) => ({ error: failureOf(e, "claude-code"), metrics: metricsOf() }),
      );
    },
  };

  async function run(inputs: FunctionInputs, ctx: ExecServices): Promise<{ text: string; costUsd?: number }> {
      const config = configOf(inputs);
      const prompt = typeof inputs.prompt === "string" ? inputs.prompt : String(inputs.prompt ?? "");
      const approve = ctx.approve;
      const sessionId = typeof config.sessionId === "string" ? config.sessionId : "delegated";

      // The run is driven by the caller's abort signal directly. A run that completes without aborting
      // must not leave a listener attached to a possibly long-lived, shared `ctx.abortSignal`.
      const signal = ctx.abortSignal ?? new AbortController().signal;

      // Resolve each logical tool to NATIVE (the agent's built-in, aliased) or MCP-INJECTED (our impl,
      // ctx-bound). A tool is native when `injectTools:false` or it has a `nativeTools` entry; else it is
      // injected. The engine hands a delegated runtime RAW tools, and authorization flows through
      // `canUseTool` → `ctx.approve`, so injected tools are not double-gated.
      // A per-tool `deny` in the authored baseline needs no human, so it must reach the agent as
      // CONFIGURATION rather than waiting for an approval that will never be asked for. Native names are
      // what the agent addresses, so an aliased tool is denied under its `native` name.
      const denied = Object.entries(ctx.policy?.baseline?.tools ?? {})
        .filter(([, mode]) => mode === "deny")
        .map(([name]) => nativeMap[name]?.native ?? name);
      const denySet = new Set(denied);

      let allowedTools: string[] | undefined;
      let mcpTools: Record<string, InjectedTool> | undefined;
      if (ctx.tools) {
        const native: string[] = [];
        const injected: Record<string, InjectedTool> = {};
        for (const [name, tool] of Object.entries(ctx.tools)) {
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

      const queryOptions: AgentQueryOptions = {
        prompt,
        cwd: ctx.workspace?.root,
        allowedTools,
        ...(denied.length > 0 ? { disallowedTools: denied } : {}),
        mcpTools,
        // The injected-tool input gate is sync (agents-api `seam.ts`); the ctx seam is maybe-async —
        // narrow FAIL-CLOSED (json's `syncOnly`) rather than let an async validator read as a pass.
        ...(ctx.validator !== undefined ? { validator: syncOnly(ctx.validator) } : {}),
        permissionMode: permissionModeOf(config),
        // Route the agent's native tool-approval callback through our approver (DESIGN §5.1, "Delegated approval fidelity").
        canUseTool: approve
          ? async (req) => {
              const decision = await approve({ tool: req.toolName, input: req.input, sessionId });
              return decision.decision === "allow" ? { allow: true } : { allow: false, reason: `denied by permission policy` };
            }
          : undefined,
        abortSignal: signal,
      };

      let result: { text: string; costUsd?: number } | undefined;
      try {
        for await (const msg of query(queryOptions)) {
          if (msg.error) throw new ClaudeCodeError(`claude-code agent error: ${msg.error}`);
          if (msg.type === "result" && msg.result) result = msg.result;
        }
      } catch (e) {
        if (signal.aborted) throw new ClaudeCodeError("aborted", true);
        if (e instanceof ClaudeCodeError) throw e;
        throw new ClaudeCodeError(`claude-code query threw: ${(e as Error).message}`);
      }
      if (signal.aborted) throw new ClaudeCodeError("aborted", true);
      if (!result) throw new ClaudeCodeError("claude-code produced no result message");

      // A delegated agent spends real money inside its own loop, so the charge lands after the fact:
      // settle it against the wallet when one is injected. Absent meter ⇒ unmetered, as everywhere else.
      if (result.costUsd !== undefined && ctx.meter) await debitSpentCost(ctx.meter, result.costUsd);
    return result;
  }
}

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
async function debitSpentCost(meter: BudgetMeter, costUsd: number): Promise<void> {
  if (meter.debit) return meter.debit(costUsd);
  const reservation = await meter.reserve(costUsd);
  await reservation?.settle(costUsd);
}
