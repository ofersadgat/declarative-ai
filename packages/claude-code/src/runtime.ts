/**
 * The `claude-code` DELEGATED runtime adapter (RUNTIMES-AND-PERMISSIONS.md §1, build order 5). Unlike the
 * composed `llm` runtime (we drive the tool loop), a delegated agent runs ITS OWN loop; we configure it —
 * a prompt, a workspace `cwd`, an allowed-tools list, a permission mode — and route its native tool-approval
 * callback back through OUR approver (`ctx.approve`), so the human-gate UX is uniform across runtimes. The
 * agent is reached through the injectable {@link AgentQuery} seam (default: the real SDK; tests: a fake).
 */
import type { ExecHandle, ExecServices, ExecutorCapabilities, NativeToolRef, Outcome, Runtime, RuntimeOp } from "@declarative-ai/core";
import { sdkAgentQuery } from "./sdkQuery";
import type { AgentPermissionMode, AgentQuery, AgentQueryOptions, InjectedTool } from "./seam";

/** Delegated agents: schema-constrained output isn't guaranteed (they answer in text), they mutate the
 *  workspace, run their own non-deterministic loop (not memoizable), gate tools via a callback, and need a
 *  real process (not edge-safe). Interactive: tool approvals route to our UI. */
const DELEGATED_CAPS: ExecutorCapabilities = {
  structuredOutput: false,
  sessionResume: false,
  streaming: true,
  interactive: true,
  mutatesWorkspace: true,
  policyEnforcement: "callback",
  memoizable: false,
  runtime: "node",
};

const PERMISSION_MODES: readonly AgentPermissionMode[] = ["default", "plan", "acceptEdits", "bypassPermissions"];

export interface ClaudeCodeRuntimeOptions {
  /** The agent-query seam. Default: {@link sdkAgentQuery} (lazily loads `@anthropic-ai/claude-agent-sdk`). */
  query?: AgentQuery;
  /** Override the advertised capabilities (e.g. a variant that supports structured output). */
  capabilities?: ExecutorCapabilities;
  /** Inject `runtime.tools` into the agent over MCP so it calls OUR impls (identical behavior to the `llm`
   *  runtime). Default `true`. Set `false` to instead pass every tool name as a NATIVE allow-list (the agent
   *  uses its own built-ins by that name). */
  injectTools?: boolean;
  /** Per-logical-name overrides (RUNTIMES-AND-PERMISSIONS.md §3): a tool listed here resolves to the agent's
   *  NATIVE built-in `ref.native` (aliased) instead of being MCP-injected — so a run can use the agent's own
   *  `Read` for `read_file` while still injecting our `bash`. Ignored tools default to injection. */
  nativeTools?: Record<string, NativeToolRef>;
}

async function* noEvents(): AsyncGenerator<never> {}

function failed(reason: string, startMs: number): Outcome {
  return { rawText: "", finishReason: "error", metrics: { startMs, durationMs: Date.now() - startMs }, error: { classification: "permanent", reason } };
}

/** Read an author-supplied permission mode from the op config, ignoring an unknown value. */
function permissionModeOf(config: Record<string, unknown>): AgentPermissionMode | undefined {
  const m = config["permissionMode"];
  return typeof m === "string" && (PERMISSION_MODES as readonly string[]).includes(m) ? (m as AgentPermissionMode) : undefined;
}

/**
 * Build the `claude-code` runtime. Register it under a name in `registry.runtimes` (e.g. `"claude-code"`);
 * a state's `runtime.name` selects it. Its `prompt` and `tools` (allow-list) become the agent query;
 * `ctx.workspace` is the cwd and `ctx.approve` gates its tool calls.
 */
export function createClaudeCodeRuntime(options: ClaudeCodeRuntimeOptions = {}): Runtime {
  const query = options.query ?? sdkAgentQuery;
  const inject = options.injectTools ?? true;
  const nativeMap = options.nativeTools ?? {};
  return {
    capabilities: options.capabilities ?? DELEGATED_CAPS,
    run(op: RuntimeOp, ctx: ExecServices): ExecHandle {
      const startMs = Date.now();
      const abort = new AbortController();
      const signal = op.abortSignal ? anySignal([op.abortSignal, abort.signal]) : abort.signal;

      const approve = ctx.approve;
      const sessionId = typeof op.config["sessionId"] === "string" ? (op.config["sessionId"] as string) : "delegated";
      // Resolve each logical tool to NATIVE (the agent's built-in, aliased) or MCP-INJECTED (our impl,
      // ctx-bound). A tool is native when `injectTools:false` or it has a `nativeTools` entry; else it is
      // injected. The engine hands a delegated runtime RAW tools, and authorization flows through
      // `canUseTool` → `ctx.approve`, so injected tools are not double-gated.
      let allowedTools: string[] | undefined;
      let mcpTools: Record<string, InjectedTool> | undefined;
      if (op.tools) {
        const native: string[] = [];
        const injected: Record<string, InjectedTool> = {};
        for (const [name, tool] of Object.entries(op.tools)) {
          const ref = nativeMap[name];
          if (!inject || ref) native.push(ref ? ref.native : name);
          else injected[name] = { description: tool.description, inputSchema: tool.inputSchema, run: (input) => tool.run(input, ctx) };
        }
        allowedTools = native;
        if (Object.keys(injected).length > 0) mcpTools = injected;
      }
      const queryOptions: AgentQueryOptions = {
        prompt: op.prompt,
        cwd: ctx.workspace?.root,
        allowedTools,
        mcpTools,
        permissionMode: permissionModeOf(op.config),
        // Route the agent's native tool-approval callback through our approver (RUNTIMES-AND-PERMISSIONS §4).
        canUseTool: approve
          ? async (req) => {
              const decision = await approve({ tool: req.toolName, input: req.input, sessionId });
              return decision.decision === "allow" ? { allow: true } : { allow: false, reason: `denied by permission policy` };
            }
          : undefined,
        abortSignal: signal,
      };

      const outcome = (async (): Promise<Outcome> => {
        let result: { text: string; costUsd?: number } | undefined;
        try {
          for await (const msg of query(queryOptions)) {
            if (msg.error) return failed(`claude-code agent error: ${msg.error}`, startMs);
            if (msg.type === "result" && msg.result) result = msg.result;
          }
        } catch (e) {
          if (signal.aborted) return { rawText: "", finishReason: "error", metrics: { startMs, durationMs: Date.now() - startMs }, error: { classification: "canceled", reason: "aborted" } };
          return failed(`claude-code query threw: ${(e as Error).message}`, startMs);
        }
        if (signal.aborted) return { rawText: "", finishReason: "error", metrics: { startMs, durationMs: Date.now() - startMs }, error: { classification: "canceled", reason: "aborted" } };
        if (!result) return failed("claude-code produced no result message", startMs);
        return {
          value: result.text,
          rawText: result.text,
          finishReason: "stop",
          metrics: { startMs, durationMs: Date.now() - startMs, cost: result.costUsd },
        };
      })();

      return { events: noEvents(), outcome, cancel: async () => abort.abort() };
    },
  };
}

/** Combine abort signals (avoids relying on a specific `AbortSignal.any` lib target). */
function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort();
      break;
    }
    s.addEventListener("abort", () => controller.abort(), { once: true });
  }
  return controller.signal;
}
