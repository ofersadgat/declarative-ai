/**
 * The `llm` runtime adapter (HW-REDESIGN.md): a {@link Runtime} that runs a normalized agent operation
 * through the `llm-call` core. It ABSORBS what used to be `@declarative-ai/hw`'s `llmCallBinding` — the
 * same declarative `resolveConfig` pipeline — so hw no longer needs to know the `LlmCallDefinition` shape:
 * it just hands every runtime a uniform {@link RuntimeOp} and this adapter builds its own definition.
 *
 * Config resolution (DESIGN §1.5, §7): `defaults` ← `configs.get(op.config.configRef)` preset ← the
 * operation's inline `config`, merged family-aware and strict-parsed. Each layer may carry
 * DEFINITION-LAYER fields (`system`, `messages`, `attachments`, `timeoutMs`) alongside the config knobs;
 * `resolveConfig` splits them out. The rendered prompt is THE operation prompt (config-layer `messages`
 * become preamble turns with the prompt appended as the final user turn); a config-layer `prompt` is an
 * ERROR. A malformed merged config surfaces as a `permanent` operation failure (never a throw).
 */
import {
  resolveConfig,
  type ConfigurationRegistry,
  type ExecHandle,
  type ExecServices,
  type ExecutionSpec,
  type Executor,
  type Runtime,
  type RuntimeOp,
} from "@declarative-ai/core";
import { createLlmCallExecutor, emptyEvents } from "./executor";

/** Options for {@link createLlmRuntime}. */
export interface LlmRuntimeOptions {
  /** Provider-wide default config for every call through this runtime (was the `llmCallBinding` defaults
   *  arg): model, sampling, a shared `system`/`messages` preamble, a per-call `timeoutMs`, … */
  defaults?: Record<string, unknown>;
  /** Named-config registry: a state's `config.configRef` resolves to a preset merged UNDER the inline
   *  config (over `defaults`). */
  configs?: ConfigurationRegistry;
  /** The composed `llm-call` executor stack this runtime delegates to (e.g. the core with `withRepair`
   *  composed on). Defaults to a bare {@link createLlmCallExecutor}. */
  executor?: Executor;
}

/** An already-failed handle (permanent) — used when building the definition throws, so the runtime honors
 *  the never-throws contract just like an {@link Executor}. */
function failedHandle(reason: string): ExecHandle {
  return {
    events: emptyEvents(),
    outcome: Promise.resolve({
      rawText: "",
      finishReason: "error",
      metrics: { durationMs: 0 },
      error: { classification: "permanent" as const, reason },
    }),
    cancel: async () => {},
  };
}

/**
 * Build the `llm` runtime. Register it under a name in `registry.runtimes` (e.g. `"llm"`); a state's
 * `runtime.name` selects it and its `prompt`/`config` become the {@link RuntimeOp} this runs.
 */
export function createLlmRuntime(options: LlmRuntimeOptions = {}): Runtime {
  const executor = options.executor ?? createLlmCallExecutor();
  const defaults = options.defaults ?? {};
  return {
    capabilities: executor.capabilities,
    run(op: RuntimeOp, ctx: ExecServices): ExecHandle {
      let definition: Record<string, unknown>;
      try {
        const { configRef, ...inline } = op.config as { configRef?: unknown } & Record<string, unknown>;
        const preset = typeof configRef === "string" ? options.configs?.get(configRef) : undefined;
        const { config: resolved, definition: defLayer } = resolveConfig([defaults, preset, inline]);
        if (defLayer.prompt !== undefined) {
          throw new Error(
            "config supplies `prompt`, but a runtime operation's prompt is rendered from its template/skill — remove it (use `system` for instructions, or `messages` for preamble turns)",
          );
        }
        const preamble = defLayer.messages as unknown[] | undefined;
        definition = {
          ...defLayer,
          ...resolved,
          ...(preamble !== undefined ? { messages: [...preamble, { role: "user", content: op.prompt }] } : { prompt: op.prompt }),
          ...(op.system !== undefined ? { system: op.system } : {}),
          ...(op.timeoutMs !== undefined ? { timeoutMs: op.timeoutMs } : {}),
        };
        // Registered tools (op.tools) become FUNCTION tool DECLARATIONS on the call, appended to any the
        // config already carries (e.g. provider tools). Their EXECUTORS travel via ctx.tools (below), which
        // the llm-call executor adapts into the in-loop tool set — turning this into a bounded agent loop.
        if (op.tools !== undefined) {
          const toolDefs = Object.entries(op.tools).map(([name, t]) => ({
            name,
            ...(t.description !== undefined ? { description: t.description } : {}),
            inputSchema: t.inputSchema,
          }));
          definition.tools = [...((definition.tools as unknown[] | undefined) ?? []), ...toolDefs];
        }
      } catch (e) {
        return failedHandle(`invalid llm config: ${(e as Error).message}`);
      }
      const spec: ExecutionSpec = {
        kind: "llm-call",
        definition,
        inputs: {},
        outputSchema: op.outputSchema,
        limits: op.timeoutMs !== undefined ? { timeoutMs: op.timeoutMs } : undefined,
        abortSignal: op.abortSignal,
      };
      return executor.start(spec, op.tools !== undefined ? { ...ctx, tools: op.tools } : ctx);
    },
  };
}
