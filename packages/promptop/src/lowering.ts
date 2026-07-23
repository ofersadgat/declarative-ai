/**
 * `PromptOp → LlmCallDefinition` — the LOWERING (DESIGN §4.1).
 *
 * The op SHAPE stays in `@declarative-ai/ops`: a `PromptOp` is text slots + a config slot + a schema,
 * and it imports nothing from `llm`. The lowering is llm-specific, so it lives here, one layer up.
 * This is also the whole of what `PromptOpRunner` used to be — an op→spec lowering wrapped around an
 * executor — which is why removing the runner cost nothing: the lowering became a function and the
 * wrapper became an ordinary {@link Executor}.
 *
 * Config resolution is unchanged: `defaults` ← `configs.get(config.configRef)` preset ← the op's inline
 * `config`, merged family-aware and strict-parsed. The op's `user` text is THE prompt (config-layer
 * `messages` become preamble turns with it appended as the final user turn); a config-layer `prompt` is
 * an ERROR. The op's `output.schema` is the structured-output contract — it now travels IN the
 * definition rather than being smuggled through a spec field and cast (§5.1) — but only when the output
 * parameter's KIND is `json`; `text` and `blob` outputs lower to a schema-less (text/bytes) call.
 */
import type { InlineFamily, JsonSchema, JsonValue, PromptOp, Tool } from "@declarative-ai/exec";
import { resolveConfig, type ConfigLayer, type ConfigurationRegistry, type LlmCallDefinition, type ToolDefinition } from "@declarative-ai/llm";

export interface LoweringOptions {
  /** Provider-wide default config for every call: model, sampling, a shared `system`/`messages`
   *  preamble, a per-call `timeoutMs`, … */
  defaults?: ConfigLayer;
  /** Named-config registry: an op's `config.configRef` resolves to a preset merged UNDER the inline
   *  config (over `defaults`). */
  configs?: ConfigurationRegistry;
  /** Executable tools the call may invoke mid-loop, keyed by logical name. Each becomes a FUNCTION tool
   *  DECLARATION on the definition; the EXECUTORS travel separately (the executor adapts the SAME map
   *  into `env.toolExecutors`), which is the same declaration/environment split the model handle and
   *  validator use. Declaration and executor must come from one source: a declared tool with no
   *  executor is single-turn, so the two drifting apart degrades the loop silently. */
  tools?: Record<string, Tool>;
}

/** A `PromptOp`'s inline `config` slot, read as a config layer. The slot is typed by the
 *  `LlmConfiguration` schema, so this is a view, not a parse — `resolveConfig` does the parsing. */
export function configLayerOf(config: JsonValue): { configRef?: string; inline: ConfigLayer } {
  if (config === null || typeof config !== "object" || Array.isArray(config)) return { inline: {} };
  const { configRef, ...inline } = config as { configRef?: unknown } & Record<string, JsonValue>;
  return {
    ...(typeof configRef === "string" ? { configRef } : {}),
    inline: inline as ConfigLayer,
  };
}

/**
 * Lower a resolved `PromptOp` to the serializable call definition. THROWS on a malformed config — the
 * caller (the executor) turns that into a `permanent` outcome, honoring the never-throws contract.
 */
export function lowerPromptOp(op: PromptOp<InlineFamily>, options: LoweringOptions = {}): LlmCallDefinition {
  const { configRef, inline } = configLayerOf(op.config);
  const preset = configRef !== undefined ? options.configs?.get(configRef) : undefined;
  const { definition: resolved } = resolveConfig([options.defaults, preset, inline]);

  if (resolved.prompt !== undefined) {
    throw new Error(
      "config supplies `prompt`, but a PromptOp's prompt is its `user` text — remove it (use `system` for instructions, or `messages` for preamble turns)",
    );
  }
  const preamble = resolved.messages;
  const definition: LlmCallDefinition = {
    ...resolved,
    ...(preamble !== undefined ? { messages: [...preamble, { role: "user", content: op.user }] } : { prompt: op.user }),
    ...(op.system !== undefined ? { system: op.system } : {}),
  };

  // The output schema comes off the OP's output slot — it is the same document the pre-run checker
  // type-checked the wiring against, so the two layers cannot drift.
  //
  // But ONLY a `json`-kind parameter carries a STRUCTURED-OUTPUT contract. `kindFor` derives the kind
  // from that very schema (`ops/model.ts`): a `text` parameter's schema is `{type:"string"}` and a
  // `blob` parameter's is `{type:"string", contentMediaType:"image/png"}` — both DESCRIBE A LEAF, they
  // are not a JSON contract to decode against. Forwarding either made `executeLlmCall` attach
  // `Output.object` and `generate.ts` demand parseable JSON on the text channel, so a text op that
  // answered `four` failed as "unparseable structured output", and a blob op whose bytes arrived and
  // projected correctly was still reported failed as "structured output was empty/absent". A `text` op
  // therefore takes the TEXT path (§5.2 — a text call yields `string`) and a `blob` op the bytes path
  // (§7.1), both of which are exactly "no schema on the definition".
  //
  // The DELETE (rather than a conditional spread) matters: `schema` is a SIGNATURE key, so a
  // `defaults`/preset layer can carry one, and a config layer must not be able to force structured
  // output onto a parameter whose kind says otherwise. The op's output slot is the authority.
  if (op.output.kind === "json" && op.output.schema !== undefined) definition.schema = op.output.schema;
  else delete definition.schema;

  if (options.tools !== undefined) {
    const toolDefs: ToolDefinition[] = Object.entries(options.tools).map(([name, t]) => ({
      name,
      ...(t.description !== undefined ? { description: t.description } : {}),
      inputSchema: t.inputSchema as JsonSchema,
    }));
    definition.tools = [...(definition.tools ?? []), ...toolDefs];
  }
  return definition;
}
