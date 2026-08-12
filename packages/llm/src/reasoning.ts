/**
 * Adapt the provider-NEUTRAL `ReasoningSpec` (the standard interface — effort and/or token budget) to the
 * provider-specific `providerOptions` the AI SDK forwards. This is the ONE place provider divergence lives
 * (mirrors the structured-output schema adapter, `providers/schema/`): the search config + stored config
 * JSON stay neutral, and we translate on the way OUT.
 *
 * Provider shapes differ in WHICH knob they take:
 *  - Anthropic extended thinking takes a token BUDGET (`thinking.budgetTokens`); an effort level is mapped
 *    to a representative budget.
 *  - OpenRouter takes an effort LEVEL (`reasoning.effort`); a budget is sent as `reasoning.max_tokens`.
 *  - Native OpenAI takes an effort level too, but under its OWN key and spelling — `reasoningEffort`
 *    on `providerOptions.openai`, which `@ai-sdk/openai-compatible` lowers to `reasoning_effort`.
 *
 * That last arm is not a nicety. The AI SDK routes `providerOptions` by the PROVIDER'S OWN NAME, so a
 * request emitted under the `openrouter` key reaches a client built as `name: "openai"` and is
 * discarded without a warning — the call runs at the model's default effort while the config, the
 * catalog's `acceptsReasoning` and the UI all say the level was honoured. Every route this function
 * does not name fails exactly that way, which is why {@link LOCAL_CONFIG_SCHEMA} omits `reasoning`
 * outright rather than offering a knob nothing translates.
 *
 * NOTE: the exact option SHAPES need live verification against each provider/model (no API keys here); the
 * MAPPING logic is unit-tested, and the result is `undefined` when nothing is requested — so a no-reasoning
 * call is byte-identical to before (this can't regress existing runs).
 */
import type { JsonValue } from "@declarative-ai/json";
import type { ProviderOptions, ReasoningSpec } from "./llmConfig.js";

/** Representative thinking budgets for an effort level, for providers that only accept a budget. */
const EFFORT_BUDGET: Record<NonNullable<ReasoningSpec["effort"]>, number> = { low: 2048, medium: 8192, high: 16384, xhigh: 32768 };

/**
 * The effort level a provider that tops out at `high` is asked for.
 *
 * `xhigh` exists because a delegated Claude Code run has a tier the three-value vocabulary could not
 * name. A message-based provider that has never heard of it must not fail the call over it: asking for
 * more thought than a model offers is satisfied by giving it all of it, so the level CLAMPS. Losing
 * the request entirely would be the wrong answer; refusing it would make one transport's vocabulary
 * everyone's problem.
 */
const CLAMPED: Record<NonNullable<ReasoningSpec["effort"]>, "low" | "medium" | "high"> = {
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "high",
};

/**
 * The effort level a budget asks for, on a provider that has no budget knob.
 *
 * The inverse of {@link EFFORT_BUDGET}: the lowest level whose representative budget covers what was
 * asked for, and the top level for anything above it. Native OpenAI is the case — `reasoning_effort`
 * is the only lever it offers, so a spec carrying only a budget would otherwise be dropped, and
 * dropping it is the failure this whole module exists to prevent.
 */
function effortForBudget(budgetTokens: number): "low" | "medium" | "high" {
  if (budgetTokens <= EFFORT_BUDGET.low) return "low";
  if (budgetTokens <= EFFORT_BUDGET.medium) return "medium";
  return "high";
}

/**
 * @param opts.anthropic  Native Anthropic — asked of the ROUTER, which owns the question.
 * @param opts.openai     The native `openai` route. Read from the route prefix rather than from the
 *                        router, which exposes no predicate for it; the two are checked in this
 *                        order and a call cannot be both.
 */
export function adaptReasoning(
  spec: ReasoningSpec | undefined,
  opts: { anthropic: boolean; openai?: boolean },
): ProviderOptions | undefined {
  if (!spec || (spec.effort === undefined && spec.budgetTokens === undefined)) return undefined;
  if (opts.anthropic) {
    const budgetTokens = spec.budgetTokens ?? (spec.effort ? EFFORT_BUDGET[spec.effort] : undefined);
    if (budgetTokens === undefined) return undefined;
    return { anthropic: { thinking: { type: "enabled", budgetTokens } } };
  }
  if (opts.openai === true) {
    // `reasoningEffort`, camelCase, on the provider's own key — the compatible client parses that
    // name and emits `reasoning_effort`. A budget is converted rather than forwarded: there is no
    // field on this API to forward it to.
    const effort = spec.effort !== undefined ? CLAMPED[spec.effort] : effortForBudget(spec.budgetTokens!);
    return { openai: { reasoningEffort: effort } };
  }
  // OpenRouter (default): prefer the effort level; fall back to the token budget as `max_tokens`.
  const reasoning: Record<string, JsonValue> = spec.effort !== undefined ? { effort: CLAMPED[spec.effort] } : { max_tokens: spec.budgetTokens ?? null };
  return { openrouter: { reasoning } };
}
