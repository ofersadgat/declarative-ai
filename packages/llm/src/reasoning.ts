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
 *
 * NOTE: the exact option SHAPES need live verification against each provider/model (no API keys here); the
 * MAPPING logic is unit-tested, and the result is `undefined` when nothing is requested — so a no-reasoning
 * call is byte-identical to before (this can't regress existing runs).
 */
import type { ReasoningSpec } from "@ai-exec/core";

/** Representative thinking budgets for an effort level, for providers that only accept a budget. */
const EFFORT_BUDGET: Record<"low" | "medium" | "high", number> = { low: 2048, medium: 8192, high: 16384 };

export function adaptReasoning(spec: ReasoningSpec | undefined, opts: { anthropic: boolean }): Record<string, unknown> | undefined {
  if (!spec || (spec.effort === undefined && spec.budgetTokens === undefined)) return undefined;
  if (opts.anthropic) {
    const budgetTokens = spec.budgetTokens ?? (spec.effort ? EFFORT_BUDGET[spec.effort] : undefined);
    if (budgetTokens === undefined) return undefined;
    return { anthropic: { thinking: { type: "enabled", budgetTokens } } };
  }
  // OpenRouter (default): prefer the effort level; fall back to the token budget as `max_tokens`.
  const reasoning = spec.effort !== undefined ? { effort: spec.effort } : { max_tokens: spec.budgetTokens };
  return { openrouter: { reasoning } };
}
