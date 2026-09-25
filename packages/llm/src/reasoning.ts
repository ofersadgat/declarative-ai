/**
 * Adapt the provider-NEUTRAL `ReasoningSpec` (the standard interface — effort and/or token budget) to the
 * provider-specific `providerOptions` the AI SDK forwards. This is the ONE place provider divergence lives
 * (mirrors the structured-output schema adapter, `providers/schema/`): the search config + stored config
 * JSON stay neutral, and we translate on the way OUT.
 *
 * The request arrives already FITTED to the model (`fitReasoning`, model-catalog.ts): its level is one
 * the model lists and its budget one it takes. What is left here is the provider's SHAPE:
 *
 *  - Anthropic takes a level (`effort`) and a thinking mode. A model whose schema takes a budget gets
 *    `thinking: {type: "enabled", budgetTokens}`; one that takes only a level gets
 *    `thinking: {type: "adaptive"}` — MEASURED from Anthropic's `/v1/models` (2026-09-24): every Claude
 *    from 4.7 on is adaptive-only, so a budget sent to one asks for a mode it does not have. 4.6 takes
 *    both; 4.5 and Haiku 4.5 take only a budget.
 *  - OpenRouter takes an effort LEVEL (`reasoning.effort`); a budget is sent as `reasoning.max_tokens`.
 *  - Native OpenAI takes an effort level too, but under its OWN key and spelling — `reasoningEffort`
 *    on `providerOptions.openai`, which `@ai-sdk/openai-compatible` lowers to `reasoning_effort`.
 *
 * That last arm is not a nicety. The AI SDK routes `providerOptions` by the PROVIDER'S OWN NAME, so a
 * request emitted under the `openrouter` key reaches a client built as `name: "openai"` and is
 * discarded without a warning — the call runs at the model's default effort while the config, the
 * catalog and the UI all say the level was honoured. Every route this function does not name fails
 * exactly that way, which is why {@link LOCAL_CONFIG_SCHEMA} omits `reasoning` outright rather than
 * offering a knob nothing translates.
 *
 * A model with NO schema (not in the catalog yet) gets what this did before the catalog knew levels: a
 * budget on Anthropic, and a level clamped to `low`–`high` elsewhere.
 */
import type { JsonValue } from "@declarative-ai/json";
import type { ProviderOptions, ReasoningEffort, ReasoningSpec } from "./llmConfig.js";
import { clampEffort, EFFORT_BUDGET, effortForBudget, type ParamAcceptance } from "./model-catalog.js";

/** The levels an UNKNOWN model is assumed to take — the three every reasoning endpoint has. */
const ASSUMED_EFFORTS: readonly ReasoningEffort[] = ["low", "medium", "high"];

/**
 * @param opts.anthropic  Native Anthropic — asked of the ROUTER, which owns the question.
 * @param opts.openai     The native `openai` route. Read from the route prefix rather than from the
 *                        router, which exposes no predicate for it; the two are checked in this
 *                        order and a call cannot be both.
 * @param opts.accept     What the model takes (its catalog gate). Absent, or an unknown model's gate ⇒
 *                        the assumptions above.
 */
export function adaptReasoning(
  spec: ReasoningSpec | undefined,
  opts: { anthropic: boolean; openai?: boolean; accept?: Pick<ParamAcceptance, "efforts" | "acceptsBudget"> },
): ProviderOptions | undefined {
  if (!spec || (spec.effort === undefined && spec.budgetTokens === undefined)) return undefined;
  const levels = opts.accept?.efforts;
  const thinks = spec.effort !== undefined && spec.effort !== "none";
  if (opts.anthropic) {
    // A level rides along only where the model is KNOWN to take one; unknown ⇒ a budget alone, as
    // before the catalog could say otherwise.
    const effort = thinks && levels !== undefined && levels.length > 0 ? spec.effort : undefined;
    if (opts.accept?.acceptsBudget ?? true) {
      const budgetTokens = spec.budgetTokens ?? (thinks ? EFFORT_BUDGET[spec.effort!] : undefined);
      if (budgetTokens === undefined) return undefined;
      return { anthropic: { thinking: { type: "enabled", budgetTokens }, ...(effort !== undefined ? { effort } : {}) } };
    }
    if (effort === undefined) return undefined;
    return { anthropic: { thinking: { type: "adaptive" }, effort } };
  }
  // A level for the two effort-shaped providers: the one asked for (already one the model lists, when
  // its levels are known), else the one a budget asks for.
  const asked = spec.effort ?? effortForBudget(spec.budgetTokens!);
  if (asked === "none" && levels === undefined) return undefined;
  const effort = levels === undefined ? clampEffort(asked, ASSUMED_EFFORTS) : asked;
  if (opts.openai === true) {
    // `reasoningEffort`, camelCase, on the provider's own key — the compatible client parses that
    // name and emits `reasoning_effort`. A budget is converted rather than forwarded: there is no
    // field on this API to forward it to.
    return { openai: { reasoningEffort: effort } };
  }
  // OpenRouter (default): prefer the effort level; fall back to the token budget as `max_tokens`.
  const reasoning: Record<string, JsonValue> = spec.effort !== undefined ? { effort } : { max_tokens: spec.budgetTokens ?? null };
  return { openrouter: { reasoning } };
}
