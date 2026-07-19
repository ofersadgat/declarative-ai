/**
 * §Pricing reservation estimate — the per-call cost projection behind BOTH pre-call holds (the wallet
 * RESERVE and the search-gate budget hold) and their affordable-output clamps. Input tokens are a chars/4
 * proxy (exact enough — the input is in hand); output tokens are the genuinely unknown half, estimated
 * per-problem:
 *
 *  1. OBSERVED — this run's own settled calls, per model (`ctx.outputTokenStats`). A run targets one
 *     problem, so run-local stats ARE per-problem stats (different prompts produce very different output
 *     lengths, so a global per-model average would mislead). Multiplied by `headroomMultiplier` (the
 *     runtime-tunable HOLD knob, default {@link DEFAULT_HOLD_OUTPUT_MULTIPLIER}) — settle usually
 *     corrects DOWNWARD, and an over-reserve lives only for the call's duration.
 *  2. FALLBACK — ~2.5× the input tokens (floored), for the cold start before any call has settled.
 *  3. Always CLAMPED by the call's configured `maxOutputTokens` when one is set (a real ceiling).
 *
 * When even that estimate doesn't fit the wallet/budget, the caller flips the relationship: it computes
 * the AFFORDABLE output tokens from the remaining headroom and sets that as the call's REAL
 * `maxOutputTokens` — the reserve stops being a guess and becomes provider-enforced
 * (`affordableOutputTokens`). Known under-count carried from `search/cost.ts`: reasoning models can bill
 * thinking beyond the output cap.
 */
import { computeCostUsd } from "./model-catalog";

/** chars/4 input-token proxy over one or more prompt fragments (system + user + …). Inlined from
 *  findmyprompt `src/engine/search/cost.ts`; mirrors the input half of `estimateCallTokens`
 *  (@declarative-ai/services concurrency.ts) — kept separate from output because the rates differ. */
export function estimateInputTokens(...texts: (string | undefined)[]): number {
  let chars = 0;
  for (const t of texts) chars += t?.length ?? 0;
  return Math.ceil(chars / 4);
}

/** §budget hold price: output-token headroom when no runtime policy names one. ×2 — settle usually
 *  corrects DOWNWARD, and an over-hold lives only for the call's duration. Inlined from findmyprompt
 *  `src/engine/strategies/abstractions/searchNode.ts`. */
export const DEFAULT_HOLD_OUTPUT_MULTIPLIER = 2;

export interface OutputTokenStats {
  n: number;
  mean: number;
}

/** Minimum output-token estimate — a floor under both the fallback and the observed mean. */
const MIN_OUTPUT_TOKENS = 256;
/** Below this affordable ceiling a call can't produce a useful answer — refuse instead of clamping. */
export const MIN_USEFUL_OUTPUT_TOKENS = 128;
/** Observed means are trusted once this many calls have settled for the model. */
const MIN_OBSERVATIONS = 3;

/** Estimate a call's output tokens (see file header): observed-mean × headroom → else ~2.5 × input;
 *  clamped by the configured cap when set. `headroomMultiplier` is the runtime-tunable hold knob
 *  (`SearchGate.holdOutputMultiplier`); the 2.5× cold-start fallback is already a generous guess and is
 *  NOT multiplied. */
export function estimateOutputTokens(
  modelId: string,
  inputTokens: number,
  configMaxOutputTokens: number | undefined,
  stats?: Map<string, OutputTokenStats>,
  headroomMultiplier: number = DEFAULT_HOLD_OUTPUT_MULTIPLIER,
): number {
  const seen = stats?.get(modelId);
  const guess =
    seen && seen.n >= MIN_OBSERVATIONS
      ? Math.ceil(seen.mean * Math.max(1, headroomMultiplier))
      : Math.ceil(inputTokens * 2.5);
  const floored = Math.max(MIN_OUTPUT_TOKENS, guess);
  return configMaxOutputTokens != null ? Math.min(configMaxOutputTokens, floored) : floored;
}

/** One call's estimated USD cost for the wallet reserve. Un-priced models estimate $0 (they cost the
 *  platform nothing; the balance>0 admission floor still applies). */
export function estimateCallCostUsd(modelId: string, inputTokens: number, outputTokens: number): number {
  return computeCostUsd(modelId, inputTokens, outputTokens) ?? 0;
}

/**
 * The AFFORDABLE output-token ceiling for a tight wallet: how many output tokens `availableUsd` buys
 * after the input's cost, at the model's output rate. Returns `Infinity` for an un-priced model (nothing
 * to clamp against) and 0 when even the input doesn't fit.
 */
export function affordableOutputTokens(modelId: string, inputTokens: number, availableUsd: number): number {
  const inputUsd = computeCostUsd(modelId, inputTokens, 0);
  const perOutputToken = (computeCostUsd(modelId, 0, 1_000_000) ?? 0) / 1_000_000;
  if (inputUsd == null || perOutputToken <= 0) return Number.POSITIVE_INFINITY; // un-priced — no clamp basis
  const headroom = availableUsd - inputUsd;
  return headroom <= 0 ? 0 : Math.floor(headroom / perOutputToken);
}

/** Fold one settled call's observed output tokens into the run's per-model stats (a running mean). */
export function noteOutputTokens(stats: Map<string, OutputTokenStats>, modelId: string, outputTokens: number | undefined): void {
  if (outputTokens == null || !Number.isFinite(outputTokens) || outputTokens < 0) return;
  const cur = stats.get(modelId);
  if (!cur) {
    stats.set(modelId, { n: 1, mean: outputTokens });
    return;
  }
  cur.n += 1;
  cur.mean += (outputTokens - cur.mean) / cur.n;
}
