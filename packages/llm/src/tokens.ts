/**
 * Cheap token estimation for rate pre-admission (DESIGN §5, "Rate limiting"). It lands in `llm` rather than
 * beside the rate limiter because it estimates a PROMPT's footprint — the limiter itself is generic
 * counting machinery and needs no notion of a prompt.
 */
import type { CallTokenEstimate } from "./output";

/**
 * Estimate the token footprint of a call for rate pre-admission, input/output SPLIT (ITPM and
 * OTPM are metered separately). A rough chars/4 proxy for input + the configured output ceiling.
 * Deliberately conservative-ish: over-estimating only costs a little throughput, never a 429.
 */
export function estimateCallTokens(prompt: string, system: string | undefined, maxOutputTokens: number | undefined): CallTokenEstimate {
  const inputChars = prompt.length + (system?.length ?? 0);
  return { inputTokens: Math.ceil(inputChars / 4), outputTokens: maxOutputTokens ?? 512 };
}
