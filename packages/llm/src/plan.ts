/**
 * `plan` — the declarative-ai "dry run" (Terraform `plan` to `execute`'s `apply`). Given a resolved call
 * DECLARATION, it reports what WOULD happen without touching a provider: which provider/model serves it, its
 * content-hash identity, how a structured schema would be enforced, which requested params/modalities the
 * model actually supports, and a token + table-cost estimate. All from the local model catalog — no network.
 *
 * Pair with `resolveConfig` (core): `resolveConfig(layers)` composes + parses the config; build the
 * declaration (config + prompt + schema); `plan(declaration)` analyzes it. Nothing here executes.
 */
import { hashCanonical, isReasoningConfig } from "@declarative-ai/core";
import { estimateCallTokens } from "@declarative-ai/services";
import type { FilePart, ModelMessage } from "ai";
import { promptText } from "./generate";
import type { LlmCallDefinition } from "./llmStep";
import { ModelInfo, SAMPLING_PARAM_NAMES, type Modalities } from "./model-catalog";
import { familyForModel, type ModelFamily } from "./router";
import { adaptSchemaCached, profileForModelId, type Enforcement } from "./schema";

/** The result of planning a call — everything knowable before execution. */
export interface CallPlan {
  /** The provider + model that would serve the call. */
  provider: { family: ModelFamily; modelId: string };
  /** Content-hash identity of the declaration (the resolved content hash, per the memo-key model). */
  contentHash: string;
  /** Sampling params present in the declaration that the model does NOT accept (they'd be dropped). */
  unsupportedParams: string[];
  /** The model's declared input/output modalities (from the catalog), when known. */
  modalities?: Modalities;
  /** How a structured-output schema (if any) would be enforced: strict / advisory / text. */
  structuredOutput?: Enforcement;
  /** Token + USD table estimate (no provider call; the actual charge is reconciled at execute time). */
  estimate: { inputTokens: number; outputTokens: number; costUsd?: number };
  /** Human-readable fit issues (unsupported params, reasoning rejected, etc.). */
  issues: string[];
}

/** The input modality a media part requires, derived from its IANA media type: `image/*` → "image",
 *  `audio/*` → "audio", `video/*` → "video", anything else (pdf, text, octet-stream) → "file". */
function requiredModality(mediaType: string): string {
  const top = mediaType.split("/")[0]?.toLowerCase();
  return top === "image" || top === "audio" || top === "video" ? top : "file";
}

/** Every input modality the declaration's media actually requires — per attachment/part mediaType, so an
 *  audio input is gated as "audio", not lumped into a generic media check. */
function requiredInputModalities(def: LlmCallDefinition): Set<string> {
  const needs = new Set<string>();
  for (const a of def.attachments ?? []) needs.add(requiredModality(a.mediaType));
  const messages: ModelMessage[] = [...(def.messages ?? []), ...(Array.isArray(def.prompt) ? def.prompt : [])];
  for (const m of messages) {
    if (!Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (part.type === "image") needs.add("image");
      else if (part.type === "file") needs.add(requiredModality((part as FilePart).mediaType ?? ""));
    }
  }
  return needs;
}

/** Plan a resolved call declaration (config + prompt [+ schema]) — no execution, no network. */
export function plan(def: LlmCallDefinition & { schema?: Record<string, unknown> }): CallPlan {
  const modelId = def.model;
  // Route-prefixed `{route}/{model}` — the exact catalog key (§5). The catalog / profile / cost lookups
  // all key on this full id; `familyForModel` extracts the route for the `family` field + issue messages.
  const catalog = ModelInfo.instance;
  const gate = catalog.paramAcceptance(modelId);
  const modalities = catalog.modalities(modelId);
  const issues: string[] = [];

  // Sampling params present in the declaration that the model would filter out — judged by the SAME
  // acceptance gate `executeStructuredCall` filters with, so plan and execute cannot drift.
  const paramKeys = Object.keys(SAMPLING_PARAM_NAMES) as Array<keyof typeof SAMPLING_PARAM_NAMES>;
  const bag = def as unknown as Record<string, unknown>;
  const unsupportedParams = paramKeys.filter((k) => bag[k] !== undefined && !gate.accepts(k));
  if (unsupportedParams.length > 0) issues.push(`model ${modelId} does not accept: ${unsupportedParams.join(", ")} (dropped at call time)`);

  if (isReasoningConfig(def) && !gate.acceptsReasoning) issues.push(`model ${modelId} does not accept reasoning (dropped at call time)`);

  // Modality fit (Phase 5): each media input's REQUIRED modality (from its mediaType) vs `modalities.input`;
  // requested OUTPUT modalities vs `modalities.output`.
  const needs = requiredInputModalities(def);
  if (needs.size > 0 && modalities?.input) {
    const missing = [...needs].filter((m) => !modalities.input!.includes(m));
    if (missing.length > 0) {
      issues.push(`model ${modelId} does not accept ${missing.join(", ")} inputs (modalities.input: ${modalities.input.join(", ")})`);
    }
  }
  if (def.outputModalities && modalities?.output) {
    const unsupported = def.outputModalities.filter((m) => !modalities.output!.includes(m));
    if (unsupported.length > 0) issues.push(`model ${modelId} cannot produce output modalities: ${unsupported.join(", ")}`);
  }

  let structuredOutput: Enforcement | undefined;
  if (def.schema) {
    const profile = profileForModelId(modelId);
    if (profile) structuredOutput = adaptSchemaCached(def.schema, profile).enforce;
  }

  const est = estimateCallTokens(promptText(def), undefined, def.maxOutputTokens);
  const costUsd = catalog.computeCostUsd(modelId, est.inputTokens, est.outputTokens) ?? undefined;

  return {
    provider: { family: familyForModel(modelId), modelId },
    contentHash: hashCanonical(def),
    unsupportedParams,
    ...(modalities !== undefined ? { modalities } : {}),
    ...(structuredOutput !== undefined ? { structuredOutput } : {}),
    estimate: { inputTokens: est.inputTokens, outputTokens: est.outputTokens, ...(costUsd !== undefined ? { costUsd } : {}) },
    issues,
  };
}
