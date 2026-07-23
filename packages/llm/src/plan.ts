/**
 * `plan` — the declarative-ai "dry run" (Terraform `plan` to `execute`'s `apply`). Given a resolved call
 * DECLARATION, it reports what WOULD happen without touching a provider: which provider/model serves it, its
 * content-hash identity, how a structured schema would be enforced, which requested params/modalities the
 * model actually supports, and a token + table-cost estimate. All from the local model catalog — no network.
 *
 * Pair with `resolveConfig` (core): `resolveConfig(layers)` composes + parses the config; build the
 * declaration (config + prompt + schema); `plan(declaration)` analyzes it. Nothing here executes.
 */
import { hashCanonical, sha256Hex, type JsonSchema, type Serializable } from "@declarative-ai/json";
import { isReasoningConfig } from "./llmConfig";
import { estimateCallTokens } from "./tokens";
import type { FilePart, ModelMessage } from "ai";
import { promptText } from "./prompt";
import type { LlmCallDefinition } from "./llmConfig";
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

/**
 * The declaration reduced to something CANONICALIZABLE. `FileInput.data` may now hold raw bytes (§7
 * made blob a leaf kind), and JCS has no `Uint8Array` case — it is neither an array nor `toJSON`-able,
 * so it serializes as a plain object with ONE KEY PER BYTE. A 5 MB attachment becomes a ~50 MB string
 * and a sort over five million numeric keys before the digest is even started. Bytes are replaced by
 * their own hash, which is both fast and the identity we actually want: the same image supplied twice
 * by different means plans to the same content hash.
 */
function hashableDefinition(def: LlmCallDefinition): Serializable {
  const attachments = def.attachments;
  if (!attachments?.some((a) => a.data instanceof Uint8Array)) return def as unknown as Serializable;
  return {
    ...def,
    attachments: attachments.map((a) => (a.data instanceof Uint8Array ? { ...a, data: { bytesHash: sha256Hex(latin1(a.data)) } } : a)),
  } as unknown as Serializable;
}

/** Bytes as an injective string — every byte maps to a distinct code unit, so the digest is stable and
 *  collision-free without a base64 round trip. */
function latin1(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
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
export function plan(def: LlmCallDefinition & { schema?: JsonSchema }): CallPlan {
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
    contentHash: hashCanonical(hashableDefinition(def)),
    unsupportedParams,
    ...(modalities !== undefined ? { modalities } : {}),
    ...(structuredOutput !== undefined ? { structuredOutput } : {}),
    estimate: { inputTokens: est.inputTokens, outputTokens: est.outputTokens, ...(costUsd !== undefined ? { costUsd } : {}) },
    issues,
  };
}
