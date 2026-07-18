import { Output, jsonSchema, streamText, type LanguageModel } from "ai";
import { createLogger } from "./logger";
import { computeCost } from "./model-catalog";
import { classifyError, describeError, isRateLimit, retryAfterMs as retryAfterMsOf, type ErrorClass } from "@ai-exec/core";

/**
 * The streaming structured-output call (§5/§5.1) + metrics. ONE LLM request; the §6.1
 * runner wraps this as a WDK step and supplies the §6.2 deadline-driven `abortSignal`.
 *
 * Pipeline: adapt schema for Anthropic -> `streamText` with `Output.object` (so the model
 * emits structured JSON, constrained where supported) -> consume `fullStream` (which
 * surfaces the REAL provider error + carries usage + streams the JSON as text) -> parse,
 * reconstruct to the ORIGINAL schema, and optional-validate on the way out. `maxRetries: 0`
 * — retries are the budget-gated eval loop's (§10.4).
 *
 * Never throws for an LLM failure: a single `CallOutcome` ALWAYS carries the best-effort
 * output (`value`/`rawText`), the reasoning (`thinking`), and costed `metrics`, with an
 * OPTIONAL `error`. Preserving output + thinking on failure is what makes a failed eval
 * diagnosable and maps to the §4 `ValueResult = ValueRef & { error? }` shape.
 */

const log = createLogger("engine.providers.generate", { tag: "llm" });

/**
 * One non-output segment of the trace, positioned against the output text it accompanies.
 * Everything the model emits that ISN'T the structured output is captured here with a
 * `type` discriminator, so a v2 op (or a human) can tell model reasoning apart from
 * intermediate tool usage — and so none of it is ever mistaken for the output (§5.1):
 *  - `"reasoning"` — an extended-thinking block; `text` is the thinking.
 *  - `"tool-call"` — an INTERMEDIATE tool the model invoked mid-generation (v2 candidates
 *    with tools); `text` is the tool's serialized args and `toolName` names it. This is
 *    NOT the structured-output (jsonTool) call — that one becomes the output, never a segment.
 */
export interface ReasoningSegment {
  type: "reasoning" | "tool-call";
  text: string;
  /** Length of the accumulated OUTPUT text when this segment began — lets us see how it
   *  interleaves with / precedes the output it produced (§5.1). */
  textOffset: number;
  /** Tool name, for `type:"tool-call"` segments. */
  toolName?: string;
  /** Native provider metadata (e.g. an Anthropic thinking-block signature), preserved
   *  intact for any v2 op that feeds the trace back. */
  providerMetadata?: Record<string, unknown>;
}

/**
 * Token counts pulled off an AI SDK usage object. `inputTokens`/`outputTokens` are the
 * cache-INCLUSIVE totals; the optional breakdown (Anthropic `inputTokenDetails` /
 * `outputTokenDetails`) is what makes cost billing-accurate under prompt caching and feeds
 * the §6.2 ITPM estimator (which excludes cache reads). Absent fields = the provider didn't
 * report them (then cost falls back to flat input-rate pricing).
 */
export interface TokenCounts {
  /** Total input tokens, INCLUDING cache reads + writes (matches the provider's billed input). */
  inputTokens?: number;
  /** Total output tokens, including reasoning tokens. */
  outputTokens?: number;
  /** Uncached (fresh) input tokens — billed at the base input rate. */
  noCacheTokens?: number;
  /** Cached input tokens read on a cache hit — billed at the discounted read rate (~0.1x). */
  cacheReadTokens?: number;
  /** Input tokens written into the prompt cache — billed at the write rate (~1.25x). */
  cacheWriteTokens?: number;
  /** 1-hour-TTL subset of `cacheWriteTokens` (from `raw` usage) — billed at the 2x tier. */
  cacheWrite1hTokens?: number;
  /** Reasoning/thinking output tokens (subset of `outputTokens`), for diagnostics. */
  reasoningTokens?: number;
  /** Provider-reported grand total, when present. */
  totalTokens?: number;
}

export interface CallMetrics extends TokenCounts {
  /** USD price (§5/§10.2). The provider's reported charge when available (OpenRouter usage
   *  accounting), else computed cache-aware from the token split. */
  cost?: number;
  /** Where `cost` came from: `"provider"` = the provider's actual charge (authoritative);
   *  `"table"` = our price-table estimate. Absent when no cost could be determined. */
  costSource?: "provider" | "table";
  /** Provider's exact usage object (ground truth) — kept so `cost` is always recomputable. */
  rawUsage?: unknown;
  durationMs: number;
}

export interface CallFailure {
  classification: ErrorClass;
  /** The real underlying cause (§10.4), stored as the error artifact. */
  reason: string;
  /** Server-advised wait before the next attempt, from the `retry-after` header (§10.4) —
   *  ms. Present only on a transient rate-limit/5xx that carried the header; the eval loop's
   *  retry honors it instead of blind exponential backoff. */
  retryAfterMs?: number;
  /** True iff this was a 429 rate-limit — feeds AIMD's multiplicative decrease (§6.2.B). */
  rateLimited?: boolean;
}

export interface CallOutcome {
  /** Parsed structured value when the model produced usable output (kept even when a
   *  later validation fails); undefined only when nothing parseable was produced. */
  value?: unknown;
  /** Raw accumulated output text — always present (possibly ""); the partial on failure. */
  rawText: string;
  /** Structured reasoning trace, positioned against the output (§4/§5.1). */
  thinking?: ReasoningSegment[];
  finishReason: string;
  metrics: CallMetrics;
  /** Present iff the call failed; `value`/`rawText`/`thinking` are then best-effort partials. */
  error?: CallFailure;
}

export interface GenerateStructuredParams {
  model: LanguageModel;
  /** The routing id — for pricing and provider detection. */
  modelId: string;
  prompt: string;
  system?: string;
  /** The ORIGINAL JSON Schema (reconstruction + validation target). OMITTED for a TEXT-output op
   *  (§3.14): no schema → plain `streamText`, the raw text IS the value (the user's text→text flow). */
  schema?: Record<string, unknown>;
  abortSignal?: AbortSignal;
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
  providerOptions?: Record<string, unknown>;
  /**
   * Milestone 4 Ajv hook: throw to signal a PERMANENT validation failure of the
   * reconstructed value against the original schema (§5.1). Default: no-op.
   */
  validate?: (value: unknown, originalSchema: Record<string, unknown>) => void;
  /**
   * Reverse the model's structured answer back to the ORIGINAL schema's shape (§5.1) — the
   * `postProcess` from the schema adapter (union reconstruction, any-decode, nullable-optional drop).
   * Supplied by the executor when it PRE-adapts the schema (which also lets it set the matching strict
   * flag at model resolution); when omitted, `generateStructured` self-adapts from the model's profile.
   */
  postProcess?: (value: unknown) => unknown;
  /**
   * Whether to attach `Output.object` (i.e. request `response_format` json/json_schema) for this call.
   * Default `true`. The executor sets it `false` for a TEXT-tier model (§5.1 `enforce:"text"`): no
   * `response_format` is sent, the schema is described in the prompt instead, and the JSON is still
   * parsed out of the returned text (`schema` stays present so the parse+validate path runs). Ignored
   * when there's no `schema` (a genuine text-output op is already plain `streamText`).
   */
  attachStructuredOutput?: boolean;
}

/** Best-effort JSON stringify; "" if the value can't be serialized (cyclic, etc.). */
function safeStringify(value: unknown): string {
  if (value === undefined) return "";
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}

/** A number iff `v` is a finite number, else undefined. */
function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/**
 * Extract counts + the cache/reasoning breakdown from an AI SDK `LanguageModelUsage`
 * (the flat, post-mapping shape on `finish.totalUsage` / `finish-step.usage` /
 * `onAbort` step usage). The detail fields come off `inputTokenDetails`/`outputTokenDetails`;
 * the deprecated flat aliases (`cachedInputTokens`, `reasoningTokens`) are read as a fallback.
 * Missing detail just means the provider didn't report it — never fabricated.
 */
export function extractTokenCounts(usage: unknown): TokenCounts {
  if (usage == null || typeof usage !== "object") return {};
  const u = usage as {
    inputTokens?: unknown;
    outputTokens?: unknown;
    totalTokens?: unknown;
    cachedInputTokens?: unknown;
    reasoningTokens?: unknown;
    inputTokenDetails?: { noCacheTokens?: unknown; cacheReadTokens?: unknown; cacheWriteTokens?: unknown };
    outputTokenDetails?: { reasoningTokens?: unknown };
  };
  const inDet = u.inputTokenDetails ?? {};
  const outDet = u.outputTokenDetails ?? {};
  return {
    inputTokens: num(u.inputTokens),
    outputTokens: num(u.outputTokens),
    totalTokens: num(u.totalTokens),
    noCacheTokens: num(inDet.noCacheTokens),
    cacheReadTokens: num(inDet.cacheReadTokens) ?? num(u.cachedInputTokens),
    cacheWriteTokens: num(inDet.cacheWriteTokens),
    reasoningTokens: num(outDet.reasoningTokens) ?? num(u.reasoningTokens),
  };
}

export async function generateStructured(params: GenerateStructuredParams): Promise<CallOutcome> {
  const startMs = Date.now();

  const metricsOf = (t: TokenCounts): CallMetrics => {
    // Enrich with the 1-hour cache-write slice (from raw) so the table prices TTL tiers exactly.
    const enriched: TokenCounts = { ...t, cacheWrite1hTokens: cacheWrite1hOf(rawUsage) ?? t.cacheWrite1hTokens };
    // Prefer the provider's ACTUAL charge (OpenRouter); fall back to the cache-aware table.
    const tableCost = computeCost(params.modelId, enriched) ?? undefined;
    const cost = providerCost ?? tableCost;
    return {
      ...enriched, // carry the full cache/reasoning breakdown through for §6.2 + persistence
      cost,
      costSource: cost === undefined ? undefined : providerCost !== undefined ? "provider" : "table",
      rawUsage, // provider ground truth for retroactive recompute; closed over (set during streaming)
      durationMs: Date.now() - startMs,
    };
  };

  // Accumulated state — preserved into the outcome on every path.
  let text = "";
  // Structured-output TOOL fallback (§5.1). `Output.object` is `responseFormat:{type:"json"}`,
  // and the SDK parses the value from the TEXT channel (`parseCompleteOutput({text})`) — so
  // `text` is the real output and every provider re-emits its emulation-tool args as text
  // (Anthropic does). These two capture the rare provider that surfaces the structured-output
  // (jsonTool) call as an actual tool part instead: its args streamed (`structuredToolText`)
  // and/or handed back on the `tool-call` part (`structuredToolInput`). Used ONLY as a
  // fallback when `text` is empty, and ONLY for the structured-output tool — never another
  // tool. INTERMEDIATE tool calls do NOT land here; they go into the reasoning trace below.
  let structuredToolText = "";
  let structuredToolInput: unknown;
  const thinking: ReasoningSegment[] = [];
  const reasoningById = new Map<string, ReasoningSegment>();
  // Tool-call ids known to be the structured-output (jsonTool) tool vs intermediate tools.
  const structuredToolIds = new Set<string>();
  const toolSegById = new Map<string, ReasoningSegment>();
  let salvage: TokenCounts | undefined;
  // Provider's exact usage object (ground truth). NB the SDK's cross-step aggregate
  // (`finish.totalUsage`) DROPS `raw`; it survives only on the per-step `finish-step.usage`
  // (and the `onAbort` step usage). A structured call is single-step, so the per-step raw is
  // exact — captured here so `cost` can be recomputed from ground truth later (TTL tiers, rate
  // corrections) even though the primary `cost` is token-derived.
  let rawUsage: unknown;
  // The provider's ACTUAL charged cost, when it reports one (OpenRouter usage accounting
  // surfaces it at providerMetadata.openrouter.usage.cost). Authoritative — preferred over the
  // price-table estimate, which can't track OpenRouter's dynamic upstream routing + markup (§5).
  let providerCost: number | undefined;
  let errorPart: unknown;

  const rawOf = (u: unknown): unknown =>
    u != null && typeof u === "object" ? (u as { raw?: unknown }).raw : undefined;

  /** Pull a provider-reported USD cost off a part's providerMetadata (OpenRouter today). */
  const providerCostOf = (pm: Record<string, unknown> | undefined): number | undefined => {
    const openrouter = pm?.openrouter as { usage?: { cost?: unknown } } | undefined;
    const cost = openrouter?.usage?.cost;
    return typeof cost === "number" && Number.isFinite(cost) ? cost : undefined;
  };

  /**
   * The 1-hour-TTL slice of the cache WRITE, read from Anthropic's `raw` usage
   * (`cache_creation.ephemeral_1h_input_tokens`). The SDK's flat `cacheWriteTokens` sums the
   * 5-min + 1-hour writes, which bill at different rates (1.25x vs 2x); this recovers the 2x
   * slice so the cost is exact. Absent ⇒ all writes are 5-minute (the default).
   */
  const cacheWrite1hOf = (raw: unknown): number | undefined => {
    if (raw == null || typeof raw !== "object") return undefined;
    const cc = (raw as { cache_creation?: unknown }).cache_creation;
    if (cc == null || typeof cc !== "object") return undefined;
    const v = (cc as { ephemeral_1h_input_tokens?: unknown }).ephemeral_1h_input_tokens;
    return typeof v === "number" && Number.isFinite(v) ? v : undefined;
  };

  /** The best-effort OUTPUT text (output channel only — never an intermediate tool). */
  const outputText = (): string => {
    if (text.trim()) return text;
    if (structuredToolText.trim()) return structuredToolText;
    if (typeof structuredToolInput === "string") return structuredToolInput;
    if (structuredToolInput !== undefined) {
      try {
        return JSON.stringify(structuredToolInput);
      } catch {
        return "";
      }
    }
    return "";
  };

  /** A tool name designates the structured-output (jsonTool) emulation tool (§5.1). */
  const isStructuredOutputTool = (toolName: string | undefined): boolean =>
    toolName === "json" || toolName === "object";

  const build = (args: {
    value?: unknown;
    finishReason: string;
    tokens: TokenCounts;
    failure?: CallFailure;
  }): CallOutcome => {
    const outcome: CallOutcome = {
      value: args.value,
      rawText: outputText(),
      thinking: thinking.length > 0 ? thinking : undefined,
      finishReason: args.finishReason,
      metrics: metricsOf(args.tokens),
      error: args.failure,
    };
    if (args.failure) {
      log.warn("llm call failed", {
        modelId: params.modelId,
        classification: args.failure.classification,
        reason: args.failure.reason,
        finishReason: outcome.finishReason,
        ...outcome.metrics,
      });
    } else {
      log.info("llm call ok", {
        modelId: params.modelId,
        finishReason: outcome.finishReason,
        reasoningSegments: thinking.length,
        ...outcome.metrics,
      });
    }
    return outcome;
  };

  const startReasoning = (id: string | undefined, providerMetadata?: Record<string, unknown>): ReasoningSegment => {
    const seg: ReasoningSegment = { type: "reasoning", text: "", textOffset: text.length };
    if (providerMetadata) seg.providerMetadata = providerMetadata;
    if (id !== undefined) reasoningById.set(id, seg);
    thinking.push(seg);
    return seg;
  };

  /** Record an INTERMEDIATE tool call as a `type:"tool-call"` trace segment (never output). */
  const intermediateToolSegment = (id: string | undefined, toolName?: string): ReasoningSegment => {
    const seg: ReasoningSegment = { type: "tool-call", text: "", textOffset: text.length, toolName };
    if (id !== undefined) toolSegById.set(id, seg);
    thinking.push(seg);
    return seg;
  };

  // Pure transport (§5.1/§6.1): `params.schema` is the OUTGOING schema the executor already adapted for
  // this provider (the single adaptation site, `executeStructuredCall`, which also set the matching
  // strict flag at model resolution), and `params.postProcess` reverses it back to the original shape.
  // A TEXT-output op carries neither. We never adapt here — the model is already resolved, so we can't
  // set the strict flag, which is exactly why adaptation belongs upstream.
  const outgoing = params.schema;
  const postProcess = params.postProcess;
  // TEXT-tier (§5.1): a schema is present (so the parse+validate path below still runs) but we do NOT
  // request `response_format` — the model gets the shape in the prompt and answers with plain text.
  const attachStructuredOutput = params.attachStructuredOutput ?? true;

  log.debug("llm call start", {
    modelId: params.modelId,
    schema: params.schema ? (params.schema as { title?: string }).title : "(text)",
  });

  const result = streamText({
    model: params.model,
    system: params.system,
    prompt: params.prompt,
    maxRetries: 0,
    abortSignal: params.abortSignal,
    maxOutputTokens: params.maxOutputTokens,
    temperature: params.temperature,
    topP: params.topP,
    topK: params.topK,
    stopSequences: params.stopSequences,
    providerOptions: params.providerOptions as never,
    ...(outgoing && attachStructuredOutput ? { output: Output.object({ schema: jsonSchema(outgoing as never) }) } : {}),
    onAbort: ({ steps }) => {
      const lastUsage = steps.at(-1)?.usage;
      salvage = extractTokenCounts(lastUsage);
      rawUsage ??= rawOf(lastUsage);
    },
    // Capture the real error AND override the SDK's default onError (which console.errors).
    onError: ({ error }) => {
      errorPart ??= error;
    },
  });

  // An errored call rejects ALL of the result's lazy promises; mark them handled so a path
  // that returns without awaiting one never leaks an unhandled rejection.
  void Promise.allSettled([result.text, result.finishReason, result.totalUsage, result.reasoningText]);

  // Consume `fullStream`: the `error` part carries the REAL provider error (the SDK
  // otherwise hides it behind NoOutputGeneratedError), `abort` signals a deadline cutoff,
  // `finish`/`finish-step` carry usage, and text/reasoning deltas stream the body + trace.
  let aborted = false;
  let streamTokens: TokenCounts | undefined;
  try {
    for await (const part of result.fullStream) {
      const p = part as {
        type: string;
        id?: string;
        toolCallId?: string;
        toolName?: string;
        text?: string;
        delta?: string;
        input?: unknown;
        error?: unknown;
        usage?: unknown;
        totalUsage?: unknown;
        providerMetadata?: Record<string, unknown>;
      };
      switch (p.type) {
        case "text-delta":
          text += p.text ?? p.delta ?? "";
          break;
        case "tool-input-start":
          // Classify the tool: the structured-output (jsonTool) tool feeds the OUTPUT; any
          // other tool is INTERMEDIATE and goes to the reasoning trace, never the output.
          if (isStructuredOutputTool(p.toolName)) {
            if (p.id !== undefined) structuredToolIds.add(p.id);
          } else {
            intermediateToolSegment(p.id, p.toolName);
          }
          break;
        case "tool-input-delta":
          if (p.id !== undefined && structuredToolIds.has(p.id)) {
            structuredToolText += p.delta ?? "";
          } else if (p.id !== undefined) {
            // Intermediate tool args — accumulate into its trace segment (create if the
            // tool-input-start was missed).
            (toolSegById.get(p.id) ?? intermediateToolSegment(p.id)).text += p.delta ?? "";
          }
          break;
        case "tool-call": {
          const id = p.toolCallId ?? p.id;
          const structured = isStructuredOutputTool(p.toolName) || (id !== undefined && structuredToolIds.has(id));
          if (structured) {
            if (p.input !== undefined) structuredToolInput = p.input;
          } else {
            const existing = id !== undefined ? toolSegById.get(id) : undefined;
            if (existing) {
              // The tool streamed its args before being named — backfill `toolName` so it
              // is reliably present at the top level of the trace block.
              if (existing.toolName === undefined && p.toolName !== undefined) existing.toolName = p.toolName;
            } else {
              // An intermediate tool delivered with no streamed input deltas — record it once.
              const seg = intermediateToolSegment(id, p.toolName);
              seg.text = typeof p.input === "string" ? p.input : safeStringify(p.input);
            }
          }
          break;
        }
        case "reasoning-start":
          startReasoning(p.id, p.providerMetadata);
          break;
        case "reasoning-delta": {
          const seg = (p.id !== undefined ? reasoningById.get(p.id) : undefined) ?? startReasoning(p.id, p.providerMetadata);
          seg.text += p.text ?? p.delta ?? "";
          break;
        }
        case "reasoning-end": {
          const seg = p.id !== undefined ? reasoningById.get(p.id) : undefined;
          if (seg && p.providerMetadata) seg.providerMetadata = { ...seg.providerMetadata, ...p.providerMetadata };
          break;
        }
        case "error":
          errorPart ??= p.error;
          break;
        case "abort":
          aborted = true;
          break;
        case "finish":
          streamTokens = extractTokenCounts(p.totalUsage);
          break;
        case "finish-step":
          // `usage` carries the per-step breakdown AND `raw` (the aggregate drops raw);
          // `providerMetadata` carries OpenRouter's actual charged cost.
          streamTokens ??= extractTokenCounts(p.usage);
          rawUsage ??= rawOf(p.usage);
          providerCost ??= providerCostOf(p.providerMetadata);
          break;
      }
    }
  } catch (err) {
    if (params.abortSignal?.aborted) {
      return build({
        finishReason: "aborted",
        tokens: salvage ?? streamTokens ?? {},
        failure: { classification: "network-retriable", reason: "deadline-abort: stream cut off before completion" },
      });
    }
    return build({
      finishReason: "error",
      tokens: streamTokens ?? salvage ?? {},
      failure: { classification: classifyError(err), reason: describeError(err), retryAfterMs: retryAfterMsOf(err), rateLimited: isRateLimit(err) },
    });
  }

  if (aborted || params.abortSignal?.aborted) {
    return build({
      finishReason: "aborted",
      tokens: salvage ?? streamTokens ?? {},
      failure: { classification: "network-retriable", reason: "deadline-abort: stream cut off before completion" },
    });
  }
  if (errorPart !== undefined) {
    return build({
      finishReason: "error",
      tokens: streamTokens ?? salvage ?? {},
      failure: { classification: classifyError(errorPart), reason: describeError(errorPart), retryAfterMs: retryAfterMsOf(errorPart), rateLimited: isRateLimit(errorPart) },
    });
  }

  const tokens = streamTokens ?? salvage ?? {};
  let finishReason = "unknown";
  try {
    finishReason = await result.finishReason;
  } catch {
    // a remapped/unavailable finish reason is non-fatal; the parse below is the gate.
  }

  // Text output (no schema, §3.14): the accumulated raw text IS the value. A "length" finish just
  // means the token cap was hit — the text is still usable (truncated), not a hard failure here.
  if (!params.schema) return build({ value: text, finishReason, tokens });

  if (finishReason === "length") {
    return build({
      finishReason,
      tokens,
      failure: { classification: "api-retriable", reason: 'finishReason "length": output truncated before the JSON closed' },
    });
  }

  // Resolve the structured payload (§5.1). The output is the OUTPUT channel only — the
  // structured-output (jsonTool) tool, never an intermediate tool (those are in `thinking`):
  //  - JSON response-format -> the JSON streamed as text (`text`), what the SDK itself parses;
  //  - structured-output tool surfaced as a tool part -> its parsed args (`structuredToolInput`)
  //    or streamed args (`structuredToolText`), used ONLY when `text` is empty.
  // A truncated/absent body (the common §5.1 permanent failure) surfaces here as
  // unparseable/empty regardless of path.
  let parsed: unknown;
  if (text.trim()) {
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      return build({
        finishReason,
        tokens,
        failure: { classification: "api-retriable", reason: `unparseable structured output: ${describeError(err)}` },
      });
    }
  } else if (structuredToolInput !== undefined && typeof structuredToolInput === "object" && structuredToolInput !== null) {
    parsed = structuredToolInput;
  } else {
    const toolText =
      typeof structuredToolInput === "string" && structuredToolInput.trim()
        ? structuredToolInput
        : structuredToolText;
    if (toolText.trim()) {
      try {
        parsed = JSON.parse(toolText);
      } catch (err) {
        return build({
          finishReason,
          tokens,
          failure: { classification: "api-retriable", reason: `unparseable structured output: ${describeError(err)}` },
        });
      }
    }
  }
  if (parsed == null) {
    return build({
      finishReason,
      tokens,
      failure: { classification: "api-retriable", reason: "structured output was empty/absent despite a normal finish" },
    });
  }

  const value = postProcess ? postProcess(parsed) : parsed;

  if (params.validate) {
    try {
      params.validate(value, params.schema!);
    } catch (err) {
      // Preserve the parsed value even though it failed validation (§4 preserve-on-error).
      return build({
        value,
        finishReason,
        tokens,
        failure: { classification: "api-retriable", reason: `post-reconstruction validation failed: ${describeError(err)}` },
      });
    }
  }

  return build({ value, finishReason, tokens });
}
