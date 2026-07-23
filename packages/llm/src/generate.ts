import { Output, jsonSchema, streamText, type LanguageModel, type ModelMessage, type StopCondition, type SystemModelMessage, type ToolChoice, type ToolSet } from "ai";
import { createLogger } from "./logger";
import { ModelInfo } from "./model-catalog";
import { classifyError, decodeWithSchema, describeError, isRateLimit, retryAfterMs as retryAfterMsOf, type JsonSchema, type JsonValue } from "@declarative-ai/json";
import type { LlmCallResult, LlmFailure, LlmMetrics, LlmOutput, ReasoningSegment, TokenCounts, ToolCall, ToolResult } from "./output";
import type { GeneratedFile } from "./files";
import type { LlmCallDefinition, SamplingConfiguration } from "./llmConfig";
import { promptAsMessages, type CallPromptInput } from "./prompt";

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
 * Never throws for an LLM failure: the result ALWAYS carries the best-effort output
 * (`value`), the reasoning (`thinking`), and costed `metrics`, with an OPTIONAL `error` —
 * and a failure that produced no value carries what the model DID emit on
 * `LlmFailure.rawOutput`. Preserving output + thinking on failure is what makes a failed
 * eval diagnosable rather than empty.
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


// The schema-document type is the ONE ops `JsonSchema<T>` (API.md, "The JSON vocabulary"): the same phantom-branded
// document everywhere a schema travels, so the call's `schema` threads its output type through to
// `CallOutcome<T>.value` and the §4 Ajv boundary enforces it at runtime. `typedSchema` brands a plain
// document; the ops typed layer's `FromSchema` inference derives `T` from an `as const` literal instead.
export type { JsonSchema };
export { typedSchema } from "@declarative-ai/json";

/**
 * The ENVIRONMENT one transport call runs against: everything that is NOT the serializable declaration
 * — the live model handle, the provider-ADAPTED schema and its reverse transform, the resolved tool
 * set, the merged `providerOptions`, the boundary check, and cancellation.
 *
 * `GenerateStructuredParams` is gone (DESIGN §4.1). It held live handles and closures —
 * genuinely different from a serializable declaration — but it FLATTENED declaration and environment
 * into one bag, re-listing every decoding knob in the process. Pushing the same `(def, env)` split one
 * layer down makes the flat bag disappear and the re-listing with it.
 */
export interface GenerateEnvironment<T = JsonValue> {
  /** The resolved model handle (the router turns the declaration's `model` string into this). */
  model: LanguageModel;
  /** The schema actually SENT — provider-adapted. Absent ⇒ the declaration's own `schema` goes as-is;
   *  the reconstruction + validation target is always the declaration's ORIGINAL. */
  outgoing?: JsonSchema<T>;
  /** Reverse the model's structured answer back to the ORIGINAL schema's shape (union reconstruction,
   *  any-decode, nullable-optional drop). This is also the seam a `Jsonify`→decoded lift belongs at
   *  (API.md, "Codecs and type names"): validate the wire form, then decode. */
  postProcess?: (value: JsonValue) => JsonValue;
  /** Boundary check against the ORIGINAL schema; throws (or rejects) to signal a PERMANENT validation
   *  failure — async because a store-backed validator may have `$ref` reads to do. */
  validate?: (value: JsonValue, originalSchema: JsonSchema<T>) => void | Promise<void>;
  /** The RESOLVED tool set (declarations combined with runtime `execute` impls), keyed by name. */
  tools?: ToolSet;
  /** How the model may select among `tools`. */
  toolChoice?: ToolChoice<ToolSet>;
  /** The stop condition bounding an executed tool loop; set only when tools have executors. Absent ⇒ a
   *  single step (tool calls are returned, not executed in a loop). */
  stopWhen?: StopCondition<ToolSet> | Array<StopCondition<ToolSet>>;
  /** Merged provider options (raw config passthrough + adapted reasoning) forwarded to the SDK. */
  providerOptions?: Record<string, JsonValue>;
  /**
   * Whether to attach `Output.object` (i.e. request `response_format` json/json_schema). Default
   * `true`. The caller sets it `false` for a TEXT-tier model: no `response_format` is sent, the schema
   * is described in the prompt instead, and the JSON is still parsed out of the returned text.
   */
  attachStructuredOutput?: boolean;
  /** Which decoding knobs this model actually accepts. Capability, not declaration — which is why it
   *  is environment. Default: accept everything. */
  accepts?: (param: string) => boolean;
  abortSignal?: AbortSignal;
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

/**
 * Narrow a value that arrived off the provider WIRE to `JsonValue` — the §2.2 boundary step. Everything
 * the SDK hands us (usage objects, provider metadata, tool inputs/results, parsed output) is JSON by
 * construction, but is typed `unknown`/`any` at the SDK seam; this is the single place that claim is
 * made, and anything genuinely non-JSON (a cyclic object, a class instance) round-trips through JSON to
 * become so — never silently escaping as `unknown` into an exported type.
 */
function asJson(value: unknown): JsonValue {
  if (value === null) return null;
  const t = typeof value;
  if (t === "string" || t === "number" || t === "boolean") return value as JsonValue;
  try {
    return JSON.parse(JSON.stringify(value ?? null)) as JsonValue;
  } catch {
    return null;
  }
}

/** The object-shaped {@link asJson}: provider metadata bags, which are always string-keyed. */
function asJsonRecord(value: Record<string, unknown> | undefined): Record<string, JsonValue> | undefined {
  if (value === undefined) return undefined;
  const out = asJson(value);
  return out !== null && typeof out === "object" && !Array.isArray(out) ? out : undefined;
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

export async function generateStructured<T = JsonValue>(
  def: LlmCallDefinition<T>,
  env: GenerateEnvironment<T>,
): Promise<LlmCallResult<T>> {
  const startMs = Date.now();
  const modelId = def.model;
  const accepts = env.accepts ?? ((): boolean => true);
  // Narrow the sampling-XOR-reasoning union ONCE: a reasoning config carries the neutral `reasoning`
  // spec and no sampling knobs; a sampling config the reverse.
  const sampling: Partial<SamplingConfiguration> = "reasoning" in def ? {} : def;
  const knob = <K extends keyof SamplingConfiguration>(name: K): SamplingConfiguration[K] | undefined =>
    accepts(name) ? sampling[name] : undefined;

  const metricsOf = (t: TokenCounts): LlmMetrics => {
    // Enrich with the 1-hour cache-write slice (from raw) so the table prices TTL tiers exactly.
    const enriched: TokenCounts = { ...t, cacheWrite1hTokens: cacheWrite1hOf(rawUsage) ?? t.cacheWrite1hTokens };
    // Prefer the provider's ACTUAL charge (OpenRouter); fall back to the cache-aware table.
    const tableCost = ModelInfo.instance.computeCost(modelId, enriched) ?? undefined;
    const cost = providerCost ?? tableCost;
    return {
      ...enriched, // carry the full cache/reasoning breakdown through for §6.2 + persistence
      // `costUsd` is REQUIRED, so an un-priced call says 0 with `costSource: "unknown"` rather than
      // leaving the field absent — "free" and "we could not price it" are different claims.
      costUsd: cost ?? 0,
      costSource: cost === undefined ? "unknown" : providerCost !== undefined ? "provider" : "table",
      rawUsage: rawUsage === undefined ? undefined : asJson(rawUsage), // ground truth for retroactive recompute
      startMs,
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
  // First-class tool calls/results (REAL tools only — never the structured-output emulation tool).
  const toolCalls: ToolCall[] = [];
  const toolResults: ToolResult[] = [];
  // Files the model generated (image/audio/…) — a parallel output channel.
  const producedFiles: GeneratedFile[] = [];
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
    failure?: LlmFailure;
  }): LlmCallResult<T> => {
    // `value` is reconstructed as `unknown` (parsed JSON / postProcess output) and asserted to `T`
    // here, the single construction site — the §4 Ajv boundary is what makes that assertion sound.
    const output: LlmOutput<T> = {
      value: args.value as T | undefined,
      thinking: thinking.length > 0 ? thinking : undefined,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      toolResults: toolResults.length > 0 ? toolResults : undefined,
      files: producedFiles.length > 0 ? producedFiles : undefined,
      finishReason: args.finishReason,
    };
    const metrics = metricsOf(args.tokens);
    if (args.failure) {
      // A failure that produced no VALUE still produced TEXT (a truncation's partial, an unparseable
      // body) — that is diagnostic evidence, and the failure is where it rides (`LlmFailure.rawOutput`).
      // With a value present (a validation reject preserving its parse) the raw text is that value's
      // wire form and adds nothing.
      const raw = args.value === undefined ? outputText() : "";
      const failure: LlmFailure = raw ? { ...args.failure, rawOutput: raw } : args.failure;
      log.warn("llm call failed", {
        modelId: modelId,
        classification: failure.classification,
        reason: failure.reason,
        finishReason: output.finishReason,
        ...metrics,
      });
      // The FAILURE branch still carries the payload: a truncated generation keeps its thinking and
      // trace, which is what makes a failed call diagnosable rather than empty.
      return { error: failure, value: output, metrics };
    }
    log.info("llm call ok", {
      modelId: modelId,
      finishReason: output.finishReason,
      reasoningSegments: thinking.length,
      ...metrics,
    });
    return { value: output, metrics };
  };

  const startReasoning = (id: string | undefined, providerMetadata?: Record<string, unknown>): ReasoningSegment => {
    const seg: ReasoningSegment = { type: "reasoning", text: "", textOffset: text.length };
    if (providerMetadata) seg.providerMetadata = asJsonRecord(providerMetadata);
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
  const outgoing = env.outgoing ?? def.schema;
  const postProcess = env.postProcess;
  // TEXT-tier (§5.1): a schema is present (so the parse+validate path below still runs) but we do NOT
  // request `response_format` — the model gets the shape in the prompt and answers with plain text.
  const attachStructuredOutput = env.attachStructuredOutput ?? true;

  log.debug("llm call start", {
    modelId: modelId,
    schema: def.schema ? (def.schema as { title?: string }).title : "(text)",
  });

  // Forward the prompt as the SDK's `Prompt`: `system` plus EXACTLY ONE of `messages`/`prompt` (the SDK
  // rejects both). `messages` wins when present; otherwise the `prompt` (string or message array).
  const promptPart: { messages: ModelMessage[] } | { prompt: string | ModelMessage[] } =
    def.messages !== undefined ? { messages: def.messages } : { prompt: def.prompt ?? "" };

  const result = streamText({
    model: env.model,
    system: def.system,
    ...promptPart,
    maxRetries: 0,
    abortSignal: env.abortSignal,
    maxOutputTokens: def.maxOutputTokens,
    temperature: knob("temperature"),
    topP: knob("topP"),
    topK: knob("topK"),
    presencePenalty: knob("presencePenalty"),
    frequencyPenalty: knob("frequencyPenalty"),
    seed: accepts("seed") ? def.seed : undefined,
    stopSequences: accepts("stopSequences") ? def.stopSequences : undefined,
    providerOptions: env.providerOptions as never,
    ...(env.tools ? { tools: env.tools } : {}),
    ...(env.toolChoice ? { toolChoice: env.toolChoice } : {}),
    ...(env.stopWhen ? { stopWhen: env.stopWhen } : {}),
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
        output?: unknown;
        result?: unknown;
        error?: unknown;
        usage?: unknown;
        totalUsage?: unknown;
        providerMetadata?: Record<string, unknown>;
        file?: { base64?: string; uint8Array?: Uint8Array; mediaType?: string };
        mediaType?: string;
        data?: unknown;
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
            // Surface the REAL tool call as a first-class outcome entry (positioned trace above is kept
            // for diagnostics). `input` is the parsed args; parse a streamed string if that's all we have.
            const toolName = p.toolName ?? existing?.toolName ?? "";
            let input: unknown = p.input;
            if (typeof input === "string") {
              try {
                input = JSON.parse(input);
              } catch {
                /* keep the raw string when it isn't JSON */
              }
            }
            toolCalls.push({ ...(id !== undefined ? { toolCallId: id } : {}), toolName, input: asJson(input) });
          }
          break;
        }
        case "tool-result": {
          // A tool executed in-loop returned a result (streamText ran the executor). Not the structured-
          // output tool (that never executes). Surface it for diagnostics / the caller.
          const id = p.toolCallId ?? p.id;
          if (!isStructuredOutputTool(p.toolName) && !(id !== undefined && structuredToolIds.has(id))) {
            toolResults.push({
              ...(id !== undefined ? { toolCallId: id } : {}),
              ...(p.toolName !== undefined ? { toolName: p.toolName } : {}),
              output: asJson(p.output ?? p.result),
            });
          }
          break;
        }
        case "file": {
          // A model-GENERATED file (image/audio/…). The high-level fullStream emits `{file: GeneratedFile}`;
          // be defensive about a low-level `{mediaType, data}` shape too. It lands in a blob output slot.
          const gf = p.file;
          const mediaType = gf?.mediaType ?? p.mediaType;
          const base64 = gf?.base64 ?? (typeof p.data === "string" ? p.data : undefined);
          if (base64 !== undefined && base64.length > 0 && mediaType !== undefined) {
            // A malformed side-channel part must not sink the PRIMARY output: decoding throws inside
            // the `fullStream` loop, which would abandon it before `finish` — losing the parsed value
            // AND every usage/cost figure — and report the whole call as an error.
            const bytes = base64ToBytes(base64);
            if (bytes !== undefined) producedFiles.push({ mediaType, bytes });
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
          if (seg && p.providerMetadata) seg.providerMetadata = { ...seg.providerMetadata, ...asJsonRecord(p.providerMetadata) };
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
    if (env.abortSignal?.aborted) {
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

  if (aborted || env.abortSignal?.aborted) {
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
  if (!def.schema) return build({ value: text, finishReason, tokens });

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

  // `parsed` is `JSON.parse` output — the boundary ends here (§2.2): from this point the value is
  // `JsonValue`, and the §4 validate step below is what upholds the caller's asserted `T`.
  const parsedJson = asJson(parsed);
  const value = postProcess ? postProcess(parsedJson) : parsedJson;

  if (env.validate) {
    try {
      await env.validate(value, def.schema!);
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

  // Lift the validated WIRE value to its DECODED type (API.md, "Codecs and type names"): validate the
  // `Jsonify<T>` form (above), THEN decode. `decodeWithSchema` resolves each `x-type` node's codec by
  // name and lifts it (an epoch number → a `Date`); with no `x-type` in the schema — the overwhelming
  // common case — it is a structural passthrough that returns the value unchanged. It runs ONLY here on
  // the structured path (the text branch returned at line 557 before reaching this), and only AFTER
  // validation, because the decoded form (e.g. a `Date`) would not pass the wire schema. The result is
  // the decoded `T`, flowing into the single `as T` construction seam in `build`.
  //
  // A registered codec's `decode` is arbitrary code that MAY throw on a value that passed the wire schema
  // but lies outside the codec's domain (e.g. an out-of-range epoch handed to a Date codec). Normalize
  // that throw into a failure Outcome — as the validation reject above does — rather than letting it
  // escape `runCall` and break the never-throws seam the removed executor layer used to hold.
  try {
    return build({ value: decodeWithSchema(def.schema, value), finishReason, tokens });
  } catch (err) {
    return build({
      value,
      finishReason,
      tokens,
      failure: { classification: "api-retriable", reason: `structured output decode failed: ${describeError(err)}` },
    });
  }
}

/**
 * Decode a provider's base64 file payload to bytes — a `blob` leaf holds BYTES (§7), so the decode
 * happens here rather than leaking a transport encoding into the value the caller receives.
 *
 * Accepts the URL-SAFE alphabet (`-`/`_`) and missing padding, both of which providers emit and both of
 * which make `atob` throw. Returns `undefined` rather than throwing on anything it still cannot read:
 * this runs inside the stream loop, where an exception costs the whole call's result and metrics.
 */
function base64ToBytes(base64: string): Uint8Array | undefined {
  const normalized = base64.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(normalized.length + ((4 - (normalized.length % 4)) % 4), "=");
  let binary: string;
  try {
    binary = atob(padded);
  } catch {
    return undefined;
  }
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
