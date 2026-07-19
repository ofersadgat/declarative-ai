import { Output, jsonSchema, streamText, type LanguageModel, type ModelMessage, type StopCondition, type SystemModelMessage, type ToolChoice, type ToolSet } from "ai";
import { createLogger } from "./logger";
import { computeCost } from "./model-catalog";
import { classifyError, describeError, isRateLimit, retryAfterMs as retryAfterMsOf, sha256Hex, type ErrorClass, type FileInput, type ProducedArtifact, type SamplingConfiguration, type ToolCall, type ToolResult } from "@declarative-ai/core";

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

export interface CallOutcome<T = unknown> {
  /** Parsed structured value when the model produced usable output (kept even when a
   *  later validation fails); undefined only when nothing parseable was produced. Typed
   *  as `T`, the output type the call's `schema` describes (see {@link JsonSchema}). */
  value?: T;
  /** Raw accumulated output text — always present (possibly ""); the partial on failure. */
  rawText: string;
  /** Structured reasoning trace, positioned against the output (§4/§5.1). */
  thinking?: ReasoningSegment[];
  /** Tool calls the model requested — the primary output for a single-turn tool op (no executor supplied),
   *  or the intermediate calls of an executed loop. Excludes the structured-output emulation tool. */
  toolCalls?: ToolCall[];
  /** Results of tool calls executed in-loop (present only when executors ran). */
  toolResults?: ToolResult[];
  /** FILES the model generated (image/audio/…) — a parallel output channel, never folded into `value`.
   *  Each carries inline base64 `content` + `format` (media type) + a `contentHash`. */
  artifacts?: ProducedArtifact[];
  finishReason: string;
  metrics: CallMetrics;
  /** Present iff the call failed; `value`/`rawText`/`thinking` are then best-effort partials. */
  error?: CallFailure;
}

/**
 * A JSON Schema document, optionally BRANDED with the type `T` of the value it validates. The
 * brand (`__out`) is PHANTOM — never present at runtime, never read — so a plain
 * `Record<string, unknown>` is still a valid `JsonSchema<unknown>` (backward compatible), and any
 * `JsonSchema<T>` is still a plain record for the schema-transform/validation code. It exists only
 * to thread the output type from a call's `schema` through to `CallOutcome<T>.value`; the §4 Ajv
 * boundary is what actually ENFORCES conformance at runtime. Brand a plain schema with `typedSchema`.
 */
export type JsonSchema<T = unknown> = Record<string, unknown> & { readonly __out?: T };

/** Brand a plain JSON Schema document with the output type it produces — identity at runtime, so it
 *  can also be inlined. Lets `T` be INFERRED at the call site instead of spelled out as a type arg. */
export function typedSchema<T>(schema: Record<string, unknown>): JsonSchema<T> {
  return schema as JsonSchema<T>;
}

/**
 * The prompt inputs for a call, mirroring the AI SDK `Prompt` capability surface so no expressiveness is
 * lost: a `system` prompt (a plain string OR structured system message(s)) plus EITHER a `prompt` (a plain
 * string or a message array) OR a `messages` array — the latter two carry multi-turn conversation and
 * MULTIMODAL content (image/file parts live inside `ModelMessage`). Provide exactly ONE of
 * `prompt`/`messages`; both are optional at the type level so the shape threads cleanly through the layers,
 * and the SDK enforces the "one or the other" rule at the call. NB for the definition to stay serializable,
 * any file/image data inside messages must be a base64 string or URL (not a live `Uint8Array`).
 */
export interface CallPromptInput {
  system?: string | SystemModelMessage | SystemModelMessage[];
  prompt?: string | ModelMessage[];
  messages?: ModelMessage[];
  /** Neutral file/media inputs (pdf/image/audio/video) lowered to provider file parts + merged into the
   *  user turn at the call boundary (content-hash/path refs resolved via the injected blob store). */
  attachments?: FileInput[];
}

/** The plain-text content of a message's `content` (a string, or the text parts of a content array). */
function messageContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part && typeof part === "object" && typeof (part as { text?: unknown }).text === "string" ? (part as { text: string }).text : ""))
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

/** Normalize a prompt input to its MESSAGE-LIST form: explicit `messages` win, an array prompt IS the
 *  messages, a non-empty string prompt becomes one user turn, nothing ⇒ `[]`. The SINGLE implementation of
 *  the three-way prompt-shape branch (session transcripts, attachment lowering, repair hints). */
export function promptAsMessages(p: CallPromptInput): ModelMessage[] {
  if (p.messages) return p.messages;
  if (Array.isArray(p.prompt)) return p.prompt;
  if (typeof p.prompt === "string" && p.prompt.length > 0) return [{ role: "user", content: p.prompt }];
  return [];
}

/** Extract all plain text from a prompt input (system + prompt/messages) — used for the json-specifier
 *  check (§5.1) and cheap token estimation. Non-text parts (images/files/tool calls) contribute nothing. */
export function promptText(p: CallPromptInput): string {
  const parts: string[] = [];
  if (typeof p.system === "string") parts.push(p.system);
  else if (Array.isArray(p.system)) parts.push(p.system.map((m) => m.content).join("\n"));
  else if (p.system) parts.push(p.system.content);
  if (typeof p.prompt === "string") parts.push(p.prompt);
  else if (Array.isArray(p.prompt)) parts.push(p.prompt.map((m) => messageContentText(m.content)).join("\n"));
  if (p.messages) parts.push(p.messages.map((m) => messageContentText(m.content)).join("\n"));
  return parts.join("\n");
}

/**
 * The RESOLVED transport call `generateStructured` runs — the FLAT serializable decoding knobs reused
 * from core's `SamplingConfiguration` (temperature/topP/topK/penalties + base's maxOutputTokens/
 * stopSequences/seed) PLUS the pieces the executor reconstructs from a `StructuredCallParams` at the call
 * boundary: the live `model` handle (resolved from the config's `model` string via the router), the
 * provider-ADAPTED `schema` + `postProcess`, the merged `providerOptions` (raw config passthrough + the
 * neutral `reasoning` adapted at the boundary), the `abortSignal` (from `timeoutMs`), and the boundary
 * `validate`. It is deliberately distinct from the serializable `StructuredCallParams` — this one holds
 * live handles + closures and never persists — but reuses the decoding-knob surface (via
 * `SamplingConfiguration`, whose optional knobs are the flat superset once reasoning is adapted away)
 * instead of re-listing it. `model` (the config's string id) is replaced by the resolved handle, and the
 * neutral `reasoning` is replaced by `providerOptions`; both are re-supplied below.
 */
export interface GenerateStructuredParams<T = unknown>
  extends Omit<SamplingConfiguration, "model" | "providerOptions" | "tools" | "toolChoice" | "maxSteps">,
    CallPromptInput {
  /** The resolved model handle (the router turns the config's `model` string into this). */
  model: LanguageModel;
  /** The routing id — for pricing and provider detection. Kept alongside `model` because the handle's
   *  own id isn't a reliable pricing/routing key (OpenRouter prefixing, etc.). */
  modelId: string;
  /** The ORIGINAL/OUTGOING JSON Schema (reconstruction + validation target), branded with the output
   *  type `T`. OMITTED for a TEXT-output op (§3.14): no schema → plain `streamText`, the raw text IS
   *  the value (the user's text→text flow). */
  schema?: JsonSchema<T>;
  abortSignal?: AbortSignal;
  /** Merged provider options (raw config passthrough + adapted reasoning) forwarded to the SDK. */
  providerOptions?: Record<string, unknown>;
  /** The RESOLVED tool set (declarations from the config combined with runtime `execute` impls), keyed
   *  by name. The executor builds this; `generateStructured` just forwards it to the SDK. */
  tools?: ToolSet;
  /** How the model may select among `tools`. */
  toolChoice?: ToolChoice<ToolSet>;
  /** The stop condition bounding an executed tool loop (e.g. `stepCountIs(n)`); set only when tools have
   *  executors. Absent ⇒ a single step (tool calls are returned, not executed in a loop). */
  stopWhen?: StopCondition<ToolSet> | Array<StopCondition<ToolSet>>;
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

export async function generateStructured<T = unknown>(params: GenerateStructuredParams<T>): Promise<CallOutcome<T>> {
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
  // First-class tool calls/results (REAL tools only — never the structured-output emulation tool).
  const toolCalls: ToolCall[] = [];
  const toolResults: ToolResult[] = [];
  // Files the model generated (image/audio/…) — a parallel output channel.
  const producedFiles: ProducedArtifact[] = [];
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
  }): CallOutcome<T> => {
    // `value` is reconstructed as `unknown` (parsed JSON / postProcess output) and asserted to `T`
    // here, the single construction site — the §4 Ajv boundary is what makes that assertion sound.
    const outcome: CallOutcome<T> = {
      value: args.value as T | undefined,
      rawText: outputText(),
      thinking: thinking.length > 0 ? thinking : undefined,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      toolResults: toolResults.length > 0 ? toolResults : undefined,
      artifacts: producedFiles.length > 0 ? producedFiles : undefined,
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

  // Forward the prompt as the SDK's `Prompt`: `system` plus EXACTLY ONE of `messages`/`prompt` (the SDK
  // rejects both). `messages` wins when present; otherwise the `prompt` (string or message array).
  const promptPart: { messages: ModelMessage[] } | { prompt: string | ModelMessage[] } =
    params.messages !== undefined ? { messages: params.messages } : { prompt: params.prompt ?? "" };

  const result = streamText({
    model: params.model,
    system: params.system,
    ...promptPart,
    maxRetries: 0,
    abortSignal: params.abortSignal,
    maxOutputTokens: params.maxOutputTokens,
    temperature: params.temperature,
    topP: params.topP,
    topK: params.topK,
    presencePenalty: params.presencePenalty,
    frequencyPenalty: params.frequencyPenalty,
    seed: params.seed,
    stopSequences: params.stopSequences,
    providerOptions: params.providerOptions as never,
    ...(params.tools ? { tools: params.tools } : {}),
    ...(params.toolChoice ? { toolChoice: params.toolChoice } : {}),
    ...(params.stopWhen ? { stopWhen: params.stopWhen } : {}),
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
            toolCalls.push({ ...(id !== undefined ? { toolCallId: id } : {}), toolName, input });
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
              output: p.output ?? p.result,
            });
          }
          break;
        }
        case "file": {
          // A model-GENERATED file (image/audio/…). The high-level fullStream emits `{file: GeneratedFile}`;
          // be defensive about a low-level `{mediaType, data}` shape too. Captured as an inline artifact.
          const gf = p.file;
          const mediaType = gf?.mediaType ?? p.mediaType;
          const base64 = gf?.base64 ?? (typeof p.data === "string" ? p.data : undefined);
          if (base64 !== undefined && base64.length > 0 && mediaType !== undefined) {
            producedFiles.push({ name: `file-${producedFiles.length}`, content: base64, format: mediaType, contentHash: sha256Hex(base64) });
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
