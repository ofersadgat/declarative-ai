/**
 * What a model call PRODUCES, and what it cost.
 *
 * All of this used to live in `@declarative-ai/json` — `ReasoningSegment`, `ToolCall`, `ToolResult`,
 * and the token half of `Metrics` — because the old execution `Outcome` named them all at once and
 * `json` was the only package both `llm` and `exec` could see. With that `Outcome` gone, nothing below
 * this package refers to any of it: a reasoning trace, a tool call, and a token count are provider
 * vocabulary, and `json` is a package about JSON.
 *
 * The shape is payload-and-envelope: {@link LlmOutput} is what the call produced, and it rides inside
 * the SAME `Result` envelope every layer uses. That is what removed the third near-identical result
 * record — `CallOutcome` had its own `CallFailure` (field-for-field identical to `Failure`) and its own
 * metrics, so a provider failure had to be re-classified on its way anywhere useful.
 */
import type { Failure, JsonValue, ResultWithMetrics } from "@declarative-ai/json";
import type { GeneratedFile } from "./files";

/**
 * One non-output segment of the trace, positioned against the output text it accompanies. Everything
 * the model emits that ISN'T the structured output is captured here with a `type` discriminator, so a
 * caller (or a human) can tell model reasoning apart from intermediate tool usage — and so none of it
 * is ever mistaken for the output (§5.1):
 *  - `"reasoning"` — an extended-thinking block; `text` is the thinking.
 *  - `"tool-call"` — an INTERMEDIATE tool the model invoked mid-generation; `text` is the tool's
 *    serialized args and `toolName` names it. This is NOT the structured-output (jsonTool) call —
 *    that one becomes the output, never a segment.
 */
export interface ReasoningSegment {
  type: "reasoning" | "tool-call";
  text: string;
  /** Length of the accumulated OUTPUT text when this segment began. */
  textOffset: number;
  /** Tool name, for `type:"tool-call"` segments. */
  toolName?: string;
  /** Native provider metadata (e.g. an Anthropic thinking-block signature), preserved intact —
   *  provider ground truth is open by nature but JSON by construction (§2.2). */
  providerMetadata?: Record<string, JsonValue>;
}

/** A tool invocation the model requested. For a single-turn (unexecuted) tool op these ARE the primary
 *  output the caller acts on; in an executed loop they are the intermediate calls. */
export interface ToolCall {
  toolCallId?: string;
  toolName: string;
  /** The parsed tool input the model produced (validated against the tool's input schema). */
  input: JsonValue;
}

/** The result of an EXECUTED tool call, fed back to the model during a tool loop. */
export interface ToolResult {
  toolCallId?: string;
  toolName?: string;
  output: JsonValue;
}

/** Token counts pulled off a provider usage object. */
export interface TokenCounts {
  /** Total input tokens, INCLUDING cache reads + writes (the provider's billed input). */
  inputTokens?: number;
  /** Total output tokens, including reasoning tokens. */
  outputTokens?: number;
  /** Uncached (fresh) input tokens — billed at the base input rate. */
  noCacheTokens?: number;
  /** Cached input tokens read on a cache hit — billed at the discounted read rate (~0.1x). */
  cacheReadTokens?: number;
  /** Input tokens written to the prompt cache — billed at the write rate (~1.25x for 5-min). */
  cacheWriteTokens?: number;
  /** 1-hour-TTL subset of `cacheWriteTokens` — billed at the ~2x tier. */
  cacheWrite1hTokens?: number;
  /** Reasoning/thinking output tokens (subset of `outputTokens`). */
  reasoningTokens?: number;
  /** Provider-reported grand total, when present. */
  totalTokens?: number;
}

/**
 * What ONE provider call cost.
 *
 * A flat record that SATISFIES `exec`'s `ExecMetrics` (`durationMs`, `startMs?`) and its
 * `BudgetMetrics` (`costUsd`, `costSource`) structurally, without importing either — which is what
 * keeps this package's dependency list at `@declarative-ai/json` alone. A budget-aware wrapper
 * constrains `M extends BudgetMetrics` and this satisfies the constraint; `exec` itself only ever sees
 * the timing fields and merges the rest opaquely.
 */
export interface LlmMetrics extends TokenCounts {
  durationMs: number;
  startMs?: number;
  /** USD this call cost. Required: "free" and "unknown" are different claims, and `costSource` carries
   *  the second one. */
  costUsd: number;
  /** The provider's actual charge (authoritative), our price table, or no better than a guess. */
  costSource: "provider" | "table" | "unknown";
  /** Provider's exact usage object (ground truth) — kept so `costUsd` is always recomputable. Open by
   *  nature but JSON by construction (§2.2). */
  rawUsage?: JsonValue;
}

/** How much a cost figure can be trusted: the provider's own charge beats our price table, which beats
 *  a guess. The order the {@link mergeLlmMetrics} fold keeps. */
const COST_SOURCE_RANK: Record<LlmMetrics["costSource"], number> = { provider: 2, table: 1, unknown: 0 };

/** Sum two calls' measurements: tokens and money add, the trace-level facts take the latest, the start
 *  time is the FIRST observation. This is the algebra a prompt executor registers, so exec can
 *  aggregate retry attempts without knowing what any of these fields mean. */
export function mergeLlmMetrics(a: LlmMetrics, b: LlmMetrics): LlmMetrics {
  const sum = (x: number | undefined, y: number | undefined): number | undefined =>
    x === undefined && y === undefined ? undefined : (x ?? 0) + (y ?? 0);
  return {
    durationMs: a.durationMs + b.durationMs,
    startMs: a.startMs ?? b.startMs,
    costUsd: a.costUsd + b.costUsd,
    // The MORE AUTHORITATIVE of the two, not the latest. Taking `b`'s unconditionally let a retry
    // relabel real spend as un-priced — merging a billed `{0.004, "table"}` with an attempt that
    // measured nothing (`{0, "unknown"}`) yielded `{0.004, "unknown"}`, destroying the provenance a
    // settling budget reads to know whether the number it is charging is real. Rank keeps the best
    // evidence behind the summed figure; an all-`unknown` fold still says `unknown`, which is honest.
    costSource: COST_SOURCE_RANK[a.costSource] >= COST_SOURCE_RANK[b.costSource] ? a.costSource : b.costSource,
    rawUsage: b.rawUsage ?? a.rawUsage,
    inputTokens: sum(a.inputTokens, b.inputTokens),
    outputTokens: sum(a.outputTokens, b.outputTokens),
    noCacheTokens: sum(a.noCacheTokens, b.noCacheTokens),
    cacheReadTokens: sum(a.cacheReadTokens, b.cacheReadTokens),
    cacheWriteTokens: sum(a.cacheWriteTokens, b.cacheWriteTokens),
    cacheWrite1hTokens: sum(a.cacheWrite1hTokens, b.cacheWrite1hTokens),
    reasoningTokens: sum(a.reasoningTokens, b.reasoningTokens),
    totalTokens: sum(a.totalTokens, b.totalTokens),
  };
}

/**
 * What a model call produced — the payload, not the envelope.
 *
 * Every field here is meaningless for a function op, which is exactly why none of it sits on the
 * execution result any more. `promptop` reads this to fill the op's output parameter and does not
 * forward it; a caller who wants the trace asks the llm layer, which is the layer that has it.
 */
export interface LlmOutput<T = JsonValue> {
  /**
   * The call's output value: the parsed (and post-processed, decoded) structured value, or in TEXT mode
   * the output text itself. Absent only when the model produced nothing usable — and then the raw text
   * it DID produce travels on the failure ({@link LlmFailure.rawOutput}), because unusable output is
   * diagnostic evidence, not a value. There is deliberately no separate `rawText` field: on success it
   * was informationally the value again (its wire form), and carrying both invited them to drift.
   *
   * Reads nest through the envelope as `result.value.value` — the envelope's `value` is this payload,
   * and the payload's `value` is the op's output value.
   */
  value?: T;
  thinking?: ReasoningSegment[];
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
  /** FILES the model generated (image/audio/…). They land in a `blob`-kind output parameter; there is
   *  deliberately no parallel `artifacts` channel on the execution result (DESIGN §3.7). */
  files?: GeneratedFile[];
  finishReason: string;
  /** Provider-assigned handle to resume from, when the provider is stateful. This is the ONLY honest
   *  session field: it is something the call PRODUCED. The execution envelope used to carry a
   *  `session.id` that was just the caller's own logical key echoed back, read by nobody. */
  providerSessionId?: string;
}

/**
 * The llm layer's failure: the shared classified {@link Failure}, plus the raw output text of a call
 * whose result could not become a value — truncated, unparseable, or empty structured output. It rides
 * the FAILURE because that is the only case it is not redundant: a successful parse IS the raw text,
 * decoded. A plain `Failure` is structurally an `LlmFailure`, so failures built by lower layers
 * (fail-fast, transport errors) need nothing extra.
 */
export interface LlmFailure extends Failure {
  /** What the model actually emitted, when the failure is that it wasn't usable. */
  rawOutput?: string;
}

/** One call's estimated token footprint, input/output split — what rate pre-admission is priced on.
 *  Structurally the same shape `exec`'s rate-limit seam names, satisfied without importing it. */
export interface CallTokenEstimate {
  inputTokens: number;
  outputTokens: number;
}

export interface CallEstimate extends CallTokenEstimate {
  modelId?: string;
}

/** One provider call's result: the shared envelope, this layer's payload, this layer's failure and
 *  metrics. */
export type LlmCallResult<T = JsonValue> = ResultWithMetrics<LlmOutput<T>, LlmFailure, LlmMetrics>;
