/**
 * Reading an agent's stream — ONE mapping, both transports.
 *
 * The SDK and the CLI carry the SAME messages. The Agent SDK drives the binary as a subprocess and
 * hands its `stream-json` lines through with their field names untouched, so `{"type":"assistant",
 * "message":{…}}` arriving over an in-process iterator and the same object arriving as a line on stdout
 * are the same object. Two mappings would be two chances to drop the same field, and the field that got
 * dropped was every field: both paths collapsed everything but the terminal `result` to
 * `{type: "other"}`.
 *
 * Pure by design. The transports are the parts that cannot be tested here — one needs an optional peer
 * dependency, the other a subprocess — so everything that can be a function is one, and a captured
 * transcript replays against it directly.
 *
 * ✅ VERIFIED against `claude 2.1.142` and `@anthropic-ai/claude-agent-sdk 0.3.223` — a live run's own
 * lines, read back:
 *
 * ```
 * {"type":"system","subtype":"init","tools":[…],"mcp_servers":[…],"model":"…","permissionMode":"default"}
 * {"type":"assistant","message":{"content":[{"type":"text","text":"Hi!"}],"usage":{…}},"session_id":"…"}
 * {"type":"rate_limit_event","rate_limit_info":{…}}
 * {"type":"result","subtype":"success","is_error":false,"result":"Hi!","stop_reason":"end_turn",
 *  "session_id":"…","total_cost_usd":0.189,"usage":{"input_tokens":6,"cache_creation_input_tokens":30277,
 *  "cache_read_input_tokens":0,"output_tokens":8,"cache_creation":{"ephemeral_1h_input_tokens":30277,
 *  "ephemeral_5m_input_tokens":0}},"modelUsage":{…},"permission_denials":[]}
 * ```
 */
import type { JsonValue } from "@declarative-ai/exec";
import type { AgentReasoning, AgentStreamMessage, AgentTokenCounts, AgentToolCall, AgentToolResult } from "./seam.js";

/** A finite number, or nothing. Never a coerced zero — absent and zero are different claims. */
function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

/** A plain JSON object, or nothing. */
function bag(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

/** Sum the defined operands, or nothing when none were reported. */
function total(...parts: Array<number | undefined>): number | undefined {
  const present = parts.filter((p): p is number => p !== undefined);
  return present.length > 0 ? present.reduce((a, b) => a + b, 0) : undefined;
}

/**
 * Anthropic's usage object in the neutral vocabulary.
 *
 * The one reading that is easy to get backwards, and expensive when you do: **`input_tokens` is the
 * FRESH input only.** Cache reads and cache writes are reported alongside it, not inside it, so the
 * provider's billed input is the sum of the three. Treating `input_tokens` as the total under-reports
 * a cache-heavy call by an order of magnitude — the live run behind this file's header billed 6 fresh
 * input tokens and 30,277 cache-creation tokens.
 *
 * The TTL split is kept because it is priced separately: a 1-hour cache write is roughly twice the base
 * rate where a 5-minute one is roughly 1.25×.
 */
export function readAgentUsage(usage: unknown): AgentTokenCounts | undefined {
  const u = bag(usage);
  if (u === undefined) return undefined;
  const noCacheTokens = num(u["input_tokens"]);
  const cacheReadTokens = num(u["cache_read_input_tokens"]);
  const cacheWriteTokens = num(u["cache_creation_input_tokens"]);
  const outputTokens = num(u["output_tokens"]);
  const inputTokens = total(noCacheTokens, cacheReadTokens, cacheWriteTokens);
  const counts: AgentTokenCounts = {
    ...(inputTokens !== undefined ? { inputTokens } : {}),
    ...(outputTokens !== undefined ? { outputTokens } : {}),
    ...(noCacheTokens !== undefined ? { noCacheTokens } : {}),
    ...(cacheReadTokens !== undefined ? { cacheReadTokens } : {}),
    ...(cacheWriteTokens !== undefined ? { cacheWriteTokens } : {}),
    ...(num(bag(u["cache_creation"])?.["ephemeral_1h_input_tokens"]) !== undefined
      ? { cacheWrite1hTokens: num(bag(u["cache_creation"])!["ephemeral_1h_input_tokens"])! }
      : {}),
    ...(total(inputTokens, outputTokens) !== undefined ? { totalTokens: total(inputTokens, outputTokens)! } : {}),
  };
  return Object.keys(counts).length > 0 ? counts : undefined;
}

/**
 * Why the run ended, in the vocabulary a PROVIDER call reports — so the two are comparable.
 *
 * Two fields carry it and both are needed. `subtype` says whether the agent's own loop ran out of
 * something (turns, budget, retries); `stop_reason` says why the last model turn stopped. A run that
 * hit its turn cap reports `subtype: "error_max_turns"` with a perfectly ordinary `stop_reason`, so
 * reading only the second would report a truncated run as a clean stop — which is exactly what
 * hardcoding `"stop"` did, for every run.
 */
/**
 * Terminal reasons that mean the turn was ENDED DELIBERATELY, not that it broke.
 *
 * ✅ OBSERVED (claude 2.1.142, via a live `Query.interrupt()`): an interrupted turn comes back as
 * `{"is_error":true,"subtype":"error_during_execution","terminal_reason":"aborted_streaming",
 * "stop_reason":null,"result":<absent>}`. Read at face value that is a failed run — which is how an
 * interrupt would settle the handle with an error and discard an answer the agent had already written.
 * It is the same class of trap as `is_error` arriving under `subtype: "success"`, in the other
 * direction.
 */
const ABORTED_TERMINAL_REASONS = new Set(["aborted_streaming", "aborted_tools", "background_requested"]);

export function agentFinishReason(msg: Record<string, unknown>): string | undefined {
  // The deliberate end wins over the `subtype` that accompanies it: `error_during_execution` describes
  // how the loop exited, and here that exit was requested.
  if (typeof msg["terminal_reason"] === "string" && ABORTED_TERMINAL_REASONS.has(msg["terminal_reason"])) return "aborted";
  if (msg["terminal_reason"] === "max_turns") return "max-turns";
  if (msg["terminal_reason"] === "budget_exhausted") return "max-budget";
  switch (msg["subtype"]) {
    case "error_max_turns":
      return "max-turns";
    case "error_max_budget_usd":
      return "max-budget";
    case "error_during_execution":
    case "error_max_structured_output_retries":
      return "error";
    default:
      break;
  }
  switch (msg["stop_reason"]) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool-calls";
    case "refusal":
      return "content-filter";
    case null:
    case undefined:
      // `subtype: "success"` with nothing more specific: the loop finished on its own terms.
      return msg["subtype"] === "success" ? "stop" : undefined;
    default:
      return "unknown";
  }
}

/**
 * The content blocks of one turn, split into the three projections.
 *
 * A `thinking` block's SIGNATURE is preserved under `providerMetadata`, not dropped: a signed block
 * must come back byte-identical next turn or the provider rejects the conversation.
 */
function readContent(content: unknown): Pick<AgentStreamMessage, "text" | "thinking" | "toolCalls" | "toolResults"> {
  if (typeof content === "string") return content.length > 0 ? { text: content } : {};
  if (!Array.isArray(content)) return {};
  const thinking: AgentReasoning[] = [];
  const toolCalls: AgentToolCall[] = [];
  const toolResults: AgentToolResult[] = [];
  let text = "";
  for (const raw of content) {
    const block = bag(raw);
    if (block === undefined) continue;
    switch (block["type"]) {
      case "text":
        text += typeof block["text"] === "string" ? block["text"] : "";
        break;
      case "thinking":
      case "redacted_thinking": {
        // A redacted block has no readable text and MUST still round-trip: its `data` is the whole of
        // it, and dropping the block breaks the next turn rather than merely losing a trace.
        const meta: Record<string, JsonValue> = {};
        for (const key of ["signature", "data"]) {
          const value = str(block[key]);
          if (value !== undefined) meta[key] = value;
        }
        thinking.push({
          text: typeof block["thinking"] === "string" ? block["thinking"] : "",
          ...(Object.keys(meta).length > 0 ? { providerMetadata: { anthropic: meta } } : {}),
        });
        break;
      }
      case "tool_use":
        toolCalls.push({
          ...(str(block["id"]) !== undefined ? { toolCallId: str(block["id"])! } : {}),
          toolName: str(block["name"]) ?? "",
          input: (block["input"] ?? null) as JsonValue,
        });
        break;
      case "tool_result":
        toolResults.push({
          ...(str(block["tool_use_id"]) !== undefined ? { toolCallId: str(block["tool_use_id"])! } : {}),
          output: (block["content"] ?? null) as JsonValue,
        });
        break;
      default:
        break;
    }
  }
  return {
    ...(text.length > 0 ? { text } : {}),
    ...(thinking.length > 0 ? { thinking } : {}),
    ...(toolCalls.length > 0 ? { toolCalls } : {}),
    ...(toolResults.length > 0 ? { toolResults } : {}),
  };
}

/**
 * Agent failure codes that a RETRY could plausibly get past.
 *
 * ✅ The vocabulary is the SDK's own `SDKAssistantMessageError`: `authentication_failed`,
 * `oauth_org_not_allowed`, `billing_error`, `rate_limit`, `overloaded`, `invalid_request`,
 * `model_not_found`, `server_error`, `unknown`, `max_output_tokens`. Only three of them describe a
 * condition that changes on its own; the rest are facts about the account, the request, or the model,
 * and retrying them burns the retry budget to arrive at the same answer.
 *
 * Default-PERMANENT is deliberate. An unrecognised code from a later version retries zero times rather
 * than an unknown number of times against an unknown condition.
 */
export const RETRIABLE_AGENT_ERROR_CODES: ReadonlySet<string> = new Set(["rate_limit", "overloaded", "server_error"]);

/** Would retrying this failure plausibly help? Absent code ⇒ no, per the default above. */
export function isRetriableAgentError(code: string | undefined): boolean {
  return code !== undefined && RETRIABLE_AGENT_ERROR_CODES.has(code);
}

/** The text a `stream_event` adds, or nothing when the event is about something else. */
function readDelta(event: unknown): string | undefined {
  const e = bag(event);
  if (e?.["type"] !== "content_block_delta") return undefined;
  const delta = bag(e["delta"]);
  // Only the OUTPUT text. A `thinking_delta` is the agent reasoning aloud, and streaming it into
  // `output_partial` would put the reasoning into the answer — the one thing §5.1 says must never happen.
  return delta?.["type"] === "text_delta" ? str(delta["text"]) : undefined;
}

/**
 * Normalize ONE message off an agent's stream.
 *
 * Everything with no neutral home becomes a `provider_event` carrying the raw message, rather than
 * being discarded. That is the passthrough rule: `exec` must not learn Claude's vocabulary, and a host
 * that wants to render a compaction boundary or a rate-limit window must not be prevented from doing so
 * because the neutral layer had no name for it.
 */
export function readAgentMessage(msg: Record<string, unknown>): AgentStreamMessage {
  switch (msg["type"]) {
    case "result": {
      const text = str(msg["result"]) ?? str(msg["text"]) ?? "";
      // A failed run is reported IN the result message, not as an exception: `is_error: true` arrives
      // alongside `subtype: "success"`, carrying the failure text where the answer would be. Yielding it
      // as a result would hand the caller `Not logged in · Please run /login` as the agent's answer,
      // with `finishReason: "stop"` — indistinguishable from a run that found nothing.
      const usage = readAgentUsage(msg["usage"]);
      const finishReason = agentFinishReason(msg);
      // An INTERRUPTED run is not a failed one, whatever `is_error` says. The turn ended because
      // someone asked it to; the work it did up to that point is the answer, and it rides on the
      // assistant turns already accumulated rather than on `result`, which an aborted run leaves empty.
      // Settling this as a failure would discard exactly what the person pressing Stop wanted to see.
      if (msg["is_error"] === true && finishReason !== "aborted") {
        return { type: "other", error: text.length > 0 ? text : "the agent reported a failed run" };
      }
      return {
        type: "result",
        result: {
          text,
          // The schema-constrained answer, on its OWN field beside the prose — which is how both
          // transports report it when `--json-schema` / `outputFormat` asked for one. `result` stays a
          // summary of the work, so a caller that wanted the value must read this and not parse that.
          ...(msg["structured_output"] !== undefined ? { structured: msg["structured_output"] as JsonValue } : {}),
          ...(num(msg["total_cost_usd"]) !== undefined ? { costUsd: num(msg["total_cost_usd"])! } : {}),
          // The id this run ENDED in — a new one after a fork. Dropping it leaves the next call with no
          // handle to resume, which is the half of the session story that fails silently.
          ...(str(msg["session_id"]) !== undefined ? { sessionId: str(msg["session_id"])! } : {}),
          ...(finishReason !== undefined ? { finishReason } : {}),
          ...(usage !== undefined ? { usage } : {}),
          // `modelUsage` alongside `usage` deliberately: the first is per-model and covers subagents and
          // compaction, the second is the main loop only. Keeping the whole object is what makes
          // `costUsd` recomputable when our reading of either turns out to be incomplete.
          ...(msg["usage"] !== undefined || msg["modelUsage"] !== undefined
            ? { rawUsage: { ...(msg["usage"] !== undefined ? { usage: msg["usage"] } : {}), ...(msg["modelUsage"] !== undefined ? { modelUsage: msg["modelUsage"] } : {}) } as JsonValue }
            : {}),
        },
      };
    }
    case "assistant":
    case "user": {
      const inner = bag(msg["message"]);
      return {
        type: msg["type"] as "assistant" | "user",
        // ✅ OBSERVED on a not-logged-in run: the turn that carries the failure text also carries
        // `error: "authentication_failed"` and `message.model: "<synthetic>"`. The prose reaches the
        // caller either way — the terminal result repeats it — but the CODE is the only part that can
        // decide whether retrying is sound, and it appears HERE and nowhere else.
        ...(str(msg["error"]) !== undefined ? { errorCode: str(msg["error"])! } : {}),
        // VERBATIM. `LlmOutput.messages` is documented as the provider's own log rather than a
        // reconstruction, and a reconstruction is exactly what the synthesized one-turn log was.
        ...(inner !== undefined ? { message: inner as JsonValue } : {}),
        ...readContent(inner?.["content"]),
      };
    }
    case "stream_event": {
      const delta = readDelta(msg["event"]);
      return delta !== undefined ? { type: "partial", delta } : { type: "provider_event", event: msg as JsonValue };
    }
    default:
      // `system` in all its subtypes (init, compact_boundary, hook_started, task_progress, api_retry,
      // permission_denied, …), `rate_limit_event`, and whatever a later version adds.
      return { type: "provider_event", event: msg as JsonValue };
  }
}
