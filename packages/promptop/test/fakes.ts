/**
 * Shared fakes for the promptop suites: a resolved `PromptOp` builder, a scripted `CallRunner`, and
 * the two canonical `LlmCallResult`s. The op — not a spec — is the payload now (DESIGN §3.1),
 * so building one is the setup step every test shares.
 */
import type { InlineFamily, JsonValue, ExecResult, ResolvedValue, PromptOp, SessionStore } from "@declarative-ai/exec";
import { MapSessionStore, isOk } from "@declarative-ai/exec";
import type { Failure } from "@declarative-ai/exec";
import type { ModelMessage } from "ai";
import type { LlmCallResult, LlmCallDefinition, LlmMetrics, LlmOutput } from "@declarative-ai/llm";
import type { CallDeps, CallRunner } from "../src/executor";

export const OUTPUT_SCHEMA = { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] } as const;

/** A resolved prompt op — every binding a literal, which is what an executor is handed. */
export function promptOp(over: Partial<PromptOp<InlineFamily>> = {}, config: Record<string, JsonValue> = {}): PromptOp<InlineFamily> {
  return {
    kind: "prompt",
    user: "What is 2+2?",
    system: "Answer tersely.",
    config: { model: "anthropic/claude-haiku-4-5", maxOutputTokens: 100, ...config },
    input: {},
    output: { name: "output", kind: "json", schema: OUTPUT_SCHEMA },
    ...over,
  };
}

const FAKE_METRICS = { inputTokens: 10, outputTokens: 5, costUsd: 0.001, costSource: "table" as const, durationMs: 20 };

/** A successful call. `over` patches the PAYLOAD (what the model produced); `error` and `metrics`
 *  patch the envelope. */
export function okOutcome(over: Partial<LlmOutput> & { error?: Failure; metrics?: Partial<LlmMetrics> } = {}): LlmCallResult {
  const { error, metrics: metricsOver, ...payload } = over;
  const value: LlmOutput = { parsed: { answer: "4" }, rawText: '{"answer":"4"}', finishReason: "stop", ...payload };
  const metrics: LlmMetrics = { ...FAKE_METRICS, ...metricsOver };
  return error ? { error, value, metrics } : { value, metrics };
}

export function validationFailure(errors = "data/answer must be string"): LlmCallResult {
  return {
    // The failure keeps its payload — the value that FAILED validation is what a repair turn reads.
    value: { parsed: { answer: 7 }, rawText: '{"answer":7}', finishReason: "stop" },
    metrics: FAKE_METRICS,
    error: {
      classification: "api-retriable",
      reason: `post-reconstruction validation failed: output failed schema validation: ${errors}`,
    },
  };
}

/** A fake runner that records every invocation and replays a scripted outcome sequence (the last
 *  outcome repeats once the script is exhausted). */
export function fakeRunner(script: LlmCallResult[]): {
  runner: CallRunner;
  calls: { def: LlmCallDefinition; env: CallDeps; timeoutMs?: number }[];
} {
  const calls: { def: LlmCallDefinition; env: CallDeps; timeoutMs?: number }[] = [];
  const runner: CallRunner = async (def, env, timeoutMs) => {
    calls.push({ def, env, timeoutMs });
    return script[Math.min(calls.length - 1, script.length - 1)]!;
  };
  return { runner, calls };
}

/** A trivial in-memory MemoCache. */
export function memoCache(): { cache: Map<string, ExecResult<ResolvedValue>>; get: (k: string) => ExecResult<ResolvedValue> | undefined; set: (k: string, v: ExecResult<ResolvedValue>) => void } {
  const cache = new Map<string, ExecResult<ResolvedValue>>();
  return { cache, get: (k) => cache.get(k), set: (k, v) => void cache.set(k, v) };
}

/**
 * A transcript store in its llm view (`ModelMessage` turns) plus the `ExecServices`-shaped seam view of
 * the SAME store. One store, two views: `ExecServices.sessions` is declared at the JSON base because it
 * is shared across consumers that pin different message shapes (hw stores `Turn`s, promptop
 * `ModelMessage`s), and each side reads exactly the messages it wrote.
 */
export function transcripts(): { store: MapSessionStore<ModelMessage>; seam: SessionStore } {
  const store = new MapSessionStore<ModelMessage>();
  return { store, seam: store as unknown as SessionStore };
}

/** Read a result's failure, or `undefined` when it succeeded — `error` is not a property of the union. */
export function errorOf<O, M extends { durationMs: number }>(r: ExecResult<O, M>): Failure | undefined {
  return isOk(r) ? undefined : r.error;
}
