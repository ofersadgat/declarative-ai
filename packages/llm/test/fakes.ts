/**
 * Shared fakes for the @declarative-ai/llm test suites — the mock stream plumbing, a fake provider
 * router, scripted `CallRunner`s, and `ExecutionSpec` builders — so a change to the mock stream shape,
 * the `ProviderRouter` interface, or the spec/outcome shapes lands in ONE place instead of five files.
 */
import { simulateReadableStream, type LanguageModel } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import type { ExecutionSpec, Outcome } from "@declarative-ai/core";
import type { CallOutcome } from "../src/generate";
import type { CallRunner, CallRunnerDeps, LlmCallDefinition } from "../src/executor";
import type { StructuredCallParams } from "../src/llmStep";
import type { ProviderRouter } from "../src/router";

/** AI SDK 6 `LanguageModelV3Usage`: token totals nest under `.total`. */
export const usage = (input: number, output: number): Record<string, unknown> => ({
  inputTokens: { total: input },
  outputTokens: { total: output },
});

/** Wrap raw chunk objects in the SDK's simulated readable stream. */
export const stream = (chunks: Record<string, unknown>[]): { stream: ReadableStream<never> } => ({
  stream: simulateReadableStream({ chunks: chunks as never }),
});

/** A mock model that streams the given chunks. */
export function streamingModel(chunks: Record<string, unknown>[]): MockLanguageModelV3 {
  return new MockLanguageModelV3({ doStream: async () => stream(chunks) });
}

/** The one flat test schema most suites validate against. */
export const flatSchema = { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] };

/** A router that always resolves to the given mock model. */
export const fakeRouter = (model: MockLanguageModelV3): ProviderRouter => ({
  resolveModel: () => model as unknown as LanguageModel,
  isAnthropic: () => false,
});

/** The default test declaration. */
export const DEF: LlmCallDefinition = {
  model: "anthropic/claude-haiku-4-5",
  prompt: "What is 2+2?",
  system: "Answer tersely.",
  maxOutputTokens: 100,
  timeoutMs: 30_000,
};

export function specOf(overrides: Partial<ExecutionSpec> = {}, def: LlmCallDefinition = DEF): ExecutionSpec {
  return {
    kind: "llm-call",
    definition: def,
    inputs: {},
    outputSchema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
    ...overrides,
  };
}

export function okOutcome(partial: Partial<CallOutcome> = {}): CallOutcome {
  return {
    value: { answer: "4" },
    rawText: '{"answer":"4"}',
    finishReason: "stop",
    metrics: { inputTokens: 10, outputTokens: 5, cost: 0.001, costSource: "table", durationMs: 20 },
    ...partial,
  };
}

export function validationFailure(errors = "data/answer must be string"): CallOutcome {
  return {
    value: { answer: 7 },
    rawText: '{"answer":7}',
    finishReason: "stop",
    metrics: { inputTokens: 10, outputTokens: 5, cost: 0.001, costSource: "table", durationMs: 20 },
    error: {
      classification: "api-retriable",
      reason: `post-reconstruction validation failed: output failed schema validation: ${errors}`,
    },
  };
}

/** A fake runner that records every invocation and replays a scripted outcome sequence (the last outcome
 *  repeats once the script is exhausted). */
export function fakeRunner(script: CallOutcome[]): { runner: CallRunner; calls: { params: StructuredCallParams; deps: CallRunnerDeps }[] } {
  const calls: { params: StructuredCallParams; deps: CallRunnerDeps }[] = [];
  const runner: CallRunner = async (params, deps) => {
    calls.push({ params, deps });
    const next = script[Math.min(calls.length - 1, script.length - 1)];
    return next!;
  };
  return { runner, calls };
}

/** A trivial in-memory MemoCache. */
export function memoCache(): { cache: Map<string, Outcome>; get: (k: string) => Outcome | undefined; set: (k: string, v: Outcome) => void } {
  const cache = new Map<string, Outcome>();
  return { cache, get: (k) => cache.get(k), set: (k, v) => void cache.set(k, v) };
}
