/**
 * Shared fakes for the @declarative-ai/llm test suites — the mock stream plumbing and a fake provider
 * router — so a change to the mock stream shape or the `ModelRouter` interface lands in ONE place.
 */
import { isOk, type Failure } from "@declarative-ai/json";
import type { LlmCallResult, LlmOutput } from "../src/output";
import { simulateReadableStream, type LanguageModel } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import type { JsonValue } from "@declarative-ai/json";
import { generateStructured, type GenerateEnvironment } from "../src/generate";
import type { LlmCallDefinition, ReasoningSpec, SamplingConfiguration } from "../src/llmConfig";
import type { CallSignature } from "../src/prompt";
import type { ModelRouter } from "../src/router";

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
export const fakeRouter = (model: MockLanguageModelV3): ModelRouter => ({
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

const FAKE_METRICS = { inputTokens: 10, outputTokens: 5, costUsd: 0.001, costSource: "table" as const, durationMs: 20 };

export function okOutcome(partial: Partial<LlmCallResult> = {}): LlmCallResult {
  return {
    value: { parsed: { answer: "4" }, rawText: '{"answer":"4"}', finishReason: "stop" },
    metrics: FAKE_METRICS,
    ...partial,
  };
}

export function validationFailure(errors = "data/answer must be string"): LlmCallResult {
  return {
    // The failure keeps its payload: the value that FAILED validation is what a repair turn reads.
    value: { parsed: { answer: 7 }, rawText: '{"answer":7}', finishReason: "stop" },
    metrics: FAKE_METRICS,
    error: {
      classification: "api-retriable",
      reason: `post-reconstruction validation failed: output failed schema validation: ${errors}`,
    },
  };
}

/**
 * The old FLAT call bag, kept as a TEST SHIM only. `generateStructured` now takes `(definition,
 * environment)` (DESIGN §4.1); these suites exercise transport behavior — streaming, usage
 * accounting, reconstruction, failure classification — not the split, so they keep their compact
 * literals and this one function does the splitting.
 */
type FlatDef<T> = Omit<Partial<SamplingConfiguration>, "model" | "tools" | "toolChoice"> & Partial<CallSignature<T>> & { reasoning?: ReasoningSpec };
type FlatCall<T> = FlatDef<T> & GenerateEnvironment<T> & { modelId?: string };

export function generateFlat<T = JsonValue>(flat: FlatCall<T>): Promise<LlmCallResult<T>> {
  const { model, modelId, outgoing, postProcess, validate, tools, toolChoice, stopWhen, providerOptions, attachStructuredOutput, accepts, abortSignal, ...def } =
    flat;
  return generateStructured<T>({ model: modelId ?? "anthropic/claude-haiku-4-5", ...def } as LlmCallDefinition<T>, {
    model,
    outgoing,
    postProcess,
    validate,
    tools,
    toolChoice,
    stopWhen,
    providerOptions,
    attachStructuredOutput,
    accepts,
    abortSignal,
  });
}

/** Read a call's failure, or `undefined` when it succeeded — `error` is not a property of the union. */
export function errorOf<T>(r: LlmCallResult<T>): Failure | undefined {
  return isOk(r) ? undefined : r.error;
}

/** The payload a call produced, on either branch (a failure keeps its partial). */
export function outputOf<T>(r: LlmCallResult<T>): LlmOutput<T> | undefined {
  return r.value;
}
