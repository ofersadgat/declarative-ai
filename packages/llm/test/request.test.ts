import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, expectTypeOf, it } from "vitest";
import { type CallOutcome, typedSchema } from "../src/generate";
import { executeRequest, type LlmCallEnvironment } from "../src/llmStep";
import { fakeRouter, flatSchema, streamingModel, usage } from "./fakes";

function okModel(): MockLanguageModelV3 {
  return streamingModel([
    { type: "stream-start", warnings: [] },
    { type: "text-start", id: "1" },
    { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
    { type: "text-end", id: "1" },
    { type: "finish", finishReason: "stop", usage: usage(3, 2) },
  ]);
}

describe("executeRequest — declaration + env convenience", () => {
  it("splits the request and delegates to the base call", async () => {
    const env: LlmCallEnvironment = { modelRouter: fakeRouter(okModel()) };
    const out = await executeRequest({
      model: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: flatSchema,
      timeoutMs: 30_000,
      env,
    });
    expect(out.error).toBeUndefined();
    expect(out.value).toEqual({ answer: "4" });
  });

  it("throws when the environment has no provider router", async () => {
    await expect(
      executeRequest({ model: "m", prompt: "x", timeoutMs: 1000, env: {} }),
    ).rejects.toThrow(/env.modelRouter is required/);
  });

  it("threads a typed schema through to a typed outcome (no explicit <T> at the call site)", () => {
    interface Answer {
      answer: string;
    }
    // Type-level only — never executed. `T` is INFERRED from the branded `schema`, so the
    // outcome is `CallOutcome<Answer>` without spelling out `executeRequest<Answer>`. The
    // sampling-XOR-reasoning union in the config does NOT bury the inference site.
    const p = executeRequest({
      model: "anthropic/claude-haiku-4-5",
      prompt: "q",
      timeoutMs: 1000,
      schema: typedSchema<Answer>({ type: "object", properties: { answer: { type: "string" } }, required: ["answer"] }),
      env: {},
    }).catch(() => undefined);
    expectTypeOf(p).toEqualTypeOf<Promise<CallOutcome<Answer> | undefined>>();
  });
});
