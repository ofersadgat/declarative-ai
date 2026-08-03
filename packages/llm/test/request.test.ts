import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, expectTypeOf, it } from "vitest";
import { typedSchema } from "../src/generate.js";
import type { LlmCallResult } from "../src/output.js";
import { executeRequest, type LlmCallEnvironment } from "../src/call.js";
import { fakeRouter, flatSchema, stream, streamingModel, usage, errorOf } from "./fakes.js";

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
    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.value).toEqual({ answer: "4" });
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
    // outcome is `LlmCallResult<Answer>` without spelling out `executeRequest<Answer>`. The
    // sampling-XOR-reasoning union in the config does NOT bury the inference site.
    const p = executeRequest({
      model: "anthropic/claude-haiku-4-5",
      prompt: "q",
      timeoutMs: 1000,
      schema: typedSchema<Answer>({ type: "object", properties: { answer: { type: "string" } }, required: ["answer"] }),
      env: {},
    }).catch(() => undefined);
    expectTypeOf(p).toEqualTypeOf<Promise<LlmCallResult<Answer> | undefined>>();
  });

  it("a SCHEMA-LESS request is typed as a TEXT call — it yields `string` (§5.2)", async () => {
    // The overload lived on `executeLlmCall` but not here, so this was the one remaining path where a
    // text call's `parsed` was a runtime string typed `JsonValue`. Not unsound — just a claim the types
    // did not express. The discrimination is on the ABSENCE of a schema, so nothing is asserted.
    const out = await executeRequest({
      model: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      timeoutMs: 30_000,
      env: {
        modelRouter: fakeRouter(
          streamingModel([
            { type: "stream-start", warnings: [] },
            { type: "text-start", id: "1" },
            { type: "text-delta", id: "1", delta: "four" },
            { type: "text-end", id: "1" },
            { type: "finish", finishReason: "stop", usage: usage(1, 1) },
          ]),
        ),
      },
    });
    expectTypeOf(out).toEqualTypeOf<LlmCallResult<string>>();
    const value: string | undefined = out.value?.value;
    expect(value).toBe("four");
  });
});

describe("executeLlmCall — injectable schema-profile resolution", () => {
  it("uses env.schemaProfile over the built-in catalog profile", async () => {
    const { ADVISORY } = await import("../src/schema/index.js");
    const { executeLlmCall } = await import("../src/call.js");
    const seen: string[] = [];
    // The catalog default for this model is the STRICT Anthropic tier, which puts the schema on the
    // wire and leaves the system prompt alone. Injecting the TEXT floor instead flips both: no
    // `responseFormat`, and the schema described in the prompt. Observing that flip is what proves the
    // injected resolver was consulted rather than the catalog.
    let sent: { responseFormat?: unknown; prompt?: unknown } = {};
    const model = new MockLanguageModelV3({
      doStream: async (options) => {
        sent = options as typeof sent;
        return stream([
          { type: "text-start", id: "1" },
          { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
          { type: "text-end", id: "1" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]);
      },
    });
    const out = await executeLlmCall(
      { model: "anthropic/claude-haiku-4-5", prompt: "what is 2+2?", schema: flatSchema, timeoutMs: 5000 },
      {
        modelRouter: fakeRouter(model),
        schemaProfile: (modelId) => {
          seen.push(modelId);
          return ADVISORY;
        },
      },
    );
    expect(seen).toEqual(["anthropic/claude-haiku-4-5"]);
    expect(errorOf(out)).toBeUndefined();
    expect(sent.responseFormat).toBeUndefined(); // text tier sends none
    expect(JSON.stringify(sent.prompt)).toMatch(/conforming to this JSON Schema/);
  });
});
