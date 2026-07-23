import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, expectTypeOf, it } from "vitest";
import { typedSchema } from "../src/generate";
import type { LlmCallResult } from "../src/output";
import { executeRequest, type LlmCallEnvironment } from "../src/call";
import { fakeRouter, flatSchema, streamingModel, usage, errorOf } from "./fakes";

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
    const { JSON_OBJECT } = await import("../src/schema");
    const { executeLlmCall } = await import("../src/call");
    const seen: string[] = [];
    // An advisory profile that REQUIRES the word "json" in the prompt: with no "json" anywhere the
    // call must fail fast (permanent) BEFORE hitting the provider. The default (catalog) profile for
    // this model is the strict Anthropic tier, which never fails this way — so reaching the failure
    // proves the injected resolver was consulted.
    const out = await executeLlmCall(
      { model: "anthropic/claude-haiku-4-5", prompt: "what is 2+2?", schema: flatSchema, timeoutMs: 5000 },
      {
        modelRouter: fakeRouter(okModel()),
        schemaProfile: (modelId) => {
          seen.push(modelId);
          return { ...JSON_OBJECT, promptRequiresJSONSpecifier: true };
        },
      },
    );
    expect(seen).toEqual(["anthropic/claude-haiku-4-5"]);
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/promptRequiresJSONSpecifier/);
  });
});
