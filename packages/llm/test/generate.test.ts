import { MockLanguageModelV3 } from "ai/test";
import { beforeAll, describe, expect, expectTypeOf, it } from "vitest";
import type { JsonValue } from "@declarative-ai/json";
import { typedSchema } from "../src/generate";
import { ModelInfo } from "../src/model-catalog";
import { flatSchema, generateFlat, stream, streamingModel, usage, errorOf } from "./fakes";

// Pin the process-wide catalog to fixed rates so the cost assertions below are deterministic and
// decoupled from the live snapshot (`DEFAULT_MODELS`), which `npm run update:model-info` can re-price.
// (Vitest isolates module state per test file, so this override doesn't leak to other suites.)
beforeAll(() => {
  ModelInfo.instance = new ModelInfo([
    { route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 },
    { route: "openrouter", model: "openai/gpt-4.1-mini", inputPerMillion: 0.4, outputPerMillion: 1.6 },
  ]);
});

function throwingModel(err: unknown): MockLanguageModelV3 {
  return new MockLanguageModelV3({
    doStream: async () => {
      throw err;
    },
  });
}

// A usage object carrying the Anthropic cache split (noCache + cacheRead + cacheWrite) plus
// the provider `raw` usage (ground truth, incl. cache-write TTL detail the split doesn't carry).
const cachedUsage = (noCache: number, cacheRead: number, cacheWrite: number, output: number) => ({
  inputTokens: {
    total: noCache + cacheRead + cacheWrite,
    noCache,
    cacheRead,
    cacheWrite,
  },
  outputTokens: { total: output },
  raw: {
    input_tokens: noCache,
    cache_read_input_tokens: cacheRead,
    cache_creation_input_tokens: cacheWrite,
    cache_creation: { ephemeral_5m_input_tokens: cacheWrite, ephemeral_1h_input_tokens: 0 },
    output_tokens: output,
  },
});

describe("generateStructured (§5.1) — streaming structured call + metrics", () => {
  it("returns the parsed value + raw text + costed metrics, no error, on a normal finish", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(10, 5) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.parsed).toEqual({ answer: "4" });
    expect(out.value?.rawText).toBe('{"answer":"4"}');
    expect(out.metrics.inputTokens).toBe(10);
    expect(out.metrics.outputTokens).toBe(5);
    expect(out.metrics.costUsd).toBeCloseTo(0.000035, 12); // claude-haiku-4-5 $1/$5: (10*1 + 5*5)/1e6
  });

  it("captures the cache/reasoning token breakdown and prices it cache-aware", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        // 1M fresh + 1M cache-read input, 0 output.
        { type: "finish", finishReason: "stop", usage: cachedUsage(1_000_000, 1_000_000, 0, 0) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.metrics.inputTokens).toBe(2_000_000); // cache-inclusive total
    expect(out.metrics.noCacheTokens).toBe(1_000_000);
    expect(out.metrics.cacheReadTokens).toBe(1_000_000);
    // claude-haiku-4-5 $1/M base; cache read at 0.1x => 1.0 + 0.1 = 1.1 (NOT 2 * 1.0 = 2.0).
    expect(out.metrics.costUsd).toBeCloseTo(1.1, 10);
    // Provider ground truth is preserved for retroactive recompute (TTL tiers, rate fixes).
    expect((out.metrics.rawUsage as { cache_read_input_tokens?: number }).cache_read_input_tokens).toBe(
      1_000_000,
    );
  });

  it("prices the 1-hour cache-write tier from raw usage (2x, not 1.25x)", async () => {
    // 1M cache writes, of which 1M are 1-hour TTL — the raw `cache_creation` split carries it.
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        {
          type: "finish",
          finishReason: "stop",
          usage: {
            inputTokens: { total: 1_000_000, noCache: 0, cacheRead: 0, cacheWrite: 1_000_000 },
            outputTokens: { total: 0 },
            raw: {
              input_tokens: 0,
              cache_creation_input_tokens: 1_000_000,
              cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 1_000_000 },
              output_tokens: 0,
            },
          },
        },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.metrics.cacheWriteTokens).toBe(1_000_000);
    expect(out.metrics.cacheWrite1hTokens).toBe(1_000_000); // recovered from raw.cache_creation
    // claude-haiku-4-5 $1/M base; 1-hour write at 2x => 2.0 (NOT the 5-min 1.25x => 1.25).
    expect(out.metrics.costUsd).toBeCloseTo(1 * 2.0, 10);
  });

  it("prefers OpenRouter's reported cost over the price-table estimate", async () => {
    // OpenRouter usage accounting surfaces the ACTUAL charge in providerMetadata.openrouter.
    // generateStructured must use that (authoritative), not tokens × our table rate.
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        {
          type: "finish",
          finishReason: "stop",
          usage: usage(1000, 1000),
          providerMetadata: { openrouter: { usage: { cost: 0.0123, completionTokens: 1000 } } },
        },
      ]),
      modelId: "openrouter/openai/gpt-4.1-mini",
      prompt: "x",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.metrics.costUsd).toBe(0.0123); // provider charge, NOT (1000*0.4 + 1000*1.6)/1e6
    expect(out.metrics.costSource).toBe("provider");
  });

  it("falls back to the price table (costSource=table) when the provider reports no cost", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(10, 5) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
    });

    expect(out.metrics.costUsd).toBeCloseTo((10 * 1 + 5 * 5) / 1e6, 12); // claude-haiku-4-5 $1/$5
    expect(out.metrics.costSource).toBe("table");
  });

  it("preserves partial output text on a truncated (unparseable) body, classified api-retriable", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "length", usage: usage(7, 99) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "ramble",
      schema: flatSchema,
    });

    expect(errorOf(out)?.classification).toBe("api-retriable");
    expect(errorOf(out)?.reason).toMatch(/unparseable|truncated|empty/);
    expect(out.value?.rawText).toBe('{"answer":'); // partial output preserved
    expect(out.value?.parsed).toBeUndefined();
  });

  it("classifies a thrown 429 as transient", async () => {
    const out = await generateFlat({
      model: throwingModel(Object.assign(new Error("rate limited"), { status: 429 })),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
    });
    expect(errorOf(out)?.classification).toBe("network-retriable");
  });

  it("preserves the parsed value even when post-reconstruction validation fails (§4)", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(1, 1) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
      validate: () => {
        throw new Error("does not satisfy original schema");
      },
    });

    expect(errorOf(out)?.classification).toBe("api-retriable");
    expect(errorOf(out)?.reason).toContain("validation failed");
    expect(out.value?.parsed).toEqual({ answer: "4" }); // value preserved despite the failure
  });

  it("parses structured output delivered via a json tool call (no text-delta)", async () => {
    // The jsonTool emulation path (§5.1): some providers surface structured output as a
    // forced `json` tool call rather than text. The SDK streams the args as tool-input-delta
    // and hands back the parsed/serialized args on the tool-call part. generateStructured
    // must resolve the value from that path transparently.
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "tool-input-start", id: "t1", toolName: "json" },
        { type: "tool-input-delta", id: "t1", delta: '{"answer":"4"}' },
        { type: "tool-input-end", id: "t1" },
        { type: "tool-call", toolCallId: "t1", toolName: "json", input: '{"answer":"4"}' },
        { type: "finish", finishReason: "tool-calls", usage: usage(8, 6) },
      ]),
      modelId: "openrouter/openai/gpt-4.1-mini",
      prompt: "what is 2+2?",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.parsed).toEqual({ answer: "4" });
    expect(out.value?.rawText).toBe('{"answer":"4"}'); // partial salvage works on the tool path too
  });

  it("captures reasoning as positioned segments, not a single blob", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "reasoning-start", id: "r1" },
        { type: "reasoning-delta", id: "r1", delta: "2 plus 2 " },
        { type: "reasoning-delta", id: "r1", delta: "is 4" },
        { type: "reasoning-end", id: "r1" },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(3, 2) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.thinking).toEqual([{ type: "reasoning", text: "2 plus 2 is 4", textOffset: 0 }]);
    expect(out.value?.parsed).toEqual({ answer: "4" });
  });

  it("does NOT use an intermediate tool's args as output; captures them in the trace", async () => {
    // A v2-style candidate that calls a real tool mid-generation, THEN emits the structured
    // output as text. The intermediate tool's args must never become the output — they belong
    // in the reasoning trace as a type:"tool-call" segment.
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "tool-input-start", id: "calc1", toolName: "calculator" },
        { type: "tool-input-delta", id: "calc1", delta: '{"expr":"2+2"}' },
        { type: "tool-input-end", id: "calc1" },
        { type: "tool-call", toolCallId: "calc1", toolName: "calculator", input: '{"expr":"2+2"}' },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(12, 7) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2? use the calculator.",
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    // Output is the TEXT channel, not the calculator's args.
    expect(out.value?.parsed).toEqual({ answer: "4" });
    expect(out.value?.rawText).toBe('{"answer":"4"}');
    // The intermediate tool is preserved as a discriminated trace segment, not output.
    expect(out.value?.thinking).toEqual([
      { type: "tool-call", text: '{"expr":"2+2"}', textOffset: 0, toolName: "calculator" },
    ]);
  });

  it("backfills toolName onto the trace block when args stream before the tool is named", async () => {
    // tool-input-start is absent: deltas arrive first, the name only on tool-call. toolName
    // must still end up as top-level data on the block.
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "tool-input-delta", id: "x", delta: '{"q":"hi"}' },
        { type: "tool-call", toolCallId: "x", toolName: "search", input: '{"q":"hi"}' },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(5, 5) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
    });

    expect(out.value?.parsed).toEqual({ answer: "4" });
    expect(out.value?.thinking).toEqual([
      { type: "tool-call", text: '{"q":"hi"}', textOffset: 0, toolName: "search" },
    ]);
  });
});

describe("generateStructured — call-capability passthrough", () => {
  it("forwards presencePenalty / frequencyPenalty / seed to the underlying call", async () => {
    let captured: Record<string, unknown> | undefined;
    const model = new MockLanguageModelV3({
      doStream: async (options) => {
        captured = options as unknown as Record<string, unknown>;
        return stream([
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "1" },
          { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
          { type: "text-end", id: "1" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]);
      },
    });

    const out = await generateFlat({
      model,
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "x",
      schema: flatSchema,
      presencePenalty: 0.5,
      frequencyPenalty: 0.25,
      seed: 42,
    });

    expect(errorOf(out)).toBeUndefined();
    // The new decoding knobs from core's LlmConfiguration reach the SDK call unchanged.
    expect(captured?.presencePenalty).toBe(0.5);
    expect(captured?.frequencyPenalty).toBe(0.25);
    expect(captured?.seed).toBe(42);
  });

  it("forwards a structured system + a multimodal message array (full Prompt expressiveness)", async () => {
    let captured: Record<string, unknown> | undefined;
    const model = new MockLanguageModelV3({
      doStream: async (options) => {
        captured = options as unknown as Record<string, unknown>;
        return stream([
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "1" },
          { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
          { type: "text-end", id: "1" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]);
      },
    });

    const out = await generateFlat({
      model,
      modelId: "anthropic/claude-haiku-4-5",
      system: [{ role: "system", content: "be terse" }],
      messages: [{ role: "user", content: [{ type: "text", text: "what is 2+2?" }] }],
      schema: flatSchema,
    });

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.parsed).toEqual({ answer: "4" });
    // The SDK lowered our structured system + message array to the provider prompt without dropping either.
    const wire = JSON.stringify(captured?.prompt);
    expect(wire).toContain("be terse");
    expect(wire).toContain("what is 2+2?");
  });
});

describe("generateStructured — typed output", () => {
  interface Answer {
    answer: string;
  }

  it("threads the schema's output type through to LlmCallResult.value (typedSchema infers T)", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(3, 2) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: typedSchema<Answer>(flatSchema),
    });

    // Compile-time: `T` flows from the branded schema to the value (enforced by `npm run typecheck`).
    expectTypeOf(out.value?.parsed).toEqualTypeOf<Answer | undefined>();
    // Runtime: the value is still the parsed object.
    expect(out.value?.parsed?.answer).toBe("4");
  });

  it("an unbranded schema yields the JsonValue default, never unknown (§2.2)", async () => {
    const out = await generateFlat({
      model: streamingModel([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
        { type: "text-end", id: "1" },
        { type: "finish", finishReason: "stop", usage: usage(3, 2) },
      ]),
      modelId: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: flatSchema, // a plain schema document — no phantom brand
    });

    // A dynamically-built call still yields JSON, never "anything" — the §2.2 generic-default rule.
    expectTypeOf(out.value?.parsed).toEqualTypeOf<JsonValue | undefined>();
    expect(out.value?.parsed).toEqual({ answer: "4" });
  });
});

describe("a generated FILE part never sinks the primary output", () => {
  const textThenFile = (data: string) => [
    { type: "text-start", id: "t" },
    { type: "text-delta", id: "t", delta: '{"answer":"42"}' },
    { type: "text-end", id: "t" },
    { type: "file", mediaType: "image/png", data },
    { type: "finish", finishReason: "stop", usage: usage(10, 5) },
  ];

  it("decodes standard base64 into bytes", async () => {
    const out = await generateFlat({ model: streamingModel(textThenFile("aGVsbG8=")), schema: flatSchema, prompt: "x" });
    expect(out.value?.parsed).toEqual({ answer: "42" });
    expect(out.value?.files?.[0]!.bytes).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
  });

  it("decodes the URL-SAFE alphabet and missing padding, which `atob` alone rejects", async () => {
    // "hello?" base64url-encoded, unpadded — `atob` throws on both the `_` and the short length.
    const out = await generateFlat({ model: streamingModel(textThenFile("aGVsbG8_")), schema: flatSchema, prompt: "x" });
    expect(out.value?.parsed).toEqual({ answer: "42" });
    expect(out.value?.files?.[0]!.bytes).toEqual(new Uint8Array([104, 101, 108, 108, 111, 63]));
  });

  it("drops an undecodable payload but KEEPS the value and the usage metrics", async () => {
    // The decode used to throw inside the `fullStream` loop, abandoning it before `finish` — so the
    // parsed answer AND every token/cost figure were lost, and the call reported as an error.
    const out = await generateFlat({ model: streamingModel(textThenFile("!!!not base64!!!")), schema: flatSchema, prompt: "x" });
    expect(out.value?.parsed).toEqual({ answer: "42" });
    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.files).toBeUndefined();
    expect(out.metrics.inputTokens).toBe(10);
    expect(out.metrics.outputTokens).toBe(5);
  });
});
