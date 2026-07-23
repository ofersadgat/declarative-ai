import { jsonSchema, stepCountIs, tool } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { generateStructured } from "../src/generate";
import { executeLlmCall, type CallDeps } from "../src/call";
import { fakeRouter, generateFlat, stream, usage, errorOf } from "./fakes";

const TOOL_CALL_STEP = [
  { type: "stream-start", warnings: [] },
  { type: "tool-input-start", id: "t1", toolName: "get_weather" },
  { type: "tool-input-delta", id: "t1", delta: '{"city":"NYC"}' },
  { type: "tool-input-end", id: "t1" },
  { type: "tool-call", toolCallId: "t1", toolName: "get_weather", input: '{"city":"NYC"}' },
  { type: "finish", finishReason: "tool-calls", usage: usage(5, 5) },
];
const TEXT_STEP = [
  { type: "stream-start", warnings: [] },
  { type: "text-start", id: "1" },
  { type: "text-delta", id: "1", delta: "It is sunny in NYC." },
  { type: "text-end", id: "1" },
  { type: "finish", finishReason: "stop", usage: usage(5, 3) },
];

const weatherSchema = { type: "object", properties: { city: { type: "string" } }, required: ["city"] };

describe("tools — single-turn (no executor)", () => {
  it("forwards tools + toolChoice to the call and surfaces the tool call in the outcome", async () => {
    let captured: Record<string, unknown> | undefined;
    const model = new MockLanguageModelV3({
      doStream: async (options) => {
        captured = options as unknown as Record<string, unknown>;
        return stream(TOOL_CALL_STEP);
      },
    });

    const out = await generateFlat({
      model,
      modelId: "claude-haiku-4-5",
      prompt: "weather in NYC?",
      tools: { get_weather: tool({ description: "look up weather", inputSchema: jsonSchema(weatherSchema) }) },
      toolChoice: "required",
    });

    expect(errorOf(out)).toBeUndefined();
    // The tools + toolChoice reached the underlying call (the SDK lowers the keyed ToolSet to an array).
    expect((captured?.tools as Array<{ name?: string }>).map((t) => t.name)).toContain("get_weather");
    expect(JSON.stringify(captured?.toolChoice)).toContain("required");
    // The model's tool call is surfaced first-class (parsed input), not executed.
    expect(out.value?.toolCalls).toHaveLength(1);
    expect(out.value?.toolCalls?.[0]).toMatchObject({ toolName: "get_weather", input: { city: "NYC" } });
    expect(out.value?.toolResults).toBeUndefined();
  });
});

describe("tools — executed loop", () => {
  it("runs the executor, feeds the result back, and surfaces the call + result", async () => {
    let step = 0;
    let executedWith: unknown;
    const model = new MockLanguageModelV3({
      doStream: async () => stream(step++ === 0 ? TOOL_CALL_STEP : TEXT_STEP),
    });

    const out = await generateFlat({
      model,
      modelId: "claude-haiku-4-5",
      prompt: "weather in NYC?",
      tools: {
        get_weather: tool({
          description: "look up weather",
          inputSchema: jsonSchema(weatherSchema),
          execute: async (input) => {
            executedWith = input;
            return { tempF: 72, city: (input as { city: string }).city };
          },
        }),
      },
      stopWhen: stepCountIs(4),
    });

    expect(errorOf(out)).toBeUndefined();
    expect(executedWith).toEqual({ city: "NYC" }); // executor ran with the parsed input
    expect(out.value?.value).toBe("It is sunny in NYC."); // final model turn after the tool result
    expect(out.value?.toolCalls?.[0]).toMatchObject({ toolName: "get_weather", input: { city: "NYC" } });
    expect(out.value?.toolResults?.[0]?.output).toMatchObject({ tempF: 72, city: "NYC" });
  });
});

describe("executeLlmCall — tool declarations + injected executors", () => {
  it("builds the tool set from serializable declarations and runs the loop", async () => {
    let step = 0;
    let executed = false;
    const model = new MockLanguageModelV3({
      doStream: async () => stream(step++ === 0 ? TOOL_CALL_STEP : TEXT_STEP),
    });
    const deps: CallDeps = {
      modelRouter: fakeRouter(model),
      toolExecutors: {
        get_weather: async () => {
          executed = true;
          return { tempF: 72 };
        },
      },
    };

    const out = await executeLlmCall(
      {
        model: "anthropic/claude-haiku-4-5",
        prompt: "weather in NYC?",
        tools: [{ name: "get_weather", description: "look up weather", inputSchema: weatherSchema }],
        maxSteps: 4,
        timeoutMs: 30_000,
      },
      deps,
    );

    expect(errorOf(out)).toBeUndefined();
    expect(executed).toBe(true); // the injected executor ran (declaration → ToolSet → loop)
    expect(out.value?.value).toBe("It is sunny in NYC.");
    expect(out.value?.toolCalls?.[0]).toMatchObject({ toolName: "get_weather", input: { city: "NYC" } });
    expect(out.value?.toolResults?.[0]?.output).toMatchObject({ tempF: 72 });
  });

  it("declaration WITHOUT an injected executor is single-turn (call returned, not run)", async () => {
    const model = new MockLanguageModelV3({ doStream: async () => stream(TOOL_CALL_STEP) });

    const out = await executeLlmCall(
      {
        model: "anthropic/claude-haiku-4-5",
        prompt: "weather in NYC?",
        tools: [{ name: "get_weather", inputSchema: weatherSchema }],
        timeoutMs: 30_000,
      },
      { modelRouter: fakeRouter(model) }, // no toolExecutors
    );

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.toolCalls?.[0]).toMatchObject({ toolName: "get_weather", input: { city: "NYC" } });
    expect(out.value?.toolResults).toBeUndefined();
  });
});
