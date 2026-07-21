import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import type { ConfigurationRegistry, RuntimeOp, Tool } from "@declarative-ai/core";
import { createLlmCallExecutor } from "../src/executor";
import { createLlmRuntime } from "../src/runtime";
import { fakeRouter, fakeRunner, flatSchema, okOutcome, stream, usage } from "./fakes";

const opOf = (overrides: Partial<RuntimeOp> = {}): RuntimeOp => ({
  prompt: "hello",
  config: {},
  outputSchema: flatSchema,
  ...overrides,
});

describe("createLlmRuntime — absorbs the llmCallBinding config pipeline", () => {
  it("merges defaults ← inline config and renders the prompt into the definition", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const runtime = createLlmRuntime({
      defaults: { model: "anthropic/claude-sonnet-5", temperature: 0.3 },
      executor: createLlmCallExecutor({ runner }),
    });

    const outcome = await runtime.run(opOf({ prompt: "why?", config: { temperature: 0.7 } }), {}).outcome;

    expect(outcome.error).toBeUndefined();
    expect(calls).toHaveLength(1);
    const params = calls[0]!.params as unknown as Record<string, unknown>;
    expect(params.model).toBe("anthropic/claude-sonnet-5"); // from defaults
    expect(params.temperature).toBe(0.7); // inline wins over default
    expect(params.prompt).toBe("why?"); // the op prompt IS the call prompt
  });

  it("resolves a configRef preset UNDER the inline config", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const presets: Record<string, Record<string, unknown>> = {
      critic: { model: "anthropic/claude-opus-4-8", temperature: 0.1 },
    };
    const configs: ConfigurationRegistry = { get: (name) => presets[name] };
    const runtime = createLlmRuntime({
      defaults: { model: "anthropic/claude-sonnet-5" },
      configs,
      executor: createLlmCallExecutor({ runner }),
    });

    await runtime.run(opOf({ config: { configRef: "critic" } }), {}).outcome;

    const params = calls[0]!.params as unknown as Record<string, unknown>;
    expect(params.model).toBe("anthropic/claude-opus-4-8"); // preset over default
    expect(params.temperature).toBe(0.1);
  });

  it("returns a permanent failure (never throws) when config supplies a prompt", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    const runtime = createLlmRuntime({
      defaults: { model: "anthropic/claude-sonnet-5" },
      executor: createLlmCallExecutor({ runner }),
    });

    const outcome = await runtime.run(opOf({ config: { prompt: "no" } }), {}).outcome;

    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/prompt/);
  });
});

describe("createLlmRuntime — op.tools become an executed bounded loop (RUNTIMES-AND-PERMISSIONS.md §2)", () => {
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

  it("declares the tool, runs its executor in-loop with the shared ctx, and completes", async () => {
    let step = 0;
    let executedWith: unknown;
    const model = new MockLanguageModelV3({ doStream: async () => stream(step++ === 0 ? TOOL_CALL_STEP : TEXT_STEP) });
    const getWeather: Tool = {
      description: "look up weather",
      inputSchema: weatherSchema,
      run: (input) => {
        executedWith = input;
        return { tempF: 72, city: (input as { city: string }).city };
      },
    };
    const runtime = createLlmRuntime({ defaults: { model: "anthropic/claude-haiku-4-5" } });

    const outcome = await runtime.run(
      opOf({ prompt: "weather in NYC?", outputSchema: undefined, tools: { get_weather: getWeather } }),
      { modelRouter: fakeRouter(model) },
    ).outcome;

    expect(outcome.error).toBeUndefined();
    expect(executedWith).toEqual({ city: "NYC" }); // the registered tool's `run` executed with the parsed input
    expect(step).toBe(2); // the loop did the follow-up turn after feeding the tool result back
  });
});
