import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { executeLlmCall } from "../src/call.js";
import { createModelRouter } from "../src/router.js";
import { errorOf, outputOf } from "./fakes.js";

/**
 * The `local` route END TO END, against a stub OpenAI-compatible server.
 *
 * The point of driving real HTTP rather than a `MockLanguageModelV3` is that the thing under test IS
 * the transport: `@ai-sdk/openai-compatible` wiring, SSE decoding, and whether streamed `usage`
 * survives into `LlmMetrics`. A mock model would skip every one of those and still pass.
 */

/** One OpenAI chat-completion SSE chunk. */
const chunk = (payload: Record<string, unknown>): string =>
  `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", created: 1, model: "stub", ...payload })}\n\n`;

interface Recorded {
  path: string;
  body: Record<string, unknown>;
}

let server: Server;
let baseURL: string;
const recorded: Recorded[] = [];
/** The text the next response streams, as delta pieces. */
let nextDeltas: string[] = ["hello"];

beforeAll(async () => {
  server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const parts: Buffer[] = [];
    req.on("data", (d: Buffer) => parts.push(d));
    req.on("end", () => {
      recorded.push({ path: req.url ?? "", body: JSON.parse(Buffer.concat(parts).toString() || "{}") as Record<string, unknown> });
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      res.write(chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }));
      for (const piece of nextDeltas) {
        res.write(chunk({ choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
      }
      res.write(
        chunk({
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
        }),
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseURL = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const routerFor = (over: Partial<Parameters<typeof createModelRouter>[0]> = {}) =>
  createModelRouter({ skipDispatcher: true, local: { baseURL }, ...over });

describe("the `local` route, end to end over HTTP", () => {
  it("completes a TEXT call and reports the server's token usage", async () => {
    recorded.length = 0;
    nextDeltas = ["Red", ", Yellow", ", Blue."];
    const out = await executeLlmCall(
      { model: "local/qwen2.5-32b-instruct", prompt: "Name three primary colors." },
      { modelRouter: routerFor() },
    );
    expect(errorOf(out)).toBeUndefined();
    expect(outputOf(out)?.value).toBe("Red, Yellow, Blue.");
    // The usage came off the streamed final chunk — which only arrives because `includeUsage` asked
    // for it. Local inference is free, so this is about metrics being true, not about billing.
    expect(out.metrics.inputTokens).toBe(11);
    expect(out.metrics.outputTokens).toBe(7);
    expect(recorded[0]!.path).toBe("/v1/chat/completions");
    expect(recorded[0]!.body.stream_options).toEqual({ include_usage: true });
  });

  it("an UNPRICED local model costs 0 and says so — `unknown`, not a fabricated free", async () => {
    // There is no catalog row for this id, so nothing can price it. `costSource` is the field that
    // keeps "we could not price it" distinct from "it was free", and a local model is genuinely the
    // latter — but only once a zero-rate row says so. Until then the honest answer is `unknown`.
    nextDeltas = ["ok"];
    const out = await executeLlmCall({ model: "local/unpriced-model", prompt: "hi" }, { modelRouter: routerFor() });
    expect(out.metrics.costUsd).toBe(0);
    expect(out.metrics.costSource).toBe("unknown");
  });

  it("sends the schema as an advisory json_object hint, NOT the OpenAI strict dialect", async () => {
    // `local/` resolves to LOCAL_JSON_OBJECT (the object tier). The regression this pins: falling into
    // the OpenRouter capability derivation would send `response_format: json_schema` with a strict
    // dialect that most local servers reject outright.
    recorded.length = 0;
    nextDeltas = ['{"answer":"4"}'];
    const out = await executeLlmCall(
      {
        model: "local/qwen2.5-32b-instruct",
        prompt: "What is 2+2?",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] } as never,
      },
      { modelRouter: routerFor() },
    );
    expect(errorOf(out)).toBeUndefined();
    expect(outputOf(out)?.value).toEqual({ answer: "4" });
    expect(recorded[0]!.body.response_format).toEqual({ type: "json_object" });
  });

  it("describes the schema in the prompt, since json_object mode does not carry it", async () => {
    // `{type:"json_object"}` forces JSON SYNTAX and says nothing about SHAPE. Without this the model is
    // told to emit JSON and left to guess the fields, with Ajv rejecting whatever comes back.
    recorded.length = 0;
    nextDeltas = ['{"answer":"4"}'];
    await executeLlmCall(
      {
        model: "local/qwen2.5-32b-instruct",
        system: "Answer tersely.",
        prompt: "What is 2+2?",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] } as never,
      },
      { modelRouter: routerFor() },
    );
    const messages = recorded[0]!.body.messages as { role: string; content: string }[];
    expect(messages[0]!.role).toBe("system");
    expect(messages[0]!.content).toMatch(/^Answer tersely\./); // the authored prompt still leads
    expect(messages[0]!.content).toMatch(/conforming to this JSON Schema/);
    expect(messages[0]!.content).toContain('"answer"'); // ...and the shape is actually in there
    // The OpenAI/DashScope json-specifier directive is NOT spliced in: LOCAL_JSON_OBJECT drops that
    // contract, and the schema hint would satisfy it anyway.
    expect(JSON.stringify(messages)).not.toMatch(/Return your answer as a single valid JSON object/);
  });

  it("a server capped to no-json_schema still gets the json_object hint", async () => {
    // The cap downgrades json_schema → json_object; it does not suppress `response_format`. Suppressing
    // it entirely is the TEXT tier's job and comes from the schema profile, not from server config.
    recorded.length = 0;
    nextDeltas = ['{"answer":"4"}'];
    await executeLlmCall(
      {
        model: "local/plain-completion-shim",
        prompt: "What is 2+2?",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] } as never,
      },
      { modelRouter: routerFor({ local: { baseURL, supportsStructuredOutputs: false } }) },
    );
    expect(recorded[0]!.body.response_format).toEqual({ type: "json_object" });
  });
});
