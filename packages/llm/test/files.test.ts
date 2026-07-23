import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { executeLlmCall } from "../src/call";
import { ModelInfo } from "../src/model-catalog";
import { plan } from "../src/plan";
import { fakeRouter, generateFlat, stream, usage, errorOf } from "./fakes";

const TEXT = [
  { type: "stream-start", warnings: [] },
  { type: "text-start", id: "1" },
  { type: "text-delta", id: "1", delta: "ok" },
  { type: "text-end", id: "1" },
  { type: "finish", finishReason: "stop", usage: usage(1, 1) },
];

describe("file OUTPUT capture (DESIGN §3.7)", () => {
  it("captures a model-generated file as BYTES, not as a parallel artifacts channel", async () => {
    const model = new MockLanguageModelV3({
      doStream: async () =>
        stream([
          { type: "stream-start", warnings: [] },
          { type: "file", mediaType: "image/png", data: "aGVsbG8=" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]),
    });
    const out = await generateFlat({ model, modelId: "m", prompt: "draw a cat" });
    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.files).toHaveLength(1);
    expect(out.value?.files![0]!.mediaType).toBe("image/png");
    // A blob leaf holds bytes — the transport's base64 never escapes into the value (§7).
    expect(Array.from(out.value!.files![0]!.bytes)).toEqual([...new TextEncoder().encode("hello")]);
    expect("artifacts" in out).toBe(false);
  });
});

describe("file INPUT lowering (attachments)", () => {
  it("lowers a base64 attachment into a provider file part merged with the prompt", async () => {
    let captured: Record<string, unknown> | undefined;
    const model = new MockLanguageModelV3({
      doStream: async (opts) => {
        captured = opts as unknown as Record<string, unknown>;
        return stream(TEXT);
      },
    });
    const out = await executeLlmCall(
      { model: "openrouter/m", prompt: "describe", attachments: [{ mediaType: "image/png", data: { base64: "aGVsbG8=" } }], timeoutMs: 30_000 },
      { modelRouter: fakeRouter(model) },
    );
    expect(errorOf(out)).toBeUndefined();
    const wire = JSON.stringify(captured?.prompt);
    expect(wire).toContain("image/png");
    expect(wire).toContain("describe"); // text + file merged into one user turn
  });

  it("takes raw BYTES directly — there is no blob store to inject (§7.2)", async () => {
    let captured: Record<string, unknown> | undefined;
    const model = new MockLanguageModelV3({
      doStream: async (opts) => {
        captured = opts as unknown as Record<string, unknown>;
        return stream(TEXT);
      },
    });
    const out = await executeLlmCall(
      { model: "openrouter/m", prompt: "x", attachments: [{ mediaType: "application/pdf", data: new Uint8Array([1, 2, 3]) }], timeoutMs: 30_000 },
      { modelRouter: fakeRouter(model) },
    );
    expect(errorOf(out)).toBeUndefined();
    expect(JSON.stringify(captured?.prompt)).toContain("application/pdf");
  });

  it("passes a URL attachment through as a URL — the SDK fetches it, we never do (§7.2)", async () => {
    const model = new MockLanguageModelV3({ doStream: async () => stream(TEXT) });
    const out = await executeLlmCall(
      { model: "openrouter/m", prompt: "x", attachments: [{ mediaType: "application/pdf", data: { url: "https://example.invalid/a.pdf" } }], timeoutMs: 30_000 },
      { modelRouter: fakeRouter(model) },
    );
    // The download is the SDK's, and it fails here because the host does not exist — which is exactly
    // the evidence that the URL travelled through our lowering untouched, with no `fetch` of our own.
    expect(errorOf(out)?.reason).toContain("https://example.invalid/a.pdf");
  });
});

describe("plan — modality gating", () => {
  it("flags media inputs to a text-only model, and passes for a vision model", () => {
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/text-only", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text"], output: ["text"] } });
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/vision", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text", "image"], output: ["text"] } });
    const withImg = (model: string) => ({ model, prompt: "look", attachments: [{ mediaType: "image/png", data: { base64: "x" } }], timeoutMs: 1000 });

    expect(plan(withImg("openrouter/test/text-only")).issues.some((i) => /does not accept image inputs/.test(i))).toBe(true);
    expect(plan(withImg("openrouter/test/vision")).issues.some((i) => /inputs/.test(i))).toBe(false);
  });

  it("flags a requested output modality the model cannot produce", () => {
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/text-only", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text"], output: ["text"] } });
    const p = plan({ model: "openrouter/test/text-only", prompt: "draw", outputModalities: ["image"], timeoutMs: 1000 });
    expect(p.issues.some((i) => /cannot produce output modalities: image/.test(i))).toBe(true);
  });
});
