import { MockLanguageModelV3 } from "ai/test";
import type { BlobStore } from "@declarative-ai/core";
import { describe, expect, it } from "vitest";
import { generateStructured } from "../src/generate";
import { executeStructuredCall } from "../src/llmStep";
import { defaultModelCatalog, type ModelInfo } from "../src/model-catalog";
import { plan } from "../src/plan";
import { fakeRouter, stream, usage } from "./fakes";

const TEXT = [
  { type: "stream-start", warnings: [] },
  { type: "text-start", id: "1" },
  { type: "text-delta", id: "1", delta: "ok" },
  { type: "text-end", id: "1" },
  { type: "finish", finishReason: "stop", usage: usage(1, 1) },
];

describe("file OUTPUT capture", () => {
  it("captures a model-generated file into outcome.artifacts (base64 + mediaType + contentHash)", async () => {
    const model = new MockLanguageModelV3({
      doStream: async () =>
        stream([
          { type: "stream-start", warnings: [] },
          { type: "file", mediaType: "image/png", data: "aGVsbG8=" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]),
    });
    const out = await generateStructured({ model, modelId: "m", prompt: "draw a cat" });
    expect(out.error).toBeUndefined();
    expect(out.artifacts).toHaveLength(1);
    expect(out.artifacts![0]).toMatchObject({ content: "aGVsbG8=", format: "image/png" });
    expect(out.artifacts![0]!.contentHash).toMatch(/^[0-9a-f]{64}$/);
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
    const out = await executeStructuredCall(
      { model: "openrouter/m", prompt: "describe", attachments: [{ mediaType: "image/png", data: { base64: "aGVsbG8=" } }], timeoutMs: 30_000 },
      { providers: fakeRouter(model) },
    );
    expect(out.error).toBeUndefined();
    const wire = JSON.stringify(captured?.prompt);
    expect(wire).toContain("image/png");
    expect(wire).toContain("describe"); // text + file merged into one user turn
  });

  it("resolves a contentHash attachment via the injected blob store", async () => {
    let loadedRef: unknown;
    const blobs: BlobStore = {
      load: async (ref) => {
        loadedRef = ref;
        return { bytes: new Uint8Array([1, 2, 3]) };
      },
      put: async () => ({ contentHash: "h" }),
    };
    const model = new MockLanguageModelV3({ doStream: async () => stream(TEXT) });
    const out = await executeStructuredCall(
      { model: "openrouter/m", prompt: "x", attachments: [{ mediaType: "application/pdf", data: { contentHash: "abc" } }], timeoutMs: 30_000 },
      { providers: fakeRouter(model), blobs },
    );
    expect(out.error).toBeUndefined();
    expect(loadedRef).toEqual({ contentHash: "abc" });
  });

  it("fails permanently when a contentHash attachment has no blob store to resolve it", async () => {
    const model = new MockLanguageModelV3({ doStream: async () => stream(TEXT) });
    const out = await executeStructuredCall(
      { model: "openrouter/m", prompt: "x", attachments: [{ mediaType: "application/pdf", data: { contentHash: "abc" } }], timeoutMs: 30_000 },
      { providers: fakeRouter(model) },
    );
    expect(out.error?.classification).toBe("permanent");
    expect(out.error?.reason).toMatch(/blob store/);
  });
});

describe("plan — modality gating", () => {
  it("flags media inputs to a text-only model, and passes for a vision model", () => {
    defaultModelCatalog.upsert({ modelPrefix: "test/text-only", modalities: { input: ["text"], output: ["text"] } } as ModelInfo);
    defaultModelCatalog.upsert({ modelPrefix: "test/vision", modalities: { input: ["text", "image"], output: ["text"] } } as ModelInfo);
    const withImg = (model: string) => ({ model, prompt: "look", attachments: [{ mediaType: "image/png", data: { base64: "x" } }], timeoutMs: 1000 });

    expect(plan(withImg("openrouter/test/text-only")).issues.some((i) => /does not accept image inputs/.test(i))).toBe(true);
    expect(plan(withImg("openrouter/test/vision")).issues.some((i) => /inputs/.test(i))).toBe(false);
  });

  it("flags a requested output modality the model cannot produce", () => {
    defaultModelCatalog.upsert({ modelPrefix: "test/text-only", modalities: { input: ["text"], output: ["text"] } } as ModelInfo);
    const p = plan({ model: "openrouter/test/text-only", prompt: "draw", outputModalities: ["image"], timeoutMs: 1000 });
    expect(p.issues.some((i) => /cannot produce output modalities: image/.test(i))).toBe(true);
  });
});
