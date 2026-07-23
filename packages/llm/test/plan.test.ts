import { describe, expect, it } from "vitest";
import type { JsonSchema } from "@declarative-ai/json";
import { plan } from "../src/plan";
import type { LlmCallDefinition } from "../src/llmConfig";
import { ModelInfo } from "../src/model-catalog";
import { flatSchema } from "./fakes";

const baseDef: LlmCallDefinition & { schema?: JsonSchema } = {
  model: "anthropic/claude-haiku-4-5",
  system: "be terse",
  prompt: "what is 2+2?",
  maxOutputTokens: 100,
  temperature: 0.7,
  timeoutMs: 30_000,
  schema: flatSchema,
};

describe("plan — declarative dry run (no execution)", () => {
  it("reports provider, a content-hash identity, and a token+cost estimate", () => {
    const p = plan(baseDef);
    expect(p.provider).toEqual({ family: "anthropic", modelId: "anthropic/claude-haiku-4-5" });
    expect(p.contentHash).toMatch(/^[0-9a-f]{64}$/);
    expect(p.estimate.inputTokens).toBeGreaterThan(0);
    expect(p.estimate.outputTokens).toBe(100);
    expect(p.estimate.costUsd).toBeGreaterThan(0); // haiku has a price table entry
  });

  it("content hash is stable for the same declaration and changes with the prompt", () => {
    expect(plan(baseDef).contentHash).toBe(plan({ ...baseDef }).contentHash);
    expect(plan({ ...baseDef, prompt: "different" }).contentHash).not.toBe(plan(baseDef).contentHash);
  });

  it("reports the structured-output enforcement when a schema is present, none for text output", () => {
    expect(["strict", "advisory", "text"]).toContain(plan(baseDef).structuredOutput);
    const { schema, ...textDef } = baseDef;
    void schema;
    expect(plan(textDef).structuredOutput).toBeUndefined();
  });

  it("routes an openrouter-prefixed model to the openrouter family", () => {
    expect(plan({ ...baseDef, model: "openrouter/openai/gpt-4.1-mini" }).provider.family).toBe("openrouter");
  });

  it("exposes unsupportedParams as an array (params the model would drop)", () => {
    expect(Array.isArray(plan(baseDef).unsupportedParams)).toBe(true);
  });
});

describe("plan — per-mediaType modality gating", () => {
  it("gates each attachment by its REQUIRED modality, not a generic media check", () => {
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/audio-only", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text", "audio"], output: ["text"] } });
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/image-only", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text", "image"], output: ["text"] } });
    const withAtt = (model: string, mediaType: string): LlmCallDefinition => ({
      model,
      prompt: "listen",
      attachments: [{ mediaType, data: { base64: "x" } }],
      timeoutMs: 1000,
    });

    // An audio input to an audio-capable model FITS (was falsely flagged by the old image/file-only check).
    expect(plan(withAtt("openrouter/test/audio-only", "audio/mp3")).issues).toEqual([]);
    // An audio input to an image-only model is flagged (was falsely passing before).
    expect(plan(withAtt("openrouter/test/image-only", "audio/mp3")).issues.some((i) => /does not accept audio inputs/.test(i))).toBe(true);
    // A pdf requires the "file" modality.
    expect(plan(withAtt("openrouter/test/image-only", "application/pdf")).issues.some((i) => /does not accept file inputs/.test(i))).toBe(true);
  });

  it("gates media parts inside messages by their mediaType too", () => {
    ModelInfo.instance.upsert({ route: "openrouter", model: "test/image-only", inputPerMillion: 0, outputPerMillion: 0, modalities: { input: ["text", "image"], output: ["text"] } });
    const p = plan({
      model: "openrouter/test/image-only",
      messages: [{ role: "user", content: [{ type: "file", mediaType: "video/mp4", data: "x" }] }],
      timeoutMs: 1000,
    });
    expect(p.issues.some((i) => /does not accept video inputs/.test(i))).toBe(true);
  });
});

describe("content hashing with binary attachments", () => {
  // `FileInput.data` may hold raw bytes since blob became a leaf kind (§7), and JCS has no
  // `Uint8Array` case — it serializes as an object with ONE KEY PER BYTE. A megabyte attachment
  // therefore built a multi-megabyte string and sorted millions of numeric keys before hashing.
  const withBytes = (bytes: Uint8Array): LlmCallDefinition => ({
    ...baseDef,
    attachments: [{ mediaType: "image/png", data: bytes }],
  });

  it("hashes a large byte attachment promptly", () => {
    const big = new Uint8Array(2_000_000).fill(7);
    const started = Date.now();
    expect(plan(withBytes(big)).contentHash).toMatch(/^[0-9a-f]{64}$/);
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it("is stable for identical bytes and distinct for different bytes", () => {
    expect(plan(withBytes(new Uint8Array([1, 2, 3]))).contentHash).toBe(plan(withBytes(new Uint8Array([1, 2, 3]))).contentHash);
    expect(plan(withBytes(new Uint8Array([1, 2, 3]))).contentHash).not.toBe(plan(withBytes(new Uint8Array([1, 2, 4]))).contentHash);
  });

  it("leaves a byte-free declaration's hash alone", () => {
    expect(plan(baseDef).contentHash).toBe(plan(baseDef).contentHash);
    expect(plan(baseDef).contentHash).not.toBe(plan(withBytes(new Uint8Array([1]))).contentHash);
  });
});
