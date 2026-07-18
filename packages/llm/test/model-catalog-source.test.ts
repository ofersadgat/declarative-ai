import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ModelCatalog } from "../src/model-catalog";
import {
  anthropicRejectsSampling,
  claudeSupportedParameters,
  claudeSupportsThinking,
  completeSeedRow,
  makeAnthropicDocsSource,
  makeOpenRouterSource,
  parseAnthropicDocsPricing,
  parseOpenRouterModels,
  refreshModelCatalog,
  sanitizePricingRows,
  validatePricingRows,
  type PricingSource,
} from "../src/model-catalog-source";

const fixture = readFileSync(
  fileURLToPath(new URL("./fixtures/anthropic-pricing.md", import.meta.url)),
  "utf8",
);

describe("parseAnthropicDocsPricing", () => {
  it("parses the docs table into rows with exact cache rates", () => {
    const rows = parseAnthropicDocsPricing(fixture);
    const byId = Object.fromEntries(rows.map((r) => [r.modelPrefix, r]));

    expect(byId["claude-opus-4-8"]).toEqual({
      modelPrefix: "claude-opus-4-8",
      inputPerMillion: 5,
      outputPerMillion: 25,
      cacheReadPerMillion: 0.5,
      cacheWritePerMillion: 6.25,
      cacheWrite1hPerMillion: 10,
      family: "anthropic",
      provider: "Anthropic",
      label: "claude-opus-4-8",
      canonicalId: "claude-opus-4-8",
      servingProvider: "anthropic",
      // Stamped family defaults — the docs table has no modality/capability columns. Opus 4.8 supports
      // extended thinking (`reasoning`) but REJECTS the sampling knobs (temperature/top_p/top_k) — mirrors
      // the AI SDK's rejectsSamplingParameters, so the refresh can't re-introduce the ignored-param warning.
      modalities: { input: ["text", "image", "file"], output: ["text"] },
      supportedParameters: ["max_tokens", "stop", "tools", "tool_choice", "reasoning"],
      source: "anthropic-docs",
    });
    expect(byId["claude-haiku-4-5"]).toMatchObject({ inputPerMillion: 1, outputPerMillion: 5 });
    // Claude 4.5 Haiku supports extended thinking → reasoning is in its supported params.
    expect(byId["claude-haiku-4-5"]?.supportedParameters).toContain("reasoning");
    // The name→prefix map drops "([deprecated])" notes and dots → hyphens.
    expect(byId["claude-sonnet-4"]).toMatchObject({ inputPerMillion: 3 });
    expect(byId["claude-sonnet-4"]?.supportedParameters).toContain("reasoning");
    expect(byId["claude-haiku-3-5"]).toMatchObject({ inputPerMillion: 0.8, outputPerMillion: 4 });
    // Claude 3.5 Haiku does NOT support extended thinking → no reasoning branch (sampling-only).
    expect(byId["claude-haiku-3-5"]?.supportedParameters).not.toContain("reasoning");
    expect(byId["claude-haiku-3-5"]?.supportedParameters).toContain("temperature");
  });

  it("parses an HTML <table> form too (production fetch may return HTML)", () => {
    const html = `
      <table>
        <tr><th>Model</th><th>Base Input Tokens</th><th>5m Cache Writes</th><th>1h Cache Writes</th><th>Cache Hits</th><th>Output Tokens</th></tr>
        <tr><td>Claude Haiku 4.5</td><td>$1 / MTok</td><td>$1.25 / MTok</td><td>$2 / MTok</td><td>$0.10 / MTok</td><td>$5 / MTok</td></tr>
        <tr><td>Claude Sonnet 4.6</td><td>$3 / MTok</td><td>$3.75 / MTok</td><td>$6 / MTok</td><td>$0.30 / MTok</td><td>$15 / MTok</td></tr>
        <tr><td>Claude Opus 4.8</td><td>$5 / MTok</td><td>$6.25 / MTok</td><td>$10 / MTok</td><td>$0.50 / MTok</td><td>$25 / MTok</td></tr>
      </table>`;
    const rows = parseAnthropicDocsPricing(html);
    expect(rows.find((r) => r.modelPrefix === "claude-haiku-4-5")).toMatchObject({
      inputPerMillion: 1,
      outputPerMillion: 5,
      cacheReadPerMillion: 0.1,
      cacheWrite1hPerMillion: 2,
    });
  });

  it("returns [] when no recognizable pricing table is present", () => {
    expect(parseAnthropicDocsPricing("# Some page\n\nNo table here.")).toEqual([]);
  });
});

describe("parseOpenRouterModels", () => {
  const json = JSON.stringify({
    data: [
      { id: "anthropic/claude-opus-4.8", created: 1730000000, pricing: { prompt: "0.000005", completion: "0.000025", input_cache_read: "0.0000005", input_cache_write: "0.00000625" } },
      { id: "openai/gpt-4.1-mini", pricing: { prompt: "0.0000004", completion: "0.0000016" } },
      { id: "some/dynamic-model", pricing: { prompt: "-1", completion: "-1" } }, // unavailable -> dropped
      { id: "broken", pricing: null }, // no pricing -> dropped
    ],
  });

  it("converts per-token string prices to per-million and keeps cache rates", () => {
    const rows = parseOpenRouterModels(json);
    const byId = Object.fromEntries(rows.map((r) => [r.modelPrefix, r]));
    expect(byId["anthropic/claude-opus-4.8"]).toEqual({
      modelPrefix: "anthropic/claude-opus-4.8",
      inputPerMillion: 5,
      outputPerMillion: 25,
      cacheReadPerMillion: 0.5,
      cacheWritePerMillion: 6.25,
      releasedAt: 1730000000, // captured from `created`
      family: "openrouter",
      provider: "Anthropic", // vendor slug → display name
      label: "claude-opus-4.8", // vendor/ prefix dropped
      canonicalId: "claude-opus-4-8", // dots→hyphens: collapses onto the native `claude-opus-4-8` row
      servingProvider: "openrouter", // served via OpenRouter, NOT the native Anthropic API
      source: "openrouter-models",
      // No supportedParameters/modalities stamped: the feed omitted them and OR rows are NOT given
      // native-Claude caps (the read-time fallback resolves them if needed).
    });
    expect(byId["openai/gpt-4.1-mini"]).toMatchObject({ inputPerMillion: 0.4, outputPerMillion: 1.6 });
    expect(byId["openai/gpt-4.1-mini"]?.releasedAt).toBeUndefined(); // no `created` → unset
    expect(byId["some/dynamic-model"]).toBeUndefined(); // -1 dropped
    expect(byId["broken"]).toBeUndefined();
  });

  it("captures capabilities: supported_parameters, context_length, max output, modalities", () => {
    const withCaps = JSON.stringify({
      data: [
        {
          id: "openai/gpt-5-nano",
          pricing: { prompt: "0.00000025", completion: "0.000001" },
          context_length: 400000,
          supported_parameters: ["reasoning", "response_format", "structured_outputs", "max_tokens"],
          top_provider: { max_completion_tokens: 128000 },
          architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
        },
      ],
    });
    const [row] = parseOpenRouterModels(withCaps);
    expect(row).toMatchObject({
      modelPrefix: "openai/gpt-5-nano",
      provider: "OpenAI",
      family: "openrouter",
      supportedParameters: ["reasoning", "response_format", "structured_outputs", "max_tokens"],
      contextLength: 400000,
      maxOutputTokens: 128000,
      modalities: { input: ["text", "image"], output: ["text"] },
    });
    // The reasoning model does NOT list temperature/top_p/top_k — exactly why require_parameters would
    // 404 it if we sent them; the executor filters them out (see llmStep / param-filter test).
    expect(row?.supportedParameters).not.toContain("temperature");
  });

  it("returns [] for non-JSON or a missing data array", () => {
    expect(parseOpenRouterModels("not json")).toEqual([]);
    expect(parseOpenRouterModels("{}")).toEqual([]);
  });
});

describe("sanitizePricingRows (lenient per-row drop)", () => {
  it("keeps the good rows and drops the bad ones", () => {
    const { rows, dropped } = sanitizePricingRows([
      { modelPrefix: "ok", inputPerMillion: 1, outputPerMillion: 5 },
      { modelPrefix: "bad-output", inputPerMillion: 10, outputPerMillion: 1 },
      { modelPrefix: "ok", inputPerMillion: 1, outputPerMillion: 5 }, // duplicate
    ]);
    expect(rows.map((r) => r.modelPrefix)).toEqual(["ok"]);
    expect(dropped.map((d) => d.modelPrefix).sort()).toEqual(["bad-output", "ok"]);
  });

  it("KEEPS a zero-output embedding model (input-only billing is legitimate, not a bad scrape)", () => {
    // Carve-out (review finding #2): embedding / input-only models bill $0 output. They must
    // NOT be flagged by the output>0 or output<input heuristics, or a lenient feed would drop
    // every embedding model.
    const { rows, dropped } = sanitizePricingRows([
      { modelPrefix: "text-embedding-3-small", inputPerMillion: 0.02, outputPerMillion: 0 },
    ]);
    expect(rows.map((r) => r.modelPrefix)).toEqual(["text-embedding-3-small"]);
    expect(dropped).toEqual([]);
  });

  it("still drops a NEGATIVE-output row (zero is fine, negative is a bad scrape)", () => {
    const { rows, dropped } = sanitizePricingRows([
      { modelPrefix: "broken", inputPerMillion: 1, outputPerMillion: -1 },
    ]);
    expect(rows).toEqual([]);
    expect(dropped[0]?.modelPrefix).toBe("broken");
  });
});

describe("validatePricingRows", () => {
  it("accepts a sane batch", () => {
    expect(validatePricingRows(parseAnthropicDocsPricing(fixture)).ok).toBe(true);
  });

  it("rejects a too-small / inconsistent batch", () => {
    expect(validatePricingRows([]).ok).toBe(false);
    const bad = validatePricingRows([
      { modelPrefix: "x", inputPerMillion: 10, outputPerMillion: 1 }, // output < input
      { modelPrefix: "y", inputPerMillion: -1, outputPerMillion: 5 }, // non-positive input
      { modelPrefix: "z", inputPerMillion: 1, outputPerMillion: 2, cacheReadPerMillion: 5 }, // read ≥ input
    ]);
    expect(bad.ok).toBe(false);
    expect(bad.problems.length).toBeGreaterThanOrEqual(3);
  });
});

describe("refreshModelCatalog", () => {
  const sourceFrom = (text: string): PricingSource =>
    makeAnthropicDocsSource(async () => text, "https://example/pricing");

  it("applies only changed rows and reports added/updated", async () => {
    const table = new ModelCatalog([
      { modelPrefix: "claude-haiku-4-5", inputPerMillion: 0.8, outputPerMillion: 4 }, // stale → updated
    ]);
    const report = await refreshModelCatalog({ sources: [sourceFrom(fixture)], table });

    expect(report.bySource[0]).toMatchObject({ name: "anthropic-docs", skipped: false });
    expect(report.updated).toContain("claude-haiku-4-5");
    expect(report.added).toContain("claude-opus-4-8");
    // The stale row was corrected to the scraped value.
    expect(table.computeCostUsd("claude-haiku-4-5", 1_000_000, 0)).toBeCloseTo(1, 10);
  });

  it("a second refresh with identical data applies nothing (idempotent diff)", async () => {
    const table = new ModelCatalog();
    await refreshModelCatalog({ sources: [sourceFrom(fixture)], table });
    const second = await refreshModelCatalog({ sources: [sourceFrom(fixture)], table });
    expect(second.added).toEqual([]);
    expect(second.updated).toEqual([]);
    expect(second.bySource[0]!.applied).toBe(0);
  });

  it("NEVER overwrites good prices when a source fails to fetch", async () => {
    const table = new ModelCatalog([{ modelPrefix: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 }]);
    const failing: PricingSource = {
      name: "boom",
      fetchRows: async () => {
        throw new Error("network down");
      },
    };
    const report = await refreshModelCatalog({ sources: [failing], table });
    expect(report.bySource[0]).toMatchObject({ skipped: true, error: "network down" });
    // Existing price is untouched.
    expect(table.computeCostUsd("claude-haiku-4-5", 1_000_000, 0)).toBeCloseTo(1, 10);
  });

  it("a lenient source applies its good rows even when the batch has bad ones", async () => {
    const table = new ModelCatalog();
    // 2 good rows + 1 unpriceable (-1, dropped) + 1 inconsistent (output<input, dropped).
    const json = JSON.stringify({
      data: [
        { id: "a/good-1", pricing: { prompt: "0.000001", completion: "0.000005" } },
        { id: "a/good-2", pricing: { prompt: "0.000002", completion: "0.000008" } },
        { id: "a/dyn", pricing: { prompt: "-1", completion: "-1" } },
        { id: "a/weird", pricing: { prompt: "0.00001", completion: "0.000001" } }, // output<input
      ],
    });
    const report = await refreshModelCatalog({
      sources: [makeOpenRouterSource(async () => json, "https://example/models")],
      table,
      minRows: 1,
    });
    expect(report.bySource[0]).toMatchObject({ name: "openrouter-models", skipped: false });
    expect(report.added.sort()).toEqual(["a/good-1", "a/good-2"]); // weird + dyn dropped, not fatal
    expect(table.hasPricing("a/weird")).toBe(false);
  });

  it("NEVER overwrites good prices when a scrape fails validation", async () => {
    const table = new ModelCatalog([{ modelPrefix: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 }]);
    // A garbled page → parser yields nothing → validation fails → skip.
    const report = await refreshModelCatalog({ sources: [sourceFrom("garbled, no table")], table });
    expect(report.bySource[0]!.skipped).toBe(true);
    expect(table.computeCostUsd("claude-haiku-4-5", 1_000_000, 0)).toBeCloseTo(1, 10);
  });
});

describe("claudeSupportedParameters — the ingestion path's Claude capability authority", () => {
  const has = (id: string, p: string): boolean => claudeSupportedParameters(id).includes(p);

  it("opus-4-7/4-8 REJECT the sampling knobs but keep stop + reasoning (mirrors the AI SDK)", () => {
    for (const id of ["claude-opus-4-8", "claude-opus-4-7", "anthropic/claude-opus-4.8", "claude-opus-4-8-20260115"]) {
      expect(anthropicRejectsSampling(id)).toBe(true);
      expect(has(id, "temperature")).toBe(false);
      expect(has(id, "top_p")).toBe(false);
      expect(has(id, "top_k")).toBe(false);
      expect(has(id, "stop")).toBe(true);
      expect(has(id, "reasoning")).toBe(true);
    }
  });

  it("Claude 4+ (non-opus-4-7/4-8) accept sampling AND thinking", () => {
    for (const id of ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5"]) {
      expect(anthropicRejectsSampling(id)).toBe(false);
      expect(has(id, "temperature")).toBe(true);
      expect(has(id, "reasoning")).toBe(true);
    }
  });

  it("pre-4 Claude (3.5 / 3) accept sampling but NOT thinking", () => {
    for (const id of ["claude-haiku-3-5", "claude-3-haiku"]) {
      expect(has(id, "temperature")).toBe(true);
      expect(claudeSupportsThinking(id)).toBe(false);
      expect(has(id, "reasoning")).toBe(false);
    }
  });
});

describe("completeSeedRow — identity always, Claude caps for native rows only", () => {
  it("stamps identity + correct Claude caps on a native price-only row", () => {
    const row = completeSeedRow({ modelPrefix: "claude-opus-4-8", inputPerMillion: 5, outputPerMillion: 25 });
    expect(row.canonicalId).toBe("claude-opus-4-8");
    expect(row.servingProvider).toBe("anthropic");
    expect(row.provider).toBe("Anthropic");
    expect(row.supportedParameters).not.toContain("temperature"); // opus-4-8 rejects sampling
    expect(row.supportedParameters).toContain("reasoning");
    expect(row.modalities).toEqual({ input: ["text", "image", "file"], output: ["text"] });
  });

  it("stamps identity only (no caps) on a NON-Claude row — the feed/refresh owns those", () => {
    const row = completeSeedRow({ modelPrefix: "gpt-4o", inputPerMillion: 2.5, outputPerMillion: 10 });
    expect(row.canonicalId).toBe("gpt-4o");
    expect(row.servingProvider).toBe("openrouter");
    expect(row.supportedParameters).toBeUndefined();
    expect(row.modalities).toBeUndefined();
  });
});
