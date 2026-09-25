import { describe, expect, it } from "vitest";
import { acceptanceOf, keyForModel, ModelInfo, type ModelInfoInterface } from "../src/model-catalog.js";
import {
  anthropicRejectsSampling,
  makeAnthropicModelsSource,
  makeOpenRouterSources,
  mergeRow,
  nativeMirrors,
  parseAnthropicModels,
  parseOpenRouterModels,
  refreshModelCatalog,
  sanitizeCatalogRows,
  validateCatalogRows,
  type CatalogSource,
} from "../src/model-catalog-source.js";

/** The call fields a row's `parameters` schema accepts. */
function fields(row: ModelInfoInterface | undefined): string[] {
  return Object.keys((row?.parameters?.["properties"] as Record<string, unknown> | undefined) ?? {});
}

/** A source that answers with fixed rows — the orchestrator's view of any source. */
const fixed = (name: string, rows: ModelInfoInterface[], extra: Partial<CatalogSource> = {}): CatalogSource => ({ name, fetchRows: async () => rows, ...extra });

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
    const byId = Object.fromEntries(rows.map((r) => [r.model, r]));
    expect(byId["anthropic/claude-opus-4.8"]).toEqual({
      route: "openrouter",
      model: "anthropic/claude-opus-4.8",
      inputPerMillion: 5,
      outputPerMillion: 25,
      cacheReadPerMillion: 0.5,
      cacheWritePerMillion: 6.25,
      releasedAt: 1730000000, // captured from `created`
      provider: "Anthropic", // vendor slug → display name
      label: "claude-opus-4.8", // vendor/ prefix dropped
      canonicalId: "claude-opus-4-8", // dots→hyphens: collapses onto the native `claude-opus-4-8` row
      source: "openrouter-models",
    });
    expect(byId["openai/gpt-4.1-mini"]).toMatchObject({ inputPerMillion: 0.4, outputPerMillion: 1.6 });
    expect(byId["openai/gpt-4.1-mini"]?.releasedAt).toBeUndefined(); // no `created` → unset
    expect(byId["some/dynamic-model"]).toBeUndefined(); // -1 dropped
    expect(byId["broken"]).toBeUndefined();
  });

  it("reads the 1-hour cache write and the long-context tier", () => {
    // MEASURED 2026-09-24: anthropic/claude-opus-5.5's 1h write, and openai/gpt-5.6-luna-pro's override.
    const [row] = parseOpenRouterModels(
      JSON.stringify({
        data: [
          {
            id: "openai/gpt-5.6-luna-pro",
            pricing: {
              prompt: "0.0000002",
              completion: "0.0000012",
              input_cache_write_1h: "0.0000004",
              overrides: [{ min_prompt_tokens: 272000, prompt: "0.0000004", completion: "0.0000018", input_cache_read: "0.00000004" }],
            },
          },
        ],
      }),
    );
    expect(row).toMatchObject({
      cacheWrite1hPerMillion: 0.4,
      longContext: { thresholdTokens: 272000, inputPerMillion: 0.4, outputPerMillion: 1.8, cacheReadPerMillion: 0.04 },
    });
  });

  it("captures capabilities: supported_parameters, context_length, max output, modalities", () => {
    const [row] = parseOpenRouterModels(
      JSON.stringify({
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
      }),
    );
    expect(row).toMatchObject({
      route: "openrouter",
      model: "openai/gpt-5-nano",
      provider: "OpenAI",
      structuredOutput: "schema",
      contextLength: 400000,
      maxOutputTokens: 128000,
      modalities: { input: ["text", "image"], output: ["text"] },
    });
    expect(fields(row)).toEqual(["maxOutputTokens", "reasoning"]);
    expect(row?.parameters?.["properties"]).toMatchObject({ maxOutputTokens: { maximum: 128000 } });
    // The reasoning model does NOT list temperature/top_p/top_k — exactly why require_parameters would
    // 404 it if we sent them; the executor filters them out.
    expect(fields(row)).not.toContain("temperature");
  });

  it("reads the per-model `reasoning` object into the levels the model takes", () => {
    // MEASURED 2026-09-24 on anthropic/claude-opus-5.5 (trimmed).
    const [row] = parseOpenRouterModels(
      JSON.stringify({
        data: [
          {
            id: "anthropic/claude-opus-5.5",
            pricing: { prompt: "0.000004", completion: "0.00002" },
            supported_parameters: ["include_reasoning", "max_tokens", "reasoning", "reasoning_effort", "structured_outputs", "temperature", "verbosity"],
            reasoning: { mandatory: true, supported_efforts: ["max", "xhigh", "high", "medium", "low"], default_effort: "high" },
          },
        ],
      }),
    );
    expect(fields(row).sort()).toEqual(["maxOutputTokens", "reasoning", "temperature"]);
    expect(row?.parameters?.["required"]).toEqual(["reasoning"]);
    expect(row?.parameters?.["properties"]).toMatchObject({
      reasoning: { properties: { effort: { enum: ["low", "medium", "high", "xhigh", "max"], default: "high" } } },
    });
  });

  it("returns [] for non-JSON or a missing data array", () => {
    expect(parseOpenRouterModels("not json")).toEqual([]);
    expect(parseOpenRouterModels("{}")).toEqual([]);
  });
});

describe("nativeMirrors — OpenRouter's rows for a provider's own models, as that provider's route", () => {
  const rows = parseOpenRouterModels(
    JSON.stringify({
      data: [
        { id: "anthropic/claude-opus-4.8", pricing: { prompt: "0.000005", completion: "0.000025" } },
        { id: "anthropic/claude-3.7-sonnet:thinking", pricing: { prompt: "0.000003", completion: "0.000015" } },
        {
          id: "openai/gpt-5.6-terra",
          pricing: { prompt: "0.000002", completion: "0.000012" },
          supported_parameters: ["reasoning", "max_tokens"],
          reasoning: { supported_efforts: ["low", "high"] },
        },
        { id: "qwen/qwen3-max", pricing: { prompt: "0.000001", completion: "0.000004" } },
      ],
    }),
  );

  it("restates anthropic/ and openai/ rows under the native id each API takes, and nothing else", () => {
    const mirrors = nativeMirrors(rows);
    expect(mirrors.map(keyForModel)).toEqual(["anthropic/claude-opus-4-8", "openai/gpt-5.6-terra"]);
    expect(mirrors[0]).toMatchObject({ inputPerMillion: 5, outputPerMillion: 25, canonicalId: "claude-opus-4-8" });
  });

  it("drops the budget from an openai mirror — OpenAI's own API has only levels", () => {
    const openai = nativeMirrors(rows).find((r) => r.route === "openai");
    const gate = acceptanceOf(openai?.parameters);
    expect(gate.efforts).toEqual(["low", "high"]);
    expect(gate.acceptsBudget).toBe(false);
  });
});

describe("parseAnthropicModels — Anthropic's /v1/models", () => {
  // MEASURED 2026-09-24 (trimmed to the fields read): an adaptive-only model and a dated budget-only one.
  const page = JSON.stringify({
    data: [
      {
        id: "claude-opus-5-5",
        created_at: "2026-09-21T16:24:00Z",
        max_input_tokens: 1000000,
        max_tokens: 128000,
        capabilities: {
          effort: { supported: true, low: { supported: true }, medium: { supported: true }, high: { supported: true }, xhigh: { supported: true }, max: { supported: true } },
          thinking: { supported: true, types: { enabled: { supported: false }, adaptive: { supported: true } } },
          structured_outputs: { supported: true },
          image_input: { supported: true },
          pdf_input: { supported: true },
        },
      },
      {
        id: "claude-haiku-4-5-20251001",
        max_input_tokens: 200000,
        max_tokens: 64000,
        capabilities: { effort: { supported: false }, thinking: { supported: true, types: { enabled: { supported: true } } } },
      },
    ],
    has_more: true,
    last_id: "claude-haiku-4-5-20251001",
  });

  it("reads exact levels and no budget for an adaptive-only model", () => {
    const { rows } = parseAnthropicModels(page);
    const opus = rows.find((r) => r.model === "claude-opus-5-5");
    expect(opus).toMatchObject({ route: "anthropic", contextLength: 1000000, maxOutputTokens: 128000, structuredOutput: "schema", releasedAt: 1790007840 });
    expect(opus?.modalities).toEqual({ input: ["text", "image", "file"], output: ["text"] });
    expect(opus?.inputPerMillion).toBeUndefined(); // no prices here — the mirror has them
    const gate = acceptanceOf(opus?.parameters);
    expect(gate.efforts).toEqual(["low", "medium", "high", "xhigh", "max"]);
    expect(gate.acceptsBudget).toBe(false);
  });

  it("reads a budget-only model, and adds the alias a dated id is also reached by", () => {
    const { rows } = parseAnthropicModels(page);
    expect(rows.map((r) => r.model)).toEqual(["claude-opus-5-5", "claude-haiku-4-5-20251001", "claude-haiku-4-5"]);
    const gate = acceptanceOf(rows[2]?.parameters);
    expect(gate.efforts).toEqual([]);
    expect(gate.acceptsBudget).toBe(true);
    expect(gate.maxBudget).toBe(63999);
    expect(rows[2]?.canonicalId).toBe("claude-haiku-4-5");
  });

  it("says where the next page starts", () => {
    expect(parseAnthropicModels(page)).toMatchObject({ hasMore: true, lastId: "claude-haiku-4-5-20251001" });
  });

  it("pages through the listing with the account's key", async () => {
    const asked: Array<{ url: string; headers?: Record<string, string> }> = [];
    const source = makeAnthropicModelsSource(async (url, init) => {
      asked.push({ url, ...(init?.headers !== undefined ? { headers: init.headers } : {}) });
      return asked.length === 1 ? page : JSON.stringify({ data: [], has_more: false });
    }, "sk-test");
    await source.fetchRows();
    expect(asked.map((a) => a.url)).toEqual([
      "https://api.anthropic.com/v1/models?limit=100",
      "https://api.anthropic.com/v1/models?limit=100&after_id=claude-haiku-4-5-20251001",
    ]);
    expect(asked[0]?.headers).toMatchObject({ "x-api-key": "sk-test", "anthropic-version": "2023-06-01" });
  });

  it("keeps the sampling knobs off the models the AI SDK says reject them", () => {
    for (const id of ["claude-opus-4-8", "claude-opus-4-7", "anthropic/claude-opus-4.8"]) expect(anthropicRejectsSampling(id)).toBe(true);
    expect(anthropicRejectsSampling("claude-opus-5-5")).toBe(false);
  });
});

describe("sanitizeCatalogRows (lenient per-row drop)", () => {
  it("keeps the good rows and drops the bad ones", () => {
    const { rows, dropped } = sanitizeCatalogRows([
      { route: "openrouter", model: "ok", inputPerMillion: 1, outputPerMillion: 5 },
      { route: "openrouter", model: "bad-output", inputPerMillion: 10, outputPerMillion: 1 },
      { route: "openrouter", model: "ok", inputPerMillion: 1, outputPerMillion: 5 }, // duplicate
    ]);
    expect(rows.map(keyForModel)).toEqual(["openrouter/ok"]);
    expect(dropped.map((d) => d.model).sort()).toEqual(["openrouter/bad-output", "openrouter/ok"]);
  });

  it("KEEPS a zero-output embedding model (input-only billing is legitimate, not a bad read)", () => {
    const { rows, dropped } = sanitizeCatalogRows([{ route: "openrouter", model: "openai/text-embedding-3-small", inputPerMillion: 0.02, outputPerMillion: 0 }]);
    expect(rows.map((r) => r.model)).toEqual(["openai/text-embedding-3-small"]);
    expect(dropped).toEqual([]);
  });

  it("still drops a NEGATIVE-output row (zero is fine, negative is a bad read)", () => {
    const { rows, dropped } = sanitizeCatalogRows([{ route: "openrouter", model: "broken", inputPerMillion: 1, outputPerMillion: -1 }]);
    expect(rows).toEqual([]);
    expect(dropped[0]?.model).toBe("openrouter/broken");
  });

  it("keeps a row with NO price (priced elsewhere), and drops one with half a price", () => {
    const { rows, dropped } = sanitizeCatalogRows([
      { route: "anthropic", model: "claude-opus-5-5" },
      { route: "anthropic", model: "half", inputPerMillion: 1 },
    ]);
    expect(rows.map(keyForModel)).toEqual(["anthropic/claude-opus-5-5"]);
    expect(dropped[0]?.problems).toEqual(["half a price (input or output rate missing)"]);
  });
});

describe("validateCatalogRows", () => {
  it("accepts a sane batch", () => {
    expect(validateCatalogRows(parseAnthropicModels(JSON.stringify({ data: [{ id: "a" }, { id: "b" }, { id: "c" }] })).rows).ok).toBe(true);
  });

  it("rejects a too-small / inconsistent batch", () => {
    expect(validateCatalogRows([]).ok).toBe(false);
    const bad = validateCatalogRows([
      { route: "openrouter", model: "x", inputPerMillion: 10, outputPerMillion: 1 }, // output < input
      { route: "openrouter", model: "y", inputPerMillion: -1, outputPerMillion: 5 }, // non-positive input
      { route: "openrouter", model: "z", inputPerMillion: 1, outputPerMillion: 2, cacheReadPerMillion: 5 }, // read ≥ input
    ]);
    expect(bad.ok).toBe(false);
    expect(bad.problems.length).toBeGreaterThanOrEqual(3);
  });
});

describe("mergeRow — a source's row laid over the table's", () => {
  const current: ModelInfoInterface = { route: "anthropic", model: "m", inputPerMillion: 5, outputPerMillion: 25, contextLength: 1000000, source: "anthropic-models" };

  it("keeps every field the incoming row does not state", () => {
    expect(mergeRow(current, { route: "anthropic", model: "m", maxOutputTokens: 128000 })).toEqual({ ...current, maxOutputTokens: 128000 });
  });

  it("a FILLING row refreshes rates and fills gaps, and overwrites nothing else", () => {
    const merged = mergeRow(current, { route: "anthropic", model: "m", inputPerMillion: 4, outputPerMillion: 20, contextLength: 200000, releasedAt: 1, source: "openrouter-models" }, true);
    expect(merged).toEqual({ ...current, inputPerMillion: 4, outputPerMillion: 20, releasedAt: 1 });
  });
});

describe("refreshModelCatalog", () => {
  it("merges each source over the table in order, and reports what it wrote", async () => {
    const table = new ModelInfo([{ route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 0.8, outputPerMillion: 4 }]);
    const report = await refreshModelCatalog({
      table,
      minRows: 1,
      sources: [
        fixed("prices", [{ route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 }]),
        fixed("caps", [
          { route: "anthropic", model: "claude-haiku-4-5", contextLength: 200000 },
          { route: "anthropic", model: "claude-opus-5-5", contextLength: 1000000 },
        ]),
      ],
    });
    expect(report.updated).toEqual(["anthropic/claude-haiku-4-5"]);
    expect(report.added).toEqual(["anthropic/claude-opus-5-5"]);
    expect(report.changed.map(keyForModel).sort()).toEqual(["anthropic/claude-haiku-4-5", "anthropic/claude-opus-5-5"]);
    // Both sources' facts on one row: the price from the first, the context from the second.
    expect(table.lookup("anthropic/claude-haiku-4-5")).toMatchObject({ inputPerMillion: 1, contextLength: 200000 });
  });

  it("a second refresh with identical data applies nothing (idempotent diff)", async () => {
    const table = new ModelInfo([]);
    const source = fixed("s", [{ route: "openrouter", model: "a/b", inputPerMillion: 1, outputPerMillion: 5 }]);
    await refreshModelCatalog({ sources: [source], table, minRows: 1 });
    const second = await refreshModelCatalog({ sources: [source], table, minRows: 1 });
    expect(second.added).toEqual([]);
    expect(second.updated).toEqual([]);
    expect(second.changed).toEqual([]);
  });

  it("NEVER overwrites good data when a source fails to fetch", async () => {
    const table = new ModelInfo([{ route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 }]);
    const failing: CatalogSource = {
      name: "boom",
      fetchRows: async () => {
        throw new Error("network down");
      },
    };
    const report = await refreshModelCatalog({ sources: [failing], table });
    expect(report.bySource[0]).toMatchObject({ skipped: true, error: "network down" });
    expect(table.computeCostUsd("anthropic/claude-haiku-4-5", 1_000_000, 0)).toBeCloseTo(1, 10);
  });

  it("a lenient source applies its good rows even when the batch has bad ones", async () => {
    const table = new ModelInfo([]);
    const json = JSON.stringify({
      data: [
        { id: "a/good-1", pricing: { prompt: "0.000001", completion: "0.000005" } },
        { id: "a/good-2", pricing: { prompt: "0.000002", completion: "0.000008" } },
        { id: "a/dyn", pricing: { prompt: "-1", completion: "-1" } },
        { id: "a/weird", pricing: { prompt: "0.00001", completion: "0.000001" } }, // output<input
      ],
    });
    const [own] = makeOpenRouterSources(async () => json, "https://example/models");
    const report = await refreshModelCatalog({ sources: [own], table, minRows: 1 });
    expect(report.bySource[0]).toMatchObject({ name: "openrouter-models", skipped: false });
    expect(report.added.sort()).toEqual(["openrouter/a/good-1", "openrouter/a/good-2"]);
    expect(table.hasPricing("openrouter/a/weird")).toBe(false);
  });

  it("the OpenRouter mirrors never undo what the native route's own list said, on a later refresh too", async () => {
    const json = JSON.stringify({
      data: [{ id: "anthropic/claude-opus-5.5", pricing: { prompt: "0.000004", completion: "0.00002" }, context_length: 200000, supported_parameters: ["temperature"] }],
    });
    const table = new ModelInfo([]);
    const anthropic = fixed("anthropic-models", [{ route: "anthropic", model: "claude-opus-5-5", contextLength: 1000000, source: "anthropic-models" }], { fills: false });
    await refreshModelCatalog({ sources: [...makeOpenRouterSources(async () => json), anthropic], table, minRows: 1 });
    // A later refresh that could not reach Anthropic: the mirror refreshes the price, and leaves the rest.
    const json2 = json.replace('"0.000004"', '"0.000003"');
    await refreshModelCatalog({ sources: makeOpenRouterSources(async () => json2), table, minRows: 1 });
    expect(table.lookup("anthropic/claude-opus-5-5")).toMatchObject({ inputPerMillion: 3, contextLength: 1000000, source: "anthropic-models" });
  });

  it("NEVER overwrites good data when a strict source fails validation", async () => {
    const table = new ModelInfo([{ route: "anthropic", model: "claude-haiku-4-5", inputPerMillion: 1, outputPerMillion: 5 }]);
    const report = await refreshModelCatalog({ sources: [fixed("tiny", [])], table });
    expect(report.bySource[0]!.skipped).toBe(true);
    expect(table.computeCostUsd("anthropic/claude-haiku-4-5", 1_000_000, 0)).toBeCloseTo(1, 10);
  });
});
