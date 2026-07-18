import { describe, expect, it } from "vitest";
import {
  canonicalIdFor,
  computeCost,
  computeCostUsd,
  deriveIdentity,
  hasPricing,
  isReasoningModel,
  ModelCatalog,
  SAMPLING_PARAM_NAMES,
  servingProviderFor,
} from "../src/model-catalog";
import type { ProviderSchemaProfile } from "../src/schema";

describe("pricing (default table)", () => {
  it("prices a known model by tokens", () => {
    // claude-haiku-4-5 seed: $1/M in, $5/M out (verified against the Anthropic docs table).
    expect(computeCostUsd("claude-haiku-4-5", 1_000_000, 1_000_000)).toBeCloseTo(6, 10);
    expect(computeCostUsd("claude-haiku-4-5", 10, 5)).toBeCloseTo((10 * 1 + 5 * 5) / 1e6, 12);
  });

  it("matches dated variants via prefix", () => {
    expect(computeCostUsd("claude-opus-4-8-20260115", 0, 0)).toBe(0);
    expect(hasPricing("claude-opus-4-8-20260115")).toBe(true);
  });

  it("falls back to the bare id (direct-provider price) for a vendor-prefixed id", () => {
    expect(computeCostUsd("openai/gpt-4.1-mini", 1_000_000, 0)).toBeCloseTo(0.4, 10);
  });

  it("prefers a vendor-specific row (full id) over the bare-id fallback", () => {
    const table = new ModelCatalog([
      { modelPrefix: "claude-opus-4-8", inputPerMillion: 15, outputPerMillion: 75 }, // direct
      { modelPrefix: "anthropic/claude-opus-4-8", inputPerMillion: 18, outputPerMillion: 90 }, // OpenRouter markup
    ]);
    // Direct id -> direct price; OpenRouter-routed id -> the vendor-specific (full-id) row.
    expect(table.computeCostUsd("claude-opus-4-8", 1_000_000, 0)).toBeCloseTo(15, 10);
    expect(table.computeCostUsd("anthropic/claude-opus-4-8", 1_000_000, 0)).toBeCloseTo(18, 10);
  });

  it("returns null for an unknown model rather than guessing", () => {
    expect(computeCostUsd("some-unknown-model", 100, 100)).toBeNull();
    expect(hasPricing("some-unknown-model")).toBe(false);
  });

  it("treats null token counts as zero", () => {
    expect(computeCostUsd("claude-haiku-4-5", null, null)).toBe(0);
  });
});

describe("computeCost (cache-aware, §5/§10.2)", () => {
  // A fixed local table so the arithmetic is decoupled from seed drift (the cron updates seeds).
  // $0.8/M in, $4/M out; default multipliers: read 0.1x, write-5m 1.25x, write-1h 2x.
  const inRate = 0.8;
  const t = new ModelCatalog([{ modelPrefix: "hk", inputPerMillion: 0.8, outputPerMillion: 4 }]);

  it("prices cache reads at the discounted rate, not the full input rate", () => {
    // 1M fresh input + 1M cache-read + 0 output. The cache read should cost 0.1x.
    const cost = t.computeCost("hk", {
      inputTokens: 2_000_000, // cache-inclusive total (ignored when the split is present)
      outputTokens: 0,
      noCacheTokens: 1_000_000,
      cacheReadTokens: 1_000_000,
    });
    expect(cost).toBeCloseTo(inRate * 1 + inRate * 0.1 * 1, 10); // 0.8 + 0.08 = 0.88
  });

  it("prices cache writes at the write rate", () => {
    const cost = t.computeCost("hk", {
      inputTokens: 1_000_000,
      outputTokens: 0,
      noCacheTokens: 0,
      cacheWriteTokens: 1_000_000,
    });
    expect(cost).toBeCloseTo(inRate * 1.25, 10); // 1.0
  });

  it("over-counts vs flat pricing exactly by the cache discount (regression guard)", () => {
    const usage = { noCacheTokens: 0, cacheReadTokens: 1_000_000, outputTokens: 0 };
    const accurate = t.computeCost("hk", usage)!;
    const flatWrong = t.computeCostUsd("hk", 1_000_000, 0)!; // old behavior: full input rate
    expect(accurate).toBeCloseTo(0.08, 10);
    expect(flatWrong).toBeCloseTo(0.8, 10);
    expect(flatWrong / accurate).toBeCloseTo(10, 6); // a 10x over-charge on a pure cache hit
  });

  it("falls back to flat input pricing when no cache split is reported", () => {
    expect(t.computeCost("hk", { inputTokens: 10, outputTokens: 5 })).toBeCloseTo(
      (10 * 0.8 + 5 * 4) / 1e6,
      12,
    );
  });

  it("honors explicit per-row cache rates and custom multipliers", () => {
    const explicit = new ModelCatalog([
      { modelPrefix: "m", inputPerMillion: 10, outputPerMillion: 30, cacheReadPerMillion: 0.5 },
    ]);
    expect(explicit.computeCost("m", { cacheReadTokens: 1_000_000 })).toBeCloseTo(0.5, 10);

    const custom = new ModelCatalog([{ modelPrefix: "m", inputPerMillion: 10, outputPerMillion: 30 }], {
      cacheReadMultiplier: 0.2,
    });
    expect(custom.computeCost("m", { cacheReadTokens: 1_000_000 })).toBeCloseTo(2, 10); // 10 * 0.2
  });

  it("prices 1-hour cache writes at 2x and 5-min writes at 1.25x", () => {
    // 1M writes, of which 400k are 1-hour TTL. 5-min slice = 600k.
    const cost = t.computeCost("hk", {
      noCacheTokens: 0,
      cacheWriteTokens: 1_000_000,
      cacheWrite1hTokens: 400_000,
      outputTokens: 0,
    });
    // 600k * (0.8*1.25) + 400k * (0.8*2.0) = 600k*1.0 + 400k*1.6 = 0.6 + 0.64 = 1.24
    expect(cost).toBeCloseTo((600_000 * 0.8 * 1.25 + 400_000 * 0.8 * 2.0) / 1e6, 10);
  });

  it("treats all writes as 5-min when no 1-hour slice is reported", () => {
    expect(
      t.computeCost("hk", { noCacheTokens: 0, cacheWriteTokens: 1_000_000, outputTokens: 0 }),
    ).toBeCloseTo(0.8 * 1.25, 10); // 1.0
  });

  it("clamps a 1-hour slice that exceeds total writes", () => {
    const cost = t.computeCost("hk", {
      cacheWriteTokens: 100,
      cacheWrite1hTokens: 999_999, // nonsense; must clamp to 100
      outputTokens: 0,
    });
    expect(cost).toBeCloseTo((100 * 0.8 * 2.0) / 1e6, 12);
  });
});

describe("long-context tier (mechanism)", () => {
  // Local table: current Anthropic models price 1M context at STANDARD rates (no premium seeded),
  // so the tier mechanism is tested against a synthetic row, not a live seed.
  const t = new ModelCatalog([
    {
      modelPrefix: "lc",
      inputPerMillion: 3,
      outputPerMillion: 15,
      longContext: { thresholdTokens: 200_000, inputPerMillion: 6, outputPerMillion: 22.5 },
    },
    { modelPrefix: "flat", inputPerMillion: 3, outputPerMillion: 15 },
  ]);

  it("uses base rates below the threshold and premium rates above it", () => {
    const below = t.computeCost("lc", { inputTokens: 100_000, outputTokens: 0, noCacheTokens: 100_000 });
    expect(below).toBeCloseTo((100_000 * 3) / 1e6, 10);

    const above = t.computeCost("lc", { inputTokens: 300_000, outputTokens: 10_000, noCacheTokens: 300_000 });
    // ALL tokens repriced at the premium set: 300k input @ $6, 10k output @ $22.50.
    expect(above).toBeCloseTo((300_000 * 6 + 10_000 * 22.5) / 1e6, 10);
  });

  it("scales cache rates with the premium input rate above the threshold", () => {
    // Above 200K: input $6 → cache read 0.6. 250k read + 250k input.
    const cost = t.computeCost("lc", {
      inputTokens: 500_000,
      noCacheTokens: 250_000,
      cacheReadTokens: 250_000,
      outputTokens: 0,
    });
    expect(cost).toBeCloseTo((250_000 * 6 + 250_000 * 6 * 0.1) / 1e6, 10);
  });

  it("a row without a long-context tier keeps base rates at any size", () => {
    const cost = t.computeCost("flat", { inputTokens: 1_000_000, outputTokens: 0, noCacheTokens: 1_000_000 });
    expect(cost).toBeCloseTo((1_000_000 * 3) / 1e6, 10);
  });

  it("returns null for an unknown model", () => {
    expect(computeCost("nope", { inputTokens: 100, cacheReadTokens: 50 })).toBeNull();
  });
});

describe("ModelCatalog (updatable)", () => {
  it("supports upsert / remove at runtime", () => {
    const table = new ModelCatalog();
    expect(table.hasPricing("new-model-1")).toBe(false);

    table.upsert({ modelPrefix: "new-model-1", inputPerMillion: 2, outputPerMillion: 6 });
    expect(table.computeCostUsd("new-model-1-v2", 1_000_000, 1_000_000)).toBeCloseTo(8, 10);

    // re-price (prices change)
    table.upsert({ modelPrefix: "new-model-1", inputPerMillion: 1, outputPerMillion: 1 });
    expect(table.computeCostUsd("new-model-1", 1_000_000, 0)).toBeCloseTo(1, 10);

    table.remove("new-model-1");
    expect(table.hasPricing("new-model-1")).toBe(false);
  });

  it("matches longest prefix wins, regardless of insertion order", () => {
    const table = new ModelCatalog([
      { modelPrefix: "foo", inputPerMillion: 1, outputPerMillion: 1 },
      { modelPrefix: "foo-pro", inputPerMillion: 10, outputPerMillion: 10 },
    ]);
    // "foo-pro-x" matches both "foo" and "foo-pro"; the longer prefix wins.
    expect(table.computeCostUsd("foo-pro-x", 1_000_000, 0)).toBeCloseTo(10, 10);
    expect(table.computeCostUsd("foo-lite", 1_000_000, 0)).toBeCloseTo(1, 10);
  });

  it("bulk-loads rows (DB hydration path)", () => {
    const table = new ModelCatalog();
    table.load([
      { modelPrefix: "a", inputPerMillion: 1, outputPerMillion: 2 },
      { modelPrefix: "b", inputPerMillion: 3, outputPerMillion: 4 },
    ]);
    expect(table.list()).toHaveLength(2);
  });
});

describe("capabilities (param filtering substrate, §5.1)", () => {
  it("flags OpenAI reasoning families (GPT-5*, o-series) and nothing else", () => {
    for (const id of ["gpt-5-nano", "gpt-5.4-mini", "openai/gpt-5-mini", "o3", "o4-mini", "o3-mini"]) {
      expect(isReasoningModel(id)).toBe(true);
    }
    for (const id of ["gpt-4.1-mini", "gpt-4o", "claude-opus-4-8", "anthropic/claude-haiku-4-5", "gemini-2.5-pro"]) {
      expect(isReasoningModel(id)).toBe(false);
    }
  });

  it("returns the RECORDED supportedParameters when present (data wins over heuristic)", () => {
    const c = new ModelCatalog([
      {
        modelPrefix: "openai/gpt-5-nano",
        inputPerMillion: 0.25,
        outputPerMillion: 1,
        supportedParameters: ["temperature", "response_format"], // pretend it DID accept temperature
      },
    ]);
    expect(c.supportedParametersFor("openai/gpt-5-nano")).toContain("temperature");
  });

  it("falls back to a temperature-LESS set for a reasoning model with no recorded capabilities", () => {
    const c = new ModelCatalog(); // empty — cold catalog
    const supported = c.supportedParametersFor("openai/gpt-5-nano");
    expect(supported).toBeDefined();
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.temperature);
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.topP);
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.topK);
    // structured output still works → response_format/structured_outputs ARE present.
    expect(supported).toContain("response_format");
    expect(supported).toContain("structured_outputs");
  });

  it("returns undefined for an unknown non-reasoning model (⇒ caller sends everything)", () => {
    const c = new ModelCatalog();
    expect(c.supportedParametersFor("some/unknown-chat-model")).toBeUndefined();
    expect(c.supportedParametersFor("gpt-4.1-mini")).toBeUndefined();
  });

  it("native Claude capabilities are NOT computed by the catalog — they come from the recorded row (table)", () => {
    const c = new ModelCatalog(); // cold catalog: no `claude-opus-4-8` row
    // The catalog no longer synthesizes Claude caps; an un-seeded model just returns undefined (send all).
    expect(c.supportedParametersFor("claude-opus-4-8")).toBeUndefined();
    // With the row present (as migration 0011 seeds it), the recorded sampling-less set is returned.
    const seeded = new ModelCatalog([
      { modelPrefix: "claude-opus-4-8", inputPerMillion: 5, outputPerMillion: 25, supportedParameters: ["max_tokens", "stop", "tools", "tool_choice", "reasoning"] },
    ]);
    const supported = seeded.supportedParametersFor("claude-opus-4-8");
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.temperature);
    expect(supported).toContain(SAMPLING_PARAM_NAMES.stopSequences);
  });

  it("exposes the recorded (resolved) schema profile, longest-prefix matched", () => {
    const profile = { id: "openrouter:strict", supportsStructuredOutput: "schema" } as unknown as ProviderSchemaProfile;
    const c = new ModelCatalog([
      { modelPrefix: "openai/gpt-5", inputPerMillion: 1, outputPerMillion: 4, schemaProfile: profile },
    ]);
    expect(c.schemaProfileFor("openai/gpt-5-2026-01-01")?.id).toBe("openrouter:strict");
    expect(c.schemaProfileFor("anthropic/claude-opus-4-8")).toBeUndefined();
  });
});

describe("model identity derivation (canonical id / serving provider)", () => {
  it("canonicalIdFor collapses native + OpenRouter routes onto one id (drop vendor, dots→hyphens)", () => {
    expect(canonicalIdFor("claude-opus-4-8")).toBe("claude-opus-4-8");
    expect(canonicalIdFor("anthropic/claude-opus-4.8")).toBe("claude-opus-4-8");
    expect(canonicalIdFor("openai/gpt-4.1")).toBe("gpt-4-1");
  });

  it("servingProviderFor distinguishes native Anthropic from OpenRouter by the id shape", () => {
    expect(servingProviderFor("claude-opus-4-8")).toBe("anthropic"); // bare claude- → native API
    expect(servingProviderFor("anthropic/claude-opus-4.8")).toBe("openrouter"); // vendor/ → OpenRouter
    expect(servingProviderFor("openai/gpt-4.1")).toBe("openrouter");
    expect(servingProviderFor("gpt-4o")).toBe("openrouter"); // bare non-claude routes via openai/ on OR
  });

  it("deriveIdentity fills generic identity but NEVER capabilities (that's the table/ingestion path's job)", () => {
    const row = deriveIdentity({ modelPrefix: "anthropic/claude-opus-4.8", inputPerMillion: 5, outputPerMillion: 25 });
    expect(row.canonicalId).toBe("claude-opus-4-8");
    expect(row.servingProvider).toBe("openrouter");
    expect(row.family).toBe("openrouter");
    expect(row.provider).toBe("Anthropic");
    expect(row.supportedParameters).toBeUndefined(); // identity only — no capability synthesis here
    // Fill-only: an explicit value is kept.
    const fed = deriveIdentity({ modelPrefix: "x/y", inputPerMillion: 1, outputPerMillion: 1, supportedParameters: ["temperature"] });
    expect(fed.supportedParameters).toEqual(["temperature"]);
  });
});
