import { describe, expect, it } from "vitest";
import {
  canonicalIdFor,
  deriveIdentity,
  isReasoningModel,
  ModelInfo,
  SAMPLING_PARAM_NAMES,
} from "../src/model-catalog";
import type { ProviderSchemaProfile } from "../src/schema";

describe("pricing (default catalog)", () => {
  const catalog = ModelInfo.instance;

  it("prices a known model by tokens (keyed on the full {route}/{model} id)", () => {
    // Read the rates off the row so this stays correct across snapshot refreshes — it verifies the default
    // catalog is WIRED + computes, not a specific price (the arithmetic is pinned in the local-table blocks).
    const id = "anthropic/claude-haiku-4-5";
    const { inputPerMillion: i, outputPerMillion: o } = catalog.lookup(id)!;
    expect(catalog.computeCostUsd(id, 1_000_000, 1_000_000)).toBeCloseTo(i + o, 10);
    expect(catalog.computeCostUsd(id, 10, 5)).toBeCloseTo((10 * i + 5 * o) / 1e6, 12);
  });

  it("matches EXACTLY — a dated/variant id is unknown unless seeded (no prefix fallback)", () => {
    expect(catalog.hasPricing("anthropic/claude-opus-4-8")).toBe(true);
    expect(catalog.hasPricing("anthropic/claude-opus-4-8-20260115")).toBe(false);
    expect(catalog.computeCostUsd("anthropic/claude-opus-4-8-20260115", 1_000_000, 0)).toBeNull();
  });

  it("prices an OpenAI model on its OpenRouter route id", () => {
    const id = "openrouter/openai/gpt-4.1-mini";
    const { inputPerMillion: i } = catalog.lookup(id)!;
    expect(catalog.computeCostUsd(id, 1_000_000, 0)).toBeCloseTo(i, 10);
  });

  it("keeps native and OpenRouter routes as DISTINCT keys (different prices)", () => {
    const table = new ModelInfo([
      { route: "anthropic", model: "claude-opus-4-8", inputPerMillion: 15, outputPerMillion: 75 }, // native
      { route: "openrouter", model: "anthropic/claude-opus-4.8", inputPerMillion: 18, outputPerMillion: 90 }, // OR markup
    ]);
    expect(table.computeCostUsd("anthropic/claude-opus-4-8", 1_000_000, 0)).toBeCloseTo(15, 10);
    expect(table.computeCostUsd("openrouter/anthropic/claude-opus-4.8", 1_000_000, 0)).toBeCloseTo(18, 10);
  });

  it("returns null for an unknown model rather than guessing", () => {
    expect(catalog.computeCostUsd("openrouter/some-unknown-model", 100, 100)).toBeNull();
    expect(catalog.hasPricing("openrouter/some-unknown-model")).toBe(false);
  });

  it("treats null token counts as zero", () => {
    expect(catalog.computeCostUsd("anthropic/claude-haiku-4-5", null, null)).toBe(0);
  });

  it("affordableOutputTokens: output tokens the remaining balance buys after the input cost", () => {
    // $15/1M input, $75/1M output. Input 1M costs $15; each output token costs $75/1M.
    const c = new ModelInfo([{ route: "anthropic", model: "m", inputPerMillion: 15, outputPerMillion: 75 }]);
    // $90 available, $15 spent on input → $75 headroom → 1,000,000 output tokens.
    expect(c.affordableOutputTokens("anthropic/m", 1_000_000, 90)).toBe(1_000_000);
    // Balance below the input cost → nothing affordable.
    expect(c.affordableOutputTokens("anthropic/m", 1_000_000, 10)).toBe(0);
    // Un-priced model → no clamp basis.
    expect(c.affordableOutputTokens("openrouter/unknown", 100, 5)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("compile-time model keys (generic ModelInfo)", () => {
  it("a literal seed types the methods to its exact keys (unknown model fails to COMPILE)", () => {
    // Constructed with a literal array (the `const` type param preserves the keys), so the method params
    // are the exact `${route}/${model}` union — the @ts-expect-error below is verified by `tsc`.
    const c = new ModelInfo([
      { route: "anthropic", model: "claude-opus-4-8", inputPerMillion: 5, outputPerMillion: 25 },
      { route: "openrouter", model: "openai/gpt-5", inputPerMillion: 3.75, outputPerMillion: 15 },
    ]);
    expect(c.computeCostUsd("anthropic/claude-opus-4-8", 1_000_000, 0)).toBeCloseTo(5, 10);
    expect(c.hasPricing("openrouter/openai/gpt-5")).toBe(true);
    // @ts-expect-error — "anthropic/nope" is not one of the seeded `${route}/${model}` keys.
    expect(c.computeCost("anthropic/nope", { inputTokens: 1 })).toBeNull();
  });
});

describe("computeCost (cache-aware, §5/§10.2)", () => {
  // A fixed local catalog so the arithmetic is decoupled from seed drift (the cron updates seeds).
  // $0.8/M in, $4/M out; default multipliers: read 0.1x, write-5m 1.25x, write-1h 2x.
  const inRate = 0.8;
  const t = new ModelInfo([{ route: "openrouter", model: "hk", inputPerMillion: 0.8, outputPerMillion: 4 }]);

  it("prices cache reads at the discounted rate, not the full input rate", () => {
    // 1M fresh input + 1M cache-read + 0 output. The cache read should cost 0.1x.
    const cost = t.computeCost("openrouter/hk", {
      inputTokens: 2_000_000, // cache-inclusive total (ignored when the split is present)
      outputTokens: 0,
      noCacheTokens: 1_000_000,
      cacheReadTokens: 1_000_000,
    });
    expect(cost).toBeCloseTo(inRate * 1 + inRate * 0.1 * 1, 10); // 0.8 + 0.08 = 0.88
  });

  it("prices cache writes at the write rate", () => {
    const cost = t.computeCost("openrouter/hk", {
      inputTokens: 1_000_000,
      outputTokens: 0,
      noCacheTokens: 0,
      cacheWriteTokens: 1_000_000,
    });
    expect(cost).toBeCloseTo(inRate * 1.25, 10); // 1.0
  });

  it("over-counts vs flat pricing exactly by the cache discount (regression guard)", () => {
    const usage = { noCacheTokens: 0, cacheReadTokens: 1_000_000, outputTokens: 0 };
    const accurate = t.computeCost("openrouter/hk", usage)!;
    const flatWrong = t.computeCostUsd("openrouter/hk", 1_000_000, 0)!; // old behavior: full input rate
    expect(accurate).toBeCloseTo(0.08, 10);
    expect(flatWrong).toBeCloseTo(0.8, 10);
    expect(flatWrong / accurate).toBeCloseTo(10, 6); // a 10x over-charge on a pure cache hit
  });

  it("falls back to flat input pricing when no cache split is reported", () => {
    expect(t.computeCost("openrouter/hk", { inputTokens: 10, outputTokens: 5 })).toBeCloseTo(
      (10 * 0.8 + 5 * 4) / 1e6,
      12,
    );
  });

  it("honors explicit per-row cache rates and custom multipliers", () => {
    const explicit = new ModelInfo([
      { route: "openrouter", model: "m", inputPerMillion: 10, outputPerMillion: 30, cacheReadPerMillion: 0.5 },
    ]);
    expect(explicit.computeCost("openrouter/m", { cacheReadTokens: 1_000_000 })).toBeCloseTo(0.5, 10);

    const custom = new ModelInfo([{ route: "openrouter", model: "m", inputPerMillion: 10, outputPerMillion: 30 }], {
      cacheReadMultiplier: 0.2,
    });
    expect(custom.computeCost("openrouter/m", { cacheReadTokens: 1_000_000 })).toBeCloseTo(2, 10); // 10 * 0.2
  });

  it("prices 1-hour cache writes at 2x and 5-min writes at 1.25x", () => {
    // 1M writes, of which 400k are 1-hour TTL. 5-min slice = 600k.
    const cost = t.computeCost("openrouter/hk", {
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
      t.computeCost("openrouter/hk", { noCacheTokens: 0, cacheWriteTokens: 1_000_000, outputTokens: 0 }),
    ).toBeCloseTo(0.8 * 1.25, 10); // 1.0
  });

  it("clamps a 1-hour slice that exceeds total writes", () => {
    const cost = t.computeCost("openrouter/hk", {
      cacheWriteTokens: 100,
      cacheWrite1hTokens: 999_999, // nonsense; must clamp to 100
      outputTokens: 0,
    });
    expect(cost).toBeCloseTo((100 * 0.8 * 2.0) / 1e6, 12);
  });
});

describe("long-context tier (mechanism)", () => {
  // Local catalog: current Anthropic models price 1M context at STANDARD rates (no premium seeded),
  // so the tier mechanism is tested against a synthetic row, not a live seed.
  // Weakly typed (`: ModelInfo`) so the "unknown model → null" probe below can pass a string key.
  const t: ModelInfo = new ModelInfo([
    {
      route: "anthropic",
      model: "lc",
      inputPerMillion: 3,
      outputPerMillion: 15,
      longContext: { thresholdTokens: 200_000, inputPerMillion: 6, outputPerMillion: 22.5 },
    },
    { route: "anthropic", model: "flat", inputPerMillion: 3, outputPerMillion: 15 },
  ]);

  it("uses base rates below the threshold and premium rates above it", () => {
    const below = t.computeCost("anthropic/lc", { inputTokens: 100_000, outputTokens: 0, noCacheTokens: 100_000 });
    expect(below).toBeCloseTo((100_000 * 3) / 1e6, 10);

    const above = t.computeCost("anthropic/lc", { inputTokens: 300_000, outputTokens: 10_000, noCacheTokens: 300_000 });
    // ALL tokens repriced at the premium set: 300k input @ $6, 10k output @ $22.50.
    expect(above).toBeCloseTo((300_000 * 6 + 10_000 * 22.5) / 1e6, 10);
  });

  it("scales cache rates with the premium input rate above the threshold", () => {
    // Above 200K: input $6 → cache read 0.6. 250k read + 250k input.
    const cost = t.computeCost("anthropic/lc", {
      inputTokens: 500_000,
      noCacheTokens: 250_000,
      cacheReadTokens: 250_000,
      outputTokens: 0,
    });
    expect(cost).toBeCloseTo((250_000 * 6 + 250_000 * 6 * 0.1) / 1e6, 10);
  });

  it("a row without a long-context tier keeps base rates at any size", () => {
    const cost = t.computeCost("anthropic/flat", { inputTokens: 1_000_000, outputTokens: 0, noCacheTokens: 1_000_000 });
    expect(cost).toBeCloseTo((1_000_000 * 3) / 1e6, 10);
  });

  it("returns null for an unknown model", () => {
    expect(t.computeCost("anthropic/nope", { inputTokens: 100, cacheReadTokens: 50 })).toBeNull();
  });
});

describe("ModelInfo (updatable)", () => {
  it("supports upsert / remove at runtime", () => {
    const table = new ModelInfo([]);
    expect(table.hasPricing("openrouter/new-model-1")).toBe(false);

    table.upsert({ route: "openrouter", model: "new-model-1", inputPerMillion: 2, outputPerMillion: 6 });
    expect(table.computeCostUsd("openrouter/new-model-1", 1_000_000, 1_000_000)).toBeCloseTo(8, 10);

    // re-price (prices change)
    table.upsert({ route: "openrouter", model: "new-model-1", inputPerMillion: 1, outputPerMillion: 1 });
    expect(table.computeCostUsd("openrouter/new-model-1", 1_000_000, 0)).toBeCloseTo(1, 10);

    table.remove("openrouter/new-model-1");
    expect(table.hasPricing("openrouter/new-model-1")).toBe(false);
  });

  it("matches on the EXACT {route}/{model} key (no prefix matching)", () => {
    // Weakly typed so the "extends a key but isn't it → null" probe below can pass a string key.
    const table: ModelInfo = new ModelInfo([
      { route: "openrouter", model: "foo", inputPerMillion: 1, outputPerMillion: 1 },
      { route: "openrouter", model: "foo-pro", inputPerMillion: 10, outputPerMillion: 10 },
    ]);
    expect(table.computeCostUsd("openrouter/foo", 1_000_000, 0)).toBeCloseTo(1, 10);
    expect(table.computeCostUsd("openrouter/foo-pro", 1_000_000, 0)).toBeCloseTo(10, 10);
    // A longer id that merely EXTENDS a key no longer resolves — it's a different model.
    expect(table.computeCostUsd("openrouter/foo-pro-x", 1_000_000, 0)).toBeNull();
  });

  it("bulk-loads rows (hydration path)", () => {
    const table = new ModelInfo([]);
    table.load([
      { route: "openrouter", model: "a", inputPerMillion: 1, outputPerMillion: 2 },
      { route: "openrouter", model: "b", inputPerMillion: 3, outputPerMillion: 4 },
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
    const c = new ModelInfo([
      {
        route: "openrouter",
        model: "openai/gpt-5-nano",
        inputPerMillion: 0.25,
        outputPerMillion: 1,
        supportedParameters: ["temperature", "response_format"], // pretend it DID accept temperature
      },
    ]);
    expect(c.supportedParameters("openrouter/openai/gpt-5-nano")).toContain("temperature");
  });

  it("falls back to a temperature-LESS set for a reasoning model with no recorded capabilities", () => {
    const c = new ModelInfo([]); // empty — cold catalog
    const supported = c.supportedParameters("openrouter/openai/gpt-5-nano");
    expect(supported).toBeDefined();
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.temperature);
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.topP);
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.topK);
    // structured output still works → response_format/structured_outputs ARE present.
    expect(supported).toContain("response_format");
    expect(supported).toContain("structured_outputs");
  });

  it("returns undefined for an unknown non-reasoning model (⇒ caller sends everything)", () => {
    const c = new ModelInfo([]);
    expect(c.supportedParameters("openrouter/some/unknown-chat-model")).toBeUndefined();
    expect(c.supportedParameters("openrouter/openai/gpt-4.1-mini")).toBeUndefined();
  });

  it("native Claude capabilities are NOT computed by the catalog — they come from the recorded row", () => {
    const c = new ModelInfo([]); // cold catalog: no claude-opus-4-8 row
    // The catalog no longer synthesizes Claude caps; an un-seeded model just returns undefined (send all).
    expect(c.supportedParameters("anthropic/claude-opus-4-8")).toBeUndefined();
    // With the row present (as the ingestion path seeds it), the recorded sampling-less set is returned.
    const seeded = new ModelInfo([
      {
        route: "anthropic",
        model: "claude-opus-4-8",
        inputPerMillion: 5,
        outputPerMillion: 25,
        supportedParameters: ["max_tokens", "stop", "tools", "tool_choice", "reasoning"],
      },
    ]);
    const supported = seeded.supportedParameters("anthropic/claude-opus-4-8");
    expect(supported).not.toContain(SAMPLING_PARAM_NAMES.temperature);
    expect(supported).toContain(SAMPLING_PARAM_NAMES.stopSequences);
  });

  it("exposes the recorded (resolved) schema profile, exact-key matched", () => {
    const profile = { id: "openrouter:strict", supportsStructuredOutput: "schema" } as unknown as ProviderSchemaProfile;
    // Weakly typed so the exact-only probes (dated variant, other route) can pass string keys.
    const c: ModelInfo = new ModelInfo([
      { route: "openrouter", model: "openai/gpt-5", inputPerMillion: 1, outputPerMillion: 4, schemaProfile: profile },
    ]);
    expect(c.schemaProfile("openrouter/openai/gpt-5")?.id).toBe("openrouter:strict");
    expect(c.schemaProfile("openrouter/openai/gpt-5-2026-01-01")).toBeUndefined(); // exact only
    expect(c.schemaProfile("anthropic/claude-opus-4-8")).toBeUndefined();
  });
});

describe("model identity derivation (canonical id / display)", () => {
  it("canonicalIdFor collapses native + OpenRouter routes onto one id (drop vendor, dots→hyphens)", () => {
    expect(canonicalIdFor("claude-opus-4-8")).toBe("claude-opus-4-8");
    expect(canonicalIdFor("anthropic/claude-opus-4.8")).toBe("claude-opus-4-8");
    expect(canonicalIdFor("openai/gpt-4.1")).toBe("gpt-4-1");
  });

  it("deriveIdentity fills generic identity but NEVER capabilities (that's the ingestion path's job)", () => {
    const row = deriveIdentity({ route: "openrouter", model: "anthropic/claude-opus-4.8", inputPerMillion: 5, outputPerMillion: 25 });
    expect(row.canonicalId).toBe("claude-opus-4-8");
    expect(row.provider).toBe("Anthropic");
    expect(row.label).toBe("claude-opus-4.8");
    expect(row.supportedParameters).toBeUndefined(); // identity only — no capability synthesis here
    // Fill-only: an explicit value is kept.
    const fed = deriveIdentity({ route: "openrouter", model: "x/y", inputPerMillion: 1, outputPerMillion: 1, supportedParameters: ["temperature"] });
    expect(fed.supportedParameters).toEqual(["temperature"]);
  });
});
