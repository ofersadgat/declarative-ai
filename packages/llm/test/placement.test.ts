import { existsSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EmbeddedModelStore, catalogRowForGguf, classifyPlacement, embeddedPlacementProbe } from "../src/embedded.js";
import { ModelInfo } from "../src/model-catalog.js";
import { ResidencyManager } from "../src/residency.js";
import { createModelRouter } from "../src/router.js";
import { executeLlmCall } from "../src/call.js";
import { errorOf } from "./fakes.js";

const GGUF =
  process.env.DAI_TEST_GGUF ??
  path.join(
    process.env.TEMP ?? "/tmp",
    "claude/C--UbuntuCode-declarative-ai/875c7345-d8b5-48b1-bd9c-b2b100d1b991/scratchpad/nlc-spike/models/hf_Qwen_qwen2.5-0.5b-instruct-q4_k_m.gguf",
  );
const withWeights = existsSync(GGUF) ? it : it.skip;

const GB = 1024 ** 3;

describe("classifyPlacement", () => {
  // The numbers are the real ones the spike measured for Qwen2.5-32B-Q4_K_M (18.48 GB, 65 layers)
  // against simulated machines — the same arithmetic, checked without needing that hardware present.
  const layers = 65;

  it("full offload that fits free VRAM is tier 1", () => {
    const p = classifyPlacement({ gpuLayers: 65, contextSize: 4096, totalVramUsage: 19.37 * GB, totalRamUsage: 0.43 * GB }, layers, {
      vramFree: 24 * GB,
      ramFree: 128 * GB,
    });
    expect(p.tier).toBe("vram");
    expect(p.gpuLayers).toBe(65);
  });

  it("PARTIAL offload is tier 2, even when everything fits comfortably in RAM", () => {
    // The 4090 at 32k context: 57/65 layers, 3.78 GB in RAM. Works, slower — and the user must be told.
    const p = classifyPlacement({ gpuLayers: 57, contextSize: 32_768, totalVramUsage: 23.05 * GB, totalRamUsage: 3.78 * GB }, layers, {
      vramFree: 24 * GB,
      ramFree: 128 * GB,
    });
    expect(p.tier).toBe("ram");
  });

  it("a full-layer offload that EXCEEDS free VRAM is still tier 2", () => {
    // Free VRAM moves between predicting and loading — something else took the card. All layers being
    // assigned is not by itself proof they fit.
    const p = classifyPlacement({ gpuLayers: 65, contextSize: 4096, totalVramUsage: 19 * GB, totalRamUsage: 0.4 * GB }, layers, {
      vramFree: 11 * GB, // exactly the situation nvidia-smi showed mid-spike
      ramFree: 128 * GB,
    });
    expect(p.tier).toBe("ram");
  });

  it("a working set past free RAM is tier 3, NOT merely a partial offload", () => {
    // The ordering that matters. Swap is reached by the same partial-offload path as tier 2, so testing
    // "did layers come off the GPU" first would label a model about to thrash the pagefile as merely
    // degraded — the one mistake the three-tier split exists to prevent.
    const p = classifyPlacement({ gpuLayers: 16, contextSize: 32_768, totalVramUsage: 6.99 * GB, totalRamUsage: 19.84 * GB }, layers, {
      vramFree: 8 * GB,
      ramFree: 16 * GB, // 19.84 > 16 — the remainder goes to the pagefile
    });
    expect(p.tier).toBe("swap");
  });

  it("no GPU at all with enough RAM is tier 2, not tier 3", () => {
    // CPU-only inference is slow but legitimate; calling it "swap" would refuse a machine that works.
    const p = classifyPlacement({ gpuLayers: 0, contextSize: 4096, totalVramUsage: 0.89 * GB, totalRamUsage: 19.51 * GB }, layers, {
      vramFree: 0,
      ramFree: 64 * GB,
    });
    expect(p.tier).toBe("ram");
  });

  it("carries the prediction through verbatim for the policy to read", () => {
    const p = classifyPlacement({ gpuLayers: 21, contextSize: 8192, totalVramUsage: 6.74 * GB, totalRamUsage: 13.07 * GB }, layers, {
      vramFree: 12 * GB,
      ramFree: 32 * GB,
    });
    expect(p).toMatchObject({ gpuLayers: 21, totalLayers: 65, contextSize: 8192, vramBytes: 6.74 * GB, ramBytes: 13.07 * GB });
  });
});

describe("embeddedPlacementProbe", () => {
  const store = new EmbeddedModelStore((id) => (id === "known" ? { modelPath: GGUF, contextSize: 512 } : undefined));

  it("ABSTAINS for a model id it cannot map to a provider id", async () => {
    // A remote model reaching the probe is normal — the manager asks about whatever it is given.
    const probe = embeddedPlacementProbe(store, () => undefined);
    await expect(probe.predict("anthropic/claude-haiku-4-5")).resolves.toBeUndefined();
  });

  it("ABSTAINS for a model with no embedded configuration", async () => {
    const probe = embeddedPlacementProbe(store, (id) => id.split("/")[1]);
    await expect(probe.predict("embedded/unconfigured")).resolves.toBeUndefined();
  });

  it("ABSTAINS rather than throwing when the weights cannot be read", async () => {
    // Admitting unjudged is the deliberate choice: a probe that guessed would be worse than one that
    // says nothing, and a probe fault must not fail a call the machine could have run.
    const missing = new EmbeddedModelStore({ modelPath: "/definitely/not/here.gguf" });
    const probe = embeddedPlacementProbe(missing, (id) => id.split("/")[1]);
    await expect(probe.predict("embedded/anything")).resolves.toBeUndefined();
  }, 30_000);

  withWeights(
    "predicts a real placement for real weights on this machine",
    async () => {
      const probe = embeddedPlacementProbe(store, (id) => id.split("/")[1]);
      const placement = await probe.predict("embedded/known");
      expect(placement).toBeDefined();
      expect(placement!.totalLayers).toBeGreaterThan(0);
      expect(placement!.gpuLayers).toBeLessThanOrEqual(placement!.totalLayers);
      expect(placement!.contextSize).toBeGreaterThan(0);
      // A 0.5B model on any machine that can run this suite fits entirely.
      expect(placement!.tier).toBe("vram");
    },
    120_000,
  );
});

describe("catalogRowForGguf", () => {
  withWeights(
    "reads the row out of the weights themselves",
    async () => {
      const row = await catalogRowForGguf({
        model: "qwen2.5-0.5b-instruct-q4_k_m",
        source: GGUF,
        downloads: [{ source: "hf", uri: "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf" }],
      });
      expect(row.route).toBe("embedded");
      // Quantization comes from the HEADER (`general.file_type`), not the filename — a renamed file
      // still reports what it really is.
      expect(row.quantization).toBe("Q4_K_M");
      expect(row.weightsMb).toBeGreaterThan(400);
      expect(row.contextLength).toBe(32_768);
      expect(row.openWeights).toBe(true);
      expect(row.downloads).toHaveLength(1);
    },
    120_000,
  );

  withWeights(
    "the zero rates make cost read FREE rather than unmeasured",
    async () => {
      // The distinction this codebase keeps deliberately: `costUsd: 0, costSource: "unknown"` means we
      // could not price the call; `"table"` means we priced it and the answer is nothing.
      const row = await catalogRowForGguf({ model: "priced-local", source: GGUF });
      const table = new ModelInfo([row]);
      expect(table.computeCost("embedded/priced-local", { inputTokens: 1_000_000, outputTokens: 1_000_000 })).toBe(0);
      // ...whereas a model with no row cannot be priced at all.
      expect(table.computeCost("embedded/absent" as never, { inputTokens: 1000 })).toBeNull();
    },
    120_000,
  );

  withWeights(
    "a generated row makes a real call report `table`, not `unknown`",
    async () => {
      const row = await catalogRowForGguf({ model: "costed", source: GGUF });
      ModelInfo.instance.load([row]);
      const router = createModelRouter({ skipDispatcher: true, embedded: { modelPath: GGUF, contextSize: 512 } });
      try {
        const out = await executeLlmCall({ model: "embedded/costed", prompt: "say ok", maxOutputTokens: 4 }, { modelRouter: router });
        expect(errorOf(out)).toBeUndefined();
        expect(out.metrics.costUsd).toBe(0);
        expect(out.metrics.costSource).toBe("table"); // free, and known to be free
      } finally {
        await router.close?.();
        ModelInfo.instance.remove("embedded/costed" as never);
      }
    },
    180_000,
  );
});

describe("residency manager wired to a router", () => {
  withWeights(
    "EVICTS a model through the router, actually freeing its memory",
    async () => {
      // The seam between the two halves. `maxResident` is bookkeeping unless `unload` reaches the store
      // that holds the gigabytes — this is the test that the wiring in between is real.
      const router = createModelRouter({
        skipDispatcher: true,
        embedded: (id) => ({ modelPath: GGUF, contextSize: 512, sequences: 1, ...(id ? {} : {}) }),
      });
      const unloaded: string[] = [];
      const manager = new ResidencyManager({
        maxResident: 1,
        maxConcurrentPerModel: 1,
        unload: async (id) => {
          unloaded.push(id);
          await router.unloadModel?.(id);
        },
      });
      try {
        const call = async (modelId: string) => {
          const lease = await manager.acquire(modelId);
          try {
            return await executeLlmCall({ model: modelId, prompt: "say ok", maxOutputTokens: 4 }, { modelRouter: router });
          } finally {
            lease.release();
          }
        };
        expect(errorOf(await call("embedded/a"))).toBeUndefined();
        expect(manager.residentModels()).toEqual(["embedded/a"]);

        // A second model with room for only one forces the swap.
        expect(errorOf(await call("embedded/b"))).toBeUndefined();
        expect(unloaded).toEqual(["embedded/a"]);
        expect(manager.residentModels()).toEqual(["embedded/b"]);

        // And `a` reloads cleanly afterwards — an unload that half-freed would fail here.
        expect(errorOf(await call("embedded/a"))).toBeUndefined();
        expect(unloaded).toEqual(["embedded/a", "embedded/b"]);
      } finally {
        await manager.close();
        await router.close?.();
      }
    },
    300_000,
  );
});
