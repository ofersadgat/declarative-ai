/**
 * Locally-served models whose WEIGHTS are missing (§5).
 *
 * The check is possible because the loader merges the `environment` chain before validation runs, so a
 * state's operation carries its RESOLVED model — including one inherited from an ancestor's defaults or
 * from a per-mount layer. That is what makes a workflow's whole model working-set knowable before
 * anything runs, instead of discovered one failed call at a time.
 */
import { describe, expect, it } from "vitest";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";

/** Weights present for `qwen-small`, absent for `qwen-large`, and nothing to say about anything else. */
const present = (modelId: string): boolean | undefined => {
  if (!modelId.startsWith("embedded/")) return undefined; // not a model this caller manages
  return modelId === "embedded/qwen-small";
};

const leaf = (extra: Partial<StateDef> = {}): StateDef => ({
  label: "Leaf",
  outputs: { summary: { schema: { type: "string" } } },
  ...extra,
});

/** One state that runs a prompt, with the model declared inline. */
const inline = (model?: string): Record<string, StateDef> => ({
  root: leaf({ operation: { kind: "prompt", prompt: "go", ...(model !== undefined ? { model } : {}) } }),
});

const weightsFindings = (report: { warnings: { message: string; path: string; stateId: string }[] }) =>
  report.warnings.filter((w) => /weights/.test(w.message));

describe("missing local weights", () => {
  it("is not reported at all without the predicate — the document alone cannot know", () => {
    const report = validateBundle(loadBundle(inline("embedded/qwen-large"), "root"));
    expect(weightsFindings(report)).toEqual([]);
  });

  it("WARNS for a local model whose weights are absent, naming it and the path", () => {
    const report = validateBundle(loadBundle(inline("embedded/qwen-large"), "root"), { weightsPresent: present });
    const found = weightsFindings(report);
    expect(found).toHaveLength(1);
    expect(found[0]!.message).toMatch(/model 'embedded\/qwen-large' is served locally but its weights are not present/);
    expect(found[0]!.path).toBe("operation.config.model");
  });

  it("stays a WARNING, so a workflow whose local branch is never entered still runs", () => {
    // Same reasoning that keeps an unregistered `functionRef` a warning: a state the run never enters
    // never needs its weights, and the pre-run gate must not block on it.
    const report = validateBundle(loadBundle(inline("embedded/qwen-large"), "root"), { weightsPresent: present });
    expect(report.errors).toEqual([]);
  });

  it("says nothing when the weights ARE present", () => {
    const report = validateBundle(loadBundle(inline("embedded/qwen-small"), "root"), { weightsPresent: present });
    expect(weightsFindings(report)).toEqual([]);
  });

  it("says nothing about a REMOTE model — `undefined` means not mine", () => {
    // The predicate is how the caller keeps route knowledge out of hw. A remote model has no weights to
    // be missing, and only an explicit `false` is a finding.
    const report = validateBundle(loadBundle(inline("anthropic/claude-haiku-4-5"), "root"), { weightsPresent: present });
    expect(weightsFindings(report)).toEqual([]);
  });

  it("checks a model INHERITED from an ancestor's environment, not just an inline one", () => {
    // The whole point: `environment` is a defaults layer merged down the tree, so the model a leaf runs
    // may be declared nowhere near it.
    const defs: Record<string, StateDef> = {
      root: { label: "Root", environment: { kind: "prompt", model: "embedded/qwen-large" }, children: { child: { state: "root/leaf" } } },
      "root/leaf": leaf({ operation: { prompt: "go" } }),
    };
    const found = weightsFindings(validateBundle(loadBundle(defs, "root"), { weightsPresent: present }));
    expect(found).toHaveLength(1);
    expect(found[0]!.stateId).toBe("root/leaf");
  });

  describe("placement tiers", () => {
    const placement = (id: string): "vram" | "ram" | "swap" | undefined => {
      if (!id.startsWith("embedded/")) return undefined;
      if (id.endsWith("-huge")) return "swap";
      if (id.endsWith("-big")) return "ram";
      return "vram";
    };

    it("says nothing for a model that fits VRAM", () => {
      const report = validateBundle(loadBundle(inline("embedded/qwen-small"), "root"), { placement });
      expect(report.errors).toEqual([]);
      expect(report.warnings.filter((w) => /spill/.test(w.message))).toEqual([]);
    });

    it("WARNS on a spill into system RAM — it works, slower, and the author should know", () => {
      const report = validateBundle(loadBundle(inline("embedded/qwen-big"), "root"), { placement });
      expect(report.errors).toEqual([]);
      expect(report.warnings.map((w) => w.message).join("\n")).toMatch(/spill out of VRAM into system memory/);
    });

    it("ERRORS on a spill into swap — it does not run usefully", () => {
      const report = validateBundle(loadBundle(inline("embedded/qwen-huge"), "root"), { placement });
      expect(report.errors).toHaveLength(1);
      expect(report.errors[0]!.message).toMatch(/spill past system memory into swap/);
      expect(report.errors[0]!.path).toBe("operation.config.model");
    });

    it("the policy is the ACKNOWLEDGMENT surface — a host may downgrade swap to a warning", () => {
      // "yes, run the 70B degraded anyway". It lives in the host's configuration, never in a workflow:
      // a document that could downgrade its own safety check would not be a check.
      const report = validateBundle(loadBundle(inline("embedded/qwen-huge"), "root"), {
        placement,
        placementPolicy: { swap: "warn" },
      });
      expect(report.errors).toEqual([]);
      expect(report.warnings.map((w) => w.message).join("\n")).toMatch(/into swap/);
    });

    it("...and may tighten a RAM spill into an error, or silence it entirely", () => {
      const strict = validateBundle(loadBundle(inline("embedded/qwen-big"), "root"), { placement, placementPolicy: { ram: "error" } });
      expect(strict.errors).toHaveLength(1);
      const quiet = validateBundle(loadBundle(inline("embedded/qwen-big"), "root"), { placement, placementPolicy: { ram: "ignore" } });
      expect(quiet.errors).toEqual([]);
      expect(quiet.warnings.filter((w) => /spill/.test(w.message))).toEqual([]);
    });

    it("is not checked at all without the predicate, and ignores remote models", () => {
      expect(validateBundle(loadBundle(inline("embedded/qwen-huge"), "root")).errors).toEqual([]);
      expect(validateBundle(loadBundle(inline("anthropic/claude-haiku-4-5"), "root"), { placement }).errors).toEqual([]);
    });

    it("reports a model INHERITED from an ancestor's environment", () => {
      // The property that makes this worth doing at validation: the working set is knowable statically,
      // including models declared nowhere near the state that runs them.
      const defs: Record<string, StateDef> = {
        root: { label: "Root", environment: { kind: "prompt", model: "embedded/qwen-huge" }, children: { child: { state: "root/leaf" } } },
        "root/leaf": leaf({ operation: { prompt: "go" } }),
      };
      const report = validateBundle(loadBundle(defs, "root"), { placement });
      expect(report.errors).toHaveLength(1);
      expect(report.errors[0]!.stateId).toBe("root/leaf");
    });
  });

  it("reports the MOUNT that names an unavailable model, not the one that does not", () => {
    // Per-mount `environment` is how one state runs under two models; two mounts load as two variants,
    // so the finding must follow the mount that actually asked for the missing weights.
    const defs: Record<string, StateDef> = {
      root: {
        label: "Root",
        environment: { kind: "prompt" },
        children: {
          small: { state: "root/leaf", environment: { model: "embedded/qwen-small" } },
          large: { state: "root/leaf", environment: { model: "embedded/qwen-large" } },
        },
      },
      "root/leaf": leaf({ operation: { prompt: "go" } }),
    };
    const found = weightsFindings(validateBundle(loadBundle(defs, "root"), { weightsPresent: present }));
    expect(found).toHaveLength(1);
    expect(found[0]!.message).toMatch(/qwen-large/);
  });
});
