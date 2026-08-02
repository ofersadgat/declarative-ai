/** State path references (§2.1) — the grammar an `id` or a `children[].state` is written in. */
import { describe, expect, it } from "vitest";
import { isBareStateId, resolveStateRef, stateFilePath, StateRefError } from "../src/ref.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";

const roots = { JAIRA: "/proj/.jaira", PROJECT: "/proj" };
const opts = { defaultRoot: "/proj/.jaira/workflows", roots };

describe("resolveStateRef", () => {
  it("leaves a bare reference bare — it is already the canonical id", () => {
    expect(resolveStateRef("feature/plan", opts)).toBe("feature/plan");
    expect(resolveStateRef("feature//plan/", opts)).toBe("feature/plan");
  });

  it("folds a $VAR reference back to its bare spelling when it lands under the default root", () => {
    // The point of folding: the id keys the snapshot hash and the event log, so two spellings of
    // the same state must not produce two identities.
    expect(resolveStateRef("$JAIRA/workflows/feature/plan", opts)).toBe("feature/plan");
    expect(resolveStateRef("$PROJECT/.jaira/workflows/feature/plan", opts)).toBe("feature/plan");
  });

  it("keeps an out-of-tree reference absolute", () => {
    expect(resolveStateRef("$PROJECT/shared/review", opts)).toBe("/proj/shared/review");
    expect(resolveStateRef("/opt/workflows/review", opts)).toBe("/opt/workflows/review");
    expect(resolveStateRef("file:/opt/workflows/review", opts)).toBe("/opt/workflows/review");
  });

  it("resolves ./ and ../ against the referring state's own id, not its file's directory", () => {
    // `feature/plan.json` sits in `feature/`, but its children live in `feature/plan/` — resolving
    // against the file's directory would make relative references useless exactly where they help.
    expect(resolveStateRef("./goals", { ...opts, from: "feature/plan" })).toBe("feature/plan/goals");
    expect(resolveStateRef("../shared/lint", { ...opts, from: "feature/plan" })).toBe("feature/shared/lint");
  });

  it("normalizes a Windows path and a drive letter to one POSIX spelling", () => {
    expect(resolveStateRef("C:\\wf\\review", opts)).toBe("C:/wf/review");
    expect(isBareStateId("C:/wf/review")).toBe(false);
  });

  it("refuses what it cannot resolve rather than guessing", () => {
    expect(() => resolveStateRef("", opts)).toThrow(StateRefError);
    // `feature/plan` → `..` is `feature`, `../..` is the root itself, `../../..` is off the end.
    expect(resolveStateRef("../../shared", { ...opts, from: "feature/plan" })).toBe("shared");
    expect(() => resolveStateRef("../../../escape", { ...opts, from: "feature/plan" })).toThrow(
      /climbs above the workflow root/,
    );
    expect(() => resolveStateRef("./x", opts)).toThrow(/no referring state/);
    expect(() => resolveStateRef("$NOPE/x", opts)).toThrow(/unknown root '\$NOPE'/);
    expect(() => resolveStateRef("https://example.com/x", opts)).toThrow(/unknown scheme 'https:'/);
  });

  it("maps a canonical id back to the file it reads from", () => {
    expect(stateFilePath("feature/plan", "/proj/.jaira/workflows")).toBe("/proj/.jaira/workflows/feature/plan");
    expect(stateFilePath("/opt/wf/review", "/proj/.jaira/workflows")).toBe("/opt/wf/review");
  });
});

describe("references inside a bundle", () => {
  const files = (childRef: string): Record<string, StateDef> => ({
    "feature/plan": {
      label: "Plan",
      children: { goals: { state: childRef } },
    },
    "feature/plan/goals": {
      label: "Goals",
      outputs: { goals: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: "go" },
    },
  });

  it("resolves a relative child reference to the same state a bare one names", () => {
    for (const ref of ["feature/plan/goals", "./goals", "$JAIRA/workflows/feature/plan/goals"]) {
      const bundle = loadBundle(files(ref), "feature/plan", opts);
      expect(bundle.states["feature/plan"]!.children!.goals!.state).toBe("feature/plan/goals");
      expect(Object.keys(bundle.states).sort()).toEqual(["feature/plan", "feature/plan/goals"]);
    }
  });

  it("reports a bad reference against the field that wrote it", () => {
    expect(() => loadBundle(files("$NOPE/goals"), "feature/plan", opts)).toThrow(
      /feature\/plan: children\.goals\.state: unknown root/,
    );
  });

  it("fetches an out-of-tree state through loadState", () => {
    const external: StateDef = {
      label: "Shared",
      outputs: { goals: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: "shared" },
    };
    const bundle = loadBundle(files("/opt/lib/goals"), "feature/plan", {
      ...opts,
      loadState: (id) => (id === "/opt/lib/goals" ? external : undefined),
    });
    expect(Object.keys(bundle.states).sort()).toEqual(["/opt/lib/goals", "feature/plan"]);
    expect(bundle.states["feature/plan"]!.children!.goals!.state).toBe("/opt/lib/goals");
  });

  it("accepts a declared id that spells the same path differently", () => {
    const defs = files("./goals");
    defs["feature/plan"]!.id = "$JAIRA/workflows/feature/plan";
    expect(() => loadBundle(defs, "feature/plan", opts)).not.toThrow();
  });

  it("still rejects a declared id naming a different state", () => {
    const defs = files("./goals");
    defs["feature/plan"]!.id = "feature/other";
    expect(() => loadBundle(defs, "feature/plan", opts)).toThrow(/does not match path-derived id/);
  });
});
