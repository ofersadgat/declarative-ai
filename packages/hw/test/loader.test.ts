import { describe, expect, it } from "vitest";
import { loadBundle, snapshotHash, stateIdFromPath, WorkflowLoadError } from "../src/loader";
import { validateBundle } from "../src/validate";
import { FANOUT_ID, PLAN_ID, specFanoutFiles, specPlanningFiles } from "./fixtures";

describe("stateIdFromPath", () => {
  it("strips extensions and normalizes separators", () => {
    expect(stateIdFromPath("feature/plan.json")).toBe("feature/plan");
    expect(stateIdFromPath("feature\\plan\\critique.state.json")).toBe("feature/plan/critique");
    expect(stateIdFromPath("feature/plan")).toBe("feature/plan");
  });
});

describe("loadBundle", () => {
  it("loads the spec planning workflow and derives ids", () => {
    const bundle = loadBundle(specPlanningFiles(), PLAN_ID);
    expect(Object.keys(bundle.states).sort()).toEqual([
      "feature/plan",
      "feature/plan/context",
      "feature/plan/critique",
      "feature/plan/critique/address_weaknesses",
      "feature/plan/critique/human_review",
      "feature/plan/goals",
    ]);
    expect(bundle.states[PLAN_ID]!.id).toBe(PLAN_ID);
  });

  it("rejects a mismatched declared id", () => {
    const files = specPlanningFiles();
    (files[PLAN_ID] as { id?: string }).id = "some/other/id";
    expect(() => loadBundle(files, PLAN_ID)).toThrow(WorkflowLoadError);
  });

  it("rejects a missing root and non-object files", () => {
    expect(() => loadBundle(specPlanningFiles(), "nope")).toThrow(/root state/);
    expect(() => loadBundle({ x: 42 }, "x")).toThrow(/not a JSON object/);
  });

  it("restricts the bundle to the closure reachable from the root", () => {
    const files = { ...specPlanningFiles(), unrelated: { label: "Unrelated" } };
    const bundle = loadBundle(files, PLAN_ID);
    expect(bundle.states["unrelated"]).toBeUndefined();
  });
});

describe("snapshotHash (SPEC §12)", () => {
  it("is stable across load order and unreachable files, and changes with content", () => {
    const a = snapshotHash(loadBundle(specPlanningFiles(), PLAN_ID));
    const reordered = Object.fromEntries(Object.entries(specPlanningFiles()).reverse());
    const b = snapshotHash(loadBundle(reordered, PLAN_ID));
    expect(b).toBe(a);
    const withNoise = snapshotHash(loadBundle({ ...specPlanningFiles(), extra: { label: "x" } }, PLAN_ID));
    expect(withNoise).toBe(a);
    const files = specPlanningFiles();
    files["feature/plan/goals"]!.label = "Changed";
    expect(snapshotHash(loadBundle(files, PLAN_ID))).not.toBe(a);
  });

  it("a derived id hashes the same as an explicitly declared matching id", () => {
    const files = specPlanningFiles();
    const a = snapshotHash(loadBundle(files, PLAN_ID));
    (files["feature/plan/goals"] as { id?: string }).id = "feature/plan/goals";
    expect(snapshotHash(loadBundle(files, PLAN_ID))).toBe(a);
  });
});

describe("validateBundle on the spec examples", () => {
  it("planning workflow validates clean (no errors)", () => {
    const report = validateBundle(loadBundle(specPlanningFiles(), PLAN_ID));
    expect(report.errors).toEqual([]);
    expect(report.warnings).toEqual([]);
  });

  it("fan-out workflow validates clean", () => {
    const report = validateBundle(loadBundle(specFanoutFiles(), FANOUT_ID));
    expect(report.errors).toEqual([]);
  });
});

describe("validateBundle failure modes", () => {
  it("flags unknown child states, bad sequence entries, and bad transition targets", () => {
    const files = specPlanningFiles();
    const plan = files[PLAN_ID]!;
    plan.children!["ghost"] = { state: "feature/plan/ghost" };
    plan.sequence = [...plan.sequence!, "ghost", "goals"];
    plan.transitions = [...plan.transitions!, { to: "nowhere" }, { to: "terminate.sideways" }];
    const report = validateBundle(loadBundle(files, PLAN_ID));
    const messages = report.errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/unknown state 'feature\/plan\/ghost'/);
    expect(messages).toMatch(/duplicate sequence entry 'goals'/);
    expect(messages).toMatch(/'nowhere' is neither a declared child/);
    expect(messages).toMatch(/'terminate.sideways' is neither/);
  });

  it("flags unparseable and undeclared-reference expressions", () => {
    const files = specPlanningFiles();
    files[PLAN_ID]!.transitions![0]!.when = "children.critique.outputs.outcome ===";
    files[PLAN_ID]!.transitions![2]!.when = "children.nonchild.outcome === 'success'";
    files[PLAN_ID]!.outputs!["outcome"]!.from = "bogusroot.x";
    const report = validateBundle(loadBundle(files, PLAN_ID));
    const messages = report.errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/does not parse/);
    expect(messages).toMatch(/undeclared child 'nonchild'/);
    expect(messages).toMatch(/unknown reference root 'bogusroot'/);
  });

  it("flags unwired required child inputs and unknown wired names", () => {
    const files = specPlanningFiles();
    const plan = files[PLAN_ID]!;
    delete plan.children!["critique"]!.inputs!["plan_doc"];
    plan.children!["goals"]!.inputs!["not_an_input"] = { value: 1 };
    const report = validateBundle(loadBundle(files, PLAN_ID));
    const messages = report.errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/required child input 'plan_doc' is not wired/);
    expect(messages).toMatch(/declares no input 'not_an_input'/);
  });

  it("warns on an unguarded cycle back into the sequence", () => {
    const files = specPlanningFiles();
    const plan = files[PLAN_ID]!;
    delete plan.limits;
    plan.transitions = [{ to: "goals", when: "children.critique.outputs.outcome === 'needs_changes'" }];
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.warnings.map((w) => w.message).join("\n")).toMatch(/can cycle/);
  });

  it("does not warn when the cycle is guarded by run.iteration", () => {
    const files = specPlanningFiles();
    const plan = files[PLAN_ID]!;
    delete plan.limits;
    plan.transitions = [{ to: "goals", when: "children.critique.outputs.outcome === 'needs_changes' && run.iteration < 3" }];
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.warnings.filter((w) => /can cycle/.test(w.message))).toEqual([]);
  });

  it("flags passthrough misuse and field-type errors", () => {
    const files = specPlanningFiles();
    files[PLAN_ID]!.inputs!["bad"] = { type: "passthrough" };
    files[PLAN_ID]!.outputs!["orphan"] = { type: "passthrough" }; // no from
    files["feature/plan/goals"]!.outputs!["weird"] = { type: "wibble" };
    const report = validateBundle(loadBundle(files, PLAN_ID));
    const messages = report.errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/'passthrough' is only valid for outputs/);
    expect(messages).toMatch(/passthrough outputs require a 'from' expression/);
    expect(messages).toMatch(/unknown field type 'wibble'/);
  });
});
