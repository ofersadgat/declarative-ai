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

describe("desugaring (API.md, \"Binding desugaring\")", () => {
  const critique = () => loadBundle(specPlanningFiles(), PLAN_ID).states["feature/plan/critique"]!;

  it("lowers every authored sugar to a base Ref case", () => {
    const plan = loadBundle(specPlanningFiles(), PLAN_ID).states[PLAN_ID]!;
    // `{ input: "issue" }` → a `scope.get` producer edge.
    const issueWire = plan.children!["goals"]!.inputs!["issue"]!;
    expect(issueWire).toMatchObject({ op: { kind: "function", functionRef: "scope.get" } });
    // `{ child, output }` → a producer edge on the child + a `select` projection.
    const goalsWire = plan.children!["context"]!.inputs!["goals"]!;
    expect(goalsWire).toMatchObject({ op: { kind: "function", functionRef: "select" } });
    expect((goalsWire as { op: { input: Record<string, { binding?: unknown }> } }).op.input.value!.binding).toEqual({ op: "goals" });
    // A literal stays a literal.
    expect(plan.children!["critique"]!.inputs!["severity_threshold"]).toEqual({ text: "significant" });
    // `{ expr }` → an `expr.eval` producer.
    expect(plan.outputs!["outcome"]!.binding).toMatchObject({ op: { kind: "function", functionRef: "expr.eval" } });
  });

  it("an operation's declared output is what the OPERATION produces — bound outputs excluded", () => {
    // `human_decision` is derived from a child when the state terminates, so requiring it of the
    // operation would be a contract the operation cannot meet.
    const op = critique().operation!;
    const props = op.output.schema!.properties as Record<string, unknown>;
    expect(Object.keys(props).sort()).toEqual(["critique_report", "outcome", "weaknesses"]);
    expect(op.output.schema!.required).not.toContain("human_decision");
  });

  it("a prompt op's render variable is an ordinary bound input slot", () => {
    const files = specPlanningFiles();
    // A render variable (the successor to the removed `operation.params` sugar) is authored as an
    // operation input with a literal binding, and reaches the template under `{{inputs.style}}`.
    files["feature/plan/goals"]!.operation = {
      kind: "prompt",
      prompt: { template: "Extract goals ({{inputs.style}})." },
      config: { model: "planner" },
      input: { style: { kind: "text", binding: { text: "terse" } } },
    };
    const goals = loadBundle(files, PLAN_ID).states["feature/plan/goals"]!;
    expect(goals.operation!.input["style"]).toEqual({ kind: "text", binding: { text: "terse" } });
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
    files[PLAN_ID]!.outputs!["outcome"]!.binding = { expr: "bogusroot.x" };
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
    plan.children!["goals"]!.inputs!["not_an_input"] = { json: 1 };
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

  it("flags an unknown slot kind", () => {
    const files = specPlanningFiles();
    files["feature/plan/goals"]!.outputs!["weird"] = { kind: "wibble" as "json" };
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/unknown slot kind 'wibble'/);
  });

  it("type-checks a producer edge against the consuming slot (§7.3)", () => {
    const files = specPlanningFiles();
    // The `goals` child produces a string ARRAY; wire it into the string-typed `issue` slot instead.
    files[PLAN_ID]!.children!["context"]!.inputs!.issue = { child: "goals", output: "goals" };
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/not type-compatible/);
  });

  it("requires a guard to infer to boolean (§7.2, no truthiness coercion)", () => {
    const files = specPlanningFiles();
    files[PLAN_ID]!.transitions![0]!.when = "run.iteration";
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/guard must infer to boolean/);
  });

  it("rejects a producer edge not proven to have run (§7.2 reachability)", () => {
    const files = specPlanningFiles();
    const critique = files["feature/plan/critique"]!;
    // `human_review` is reachable only through a CONDITIONAL transition, so a REQUIRED slot reading
    // it could observe nothing — the hole the strict rule closes.
    critique.outputs!["weaknesses"]!.binding = { child: "human_review", output: "decision" };
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/not proven to have run/);
  });

  it("an optional/defaulted slot is the explicit opt-out from the reachability rule", () => {
    const files = specPlanningFiles();
    // The fixture's `human_decision` reads that same conditionally-reached child, but declares
    // `optional: true` — the author saying absence is acceptable here.
    expect(files["feature/plan/critique"]!.outputs!["human_decision"]!.optional).toBe(true);
    const report = validateBundle(loadBundle(files, PLAN_ID));
    expect(report.errors.filter((e) => /not proven to have run/.test(e.message))).toEqual([]);
  });
});
