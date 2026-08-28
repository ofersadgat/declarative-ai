/**
 * "This state will terminate immediately" — when that is a complaint, and when it is the design.
 *
 * The check used to be a flat `operation || children`, which made it fire on the one shape it should
 * be quietest about: a state whose every output is a BINDING. Those are resolved when the state
 * terminates (§3.7), so such a state has no operation and no children *because* it needs neither —
 * the arithmetic IS the state. Warning on it put an unactionable line under every scoring and verdict
 * state in a workflow, which is how a warning stops being read at all.
 *
 * What must still warn is the state that genuinely does nothing: nothing to run, and nothing to
 * compute from what is already there.
 */
import { describe, expect, it } from "vitest";
import type { StateDef } from "../src/format.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";

const warnings = (files: Record<string, StateDef>, rootId: string) =>
  validateBundle(loadBundle(files, rootId))
    .warnings.map((w) => `${w.stateId} ${w.path}: ${w.message}`)
    .join("\n");

const inert = (outputs: StateDef["outputs"]): Record<string, StateDef> => ({
  root: {
    label: "Root",
    inputs: { rank: { schema: { type: "integer" } } },
    ...(outputs !== undefined ? { outputs } : {}),
  } as StateDef,
});

describe("a state with no operation and no children", () => {
  it("is not warned about when its outputs are computed from what it was handed", () => {
    const files = inert({
      score: { schema: { type: "number" }, binding: { expr: "max(0, 1 - .inputs.rank / 3)" } },
    });
    expect(warnings(files, "root")).toBe("");
  });

  it("is not warned about when the binding is a plain read either", () => {
    const files = inert({ rank: { schema: { type: "integer" }, binding: ".inputs.rank" } });
    expect(warnings(files, "root")).toBe("");
  });

  it("IS warned about when it declares no outputs at all", () => {
    expect(warnings(inert(undefined), "root")).toContain("having done nothing");
  });

  it("IS warned about when every output is produced — nothing exists to fill them", () => {
    // No operation, so `verdict` can never arrive; the state fails at run time with "required output
    // was not produced". The unbound-output check reports the slot; this one reports the state.
    const files = inert({ verdict: { schema: { type: "string" } } });
    expect(warnings(files, "root")).toContain("having done nothing");
  });

  it("says nothing once the state has an operation, bound outputs or not", () => {
    const files: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { verdict: { schema: { type: "string" }, binding: ".operation.output.verdict" } },
        operation: { kind: "prompt", model: "m", prompt: "go", output: { verdict: { schema: { type: "string" } } } },
      } as StateDef,
    };
    expect(warnings(files, "root")).toBe("");
  });
});
