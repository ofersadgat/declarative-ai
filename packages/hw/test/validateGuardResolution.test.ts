/**
 * A guard whose CALL resolves to nothing is an error, not a rule that quietly stops existing.
 *
 * The loader carries a guard's lowering failure as data (`LoadedTransition.whenError`) rather than
 * throwing, so one bad guard cannot hide every other mistake in a workflow — and the engine skips a
 * transition carrying one, because reading an unparseable guard as unconditional would be the worst
 * available interpretation of a typo. Both halves are right. What was missing is the third: somebody
 * has to SAY SO.
 *
 * Until it did, a mistyped callee produced a workflow with one fewer rule. Lint clean, run silent,
 * and — where the rule was the one offering a person a decision — an offer that was simply never
 * made, with nothing anywhere to explain why.
 */
import { describe, expect, it } from "vitest";
import type { EntrySignature } from "@declarative-ai/exec";
import type { StateDef } from "../src/format.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";

/** A host callee, contributed the way a registry contributes one (SPEC §7.1). */
const REGISTRY: ReadonlyMap<string, EntrySignature> = new Map<string, EntrySignature>([
  [
    "did_it_happen",
    {
      signature: {
        input: { event: { kind: "text", index: 0, schema: { type: "string" } } },
        output: { name: "happened", kind: "json", schema: { type: "boolean" } },
      },
    },
  ],
]);

const files = (guard: string): Record<string, StateDef> =>
  ({
    root: {
      label: "Root",
      inputs: { n: { schema: { type: "number" } } },
      children: { next: { state: "next" } },
      transitions: [{ to: "next", when: guard }, { to: "terminate.success" }],
    },
    next: { label: "Next", operation: { kind: "prompt", model: "m", prompt: "go" } },
  }) as unknown as Record<string, StateDef>;

const errorsFor = (guard: string, functions?: ReadonlyMap<string, EntrySignature>) =>
  validateBundle(loadBundle(files(guard), "root", functions ? { functions } : {}))
    .errors.map((e) => `${e.path}: ${e.message}`)
    .join("\n");

describe("a guard that calls something", () => {
  it("is fine when the callee is on the path", () => {
    expect(errorsFor("did_it_happen('drag')", REGISTRY)).toBe("");
  });

  it("is an ERROR when the callee resolves nowhere", () => {
    const errors = errorsFor("did_it_happen_TYPO('drag')", REGISTRY);
    expect(errors).toContain("transitions[0].when");
    expect(errors).toContain("guard could not be resolved");
  });

  it("names the callee that could not be found, so the typo is visible", () => {
    expect(errorsFor("did_it_happen_TYPO('drag')", REGISTRY)).toContain("did_it_happen_TYPO");
  });

  it("reports a callee the registry would have supplied but was not passed", () => {
    // The registry is a CONTRIBUTOR on the search path, so leaving it off is leaving a path entry
    // off. The honest answer is that the name resolves nowhere — which is exactly what a run against
    // that same partial registry would find.
    expect(errorsFor("did_it_happen('drag')")).toContain("guard could not be resolved");
  });

  it("still reports the ordinary guard errors when the guard DOES resolve", () => {
    // The new check must not swallow the old ones: a non-boolean guard is still the error it was.
    expect(errorsFor(".inputs.n", REGISTRY)).toContain("guard must infer to boolean");
  });

  it("says nothing about a guard with no call in it at all", () => {
    expect(errorsFor(".inputs.n > 3", REGISTRY)).toBe("");
  });

  it("reports a guard on a CHILD MOUNT the same way", () => {
    // A child's transitions are the state's, narrowed to the round that child finishes in, so a
    // mistyped callee there is the same mistake and reads the same.
    const mounted = {
      root: {
        label: "Root",
        children: { next: { state: "next", transitions: [{ to: "terminate.success", when: "nope('x')" }] } },
      },
      next: { label: "Next", operation: { kind: "prompt", model: "m", prompt: "go" } },
    } as unknown as Record<string, StateDef>;
    const errors = validateBundle(loadBundle(mounted, "root", { functions: REGISTRY }))
      .errors.map((e) => `${e.path}: ${e.message}`)
      .join("\n");
    expect(errors).toContain("children.next.transitions[0].when");
    expect(errors).toContain("guard could not be resolved");
  });
});
