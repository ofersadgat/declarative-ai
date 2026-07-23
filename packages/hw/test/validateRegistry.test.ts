import type { WorkflowMetrics } from "../src/ports";
/**
 * Validation as a function of *(document, registry)* (DESIGN §7).
 *
 * The redesign made registry entries a discriminated union with REQUIRED, total capabilities precisely
 * so the checker could read them before anything runs — "an interactive function in a search-only
 * workflow" is not decidable from the document alone. This is that check.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, pureFunction, runtimeFunction, HOST_CAPABILITIES, PURE_CAPABILITIES, RUNTIME_CAPABILITIES, type ExecServices, type FunctionResult,
  type FunctionRegistry,
  type ResolvedValue,
} from "@declarative-ai/exec";
import { loadBundle } from "../src/loader";
import { validateBundle } from "../src/validate";
import { PLAN_ID, specPlanningFiles } from "./fixtures";

/** The fixture's one function state — `choose_option`, a human-approval gate. */
const HR = "feature/plan/critique/human_review";
const noop = async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({ value: {} });

function registryWith(capabilities: { interactive: boolean; readOnly: boolean; memoizable: boolean }) {
  const r = new Map();
  r.set("choose_option", hostFunction(noop, capabilities));
  return r;
}

const bundle = () => loadBundle(specPlanningFiles(), PLAN_ID);

describe("an unregistered functionRef", () => {
  it("is not reported at all when no registry is supplied — the document alone cannot know", () => {
    const report = validateBundle(bundle());
    expect(report.errors).toEqual([]);
    expect(report.warnings).toEqual([]);
  });

  it("WARNS when a registry is supplied, naming the missing function", () => {
    const report = validateBundle(bundle(), { functions: new Map() });
    expect(report.errors).toEqual([]);
    expect(report.warnings.map((w) => w.message).join("\n")).toMatch(/no function 'choose_option' is registered/);
    expect(report.warnings[0]!.stateId).toBe(HR);
    expect(report.warnings[0]!.path).toBe("operation.function");
  });

  // Leaving a function unregistered is how a search context REFUSES a human gate, and a state the run
  // never enters never needs its function. So the pre-run gate must not block on it.
  it("does NOT become an error by default — that is what keeps an unreached gate runnable", () => {
    const report = validateBundle(bundle(), { functions: new Map() });
    expect(report.errors).toEqual([]);
  });

  it("becomes an error under `strict`, for a lint surface that wants every reference to resolve", () => {
    const report = validateBundle(bundle(), { functions: new Map(), strict: true });
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/no function 'choose_option' is registered/);
  });

  it("says nothing once the function IS registered", () => {
    const report = validateBundle(bundle(), { functions: registryWith(HOST_CAPABILITIES), strict: true });
    expect(report.errors).toEqual([]);
    expect(report.warnings).toEqual([]);
  });

  it("never flags the loader's own synthesized resolvers, which are engine built-ins", () => {
    // The fixture's `{ expr }` / `{ child, output }` sugar lowers onto RESOLVER_REFS function ops.
    // An empty registry must not report those as missing.
    const report = validateBundle(bundle(), { functions: new Map(), strict: true });
    expect(report.errors.map((e) => e.message).join("\n")).not.toMatch(/resolver|expr|select/i);
  });
});

describe("interactive functions in a non-interactive context", () => {
  const interactive = { interactive: true, readOnly: true, memoizable: false };
  const batch = { interactive: false, readOnly: true, memoizable: false };

  it("is an ERROR when the caller asserts a non-interactive context", () => {
    const report = validateBundle(bundle(), { functions: registryWith(interactive), interactive: false });
    const messages = report.errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/function 'choose_option' is interactive/);
    expect(report.errors[0]!.stateId).toBe(HR);
  });

  it("is fine when the same workflow is validated for an interactive context", () => {
    expect(validateBundle(bundle(), { functions: registryWith(interactive), interactive: true }).errors).toEqual([]);
  });

  it("is not checked at all when the caller says nothing about interactivity", () => {
    expect(validateBundle(bundle(), { functions: registryWith(interactive) }).errors).toEqual([]);
  });

  it("passes a NON-interactive entry through the same assertion", () => {
    expect(validateBundle(bundle(), { functions: registryWith(batch), interactive: false }).errors).toEqual([]);
  });

  it("reads capabilities off a `runtime` entry too, not just `host`", () => {
    const r = new Map();
    r.set("choose_option", runtimeFunction(noop, { ...RUNTIME_CAPABILITIES, interactive: true }));
    const report = validateBundle(bundle(), { functions: r, interactive: false });
    expect(report.errors.map((e) => e.message).join("\n")).toMatch(/is interactive/);
  });

  it("never flags a `pure` entry, which has no interactivity axis to read", () => {
    const r = new Map();
    r.set("choose_option", pureFunction(() => ({ value: {} }), PURE_CAPABILITIES));
    expect(validateBundle(bundle(), { functions: r, interactive: false }).errors).toEqual([]);
  });
});
