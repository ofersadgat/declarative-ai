/**
 * A child is every pass it took — `.children.<key>` is the sequence, `[-1]` the current one.
 *
 * The bug this exists for: a sequence reset deleted the record before the new inputs resolved, so a
 * re-plan loop could not read what the previous pass said. `prior_findings` wired to
 * `.children.critique.output.findings` was authorable, type-checked, and resolved to nothing on
 * every iteration — no park, no error, no diagnostic, because the input was optional.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/**
 * `draft` records what it was told about the previous pass; `critique` counts its own passes and
 * sends the draft back until the limit. Three passes of the shape `feature/product` runs.
 */
function replanBundle(): Record<string, StateDef> {
  return {
    root: {
      label: "Root",
      limits: { max_iterations: 2 },
      children: {
        draft: {
          state: "root/draft",
          inputs: {
            // THE WIRE THIS IS ABOUT. At the moment `draft` runs in pass n, `critique[-1]` is pass
            // n's critique, which has not run — so the findings that sent it back are one before.
            prior: { expr: ".children.critique[-2].output.note" },
          },
        },
        critique: {
          state: "root/critique",
          transitions: [{ to: "draft", when: ".run.iteration < .limits.max_iterations" }],
        },
      },
    },
    "root/draft": {
      label: "draft",
      inputs: { prior: { schema: { type: "string" }, optional: true } },
      outputs: { saw: { schema: { type: "string" } } },
      operation: { kind: "function", function: "draft" },
    },
    "root/critique": {
      label: "critique",
      outputs: { note: { schema: { type: "string" } } },
      operation: { kind: "function", function: "critique" },
    },
  };
}

async function runReplan(): Promise<{ outcome: string; sawPerPass: Array<string | undefined> }> {
  const sawPerPass: Array<string | undefined> = [];
  let pass = 0;
  const registry = newRegistry();
  registry.functions.set(
    "draft",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const prior = (inputs as { prior?: string }).prior;
      sawPerPass.push(prior);
      return ok({ saw: prior ?? "(nothing)" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  registry.functions.set(
    "critique",
    hostFunction(async () => {
      pass += 1;
      return ok({ note: `finding ${pass}` }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({ bundle: loadBundle(replanBundle(), "root"), registry });
  const result = await engine.run({ inputs: {} });
  return { outcome: result.outcome, sawPerPass };
}

describe("a loop reads the pass before it", () => {
  it("carries the previous pass's output into the re-run child", async () => {
    const { outcome, sawPerPass } = await runReplan();
    expect(outcome).toBe("success");
    // Pass 0 has nothing before it. Passes 1 and 2 each see the critique that sent them back —
    // which is the whole feature: three DIFFERENT drafts instead of the same one three times.
    expect(sawPerPass).toEqual([undefined, "finding 1", "finding 2"]);
  });

  it("keeps every pass addressable, oldest first", async () => {
    const registry = newRegistry();
    const seen: Array<unknown> = [];
    registry.functions.set(
      "note",
      hostFunction(async (inputs: Record<string, unknown>) => {
        seen.push((inputs as { all?: unknown }).all);
        return ok({ n: String(seen.length) }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    registry.functions.set("tick", hostFunction(async () => ok({ n: "x" }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
    const engine = new WorkflowEngine({
      bundle: loadBundle(
        {
          root: {
            label: "Root",
            limits: { max_iterations: 2 },
            children: {
              a: { state: "root/a" },
              b: {
                state: "root/b",
                // `a` re-runs each pass, so its history grows; `length` reads how many there are.
                inputs: { all: { expr: ".children.a.length" } },
                transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }],
              },
            },
          },
          "root/a": { label: "a", outputs: { n: { schema: { type: "string" } } }, operation: { kind: "function", function: "tick" } },
          "root/b": {
            label: "b",
            inputs: { all: { schema: { type: "number" }, optional: true } },
            outputs: { n: { schema: { type: "string" } } },
            operation: { kind: "function", function: "note" },
          },
        },
        "root",
      ),
      registry,
    });
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    // One entry per pass, growing — history accumulates rather than being overwritten.
    expect(seen).toEqual([1, 2, 3]);
  });

  it("a child the reset does not clear reads the same in every pass", async () => {
    // `context` runs once, before the loop's target. Carried forward by reference, so index i means
    // the same pass for every key and a guard can line them up.
    const registry = newRegistry();
    const priors: Array<unknown> = [];
    registry.functions.set("ctx", hostFunction(async () => ok({ v: "fixed" }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
    registry.functions.set(
      "use",
      hostFunction(async (inputs: Record<string, unknown>) => {
        priors.push((inputs as { first?: unknown }).first);
        return ok({ n: "ok" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    const engine = new WorkflowEngine({
      bundle: loadBundle(
        {
          root: {
            label: "Root",
            limits: { max_iterations: 2 },
            children: {
              context: { state: "root/context" },
              work: {
                state: "root/work",
                inputs: { first: { expr: ".children.context[0].output.v" } },
                transitions: [{ to: "work", when: ".run.iteration < .limits.max_iterations" }],
              },
            },
          },
          "root/context": { label: "context", outputs: { v: { schema: { type: "string" } } }, operation: { kind: "function", function: "ctx" } },
          "root/work": {
            label: "work",
            inputs: { first: { schema: { type: "string" }, optional: true } },
            outputs: { n: { schema: { type: "string" } } },
            operation: { kind: "function", function: "use" },
          },
        },
        "root",
      ),
      registry,
    });
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(priors).toEqual(["fixed", "fixed", "fixed"]);
  });

  it("the bare spelling still means the CURRENT pass", async () => {
    // `.children.a.output.n` and `.children.a[-1].output.n` are one value, so nothing that was
    // written before this change had to be rewritten to keep meaning what it meant.
    const registry = newRegistry();
    let bare: unknown;
    let indexed: unknown;
    registry.functions.set("tick", hostFunction(async () => ok({ n: "v1" }) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
    registry.functions.set(
      "read",
      hostFunction(async (inputs: Record<string, unknown>) => {
        bare = (inputs as { bare?: unknown }).bare;
        indexed = (inputs as { indexed?: unknown }).indexed;
        return ok({ n: "ok" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    const engine = new WorkflowEngine({
      bundle: loadBundle(
        {
          root: {
            label: "Root",
            children: {
              a: { state: "root/a" },
              b: {
                state: "root/b",
                inputs: { bare: ".children.a.output.n", indexed: { expr: ".children.a[-1].output.n" } },
              },
            },
          },
          "root/a": { label: "a", outputs: { n: { schema: { type: "string" } } }, operation: { kind: "function", function: "tick" } },
          "root/b": {
            label: "b",
            inputs: { bare: { schema: { type: "string" } }, indexed: { schema: { type: "string" }, optional: true } },
            outputs: { n: { schema: { type: "string" } } },
            operation: { kind: "function", function: "read" },
          },
        },
        "root",
      ),
      registry,
    });
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(bare).toBe("v1");
    expect(indexed).toBe("v1");
  });
});

describe("a running pass parks its readers, whichever spelling asks", () => {
  /**
   * `.children.a.output.x` lowers through the REF desugarer; `.children.a[-1].output.x` contains
   * brackets, fails `isPathSpelling`, and lowers as an expression instead. Two paths to one value,
   * and the dataflow join lives on the first — so a test that only used the dotted form would not
   * notice the bracketed one reading a running child as absent and proceeding without it.
   */
  async function readsWhileRunning(binding: unknown): Promise<{ outcome: string; saw: unknown }> {
    let saw: unknown = "never ran";
    const registry = newRegistry();
    registry.functions.set(
      "slow",
      hostFunction(async () => {
        await new Promise((resolve) => setTimeout(resolve, 20));
        return ok({ n: "done" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    registry.functions.set(
      "read",
      hostFunction(async (inputs: Record<string, unknown>) => {
        saw = (inputs as { seen?: unknown }).seen;
        return ok({ n: "ok" }) as ExecResult<ResolvedValue, WorkflowMetrics>;
      }, HOST),
    );
    const engine = new WorkflowEngine({
      bundle: loadBundle(
        {
          root: {
            label: "Root",
            children: {
              // `async` so the reader is entered while the producer is still in flight — which is
              // the only moment the two lowerings can disagree.
              a: { state: "root/a", async: true },
              b: { state: "root/b", inputs: { seen: binding as never } },
            },
          },
          "root/a": { label: "a", outputs: { n: { schema: { type: "string" } } }, operation: { kind: "function", function: "slow" } },
          "root/b": {
            label: "b",
            inputs: { seen: { schema: { type: "string" }, optional: true } },
            outputs: { n: { schema: { type: "string" } } },
            operation: { kind: "function", function: "read" },
          },
        },
        "root",
      ),
      registry,
    });
    const result = await engine.run({ inputs: {} });
    return { outcome: result.outcome, saw };
  }

  it("parks the DOTTED read until the pass settles", async () => {
    const { outcome, saw } = await readsWhileRunning(".children.a.output.n");
    expect(outcome).toBe("success");
    expect(saw).toBe("done");
  });

  it("parks the BRACKETED read too — the two lowerings agree", async () => {
    // Absent this, the bracketed form read a running child as `undefined`, the optional input was
    // silently skipped, and the state ran with nothing in it. That is the failure this whole
    // document exists about, reintroduced by a spelling.
    const { outcome, saw } = await readsWhileRunning({ expr: ".children.a[-1].output.n" });
    expect(outcome).toBe("success");
    expect(saw).toBe("done");
  });
});
