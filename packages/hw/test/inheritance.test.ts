/**
 * Operation inheritance (§5) and the derived sequence (§6) — the two places a state's behaviour is
 * now decided by something other than its own file.
 */
import { describe, expect, it } from "vitest";
import type { FunctionOp, PromptOp } from "@declarative-ai/exec";
import { loadBundle, sourceStateId } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { mergeOperationChain } from "../src/merge.js";
import type { StateDef } from "../src/format.js";
import type { Vfs } from "../src/reference.js";

const leaf = (extra: Partial<StateDef> = {}): StateDef => ({
  label: "Leaf",
  outputs: { report: { schema: { type: "string" } } },
  ...extra,
});

describe("environment inheritance (§5)", () => {
  const files = (): Record<string, StateDef> => ({
    root: {
      label: "Root",
      environment: {
        kind: "prompt",
        session: "review",
        tools: ["bash"],
        model: "anthropic/claude-sonnet-5",
        temperature: 0.2,
      },
      children: { child: { state: "root/mid" } },
    },
    "root/mid": {
      label: "Mid",
      environment: { temperature: 0.9 },
      children: { leaf: { state: "root/mid/leaf" } },
    },
    "root/mid/leaf": leaf({ operation: { prompt: "go" } }),
  });

  const opOf = (defs: Record<string, StateDef>, id = "root/mid/leaf"): PromptOp<never> =>
    loadBundle(defs, "root").states[id]!.operation as PromptOp<never>;

  it("supplies kind and call configuration from two levels up", () => {
    const op = opOf(files());
    expect(op.kind).toBe("prompt");
    expect(op.user).toBe("go");
    // The root's model survives the mid state's override of a DIFFERENT call setting.
    expect(op.config).toEqual({ model: "anthropic/claude-sonnet-5", temperature: 0.9 });
  });

  it("puts the merged execution environment on the loaded state, not the op", () => {
    const state = loadBundle(files(), "root").states["root/mid/leaf"]!;
    expect(state.environment).toEqual({ session: "review", tools: ["bash"] });
  });

  it("lets the nearest layer win, own operation over own environment over ancestors", () => {
    const defs = files();
    defs["root/mid/leaf"] = leaf({
      environment: { temperature: 0.5, session: "leaf-session" },
      operation: { prompt: "go", temperature: 0.1 },
    });
    const op = opOf(defs);
    expect(op.config).toEqual({ model: "anthropic/claude-sonnet-5", temperature: 0.1 });
    expect(loadBundle(defs, "root").states["root/mid/leaf"]!.environment!.session).toBe("leaf-session");
  });

  it("does NOT give an operation to a state that declares none", () => {
    // Otherwise every pure composite under an environment-declaring root would inherit its
    // ancestor's operation and start running it.
    expect(loadBundle(files(), "root").states["root/mid"]!.operation).toBeUndefined();
  });

  it("treats `operation: {}` as the opt-in to a fully inherited operation", () => {
    const defs = files();
    defs.root!.environment = { kind: "function", function: "claude-code", args: { mode: "plan" } };
    defs["root/mid/leaf"] = leaf({ operation: {} });
    const op = opOf(defs) as unknown as FunctionOp<never>;
    expect(op.kind).toBe("function");
    expect(op.functionRef).toBe("claude-code");
  });

  it("reports an operation the chain never completes as a validation error, not a load failure", () => {
    // Carried as data so ONE broken state does not abort the load and hide every other authoring
    // error in the workflow.
    const defs = files();
    delete defs.root!.environment;
    const bundle = loadBundle(defs, "root");
    const leaf = bundle.states["root/mid/leaf"]!;
    expect(leaf.operation).toBeUndefined();
    expect(leaf.operationError).toMatch(/declares no 'kind'/);
    const report = validateBundle(bundle);
    expect(report.errors).toContainEqual(
      expect.objectContaining({ stateId: "root/mid/leaf", path: "operation" }),
    );
  });

  it("gives a state mounted twice with DIFFERENT environments one entry per mount", () => {
    const defs: Record<string, StateDef> = {
      root: {
        label: "Root",
        children: { a: { state: "root/a" }, b: { state: "root/b" } },
      },
      "root/a": {
        environment: { kind: "prompt", model: "m1" },
        children: { shared: { state: "lib/shared" } },
      },
      "root/b": {
        environment: { kind: "prompt", model: "m2" },
        children: { shared: { state: "lib/shared" } },
      },
      "lib/shared": leaf({ operation: { prompt: "go" } }),
    };
    const bundle = loadBundle(defs, "root");
    const ids = Object.keys(bundle.states).filter((id) => id.startsWith("lib/shared"));
    expect(ids).toHaveLength(2);

    // Each parent points at the variant it actually runs, and each variant has its own model.
    const viaA = bundle.states["root/a"]!.children!.shared!.state;
    const viaB = bundle.states["root/b"]!.children!.shared!.state;
    expect(viaA).not.toBe(viaB);
    expect((bundle.states[viaA]!.operation as PromptOp<never>).config).toEqual({ model: "m1" });
    expect((bundle.states[viaB]!.operation as PromptOp<never>).config).toEqual({ model: "m2" });

    // The first mount keeps the plain id; only the divergent one is suffixed.
    expect(ids).toContain("lib/shared");
    expect(sourceStateId(viaB)).toBe("lib/shared");
    // Both carry the same authored source, so a snapshot round-trips either one.
    expect(bundle.source![viaB]).toEqual(defs["lib/shared"]);
  });

  /**
   * The two-reviewer case: ONE review state, mounted twice under two different agents.
   *
   * Before `children[].environment` the chain was per-STATE, so both children of a parent inherited
   * the same layer and the only way to differ was two near-duplicate state files.
   */
  it("lets two CHILDREN of one parent mount one state under different environments", () => {
    const defs: Record<string, StateDef> = {
      root: {
        label: "Review",
        children: {
          claude_review: { state: "review/agent_review", async: true, environment: { kind: "function", function: "claude-code" } },
          codex_review: { state: "review/agent_review", async: true, environment: { kind: "function", function: "codex-cli" } },
        },
      },
      "review/agent_review": leaf({ operation: {} }),
    };
    const bundle = loadBundle(defs, "root");
    const viaClaude = bundle.states["root"]!.children!.claude_review!.state;
    const viaCodex = bundle.states["root"]!.children!.codex_review!.state;
    expect(viaClaude).not.toBe(viaCodex);
    expect((bundle.states[viaClaude]!.operation as FunctionOp<never>).functionRef).toBe("claude-code");
    expect((bundle.states[viaCodex]!.operation as FunctionOp<never>).functionRef).toBe("codex-cli");
    // Both mounts still map back to the one file the author wrote.
    expect(sourceStateId(viaCodex)).toBe("review/agent_review");
    expect(validateBundle(bundle).errors).toEqual([]);
  });

  it("sits UNDER the child's own environment — a state that says what it runs still wins", () => {
    const defs: Record<string, StateDef> = {
      root: { children: { pinned: { state: "lib/pinned", environment: { kind: "function", function: "codex-cli" } } } },
      "lib/pinned": leaf({ environment: { kind: "function", function: "claude-code" }, operation: {} }),
    };
    const bundle = loadBundle(defs, "root");
    const mounted = bundle.states["root"]!.children!.pinned!.state;
    // Nearest layer wins, and the state's own is nearer than the mount's. A state meant to be varied
    // leaves `function` to the chain instead.
    expect((bundle.states[mounted]!.operation as FunctionOp<never>).functionRef).toBe("claude-code");
  });

  it("layers OVER the parent's environment, so a mount refines rather than replaces", () => {
    const defs: Record<string, StateDef> = {
      root: {
        environment: { kind: "prompt", model: "m1", session: "review" },
        children: { fast: { state: "lib/leaf", environment: { model: "m2" } } },
      },
      "lib/leaf": leaf({ operation: { prompt: "go" } }),
    };
    const bundle = loadBundle(defs, "root");
    const mounted = bundle.states["root"]!.children!.fast!.state;
    // The mount changed the model; everything else the parent supplied came through untouched.
    expect((bundle.states[mounted]!.operation as PromptOp<never>).config).toEqual({ model: "m2" });
    expect(bundle.states[mounted]!.environment?.session).toBe("review");
  });

  it("collapses back to ONE entry when two mounts declare the same thing", () => {
    // The identity hashes what a state RUNS AS, not where it was reached from — so the ordinary
    // shared-library case does not sprout duplicates just because someone wrote the layer twice.
    const defs: Record<string, StateDef> = {
      root: {
        children: {
          a: { state: "lib/leaf", environment: { kind: "function", function: "claude-code" } },
          b: { state: "lib/leaf", environment: { kind: "function", function: "claude-code" } },
        },
      },
      "lib/leaf": leaf({ operation: {} }),
    };
    const bundle = loadBundle(defs, "root");
    expect(Object.keys(bundle.states).filter((id) => id.startsWith("lib/leaf"))).toHaveLength(1);
  });

  it("reports a bad session on a mount against the line that declared it", () => {
    const defs: Record<string, StateDef> = {
      root: { children: { a: { state: "lib/leaf", environment: { session: "" } } } },
      "lib/leaf": leaf({ operation: { prompt: "go", model: "m" } }),
    };
    const report = validateBundle(loadBundle(defs, "root"));
    expect(report.errors).toContainEqual(
      expect.objectContaining({ stateId: "root", path: "children.a.environment.session" }),
    );
  });

  it("allows the same state under two parents when the environments agree", () => {
    const defs: Record<string, StateDef> = {
      root: {
        environment: { kind: "prompt", model: "m" },
        children: { a: { state: "root/a" }, b: { state: "root/b" } },
      },
      "root/a": { children: { shared: { state: "lib/shared" } } },
      "root/b": { children: { shared: { state: "lib/shared" } } },
      "lib/shared": leaf({ operation: { prompt: "go" } }),
    };
    const op = loadBundle(defs, "root").states["lib/shared"]!.operation as PromptOp<never>;
    expect(op.config).toEqual({ model: "m" });
  });

  /**
   * Expansion runs before inheritance so that no later pass has to know references exist. The
   * child path read the RAW document instead, so a transcluded `environment` reached the state
   * that declared it and never reached its children — and because the raw value is the reference
   * STRING, what the children inherited was that string spread character by character
   * (`{"0": "$", "1": "/", …}`).
   */
  it("passes a TRANSCLUDED environment down to children, not just to the state that declares it", () => {
    const ROOT = "/p/.jaira";
    const files: Record<string, string> = { [`${ROOT}/lib/env.json`]: JSON.stringify({ model: "from-fragment" }) };
    const vfs: Vfs = {
      list: (dir) => {
        const prefix = `${dir}/`;
        const names = new Set<string>();
        for (const path of Object.keys(files)) {
          if (!path.startsWith(prefix)) continue;
          const rest = path.slice(prefix.length);
          if (!rest.includes("/")) names.add(rest);
        }
        return [...names];
      },
      read: (path) => files[path],
    };

    const bundle = loadBundle(
      {
        "plan.json": { environment: "$/lib/env", operation: { kind: "prompt", prompt: "parent" } },
        "plan/goals.json": { operation: { kind: "prompt", prompt: "child" } },
      },
      "plan",
      { defaultRoot: `${ROOT}/workflows`, roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs },
    );

    const parent = bundle.states["plan"]!.operation as PromptOp<never>;
    const child = bundle.states["plan/goals"]!.operation as PromptOp<never>;
    expect(parent.config).toEqual({ model: "from-fragment" });
    expect(child.config).toEqual({ model: "from-fragment" });
  });
});

describe("the merge rules (§5)", () => {
  it("lets a nearer layer replace an inherited prompt", () => {
    const merged = mergeOperationChain([{ prompt: "inherited" }, { prompt: "nearer" }]);
    expect(merged.prompt).toBe("nearer");
  });

  /**
   * `JSON.parse` creates `__proto__` as a real OWN property rather than invoking the setter, so it
   * reaches the merge through `Object.entries` — and assigning it back out DOES invoke the setter,
   * replacing the merged object's prototype with authored content. The result then carries fields
   * that property access sees and `Object.keys`/spread/`JSON.stringify` do not.
   */
  it("refuses to let an authored layer replace the merged object's prototype", () => {
    const authored = JSON.parse('{"model":"m","__proto__":{"function":"claude-code"}}') as never;
    const merged = mergeOperationChain([{}, authored]) as Record<string, unknown>;
    expect(merged["model"]).toBe("m");
    expect(merged["function"]).toBeUndefined();
    expect(Object.getPrototypeOf(merged)).toBe(Object.prototype);
  });

  it("refuses the same through a nested map-shaped field", () => {
    const authored = JSON.parse('{"args":{"a":1,"__proto__":{"b":2}}}') as never;
    const merged = mergeOperationChain([{ args: { keep: true } }, authored]);
    expect(merged.args).toEqual({ keep: true, a: 1 });
    expect(Object.getPrototypeOf(merged.args!)).toBe(Object.prototype);
  });

  it("replaces a binding rather than merging its keys", () => {
    const merged = mergeOperationChain([
      { input: { prompt: { kind: "text", binding: ".children.a.outputs.x" } } },
      { input: { prompt: { binding: ".inputs.instruction" } } },
    ]);
    // A merged binding would be `{ child, output, input }` — not a binding at all.
    expect(merged.input!.prompt!.binding).toEqual(".inputs.instruction");
    // Everything the nearer layer left out still comes from the outer one.
    expect(merged.input!.prompt!.kind).toBe("text");
  });

  it("drops call configuration when a layer changes the kind", () => {
    // A root defaulting a model would otherwise hand `model` to every UI gate in the subtree as if
    // the author had written it there.
    const merged = mergeOperationChain([
      { kind: "prompt", model: "anthropic/claude-sonnet-5", prompt: "go", session: "s" },
      { kind: "function", function: "choose_option", args: { options: ["a", "b"] } },
    ]);
    expect(merged.args).toEqual({ options: ["a", "b"] });
    expect(merged.model).toBeUndefined();
    expect(merged.prompt).toBeUndefined();
    // The execution environment is NOT kind-specific — the gate still joins the same session.
    expect(merged.session).toBe("s");
  });

  it("replaces `tools` so an empty list drops what was inherited", () => {
    expect(mergeOperationChain([{ tools: ["bash", "write_file"] }, { tools: [] }]).tools).toEqual([]);
  });

  it("merges permissions per tool", () => {
    const merged = mergeOperationChain([
      { permissions: { default: "ask", tools: { bash: "smart" } } },
      { permissions: { tools: { write_file: "deny" } } },
    ]);
    expect(merged.permissions).toEqual({ default: "ask", tools: { bash: "smart", write_file: "deny" } });
  });
});

describe("the derived sequence (§6)", () => {
  const composite = (sequence?: string[]): Record<string, StateDef> => ({
    root: {
      label: "Root",
      children: { first: { state: "root/first" }, second: { state: "root/second" } },
      ...(sequence !== undefined ? { sequence } : {}),
    },
    "root/first": leaf({ operation: { kind: "prompt", prompt: "1" } }),
    "root/second": leaf({ operation: { kind: "prompt", prompt: "2" } }),
  });

  it("defaults to the order the children were declared in", () => {
    const root = loadBundle(composite(), "root").states.root!;
    expect(root.sequence).toEqual(["first", "second"]);
    expect(root.sequenceAuthored).toBe(false);
  });

  it("keeps an authored sequence and marks it as authored", () => {
    const root = loadBundle(composite(["second", "first"]), "root").states.root!;
    expect(root.sequence).toEqual(["second", "first"]);
    expect(root.sequenceAuthored).toBe(true);
  });

  it("treats an empty sequence as 'no spine', not as absent", () => {
    const root = loadBundle(composite([]), "root").states.root!;
    expect(root.sequence).toEqual([]);
  });

  it("gives a childless state no sequence at all", () => {
    expect(loadBundle(composite(), "root").states["root/first"]!.sequence).toBeUndefined();
  });
});

describe("output bindings (§3.4)", () => {
  const producer: StateDef = {
    label: "Context",
    outputs: {
      plan_doc: { schema: { type: "string" } },
      notes: { schema: { type: "string" }, optional: true },
    },
    operation: { kind: "prompt", prompt: "go" },
  };
  const withOutputs = (outputs: StateDef["outputs"]): Record<string, StateDef> => ({
    root: { label: "Root", outputs, children: { context: { state: "root/context" } } },
    "root/context": producer,
  });

  const selectedKey = (state: { outputs?: Record<string, { binding?: unknown }> }, slot: string): unknown => {
    const binding = state.outputs![slot]!.binding as { op?: { input?: Record<string, { binding?: { text?: string } }> } };
    return binding.op?.input?.key?.binding?.text;
  };

  it("defaults `output` to the slot's own name", () => {
    const root = loadBundle(withOutputs({ plan_doc: { binding: ".children.context.outputs.plan_doc" } }), "root").states.root!;
    expect(selectedKey(root, "plan_doc")).toBe("plan_doc");
  });

  it("still honours an explicit `output` that differs from the slot", () => {
    const root = loadBundle(withOutputs({ summary: { binding: ".children.context.outputs.notes" } }), "root").states.root!;
    expect(selectedKey(root, "summary")).toBe("notes");
  });

  it("takes the child's whole output object with `*`", () => {
    const root = loadBundle(withOutputs({ everything: { binding: ".children.context.outputs" } }), "root").states.root!;
    // No `select` projection at all — the producer edge itself is the value.
    expect(root.outputs!.everything!.binding).toEqual({ op: "context" });
  });

  it("spreads a child's outputs under a prefix", () => {
    const root = loadBundle(withOutputs({ "ctx_*": { binding: ".children.context.outputs" } }), "root").states.root!;
    expect(Object.keys(root.outputs!).sort()).toEqual(["ctx_notes", "ctx_plan_doc"]);
    expect(selectedKey(root, "ctx_plan_doc")).toBe("plan_doc");
    // Each spread slot keeps the schema and the optionality of the output it republishes.
    expect(root.outputs!.ctx_plan_doc!.schema).toEqual({ type: "string" });
    expect(root.slotMeta!["outputs.ctx_notes"]!.optional).toBe(true);
  });

  it("spreads unprefixed with a bare `*` key", () => {
    const root = loadBundle(withOutputs({ "*": { binding: ".children.context.outputs" } }), "root").states.root!;
    expect(Object.keys(root.outputs!).sort()).toEqual(["notes", "plan_doc"]);
  });

  it("lets an explicitly declared slot win over the spread", () => {
    const root = loadBundle(
      withOutputs({
        "ctx_*": { binding: ".children.context.outputs" },
        ctx_plan_doc: { schema: { type: "string" }, binding: ".children.context.outputs.notes" },
      }),
      "root",
    ).states.root!;
    expect(selectedKey(root, "ctx_plan_doc")).toBe("notes");
  });

  it("refuses a spread that does not bind a child", () => {
    expect(() => loadBundle(withOutputs({ "ctx_*": { binding: { expr: "1 + 1" } } }), "root")).toThrow(
      /must bind a child's outputs/,
    );
  });
});

describe("children inferred from the directory (§6)", () => {
  const files = (): Record<string, StateDef> => ({
    root: { label: "Root" },
    "root/zebra": leaf({ operation: { kind: "prompt", prompt: "z" } }),
    "root/apple": leaf({ operation: { kind: "prompt", prompt: "a" } }),
    "root/apple/deeper": leaf({ operation: { kind: "prompt", prompt: "d" } }),
  });

  it("takes the states one segment below, keyed by basename, alphabetically", () => {
    const root = loadBundle(files(), "root").states.root!;
    expect(Object.keys(root.children!)).toEqual(["apple", "zebra"]);
    expect(root.children!.apple!.state).toBe("root/apple");
    // …and the derived sequence follows, so alphabetical is the running order too.
    expect(root.sequence).toEqual(["apple", "zebra"]);
  });

  it("infers recursively, one level at a time", () => {
    const bundle = loadBundle(files(), "root");
    expect(Object.keys(bundle.states["root/apple"]!.children!)).toEqual(["deeper"]);
    expect(Object.keys(bundle.states).sort()).toEqual(["root", "root/apple", "root/apple/deeper", "root/zebra"]);
  });

  it("treats an empty `children` as 'none', not as 'infer'", () => {
    const defs = files();
    defs.root = { label: "Root", children: {} };
    const bundle = loadBundle(defs, "root");
    expect(bundle.states.root!.children).toEqual({});
    expect(Object.keys(bundle.states)).toEqual(["root"]);
  });

  it("lets a declared `children` win outright", () => {
    const defs = files();
    defs.root = { label: "Root", children: { only: { state: "./zebra" } } };
    const root = loadBundle(defs, "root").states.root!;
    expect(Object.keys(root.children!)).toEqual(["only"]);
  });
});

describe("a child that names itself (§7.3)", () => {
  const files = (child: Record<string, unknown>): Record<string, StateDef> => ({
    root: { label: "Root", children: { goals: child as never } },
    "root/goals": leaf({ operation: { kind: "prompt", prompt: "go" } }),
  });

  it("defaults `state` to the state its key names", () => {
    const root = loadBundle(files({}), "root").states.root!;
    expect(root.children!.goals!.state).toBe("root/goals");
  });

  it("still lets a child be mounted under a different key", () => {
    const defs = files({ state: "./goals" });
    defs.root!.children = { renamed: { state: "./goals" } };
    expect(loadBundle(defs, "root").states.root!.children!.renamed!.state).toBe("root/goals");
  });

  it("keeps the wiring a declared child adds", () => {
    const defs = files({ inputs: { issue: { text: "x" } } });
    defs["root/goals"] = leaf({
      inputs: { issue: { schema: { type: "string" } } },
      operation: { kind: "prompt", prompt: "go" },
    });
    const child = loadBundle(defs, "root").states.root!.children!.goals!;
    expect(child.state).toBe("root/goals");
    expect(child.inputs!.issue).toEqual({ text: "x" });
  });
});

/**
 * The inherited search PATH (EXPRESSIONS.md §4.2) — `path` merges like every other array field
 * (replace, not union), with a `$INHERITED` sentinel to splice rather than an exemption from the
 * rule. Exempting it would have left no way to say "ignore what I inherited".
 */
describe("the inherited search path (§4.2)", () => {
  it("replaces by default, so a subtree can deliberately shadow everything", () => {
    const merged = mergeOperationChain([{ path: ["$JAIRA/lib", "$/functions"] }, { path: ["./ops"] }]);
    expect(merged.path).toEqual(["./ops"]);
  });

  it("splices what the chain supplied wherever `$INHERITED` appears", () => {
    const base = { path: ["$JAIRA/lib", "$/functions"] };
    expect(mergeOperationChain([base, { path: ["./ops", "$INHERITED"] }]).path).toEqual([
      "./ops",
      "$JAIRA/lib",
      "$/functions",
    ]);
    expect(mergeOperationChain([base, { path: ["$INHERITED", "./fallback"] }]).path).toEqual([
      "$JAIRA/lib",
      "$/functions",
      "./fallback",
    ]);
  });

  it("expands `$INHERITED` to nothing when the chain supplied no path", () => {
    expect(mergeOperationChain([{}, { path: ["$INHERITED", "./ops"] }]).path).toEqual(["./ops"]);
  });

  it("carries an ancestor's path down untouched", () => {
    expect(mergeOperationChain([{ path: ["$/functions"] }, { model: "m" }]).path).toEqual(["$/functions"]);
  });

  /** It steers RESOLUTION, so it must survive a kind change — a prompt op resolves references too. */
  it("survives a layer changing the operation kind", () => {
    const merged = mergeOperationChain([
      { kind: "prompt", model: "m", path: ["$/functions"] },
      { kind: "function", function: "f" },
    ]);
    expect(merged.path).toEqual(["$/functions"]);
    expect(merged.model).toBeUndefined(); // call config still drops, as it always did
  });
});
