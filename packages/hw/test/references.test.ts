/**
 * The one reference type, end to end (REFERENCES.md).
 *
 * Document references are transclusion resolved at load; runtime references are bindings resolved
 * per instance. Both use one grammar, and these pin the seam between them.
 */
import { describe, expect, it } from "vitest";
import type { PromptOp } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader.js";
import { resolveReference, selectProperty, type Vfs } from "../src/reference.js";
import { validateBundle } from "../src/validate.js";

const ROOT = "/p/.jaira";
const WF = `${ROOT}/workflows`;

/** A Vfs over a flat `absolute path → text` map. */
function vfsOf(files: Record<string, string>): Vfs {
  return {
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
}

const opts = (files: Record<string, string>) => ({
  defaultRoot: WF,
  roots: { JAIRA: ROOT, PROJECT: "/p" },
  vfs: vfsOf(files),
});

describe("the grammar", () => {
  const files = {
    [`${WF}/feature/plan.json`]: "{}",
    [`${ROOT}/types/user.json`]: "{}",
    [`${ROOT}/prompts/review.md`]: "hello",
  };

  it("splits file from property by longest match against the directory", () => {
    const o = opts(files);
    expect(resolveReference("feature/plan.children.critique", o)).toMatchObject({
      file: `${WF}/feature/plan.json`,
      property: ["children", "critique"],
    });
    expect(resolveReference("$/types/user.address", o)).toMatchObject({
      file: `${ROOT}/types/user.json`,
      property: ["address"],
    });
    // An extension in the reference is just part of the longest match — no list of known suffixes.
    expect(resolveReference("$/prompts/review.md", o)).toMatchObject({
      file: `${ROOT}/prompts/review.md`,
      property: [],
    });
  });

  it("treats `$` as shorthand for `$JAIRA`", () => {
    const o = opts(files);
    expect(resolveReference("$/types/user", o).file).toBe(resolveReference("$JAIRA/types/user", o).file);
  });

  it("keeps a bare target's canonical id bare, and drops the data suffix", () => {
    expect(resolveReference("feature/plan", opts(files)).id).toBe("feature/plan");
  });

  it("reads `.foo` as a property of the current file, and `./foo` as a sibling path", () => {
    const o = { ...opts(files), from: "feature/plan" };
    expect(resolveReference(".children.critique", o)).toEqual({ property: ["children", "critique"], local: true });
    expect(resolveReference("./goals", { ...o, vfs: vfsOf({ ...files, [`${WF}/feature/plan/goals.json`]: "{}" }) }).id).toBe(
      "feature/plan/goals",
    );
  });

  it("warns when a shorter candidate also matches", () => {
    // BOTH candidates must genuinely provide `address` for this to be an ambiguity at all: since
    // SPEC §7.5.2, a `user.json` that lacks the property is not competing for the reference and
    // warning about it would be noise. Here it has one, so removing either file really would change
    // what this reference means.
    const warnings: string[] = [];
    const ambiguous = {
      ...files,
      [`${ROOT}/types/user.json`]: '{"address": {}}',
      [`${ROOT}/types/user.address.json`]: "{}",
    };
    const o = { ...opts(ambiguous), onWarn: (m: string) => warnings.push(m) };
    expect(resolveReference("$/types/user.address", o).file).toBe(`${ROOT}/types/user.address.json`);
    expect(warnings.join("\n")).toMatch(/also matches/);
  });

  it("does NOT warn when the shorter candidate lacks the property", () => {
    // `user.json` is `{}`, so it never held `address` — nothing is being shadowed and there is
    // nothing to report.
    const warnings: string[] = [];
    const o = { ...opts({ ...files, [`${ROOT}/types/user.address.json`]: "{}" }), onWarn: (m: string) => warnings.push(m) };
    expect(resolveReference("$/types/user.address", o).file).toBe(`${ROOT}/types/user.address.json`);
    expect(warnings.join("\n")).not.toMatch(/also matches/);
  });

  it("errors on a directory, an unknown root, and a missing file", () => {
    const o = opts(files);
    expect(() => resolveReference("$/types/", o)).toThrow(/directory/);
    expect(() => resolveReference("$NOPE/x", o)).toThrow(/unknown root/);
    expect(() => resolveReference("$/types/ghost", o)).toThrow(/matches no file/);
    expect(() => resolveReference("https://example.com/x", o)).toThrow(/unknown scheme/);
  });
});

describe("transclusion", () => {
  it("splices an operation in from a fragment, and lets siblings override it", () => {
    const files = {
      [`${ROOT}/lib/review.json`]: JSON.stringify({
        operation: { kind: "prompt", prompt: "Review it.", model: "base", temperature: 0.2 },
      }),
      [`${WF}/r.json`]: JSON.stringify({
        outputs: { verdict: { schema: { type: "string" } } },
        operation: { $ref: "$/lib/review.operation", model: "override" },
      }),
    };
    const op = loadBundle({ "r.json": JSON.parse(files[`${WF}/r.json`]!) }, "r", opts(files)).states.r!
      .operation as PromptOp<never>;
    expect(op.user).toBe("Review it.");
    // The sibling wins; everything it did not mention survives from the fragment.
    expect(op.config).toEqual({ model: "override", temperature: 0.2 });
  });

  it("loads a prompt from a .md file as text", () => {
    const files = {
      [`${ROOT}/prompts/goals.md`]: "Extract goals from {{.inputs.issue}}.",
      [`${WF}/g.json`]: "{}",
    };
    const def = { operation: { kind: "prompt", prompt: { $ref: "$/prompts/goals.md" }, model: "m" } };
    const op = loadBundle({ "g.json": def }, "g", opts(files)).states.g!.operation as PromptOp<never>;
    expect(op.user).toBe("Extract goals from {{.inputs.issue}}.");
  });

  it("reads a YAML fragment as an object", () => {
    const files = {
      [`${ROOT}/lib/env.yaml`]: "kind: prompt\nmodel: from-yaml\n",
      [`${WF}/y.json`]: "{}",
    };
    const def = { environment: "$/lib/env", operation: { prompt: "go" } };
    const op = loadBundle({ "y.json": def }, "y", opts(files)).states.y!.operation as PromptOp<never>;
    expect(op.kind).toBe("prompt");
    expect(op.config).toEqual({ model: "from-yaml" });
  });

  it("composes a schema out of a shared type library", () => {
    const files = {
      [`${ROOT}/types/markdown.json`]: JSON.stringify({ type: "string", contentMediaType: "text/markdown" }),
      [`${WF}/s.json`]: "{}",
    };
    const def = {
      outputs: { doc: { schema: "$/types/markdown" } },
      operation: { kind: "prompt", prompt: "go", model: "m" },
    };
    const state = loadBundle({ "s.json": def }, "s", opts(files)).states.s!;
    expect(state.outputs!.doc!.schema).toEqual({ type: "string", contentMediaType: "text/markdown" });
    // …and the derived kind follows the expanded schema, so the artifact rule still fires.
    expect(state.outputs!.doc!.kind).toBe("blob");
  });

  it("leaves JSON Schema's own $ref alone", () => {
    const files = { [`${WF}/j.json`]: "{}" };
    const def = {
      outputs: { doc: { schema: { $ref: "#/definitions/thing", definitions: { thing: { type: "string" } } } } },
      operation: { kind: "prompt", prompt: "go", model: "m" },
    };
    const state = loadBundle({ "j.json": def }, "j", opts(files)).states.j!;
    expect((state.outputs!.doc!.schema as { $ref?: string }).$ref).toBe("#/definitions/thing");
  });

  it("refuses a cycle", () => {
    const files = {
      [`${ROOT}/lib/a.json`]: JSON.stringify({ operation: { $ref: "$/lib/b.operation" } }),
      [`${ROOT}/lib/b.json`]: JSON.stringify({ operation: { $ref: "$/lib/a.operation" } }),
      [`${WF}/c.json`]: "{}",
    };
    expect(() => loadBundle({ "c.json": { operation: "$/lib/a.operation" } }, "c", opts(files))).toThrow(/cycle/);
  });

  it("does NOT treat a prompt that looks like a path as a reference", () => {
    // `prompt` is a string position, so a string there is a string — the whole point of §3.
    const files = { [`${WF}/p.json`]: "{}" };
    const def = { operation: { kind: "prompt", prompt: "Review feature/plan.outputs.summary", model: "m" } };
    const op = loadBundle({ "p.json": def }, "p", opts(files)).states.p!.operation as PromptOp<never>;
    expect(op.user).toBe("Review feature/plan.outputs.summary");
  });
});

describe("runtime references", () => {
  const files = { [`${WF}/w.json`]: "{}" };
  const bundle = (defs: Record<string, unknown>) => loadBundle(defs, "w", opts(files));

  it("lowers a child read to the same edge the tagged form produces", () => {
    const defs = {
      "w.json": {
        inputs: { issue: { schema: { type: "string" } } },
        outputs: { plan: { schema: { type: "string" }, binding: ".children.c.outputs.plan" } },
        children: { c: { state: "./c", inputs: { issue: ".inputs.issue" } } },
      },
      "w/c.json": {
        inputs: { issue: { schema: { type: "string" } } },
        outputs: { plan: { schema: { type: "string" } } },
        operation: { kind: "prompt", prompt: "go", model: "m" },
      },
    };
    const b = bundle(defs);
    expect(b.states.w!.outputs!.plan!.binding).toMatchObject({ op: { kind: "function", functionRef: "select" } });
    expect(b.states.w!.children!.c!.inputs!.issue).toMatchObject({ op: { kind: "function", functionRef: "scope.get" } });
    expect(validateBundle(b).errors).toEqual([]);
  });

  it("reads the whole outputs object without a projection", () => {
    const defs = {
      "w.json": {
        outputs: { all: { binding: ".children.c.outputs" } },
        children: { c: { state: "./c" } },
      },
      "w/c.json": { outputs: { plan: { schema: { type: "string" } } }, operation: { kind: "prompt", prompt: "go", model: "m" } },
    };
    expect(bundle(defs).states.w!.outputs!.all!.binding).toEqual({ op: "c" });
  });

  it("rejects a namespace that does not exist", () => {
    const defs = {
      "w.json": { outputs: { x: { binding: ".nope.thing" } }, operation: { kind: "prompt", prompt: "go", model: "m" } },
    };
    expect(() => bundle(defs)).toThrow(/not a runtime namespace/);
  });

  it("insists a binding reference start with a dot", () => {
    const child = { outputs: { x: { schema: { type: "string" } } }, operation: { kind: "prompt", prompt: "go", model: "m" } };
    const withBinding = (binding: string) => ({
      "w.json": { outputs: { x: { binding } }, children: { c: { state: "./c" } } },
      "w/c.json": child,
    });
    // With the dot it reads the child. Without it, the same text is a DOCUMENT reference — resolved
    // against the filesystem, where nothing of that name exists.
    expect(bundle(withBinding(".children.c.outputs.x")).states.w!.outputs!.x!.binding).toMatchObject({
      op: { kind: "function", functionRef: "select" },
    });
    expect(() => bundle(withBinding("children.c.outputs.x"))).toThrow(/matches no file/);
  });

  it("reads a string binding with operators or a call as an expression", () => {
    // The counterpart: a string expansion cannot read as a path is left for the desugarer, so an
    // expression needs no `{ expr }` wrapper to be one (EXPRESSIONS.md §1).
    const defs = {
      "w.json": {
        inputs: { n: { schema: { type: "integer" } } },
        outputs: { doubled: { binding: "add(.inputs.n, .inputs.n)" } },
        operation: { kind: "prompt", prompt: "go", model: "m" },
      },
    };
    expect(bundle(defs).states.w!.outputs!.doubled!.binding).toMatchObject({ op: { kind: "function", functionRef: "add" } });
  });

  it("reads a bracket index as an expression, not as a path segment", () => {
    // Brackets belong to the expression grammar (sugar for `at`), so `.inputs.xs[-1]` must reach
    // the desugarer whole — split as a runtime PATH it would name the nonexistent input 'xs[-1]'.
    const defs = {
      "w.json": {
        inputs: { xs: { schema: { type: "array", items: { type: "integer" } } } },
        outputs: { last: { binding: ".inputs.xs[-1]" } },
        operation: { kind: "prompt", prompt: "go", model: "m" },
      },
    };
    expect(bundle(defs).states.w!.outputs!.last!.binding).toMatchObject({ op: { kind: "function", functionRef: "at" } });
  });

  /**
   * A property path walks OWN properties. Transclusion splices what it finds into the document, so
   * an inherited hit put a FUNCTION where a node was expected instead of reporting that the file has
   * no such property.
   */
  it("does not walk a property path onto the prototype", () => {
    expect(() => selectProperty({ a: 1 }, ["constructor"], "$/types/user.constructor")).toThrow(/has no 'constructor'/);
    expect(() => selectProperty({ a: 1 }, ["toString"], "$/types/user.toString")).toThrow(/has no 'toString'/);
    expect(selectProperty({ a: 1 }, ["a"], "$/types/user.a")).toBe(1);
  });
});

/**
 * The search PATH (EXPRESSIONS.md §4): a bare reference is tried against each root in turn and the
 * first match wins, with shell `PATH` semantics — and with identity folding back from EVERY entry,
 * which is what makes the path a LAYERING mechanism rather than only a convenience.
 */
describe("a bare reference searches the path", () => {
  const LIB = "/p/.jaira/lib";
  const files = {
    [`${WF}/local.json`]: "{}",
    [`${WF}/both.json`]: JSON.stringify({ label: "from workflows" }),
    [`${LIB}/both.json`]: JSON.stringify({ label: "from lib" }),
    [`${LIB}/shared.json`]: JSON.stringify({ label: "from lib" }),
  };
  const withPath = (onWarn?: (m: string) => void) => ({
    defaultRoot: [WF, LIB],
    roots: { JAIRA: ROOT, PROJECT: "/p" },
    vfs: vfsOf(files),
    ...(onWarn ? { onWarn } : {}),
  });

  it("finds a reference at the first entry", () => {
    expect(resolveReference("local", withPath()).file).toBe(`${WF}/local.json`);
  });

  it("falls through to a later entry when the first has no match", () => {
    expect(resolveReference("shared", withPath()).file).toBe(`${LIB}/shared.json`);
  });

  it("lets an earlier entry shadow a later one", () => {
    expect(resolveReference("both", withPath()).file).toBe(`${WF}/both.json`);
  });

  /**
   * §4.1 — a canonical id keys the snapshot hash, the event log and task rows, and it folds back
   * from every entry so that `shared` names one state whichever layer supplied it. Two files at two
   * entries therefore share an id, which is not a collision but an OVERRIDE: resolution has already
   * picked the winner before identity is asked. What keeps a RUN honest is that execution reads a
   * pinned snapshot of the resolved bundle rather than re-resolving the live path.
   */
  it("gives a bare id to a match under ANY entry, so a layer and its override share one", () => {
    expect(resolveReference("local", withPath()).id).toBe("local");
    expect(resolveReference("both", withPath()).id).toBe("both");
    // Found further along ⇒ still bare. This is the rule the base root exists for.
    expect(resolveReference("shared", withPath()).id).toBe("shared");
  });

  it("reports a reference that matches nowhere, naming the roots it tried", () => {
    expect(() => resolveReference("absent", withPath())).toThrow(/matches no file on the path/);
  });

  it("warns when an earlier entry shadows a later one", () => {
    const warnings: string[] = [];
    resolveReference("both", withPath((m) => warnings.push(m)));
    expect(warnings.join("\n")).toMatch(/also matches further along the path/);
    // No shadow, no warning.
    const quiet: string[] = [];
    resolveReference("shared", withPath((m) => quiet.push(m)));
    expect(quiet).toEqual([]);
  });

  /** The non-bare forms are unchanged: only a bare reference searches. */
  it("does not search for rooted, absolute or relative references", () => {
    expect(resolveReference("$JAIRA/lib/shared", withPath()).file).toBe(`${LIB}/shared.json`);
    expect(() => resolveReference("./shared", { ...withPath(), from: "local" })).toThrow(/matches no file/);
  });
});

/** The path, inherited through `environment` and used to resolve a bare reference (§4 end to end). */
describe("an inherited path resolves a bare reference", () => {
  const LIB = "/p/.jaira/lib";
  const files = {
    [`${WF}/plan.json`]: "{}",
    [`${WF}/plan/leaf.json`]: "{}",
    [`${LIB}/review.json`]: JSON.stringify({ kind: "prompt", prompt: "Review it.", model: "from-lib" }),
  };
  const load = (defs: Record<string, unknown>) =>
    loadBundle(defs, "plan", { defaultRoot: WF, roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs: vfsOf(files) });

  it("finds a fragment the root's `path` put on the search list", () => {
    // `review` is bare and lives nowhere under the workflows root — only the inherited path finds it.
    const state = load({
      "plan.json": { environment: { path: [WF, LIB] }, children: { leaf: { state: "./leaf" } } },
      "plan/leaf.json": { outputs: { v: { schema: { type: "string" } } }, operation: "review" },
    }).states["plan/leaf"]!;
    expect((state.operation as PromptOp<never>).user).toBe("Review it.");
    expect((state.operation as PromptOp<never>).config).toEqual({ model: "from-lib" });
  });

  it("does not find it without the path", () => {
    expect(() =>
      load({
        "plan.json": { children: { leaf: { state: "./leaf" } } },
        "plan/leaf.json": { outputs: { v: { schema: { type: "string" } } }, operation: "review" },
      }),
    ).toThrow(/matches no file/);
  });

  /**
   * §9's cycle, and where it breaks: the path a state resolves under is read off the RAW
   * `environment`, because a reference cannot be resolved with a path that has not been loaded yet.
   */
  it("does not apply a path declared inside a transcluded environment to that same transclusion", () => {
    const withEnvFragment = {
      ...files,
      [`${LIB}/env.json`]: JSON.stringify({ kind: "prompt", model: "m", path: [WF, LIB] }),
    };
    // The environment itself is reachable (it is `$`-rooted), and its path then applies to the
    // state's OTHER references — but not to locating the environment fragment.
    const bundle = loadBundle(
      {
        "plan.json": { environment: "$JAIRA/lib/env", children: { leaf: { state: "./leaf" } } },
        "plan/leaf.json": { outputs: { v: { schema: { type: "string" } } }, operation: { prompt: "go" } },
      },
      "plan",
      { defaultRoot: WF, roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs: vfsOf(withEnvFragment) },
    );
    expect((bundle.states["plan/leaf"]!.operation as PromptOp<never>).config).toEqual({ model: "m" });
  });
});

/**
 * Calling an operation from an expression (EXPRESSIONS.md §3) — the callee resolved along the path.
 *
 * The callee is an operation DOCUMENT: the same shape an `operation` block is written in. That is
 * what makes a built-in and a project's own indistinguishable — they differ only in where the name
 * resolves.
 */
describe("an expression calls an operation resolved along the path", () => {
  const FUNCTIONS = "/p/.jaira/functions";
  const files = {
    [`${WF}/plan.json`]: "{}",
    // An operation document, declaring the parameters a positional call binds against (§3.3).
    [`${FUNCTIONS}/classify.json`]: JSON.stringify({
      kind: "prompt",
      prompt: "Classify {{.inputs.text}}",
      model: "m",
      input: { text: { kind: "text", index: 0 } },
    }),
  };
  const load = (def: unknown) =>
    loadBundle({ "plan.json": def }, "plan", {
      defaultRoot: [WF, FUNCTIONS],
      roots: { JAIRA: ROOT, PROJECT: "/p" },
      vfs: vfsOf(files),
    });

  it("lowers a call to a producer edge on the resolved operation, with the argument bound", () => {
    const state = load({
      inputs: { issue: { schema: { type: "string" } } },
      outputs: { verdict: { binding: { expr: "classify(.inputs.issue)" } } },
      operation: { kind: "prompt", prompt: "go", model: "m" },
    }).states.plan!;

    const binding = state.outputs!.verdict!.binding as {
      op: { kind: string; user?: string };
      parameters?: Record<string, { binding?: unknown }>;
    };
    // The edge carries the RESOLVED operation, not a name to look up later.
    expect(binding.op.kind).toBe("prompt");
    expect(binding.op.user).toBe("Classify {{.inputs.text}}");
    // …and the positional argument bound to the parameter the document declared at index 0.
    expect(Object.keys(binding.parameters ?? {})).toEqual(["text"]);
    expect(binding.parameters!.text!.binding).toBeDefined();
  });

  it("nests a call inside an operator, since an operator IS an application", () => {
    const state = load({
      inputs: { issue: { schema: { type: "string" } } },
      outputs: { hot: { binding: { expr: "classify(.inputs.issue).severity === 'high'" } } },
      operation: { kind: "prompt", prompt: "go", model: "m" },
    }).states.plan!;
    const binding = state.outputs!.hot!.binding as { op: { functionRef?: string } };
    expect(binding.op.functionRef).toBe("op.strictEq");
  });

  it("reports a name that resolves to no operation", () => {
    expect(() =>
      load({
        outputs: { v: { binding: { expr: "nosuchthing(1)" } } },
        operation: { kind: "prompt", prompt: "go", model: "m" },
      }),
    ).toThrow(/matches no file|not a known operation/);
  });

  /**
   * A callee is ALWAYS a reference, so children and callees are separate namespaces. A child named
   * `classify` and a callee `classify` coexist: the child is wired and read as a child, the call
   * resolves along the path. "Calling a child" was dropped rather than deferred — a child is a
   * STATE, and calling one would duplicate what wiring it already does.
   */
  it("lets a child and a callee share a name without interacting", () => {
    const state = load({
      children: { classify: { state: "./classify" } },
      outputs: { v: { binding: { expr: "classify(1)" } } },
      operation: { kind: "prompt", prompt: "go", model: "m" },
    }).states.plan!;
    // The binding resolved to the PATH operation, not to the child.
    const binding = state.outputs!.v!.binding as { op: { kind: string } };
    expect(binding.op.kind).toBe("prompt");
    // …and the child is still declared, untouched.
    expect(Object.keys(state.children ?? {})).toEqual(["classify"]);
  });
});
