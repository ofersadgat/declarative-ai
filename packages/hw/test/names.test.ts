/**
 * Scoped names in any position (NAMES.md).
 *
 * `session.test.ts` pins the mechanic through its first user. What is pinned here is everything the
 * second reading of a variable string adds: where a name is SCOPED once names can be declared, what
 * an inner entry of the same name means, that one identity has one configuration and one type, and
 * what a name reads as in a value position.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { Vfs } from "../src/reference.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";
import { InMemoryPersistence, type WorkflowMetrics } from "../src/ports.js";
import { newRegistry, ok } from "./fakes.js";

const ROOT = "/p/.jaira";
const WF = `${ROOT}/workflows`;

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

/** Load through the real reference pass: state files by id, plus any fragments by absolute path. */
const load = (states: Record<string, unknown>, rootId: string, fragments: Record<string, unknown> = {}) => {
  const onDisk: Record<string, string> = {};
  const files: Record<string, StateDef> = {};
  for (const [id, def] of Object.entries(states)) {
    onDisk[`${WF}/${id}.json`] = JSON.stringify(def);
    files[`${id}.json`] = def as StateDef;
  }
  for (const [path, value] of Object.entries(fragments)) onDisk[path] = typeof value === "string" ? value : JSON.stringify(value);
  return loadBundle(files, rootId, { defaultRoot: WF, roots: { JAIRA: ROOT }, vfs: vfsOf(onDisk) });
};

const errorsOf = (states: Record<string, unknown>, rootId: string): string[] =>
  validateBundle(load(states, rootId)).errors.map((e) => `${e.stateId ?? ""}: ${e.message}`);

const leaf = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({
  outputs: {},
  operation: { prompt: "go", model: "m", ...extra },
});

describe("where a name is scoped (§3)", () => {
  it("is the state that wrote a VISIBLE entry — so two children that write one word share it", () => {
    const bundle = load(
      {
        root: { environment: { names: { review: {} } }, children: { a: {}, b: {} } },
        "root/a": leaf({ session: "review" }),
        "root/b": leaf({ session: "review" }),
      },
      "root",
    );
    expect(bundle.states["root/a"]!.environment!.session).toEqual({ $ref: "review", $in: "root" });
    expect(bundle.states["root/b"]!.environment!.session).toEqual({ $ref: "review", $in: "root" });
  });

  it("is the writer of the USE when no entry is visible — what a session name has always meant", () => {
    const bundle = load({ root: { children: { a: {}, b: {} } }, "root/a": leaf({ session: "review" }), "root/b": leaf({ session: "review" }) }, "root");
    expect(bundle.states["root/a"]!.environment!.session).toEqual({ $ref: "review", $in: "root/a" });
    expect(bundle.states["root/b"]!.environment!.session).toEqual({ $ref: "review", $in: "root/b" });
  });

  it("lets `$in` redirect either", () => {
    const bundle = load(
      {
        root: { environment: { names: { review: {} } }, children: { mid: {} } },
        "root/mid": { children: { a: {} } },
        "root/mid/a": leaf({ session: { $ref: "review", $in: "parent" } }),
      },
      "root",
    );
    expect(bundle.states["root/mid/a"]!.environment!.session).toEqual({ $ref: "review", $in: "root/mid" });
  });
});

describe("an inner entry of the same name (§4)", () => {
  const tree = (inner: Record<string, unknown>) => ({
    root: { environment: { names: { impl: { from: "main", depth: 1 } } }, children: { mid: {} } },
    "root/mid": { environment: { names: { impl: inner } }, children: { a: {} } },
    "root/mid/a": leaf({ args: undefined }),
  });

  it("OVERRIDES by default: a new name scoped here, configured only by this block", () => {
    const bundle = load(tree({ from: "dev" }), "root");
    expect(bundle.states["root/mid"]!.names).toEqual({ impl: { from: "dev" } });
    expect(bundle.states.root!.names).toEqual({ impl: { from: "main", depth: 1 } });
  });

  it("pastes the ENCLOSING entry for `$ref` to its own name — templating, not a cycle", () => {
    const bundle = load(tree({ $ref: "impl", from: "dev" }), "root");
    expect(bundle.states["root/mid"]!.names).toEqual({ impl: { from: "dev", depth: 1 } });
    // Nothing links the two afterwards: the outer one is untouched.
    expect(bundle.states.root!.names).toEqual({ impl: { from: "main", depth: 1 } });
  });

  it("refuses a paste with nothing enclosing to paste", () => {
    expect(errorsOf({ root: { environment: { names: { impl: { $ref: "impl", from: "dev" } } } } }, "root")).toEqual([
      expect.stringMatching(/'impl' means the ENCLOSING 'impl', and no enclosing scope declares one/),
    ]);
  });

  it("CONTRIBUTES with `$in`: the ancestor's identity, with this added — and no new scope", () => {
    const bundle = load(tree({ $in: "parent", branch: "x" }), "root");
    expect(bundle.states.root!.names).toEqual({ impl: { from: "main", depth: 1, branch: "x" } });
    expect(bundle.states["root/mid"]!.names).toBeUndefined();
  });

  it("collects contributions before any use resolves, so a SIBLING sees them", async () => {
    const seen = await argsSeen(
      {
        root: { environment: { names: { mr: { to: "origin" } } }, children: { reader: {}, writer: {} }, sequence: ["reader", "writer"] },
        // `reader` runs first and is declared first, and still reads what `writer` contributes.
        "root/reader": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr" } } } },
        "root/writer": { environment: { names: { mr: { $in: "parent", target: "main" } } }, outputs: {}, operation: { function: "record", args: {} } },
      },
      "root",
    );
    expect(seen[0]!.remote).toEqual({ to: "origin", target: "main", $key: "mr#/" });
  });

  it("does not deep-merge `names` down the tree — the nearest entry wins WHOLE", () => {
    const bundle = load(tree({ from: "dev" }), "root");
    expect(bundle.states["root/mid/a"] && bundle.states["root/mid"]!.names!.impl).not.toHaveProperty("depth");
  });
});

describe("one identity has one configuration, and one type (§4, §5)", () => {
  it("reports two writers that disagree, at the second one", () => {
    const errors = errorsOf(
      {
        root: { environment: { names: { impl: { from: "main" } } }, children: { a: {} } },
        "root/a": { environment: { names: { impl: { $in: "parent", from: "dev" } } }, children: {} },
      },
      "root",
    );
    expect(errors).toEqual([expect.stringMatching(/^root\/a: 'impl' is given 'from': "dev" here and "main" at root: environment\.names\.impl/)]);
  });

  it("reports conflicting override keys at two USE sites the same way", () => {
    const errors = errorsOf(
      {
        root: { environment: { names: { mr: {} } }, children: { a: {}, b: {} } },
        "root/a": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr", draft: true } } } },
        "root/b": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr", draft: false } } } },
      },
      "root",
    );
    expect(errors.filter((e) => /is given 'draft'/.test(e))).toHaveLength(1);
  });

  it("is quiet when the writers agree", () => {
    const errors = errorsOf(
      {
        root: { environment: { names: { mr: {} } }, children: { a: {}, b: {} } },
        "root/a": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr", draft: true } } } },
        "root/b": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr", draft: true } } } },
      },
      "root",
    );
    expect(errors.filter((e) => /is given/.test(e))).toEqual([]);
  });

  it("refuses a name bound as two types, AT THE BIND POINT that disagrees", () => {
    const errors = errorsOf(
      {
        root: { environment: { names: { impl: {} }, session: "impl" }, children: { a: {} } },
        "root/a": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "impl" } } } },
      },
      "root",
    );
    expect(errors).toEqual([expect.stringMatching(/^root\/a: 'impl' is a session \(bound at root: environment\.session\) and cannot also be a value here/)]);
  });
});

describe("the second reading of a variable string (§1, §6)", () => {
  it("reads a bare string in an object position as a declared NAME before it reads it as a file", () => {
    const bundle = load(
      {
        root: { environment: { names: { strict: { profile: "read-only" } } }, children: { a: {} } },
        "root/a": leaf({ permissions: "strict" }),
      },
      "root",
    );
    expect(bundle.states["root/a"]!.fields?.map((f) => f.path)).toContain("environment.permissions");
  });

  it("still resolves an undeclared bare string as a file off the default root", () => {
    const bundle = load({ root: { operation: "lib/op.operation", outputs: {} }, "lib/op": { operation: { prompt: "from the file", model: "m" } } }, "root");
    expect((bundle.states.root!.operation as { user?: string }).user).toBe("from the file");
  });

  it("never reads a `$…` string as a name, whatever is declared", () => {
    const bundle = load(
      { root: { environment: { names: { "$/lib/op": {} } as never }, outputs: {}, operation: { $ref: "$/lib/op.operation" } } },
      "root",
      { [`${ROOT}/lib/op.json`]: { operation: { prompt: "a file", model: "m" } } },
    );
    expect((bundle.states.root!.operation as { user?: string }).user).toBe("a file");
  });

  it("makes an undefined name that no position can provide a LOAD error, said as one", () => {
    expect(() => load({ root: { outputs: {}, operation: { function: "record", args: { remote: { $ref: "nowhere" } } } } }, "root")).toThrow(
      /'nowhere' is not a name any enclosing scope declares \(environment\.names\), and this position cannot provide one/,
    );
  });

  it("drops an alternative that REFERENCES something that is not there — that is what 'usable' means", () => {
    const warnings: string[] = [];
    const onDisk = {
      [`${WF}/root.json`]: "{}",
      [`${ROOT}/roles.json`]: JSON.stringify({ plan: { model: "from-the-layer" } }),
    };
    const def = { outputs: {}, operation: { prompt: "go", model: { $any: [{ $ref: "$/missing.plan" }, { $ref: "$/roles.plan" }, null] } } };
    const bundle = loadBundle({ "root.json": def as never }, "root", { defaultRoot: WF, roots: { JAIRA: ROOT }, vfs: vfsOf(onDisk), onWarn: (m) => warnings.push(m) });
    expect(warnings).toEqual([expect.stringMatching(/operation\.model\.\$any\.0: alternative skipped/)]);
    // `null` is a value, not an absence: it stays in the list.
    expect(JSON.stringify(bundle.states.root!.fields)).toContain("from-the-layer");
    expect(() =>
      loadBundle({ "root.json": { outputs: {}, operation: { prompt: "go", model: { $any: [{ $ref: "$/missing.plan" }] } } } as never }, "root", {
        defaultRoot: WF,
        roots: { JAIRA: ROOT },
        vfs: vfsOf(onDisk),
      }),
    ).toThrow(/no alternative is usable/);
  });

  it("needs no entry where the position provides — `session` creates what the name is bound to", () => {
    expect(errorsOf({ root: leaf({ session: "draft" }) }, "root")).toEqual([]);
  });
});

/** Run a tree whose leaves call `record`, and return what each call was handed, in order. */
async function argsSeen(states: Record<string, unknown>, rootId: string): Promise<Array<Record<string, unknown>>> {
  const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };
  const seen: Array<Record<string, unknown>> = [];
  const registry = newRegistry();
  registry.functions.set(
    "record",
    hostFunction(async (inputs: Record<string, unknown>) => {
      seen.push(inputs);
      return ok({}) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  const engine = new WorkflowEngine({ bundle: load(states, rootId), registry, validator: new SchemaValidator(), persistence: new InMemoryPersistence() });
  const result = await engine.run({ inputs: {} });
  expect(result.failure?.reason).toBeUndefined();
  return seen;
}

describe("a name in a VALUE position (§3, §10)", () => {
  it("reads as its configuration, with the identity's key beside it", async () => {
    const seen = await argsSeen(
      {
        root: { environment: { names: { review: { to: "origin" } } }, children: { gate: {} } },
        "root/gate": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "review", target: "main" }, prompt: "Review it" } } },
      },
      "root",
    );
    expect(seen).toEqual([{ prompt: "Review it", remote: { to: "origin", target: "main", $key: "review#/" } }]);
  });

  it("keys on the scoping INSTANCE: one identity above a loop, a new one per pass inside it", async () => {
    const loop = (names: "above" | "inside") => ({
      root: {
        ...(names === "above" ? { environment: { names: { mr: {} } } } : {}),
        limits: { max_iterations: 2 },
        children: { pass: { transitions: [{ to: "pass", when: ".run.iteration < .limits.max_iterations" }] } },
      },
      "root/pass": { ...(names === "inside" ? { environment: { names: { mr: {} } } } : {}), children: { gate: {} } },
      "root/pass/gate": { outputs: {}, operation: { function: "record", args: { remote: { $ref: "mr" } } } },
    });
    const keys = async (names: "above" | "inside") => (await argsSeen(loop(names), "root")).map((call) => (call.remote as { $key: string }).$key);
    // Above the loop the scope is one instance on every pass; inside it, each pass is a new one. No
    // rule about loops says so — the key is the anchoring instance's ADDRESS, and an address counts
    // entries.
    const above = await keys("above");
    expect(above.length).toBeGreaterThan(1);
    expect(new Set(above)).toEqual(new Set(["mr#/"]));
    const inside = await keys("inside");
    expect(inside.length).toBe(above.length);
    expect(new Set(inside).size).toBe(inside.length);
    expect(inside[0]).toBe("mr#pass");
  });

  it("reads a name INSIDE a literal argument in place, at any depth — objects and arrays", async () => {
    const seen = await argsSeen(
      {
        root: {
          environment: { names: { mr: { to: "origin" } } },
          inputs: { who: { schema: { type: "string" }, default: "mara" } },
          outputs: {},
          operation: {
            function: "record",
            args: {
              options: { label: "gate", remote: { $ref: "mr", draft: true }, reviewers: ["ann", { $binding: ".inputs.who" }], nested: { deep: [{ $ref: "mr" }] } },
              plain: { untouched: [1, 2] },
            },
          },
        },
      },
      "root",
    );
    const remote = { to: "origin", draft: true, $key: "mr#/" };
    // (`who` arrives too: a state input fills a function's slot of the same name, as it always has.)
    expect(seen).toEqual([{ options: { label: "gate", remote, reviewers: ["ann", "mara"], nested: { deep: [remote] } }, plain: { untouched: [1, 2] }, who: "mara" }]);
  });
});

describe("a function's default arguments (§7)", () => {
  const gate = (args?: Record<string, unknown>, fn = "record") => ({ outputs: {}, operation: { function: fn, ...(args !== undefined ? { args } : {}) } });

  it("takes the state's own `args`, then the NEAREST block, merged per key down the tree", async () => {
    const seen = await argsSeen(
      {
        root: { environment: { functions: { record: { args: { remote: { to: "origin" }, draft: false, tone: "plain" } } } }, children: { mid: {} } },
        "root/mid": { environment: { functions: { record: { args: { draft: true } } } }, children: { gate: {} } },
        "root/mid/gate": gate({ tone: "blunt", remote: { target: "main" } }),
      },
      "root",
    );
    expect(seen).toEqual([{ remote: { to: "origin", target: "main" }, draft: true, tone: "blunt" }]);
  });

  it("is not the operation's `args`, so a change of KIND does not drop it", async () => {
    // The root's own layer is a PROMPT's; a function leaf drops every kind-specific field it
    // inherited (`KIND_SPECIFIC`) — which is exactly what an `args` default written there would be.
    const seen = await argsSeen(
      {
        root: { environment: { prompt: "a prompt default", model: "m", functions: { record: { args: { draft: true } } } }, children: { gate: {} } },
        "root/gate": gate({}),
      },
      "root",
    );
    expect(seen).toEqual([{ draft: true }]);
  });

  it("reaches the function it names and no other", async () => {
    const seen = await argsSeen(
      { root: { environment: { functions: { other: { args: { draft: true } } } }, children: { gate: {} } }, "root/gate": gate({ tone: "plain" }) },
      "root",
    );
    expect(seen).toEqual([{ tone: "plain" }]);
  });

  it("lets a nearer `null` take a default away", async () => {
    const seen = await argsSeen(
      { root: { environment: { functions: { record: { args: { remote: { to: "origin" } } } } }, children: { gate: {} } }, "root/gate": gate({ remote: null }) },
      "root",
    );
    expect(seen).toEqual([{ remote: null }]);
  });

  it("carries a scoped name, anchored where the DEFAULT was written — the same request on every gate below", async () => {
    const seen = await argsSeen(
      {
        root: {
          environment: { names: { review: {} }, functions: { record: { args: { remote: { $ref: "review", to: "origin" } } } } },
          children: { a: {}, b: {} },
        },
        "root/a": gate({}),
        "root/b": gate({}),
      },
      "root",
    );
    expect(seen.map((call) => call.remote)).toEqual([
      { to: "origin", $key: "review#/" },
      { to: "origin", $key: "review#/" },
    ]);
  });

  it("replaces, rather than deep-merges, where either side is a name", async () => {
    const seen = await argsSeen(
      {
        root: { environment: { names: { review: {} }, functions: { record: { args: { remote: { $ref: "review" } } } } }, children: { gate: {} } },
        "root/gate": gate({ remote: { to: "elsewhere" } }),
      },
      "root",
    );
    expect(seen).toEqual([{ remote: { to: "elsewhere" } }]);
  });

  it("is TYPED by the function's signature, at the block that wrote the default", () => {
    const functions = new Map([
      [
        "record",
        {
          kind: "host",
          capabilities: { interactive: false, readOnly: true, memoizable: false },
          impl: async () => ({ value: {} }),
          signature: {
            input: { remote: { kind: "json", schema: { type: "object" }, optional: true }, draft: { kind: "json", schema: { type: "boolean" }, optional: true } },
            output: { name: "output", kind: "json", schema: {} },
          },
        },
      ],
    ]);
    const bundle = load(
      { root: { environment: { functions: { record: { args: { remtoe: {}, draft: "yes" } } } }, children: { gate: {} } }, "root/gate": gate({}) },
      "root",
    );
    const errors = validateBundle(bundle, { functions: functions as never }).errors.filter((e) => e.stateId === "root");
    expect(errors.map((e) => `${e.path}: ${e.message}`)).toEqual([
      expect.stringMatching(/^environment\.functions\.record\.args\.remtoe: 'record' has no parameter 'remtoe' — it takes remote, draft/),
      expect.stringMatching(/^environment\.functions\.record\.args\.draft: default for 'record' parameter 'draft' is not the type it takes/),
    ]);
  });
});

describe("the `$`-key renames (§2)", () => {
  const refused = (states: Record<string, unknown>): string => {
    try {
      const errors = errorsOf(states, "root");
      return errors.join("\n");
    } catch (e) {
      return (e as Error).message;
    }
  };

  it("refuses `expr`, naming `$expr`", () => {
    expect(refused({ root: { outputs: { n: { schema: {}, binding: { expr: "1 + 1" } } } } })).toMatch(/'expr' is spelled '\$expr'/);
    expect(refused({ root: leaf({ model: { expr: ".inputs.m" } }) })).toMatch(/operation\.model: 'expr' is spelled '\$expr'/);
  });

  it("refuses the `{ binding }` wrapper in an argument bag, naming `$binding`", () => {
    expect(refused({ root: { inputs: { t: { schema: {} } }, outputs: {}, operation: { function: "record", args: { text: { binding: ".inputs.t" } } } } })).toMatch(
      /operation\.args\.text: the '\{ "binding": … \}' wrapper is spelled '\{ "\$binding": … \}'/,
    );
  });

  it("refuses the old session keys, each naming its replacement", () => {
    for (const [old, now, decl] of [
      ["name", "$ref", { name: "review" }],
      ["in", "$in", { $ref: "review", in: "parent" }],
      ["join", "$join", { join: "nearest" }],
      ["fork", "$fork", { $ref: "review", fork: true }],
      ["expr", "$expr", { expr: ".inputs.thread" }],
    ] as const) {
      expect(refused({ root: leaf({ session: decl }) })).toContain(`session declares '${old}'; the key is spelled '${now}'`);
    }
  });

  it("leaves a slot's own `binding` key bare — it is structure, not payload", () => {
    expect(errorsOf({ root: { inputs: { t: { schema: {} } }, outputs: { o: { schema: {}, binding: ".inputs.t" } } } }, "root")).toEqual([]);
  });
});

describe("a name is usable only where a VALUE is read (§5, §6)", () => {
  it("passes a declared name over in a STRUCTURAL position — the string means the file it always did", () => {
    const bundle = load(
      { root: { environment: { names: { shared: {} } }, outputs: {}, operation: { function: "record", input: "shared" } } },
      "root",
      { [`${WF}/shared.json`]: { text: { schema: { type: "string" }, default: "hi" } } },
    );
    expect(Object.keys(bundle.states.root!.operation!.input)).toEqual(["text"]);
  });

  it("says so when there is no such file either, instead of loading `$ref` and `$in` as slots", () => {
    expect(() => load({ root: { environment: { names: { shared: {} } }, outputs: {}, operation: { function: "record", input: "shared" } } }, "root")).toThrow(
      /operation\.input: 'shared' is a declared name, but this position holds structure rather than a value/,
    );
  });

  it("still reads one in a value position that holds an object — `permissions`", () => {
    const bundle = load({ root: { environment: { names: { strict: { profile: "read-only" } } }, outputs: {}, operation: { prompt: "go", model: "m", permissions: "strict" } } }, "root");
    expect(validateBundle(bundle).errors).toEqual([]);
    expect(bundle.states.root!.fields?.map((f) => f.path)).toContain("environment.permissions");
  });
});

describe("an argument is the function's own data, whatever it is called", () => {
  it("reads a name in an argument called `workspace`, `session`, `schema` or `names`", async () => {
    for (const arg of ["workspace", "session", "schema", "names"]) {
      const seen = await argsSeen({ root: { environment: { names: { impl: { from: "main" } } }, outputs: {}, operation: { function: "record", args: { [arg]: { $ref: "impl" } } } } }, "root");
      expect(seen).toEqual([{ [arg]: { from: "main", $key: "impl#/" } }]);
    }
  });

  it("leaves a SLOT called `args` a slot — its schema's `$ref` is JSON Schema's", () => {
    const bundle = load({ root: { inputs: { args: { schema: { $defs: { a: { type: "string" } }, $ref: "#/$defs/a" }, optional: true } }, outputs: {} } }, "root");
    expect(bundle.states.root!.nameBinds).toBeUndefined();
    expect(bundle.states.root!.sessionError).toBeUndefined();
  });

  it("replaces a name NESTED in a default with the state's own literal, whole — nothing is dropped", async () => {
    const seen = await argsSeen(
      {
        root: { environment: { names: { review: { to: "origin" } }, functions: { record: { args: { opts: { keep: 1, remote: { $ref: "review" } } } } } }, children: { gate: {} } },
        "root/gate": { outputs: {}, operation: { function: "record", args: { opts: { remote: { draft: true } } } } },
      },
      "root",
    );
    expect(seen).toEqual([{ opts: { keep: 1, remote: { draft: true } } }]);
  });
});

describe("one identity, one configuration — compared by VALUE", () => {
  it("does not mistake a different key order for a disagreement", () => {
    expect(
      errorsOf(
        {
          root: { environment: { names: { r: { opts: [{ a: 1, b: 2 }], at: { x: 1, y: 2 } } } }, children: { c: {} } },
          "root/c": { environment: { names: { r: { $in: "parent", opts: [{ b: 2, a: 1 }], at: { y: 2, x: 1 } } } }, outputs: {} },
        },
        "root",
      ),
    ).toEqual([]);
  });
});

describe("the plain keys beside a session's `$ref` are overrides of the NAME (§2, §4)", () => {
  const tree = (a: unknown, b: unknown) => ({
    root: { environment: { names: { draft: { retention: "run" } } }, children: { a: {}, b: {} } },
    "root/a": leaf({ session: a }),
    "root/b": leaf({ session: b }),
  });

  it("configures the identity, and stays off the declaration that travels", () => {
    const bundle = load(tree({ $ref: "draft", title: "Draft" }, "draft"), "root");
    expect(bundle.states.root!.names).toEqual({ draft: { retention: "run", title: "Draft" } });
    expect(bundle.states["root/a"]!.environment!.session).toEqual({ $ref: "draft", $in: "root" });
  });

  it("is a lint error when two uses disagree, like any other writer of the name", () => {
    expect(errorsOf(tree({ $ref: "draft", title: "A" }, { $ref: "draft", title: "B" }), "root")).toEqual([expect.stringMatching(/'draft' is given 'title': "B" here and "A"/)]);
  });

  it("refuses a `$`-key a session does not take", () => {
    expect(errorsOf(tree({ $ref: "draft", $keep: true }, "draft"), "root")).toEqual([expect.stringMatching(/'\$keep' means nothing on a session/)]);
  });

  it("is TYPE CHECKED against what the host says the position takes — at the bind point", () => {
    const positions = {
      session: { type: "object", properties: { retention: { type: "string" }, title: { type: "string" } }, additionalProperties: false },
      workspace: { type: "object", properties: { from: { type: "string" } }, required: ["from"] },
    };
    const check = (states: Record<string, unknown>) => validateBundle(load(states, "root"), { positions: positions as never }).errors.map((e) => `${e.stateId} ${e.path}: ${e.message}`);
    expect(check(tree({ $ref: "draft", title: "Draft" }, "draft"))).toEqual([]);
    expect(check(tree({ $ref: "draft", title: 7 }, "draft"))).toEqual([expect.stringMatching(/^root\/a operation\.session: 'draft' is configured as .*"title":7.* not what a session takes/)]);
    // The same check, for the other providing position — an override, and a name with no entry at all.
    expect(check({ root: { outputs: {}, operation: { function: "record", workspace: { $ref: "impl", from: 3 } } } })).toEqual([expect.stringMatching(/^root operation\.workspace: 'impl' is configured as \{"from":3\}/)]);
    expect(check({ root: { outputs: {}, operation: { function: "record", workspace: "impl" } } })).toEqual([expect.stringMatching(/'impl' is configured as \{\}, which is not what a workspace takes/)]);
  });
});

describe("a name in a value position is typed where it is read (§5)", () => {
  const functions = new Map([
    [
      "record",
      {
        kind: "host",
        capabilities: { interactive: false, readOnly: true, memoizable: false },
        impl: async () => ({ value: {} }),
        signature: {
          input: { remote: { kind: "json", schema: { type: "object", properties: { to: { type: "string" } }, required: ["to"] }, optional: true } },
          output: { name: "output", kind: "json", schema: {} },
        },
      },
    ],
  ]);
  const gate = (remote: unknown) => ({ root: { environment: { names: { review: {} } }, outputs: {}, operation: { function: "record", args: { remote } } } });
  const errors = (states: Record<string, unknown>) => validateBundle(load(states, "root"), { functions: functions as never }).errors.map((e) => `${e.path}: ${e.message}`);

  it("accepts a name whose configuration — overrides included — is what the parameter takes", () => {
    expect(errors(gate({ $ref: "review", to: "origin" }))).toEqual([]);
  });

  it("refuses one that is not, at the bind point", () => {
    expect(errors(gate({ $ref: "review", to: 5 }))).toEqual([expect.stringMatching(/^operation\.input\.remote/)]);
    expect(errors(gate({ $ref: "review" }))).toEqual([expect.stringMatching(/^operation\.input\.remote/)]);
  });
});

describe("a mount layer's function defaults are this state's writing too (§7)", () => {
  it("checks them against the function, at the line that wrote them", () => {
    const functions = new Map([
      [
        "record",
        {
          kind: "host",
          capabilities: { interactive: false, readOnly: true, memoizable: false },
          impl: async () => ({ value: {} }),
          signature: { input: { draft: { kind: "json", schema: { type: "boolean" }, optional: true } }, output: { name: "output", kind: "json", schema: {} } },
        },
      ],
    ]);
    const bundle = load(
      { root: { children: { gate: { environment: { functions: { record: { args: { draft: "yes" } } } } } } }, "root/gate": { outputs: {}, operation: { function: "record" } } },
      "root",
    );
    expect(
      validateBundle(bundle, { functions: functions as never })
        .errors.filter((e) => e.stateId === "root")
        .map((e) => `${e.path}: ${e.message}`),
    ).toEqual([expect.stringMatching(/^children\.gate\.environment\.functions\.record\.args\.draft: default for 'record' parameter 'draft' is not the type it takes/)]);
  });
});

describe("what is USABLE among alternatives (§6)", () => {
  const states = { root: { outputs: {}, operation: { prompt: "go", model: { $any: [{ $ref: "$/roles.plan" }, { model: "fallback" }] } } } };
  const chosen = (roles: unknown) =>
    load(states, "root", roles === undefined ? {} : { [`${ROOT}/roles.json`]: roles }).states.root!.fields?.find((f) => f.path === "operation.config.model")?.ref;

  it("skips an alternative whose OWN reference finds nothing — no file, or no such property", () => {
    expect(chosen(undefined)).toEqual({ json: { model: "fallback" } });
    expect(chosen({ review: { model: "r" } })).toEqual({ json: { model: "fallback" } });
    expect(chosen({ plan: { model: "planner" } })).toEqual({ json: { model: "planner" } });
  });

  it("does NOT skip one that is there and broken inside — a typo is a mistake, not an absence", () => {
    expect(() => chosen({ plan: { model: "planner", reasoning: { $ref: "$/typo-missing.r" } } })).toThrow(/typo-missing/);
  });
});
