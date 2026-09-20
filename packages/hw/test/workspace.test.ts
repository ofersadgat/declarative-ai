/**
 * `environment.workspace` — the resource bundle, as a position of its own (NAMES.md §10).
 *
 * `engine.test.ts` pins the split from the session: many sessions in one workspace, one session
 * across several. What is pinned here is that a workspace is a scoped name like any other — where it
 * is scoped, what the host is told, and that a loop and a fan-out need no rule of their own.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type JsonValue, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { StateDef } from "../src/format.js";
import { InMemoryPersistence, type WorkflowMetrics } from "../src/ports.js";
import { normalizeWorkspace, validateWorkspaceDecl } from "../src/workspace.js";
import { newRegistry, ok } from "./fakes.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/** Run a tree whose leaves call `work`, and return what the host was asked for, one entry per call. */
async function asked(files: Record<string, unknown>, inputs: Record<string, ResolvedValue> = {}): Promise<Array<{ key: string; declared?: { name: string; configuration: JsonValue } }>> {
  const seen: Array<{ key: string; declared?: { name: string; configuration: JsonValue } }> = [];
  const registry = newRegistry();
  registry.functions.set("work", hostFunction(async () => ok({}) as ExecResult<ResolvedValue, WorkflowMetrics>, HOST));
  const engine = new WorkflowEngine({
    bundle: loadBundle(files as Record<string, StateDef>, "root"),
    registry,
    validator: new SchemaValidator(),
    persistence: new InMemoryPersistence(),
    workspaceFor: (key, declared) => {
      seen.push({ key, ...(declared !== undefined ? { declared } : {}) });
      return undefined;
    },
  });
  const result = await engine.run({ inputs });
  expect(result.failure?.reason).toBeUndefined();
  return seen;
}

const leaf = (environment?: Record<string, unknown>): Record<string, unknown> => ({
  outputs: {},
  operation: { function: "work" },
  ...(environment !== undefined ? { environment } : {}),
});

describe("what may be declared", () => {
  it("accepts a name, a scoped name, a join and null", () => {
    for (const decl of ["impl", { $ref: "impl", $in: "parent" }, { $ref: "impl", from: "main" }, { $join: "nearest" }, null, undefined]) {
      expect(validateWorkspaceDecl(decl)).toBeUndefined();
    }
  });

  it("refuses a computed one, and says why a bundle cannot be", () => {
    expect(validateWorkspaceDecl({ $expr: ".inputs.ws" })).toMatch(/a resource bundle is fixed when its instance is created/);
    expect(validateWorkspaceDecl("")).toMatch(/write null for a fresh private workspace/);
    expect(validateWorkspaceDecl("#mine")).toMatch(/reserved/);
  });

  it("takes an ancestor's declaration with `$join`, whatever it is called", () => {
    const ancestry = [{ id: "root", source: "root", workspace: { $ref: "impl", $in: "root" } }, { id: "mid", source: "mid" }];
    expect(normalizeWorkspace({ $join: "nearest" }, { id: "leaf", source: "leaf" }, ancestry)).toEqual({ workspace: { $ref: "impl", $in: "root" } });
    expect(normalizeWorkspace({ $join: "parent" }, { id: "leaf", source: "leaf" }, ancestry)).toMatchObject({
      error: expect.stringMatching(/requires 'mid' to declare a workspace and it declares none/),
    });
  });
});

describe("the bundle an operation runs in", () => {
  it("is the run's own when nothing names one — and the host is told no name", async () => {
    expect(await asked({ root: leaf() })).toEqual([{ key: "default" }]);
  });

  it("is a FRESH private one for `null` — per pass inside a loop, and shared with the subtree below it", async () => {
    const seen = await asked({
      root: {
        environment: { workspace: "impl" },
        limits: { max_iterations: 2 },
        children: { pass: { transitions: [{ to: "pass", when: ".run.iteration < .limits.max_iterations" }] } },
      },
      "root/pass": { environment: { workspace: null }, children: { a: {}, b: {} } },
      "root/pass/a": leaf(),
      "root/pass/b": leaf(),
    });
    const keys = seen.map((s) => s.key);
    // Two leaves per pass share their pass's bundle; the next pass gets another; none is `impl`.
    expect(keys.slice(0, 2)).toEqual(["#pass", "#pass"]);
    expect(new Set(keys).size).toBe(keys.length / 2);
    expect(seen.every((s) => s.declared === undefined)).toBe(true);
  });

  it("is the run's bundle, explicitly, by a name scoped at the run", async () => {
    const seen = await asked({
      root: { children: { mid: {} } },
      "root/mid": { environment: { workspace: null }, children: { a: {} } },
      "root/mid/a": leaf({ workspace: { $ref: "main", $in: "global" } }),
    });
    expect(seen.map((s) => s.key)).toEqual(["main#/"]);
  });

  it("is the named workspace, handed to the host with what `names` configured it with", async () => {
    const seen = await asked({
      root: { environment: { names: { impl: { from: "main" } }, workspace: "impl" }, children: { a: {} } },
      "root/a": leaf(),
    });
    expect(seen).toEqual([{ key: "impl#/", declared: { name: "impl", configuration: { from: "main" } } }]);
  });

  it("takes plain keys beside the `$ref` as the name's configuration too", async () => {
    const seen = await asked({ root: leaf({ workspace: { $ref: "impl", from: "dev" } }) });
    expect(seen).toEqual([{ key: "impl#/", declared: { name: "impl", configuration: { from: "dev" } } }]);
  });

  it("scopes at the state that wrote a VISIBLE entry, so two children that write one word share a worktree", async () => {
    const seen = await asked({
      root: { environment: { names: { impl: {} } }, children: { a: {}, b: {} } },
      "root/a": leaf({ workspace: "impl" }),
      "root/b": leaf({ workspace: "impl" }),
    });
    expect(seen.map((s) => s.key)).toEqual(["impl#/", "impl#/"]);
  });

  it("…and at the writer of the USE when there is none, so they do not", async () => {
    const seen = await asked({ root: { children: { a: {}, b: {} } }, "root/a": leaf({ workspace: "impl" }), "root/b": leaf({ workspace: "impl" }) });
    expect(seen.map((s) => s.key)).toEqual(["impl#a", "impl#b"]);
  });

  it("is one worktree on every pass when named ABOVE a loop, and a fresh one per pass INSIDE it", async () => {
    const loop = (where: "above" | "inside") => ({
      root: {
        ...(where === "above" ? { environment: { workspace: "impl" } } : {}),
        limits: { max_iterations: 2 },
        children: { pass: { transitions: [{ to: "pass", when: ".run.iteration < .limits.max_iterations" }] } },
      },
      "root/pass": leaf(where === "inside" ? { workspace: "impl" } : undefined),
    });
    const above = (await asked(loop("above"))).map((s) => s.key);
    expect(above.length).toBeGreaterThan(1);
    expect(new Set(above)).toEqual(new Set(["impl#/"]));
    const inside = (await asked(loop("inside"))).map((s) => s.key);
    expect(new Set(inside).size).toBe(inside.length);
  });

  it("is one per ELEMENT of a fan-out — `key[i]`", async () => {
    const seen = await asked(
      {
        root: {
          inputs: { parts: { schema: { type: "array", items: { type: "string" } } } },
          children: { part: { inputs: { name: { $expr: ".inputs.parts", each: true } } } },
        },
        "root/part": { ...leaf({ workspace: "impl" }), inputs: { name: { schema: { type: "string" } } } },
      },
      { parts: ["x", "y"] },
    );
    expect(seen.map((s) => s.key).sort()).toEqual(["impl#part[0]", "impl#part[1]"]);
  });

  it("refuses one name bound as a workspace AND a session, at the bind that disagrees", () => {
    const bundle = loadBundle(
      { root: { environment: { names: { impl: {} }, workspace: "impl" }, children: { a: {} } }, "root/a": leaf({ session: "impl" }) } as Record<string, StateDef>,
      "root",
    );
    expect(validateBundle(bundle).errors.map((e) => `${e.stateId}: ${e.message}`)).toEqual([
      expect.stringMatching(/^root\/a: 'impl' is a workspace \(bound at root: environment\.workspace\) and cannot also be a session here/),
    ]);
  });
});

describe("ABSENT inherits the enclosing instance's bundle", () => {
  it("so a child under a state that names one for its OWN operation works where that operation does", async () => {
    // Unlike the rest of `operation`, which a child never inherits: a bundle belongs to an instance,
    // and a child that names none is working where its parent is.
    const seen = await asked({
      root: { outputs: {}, operation: { function: "work", workspace: "mine" }, children: { kid: {} } },
      "root/kid": leaf(),
    });
    expect(seen.map((s) => s.key)).toEqual(["mine#/", "mine#/"]);
  });

  it("and a child the chain DOES name one for resolves its own", async () => {
    const seen = await asked({
      root: { outputs: {}, environment: { workspace: "shared" }, operation: { function: "work", workspace: "mine" }, children: { kid: {} } },
      "root/kid": leaf(),
    });
    expect(seen.map((s) => s.key)).toEqual(["mine#/", "shared#/"]);
  });
});
