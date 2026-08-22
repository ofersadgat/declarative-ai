/**
 * Transpiling and running a user module (SPEC §7.5.4, §7.5.6).
 *
 * The first slice that executes anything, so these pin behaviour rather than shape:
 *
 *  - the require path is the search path plus each entry's `node_modules`, in that order;
 *  - a specifier is a REFERENCE — `$JAIRA/…`, relative, absolute, bare — not Node's algorithm;
 *  - a bare specifier finds a plain file before any `node_modules`, so bare ≠ dependency;
 *  - the closure is walked from the SOURCE's imports, so a `require(` inside a string is not one;
 *  - an unapproved file is an error, never a quiet resolution elsewhere;
 *  - a cycle terminates with a partial namespace, as CommonJS does;
 *  - nothing executes until `execute` is called, which is what leaves room for an approval gate.
 */
import { describe, expect, it } from "vitest";
import { dirOf, ModuleLoadError, prepareModules, requirePathFor, resolveSpecifier } from "../src/moduleLoader.js";
import { resolveReference, selectProperty, type Vfs } from "../src/reference.js";
import { buildModuleIndex } from "../src/moduleIndex.js";
import { synthesizeBody } from "../src/functionBody.js";

const FN = "/p/.jaira/functions";
const OPS = "/p/ops";

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

const prepare = (files: Record<string, string>, entries: string[], extra = {}) =>
  prepareModules(entries, {
    vfs: vfsOf(files),
    requirePath: requirePathFor([FN, OPS]),
    roots: { JAIRA: "/p/.jaira" },
    ...extra,
  });

describe("the require path", () => {
  it("is each search entry followed by that entry's node_modules", () => {
    expect(requirePathFor([FN, OPS])).toEqual([FN, `${FN}/node_modules`, OPS, `${OPS}/node_modules`]);
  });

  it("tolerates a trailing slash and backslashes", () => {
    expect(requirePathFor(["C:\\p\\ops\\"])).toEqual(["C:/p/ops", "C:/p/ops/node_modules"]);
  });
});

describe("a specifier is a reference", () => {
  const files = {
    [`${FN}/helper.ts`]: "export const x = 1;",
    [`${FN}/node_modules/zod/index.js`]: "module.exports = { z: 1 };",
    [`/p/.jaira/lib/review.ts`]: "export const r = 1;",
    [`/opt/shared/thing.js`]: "module.exports = 1;",
    [`${FN}/nested/deep.ts`]: "export const d = 1;",
  };
  const options = { vfs: vfsOf(files), requirePath: requirePathFor([FN, OPS]), roots: { JAIRA: "/p/.jaira" } };

  it("resolves a root variable, the same $JAIRA a state file uses", () => {
    expect(resolveSpecifier("$JAIRA/lib/review", FN, options)).toBe("/p/.jaira/lib/review.ts");
  });

  it("resolves a relative specifier against the requiring file's directory", () => {
    expect(resolveSpecifier("./helper", FN, options)).toBe(`${FN}/helper.ts`);
    expect(resolveSpecifier("../helper", `${FN}/nested`, options)).toBe(`${FN}/helper.ts`);
  });

  it("resolves a .js specifier to the .ts source, as TypeScript's own ESM rules require", () => {
    // A `.ts` module importing `./helper.js` is the correct spelling — the import names the EMITTED
    // file. Without this a module's imports would be unwritable in the dialect it is written in.
    expect(resolveSpecifier("./helper.js", FN, options)).toBe(`${FN}/helper.ts`);
  });

  it("prefers the exact spelling when both a .js and a .ts sit there", () => {
    const both = { ...files, [`${FN}/helper.js`]: "module.exports = {};" };
    expect(resolveSpecifier("./helper.js", FN, { ...options, vfs: vfsOf(both) })).toBe(`${FN}/helper.js`);
  });

  it("resolves an absolute specifier", () => {
    expect(resolveSpecifier("/opt/shared/thing", FN, options)).toBe("/opt/shared/thing.js");
  });

  it("resolves a bare specifier along the require path", () => {
    expect(resolveSpecifier("zod", FN, options)).toBe(`${FN}/node_modules/zod/index.js`);
  });

  it("finds a plain file BEFORE any node_modules — bare is not the same as a dependency", () => {
    // The entry itself precedes its own `node_modules`, so a local `helper.ts` wins over a package.
    const shadowed = {
      ...files,
      [`${FN}/node_modules/helper/index.js`]: "module.exports = 'the package';",
    };
    const opts = { ...options, vfs: vfsOf(shadowed) };
    expect(resolveSpecifier("helper", FN, opts)).toBe(`${FN}/helper.ts`);
  });

  it("returns undefined when nothing answers", () => {
    expect(resolveSpecifier("nope", FN, options)).toBeUndefined();
  });

  it("refuses an unknown root rather than searching for it", () => {
    expect(() => resolveSpecifier("$NOPE/x", FN, options)).toThrow(/unknown root/);
  });
});

describe("running a module", () => {
  it("executes a default export and returns its namespace", async () => {
    const files = { [`${FN}/score.ts`]: "export default function (n: number) { return n * 2; }" };
    const prepared = await prepare(files, [`${FN}/score.ts`]);
    const namespace = prepared.execute(`${FN}/score.ts`);
    expect((namespace.default as (n: number) => number)(21)).toBe(42);
  });

  it("follows a relative import and runs the dependency", async () => {
    const files = {
      [`${FN}/main.ts`]: "import { double } from './helper.js';\nexport default (n: number) => double(n) + 1;",
      [`${FN}/helper.ts`]: "export function double(n: number) { return n * 2; }",
    };
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    expect((prepared.execute(`${FN}/main.ts`).default as (n: number) => number)(10)).toBe(21);
  });

  it("follows a $ROOT import", async () => {
    const files = {
      [`${FN}/main.ts`]: "import { r } from '$JAIRA/lib/review.js';\nexport default () => r;",
      ["/p/.jaira/lib/review.ts"]: "export const r = 'reviewed';",
    };
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    expect((prepared.execute(`${FN}/main.ts`).default as () => string)()).toBe("reviewed");
  });

  it("honours a CommonJS module that replaces module.exports", async () => {
    const files = { [`${FN}/legacy.js`]: "module.exports = function () { return 'cjs'; };" };
    const prepared = await prepare(files, [`${FN}/legacy.js`]);
    const produced = prepared.execute(`${FN}/legacy.js`);
    expect((produced as unknown as () => string)()).toBe("cjs");
  });

  it("runs a module once however many times it is required", async () => {
    const files = {
      [`${FN}/main.ts`]: "import { n } from './counter.js';\nimport { n as m } from './counter.js';\nexport default () => n + m;",
      [`${FN}/counter.ts`]: "globalThis.__loads = (globalThis.__loads ?? 0) + 1;\nexport const n = 1;",
    };
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    (globalThis as Record<string, unknown>).__loads = 0;
    prepared.execute(`${FN}/main.ts`);
    expect((globalThis as Record<string, unknown>).__loads).toBe(1);
  });

  it("terminates an import cycle with a partial namespace, as CommonJS does", async () => {
    const files = {
      [`${FN}/a.ts`]: "import './b.js';\nexport const a = 1;",
      [`${FN}/b.ts`]: "import './a.js';\nexport const b = 2;",
    };
    const prepared = await prepare(files, [`${FN}/a.ts`]);
    expect(prepared.execute(`${FN}/a.ts`).a).toBe(1);
  });
});

describe("preparing is not running", () => {
  it("transpiles the whole closure without executing any of it", async () => {
    // The gap between the two is where an approval gate lives; a design that had to RUN a module to
    // discover its imports would have no moment at which to ask.
    const files = {
      [`${FN}/main.ts`]: "import './effect.js';\nexport default () => 1;",
      [`${FN}/effect.ts`]: "globalThis.__ran = true;\nexport const e = 1;",
    };
    (globalThis as Record<string, unknown>).__ran = false;
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    expect((globalThis as Record<string, unknown>).__ran).toBe(false);
    expect([...prepared.emitted.keys()].sort()).toEqual([`${FN}/effect.ts`, `${FN}/main.ts`]);

    prepared.execute(`${FN}/main.ts`);
    expect((globalThis as Record<string, unknown>).__ran).toBe(true);
  });
});

describe("the closure is read from the source, not the emit", () => {
  it("does not mistake a require( inside a string for an import", async () => {
    const files = { [`${FN}/main.ts`]: `export default () => "require('./ghost.js')";` };
    // Would throw "matches nothing on the require path" if a regex over the emit had been used.
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    expect([...prepared.emitted.keys()]).toEqual([`${FN}/main.ts`]);
  });

  it("does follow a real require() call", async () => {
    const files = {
      [`${FN}/main.js`]: "const h = require('./helper.js');\nmodule.exports = () => h.v;",
      [`${FN}/helper.js`]: "module.exports = { v: 7 };",
    };
    const prepared = await prepare(files, [`${FN}/main.js`]);
    expect((prepared.execute(`${FN}/main.js`) as unknown as () => number)()).toBe(7);
  });
});

describe("refusals", () => {
  it("names the specifier and the path when an import matches nothing", async () => {
    const files = { [`${FN}/main.ts`]: "import './missing.js';\nexport default () => 1;" };
    await expect(prepare(files, [`${FN}/main.ts`])).rejects.toThrow(/imports '\.\/missing\.js'.*require path/s);
  });

  it("refuses an unapproved file outright rather than resolving elsewhere", async () => {
    const files = {
      [`${FN}/main.ts`]: "import './helper.js';\nexport default () => 1;",
      [`${FN}/helper.ts`]: "export const x = 1;",
    };
    await expect(
      prepare(files, [`${FN}/main.ts`], { approved: (f: string) => f.endsWith("main.ts") }),
    ).rejects.toThrow(/'.*helper\.ts' is not approved to run/);
  });

  it("reports a compile error against the file that would not compile", async () => {
    const files = { [`${FN}/broken.ts`]: "export default function ( { return;" };
    await expect(prepare(files, [`${FN}/broken.ts`])).rejects.toThrow(ModuleLoadError);
    await expect(prepare(files, [`${FN}/broken.ts`])).rejects.toThrow(/broken\.ts' does not compile/);
  });

  it("refuses a host builtin rather than pretending to sandbox it", async () => {
    // SPEC §7.5.4 is explicit that the require path is resolution and not containment. This loader
    // supplies no builtins, and says so, rather than silently resolving `fs` to nothing.
    const files = { [`${FN}/main.ts`]: "import { readFileSync } from 'node:fs';\nexport default () => readFileSync;" };
    const prepared = await prepare(files, [`${FN}/main.ts`]);
    expect(() => prepared.execute(`${FN}/main.ts`)).toThrow(/host module 'node:fs'/);
  });
});

describe("the transpile cache", () => {
  it("keys on content, so unchanged bytes are compiled once", async () => {
    const files = { [`${FN}/score.ts`]: "export default () => 1;" };
    const cache = new Map<string, string>();
    await prepare(files, [`${FN}/score.ts`], { cache });
    expect(cache.size).toBe(1);
    const before = [...cache.values()][0];
    await prepare(files, [`${FN}/score.ts`], { cache });
    expect(cache.size).toBe(1);
    expect([...cache.values()][0]).toBe(before);
  });
});

describe("sources supplied directly", () => {
  it("loads a synthesized body that has no file behind it", async () => {
    // How an embedded body (SPEC §7.5.1) enters: same pipeline, same artifact, no file on disk.
    const synthesized = "export default function (severity) {\n  return Math.max(0, 1 - 0.35 * severity);\n}\n";
    const prepared = await prepare({}, ["<embedded>/confidence.ts"], {
      sources: { "<embedded>/confidence.ts": synthesized },
    });
    const fn = prepared.execute("<embedded>/confidence.ts").default as (s: number) => number;
    expect(fn(2)).toBeCloseTo(0.3);
  });
});

describe("a bare symbol, resolved and then CALLED", () => {
  it("addresses the right value in the loaded module", async () => {
    // Resolution is only half of it: the property path `moduleIndex` recorded has to address the
    // right value in the loaded namespace, or a reference resolves correctly to something nobody can
    // invoke. One file, several exports, called by bare name — SPEC §7.5.2's commonest shape.
    const files = {
      [`${FN}/lib.ts`]: [
        "export function clamp(n: number) { return Math.max(0, Math.min(1, n)); }",
        "export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;",
      ].join("\n"),
    };
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([FN], { vfs });
    const resolved = resolveReference("lerp", { defaultRoot: [FN], vfs, symbols });
    expect(resolved.file).toBe(`${FN}/lib.ts`);

    const prepared = await prepare(files, [resolved.file!]);
    const fn = selectProperty(prepared.execute(resolved.file!), resolved.property, "lerp") as (
      a: number,
      b: number,
      t: number,
    ) => number;
    expect(fn(0, 10, 0.5)).toBe(5);
  });
});

describe("an embedded body, end to end", () => {
  // The proof that the two authoring forms converge: a body written in a JSON document is
  // synthesized, prepared and executed by exactly the machinery a file on disk goes through.
  const run = async (body: string, slots: string[], args: unknown[]): Promise<unknown> => {
    const { source } = await synthesizeBody("f", body, Object.fromEntries(slots.map((s) => [s, {}])));
    const path = "<embedded>/f.ts";
    const prepared = await prepare({}, [path], { sources: { [path]: source } });
    return (prepared.execute(path).default as (...a: unknown[]) => unknown)(...args);
  };

  it("runs an expression body", async () => {
    expect(await run("Math.max(0, 1 - 0.35 * severity)", ["severity"], [2])).toBeCloseTo(0.3);
  });

  it("runs a statement body", async () => {
    const body = "const scaled = severity * 0.35;\nreturn Math.max(0, 1 - scaled);";
    expect(await run(body, ["severity"], [2])).toBeCloseTo(0.3);
  });

  it("runs a parenthesised record body, returning every output at once", async () => {
    const result = await run("({ score: 1 - severity / 3, high: severity >= 2 })", ["severity"], [2]);
    expect(result).toEqual({ score: expect.closeTo(0.333, 2) as unknown, high: true });
  });

  it("binds parameters in the declared order", async () => {
    expect(await run("a - b", ["a", "b"], [10, 4])).toBe(6);
  });

  it("carries TypeScript through the compiler", async () => {
    const body = "const n: number = severity;\nreturn n * 2;";
    expect(await run(body, ["severity"], [21])).toBe(42);
  });
});

describe("dirOf", () => {
  it("takes the directory of an absolute path", () => {
    expect(dirOf("/a/b/c.ts")).toBe("/a/b");
    expect(dirOf("C:/a/b.ts")).toBe("C:/a");
  });
});
