/**
 * The module index, and symbol resolution running through it end to end (SPEC §7.5.2).
 *
 * `symbolResolution.test.ts` pins the resolver against a hand-written index. These pin the REAL one:
 * modules on disk, parsed, assembled per directory, and then resolved through `resolveReference`
 * exactly as the loader will do it.
 *
 * The motivating case is the first test, and it is the one a name-driven resolver cannot pass.
 */
import { describe, expect, it } from "vitest";
import { buildModuleIndex } from "../src/moduleIndex.js";
import { resolveReference, selectProperty, SymbolMissError, type Vfs } from "../src/reference.js";
import type { SymbolTable } from "../src/moduleExports.js";

const A = "/layers/a";
const B = "/layers/b";

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

describe("the motivating case", () => {
  const files = {
    [`${A}/mynamespace.ts`]: "export default { mynamespace: { foo: 5 } }",
    [`${A}/mynamespace2.ts`]: "export default { mynamespace: { myfunction: () => {} } }",
  };

  it("resolves a symbol into the sibling that holds it, though the other is searched first", async () => {
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([A], { vfs });
    const opts = { defaultRoot: [A], vfs, symbols };

    expect(resolveReference("mynamespace.myfunction", opts)).toMatchObject({
      file: `${A}/mynamespace2.ts`,
      property: ["default", "mynamespace", "myfunction"],
    });
    expect(resolveReference("mynamespace.foo", opts)).toMatchObject({
      file: `${A}/mynamespace.ts`,
      property: ["default", "mynamespace", "foo"],
    });
  });
});

describe("a library of helpers in one file, called by bare name", () => {
  // The shape SPEC §7.5.2 calls the commonest once a `functions/` directory has more than a few
  // things in it: one file, several exports, and every one of them a name an expression can call
  // with no file or path qualifier anywhere in the reference.
  const files = {
    [`${A}/lib.ts`]: [
      "export function clamp(n: number) { return Math.max(0, Math.min(1, n)); }",
      "export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;",
      "export function slug(s: string) { return s.toLowerCase(); }",
    ].join("\n"),
  };

  it("resolves every export by its own bare name", async () => {
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([A], { vfs });
    const opts = { defaultRoot: [A], vfs, symbols };

    // Note what is NOT in any of these references: the word `lib`.
    expect(resolveReference("clamp", opts)).toMatchObject({ file: `${A}/lib.ts`, property: ["clamp"] });
    expect(resolveReference("lerp", opts)).toMatchObject({ file: `${A}/lib.ts`, property: ["lerp"] });
    expect(resolveReference("slug", opts)).toMatchObject({ file: `${A}/lib.ts`, property: ["slug"] });
  });

  it("draws bare names from a file with no exports at all", async () => {
    const scratch = { [`${A}/helpers.ts`]: "function trim(s: string) { return s.trim(); }\nconst n = 1;" };
    const vfs = vfsOf(scratch);
    const symbols = await buildModuleIndex([A], { vfs });
    expect(resolveReference("trim", { defaultRoot: [A], vfs, symbols })).toMatchObject({
      file: `${A}/helpers.ts`,
      property: ["trim"],
    });
  });

  it("draws bare names from SEVERAL files in one directory", async () => {
    const spread = {
      [`${A}/math.ts`]: "export function clamp(n: number) { return n; }",
      [`${A}/text.ts`]: "export function slug(s: string) { return s; }",
    };
    const vfs = vfsOf(spread);
    const symbols = await buildModuleIndex([A], { vfs });
    const opts = { defaultRoot: [A], vfs, symbols };
    expect(resolveReference("clamp", opts)).toMatchObject({ file: `${A}/math.ts` });
    expect(resolveReference("slug", opts)).toMatchObject({ file: `${A}/text.ts` });
  });

  it("finds a bare name at a LATER path entry", async () => {
    const layered = {
      [`${A}/base.ts`]: "export function clamp(n: number) { return n; }",
      [`${B}/extra.ts`]: "export function slug(s: string) { return s; }",
    };
    const vfs = vfsOf(layered);
    const symbols = await buildModuleIndex([A, B], { vfs });
    expect(resolveReference("slug", { defaultRoot: [A, B], vfs, symbols })).toMatchObject({
      file: `${B}/extra.ts`,
      property: ["slug"],
    });
  });
});

describe("across the search path", () => {
  const files = {
    [`${A}/one.ts`]: "export default { text: { trim: () => {} } }",
    [`${B}/two.ts`]: "export default { text: { slug: () => {} } }",
  };

  it("falls through an entry that contributes nothing matching", async () => {
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([A, B], { vfs });
    expect(resolveReference("text.slug", { defaultRoot: [A, B], vfs, symbols })).toMatchObject({
      file: `${B}/two.ts`,
      property: ["default", "text", "slug"],
    });
  });

  it("takes the earlier entry when both contribute the symbol", async () => {
    const both = {
      [`${A}/one.ts`]: "export default { text: { slug: () => {} } }",
      [`${B}/two.ts`]: "export default { text: { slug: () => {} } }",
    };
    const vfs = vfsOf(both);
    const symbols = await buildModuleIndex([A, B], { vfs });
    expect(resolveReference("text.slug", { defaultRoot: [A, B], vfs, symbols })).toMatchObject({
      file: `${A}/one.ts`,
    });
  });

  it("names the near misses when nothing has it", async () => {
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([A, B], { vfs });
    let error: unknown;
    try {
      resolveReference("text.slugify", { defaultRoot: [A, B], vfs, symbols });
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(SymbolMissError);
    expect((error as Error).message).toContain("text.trim");
    expect((error as Error).message).toContain("text.slug");
  });
});

describe("approval gates what may contribute", () => {
  it("indexes nothing from an unapproved module", async () => {
    // An unapproved file must not even be DISCOVERABLE, or indexing is how a workflow learns it is
    // there and a dropped-in file could shadow a symbol before anyone agreed to run it.
    const files = {
      [`${A}/trusted.ts`]: "export default { text: { slug: () => {} } }",
      [`${A}/dropped.ts`]: "export default { text: { slug: () => {} } }",
    };
    const vfs = vfsOf(files);
    const symbols = await buildModuleIndex([A], { vfs, approved: (f) => f.endsWith("trusted.ts") });
    expect(resolveReference("text.slug", { defaultRoot: [A], vfs, symbols })).toMatchObject({
      file: `${A}/trusted.ts`,
    });
  });

  it("leaves the symbol unresolvable when its only contributor is unapproved", async () => {
    const vfs = vfsOf({ [`${A}/dropped.ts`]: "export default { text: { slug: () => {} } }" });
    const symbols = await buildModuleIndex([A], { vfs, approved: () => false });
    expect(() => resolveReference("text.slug", { defaultRoot: [A], vfs, symbols })).toThrow();
  });
});

describe("collisions inside one directory", () => {
  it("warns and takes the earlier FILENAME, since there is no path order to appeal to", async () => {
    const vfs = vfsOf({
      [`${A}/a_first.ts`]: "export default { text: { slug: () => {} } }",
      [`${A}/z_last.ts`]: "export default { text: { slug: () => {} } }",
    });
    const warnings: string[] = [];
    const symbols = await buildModuleIndex([A], { vfs, onWarn: (m) => warnings.push(m) });
    expect(resolveReference("text.slug", { defaultRoot: [A], vfs, symbols })).toMatchObject({
      file: `${A}/a_first.ts`,
    });
    expect(warnings.join("\n")).toContain("is contributed by both");
  });
});

describe("the cache", () => {
  it("parses one set of bytes once, however many files hold them", async () => {
    const source = "export default { text: { slug: () => {} } }";
    const vfs = vfsOf({ [`${A}/one.ts`]: source, [`${B}/copy.ts`]: source });
    const cache = new Map<string, SymbolTable>();
    await buildModuleIndex([A, B], { vfs, cache });
    // Same bytes in two places: one entry, keyed by content rather than by path.
    expect(cache.size).toBe(1);
  });

  it("is reused across builds", async () => {
    const vfs = vfsOf({ [`${A}/one.ts`]: "export default { text: { slug: () => {} } }" });
    const cache = new Map<string, SymbolTable>();
    await buildModuleIndex([A], { vfs, cache });
    const symbols = await buildModuleIndex([A], { vfs, cache });
    expect(cache.size).toBe(1);
    expect(resolveReference("text.slug", { defaultRoot: [A], vfs, symbols })).toMatchObject({
      file: `${A}/one.ts`,
    });
  });
});

describe("what the index does not touch", () => {
  it("ignores non-module files", async () => {
    const vfs = vfsOf({ [`${A}/plan.json`]: '{"text":{"slug":1}}', [`${A}/notes.md`]: "text" });
    const symbols = await buildModuleIndex([A], { vfs });
    // A document is found by the document split, not here — asking the index for one gets nothing.
    expect(symbols(A, ["text", "slug"])).toEqual({ found: false });
  });

  it("leaves a document to win its own name, but not its whole dotted subtree", async () => {
    const vfs = vfsOf({
      [`${A}/plan.json`]: "{}",
      [`${A}/lib.ts`]: "export default { plan: { inner: 1 } }",
    });
    const symbols = await buildModuleIndex([A], { vfs });
    const opts = { defaultRoot: [A], vfs, symbols };

    // The document answers for its own name outright.
    expect(resolveReference("plan", opts)).toMatchObject({ file: `${A}/plan.json`, property: [] });

    // But `plan.json` holds no `inner`, so it does not claim `plan.inner` merely by being named
    // first — a prefix match is not a match for documents either, and the module is reached.
    expect(resolveReference("plan.inner", opts)).toMatchObject({
      file: `${A}/lib.ts`,
      property: ["default", "plan", "inner"],
    });
  });

  it("still lets a document that HOLDS the property claim it", async () => {
    const vfs = vfsOf({
      [`${A}/plan.json`]: '{"inner": 1}',
      [`${A}/lib.ts`]: "export default { plan: { inner: 2 } }",
    });
    const symbols = await buildModuleIndex([A], { vfs });
    // A document that genuinely provides it wins, which is what keeps every reference that predates
    // modules meaning what it always did.
    expect(resolveReference("plan.inner", { defaultRoot: [A], vfs, symbols })).toMatchObject({
      file: `${A}/plan.json`,
      property: ["inner"],
    });
  });

  it("recovers the precise error when nothing provides the symbol", async () => {
    // The strict pass misses everywhere; the re-run is what turns "names no symbol on the path" back
    // into an error naming the file and the property it lacks, reported where the value is read.
    const vfs = vfsOf({ [`${A}/plan.json`]: "{}" });
    const symbols = await buildModuleIndex([A], { vfs });
    const resolved = resolveReference("plan.ghost", { defaultRoot: [A], vfs, symbols });
    expect(resolved).toMatchObject({ file: `${A}/plan.json`, property: ["ghost"] });
    expect(() => selectProperty({}, resolved.property, "plan.ghost")).toThrow(/has no 'ghost'/);
  });
});
