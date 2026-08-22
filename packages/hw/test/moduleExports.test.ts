/**
 * What a module contributes to the search path (SPEC §7.5.2, the contribution table).
 *
 * Every case pins the SAME two coordinates: the fully-qualified symbol an expression writes, and the
 * property path the value sits at in the export namespace. They are different things — `text.slug`
 * from a default export lives at `default.text.slug` — and the pair is what the index hands back.
 *
 * The load-bearing row is the second: for an object export the FILENAME contributes nothing, which
 * is what lets a symbol live in whichever file its author put it in.
 */
import { describe, expect, it } from "vitest";
import { moduleSymbols, moduleStem } from "../src/moduleExports.js";

/** Symbol → property path, as a plain object, for readable assertions. */
async function symbols(path: string, source: string): Promise<Record<string, readonly string[]>> {
  return Object.fromEntries(await moduleSymbols(path, source));
}

describe("a default export that is a function", () => {
  it("is named by the FILENAME — the one case where the file's name is the symbol", async () => {
    expect(await symbols("/fn/confidence.ts", "export default function confidence() {}")).toEqual({
      confidence: ["default"],
    });
  });

  it("takes the filename even when the function is anonymous or an arrow", async () => {
    expect(await symbols("/fn/score.ts", "export default (x: number) => x * 2")).toEqual({ score: ["default"] });
    expect(await symbols("/fn/score.ts", "export default function () {}")).toEqual({ score: ["default"] });
  });

  it("ignores what the function was CALLED, since the reference names the file", async () => {
    // A default export is reached by the filename, so an internal name is decoration.
    expect(await symbols("/fn/score.ts", "export default function somethingElse() {}")).toEqual({
      score: ["default"],
    });
  });
});

describe("a default export that is an object", () => {
  it("contributes one symbol per key, and the filename contributes NOTHING", async () => {
    const table = await symbols("/fn/lib.ts", "export default { text: { slug: () => {} } }");
    expect(table).toEqual({
      text: ["default", "text"],
      "text.slug": ["default", "text", "slug"],
    });
    // The whole point: nothing here is called `lib`.
    expect(table).not.toHaveProperty("lib");
  });

  it("nests to any depth", async () => {
    const table = await symbols("/fn/lib.ts", "export default { a: { b: { c: { d: 1 } } } }");
    expect(table["a.b.c.d"]).toEqual(["default", "a", "b", "c", "d"]);
  });

  it("reads shorthand and quoted keys", async () => {
    const table = await symbols("/fn/lib.ts", "const slug = () => {}; export default { text: { slug, 'kebab-case': slug } }");
    expect(table["text.slug"]).toEqual(["default", "text", "slug"]);
    expect(table["text.kebab-case"]).toEqual(["default", "text", "kebab-case"]);
  });

  it("follows an identifier to the table it names", async () => {
    const source = "const helpers = { math: { clamp: () => {} } };\nexport default helpers;";
    expect(await symbols("/fn/anything.ts", source)).toEqual({
      math: ["default", "math"],
      "math.clamp": ["default", "math", "clamp"],
    });
  });

  it("skips a computed key rather than guessing at it", async () => {
    // Contributing a symbol that does not exist is worse than contributing none: the miss is silent.
    const table = await symbols("/fn/lib.ts", "const k = 'x'; export default { [k]: 1, known: 2 }");
    expect(table).toEqual({ known: ["default", "known"] });
  });
});

describe("named exports", () => {
  it("contribute their names, and their contents where they are objects", async () => {
    expect(await symbols("/fn/lib.ts", "export const text = { slug: () => {} }")).toEqual({
      text: ["text"],
      "text.slug": ["text", "slug"],
    });
  });

  it("contribute a plain exported function", async () => {
    expect(await symbols("/fn/lib.ts", "export function helper() {}")).toEqual({ helper: ["helper"] });
  });

  it("read an `export { … }` list, including a rename", async () => {
    const source = "const inner = { slug: () => {} };\nfunction plain() {}\nexport { inner as text, plain };";
    expect(await symbols("/fn/lib.ts", source)).toEqual({
      text: ["text"],
      "text.slug": ["text", "slug"],
      plain: ["plain"],
    });
  });

  it("sit alongside a default export in the same file", async () => {
    const source = "export default function () {}\nexport const math = { clamp: () => {} };";
    expect(await symbols("/fn/score.ts", source)).toEqual({
      score: ["default"],
      math: ["math"],
      "math.clamp": ["math", "clamp"],
    });
  });
});

describe("a module with no exports at all", () => {
  it("contributes its top-level declarations", async () => {
    const source = "function trim() {}\nconst slug = () => {};\nclass Thing {}";
    expect(await symbols("/fn/lib.ts", source)).toEqual({
      trim: ["trim"],
      slug: ["slug"],
      Thing: ["Thing"],
    });
  });

  it("still nests a top-level object", async () => {
    expect(await symbols("/fn/lib.ts", "const text = { slug: () => {} };")).toEqual({
      text: ["text"],
      "text.slug": ["text", "slug"],
    });
  });

  it("does NOT fall back once anything is exported", async () => {
    // An explicit export is a statement about what the file offers; the rest is implementation.
    const source = "function privateHelper() {}\nexport function published() {}";
    const table = await symbols("/fn/lib.ts", source);
    expect(table).toEqual({ published: ["published"] });
    expect(table).not.toHaveProperty("privateHelper");
  });
});

describe("dialects", () => {
  it("reads a .js module", async () => {
    expect(await symbols("/fn/helpers.js", "export const math = { clamp: () => {} }")).toEqual({
      math: ["math"],
      "math.clamp": ["math", "clamp"],
    });
  });

  it("reads TypeScript annotations without a type checker", async () => {
    const source = "export function score(rank: number, iteration = 3): { v: number } { return { v: rank + iteration }; }";
    expect(await symbols("/fn/lib.ts", source)).toEqual({ score: ["score"] });
  });
});

describe("moduleStem", () => {
  it("strips the directory and the module extension", () => {
    expect(moduleStem("/a/b/confidence.ts")).toBe("confidence");
    expect(moduleStem("confidence.js")).toBe("confidence");
    expect(moduleStem("/a/b/confidence.mts")).toBe("confidence");
  });

  it("leaves a dotted name that is not a module extension alone", () => {
    expect(moduleStem("/a/my.helpers.ts")).toBe("my.helpers");
  });
});
