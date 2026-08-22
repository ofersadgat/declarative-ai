/**
 * A PREFIX MATCH IS NOT A MATCH — module symbol resolution (SPEC §7.5.2).
 *
 * A document contributes exactly one symbol, its own path, so a document is found by matching the
 * reference against the directory LISTING. A module contributes a set, and for an object export the
 * filename contributes nothing — so `text.slug` may live in `strings.ts`, in `lib.ts`, or beside
 * forty other helpers, in a file whose name shares not one character with the reference.
 *
 * That is why modules resolve through an INDEX rather than by name-matching. A resolver built on
 * prefix-matching answers the easy case (`text.slug` in `text.ts`) and is structurally unable to
 * answer the case the feature exists for.
 *
 * What the feature IS, one test each:
 *
 *  - a symbol resolves to the module that CONTRIBUTES it, whatever that module is called;
 *  - a file matching the reference's first segment does not capture it by being searched first;
 *  - a document always wins, so nothing that predates modules can be captured by one;
 *  - only a whole-symbol match at two entries is shadowing — a near miss is silent;
 *  - near misses are remembered, and the failure says what was found instead;
 *  - with no index supplied, resolution is exactly what it was.
 */
import { describe, expect, it } from "vitest";
import { resolveReference, SymbolMissError, type SymbolIndex, type Vfs } from "../src/reference.js";

const A = "/layers/a";
const B = "/layers/b";

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

/**
 * An index over a declared `file → { fully-qualified symbol → property path inside the file }`
 * table — standing in for what a parsed module's exports will answer once they are read.
 */
function indexOf(table: Record<string, Record<string, readonly string[]>>): SymbolIndex {
  const inDir = (file: string, dir: string): boolean =>
    file.startsWith(`${dir}/`) && !file.slice(dir.length + 1).includes("/");
  return (dir, symbol) => {
    const wanted = symbol.join(".");
    for (const [file, symbols] of Object.entries(table)) {
      if (!inDir(file, dir)) continue;
      const property = symbols[wanted];
      if (property !== undefined) return { found: true, file, property };
    }
    // Everything this directory offers under the same head — the diagnosis, not the answer.
    const head = symbol[0];
    const near = Object.entries(table)
      .filter(([file]) => inDir(file, dir))
      .map(([file, symbols]) => ({
        file,
        symbols: Object.keys(symbols).filter((s) => s === head || s.startsWith(`${head}.`)),
      }))
      .filter((p) => p.symbols.length > 0);
    return near.length > 0 ? { found: false, near } : { found: false };
  };
}

describe("a symbol resolves to whatever module contributes it", () => {
  // The motivating case, exactly: two files whose default exports share a top-level key, and the
  // one searched FIRST does not hold the symbol being asked for.
  const files = {
    [`${A}/mynamespace.ts`]: "export default { mynamespace: { foo: 5 } }",
    [`${A}/mynamespace2.ts`]: "export default { mynamespace: { myfunction: () => {} } }",
  };
  const symbols = indexOf({
    [`${A}/mynamespace.ts`]: { "mynamespace.foo": ["mynamespace", "foo"] },
    [`${A}/mynamespace2.ts`]: { "mynamespace.myfunction": ["mynamespace", "myfunction"] },
  });
  const opts = { defaultRoot: [A], vfs: vfsOf(files), symbols };

  it("finds the symbol in a SIBLING file, though the reference names the other one", () => {
    expect(resolveReference("mynamespace.myfunction", opts)).toMatchObject({
      file: `${A}/mynamespace2.ts`,
      property: ["mynamespace", "myfunction"],
    });
  });

  it("still finds the symbol the first file does hold", () => {
    expect(resolveReference("mynamespace.foo", opts)).toMatchObject({
      file: `${A}/mynamespace.ts`,
      property: ["mynamespace", "foo"],
    });
  });

  it("finds a symbol in a file whose name shares nothing with the reference", () => {
    // The filename contributes nothing for an object export, so there is no reason `text.slug`
    // should live in a file called `text`.
    const lib = { [`${A}/lib.ts`]: "export default { text: { slug: () => {} } }" };
    const resolved = resolveReference("text.slug", {
      defaultRoot: [A],
      vfs: vfsOf(lib),
      symbols: indexOf({ [`${A}/lib.ts`]: { "text.slug": ["text", "slug"] } }),
    });
    expect(resolved).toMatchObject({ file: `${A}/lib.ts`, property: ["text", "slug"] });
  });

  it("names a default-export function by its filename, with no property inside", () => {
    const one = { [`${A}/confidence.ts`]: "export default function confidence() {}" };
    expect(
      resolveReference("confidence", {
        defaultRoot: [A],
        vfs: vfsOf(one),
        symbols: indexOf({ [`${A}/confidence.ts`]: { confidence: [] } }),
      }),
    ).toMatchObject({ file: `${A}/confidence.ts`, property: [] });
  });
});

describe("the search path", () => {
  const files = {
    [`${A}/one.ts`]: "x",
    [`${B}/two.ts`]: "x",
  };

  it("falls through to the next entry when this one contributes nothing", () => {
    const resolved = resolveReference("text.slug", {
      defaultRoot: [A, B],
      vfs: vfsOf(files),
      symbols: indexOf({
        [`${A}/one.ts`]: { "text.trim": ["text", "trim"] },
        [`${B}/two.ts`]: { "text.slug": ["text", "slug"] },
      }),
    });
    expect(resolved).toMatchObject({ file: `${B}/two.ts`, property: ["text", "slug"] });
  });

  it("takes the earlier entry when both contribute the symbol", () => {
    const resolved = resolveReference("text.slug", {
      defaultRoot: [A, B],
      vfs: vfsOf(files),
      symbols: indexOf({
        [`${A}/one.ts`]: { "text.slug": ["text", "slug"] },
        [`${B}/two.ts`]: { "text.slug": ["text", "slug"] },
      }),
    });
    expect(resolved).toMatchObject({ file: `${A}/one.ts` });
  });
});

describe("a document always wins", () => {
  it("is consulted before the index, so nothing existing can be captured by a module", () => {
    const files = { [`${A}/plan.json`]: "{}", [`${A}/lib.ts`]: "x" };
    const resolved = resolveReference("plan", {
      defaultRoot: [A],
      vfs: vfsOf(files),
      // The index would happily answer `plan`; it is never asked, because the document matched.
      symbols: indexOf({ [`${A}/lib.ts`]: { plan: ["plan"] } }),
    });
    expect(resolved).toMatchObject({ file: `${A}/plan.json`, property: [] });
  });

  it("leaves a document's own property reference alone", () => {
    // The document HOLDS `address`, so it answers outright and the index is never consulted. One
    // that lacked it would fall through instead — a prefix match is not a match for documents either.
    const files = { [`${A}/user.json`]: '{"address": {"city": "x"}}' };
    let asked = false;
    const symbols: SymbolIndex = () => {
      asked = true;
      return { found: false };
    };
    expect(resolveReference("user.address", { defaultRoot: [A], vfs: vfsOf(files), symbols })).toMatchObject({
      file: `${A}/user.json`,
      property: ["address"],
    });
    expect(asked).toBe(false);
  });
});

describe("shadowing", () => {
  const files = { [`${A}/one.ts`]: "x", [`${B}/two.ts`]: "x" };

  it("is SILENT when the earlier entry merely lacks the symbol", () => {
    const warnings: string[] = [];
    resolveReference("text.slug", {
      defaultRoot: [A, B],
      vfs: vfsOf(files),
      onWarn: (m) => warnings.push(m),
      symbols: indexOf({
        [`${A}/one.ts`]: { "text.trim": ["text", "trim"] },
        [`${B}/two.ts`]: { "text.slug": ["text", "slug"] },
      }),
    });
    // A file that does not contribute the symbol is not competing for it.
    expect(warnings).toEqual([]);
  });

  it("WARNS when both entries contribute the whole symbol", () => {
    const warnings: string[] = [];
    resolveReference("text.slug", {
      defaultRoot: [A, B],
      vfs: vfsOf(files),
      onWarn: (m) => warnings.push(m),
      symbols: indexOf({
        [`${A}/one.ts`]: { "text.slug": ["text", "slug"] },
        [`${B}/two.ts`]: { "text.slug": ["text", "slug"] },
      }),
    });
    expect(warnings.join("\n")).toContain("also matches further along the path");
  });
});

describe("the failure", () => {
  const files = { [`${A}/one.ts`]: "x", [`${B}/two.ts`]: "x" };
  const opts = {
    defaultRoot: [A, B],
    vfs: vfsOf(files),
    symbols: indexOf({
      [`${A}/one.ts`]: { "text.trim": ["text", "trim"], "text.format": ["text", "format"] },
      [`${B}/two.ts`]: { "other.thing": ["other", "thing"] },
    }),
  };

  it("names what WAS found when nothing contributes the symbol", () => {
    let error: unknown;
    try {
      resolveReference("text.slug", opts);
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(SymbolMissError);
    const message = (error as Error).message;
    expect(message).toContain("text.slug");
    expect(message).toContain(`${A}/one.ts`);
    expect(message).toContain("text.trim");
    expect(message).toContain("text.format");
    // The unrelated file had nothing under this head, so it is not in the diagnosis.
    expect(message).not.toContain("other.thing");
  });

  it("carries the near misses as data, not only in the message", () => {
    try {
      resolveReference("text.slug", opts);
      expect.unreachable("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(SymbolMissError);
      expect((e as SymbolMissError).partials).toMatchObject([
        { file: `${A}/one.ts`, symbols: ["text.trim", "text.format"] },
      ]);
    }
  });

  it("stays a plain miss when nothing was near", () => {
    // "No such symbol anywhere" and "the directory had something like it" are worth keeping apart.
    let error: unknown;
    try {
      resolveReference("absent.thing", opts);
    } catch (e) {
      error = e;
    }
    expect(error).not.toBeInstanceOf(SymbolMissError);
    expect((error as Error).message).toContain("matches no file on the path");
  });
});

describe("with no index supplied", () => {
  it("resolves exactly as it did before modules existed", () => {
    const files = { [`${A}/user.json`]: "{}" };
    expect(resolveReference("user.address", { defaultRoot: [A], vfs: vfsOf(files) })).toMatchObject({
      file: `${A}/user.json`,
      property: ["address"],
    });
    expect(() => resolveReference("text.slug", { defaultRoot: [A], vfs: vfsOf(files) })).toThrow(
      /matches no file/,
    );
  });
});
