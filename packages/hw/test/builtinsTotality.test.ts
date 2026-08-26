/**
 * The built-ins are TOTAL — including for values the language itself cannot spell.
 *
 * `builtins.ts` promises "a value for every input, never throws", because a throw escapes a
 * synchronous resolver into the scheduling loop where there is no way to report it. That promise held
 * for everything an expression can write down, and not for what can arrive: a user `.ts` function's
 * return enters the dataflow unconverted whenever its schema needs no marshalling (`marshalOut`), so
 * a circular object, a structure deep enough to exhaust the stack, or a bigint all reach here.
 *
 * Tested against the implementations directly rather than through the language, precisely because
 * these inputs have no spelling in it — an expression-level test could not construct one.
 */
import { describe, expect, it } from "vitest";
import { BUILTINS } from "../src/builtins.js";

const call = (name: string, args: Record<string, unknown>): unknown => BUILTINS[name]!.fn(args);

const circular = (): Record<string, unknown> => {
  const o: Record<string, unknown> = { name: "loop" };
  o.self = o;
  return o;
};
const deep = (n: number): Record<string, unknown> => {
  const root: Record<string, unknown> = {};
  let o = root;
  for (let i = 0; i < n; i++) {
    const next: Record<string, unknown> = {};
    o.next = next;
    o = next;
  }
  return root;
};

describe("values the language cannot spell, arriving from a user function", () => {
  const hostile: [string, unknown][] = [
    ["a circular object", circular()],
    ["a 20,000-deep structure", deep(20_000)],
    ["a bigint", 10n],
    ["a function", (): void => {}],
    ["a symbol", Symbol("s")],
  ];

  for (const [label, value] of hostile) {
    it(`does not throw out of find/contains/to_json/unique for ${label}`, () => {
      expect(() => call("to_json", { value })).not.toThrow();
      expect(() => call("contains", { value: [1, 2], item: value })).not.toThrow();
      expect(() => call("find", { value: [{ total: 1 }], key: "total", match: value })).not.toThrow();
      expect(() => call("unique", { value: [value, value, 1] })).not.toThrow();
    });
  }

  it("reports a cycle rather than pretending the value is ordinary", () => {
    expect(String(call("to_json", { value: circular() }))).toContain("[circular]");
  });

  it("does not mistake a value REPEATED in two branches for a cycle", () => {
    const shared = { a: 1 };
    expect(call("to_json", { value: [shared, shared] })).toBe('[{"a":1},{"a":1}]');
  });
});

describe("serialisation still agrees with JSON.stringify wherever JSON can represent the value", () => {
  // The point of the replacement is robustness, NOT new semantics: `same` is defined in terms of this,
  // so any drift here would quietly move what `find`, `contains` and `unique` consider equal.
  const jsonValues: unknown[] = [
    null, undefined, 0, -1, 1.5, "", "hi", '"quoted"', true, false,
    [], [1, "two", null], {}, { a: 1, b: [2, { c: 3 }] },
    { nested: { deep: { enough: [1, 2, 3] } } },
    // JSON's own asymmetry: dropped from an object, nulled in an array.
    { kept: 1, dropped: undefined },
    [undefined, 1],
    // `JSON.stringify` writes these as null, and so must this.
    Number.NaN, Number.POSITIVE_INFINITY,
    new Date("2020-01-02T03:04:05.000Z"), // via toJSON
  ];

  for (const v of jsonValues) {
    it(`matches for ${JSON.stringify(v) ?? String(v)}`, () => {
      expect(call("to_json", { value: v })).toBe(JSON.stringify(v ?? null));
    });
  }
});

describe("unique", () => {
  it("keeps first occurrences, in order, by deep equality", () => {
    expect(call("unique", { value: [3, 1, 3, 2, 1] })).toEqual([3, 1, 2]);
    expect(call("unique", { value: [{ a: 1 }, { a: 1 }, { a: 2 }] })).toEqual([{ a: 1 }, { a: 2 }]);
  });

  /**
   * It used to be O(n²) with a full serialisation of BOTH sides per comparison: 8,000 distinct
   * elements took 6.5 seconds and 4,000 took one. Synchronous, so that is not a slow expression — it
   * is the process stopped, engine and UI together, by an array that is merely large.
   *
   * The bound is deliberately loose. It is not measuring speed; it is asserting the curve is no longer
   * quadratic, and the old implementation would need minutes to reach this size.
   */
  it("scales linearly enough that a large array cannot freeze the process", () => {
    const distinct = Array.from({ length: 20_000 }, (_, i) => ({ id: i }));
    const started = Date.now();
    expect((call("unique", { value: distinct }) as unknown[]).length).toBe(20_000);
    expect(Date.now() - started).toBeLessThan(2000);
  });
});
