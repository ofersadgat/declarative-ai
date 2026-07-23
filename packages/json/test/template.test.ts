import { describe, expect, it } from "vitest";
import type { SchemaDocument } from "../src/index";
import { applyTemplate, collapseHoles, resolveTypes } from "../src/index";

/** `{"__proto__": …}` written as a literal would SET the prototype, so wire-shaped inputs are parsed —
 *  which is also exactly how they arrive in production (`JSON.parse` makes it an OWN property). */
const parse = (json: string): SchemaDocument => JSON.parse(json) as SchemaDocument;
const asRecord = (v: unknown): Record<string, unknown> => v as Record<string, unknown>;

describe("applyTemplate — bindings are looked up as OWN properties (§6.1)", () => {
  it("does not resolve a parameter name off Object.prototype", () => {
    // `bindings["constructor"]` returned the `Object` FUNCTION and spliced it into a schema document,
    // where it later dies in canonicalize or Ajv — and "application is total" quietly stopped holding.
    expect(() => applyTemplate({ type: "object", properties: { p: { $param: "constructor" } } }, {})).toThrow(
      /no binding for parameter 'constructor'/,
    );
    expect(() => applyTemplate({ p: { $param: "toString" } }, {})).toThrow(/no binding for parameter 'toString'/);
    expect(() => applyTemplate({ p: { $param: "__proto__" } }, {})).toThrow(/no binding for parameter '__proto__'/);
  });

  it("collapses those same names to `{}` under a partial binding, like any other unbound variable", () => {
    expect(resolveTypes({ p: { $param: "toString" } }, {})).toEqual({ p: {} });
    expect(resolveTypes({ p: { $param: "__proto__" } }, {})).toEqual({ p: {} });
  });

  it("still binds an OWN parameter, and leaves a hole-free document alone", () => {
    expect(applyTemplate({ p: { $param: "T" } }, { T: { type: "string" } })).toEqual({ p: { type: "string" } });
    expect(applyTemplate({ type: "number" }, {})).toEqual({ type: "number" });
  });
});

describe("the substitution walk preserves a `__proto__` KEY (§6.1)", () => {
  it("keeps a constraint on a `__proto__` property through application", () => {
    // Plain `out[k] = v` hit `Object.prototype`'s setter and dropped the key, so the field silently
    // became UNCONSTRAINED after instantiation.
    const applied = applyTemplate(parse('{"type":"object","properties":{"__proto__":{"$param":"T"}}}'), {
      T: { type: "string" },
    });
    const props = asRecord(applied.properties);
    expect(Object.keys(props)).toEqual(["__proto__"]);
    expect(props["__proto__"]).toEqual({ type: "string" });
    expect(JSON.stringify(applied)).toBe('{"type":"object","properties":{"__proto__":{"type":"string"}}}');
  });

  it("keeps it through collapseHoles too", () => {
    const collapsed = collapseHoles(parse('{"properties":{"__proto__":{"type":"string"}}}'));
    expect(Object.keys(asRecord(collapsed.properties))).toEqual(["__proto__"]);
  });
});
