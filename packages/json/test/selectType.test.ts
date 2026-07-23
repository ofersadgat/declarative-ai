import { describe, expect, it } from "vitest";
import { resolveSelectOutputSchema } from "../src/index";

describe("resolveSelectOutputSchema — descent reads OWN properties (§6)", () => {
  const schema = { type: "object", properties: { a: { type: "string" }, xs: { type: "array", items: { type: "number" } } } };

  it("rejects a path segment naming an inherited property", () => {
    // `properties["__proto__"]` returned `Object.prototype`, which then passed for a schema document —
    // so a select that names nothing in the schema type-checked instead of failing.
    expect(() => resolveSelectOutputSchema(schema, "$.__proto__")).toThrow(/property '__proto__' not in schema/);
    expect(() => resolveSelectOutputSchema(schema, "$.constructor")).toThrow(/property 'constructor' not in schema/);
  });

  it("still resolves real property, index and wildcard segments", () => {
    expect(resolveSelectOutputSchema(schema, "$.a")).toEqual({ type: "string" });
    expect(resolveSelectOutputSchema(schema, "$.xs[0]")).toEqual({ type: "number" });
    expect(resolveSelectOutputSchema(schema, "$.xs[*]")).toEqual({ type: "array", items: { type: "number" } });
  });
});
