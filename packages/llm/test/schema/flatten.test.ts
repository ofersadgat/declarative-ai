import { describe, expect, it } from "vitest";
import { flattenForDepth } from "../../src/schema/flatten";
import type { SchemaNode } from "../../src/schema/profile";

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- test ergonomics on dynamic schemas.
type Any = any;

/** Max object/array nesting depth of a schema (root = 1), the metric the strict cap is checked against. */
function depthOf(node: SchemaNode, d = 1): number {
  let max = d;
  const props = (node as Any).properties;
  if (props && typeof props === "object") for (const k of Object.keys(props)) max = Math.max(max, depthOf(props[k], d + 1));
  const items = (node as Any).items;
  if (items && typeof items === "object") max = Math.max(max, depthOf(items, d + 1));
  return max;
}

describe("flattenForDepth — lossless object key-flattening", () => {
  it("hoists a required object chain into dotted keys and restores it exactly", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: {
        a: {
          type: "object",
          properties: { b: { type: "object", properties: { c: { type: "string" } }, required: ["c"] } },
          required: ["b"],
        },
      },
      required: ["a"],
    };
    const { flat, unflatten } = flattenForDepth(schema);
    expect(Object.keys(flat.properties as Any)).toEqual(["a.b.c"]);
    expect((flat as Any).required).toEqual(["a.b.c"]);
    expect(unflatten({ "a.b.c": "x" })).toEqual({ a: { b: { c: "x" } } });
  });

  it("collapses a 6-deep object chain to a single flat key", () => {
    let node: SchemaNode = { type: "object", properties: { leaf: { type: "string" } }, required: ["leaf"] };
    for (let i = 0; i < 6; i++) node = { type: "object", properties: { child: node }, required: ["child"] };
    const { flat } = flattenForDepth(node);
    expect(depthOf(node)).toBe(8);
    expect(depthOf(flat)).toBe(2); // root object + one leaf
  });

  it("recurses into arrays but never hoists ACROSS them", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: { type: "object", properties: { meta: { type: "object", properties: { id: { type: "string" } }, required: ["id"] } }, required: ["meta"] },
        },
      },
      required: ["items"],
    };
    const { flat, unflatten } = flattenForDepth(schema);
    expect(Object.keys((flat as Any).properties)).toEqual(["items"]); // array stays put
    expect(Object.keys((flat as Any).properties.items.items.properties)).toEqual(["meta.id"]); // hoisted WITHIN the item
    expect(unflatten({ items: [{ "meta.id": "1" }, { "meta.id": "2" }] })).toEqual({ items: [{ meta: { id: "1" } }, { meta: { id: "2" } }] });
  });

  it("flattens WITHIN arrays — alternating object/array depth (the foo.bar / baz.bam case)", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: {
        foo: {
          type: "object",
          properties: {
            bar: {
              type: "array",
              items: { type: "object", properties: { baz: { type: "object", properties: { bam: { type: "number" } }, required: ["bam"] } }, required: ["baz"] },
            },
          },
          required: ["bar"],
        },
      },
      required: ["foo"],
    };
    const { flat, unflatten } = flattenForDepth(schema);
    expect(Object.keys(flat.properties as Any)).toEqual(["foo.bar"]); // foo hoisted; bar stays an array
    expect(Object.keys((flat as Any).properties["foo.bar"].items.properties)).toEqual(["baz.bam"]); // hoisted INSIDE the item
    expect(depthOf(schema)).toBe(6);
    expect(depthOf(flat)).toBe(4); // root → foo.bar[] → item → baz.bam
    expect(unflatten({ "foo.bar": [{ "baz.bam": 1 }, { "baz.bam": 2 }] })).toEqual({ foo: { bar: [{ baz: { bam: 1 } }, { baz: { bam: 2 } }] } });
  });

  it("maximum-safe: hoists an OPTIONAL object that has a required child (lifted keys become optional)", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: { opt: { type: "object", properties: { r: { type: "string" }, s: { type: "number" } }, required: ["r"] } },
      required: [],
    };
    const { flat, unflatten } = flattenForDepth(schema);
    expect(Object.keys(flat.properties as Any).sort()).toEqual(["opt.r", "opt.s"]);
    expect((flat as Any).required ?? []).toEqual([]); // opt may be wholly absent → no lifted key is required
    expect(unflatten({ "opt.r": "x", "opt.s": 1 })).toEqual({ opt: { r: "x", s: 1 } });
    expect(unflatten({})).toEqual({}); // all `opt.*` absent ⇒ opt omitted (it can never be {})
  });

  it("does NOT hoist an optional object that permits {} (absent vs empty would collapse)", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: { opt: { type: "object", properties: { s: { type: "string" } }, required: [] } },
      required: [],
    };
    expect(Object.keys(flattenForDepth(schema).flat.properties as Any)).toEqual(["opt"]);
  });

  it("does NOT hoist a nullable object", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: { n: { type: ["object", "null"], properties: { x: { type: "string" } }, required: ["x"] } },
      required: ["n"],
    };
    expect(Object.keys(flattenForDepth(schema).flat.properties as Any)).toEqual(["n"]);
  });

  it("does NOT hoist an open map (additionalProperties not false)", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: { m: { type: "object", properties: { x: { type: "string" } }, required: ["x"], additionalProperties: true } },
      required: ["m"],
    };
    expect(Object.keys(flattenForDepth(schema).flat.properties as Any)).toEqual(["m"]);
  });

  it("does NOT hoist a property whose name already contains the separator", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: { "a.b": { type: "object", properties: { c: { type: "string" } }, required: ["c"] } },
      required: ["a.b"],
    };
    expect(Object.keys(flattenForDepth(schema).flat.properties as Any)).toEqual(["a.b"]);
  });

  it("collision guard: keeps a child nested when hoisting would shadow a literal sibling", () => {
    const schema: SchemaNode = {
      type: "object",
      properties: {
        a: { type: "object", properties: { b: { type: "string" } }, required: ["b"] }, // would hoist to "a.b"
        "a.b": { type: "number" }, // …but a real sibling already owns that key
      },
      required: ["a", "a.b"],
    };
    const { flat, unflatten } = flattenForDepth(schema);
    expect(Object.keys(flat.properties as Any).sort()).toEqual(["a", "a.b"]); // `a` stays nested
    expect(unflatten({ a: { b: "x" }, "a.b": 2 })).toEqual({ a: { b: "x" }, "a.b": 2 });
  });

  it("leaves {}-any leaves untouched and carries arbitrary content through unflatten", () => {
    const schema: SchemaNode = { type: "object", properties: { wrap: { type: "object", properties: { payload: {} }, required: ["payload"] } }, required: ["wrap"] };
    const { flat, unflatten } = flattenForDepth(schema);
    expect((flat.properties as Any)["wrap.payload"]).toEqual({});
    expect(unflatten({ "wrap.payload": { anything: [1, 2] } })).toEqual({ wrap: { payload: { anything: [1, 2] } } });
  });

  it("unflatten is lenient: an already-nested value passes through", () => {
    const schema: SchemaNode = { type: "object", properties: { a: { type: "object", properties: { b: { type: "string" } }, required: ["b"] } }, required: ["a"] };
    expect(flattenForDepth(schema).unflatten({ a: { b: "x" } })).toEqual({ a: { b: "x" } });
  });

  it("round-trips a meta-style signature-as-data row (object chain hoisted, arrays preserved)", () => {
    const sigSlot = (extra: SchemaNode = {}): SchemaNode => ({
      type: "object",
      properties: { kind: { type: "string" }, schema: {}, ...extra },
      required: ["kind"],
    });
    const row: SchemaNode = {
      type: "object",
      properties: {
        input: {
          type: "object",
          properties: {
            description: { type: "string" },
            signature: { type: "object", properties: { input: sigSlot(), output: sigSlot({ name: { type: "string" } }) }, required: ["input", "output"] },
          },
          required: ["description", "signature"],
        },
        output: { type: "array", items: { type: "object", properties: { input: {}, output: {} }, required: ["input"] } },
      },
      required: ["input", "output"],
    };
    const { flat, unflatten } = flattenForDepth(row);
    const keys = Object.keys(flat.properties as Any);
    expect(keys).toContain("input.description");
    expect(keys).toContain("input.signature.input.kind");
    expect(keys).toContain("input.signature.output.name");
    expect(keys).toContain("output"); // array stays a single key

    const value = {
      input: { description: "d", signature: { input: { kind: "json", schema: { type: "string" } }, output: { name: "o", kind: "text", schema: { type: "number" } } } },
      output: [{ input: "hi", output: "bye" }],
    };
    const flatValue = {
      "input.description": "d",
      "input.signature.input.kind": "json",
      "input.signature.input.schema": { type: "string" },
      "input.signature.output.name": "o",
      "input.signature.output.kind": "text",
      "input.signature.output.schema": { type: "number" },
      output: [{ input: "hi", output: "bye" }],
    };
    expect(unflatten(flatValue)).toEqual(value);
  });
});
