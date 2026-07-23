import { describe, expect, expectTypeOf, it } from "vitest";
import { collectRefs, type JsonSchema, type JsonValue, type Jsonify, type SchemaDocument, type SchemaOutput } from "../src/index";

describe("SchemaOutput<S> — the phantom, or JsonValue (§3.4: never `unknown` in an exported position)", () => {
  it("reads a document with no phantom as JsonValue, INDEX-SIGNATURE documents included", () => {
    // The regression: `SchemaDocument` matches the optional-phantom shape with nothing to infer from,
    // so the phantom inferred as `unknown` and leaked out of an exported type.
    expectTypeOf<SchemaOutput<SchemaDocument>>().toEqualTypeOf<JsonValue>();
    expectTypeOf<SchemaOutput<{ readonly [k: string]: JsonValue }>>().toEqualTypeOf<JsonValue>();
    expectTypeOf<SchemaOutput<{ type: "string" }>>().toEqualTypeOf<JsonValue>();
  });

  it("still extracts a phantom that IS bound", () => {
    expectTypeOf<SchemaOutput<JsonSchema<Date>>>().toEqualTypeOf<Date>();
    expectTypeOf<SchemaOutput<JsonSchema<{ a: string }>>>().toEqualTypeOf<{ a: string }>();
  });
});

describe("Jsonify<JsonValue> — a value that is ALREADY JSON projects to itself (§3.2)", () => {
  it("resolves without a deep-instantiation blow-up (TS2589)", () => {
    // Checking a value against `Jsonify<JsonValue>` used to fail compilation outright, and it is
    // `Codec`'s DEFAULT wire parameter — so the blow-up was latent in public API.
    const v: Jsonify<JsonValue> = { a: 1, b: [true, null] };
    expect(v).toEqual({ a: 1, b: [true, null] });
    expectTypeOf<Jsonify<JsonValue>>().toEqualTypeOf<JsonValue>();
    expectTypeOf<Jsonify<{ [k: string]: JsonValue }>>().toEqualTypeOf<{ [k: string]: JsonValue }>();
  });
});

describe("collectRefs — references, not data (§3)", () => {
  it("collects every $ref reachable through schema keywords", () => {
    const doc = { properties: { a: { $ref: "#/$defs/A" } }, items: [{ $ref: "#/$defs/B" }] };
    expect([...collectRefs(doc)].sort()).toEqual(["#/$defs/A", "#/$defs/B"]);
  });

  it("does NOT harvest a $ref out of const/enum/default — those hold DATA", () => {
    // `{ const: { $ref: "…" } }` describes a literal object that HAS a `$ref` member; treating it as a
    // reference makes the loader chase a document the schema never named.
    expect([...collectRefs({ const: { $ref: "http://not-a-schema-ref" } })]).toEqual([]);
    expect([...collectRefs({ enum: [{ $ref: "a" }] })]).toEqual([]);
    expect([...collectRefs({ default: { $ref: "b" } })]).toEqual([]);
    expect([...collectRefs({ const: { $ref: "data" }, properties: { a: { $ref: "#/real" } } })]).toEqual(["#/real"]);
  });

  it("terminates on a cyclic object graph — the parameter is `unknown`, so the input is not trusted", () => {
    const node: Record<string, unknown> = { $ref: "#/$defs/A" };
    node.self = node;
    expect([...collectRefs(node)]).toEqual(["#/$defs/A"]);
  });
});
