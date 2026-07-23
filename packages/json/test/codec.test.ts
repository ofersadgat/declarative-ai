import { describe, expect, expectTypeOf, it } from "vitest";
import {
  CodecRegistry,
  X_TYPE,
  codecs,
  decodeWithSchema,
  encodeWithSchema,
  typeNameOf,
  type Codec,
  type Decoded,
  type Jsonify,
  type JsonSchema,
  type JsonValue,
  type SchemaDocument,
} from "../src/index";

// The declaration-merging half of §3.3: a type NAME, bound globally, once.
declare module "../src/codec" {
  interface TypeRegistry {
    DateTime: { value: Date; json: number };
  }
}

describe("Jsonify<T> — the wire projection of a decoded type (§3.2)", () => {
  it("follows toJSON(), which is why ENCODE is derivable from the type", () => {
    expectTypeOf<Jsonify<Date>>().toEqualTypeOf<string>(); // Date.toJSON(): string
    expectTypeOf<Jsonify<{ at: Date }>>().toEqualTypeOf<{ at: string }>();
  });

  it("DROPS object members JSON cannot carry and NULLS array elements — JSON.stringify's two behaviors", () => {
    expectTypeOf<Jsonify<{ a: string; f: () => void }>>().toEqualTypeOf<{ a: string }>();
    expectTypeOf<Jsonify<(string | (() => void))[]>>().toEqualTypeOf<(string | null)[]>();
  });

  it("preserves tuples and optionality", () => {
    expectTypeOf<Jsonify<[string, number]>>().toEqualTypeOf<[string, number]>();
    expectTypeOf<Jsonify<{ a?: string }>>().toEqualTypeOf<{ a?: string }>();
  });

  it("leaves an already-JSON type alone", () => {
    expectTypeOf<Jsonify<{ a: string; b: number[] }>>().toEqualTypeOf<{ a: string; b: number[] }>();
  });
});

describe("JsonSchema<T> — the phantom carries the DECODED type (§3.2)", () => {
  it("is inferable at a call site, which `JsonSchema<Jsonify<T>>` would not be", () => {
    // The whole reason the phantom carries `T` and not `Jsonify<T>`: TypeScript cannot infer backwards
    // through a conditional type, so the latter would make `T` uninferable everywhere.
    function readsSchema<T>(_s: JsonSchema<T>): T {
      return undefined as T;
    }
    const schema = { type: "number", [X_TYPE]: "DateTime" } as JsonSchema<Date>;
    expectTypeOf(readsSchema(schema)).toEqualTypeOf<Date>();
  });
});

describe("Decoded<S> — what a slot decodes to (§3.3)", () => {
  it("maps an x-type node to its registered decoded type", () => {
    expectTypeOf<Decoded<{ readonly type: "number"; readonly "x-type": "DateTime" }>>().toEqualTypeOf<Date>();
  });

  it("walks objects, honoring required vs optional, and leaves plain nodes as their JSON shape", () => {
    type S = {
      readonly type: "object";
      readonly properties: { readonly at: { readonly type: "number"; readonly "x-type": "DateTime" }; readonly note: { readonly type: "string" } };
      readonly required: readonly ["at"];
    };
    expectTypeOf<Decoded<S>["at"]>().toEqualTypeOf<Date>();
    expectTypeOf<Decoded<S>["note"]>().toEqualTypeOf<string | undefined>();
  });

  it("degrades to JsonValue for a shape it cannot interpret — surfaced, never a silent unknown", () => {
    expectTypeOf<Decoded<{ readonly anyOf: readonly [] }>>().toEqualTypeOf<JsonValue>();
  });
});

describe("the codec registry — resolution by TYPE NAME, once, globally (§3.3)", () => {
  const registry = new CodecRegistry();
  registry.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });

  it("reads the name off the schema document, which is what can be stored (a closure cannot)", () => {
    expect(typeNameOf({ type: "number", [X_TYPE]: "DateTime" })).toBe("DateTime");
    expect(typeNameOf({ type: "number" })).toBeUndefined();
  });

  it("REFUSES a second, different codec for one name — a stored schema must mean one thing", () => {
    expect(() => registry.register("DateTime", { encode: () => 0, decode: () => new Date(0) })).toThrow(/already registered/);
  });

  it("leaves an UNREGISTERED type name as raw JSON — a schema may name a type this process doesn't model", () => {
    const schema = { type: "number", [X_TYPE]: "Money" };
    expect(decodeWithSchema(schema, 5)).toBe(5);
    expect(encodeWithSchema(schema, 5)).toBe(5);
  });

  it("passes plain nodes through structurally", () => {
    const schema = { type: "object", properties: { n: { type: "number" } } };
    expect(decodeWithSchema(schema, { n: 1 })).toEqual({ n: 1 });
  });
});

// The runtime walk, driven through the REAL exported entry points. An earlier version of this suite
// reimplemented the recursion locally, so the walk's own `items`, `additionalProperties` and object-key
// handling were never executed — which is how a dropped `__proto__` key survived.
describe("the runtime walk — decodeWithSchema / encodeWithSchema (§3.3)", () => {
  // The registry these use is the process-global singleton by design; vitest isolates module state per
  // test file, so registering here cannot leak into another suite.
  codecs.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });

  it("round-trips a named leaf under `properties`", () => {
    const schema = { type: "object", properties: { at: { type: "number", [X_TYPE]: "DateTime" }, note: { type: "string" } } };
    const decoded = { at: new Date(1_700_000_000_000), note: "hi" };
    const wire = encodeWithSchema(schema, decoded);
    expect(wire).toEqual({ at: 1_700_000_000_000, note: "hi" });
    expect(decodeWithSchema(schema, wire)).toEqual(decoded);
  });

  it("descends `items` — every element decoded against the one item schema", () => {
    const schema = { type: "array", items: { type: "number", [X_TYPE]: "DateTime" } };
    expect(decodeWithSchema(schema, [1, 2])).toEqual([new Date(1), new Date(2)]);
    expect(encodeWithSchema(schema, [new Date(1), new Date(2)])).toEqual([1, 2]);
    // Nested one level: an array INSIDE a property.
    const nested = { type: "object", properties: { ats: schema } };
    expect(decodeWithSchema(nested, { ats: [3] })).toEqual({ ats: [new Date(3)] });
  });

  it("falls back to `additionalProperties` for a key `properties` does not name", () => {
    const schema = {
      type: "object",
      properties: { id: { type: "string" } },
      additionalProperties: { type: "number", [X_TYPE]: "DateTime" },
    };
    expect(decodeWithSchema(schema, { id: "a", seen: 7 })).toEqual({ id: "a", seen: new Date(7) });
    expect(encodeWithSchema(schema, { id: "a", seen: new Date(7) })).toEqual({ id: "a", seen: 7 });
  });

  it("KEEPS a `__proto__` key — `JSON.parse` produces it as an own property, so wire input carries it", () => {
    // `out[k] = v` invoked `Object.prototype`'s `__proto__` setter and the key vanished from the result.
    const schema = { type: "object", properties: { a: { type: "number" } } };
    const wire = JSON.parse('{"a":1,"__proto__":{"b":2}}') as JsonValue;
    const out = decodeWithSchema(schema, wire) as Record<string, unknown>;
    expect(Object.keys(out)).toEqual(["a", "__proto__"]);
    expect(out["__proto__"]).toEqual({ b: 2 });
    expect(Object.getPrototypeOf(out)).toBe(Object.prototype); // and it is a KEY, not a re-parenting
    // Same under `additionalProperties`, which is the branch that would otherwise walk `Object.prototype`
    // as if it were the schema for that key.
    const open = { type: "object", additionalProperties: { type: "number", [X_TYPE]: "DateTime" } };
    const walked = decodeWithSchema(open, JSON.parse('{"__proto__":9}') as JsonValue) as Record<string, unknown>;
    expect(walked["__proto__"]).toEqual(new Date(9));
  });

  it("leaves a value alone when the schema constrains nothing", () => {
    expect(decodeWithSchema(undefined, { a: 1 })).toEqual({ a: 1 });
    expect(decodeWithSchema({ type: "array" }, [1, 2])).toEqual([1, 2]);
  });
});

// The composition keywords: a leaf `x-type` nested under `anyOf`/`oneOf`/`allOf` or a TUPLE `items`
// must still decode — before this the walk recursed only through `items`/`properties`, so these
// silently never lifted.
describe("the runtime walk — composition keywords (§3.3)", () => {
  // The suite above already registered `DateTime` on the process-global singleton; re-registering a
  // fresh codec object would throw (one name, one codec), so register only if absent.
  if (!codecs.has("DateTime")) codecs.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });

  it("selects the applicable `anyOf` branch by the value's JSON type — and round-trips", () => {
    const schema: SchemaDocument = { anyOf: [{ type: "number", [X_TYPE]: "DateTime" }, { type: "null" }] };
    // A number selects the DateTime branch and decodes; null selects the `null` branch and passes.
    expect(decodeWithSchema(schema, 5)).toEqual(new Date(5));
    expect(decodeWithSchema(schema, null)).toBeNull();
    expect(encodeWithSchema(schema, new Date(5))).toBe(5);
    expect(encodeWithSchema(schema, null)).toBeNull();
  });

  it("treats `oneOf` the same as `anyOf` — first branch whose declared type admits the value", () => {
    const schema: SchemaDocument = { oneOf: [{ type: "string" }, { type: "number", [X_TYPE]: "DateTime" }] };
    expect(decodeWithSchema(schema, 7)).toEqual(new Date(7));
    expect(decodeWithSchema(schema, "hi")).toBe("hi");
  });

  it("leaves the value RAW when no `anyOf` branch applies — never a wrong lift", () => {
    // A boolean matches neither branch's declared type ⇒ raw.
    const schema: SchemaDocument = { anyOf: [{ type: "number", [X_TYPE]: "DateTime" }, { type: "string" }] };
    expect(decodeWithSchema(schema, true)).toBe(true);
    // A branch carrying no `type` cannot be discriminated ⇒ the whole union stays raw.
    const untyped: SchemaDocument = { anyOf: [{ [X_TYPE]: "DateTime" }] };
    expect(decodeWithSchema(untyped, 9)).toBe(9);
  });

  it("descends a TUPLE `items: [...]` — i-th element by i-th subschema, overflow by `additionalItems`", () => {
    const schema: SchemaDocument = {
      type: "array",
      items: [{ type: "string" }, { type: "number", [X_TYPE]: "DateTime" }],
      additionalItems: { type: "number", [X_TYPE]: "DateTime" },
    };
    const decoded = decodeWithSchema(schema, ["id", 2, 3]);
    expect(decoded).toEqual(["id", new Date(2), new Date(3)]);
    expect(encodeWithSchema(schema, ["id", new Date(2), new Date(3)])).toEqual(["id", 2, 3]);
  });

  it("descends a `prefixItems` tuple — overflow by the single-schema `items` (2020-12 shape)", () => {
    const schema: SchemaDocument = {
      type: "array",
      prefixItems: [{ type: "string" }, { type: "number", [X_TYPE]: "DateTime" }],
      items: { type: "number", [X_TYPE]: "DateTime" },
    };
    expect(decodeWithSchema(schema, ["id", 2, 3])).toEqual(["id", new Date(2), new Date(3)]);
    expect(encodeWithSchema(schema, ["id", new Date(2), new Date(3)])).toEqual(["id", 2, 3]);
  });

  it("threads the value through every `allOf` branch — each decodes the members it names", () => {
    const schema: SchemaDocument = {
      allOf: [
        { type: "object", properties: { at: { type: "number", [X_TYPE]: "DateTime" } } },
        { type: "object", properties: { seen: { type: "number", [X_TYPE]: "DateTime" } } },
      ],
    };
    const wire = { at: 1, seen: 2, note: "x" };
    const decoded = { at: new Date(1), seen: new Date(2), note: "x" };
    expect(decodeWithSchema(schema, wire)).toEqual(decoded);
    expect(encodeWithSchema(schema, decoded)).toEqual(wire);
  });

  it("does not stack-overflow on a self-referential composition schema — degrades to raw at the cap", () => {
    // A hand-built cyclic schema (`s.allOf = [s]`) recurses on the SAME value; the depth bound turns it
    // into a passthrough rather than a crash.
    const s: Record<string, unknown> = { type: "number", [X_TYPE]: "DateTime" };
    s["allOf"] = [s];
    // The top-level `x-type` short-circuits to the codec before the cyclic `allOf` is ever walked.
    expect(decodeWithSchema(s as never, 4)).toEqual(new Date(4));
    // Under a wrapper with no top-level `x-type`, the cyclic branch is walked but bounded — no throw.
    const wrapper: Record<string, unknown> = {};
    wrapper["allOf"] = [wrapper];
    expect(() => decodeWithSchema(wrapper as never, 4)).not.toThrow();
  });
});

describe("Codec's default wire parameter (§3.2)", () => {
  it("is checkable — `Jsonify<JsonValue>` must resolve, not blow the instantiation depth", () => {
    // `Codec<T = JsonValue, J = Jsonify<T>>`: annotating a value with the bare `Codec` used to fail
    // compilation with TS2589, which made the DEFAULT of an exported interface unusable.
    const identity: Codec = { encode: (v) => v, decode: (v) => v };
    expect(identity.encode({ a: 1 })).toEqual({ a: 1 });
  });
});
