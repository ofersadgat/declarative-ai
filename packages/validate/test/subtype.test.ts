import { describe, expect, it } from "vitest";
import { isSubschema, type Schema } from "../src/subtype";

const sub = (a: Schema, b: Schema) => isSubschema(a, b);

describe("isSubschema (structural subtype, ported from findmyprompt)", () => {
  it("identical schemas are subtypes", () => {
    const s = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    expect(sub(s, s).ok).toBe(true);
  });

  it("a universal (unconstrained) consumer accepts anything", () => {
    expect(sub({ type: "string" }, {}).ok).toBe(true);
  });

  it("producer guaranteeing MORE required props is a subtype; missing one is not", () => {
    const consumer = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    expect(sub({ type: "object", properties: { a: { type: "string" }, b: { type: "number" } }, required: ["a", "b"] }, consumer).ok).toBe(true);
    const missing = sub({ type: "object", properties: { b: { type: "number" } }, required: ["b"] }, { ...consumer, required: ["a", "b"] });
    expect(missing.ok).toBe(false);
    expect(missing.reason).toContain("requires property 'a'");
  });

  it("rejects a property whose type doesn't narrow", () => {
    const r = sub(
      { type: "object", properties: { answer: { type: "number" } }, required: ["answer"] },
      { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
    );
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("property 'answer'");
  });

  it("treats integer as a subtype of number", () => {
    expect(sub({ type: "integer" }, { type: "number" }).ok).toBe(true);
    expect(sub({ type: "number" }, { type: "integer" }).ok).toBe(false);
  });

  it("enum/const must be a subset", () => {
    expect(sub({ const: "a" }, { enum: ["a", "b"] }).ok).toBe(true);
    expect(sub({ enum: ["a"] }, { enum: ["a", "b"] }).ok).toBe(true);
    expect(sub({ const: "c" }, { enum: ["a", "b"] }).ok).toBe(false);
    expect(sub({ type: "string" }, { enum: ["a"] }).ok).toBe(false); // unconstrained producer
  });

  it("honors closed consumers (additionalProperties:false)", () => {
    const closed = { type: "object", properties: { a: { type: "string" } }, additionalProperties: false };
    expect(sub({ type: "object", properties: { a: { type: "string" } }, additionalProperties: false }, closed).ok).toBe(true);
    const extra = sub({ type: "object", properties: { a: { type: "string" }, b: { type: "string" } }, additionalProperties: false }, closed);
    expect(extra.ok).toBe(false);
    expect(extra.reason).toContain("not permitted by closed consumer");
    // producer open against a closed consumer
    expect(sub({ type: "object", properties: { a: { type: "string" } } }, closed).ok).toBe(false);
  });

  it("recurses into array items", () => {
    expect(sub({ type: "array", items: { type: "string" } }, { type: "array", items: { type: "string" } }).ok).toBe(true);
    expect(sub({ type: "array", items: { type: "number" } }, { type: "array", items: { type: "string" } }).ok).toBe(false);
  });

  it("numeric bounds are widen-only", () => {
    expect(sub({ type: "number", minimum: 5 }, { type: "number", minimum: 0 }).ok).toBe(true);
    expect(sub({ type: "number", minimum: 0 }, { type: "number", minimum: 5 }).ok).toBe(false);
    expect(sub({ type: "number" }, { type: "number", minimum: 0 }).ok).toBe(false); // producer must bound it
  });

  it("is conservative: rejects unions and unmodeled consumer keywords", () => {
    expect(sub({ type: "string" }, { anyOf: [{ type: "string" }] }).ok).toBe(false);
    expect(sub({ anyOf: [{ type: "string" }] }, { type: "string" }).ok).toBe(false);
    const unmodeled = sub({ type: "number" }, { type: "number", multipleOf: 2 });
    expect(unmodeled.ok).toBe(false);
    expect(unmodeled.reason).toContain("unmodeled");
  });

  it("resolves $ref via the provided resolver", () => {
    const resolve = (id: string) => (id === "json:str" ? { type: "string" } : undefined);
    expect(isSubschema({ $ref: "json:str" }, { type: "string" }, resolve).ok).toBe(true);
    expect(isSubschema({ type: "number" }, { $ref: "json:str" }, resolve).ok).toBe(false);
  });

  it("flattens allOf (intersection) — schema evolution typechecks", () => {
    const consumerA = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    // evolved = old ∧ {+b}  (the "add a field" shape: allOf[{$ref old}, {+field}])
    const evolved = {
      allOf: [
        { type: "object", properties: { a: { type: "string" } }, required: ["a"] },
        { properties: { b: { type: "number" } }, required: ["b"] },
      ],
    };
    expect(sub(evolved, consumerA).ok).toBe(true); // still satisfies the old contract
    const consumerB = { type: "object", properties: { b: { type: "number" } }, required: ["b"] };
    expect(sub(evolved, consumerB).ok).toBe(true); // and the new field
    expect(sub(consumerA, consumerB).ok).toBe(false); // a producer without b does not
  });

  it("resolves + flattens allOf with $ref members", () => {
    const resolve = (id: string) => (id === "json:old" ? { type: "object", properties: { a: { type: "string" } }, required: ["a"] } : undefined);
    const evolved = { allOf: [{ $ref: "json:old" }, { properties: { b: { type: "number" } }, required: ["b"] }] };
    const consumer = { type: "object", properties: { a: { type: "string" }, b: { type: "number" } }, required: ["a", "b"] };
    expect(isSubschema(evolved, consumer, resolve).ok).toBe(true);
  });
});

describe("the cycle guard is a proof STACK, not a memo (API.md, \"The binding checker\")", () => {
  const resolve = (id: string): Schema | undefined => (id === "S" ? { type: "string" } : undefined);

  // The guard exists for SELF-REFERENTIAL schemas. Keyed on one side's `$ref` and never popped, it
  // instead behaved as "have ever touched", so two INDEPENDENT obligations against the same `$ref`'d
  // slot collapsed to one and the second was waved through.
  it("checks every sibling that shares a $ref'd consumer", () => {
    const sup: Schema = { type: "object", properties: { a: { $ref: "S" }, b: { $ref: "S" } }, required: ["a", "b"] };
    const sub: Schema = { type: "object", properties: { a: { type: "string" }, b: { type: "number" } }, required: ["a", "b"] };
    const r = isSubschema(sub, sup, resolve);
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/property 'b'/);
  });

  it("still accepts when every sibling genuinely matches", () => {
    const sup: Schema = { type: "object", properties: { a: { $ref: "S" }, b: { $ref: "S" } }, required: ["a", "b"] };
    const sub: Schema = { type: "object", properties: { a: { type: "string" }, b: { type: "string" } }, required: ["a", "b"] };
    expect(isSubschema(sub, sup, resolve).ok).toBe(true);
  });

  it("terminates on a genuinely self-referential pair", () => {
    const node = (): Schema => ({ type: "object", properties: { next: { $ref: "Node" } } });
    const cyclic = (id: string): Schema | undefined => (id === "Node" ? node() : undefined);
    expect(isSubschema({ $ref: "Node" }, { $ref: "Node" }, cyclic).ok).toBe(true);
  });

  it("rejects a consumer $ref it cannot resolve instead of accepting everything", () => {
    // `$ref` is not a CONSTRAINING keyword, so an unresolved one read as "universal" — the widest
    // possible accept, from the narrowest possible information.
    const r = isSubschema({ type: "string" }, { $ref: "schema:abc" });
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/could not be resolved/);
  });

  it("refuses a tuple-form `items` consumer rather than skipping the check", () => {
    const r = isSubschema({ type: "array", items: { type: "number" } }, { type: "array", items: [{ type: "string" }] } as unknown as Schema);
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/tuple-form/);
  });
});

describe("a literal producer is checked as a VALUE, not as a schema", () => {
  // `schemaOfValue` describes a literal with `type`/`const`; it does not restate `items`, bounds, or
  // `additionalProperties`. Compared schema-to-schema, every literal bound into a constrained slot was
  // rejected for "not restating" a constraint a literal never carries.
  it("accepts an array literal into an items-constrained slot", () => {
    const sub: Schema = { type: "array", const: ["a", "b"] };
    expect(isSubschema(sub, { type: "array", items: { type: "string" } }).ok).toBe(true);
  });

  it("rejects an array literal whose ELEMENTS violate the item schema", () => {
    const sub: Schema = { type: "array", const: ["a", 2] };
    const r = isSubschema(sub, { type: "array", items: { type: "string" } });
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/\[1\]/);
  });

  it("decides numeric bounds against the actual number", () => {
    expect(isSubschema({ type: "integer", const: 42 }, { type: "integer", minimum: 0, maximum: 100 }).ok).toBe(true);
    expect(isSubschema({ type: "integer", const: 900 }, { type: "integer", minimum: 0, maximum: 100 }).ok).toBe(false);
  });

  it("decides string length against the actual string", () => {
    expect(isSubschema({ type: "string", const: "hello" }, { type: "string", maxLength: 100 }).ok).toBe(true);
    expect(isSubschema({ type: "string", const: "hello" }, { type: "string", maxLength: 2 }).ok).toBe(false);
  });

  it("decides a closed consumer against the literal's actual keys", () => {
    const closed: Schema = { type: "object", properties: { a: { type: "number" } }, additionalProperties: false };
    expect(isSubschema({ type: "object", const: { a: 1 } }, closed).ok).toBe(true);
    const r = isSubschema({ type: "object", const: { a: 1, b: 2 } }, closed);
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/'b' not permitted/);
  });

  it("still enforces a required property the literal omits", () => {
    const r = isSubschema({ type: "object", const: { a: 1 } }, { type: "object", properties: { b: {} }, required: ["b"] });
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/requires property 'b'/);
  });
});
