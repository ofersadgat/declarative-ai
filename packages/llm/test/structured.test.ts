import { describe, expect, it } from "vitest";
import type { JsonValue } from "@declarative-ai/json";
import {
  findDiscriminators,
  parseOutputSchema,
  patchSchemaForAnthropic,
  reconstructOutput,
} from "../src/structured";

describe("patchSchemaForAnthropic (§5.1)", () => {
  it("does NOT mutate the input (original survives for reconstruction)", () => {
    const original = {
      type: "object",
      properties: { n: { type: "number", minimum: 0, maximum: 10 } },
    };
    const snapshot = structuredClone(original);
    patchSchemaForAnthropic(original);
    expect(original).toEqual(snapshot);
  });

  it("adds additionalProperties:false to every object and strips unsupported keywords", () => {
    const patched = patchSchemaForAnthropic({
      type: "object",
      properties: {
        name: { type: "string", minLength: 1, pattern: "^x" },
        nested: { type: "object", properties: { a: { type: "number", minimum: 0 } } },
      },
    }) as Record<string, any>;
    expect(patched.additionalProperties).toBe(false);
    expect(patched.properties.name.minLength).toBeUndefined();
    expect(patched.properties.name.pattern).toBeUndefined();
    expect(patched.properties.nested.additionalProperties).toBe(false);
    expect(patched.properties.nested.properties.a.minimum).toBeUndefined();
  });

  it("collapses a [type, null] union to the non-null type", () => {
    const patched = patchSchemaForAnthropic({ type: ["string", "null"] }) as Record<string, any>;
    expect(patched.type).toBe("string");
  });

  it("flattens a discriminated oneOf into one merged object", () => {
    const patched = patchSchemaForAnthropic({
      oneOf: [
        {
          type: "object",
          properties: { kind: { const: "a" }, av: { type: "number" } },
          required: ["kind", "av"],
        },
        {
          type: "object",
          properties: { kind: { const: "b" }, bv: { type: "string" } },
          required: ["kind", "bv"],
        },
      ],
    }) as Record<string, any>;
    expect(patched.type).toBe("object");
    expect(Object.keys(patched.properties).sort()).toEqual(["av", "bv", "kind"]);
    expect(patched.oneOf).toBeUndefined();
  });

  it("marks EVERY variant's required fields required when flattening (over-strict workaround) — reconstruct neutralizes it", () => {
    // KNOWN over-strictness (review finding #4): a flattened oneOf requires the UNION of all
    // variants' required fields at once, so the patched schema demands fields no single real
    // variant has (`av` AND `bv` here). This is acceptable ONLY because the patched schema is
    // sent merely to COAX the model — correctness is restored by `reconstructOutput` against the
    // ORIGINAL schema, never by trusting the patched one. The opt-in necessity guard lives in
    // `test/live/anthropic-structured.live.test.ts`: when Anthropic accepts the raw union, this
    // whole workaround (flattening + over-strict required) can be removed.
    const original: JsonValue = {
      oneOf: [
        { type: "object", properties: { kind: { const: "a" }, av: { type: "number" } }, required: ["kind", "av"] },
        { type: "object", properties: { kind: { const: "b" }, bv: { type: "string" } }, required: ["kind", "bv"] },
      ],
    };
    const patched = patchSchemaForAnthropic(structuredClone(original)) as Record<string, any>;
    // Over-strict by design: the union of both variants' required fields.
    expect((patched.required as string[]).sort()).toEqual(["av", "bv", "kind"]);
    // The model, coaxed by the flattened schema, returns BOTH variants' fields; reconstruct
    // against the original strips the unmatched variant's (wrongly-required) `bv`, yielding a
    // shape that satisfies the real `circle`/`a` variant.
    expect(reconstructOutput({ kind: "a", av: 1, bv: "leaked" }, original as any)).toEqual({ kind: "a", av: 1 });
  });
});

describe("findDiscriminators + reconstructOutput", () => {
  const unionSchema = {
    oneOf: [
      {
        type: "object",
        properties: { kind: { const: "a" }, av: { type: "number" } },
      },
      {
        type: "object",
        properties: { kind: { const: "b" }, bv: { type: "string" } },
      },
    ],
  };

  it("identifies the single-key discriminator", () => {
    expect(findDiscriminators(unionSchema.oneOf as any)).toEqual(["kind"]);
  });

  it("strips the other variant's props after flattening", () => {
    // The flattened model output carries BOTH variants' props.
    const modelOutput = { kind: "a", av: 42, bv: "leaked" };
    expect(reconstructOutput(modelOutput, unionSchema as any)).toEqual({ kind: "a", av: 42 });
  });

  it("is a no-op for a plain object schema", () => {
    const schema = { type: "object", properties: { answer: { type: "string" } } };
    const out = { answer: "4" };
    expect(reconstructOutput(out, schema as any)).toEqual(out);
  });

  it("recurses through arrays of unions", () => {
    const schema = { type: "array", items: unionSchema };
    const out = [
      { kind: "a", av: 1, bv: "x" },
      { kind: "b", av: 9, bv: "y" },
    ];
    expect(reconstructOutput(out, schema as any)).toEqual([
      { kind: "a", av: 1 },
      { kind: "b", bv: "y" },
    ]);
  });
});

describe("parseOutputSchema", () => {
  it("unwraps a double-serialized schema string", () => {
    expect(parseOutputSchema('{"type":"object","properties":{"a":{"type":"string"}}}')).toEqual({
      type: "object",
      properties: { a: { type: "string" } },
    });
  });

  it("infers type:object when properties present but type missing", () => {
    expect(parseOutputSchema({ properties: { a: { type: "string" } } })?.type).toBe("object");
  });

  it("rejects non-schema values", () => {
    expect(parseOutputSchema("not json")).toBeNull();
    expect(parseOutputSchema({ foo: "bar" })).toBeNull();
    expect(parseOutputSchema(null)).toBeNull();
  });
});
