/**
 * A signature declares NAMED, individually typed slots — and folds back to the one `Schema<I>` the
 * type-map bridge has always bound.
 *
 * `Signature.input` was a single `Parameter`, on the reading that multi-field inputs are just an `I`
 * that happens to be an object. That left a signature unable to describe the thing it is the contract
 * for: an `Operation`'s own `input` is a map, so a registered function could not say which named slots
 * it takes, and the one checker that needed to know took an object schema apart to get them back.
 *
 * These pin both halves of the change: the map is what a caller declares, and the object schema is
 * what `Schema<I>` still means.
 */
import { describe, expect, it } from "vitest";
import type { InlineFamily, Signature } from "../src/model.js";
import { isRequiredSlot } from "../src/model.js";
import { asSignature, bindSignatureTemplate, inlineDeref, signatureInputSchema, SIGNATURE_META_SCHEMA } from "../src/signatureSchema.js";

const review: Signature<InlineFamily> = {
  input: {
    text: { kind: "text", schema: { type: "string" }, index: 0 },
    tone: { kind: "text", schema: { type: "string" }, index: 1, optional: true },
    limit: { kind: "json", schema: { type: "number" }, index: 2, binding: { json: 3 } },
  },
  output: { name: "output", kind: "json", schema: { type: "object" } },
};

describe("Schema<I> is the object the named slots describe", () => {
  it("folds every slot into one object schema", () => {
    expect(signatureInputSchema(review, inlineDeref)).toEqual({
      type: "object",
      properties: { text: { type: "string" }, tone: { type: "string" }, limit: { type: "number" } },
      required: ["text"],
    });
  });

  it("counts a slot as required only when it is neither optional nor defaulted", () => {
    // The two mean different things and only their conjunction is "the caller must supply this":
    // `optional` is permission to omit, a `binding` is an answer for the caller who does.
    expect(isRequiredSlot(review.input.text!)).toBe(true);
    expect(isRequiredSlot(review.input.tone!)).toBe(false);
    expect(isRequiredSlot(review.input.limit!)).toBe(false);
  });

  it("omits `required` entirely when nothing is", () => {
    const optional: Signature<InlineFamily> = {
      input: { tone: { kind: "text", optional: true } },
      output: { name: "output", kind: "json" },
    };
    expect(signatureInputSchema(optional, inlineDeref)).toEqual({ type: "object", properties: { tone: { type: "string" } } });
  });

  it("gives a no-argument function an empty object, not an absent one", () => {
    // A function of no arguments is a function; `{}` is its honest input type, and the empty map has
    // to be a legal signature or an entry that takes nothing could not declare itself at all.
    expect(signatureInputSchema({ input: {}, output: { name: "output", kind: "json" } }, inlineDeref)).toEqual({
      type: "object",
      properties: {},
    });
  });

  it("binds the folded schema as the `$input` type variable", () => {
    const bound = bindSignatureTemplate({ type: "array", items: { $param: "input" } }, inlineDeref, review);
    expect(bound).toEqual({ type: "array", items: signatureInputSchema(review, inlineDeref) });
  });
});

describe("the structural guard reads the map", () => {
  it("accepts a signature whose slots are all parameters", () => {
    expect(asSignature<InlineFamily>(review)).toBe(review);
  });

  it("accepts an empty slot map", () => {
    const none = { input: {}, output: { name: "output", kind: "json" } };
    expect(asSignature<InlineFamily>(none)).toBe(none);
  });

  it("refuses a slot that is not a parameter", () => {
    expect(asSignature<InlineFamily>({ input: { text: "a string" }, output: { name: "o", kind: "json" } })).toBeUndefined();
  });

  it("refuses the OLD single-parameter shape, which would otherwise read as two slots named `kind` and `schema`", () => {
    // Worth pinning as its own case: `{ kind: "json", schema: … }` is a well-formed object whose keys
    // are not parameters, so a guard that only checked "is `input` an object" would let a stale
    // signature through and report every real slot as one the implementation does not accept.
    expect(asSignature<InlineFamily>({ input: { kind: "json", schema: { type: "object" } }, output: { name: "o", kind: "json" } })).toBeUndefined();
  });
});

describe("the reflective meta-schema", () => {
  it("describes the input side as a map of slots rather than as one slot", () => {
    // The names are the author's, so the witness cannot list them — every slot shares the one `$input`
    // hole, which is exactly the variable the fold above collapses them into.
    expect((SIGNATURE_META_SCHEMA.properties as Record<string, unknown>).input).toMatchObject({
      type: "object",
      additionalProperties: { title: "signature-parameter", properties: { schema: { $param: "input" } } },
    });
  });
});
