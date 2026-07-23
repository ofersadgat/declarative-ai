/**
 * Rule-by-rule coverage of `isSubschema`. `subtype.test.ts` walks one case per rule; this walks each
 * rule's BOUNDARIES.
 *
 * Why the weight: this check is the only thing standing between an authored wiring and a runtime type
 * error, and it fails in two directions. Every wrong ACCEPT is a producer feeding a slot it does not
 * fit — the bug reaches production. Every wrong REJECT is a valid workflow the author cannot express,
 * with no override. So each rule is asserted on both sides of its edge, and every rejection asserts the
 * REASON too: a bare `ok:false` tells an author nothing about what to change.
 */
import { describe, expect, it } from "vitest";
import { isSubschema, type Schema } from "../src/subtype";

type Resolve = (id: string) => Schema | undefined;

/** Assert acceptance, surfacing the rejection reason in the failure message when it doesn't hold. */
function accepts(sub: Schema, sup: Schema, resolve?: Resolve): void {
  const r = isSubschema(sub, sup, resolve);
  expect(r.reason ?? "").toBe("");
  expect(r.ok).toBe(true);
}

/** Assert rejection AND that the reason names the cause. */
function rejects(sub: Schema, sup: Schema, reason: RegExp, resolve?: Resolve): void {
  const r = isSubschema(sub, sup, resolve);
  expect(r.ok).toBe(false);
  expect(r.reason).toMatch(reason);
}

describe("the `type` gate", () => {
  it("accepts an identical type, and a producer type-set that is a SUBSET", () => {
    accepts({ type: "string" }, { type: "string" });
    accepts({ type: "string" }, { type: ["string", "number"] });
    accepts({ type: ["string", "number"] }, { type: ["string", "number", "null"] });
  });

  it("rejects any producer type outside the consumer's set", () => {
    rejects({ type: ["string", "boolean"] }, { type: ["string", "number"] }, /'boolean' not allowed by consumer string\|number/);
    rejects({ type: "object" }, { type: "array" }, /'object' not allowed by consumer array/);
  });

  it("narrows integer to number but NOT number to integer", () => {
    accepts({ type: "integer" }, { type: "number" });
    rejects({ type: "number" }, { type: "integer" }, /'number' not allowed by consumer integer/);
  });

  it("rejects an untyped producer against a typed consumer — absence is not a guarantee", () => {
    rejects({}, { type: "string" }, /producer declares none/);
  });
});

describe("enum / const", () => {
  it("requires the producer's values to be a SUBSET of the consumer's", () => {
    accepts({ enum: ["a", "b"] }, { enum: ["a", "b", "c"] });
    rejects({ enum: ["a", "b", "c"] }, { enum: ["a", "b"] }, /"c" not in consumer enum\/const/);
  });

  it("treats const as the one-element enum it is, in both directions", () => {
    accepts({ const: "a" }, { enum: ["a", "b"] });
    accepts({ enum: ["a"] }, { const: "a" });
    rejects({ const: "z" }, { enum: ["a"] }, /"z" not in consumer/);
  });

  it("rejects an unconstrained producer against an enum consumer", () => {
    rejects({ type: "string" }, { enum: ["a"] }, /consumer is enum\/const-constrained but producer is not/);
  });

  it("compares values structurally, not by identity", () => {
    accepts({ const: { a: 1 } }, { enum: [{ a: 1 }, { a: 2 }] });
    rejects({ const: { a: 9 } }, { enum: [{ a: 1 }] }, /not in consumer/);
  });
});

describe("objects", () => {
  it("requires every consumer-required property to be producer-guaranteed", () => {
    accepts({ type: "object", properties: { a: {} }, required: ["a", "b"] }, { type: "object", required: ["a"] });
    rejects({ type: "object", properties: {} }, { type: "object", properties: { a: {} }, required: ["a"] }, /requires property 'a'/);
  });

  it("lets the producer omit an OPTIONAL property the consumer merely describes", () => {
    accepts({ type: "object", properties: {} }, { type: "object", properties: { a: { type: "string" } } });
  });

  it("recurses into shared properties and reports WHICH one failed", () => {
    rejects(
      { type: "object", properties: { a: { type: "object", properties: { b: { type: "number" } } } } },
      { type: "object", properties: { a: { type: "object", properties: { b: { type: "string" } } } } },
      /property 'a': property 'b': producer type 'number'/,
    );
  });

  it("honours a closed consumer, and lets a closed producer feed an open one", () => {
    const closed: Schema = { type: "object", properties: { a: {} }, additionalProperties: false };
    accepts(closed, { type: "object", properties: { a: {} } });
    rejects({ type: "object", properties: { a: {} } }, closed, /forbids additional properties/);
  });

  it("rejects a closed producer declaring a property the closed consumer does not permit", () => {
    rejects(
      { type: "object", properties: { a: {}, b: {} }, additionalProperties: false },
      { type: "object", properties: { a: {} }, additionalProperties: false },
      /producer property 'b' not permitted/,
    );
  });
});

describe("arrays", () => {
  it("recurses into items, including nested arrays", () => {
    accepts({ type: "array", items: { type: "integer" } }, { type: "array", items: { type: "number" } });
    accepts(
      { type: "array", items: { type: "array", items: { type: "integer" } } },
      { type: "array", items: { type: "array", items: { type: "number" } } },
    );
    rejects({ type: "array", items: { type: "string" } }, { type: "array", items: { type: "number" } }, /items: producer type 'string'/);
  });

  it("rejects an item-less producer against an item-constrained consumer, but not the reverse", () => {
    rejects({ type: "array" }, { type: "array", items: { type: "string" } }, /consumer constrains array items but producer doesn't/);
    accepts({ type: "array", items: { type: "string" } }, { type: "array" });
  });
});

describe("numeric and length bounds are widen-only", () => {
  it("accepts an equal or strictly narrower producer bound", () => {
    accepts({ type: "number", minimum: 0, maximum: 10 }, { type: "number", minimum: 0, maximum: 10 });
    accepts({ type: "number", minimum: 5, maximum: 6 }, { type: "number", minimum: 0, maximum: 10 });
    accepts({ type: "string", minLength: 5, maxLength: 6 }, { type: "string", minLength: 1, maxLength: 10 });
  });

  it("rejects a wider producer bound on every axis", () => {
    rejects({ type: "number", minimum: -1 }, { type: "number", minimum: 0 }, /minimum \(-1\) below consumer minimum 0/);
    rejects({ type: "number", maximum: 11 }, { type: "number", maximum: 10 }, /maximum \(11\) above consumer maximum 10/);
    rejects({ type: "string", minLength: 0 }, { type: "string", minLength: 1 }, /minLength below consumer minLength 1/);
    rejects({ type: "string", maxLength: 99 }, { type: "string", maxLength: 10 }, /maxLength above consumer maxLength 10/);
  });

  it("rejects a producer that declares no bound at all — unbounded is not narrower", () => {
    rejects({ type: "number" }, { type: "number", minimum: 0 }, /minimum \(undefined\) below/);
  });
});

describe("allOf is flattened as an intersection", () => {
  it("unions `required` across members", () => {
    const evolved: Schema = { allOf: [{ type: "object", required: ["a"] }, { type: "object", required: ["b"] }] };
    accepts({ type: "object", properties: { a: {}, b: {} }, required: ["a", "b"] }, evolved);
    rejects({ type: "object", properties: { a: {} }, required: ["a"] }, evolved, /requires property 'b'/);
  });

  it("closes the merged consumer if ANY member is closed", () => {
    rejects(
      { type: "object", properties: { a: {} } },
      { allOf: [{ type: "object", properties: { a: {} } }, { additionalProperties: false }] },
      /forbids additional properties/,
    );
  });

  it("flattens the PRODUCER side too, and recursively through nested allOf", () => {
    accepts({ allOf: [{ type: "object", properties: { a: { type: "string" } } }, { required: ["a"] }] }, { type: "object", required: ["a"] });
    accepts({ allOf: [{ allOf: [{ type: "string" }] }] }, { type: "string" });
  });

  it("resolves $ref members while flattening — the schema-EVOLUTION case", () => {
    const resolve: Resolve = (id) => (id === "v1" ? { type: "object", properties: { a: { type: "string" } }, required: ["a"] } : undefined);
    accepts({ allOf: [{ $ref: "v1" }, { properties: { b: { type: "number" } }, required: ["b"] }] }, { $ref: "v1" }, resolve);
  });
});

describe("$ref resolution", () => {
  const resolve: Resolve = (id) => (id === "S" ? { type: "string" } : undefined);

  it("dereferences either side before comparing", () => {
    accepts({ $ref: "S" }, { type: "string" }, resolve);
    accepts({ type: "string" }, { $ref: "S" }, resolve);
    rejects({ type: "number" }, { $ref: "S" }, /'number' not allowed by consumer string/, resolve);
  });

  it("rejects a consumer $ref with no resolver, rather than reading it as universal", () => {
    rejects({ type: "string" }, { $ref: "S" }, /could not be resolved/);
  });

  it("rejects a consumer $ref the resolver does not know", () => {
    rejects({ type: "string" }, { $ref: "nope" }, /could not be resolved/, resolve);
  });

  it("does not silently accept an unresolvable PRODUCER $ref against a constrained slot", () => {
    rejects({ $ref: "nope" }, { type: "string" }, /producer declares none/, resolve);
  });
});

describe("conservatism: what the checker refuses to reason about", () => {
  it.each(["anyOf", "oneOf", "not", "if", "then", "else"])("rejects '%s' on either side", (kw) => {
    rejects({ [kw]: [{ type: "string" }] }, { type: "string" }, new RegExp("producer uses '" + kw + "'"));
    rejects({ type: "string" }, { [kw]: [{ type: "string" }] }, new RegExp("consumer uses '" + kw + "'"));
  });

  it("rejects an unmodeled CONSUMER keyword but ignores it on the producer", () => {
    rejects({ type: "string" }, { type: "string", pattern: "^a" }, /unmodeled keyword\(s\): pattern/);
    accepts({ type: "string", pattern: "^a" }, { type: "string" });
  });

  it("names EVERY unmodeled keyword, not just the first", () => {
    rejects({ type: "string" }, { type: "string", pattern: "^a", multipleOf: 2 }, /pattern, multipleOf/);
  });

  it("ignores annotation keywords, which constrain nothing", () => {
    accepts({ type: "string" }, { type: "string", format: "email", title: "T", description: "d", default: "x", examples: ["y"] });
    // These carry KIND information (`kindFor` reads them to decide `blob`), not a constraint.
    accepts({ type: "string" }, { type: "string", contentEncoding: "base64", contentMediaType: "image/png" });
  });
});

describe("`x-type` — the one extension keyword with semantics", () => {
  it("accepts only a matching type name", () => {
    accepts({ type: "number", "x-type": "DateTime" }, { type: "number", "x-type": "DateTime" });
    rejects({ type: "number", "x-type": "Money" }, { type: "number", "x-type": "DateTime" }, /'Money' does not match consumer type 'DateTime'/);
  });

  it("rejects BOTH one-sided directions — each hands over the wrong form", () => {
    rejects({ type: "number" }, { type: "number", "x-type": "DateTime" }, /producer declares no x-type/);
    rejects({ type: "number", "x-type": "DateTime" }, { type: "number" }, /consumer expects the raw encoded form/);
  });

  it("applies even against an otherwise-universal consumer — it is checked BEFORE the universal short-circuit", () => {
    rejects({ type: "number", "x-type": "DateTime" }, {}, /consumer expects the raw encoded form/);
  });

  it("still ignores every OTHER x- keyword on either side", () => {
    accepts({ type: "string", "x-ui-widget": "textarea" }, { type: "string", "x-owner": "ops", "x-team": "core" });
  });
});

describe("the universal consumer", () => {
  it("accepts anything once no constraining keyword is present", () => {
    accepts({ type: "object", properties: { a: {} }, required: ["a"] }, {});
    accepts({ type: "array", items: { type: "string" } }, { title: "anything", description: "goes" });
  });
});

describe("cycles terminate under every shape", () => {
  const graph: Record<string, Schema> = {
    A: { type: "object", properties: { b: { $ref: "B" } }, required: ["b"] },
    B: { type: "object", properties: { a: { $ref: "A" } }, required: ["a"] },
    Self: { type: "object", properties: { next: { $ref: "Self" }, v: { type: "string" } }, required: ["v"] },
  };
  const resolve: Resolve = (id) => graph[id];

  it("handles MUTUAL recursion", () => {
    accepts({ $ref: "A" }, { $ref: "A" }, resolve);
  });

  it("handles direct self-reference", () => {
    accepts({ $ref: "Self" }, { $ref: "Self" }, resolve);
  });

  it("still REJECTS a real mismatch reached THROUGH a cycle", () => {
    // The guard must suppress re-proving a pair, never suppress a disagreement.
    const producer: Resolve = (id) => (id === "Self" ? { type: "object", properties: { next: { $ref: "Self" }, v: { type: "number" } }, required: ["v"] } : graph[id]);
    const r = isSubschema({ $ref: "Self" }, { $ref: "SelfSup" }, (id) => (id === "SelfSup" ? graph["Self"] : producer(id)));
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/property 'v'/);
  });
});

describe("literal (const-bearing) producers, exhaustively", () => {
  it("checks every primitive against the consumer's type", () => {
    accepts({ type: "null", const: null }, { type: "null" });
    accepts({ type: "boolean", const: true }, { type: "boolean" });
    accepts({ type: "integer", const: 3 }, { type: "number" });
    rejects({ type: "boolean", const: true }, { type: "string" }, /value is boolean, not allowed by consumer string/);
    rejects({ type: "number", const: 1.5 }, { type: "integer" }, /value is number, not allowed by consumer integer/);
  });

  it("reports the PATH of a nested violation", () => {
    rejects(
      { type: "object", const: { a: { b: 5 } } },
      { type: "object", properties: { a: { type: "object", properties: { b: { type: "string" } } } } },
      /a\.b: value is integer/,
    );
    rejects({ type: "array", const: ["ok", 2] }, { type: "array", items: { type: "string" } }, /\[1\]: value is integer/);
  });

  it("satisfies an items constraint vacuously for an empty array", () => {
    accepts({ type: "array", const: [] }, { type: "array", items: { type: "string" } });
  });

  it("recurses through nested containers", () => {
    accepts({ type: "array", const: [[1, 2]] }, { type: "array", items: { type: "array", items: { type: "number" } } });
    rejects({ type: "array", const: [["x"]] }, { type: "array", items: { type: "array", items: { type: "number" } } }, /\[0\]\[0\]/);
  });

  it("skips consumer properties the literal does not have, unless required", () => {
    accepts({ type: "object", const: { a: 1 } }, { type: "object", properties: { a: { type: "number" }, b: { type: "string" } } });
    rejects(
      { type: "object", const: { a: 1 } },
      { type: "object", properties: { b: { type: "string" } }, required: ["b"] },
      /requires property 'b' but the literal doesn't have it/,
    );
  });

  it("is still bound by the x-type rule — a bare literal cannot fill a named-type slot", () => {
    rejects({ type: "number", const: 5 }, { type: "number", "x-type": "DateTime" }, /producer declares no x-type/);
  });

  it("is still bound by conservatism — an unmodeled consumer keyword rejects before the value is read", () => {
    rejects({ type: "string", const: "abc" }, { type: "string", pattern: "^a" }, /unmodeled keyword/);
  });

  it("resolves a $ref consumer before checking the value", () => {
    const resolve: Resolve = (id) => (id === "S" ? { type: "string", maxLength: 2 } : undefined);
    accepts({ type: "string", const: "ab" }, { $ref: "S" }, resolve);
    rejects({ type: "string", const: "abcd" }, { $ref: "S" }, /longer than consumer maxLength 2/, resolve);
  });
});
