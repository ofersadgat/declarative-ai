import { describe, expect, it } from "vitest";
import type { InlineFamily, JsonSchema, Operation } from "@declarative-ai/ops";
import { checkBinding, checkOperation, type CheckerHooks } from "../src/checker";
import { isSubschema } from "../src/subtype";

const inline: CheckerHooks<InlineFamily> = {
  producer: (ref) => (typeof ref === "string" ? undefined : ref),
};

const producerOp = (schema: JsonSchema): Operation<InlineFamily> => ({
  kind: "function",
  functionRef: "make",
  input: {},
  output: { name: "output", kind: "json", schema },
});

describe("the ONE binding checker (API.md, \"The binding checker\")", () => {
  it("accepts a producer whose output is a subtype of the consuming slot", () => {
    const op: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "consume",
      input: {
        n: {
          kind: "json",
          schema: { type: "number" },
          binding: { op: producerOp({ type: "integer" }) },
        },
      },
      output: { name: "output", kind: "json" },
    };
    expect(checkOperation(op, inline).ok).toBe(true);
  });

  it("rejects an incompatible producer with a precise reason", () => {
    const op: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "consume",
      input: { n: { kind: "json", schema: { type: "number" }, binding: { op: producerOp({ type: "string" }) } } },
      output: { name: "output", kind: "json" },
    };
    const result = checkOperation(op, inline);
    expect(result.ok).toBe(false);
    expect(result.errors[0]!.message).toMatch(/producer type 'string' not allowed/);
    expect(result.errors[0]!.path).toBe("operation.input.n");
  });

  it("types a LITERAL by its value, so it satisfies an enum-constrained consumer", () => {
    const op: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "consume",
      input: { mode: { kind: "text", schema: { type: "string", enum: ["a", "b"] }, binding: { text: "a" } } },
      output: { name: "output", kind: "json" },
    };
    expect(checkOperation(op, inline).ok).toBe(true);
  });

  it("checks KIND agreement for a higher-order slot — the op DEFINITION is the value there", () => {
    const op: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "consume",
      input: { fn: { kind: "prompt", binding: { op: producerOp({ type: "string" }) } } },
      output: { name: "output", kind: "json" },
    };
    const result = checkOperation(op, inline);
    expect(result.errors[0]!.message).toMatch(/expects a prompt op but the producer is a function op/);
  });

  it("enforces the positional `index` invariant: all-set-or-all-unset, and distinct", () => {
    const partial: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "f",
      input: { a: { kind: "json", index: 0 }, b: { kind: "json" } },
      output: { name: "output", kind: "json" },
    };
    expect(checkOperation(partial, inline).errors[0]!.message).toMatch(/all slots or none/);

    const dup: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "f",
      input: { a: { kind: "json", index: 0 }, b: { kind: "json", index: 0 } },
      output: { name: "output", kind: "json" },
    };
    expect(checkOperation(dup, inline).errors[0]!.message).toMatch(/duplicate input 'index' 0/);
  });

  it("checks signature conformance on the output side", () => {
    const op = producerOp({ type: "string" });
    const result = checkOperation(op, inline, { signature: { input: { kind: "json" }, output: { name: "output", kind: "json", schema: { type: "number" } } } });
    expect(result.errors[0]!.message).toMatch(/signature: producer type 'string'/);
  });
});

describe("x-type is CONSTRAINING — the one extension keyword with validation semantics (§3.3)", () => {
  it("accepts a producer declaring the SAME type name", () => {
    expect(isSubschema({ type: "number", "x-type": "DateTime" }, { type: "number", "x-type": "DateTime" }).ok).toBe(true);
  });

  it("REFUSES a bare-number producer for a DateTime slot — it would hand over the encoded epoch", () => {
    const r = isSubschema({ type: "number" }, { type: "number", "x-type": "DateTime" });
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/requires type 'DateTime'/);
  });

  it("REFUSES a DateTime producer for a bare-number slot — the reverse unsoundness", () => {
    const r = isSubschema({ type: "number", "x-type": "DateTime" }, { type: "number" });
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/expects the raw encoded form/);
  });

  it("REFUSES two DIFFERENT type names", () => {
    expect(isSubschema({ type: "number", "x-type": "Money" }, { type: "number", "x-type": "DateTime" }).ok).toBe(false);
  });

  it("still IGNORES every other x- keyword — they carry application metadata, not semantics", () => {
    expect(isSubschema({ type: "string", "x-ui-widget": "textarea" }, { type: "string", "x-owner": "ops" }).ok).toBe(true);
  });

  it("treats the blob content keywords as annotations, not constraints (kindFor reads them instead)", () => {
    expect(isSubschema({ type: "string" }, { type: "string", contentEncoding: "base64", contentMediaType: "image/png" }).ok).toBe(true);
  });
});

describe("the checker does not decide universality for itself (§6.2's 'never a silent pass')", () => {
  // `isUniversalSchema` lists the keywords the CHECKER can read, which is not the set `isSubschema`
  // accepts. Using it as a pre-filter meant every consumer keyword the subtype checker deliberately
  // REJECTS classified as "constrains nothing" and short-circuited to a pass — so `checkBinding` was
  // silent about exactly the schemas `isSubschema` had a precise complaint about.
  const producer = { json: 42 } as const;
  const bind = (slot: JsonSchema) => checkBinding<InlineFamily>(producer, slot, inline, "input.x");

  it.each([
    ["anyOf", { anyOf: [{ type: "string" }, { type: "boolean" }] }, /anyOf/],
    ["not", { not: { type: "number" } }, /not/],
    ["an unresolvable $ref", { $ref: "sha256:abc" }, /could not be resolved/],
    ["pattern", { pattern: "^[a-z]+$" }, /unmodeled keyword\(s\): pattern/],
    ["minItems", { minItems: 2 }, /unmodeled keyword\(s\): minItems/],
    ["uniqueItems", { uniqueItems: true }, /unmodeled keyword\(s\): uniqueItems/],
  ])("reports what isSubschema says about a %s consumer instead of passing it", (_case, slot, reason) => {
    const errors = bind(slot as JsonSchema);
    expect(errors).toHaveLength(1);
    expect(errors[0]!.message).toMatch(reason);
  });

  it("still accepts a GENUINELY universal consumer — isSubschema decides that too", () => {
    // The short-circuit is gone, not replaced by a rejection: `{}` and annotation-only documents still
    // accept anything, because `subtype.ts` answers that question itself.
    expect(bind({})).toEqual([]);
    expect(bind({ description: "anything at all", title: "x" } as JsonSchema)).toEqual([]);
  });

  it("applies the same rule to signature conformance", () => {
    const op = producerOp({ type: "string" });
    const result = checkOperation(op, inline, {
      signature: { input: { kind: "json" }, output: { name: "output", kind: "json", schema: { not: { type: "string" } } as JsonSchema } },
    });
    expect(result.ok).toBe(false);
    expect(result.errors[0]!.message).toMatch(/signature: .*not/);
  });
});

describe("inline literal bindings against constrained slots", () => {
  // These are what an author writes when they hard-code a value into a wired slot. Each was rejected
  // pre-run because the literal's derived schema does not restate the consumer's constraint.
  const bind = (binding: Parameters<typeof checkBinding<InlineFamily>>[0], slot: JsonSchema) =>
    checkBinding<InlineFamily>(binding, slot, inline, "input.x");

  it("accepts a string array into an `array of string` slot", () => {
    expect(bind({ json: ["a", "b"] }, { type: "array", items: { type: "string" } })).toEqual([]);
  });

  it("accepts an in-range number into a bounded integer slot", () => {
    expect(bind({ json: 42 }, { type: "integer", minimum: 0, maximum: 100 })).toEqual([]);
  });

  it("accepts a short string into a maxLength slot", () => {
    expect(bind({ text: "hello" }, { type: "string", maxLength: 100 })).toEqual([]);
  });

  it("accepts an exactly-shaped object into a closed slot", () => {
    expect(bind({ json: { a: 1 } }, { type: "object", properties: { a: { type: "number" } }, additionalProperties: false })).toEqual([]);
  });

  // The point is precision, not permissiveness: the same path still catches genuinely bad literals.
  it("still rejects an out-of-range number", () => {
    const errors = bind({ json: 900 }, { type: "integer", minimum: 0, maximum: 100 });
    expect(errors).toHaveLength(1);
    expect(errors[0]!.message).toMatch(/above consumer maximum/);
  });

  it("still rejects an array whose elements are the wrong type", () => {
    const errors = bind({ json: ["a", 2] }, { type: "array", items: { type: "string" } });
    expect(errors).toHaveLength(1);
    expect(errors[0]!.message).toMatch(/not allowed by consumer string/);
  });

  it("still rejects an extra key against a closed slot", () => {
    const errors = bind({ json: { a: 1, b: 2 } }, { type: "object", properties: { a: { type: "number" } }, additionalProperties: false });
    expect(errors).toHaveLength(1);
    expect(errors[0]!.message).toMatch(/'b' not permitted/);
  });
});
