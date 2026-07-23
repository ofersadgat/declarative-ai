import { describe, expect, it } from "vitest";
import type { SchemaDocument } from "@declarative-ai/json";
import { adaptSchema, __internal } from "../../src/schema/adapt";
import {
  ADVISORY,
  ANTHROPIC_AI_SDK,
  ANTHROPIC_RAW,
  JSON_OBJECT,
  OPENAI_STRICT,
} from "../../src/schema/profiles";
import type { ProviderSchemaProfile, SchemaNode } from "../../src/schema/profile";

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- test ergonomics on dynamic schemas.
type Any = any;

// Depth-mechanism tests pin an explicit small `maxDepth` (5) so they're decoupled from OPENAI_STRICT's
// real production cap (11). OPENAI_STRICT now defaults to "flatten-or-adapt"; the pure ADVISORY-fallback
// tests also pin the strategy to "adapt".
const ADAPT_PROFILE: ProviderSchemaProfile = { ...OPENAI_STRICT, maxDepth: 5, maxDepthCountStrategy: "all", maxDepthStrategy: "adapt" };

// `k` nested objects terminating in a string leaf — the worst case for depth (every level counts).
const objChain = (k: number): SchemaNode => {
  let node: SchemaNode = { type: "object", properties: { f: { type: "string" } }, required: ["f"], additionalProperties: false };
  for (let i = 1; i < k; i++) node = { type: "object", properties: { f: node }, required: ["f"], additionalProperties: false };
  return node;
};

describe("adaptSchema — meta/UI tags (every profile)", () => {
  it("strips $schema/$id/$comment/$type/$param everywhere", () => {
    const original = {
      $schema: "https://json-schema.org/draft-07",
      $type: "Widget",
      type: "object",
      properties: { a: { type: "string", $param: "T", $comment: "x" } },
      required: ["a"],
    };
    const { outgoing } = adaptSchema(original, ADVISORY) as { outgoing: Any };
    expect(outgoing.$schema).toBeUndefined();
    expect(outgoing.$type).toBeUndefined();
    expect(outgoing.properties.a.$param).toBeUndefined();
    expect(outgoing.properties.a.$comment).toBeUndefined();
    expect(outgoing.properties.a.type).toBe("string");
  });

  it("does not mutate the original", () => {
    const original = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    const snap = structuredClone(original);
    adaptSchema(original, OPENAI_STRICT);
    expect(original).toEqual(snap);
  });
});

describe("adaptSchema — OPENAI_STRICT", () => {
  it("forces additionalProperties:false and makes optionals nullable + all-required", () => {
    const { outgoing, enforce } = adaptSchema(
      { type: "object", properties: { name: { type: "string" }, age: { type: "integer" } }, required: ["name"] },
      OPENAI_STRICT,
    ) as { outgoing: Any; enforce: string };
    expect(enforce).toBe("strict");
    expect(outgoing.additionalProperties).toBe(false);
    expect(outgoing.required.sort()).toEqual(["age", "name"]); // forced all-required
    expect(outgoing.properties.name.type).toBe("string"); // required stays plain
    expect(outgoing.properties.age.type).toEqual(["integer", "null"]); // optional → nullable
  });

  it("postProcess drops the nulls we forced for originally-optional fields", () => {
    const { postProcess } = adaptSchema(
      { type: "object", properties: { name: { type: "string" }, age: { type: "integer" } }, required: ["name"] },
      OPENAI_STRICT,
    );
    expect(postProcess({ name: "x", age: null })).toEqual({ name: "x" });
    expect(postProcess({ name: "x", age: 5 })).toEqual({ name: "x", age: 5 });
  });

  it("flattens a discriminated oneOf and round-trips it back through postProcess", () => {
    const original: SchemaDocument = {
      oneOf: [
        { type: "object", properties: { kind: { const: "a" }, av: { type: "number" } }, required: ["kind", "av"] },
        { type: "object", properties: { kind: { const: "b" }, bv: { type: "string" } }, required: ["kind", "bv"] },
      ],
    };
    const { outgoing, enforce, postProcess } = adaptSchema(original, OPENAI_STRICT) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.type).toBe("object");
    expect(outgoing.oneOf).toBeUndefined();
    expect(Object.keys(outgoing.properties).sort()).toEqual(["av", "bv", "kind"]);
    expect(outgoing.additionalProperties).toBe(false);
    // The model, coaxed by the flattened schema, returns both variants' props; reconstruct picks one.
    expect(postProcess({ kind: "a", av: 1, bv: "leaked" })).toEqual({ kind: "a", av: 1 });
    expect(postProcess({ kind: "b", av: 9, bv: "y" })).toEqual({ kind: "b", bv: "y" });
  });

  it("encodes an any/untyped node as a JSON string and decodes it on the way out", () => {
    const original = { type: "object", properties: { payload: {} }, required: ["payload"] };
    const { outgoing, enforce, postProcess } = adaptSchema(original, OPENAI_STRICT) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.properties.payload.type).toBe("string"); // {} → string field
    expect(postProcess({ payload: '{"a":1,"b":[2]}' })).toEqual({ payload: { a: 1, b: [2] } });
    // A non-JSON string survives as-is (best effort).
    expect(postProcess({ payload: "plain" })).toEqual({ payload: "plain" });
  });

  it("keeps OpenAI-supported keywords and a whitelisted format, strips unsupported ones", () => {
    const { outgoing } = adaptSchema(
      {
        type: "object",
        properties: {
          s: { type: "string", minLength: 1, pattern: "^x", format: "uuid" },
          bad: { type: "string", format: "phone" },
          n: { type: "number", minimum: 0, multipleOf: 2 },
          arr: { type: "array", items: { type: "string" }, uniqueItems: true },
        },
        required: ["s", "bad", "n", "arr"],
      },
      OPENAI_STRICT,
    ) as { outgoing: Any };
    expect(outgoing.properties.s.minLength).toBe(1); // supported
    expect(outgoing.properties.s.pattern).toBe("^x"); // supported
    expect(outgoing.properties.s.format).toBe("uuid"); // whitelisted → kept
    expect(outgoing.properties.bad.format).toBeUndefined(); // not whitelisted → stripped
    expect(outgoing.properties.n.minimum).toBe(0);
    expect(outgoing.properties.n.multipleOf).toBe(2);
    expect(outgoing.properties.arr.uniqueItems).toBeUndefined(); // unsupported → stripped
  });
});

describe("adaptSchema — fitsStrict fallback to advisory", () => {
  const deep = (n: number): SchemaNode => {
    let node: SchemaNode = { type: "object", properties: { leaf: { type: "string" } }, required: ["leaf"] };
    for (let i = 0; i < n; i++) node = { type: "object", properties: { child: node }, required: ["child"] };
    return node;
  };

  it("falls back to advisory when nesting depth exceeds the cap", () => {
    const { enforce, notes, postProcess } = adaptSchema(deep(6), ADAPT_PROFILE);
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("depth-exceeded");
    // Advisory ⇒ identity post-process (no lossy transform was applied).
    expect(postProcess({ child: 1 })).toEqual({ child: 1 });
  });

  it("falls back to advisory when property count exceeds the limit", () => {
    const profile: ProviderSchemaProfile = { ...OPENAI_STRICT, limits: { maxProperties: 1 } };
    const { enforce, notes } = adaptSchema(
      { type: "object", properties: { a: { type: "string" }, b: { type: "string" } }, required: ["a", "b"] },
      profile,
    );
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("properties-exceeded");
  });

  it("the advisory outgoing schema is the meta-stripped original, untransformed", () => {
    const { outgoing } = adaptSchema(deep(6), ADAPT_PROFILE) as { outgoing: Any };
    // No additionalProperties injected (we discarded the strict transform).
    expect(outgoing.additionalProperties).toBeUndefined();
  });

  it("does not leak the discarded strict transform into NESTED advisory nodes (no in-place mutation)", () => {
    // Regression: `strictify` mutated its input in place (the top-level spread didn't deep-clone
    // `properties`/`items`), so the advisory fallback's `outgoing: base` actually shipped the lossy
    // STRICTIFIED schema beneath the surface — all-required, additionalProperties:false, and `{}`
    // rewritten to a JSON-encoded string. The top-level node escaped (its policies land on the returned
    // copy), which is why the shallow check above missed it; the corruption was one level down. This is
    // exactly what broke generate.dataset's self-bootstrap (depth 7 → advisory): the model was
    // handed a schema describing the nested `schema` holes as strings and forcing every field present.
    const leaf: SchemaNode = {
      type: "object",
      properties: { payload: {}, note: { type: "string" } }, // payload = any-hole; note = OPTIONAL
      required: ["payload"],
    };
    let node: SchemaNode = leaf;
    for (let i = 0; i < 6; i++) node = { type: "object", properties: { child: node }, required: ["child"] };

    const { outgoing, enforce } = adaptSchema(node, ADAPT_PROFILE) as { outgoing: Any; enforce: string };
    expect(enforce).toBe("advisory");

    let cur: Any = outgoing;
    for (let i = 0; i < 6; i++) cur = cur.properties.child;
    expect(cur.properties.payload).toEqual({}); // any-hole stays {}, NOT encoded to a string
    expect(cur.required).toEqual(["payload"]); // optional `note` NOT forced required
    expect(cur.additionalProperties).toBeUndefined(); // no closed-object policy injected
  });
});

describe("adaptSchema — ANTHROPIC_AI_SDK (high-capability profile, near-identity transform)", () => {
  it("leaves the schema for the SDK and goes strict for a plain schema (identity post-process)", () => {
    const original = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    const { outgoing, enforce, postProcess } = adaptSchema(original, ANTHROPIC_AI_SDK) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.additionalProperties).toBeUndefined(); // SDK adds it, not us
    expect(postProcess({ a: "x" })).toEqual({ a: "x" });
  });

  it("falls back to advisory on an any node (Anthropic can't represent {})", () => {
    const { enforce, notes } = adaptSchema(
      { type: "object", properties: { x: {} }, required: ["x"] },
      ANTHROPIC_AI_SDK,
    );
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("any-not-representable");
  });

  it("falls back to advisory on a root union (Anthropic rejects top-level anyOf)", () => {
    const { enforce, notes } = adaptSchema(
      { oneOf: [{ type: "object", properties: { a: { type: "string" } } }, { type: "object", properties: { b: { type: "string" } } }] },
      ANTHROPIC_AI_SDK,
    );
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("root-union-unsupported");
  });
});

describe("adaptSchema — ANTHROPIC_RAW (describe strategy)", () => {
  it("moves unsupported numeric/string constraints into description and keeps the field", () => {
    const { outgoing } = adaptSchema(
      { type: "object", properties: { n: { type: "number", minimum: 5, description: "count" } }, required: ["n"] },
      ANTHROPIC_RAW,
    ) as { outgoing: Any };
    expect(outgoing.properties.n.minimum).toBeUndefined();
    expect(outgoing.properties.n.description).toContain("minimum: 5");
    expect(outgoing.properties.n.description).toContain("count");
  });

  it("whitelists minItems to {0,1} and strips an out-of-range value", () => {
    const { outgoing } = adaptSchema(
      {
        type: "object",
        properties: { ok: { type: "array", items: { type: "string" }, minItems: 1 }, bad: { type: "array", items: { type: "string" }, minItems: 3 } },
        required: ["ok", "bad"],
      },
      ANTHROPIC_RAW,
    ) as { outgoing: Any };
    expect(outgoing.properties.ok.minItems).toBe(1); // allowed
    expect(outgoing.properties.bad.minItems).toBeUndefined(); // out of {0,1} → described+stripped
    expect(outgoing.properties.bad.description).toContain("minItems: 3");
  });
});

describe("adaptSchema — root-array wrap (object-root-only transports)", () => {
  const ROWS = {
    type: "array",
    title: "generated-dataset",
    items: { type: "object", properties: { input: { type: "string" }, output: { type: "string" } }, required: ["input", "output"] },
  };

  it("wraps a root array under { items } for OPENAI_STRICT and unwraps it on the way out", () => {
    const { outgoing, enforce, postProcess, notes } = adaptSchema(ROWS, OPENAI_STRICT) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.type).toBe("object");
    expect(outgoing.required).toEqual(["items"]);
    expect(outgoing.properties.items.type).toBe("array");
    expect(outgoing.title).toBe("generated-dataset"); // annotation lifted onto the wrapper
    expect(outgoing.additionalProperties).toBe(false); // object policies apply to the wrapper too
    expect(notes.map((n: Any) => n.code)).toContain("root-array-wrapped");
    // The model answers in the wrapped shape; postProcess returns the bare array.
    expect(postProcess({ items: [{ input: "a", output: "b" }] })).toEqual([{ input: "a", output: "b" }]);
  });

  it("the unwrap is lenient: a bare-array answer passes through unchanged", () => {
    const { postProcess } = adaptSchema(ROWS, OPENAI_STRICT);
    expect(postProcess([{ input: "a", output: "b" }])).toEqual([{ input: "a", output: "b" }]);
  });

  it("inner lossy transforms still reverse beneath the wrapper (nullable-optional drop)", () => {
    const rows = {
      type: "array",
      items: { type: "object", properties: { input: { type: "string" }, output: { type: "string" } }, required: ["input"] },
    };
    const { outgoing, postProcess } = adaptSchema(rows, OPENAI_STRICT) as Any;
    expect(outgoing.properties.items.items.properties.output.type).toEqual(["string", "null"]); // optional → nullable
    expect(postProcess({ items: [{ input: "a", output: null }] })).toEqual([{ input: "a" }]); // forced null dropped
  });

  it("wraps on the advisory fallback too (the dialect's JSON modes only emit objects)", () => {
    const deepRow = (n: number): SchemaNode => {
      let node: SchemaNode = { type: "object", properties: { leaf: { type: "string" } }, required: ["leaf"] };
      for (let i = 0; i < n; i++) node = { type: "object", properties: { child: node }, required: ["child"] };
      return node;
    };
    const { outgoing, enforce, postProcess, notes } = adaptSchema({ type: "array", items: deepRow(6) }, ADAPT_PROFILE) as Any;
    expect(enforce).toBe("advisory");
    expect(notes.map((n: Any) => n.code)).toContain("depth-exceeded");
    expect(outgoing.type).toBe("object"); // wrapped hint
    expect(outgoing.properties.items.type).toBe("array");
    expect(postProcess({ items: [1, 2] })).toEqual([1, 2]); // still unwrapped on the way back
  });

  it("ANTHROPIC_AI_SDK also wraps (tool input_schema requires an object root)", () => {
    const { outgoing, enforce, postProcess } = adaptSchema(ROWS, ANTHROPIC_AI_SDK) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.type).toBe("object");
    expect(postProcess({ items: [] })).toEqual([]);
  });

  it("a root-array-capable profile leaves the array alone", () => {
    const profile: ProviderSchemaProfile = { ...OPENAI_STRICT, rootArray: true };
    const { outgoing, enforce, postProcess } = adaptSchema(ROWS, profile) as Any;
    expect(enforce).toBe("strict");
    expect(outgoing.type).toBe("array");
    expect(postProcess([{ input: "a", output: "b" }])).toEqual([{ input: "a", output: "b" }]);
  });

  it("ADVISORY (text tier) keeps the true root-array shape as the hint (no decoder in play)", () => {
    const { outgoing, enforce, postProcess } = adaptSchema(ROWS, ADVISORY) as Any;
    expect(enforce).toBe("text"); // supportsStructuredOutput:false → plain-text tier
    expect(outgoing.type).toBe("array");
    expect(postProcess([1])).toEqual([1]);
  });

  it("JSON_OBJECT (object tier) wraps the root array (json_object emits an object root)", () => {
    const { outgoing, enforce, postProcess } = adaptSchema(ROWS, JSON_OBJECT) as Any;
    expect(enforce).toBe("advisory"); // json_object mode — schema is a hint, never strict
    expect(outgoing.type).toBe("object"); // rootArray:false → wrapped under { items }
    expect(postProcess({ items: [1] })).toEqual([1]); // unwrap on the way back
  });

  it("an object root is untouched by the wrap logic", () => {
    const original = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };
    const { outgoing, notes } = adaptSchema(original, OPENAI_STRICT) as Any;
    expect(outgoing.type).toBe("object");
    expect(outgoing.properties.items).toBeUndefined();
    expect(notes.map((n: Any) => n.code)).not.toContain("root-array-wrapped");
  });
});

describe("collectStats — nesting-depth metric (drives the maxDepth gate)", () => {
  const depth = (schema: SchemaNode): number => __internal.collectStats(schema).maxDepth;

  it("a primitive or empty node is depth 1; a flat object is depth 2 (the leaf level counts)", () => {
    expect(depth({ type: "string" })).toBe(1);
    expect(depth({})).toBe(1); // an `any` node
    expect(depth({ type: "object", properties: { a: { type: "string" }, b: { type: "number" } }, required: ["a", "b"] })).toBe(2);
  });

  it("each nested object adds exactly one level — a chain of k objects measures k+1", () => {
    expect(depth(objChain(1))).toBe(2);
    expect(depth(objChain(3))).toBe(4);
    expect(depth(objChain(10))).toBe(11);
    expect(depth(objChain(11))).toBe(12);
  });

  it("arrays count as a level too (items recursion) — so our depth runs ≥ OpenAI's for array-heavy schemas", () => {
    expect(depth({ type: "array", items: { type: "string" } })).toBe(2); // array → leaf
    expect(depth({ type: "array", items: { type: "object", properties: { a: { type: "string" } }, required: ["a"] } })).toBe(3); // array → item → leaf
    // object → array → object → leaf, alternating
    expect(depth({ type: "object", properties: { a: { type: "array", items: { type: "object", properties: { b: { type: "string" } }, required: ["b"] } } }, required: ["a"] })).toBe(4);
  });

  it("a schema-valued additionalProperties counts as a level", () => {
    expect(depth({ type: "object", additionalProperties: { type: "object", properties: { a: { type: "string" } }, required: ["a"] } })).toBe(3);
  });

  it("union keywords (anyOf/oneOf/allOf) recurse at the SAME depth — a union is not itself a level", () => {
    const u: SchemaNode = {
      type: "object",
      properties: { u: { anyOf: [{ type: "object", properties: { a: { type: "string" } }, required: ["a"] }, { type: "string" }] } },
      required: ["u"],
    };
    expect(depth(u)).toBe(3); // object(1) → u(2) → the variant's `a`(3); the anyOf adds no level
  });

  it('"all" is the conservative default: it counts objects AND arrays AND the leaf', () => {
    // The default never under-counts a provider's "levels". 20 nested arrays around a leaf is depth 22
    // here (a 1-object schema to OpenAI) — which is why this default would needlessly fall back, and why
    // OpenAI's profile opts into "objects" instead.
    let arrDeep: SchemaNode = { type: "string" };
    for (let i = 0; i < 20; i++) arrDeep = { type: "array", items: arrDeep };
    expect(depth({ type: "object", properties: { a: arrDeep }, required: ["a"] })).toBe(22);
  });

  it('"objects" counts only nested objects (OpenAI\'s rule) — arrays and the leaf are FREE', () => {
    const objects = (s: SchemaNode): number => __internal.collectStats(s, "objects").maxDepth;
    expect(objects(objChain(10))).toBe(10); // 10 nested objects, exactly OpenAI's "10 levels"
    expect(objects(objChain(11))).toBe(11);
    // a 20-deep array chain is ONE object-level (the matching root object), not 21.
    let arrDeep: SchemaNode = { type: "string" };
    for (let i = 0; i < 20; i++) arrDeep = { type: "array", items: arrDeep };
    expect(objects({ type: "object", properties: { a: arrDeep }, required: ["a"] })).toBe(1);
    // array-of-object: only the objects count (root + item), the array between them is free.
    expect(objects({ type: "object", properties: { rows: { type: "array", items: { type: "object", properties: { x: { type: "string" } }, required: ["x"] } } }, required: ["rows"] })).toBe(2);
  });

  it('"containers" counts objects AND arrays, but not the leaf', () => {
    const containers = (s: SchemaNode): number => __internal.collectStats(s, "containers").maxDepth;
    expect(containers(objChain(3))).toBe(3); // 3 objects, no arrays
    expect(containers({ type: "array", items: { type: "object", properties: { a: { type: "string" } }, required: ["a"] } })).toBe(2); // array + object, leaf free
  });
});

describe("maxDepth gate — the metric drives the strict/advisory boundary at OpenAI's real cap (10 objects)", () => {
  // No-flatten so the depth gate itself is observable (flatten would otherwise rescue an object chain).
  // Inherits OPENAI_STRICT's maxDepth 10 + maxDepthCountStrategy "objects", so the gate is EXACT.
  const NO_FLATTEN: ProviderSchemaProfile = { ...OPENAI_STRICT, maxDepthStrategy: "adapt" };

  it("a chain of 10 nested objects (exactly OpenAI's limit) fits strict", () => {
    const { enforce, notes } = adaptSchema(objChain(10), NO_FLATTEN);
    expect(enforce).toBe("strict");
    expect(notes.map((n) => n.code)).not.toContain("depth-exceeded");
  });

  it("a chain of 11 nested objects (over the limit) trips depth-exceeded", () => {
    const { enforce, notes } = adaptSchema(objChain(11), NO_FLATTEN);
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("depth-exceeded");
  });

  it("deeply nested ARRAYS no longer trip the gate — they're free under \"objects\" (the bug fix)", () => {
    // The reported pain: nesting arrays inside objects inflated the old count past the cap and forced a
    // failing advisory fallback. Under "objects" a 15-deep array chain is 1 object-level → strict.
    let arrDeep: SchemaNode = { type: "string" };
    for (let i = 0; i < 15; i++) arrDeep = { type: "array", items: arrDeep };
    const { enforce, notes } = adaptSchema({ type: "object", properties: { a: arrDeep }, required: ["a"] }, NO_FLATTEN);
    expect(enforce).toBe("strict");
    expect(notes.map((n) => n.code)).not.toContain("depth-exceeded");
  });
});

describe("adaptSchema — maxDepthStrategy", () => {
  // A fully-hoistable required object chain `n` levels deep.
  const chain = (n: number): SchemaNode => {
    let node: SchemaNode = { type: "object", properties: { leaf: { type: "string" } }, required: ["leaf"] };
    for (let i = 0; i < n; i++) node = { type: "object", properties: { child: node }, required: ["child"] };
    return node;
  };
  // Irreducible depth: array nesting can't be hoisted away.
  const arrayChain = (n: number): SchemaNode => {
    let node: SchemaNode = { type: "object", properties: { leaf: { type: "string" } }, required: ["leaf"] };
    for (let i = 0; i < n; i++) node = { type: "array", items: node };
    return node;
  };
  // Pin maxDepth + count strategy so the count-everything fixtures (`chain`/`arrayChain`) behave as
  // designed, independent of OPENAI_STRICT's production "objects"/10.
  const withStrategy = (s: ProviderSchemaProfile["maxDepthStrategy"]): ProviderSchemaProfile => ({ ...OPENAI_STRICT, maxDepth: 5, maxDepthCountStrategy: "all", maxDepthStrategy: s });
  const nested6 = { child: { child: { child: { child: { child: { child: { leaf: "x" } } } } } } };
  const flatKey = "child.child.child.child.child.child.leaf";

  it("default (undefined) preserves old behavior — advisory on over-depth", () => {
    const { enforce, notes } = adaptSchema(chain(6), withStrategy(undefined));
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("depth-exceeded");
  });

  it('"flatten-or-adapt": flattens an over-depth object chain to strict and round-trips', () => {
    const { enforce, outgoing, postProcess, notes } = adaptSchema(chain(6), withStrategy("flatten-or-adapt")) as Any;
    expect(enforce).toBe("strict");
    expect(notes.map((n: Any) => n.code)).toContain("depth-flattened");
    expect(outgoing.properties[flatKey]).toBeDefined(); // chain collapsed to one dotted key
    expect(postProcess({ [flatKey]: "x" })).toEqual(nested6); // model answers flat → restored nested
  });

  it('"flatten-or-adapt": falls back to advisory on irreducible (array) depth', () => {
    const { enforce, notes } = adaptSchema(arrayChain(7), withStrategy("flatten-or-adapt"));
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("depth-exceeded");
  });

  it('"flatten": strict on a flattenable chain, THROWS on irreducible depth', () => {
    expect(adaptSchema(chain(6), withStrategy("flatten")).enforce).toBe("strict");
    expect(() => adaptSchema(arrayChain(7), withStrategy("flatten"))).toThrow(/after flattening/);
  });

  it('"strict": forces strict at full depth (NOT flattened) and round-trips the deep shape', () => {
    const { enforce, outgoing, postProcess, notes } = adaptSchema(chain(6), withStrategy("strict")) as Any;
    expect(enforce).toBe("strict");
    expect(notes.map((n: Any) => n.code)).toContain("depth-strict-forced");
    expect(outgoing.properties.child).toBeDefined(); // still nested — depth override, not flatten
    expect(postProcess(nested6)).toEqual(nested6);
  });

  it('"strict": still advisory when a NON-depth blocker remains (can\'t force a closed grammar to hold {})', () => {
    // anyType:"native" makes the {} leaf unrepresentable, so a non-depth blocker coexists with the depth one.
    const profile: ProviderSchemaProfile = { ...OPENAI_STRICT, maxDepth: 5, maxDepthCountStrategy: "all", maxDepthStrategy: "strict", anyType: "native" };
    let node: SchemaNode = { type: "object", properties: { payload: {} }, required: ["payload"] };
    for (let i = 0; i < 6; i++) node = { type: "object", properties: { child: node }, required: ["child"] };
    const { enforce, notes } = adaptSchema(node, profile);
    expect(enforce).toBe("advisory");
    expect(notes.map((n) => n.code)).toContain("any-not-representable");
  });

  it('"error": throws when a schema is over the depth cap', () => {
    expect(() => adaptSchema(chain(6), withStrategy("error"))).toThrow(/maxDepthStrategy="error"/);
  });

  it("does not flatten a schema that already fits (no-op under the cap)", () => {
    const { enforce, notes } = adaptSchema({ type: "object", properties: { a: { type: "string" } }, required: ["a"] }, withStrategy("flatten-or-adapt"));
    expect(enforce).toBe("strict");
    expect(notes.map((n) => n.code)).not.toContain("depth-flattened");
  });

  it("flattens + wraps a deep signature-as-data dataset and round-trips end-to-end", () => {
    // A `generate.dataset`-style output: an array of rows whose `input` embeds a nested signature
    // (schema-as-data). Pinned to maxDepth 5 so the object-chain depth trips the cap and the flatten path
    // runs (under the real cap of 11 this exact schema fits natively — see the live probe).
    const sigSlot = { type: "object", properties: { kind: { type: "string" }, schema: {} }, required: ["kind"] };
    const row = {
      type: "object",
      properties: {
        input: {
          type: "object",
          properties: {
            description: { type: "string" },
            signature: { type: "object", properties: { input: sigSlot, output: sigSlot }, required: ["input", "output"] },
          },
          required: ["description", "signature"],
        },
        output: { type: "string" },
      },
      required: ["input", "output"],
    };
    const { enforce, outgoing, postProcess, notes } = adaptSchema({ type: "array", title: "generated-dataset", items: row }, withStrategy("flatten-or-adapt")) as Any;
    expect(enforce).toBe("strict");
    expect(notes.map((n: Any) => n.code).sort()).toEqual(["depth-flattened", "root-array-wrapped"]);
    expect(outgoing.type).toBe("object"); // root array wrapped under { items }
    const rowSchema = outgoing.properties.items.items;
    expect(Object.keys(rowSchema.properties)).toContain("input.signature.input.kind");
    // {} encoded as a JSON string field (nullable here since `schema` is optional in the slot).
    expect([].concat(rowSchema.properties["input.signature.input.schema"].type)).toContain("string");

    const modelAnswer = {
      items: [
        {
          "input.description": "classify sentiment",
          "input.signature.input.kind": "text",
          "input.signature.input.schema": '{"type":"string"}',
          "input.signature.output.kind": "json",
          "input.signature.output.schema": '{"type":"number"}',
          output: "positive",
        },
      ],
    };
    expect(postProcess(modelAnswer)).toEqual([
      {
        input: {
          description: "classify sentiment",
          signature: { input: { kind: "text", schema: { type: "string" } }, output: { kind: "json", schema: { type: "number" } } },
        },
        output: "positive",
      },
    ]);
  });
});

describe("adaptSchema — ADVISORY (text) floor", () => {
  it("uses the text tier and returns identity post-process", () => {
    const original = { type: "object", properties: { a: { type: "string", minLength: 2 } }, required: ["a"] };
    const { outgoing, enforce, postProcess, notes } = adaptSchema(original, ADVISORY) as Any;
    expect(enforce).toBe("text"); // supportsStructuredOutput:false → no response_format at all
    expect(notes[0].code).toBe("no-structured-output");
    expect(outgoing.properties.a.minLength).toBe(2); // untouched (schema sent as a prompt hint)
    expect(postProcess({ a: "xy" })).toEqual({ a: "xy" });
  });
});

describe("adaptSchema — JSON_OBJECT (object) tier", () => {
  it("is advisory (json_object) and leaves the schema untouched as a hint", () => {
    const original = { type: "object", properties: { a: { type: "string", minLength: 2 } }, required: ["a"] };
    const { outgoing, enforce, postProcess, notes } = adaptSchema(original, JSON_OBJECT) as Any;
    expect(enforce).toBe("advisory"); // json_object mode — never strict
    expect(notes[0].code).toBe("no-structured-output");
    expect(outgoing.properties.a.minLength).toBe(2);
    expect(postProcess({ a: "xy" })).toEqual({ a: "xy" });
  });
});
