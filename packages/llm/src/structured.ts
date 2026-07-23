/**
 * Structured-output machinery (§5.1) — the model-independent pieces ported from
 * alitheia `llm-service.ts`, adapted in two ways:
 *  1. **Non-mutating.** `patchSchemaForAnthropic` deep-clones first, so the ORIGINAL
 *     schema survives — we send the *adapted* schema to the model and reconstruct the
 *     result against the *original* (alitheia mutated in place, which is unsafe here).
 *  2. No DB/endpoint coupling — these are pure JSON-Schema transforms.
 *
 * The patch only exists to coax the model; correctness is enforced on the way out by
 * `reconstructOutput` + the §4 Ajv check (wired in Milestone 4). This complements,
 * never replaces, that validation.
 */

import type { JsonValue, MutableSchema } from "@declarative-ai/json";

/** The one schema-document vocabulary, in its mutable transform form (API.md, "The JSON vocabulary"). */
type SchemaNode = MutableSchema;

/** Normalize a stored schema value (unwrap double-serialized strings, infer object type). The `unknown`
 *  input is the sanctioned boundary position: this IS the parse step for a stored/wire schema. */
export function parseOutputSchema(raw: unknown): SchemaNode | null {
  if (raw == null) return null;
  let schema: unknown = raw;
  if (typeof schema === "string") {
    try {
      schema = JSON.parse(schema);
    } catch {
      return null;
    }
  }
  if (typeof schema !== "object" || Array.isArray(schema)) return null;
  const s = { ...(schema as SchemaNode) };
  if (s.properties && !s.type) s.type = "object";
  const hasStructure =
    s.type || s.properties || s.items || s.oneOf || s.anyOf || s.allOf || s.enum;
  if (!hasStructure) return null;
  return s;
}

/**
 * JSON Schema keywords Anthropic's structured output does not support; leaving them in
 * can confuse the constrained decoder into serializing complex values as strings.
 */
const UNSUPPORTED_KEYWORDS = [
  "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
  "minLength", "maxLength", "pattern", "format",
  "minItems", "maxItems", "uniqueItems",
  "minProperties", "maxProperties",
  "$schema", "$id", "$comment",
  // `$type` is a UI-only type-identity tag (the schema→component registry key, like `$param`); it is
  // not a validation keyword — strip it before the constrained decoder sees the schema.
  "$type",
];

/**
 * Adapt a JSON Schema for Anthropic structured output (§5.1), returning a NEW schema:
 *  1. every object gets `additionalProperties: false`;
 *  2. `oneOf`/`anyOf`/`allOf` are flattened into one object merging all variant props
 *     (the model fills every field; `reconstructOutput` strips the irrelevant ones);
 *  3. `type: [...]` collapses to its non-null member;
 *  4. unsupported keywords are stripped.
 */
export function patchSchemaForAnthropic(schema: JsonValue): JsonValue {
  return patchInPlace(structuredClone(schema));
}

function patchInPlace(schema: JsonValue): JsonValue {
  if (typeof schema !== "object" || schema === null || Array.isArray(schema)) return schema;
  const s = schema as SchemaNode;

  for (const kw of UNSUPPORTED_KEYWORDS) delete s[kw];

  for (const kw of ["oneOf", "anyOf", "allOf"] as const) {
    if (Array.isArray(s[kw])) {
      const variants = s[kw] as SchemaNode[];
      if (variants.every((v) => v.type === "object" || v.properties)) {
        const mergedProps: SchemaNode = {};
        const requiredSets = variants.map(
          (v) => new Set(Array.isArray(v.required) ? (v.required as string[]) : []),
        );
        const allKeys = new Set<string>();
        for (const v of variants) {
          if (v.properties && typeof v.properties === "object") {
            const props = v.properties as SchemaNode;
            for (const key of Object.keys(props)) {
              allKeys.add(key);
              const incomingValue = props[key];
              if (incomingValue === undefined) continue;
              if (!(key in mergedProps)) {
                mergedProps[key] = incomingValue;
              } else {
                const existing = mergedProps[key] as SchemaNode;
                const incoming = incomingValue as SchemaNode;
                if (existing.const !== undefined && incoming.const !== undefined) {
                  const vals = new Set<JsonValue>();
                  if (Array.isArray(existing.enum)) existing.enum.forEach((x) => vals.add(x));
                  else vals.add(existing.const);
                  vals.add(incoming.const);
                  mergedProps[key] = { type: "string", enum: [...vals] };
                } else if (Array.isArray(existing.enum) && incoming.const !== undefined) {
                  existing.enum.push(incoming.const);
                }
              }
            }
          }
        }
        const anyRequired = [...allKeys].filter((k) => requiredSets.some((rs) => rs.has(k)));
        delete s[kw];
        s.type = "object";
        s.properties = mergedProps;
        if (anyRequired.length > 0) s.required = anyRequired;
        else delete s.required;
      } else {
        const nonNull = variants.find((v) => v.type !== "null");
        if (nonNull) {
          delete s[kw];
          Object.assign(s, nonNull);
        }
      }
    }
  }

  if (Array.isArray(s.type)) {
    const types = s.type as string[];
    const collapsed = types.find((t) => t !== "null") ?? types[0];
    if (collapsed !== undefined) s.type = collapsed;
  }

  if (s.items !== undefined) s.items = patchInPlace(s.items);
  if (s.properties && typeof s.properties === "object") {
    const props = s.properties as SchemaNode;
    for (const key of Object.keys(props)) {
      const prop = props[key];
      if (prop !== undefined) props[key] = patchInPlace(prop);
    }
  }
  if (typeof s.additionalProperties === "object" && s.additionalProperties !== null) {
    s.additionalProperties = patchInPlace(s.additionalProperties);
  }
  if (s.type === "object" && !("additionalProperties" in s)) {
    s.additionalProperties = false;
  }
  return s;
}

/**
 * Discriminator field(s) of a oneOf/anyOf variant set: properties whose `const`
 * distinguishes the variants. Returns a single key when one suffices, a minimal
 * combination otherwise, or `[]` when none discriminates.
 */
export function findDiscriminators(variants: SchemaNode[]): string[] {
  const candidates = new Map<string, unknown[]>();
  for (const v of variants) {
    const props = v.properties as Record<string, SchemaNode> | undefined;
    if (!props) return [];
    for (const [key, prop] of Object.entries(props)) {
      if ("const" in prop) {
        if (!candidates.has(key)) candidates.set(key, []);
        candidates.get(key)!.push(prop.const);
      }
    }
  }
  const perVariantKeys = [...candidates.entries()].filter(
    ([, values]) => values.length === variants.length,
  );
  for (const [key, values] of perVariantKeys) {
    if (new Set(values).size === variants.length) return [key];
  }
  const keys = perVariantKeys.map(([k]) => k);
  if (keys.length === 0) return [];
  const signatures = variants.map((v) => {
    const props = v.properties as Record<string, SchemaNode>;
    return keys.map((k) => JSON.stringify(props[k]?.const)).join("\0");
  });
  if (new Set(signatures).size === variants.length) return keys;
  return [];
}

/**
 * Restore the model's flattened output to the ORIGINAL schema's shape: when a oneOf
 * was flattened, the model returned all variants' properties at once, so pick the
 * matched variant by its discriminator(s) and strip the rest. A no-op for schemas
 * without a union.
 */
export function reconstructOutput(output: JsonValue, schema: SchemaNode | null): JsonValue {
  if (output == null || schema == null) return output;
  if (typeof output !== "object") return output;

  for (const kw of ["oneOf", "anyOf"] as const) {
    if (Array.isArray(schema[kw])) {
      const variants = schema[kw] as SchemaNode[];
      const discriminators = findDiscriminators(variants);
      if (discriminators.length === 0 || Array.isArray(output)) break;
      const obj = output as Record<string, JsonValue>;
      if (discriminators.some((d) => obj[d] === undefined)) break;
      const matched = variants.find((v) => {
        const props = v.properties as Record<string, SchemaNode> | undefined;
        if (!props) return false;
        return discriminators.every((d) => props[d]?.const === obj[d]);
      });
      if (!matched) break;
      const props = (matched.properties as Record<string, SchemaNode>) ?? {};
      const trimmed: Record<string, JsonValue> = {};
      for (const key of Object.keys(props)) {
        const v = obj[key];
        if (v !== undefined) trimmed[key] = reconstructOutput(v, props[key] ?? null);
      }
      return trimmed;
    }
  }

  if (Array.isArray(output) && schema.items && typeof schema.items === "object") {
    const itemSchema = schema.items as SchemaNode;
    return output.map((item) => reconstructOutput(item, itemSchema));
  }

  if (!Array.isArray(output) && schema.properties && typeof schema.properties === "object") {
    const obj = output as Record<string, JsonValue>;
    const props = schema.properties as Record<string, SchemaNode>;
    const result: Record<string, JsonValue> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = props[key] ? reconstructOutput(value, props[key]) : value;
    }
    return result;
  }

  return output;
}
