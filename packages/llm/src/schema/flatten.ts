import type { JsonValue } from "@declarative-ai/json";
import type { SchemaNode } from "./profile";

/**
 * Lossless DEPTH-reduction by object key-flattening (§5.1, the `maxDepthStrategy: "flatten"` transform).
 * Collapses chains of nested OBJECTS into dotted keys (`a.b.c`) so a strict decoder with a nesting-depth
 * ceiling can take a schema it would otherwise reject — and provides the deterministic inverse
 * (`unflatten`) that `adaptSchema` composes into its `postProcess`. This is plain object key-flattening
 * (think lodash flatten/unflatten); the only thing on top is the guard that makes every hoist a strict
 * BIJECTION, so the round-trip never changes which values are representable:
 *
 *  - Hoist a child object `c` into `parent.c.*` ONLY when presence is unambiguous and keys are knowable:
 *      • `c` is a pure object — NOT an array, primitive, `{}`-any, union, or `$ref` (those can't enumerate
 *        keys or would need a variant/length we don't know);
 *      • `c` is NON-NULLABLE — else `c: null` is a distinct state that presence/absence of `c.*` can't encode;
 *      • `c`'s presence is determined — `c` is REQUIRED, or OPTIONAL but with ≥1 required child (so `c` can
 *        never be `{}`, making "`c` absent" ⟺ "no `c.*` keys present" a clean bijection);
 *      • `c` is CLOSED — omitted `additionalProperties` is treated as closed (strict generation forces
 *        `additionalProperties:false`, so the named keys are exhaustive); an open map is left nested.
 *  - RECURSE into array `items` to flatten WITHIN them, but never hoist ACROSS an array boundary (dynamic
 *    length). Leave `{}`/`$ref`/unions untouched.
 *  - SKIP any property whose name already contains the separator — never synthesize an ambiguous key.
 *
 * This runs on the PRE-strictify schema (original optionality intact), so the "optional-with-required-child"
 * rung is available; `adaptSchema` strictifies the flattened result afterward. `unflatten` is LENIENT: a
 * value that is already in nested form passes through untouched, so both a strict (flat) answer and an
 * advisory (nested) answer round-trip — mirroring the deliberately-lenient `unwrapRootArray`.
 */

const SEP = ".";

function isPlainObject(v: unknown): v is SchemaNode {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function typeList(node: SchemaNode): string[] | undefined {
  const t = node.type;
  if (t === undefined) return undefined;
  return Array.isArray(t) ? (t as string[]) : [t as string];
}

function isArrayNode(node: SchemaNode): boolean {
  const t = typeList(node);
  return (t ? t.includes("array") : false) || "items" in node;
}

/** A pure, non-nullable object: `type:"object"` (or `properties` with no other type) and no union/ref/null. */
function isPureObject(node: SchemaNode): boolean {
  if ("anyOf" in node || "oneOf" in node || "allOf" in node || "$ref" in node) return false;
  const t = typeList(node);
  if (t) {
    if (!t.includes("object")) return false;
    if (t.length > 1) return false; // nullable (["object","null"]) or multi-typed — not a pure object
    return true;
  }
  return "properties" in node; // untyped-but-has-properties counts as an object
}

/** Treat-as-closed: an absent or `false` `additionalProperties` is closed; `true`/a schema is an open map. */
function isClosed(node: SchemaNode): boolean {
  const ap = node.additionalProperties;
  return ap === undefined || ap === false;
}

function requiredSet(node: SchemaNode): Set<string> {
  return new Set(Array.isArray(node.required) ? (node.required as unknown[]).filter((x): x is string => typeof x === "string") : []);
}

/** May `child` (the value at `key`, `parentRequiresKey` = is `key` in the parent's `required`) be hoisted? */
function isHoistable(child: unknown, key: string, parentRequiresKey: boolean): child is SchemaNode {
  if (key.includes(SEP)) return false; // collision-safe: never build an ambiguous dotted key
  if (!isPlainObject(child)) return false;
  if (!isPureObject(child) || !isClosed(child)) return false;
  const props = isPlainObject(child.properties) ? child.properties : undefined;
  if (!props || Object.keys(props).length === 0) return false; // nothing to lift
  if (parentRequiresKey) return true;
  return requiredSet(child).size > 0; // optional: hoistable only if it can never be {}
}

interface Flattened {
  schema: SchemaNode;
  /** Reconstruct a flat value back to the nested shape this node was flattened from. Lenient. */
  restore: (value: JsonValue) => JsonValue;
}

const IDENTITY: Flattened["restore"] = (v) => v;

/** Flatten any schema node: hoist within objects, recurse through arrays, leave leaves/refs/unions alone. */
function flattenAny(node: unknown): Flattened {
  if (!isPlainObject(node)) return { schema: node as SchemaNode, restore: IDENTITY };
  if (isArrayNode(node) && isPlainObject(node.items)) {
    const inner = flattenAny(node.items);
    return {
      schema: { ...node, items: inner.schema },
      restore: (v) => (Array.isArray(v) ? v.map(inner.restore) : v),
    };
  }
  if (isPureObject(node) || "properties" in node) return flattenObject(node);
  return { schema: node, restore: IDENTITY }; // {}-any, primitive, $ref — untouched
}

/** Flatten one object node: lift every hoistable child's (already-flattened) properties up with a prefix. */
function flattenObject(node: SchemaNode): Flattened {
  const props = isPlainObject(node.properties) ? node.properties : {};
  const req = requiredSet(node);
  const siblings = new Set(Object.keys(props));
  const outProps: SchemaNode = {};
  const outRequired: string[] = [];
  const restorers: Array<(flat: Record<string, JsonValue>, out: Record<string, JsonValue>) => void> = [];

  // Keep a non-hoisted child under its own key, flattening WITHIN it (the inner is reused so a
  // collision-skip doesn't recompute). The closure undoes any in-child flattening on the way back.
  const keepNested = (key: string, keyReq: boolean, inner: Flattened): void => {
    outProps[key] = inner.schema;
    if (keyReq) outRequired.push(key);
    restorers.push((flat, out) => {
      if (key in flat && flat[key] !== undefined) out[key] = inner.restore(flat[key]);
    });
  };

  for (const [key, child] of Object.entries(props)) {
    const keyReq = req.has(key);
    if (!isHoistable(child, key, keyReq)) {
      keepNested(key, keyReq, flattenAny(child));
      continue;
    }
    const inner = flattenObject(child);
    const innerProps = isPlainObject(inner.schema.properties) ? inner.schema.properties : {};
    // Collision guard: never let a synthesized `key.x` shadow a real sibling of the same name — that
    // would be lossy. If it would, keep this child nested (still flattened within) rather than hoist.
    if (Object.keys(innerProps).some((ck) => siblings.has(`${key}${SEP}${ck}`))) {
      keepNested(key, keyReq, inner);
      continue;
    }
    const innerReq = new Set(Array.isArray(inner.schema.required) ? (inner.schema.required as string[]) : []);
    for (const [ck, cs] of Object.entries(innerProps)) {
      const fk = `${key}${SEP}${ck}`;
      outProps[fk] = cs;
      // A hoisted key is required at the parent only if the hoisted object is itself required AND the
      // sub-key is required within it. If the object is optional it may be wholly absent, so none of its
      // lifted keys can be required here.
      if (keyReq && innerReq.has(ck)) outRequired.push(fk);
    }
    const prefix = `${key}${SEP}`;
    restorers.push((flat, out) => {
      if (key in flat && flat[key] !== undefined) {
        out[key] = inner.restore(flat[key]); // lenient: an already-nested answer passes through
        return;
      }
      const sub: Record<string, JsonValue> = {};
      let present = false;
      for (const fk of Object.keys(flat)) {
        const fv = flat[fk];
        if (fk.startsWith(prefix) && fv !== undefined) {
          sub[fk.slice(prefix.length)] = fv;
          present = true;
        }
      }
      if (present) out[key] = inner.restore(sub); // all `key.*` absent ⇒ optional object omitted
    });
  }

  const schema: SchemaNode = { ...node, properties: outProps };
  if (outRequired.length > 0) schema.required = outRequired;
  else delete schema.required;

  const restore: Flattened["restore"] = (value) => {
    if (!isPlainObject(value)) return value; // lenient
    const out: Record<string, JsonValue> = {};
    for (const r of restorers) r(value as Record<string, JsonValue>, out);
    return out;
  };
  return { schema, restore };
}

export interface FlattenResult {
  /** The depth-reduced schema (dotted keys for every hoisted object chain). */
  flat: SchemaNode;
  /** The deterministic, lenient inverse — flat value → original nested value. */
  unflatten: (value: JsonValue) => JsonValue;
}

/** Flatten `schema` for depth: collapse hoistable object chains to dotted keys; return the inverse. */
export function flattenForDepth(schema: SchemaNode): FlattenResult {
  const { schema: flat, restore } = flattenAny(schema);
  return { flat, unflatten: restore };
}

// Exposed for focused unit tests; not part of the public contract.
export const __internal = { isHoistable, isPureObject, isClosed, flattenObject, flattenAny };
