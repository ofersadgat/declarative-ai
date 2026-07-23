/**
 * Pragmatic structural JSON-Schema subtype check (ported wholesale from findmyprompt,
 * API.md, "Schema subtyping"): `isSubschema(sub, sup)` answers "is every value valid under
 * `sub` necessarily valid under `sup`?" — i.e. can a producer whose output matches `sub`
 * safely feed a consumer slot that requires `sup`. No maintained JS library answers this,
 * so we hand-roll the rules (porting the spirit of the stale-but-correct
 * `is-json-schema-subset`):
 *  - `type` gate; `enum`/`const` ⊆; objects = consumer's `required` guaranteed + properties
 *    recursively compatible + `additionalProperties:false` honored; arrays recurse on `items`;
 *    numeric/length bounds widen-only.
 *
 * `allOf` IS supported — flattened into one effective schema (so schema evolution via
 * `allOf: [{$ref: old}, {+field}]` typechecks). **Conservative** otherwise: the remaining
 * unions (`anyOf`/`oneOf`/`not`, conditionals) or any unmodeled constraining keyword on the
 * consumer → REJECT with a precise reason, never a silent pass. v1 scope is
 * object/array/primitives/enum-const/$ref/allOf; consumer-side `anyOf` is the documented next
 * addition.
 */
/**
 * The schema shape this checker reads — the ONE deliberate exception to the §2.2 "no `unknown` in an
 * exported type" rule, for two reasons that hold only here:
 *
 *  - This module is a VERBATIM port (decision #11), internals and test suite included. It builds a
 *    merged schema in place when flattening `allOf`, and its tests pass heterogeneous literals that a
 *    readonly `ReadonlyJsonValue`-indexed type rejects. Retyping it would fork the port.
 *  - The `unknown` never reaches a consumer's types: callers hold a `JsonSchema<T>` / `SchemaDocument`
 *    (both assignable to this), and what comes back is a `SubtypeResult`, not a schema.
 *
 * It is a document whose keyword values this checker treats opaquely — that is exactly what `unknown`
 * says here, rather than a gap in the vocabulary.
 */
import { X_TYPE } from "@declarative-ai/json";

export type Schema = Record<string, unknown>;
export type ResolveRef = (refId: string) => Schema | undefined;

export interface SubtypeResult {
  ok: boolean;
  reason?: string;
}

const OK: SubtypeResult = { ok: true };
const fail = (reason: string): SubtypeResult => ({ ok: false, reason });

/** Keywords the checker reasons about. A constraining keyword on the CONSUMER outside this set → reject. */
const MODELED = new Set([
  "type", "enum", "const", "properties", "required", "additionalProperties", "items",
  "minimum", "maximum", "minLength", "maxLength",
  // The one extension keyword with validation semantics — see `X_TYPE` below.
  X_TYPE,
  // non-constraining metadata (ignored). `format` is ANNOTATION-ONLY in JSON Schema unless a
  // validator opts into format-assertion, so it constrains nothing here and is safe to ignore.
  // `contentEncoding`/`contentMediaType` are likewise annotations; they carry KIND information
  // (`kindFor` reads them to decide `blob`), not a constraint this checker must enforce.
  "title", "description", "$ref", "$id", "$schema", "$comment", "default", "examples", "format",
  "contentEncoding", "contentMediaType",
]);

/** Extension keywords (`x-…`) carry application metadata, never validation semantics, so they are
 *  ignored like the annotations above rather than triggering a conservative rejection — with exactly
 *  ONE exception, {@link X_TYPE}, which is handled explicitly in `isSubschema`. */
const isExtensionKeyword = (k: string): boolean => k.startsWith("x-") && k !== X_TYPE;

/** The name a value's DECODED type is registered under (`@declarative-ai/json`'s codec registry). */
const typeNameOf = (s: Schema): string | undefined => (typeof s[X_TYPE] === "string" ? (s[X_TYPE] as string) : undefined);

const UNION_KEYWORDS = ["anyOf", "oneOf", "not", "if", "then", "else"];

/**
 * Is `subIn` (a producer) acceptable where `supIn` (a consumer slot) is expected?
 *
 * `seen` is the coinductive CYCLE GUARD and nothing else: it holds the pairs currently ON THE PROOF
 * STACK, so a self-referential schema (`S = { next: { $ref: "S" } }`) terminates by assuming the pair
 * it is already proving. Two properties are load-bearing and were both wrong before:
 *
 *  - A pair is guarded only when BOTH sides are `$ref`s. A cycle cannot arise otherwise — an inline
 *    document is finite, so structural descent terminates on its own. Keying on one side alone made
 *    every distinct producer checked against the same `$ref`'d slot collide, so `{a: {$ref:"S"}, b:
 *    {$ref:"S"}}` — two INDEPENDENT obligations, not a cycle — waved the second one through.
 *  - The key is DELETED on exit. Without that, `seen` is a "have ever touched" memo shared across
 *    SIBLING recursions rather than a record of what is in flight.
 */
export function isSubschema(
  subIn: Schema,
  supIn: Schema,
  resolve?: ResolveRef,
  seen: Set<string> = new Set(),
): SubtypeResult {
  // A cycle needs a `$ref` on BOTH sides; anything else descends structurally and terminates.
  const pairKey =
    typeof subIn.$ref === "string" && typeof supIn.$ref === "string" ? `${subIn.$ref}>${supIn.$ref}` : undefined;
  if (pairKey !== undefined) {
    if (seen.has(pairKey)) return OK; // already proving this exact pair — assume it holds
    seen.add(pairKey);
  }
  try {
    return check(subIn, supIn, resolve, seen);
  } finally {
    if (pairKey !== undefined) seen.delete(pairKey);
  }
}

function check(subIn: Schema, supIn: Schema, resolve: ResolveRef | undefined, seen: Set<string>): SubtypeResult {
  // `allOf` is flattened into one effective schema (its members are an intersection) — this is how a
  // schema EVOLVES: "add a field" is `allOf: [{$ref: old}, {properties:{+field}}]`.
  const sub = flattenAllOf(deref(subIn, resolve), resolve);
  const sup = flattenAllOf(deref(supIn, resolve), resolve);

  // An UNRESOLVED `$ref` on the consumer describes a constraint we cannot read. Treating it as
  // universal (the old behaviour, because `$ref` is not in CONSTRAINING) accepted everything against
  // it — the opposite of this module's conservative contract.
  if (typeof sup.$ref === "string") return fail(`consumer '$ref: ${sup.$ref}' could not be resolved`);

  for (const kw of UNION_KEYWORDS) {
    if (kw in sub) return fail(`producer uses '${kw}' (unions/conditionals not supported in v1)`);
    if (kw in sup) return fail(`consumer uses '${kw}' (unions/conditionals not supported in v1)`);
  }
  const unmodeled = Object.keys(sup).filter((k) => !MODELED.has(k) && !isExtensionKeyword(k));
  if (unmodeled.length) return fail(`consumer uses unmodeled keyword(s): ${unmodeled.join(", ")}`);

  // `x-type` is CONSTRAINING (API.md, "Codecs and type names") — the one `x-` keyword with validation
  // semantics. Ignoring it like the other extensions would let a `DateTime` producer feed a bare-number
  // slot (silently handing over an encoded epoch) and vice versa, which is exactly the unsoundness the
  // codec registry exists to prevent. A slot declaring a type name accepts only producers declaring the
  // SAME name; whether a wider named type should accept a narrower one needs a real case before it is
  // modeled (§9).
  const supType = typeNameOf(sup);
  const subType = typeNameOf(sub);
  if (supType !== subType) {
    if (supType !== undefined && subType === undefined) {
      return fail(`consumer requires type '${supType}' but the producer declares no x-type (it would hand over the encoded form)`);
    }
    if (supType === undefined && subType !== undefined) {
      return fail(`producer declares type '${subType}' but the consumer expects the raw encoded form (no x-type)`);
    }
    return fail(`producer type '${subType}' does not match consumer type '${supType}'`);
  }

  // A consumer with no constraints accepts anything.
  if (isUniversal(sup)) return OK;

  // A producer carrying `const` is a KNOWN CONCRETE VALUE — an inline literal binding. Check the value
  // against the consumer directly rather than asking one schema to be a subtype of another: a literal
  // schema does not restate `items`/`minimum`/`maxLength`/`additionalProperties`, so the structural
  // path below rejected every literal bound into a bounded, typed-array, or closed slot. This is both
  // sounder and more precise — `42` into `{minimum: 0, maximum: 100}` is decidable, not approximable.
  if ("const" in sub) return valueSatisfies(sub.const, sup, resolve, seen, "");

  // enum / const
  const supVals = allowedValues(sup);
  if (supVals) {
    const subVals = allowedValues(sub);
    if (!subVals) return fail("consumer is enum/const-constrained but producer is not");
    for (const v of subVals) {
      if (!supVals.some((s) => deepEqual(s, v))) return fail(`producer value ${JSON.stringify(v)} not in consumer enum/const`);
    }
  }

  // type
  const supTypes = typesOf(sup);
  if (supTypes) {
    const subTypes = typesOf(sub);
    if (!subTypes) return fail(`consumer requires type ${supTypes.join("|")} but producer declares none`);
    for (const t of subTypes) {
      // `integer` values are also `number`s — a sound, common narrowing.
      const compatible = supTypes.includes(t) || (t === "integer" && supTypes.includes("number"));
      if (!compatible) return fail(`producer type '${t}' not allowed by consumer ${supTypes.join("|")}`);
    }
  }

  // object
  if (isObjectSchema(sup)) {
    const supRequired = asStringArray(sup.required);
    const subRequired = new Set(asStringArray(sub.required));
    for (const r of supRequired) {
      if (!subRequired.has(r)) return fail(`consumer requires property '${r}' but producer doesn't guarantee it`);
    }
    const supProps = asSchemaMap(sup.properties);
    const subProps = asSchemaMap(sub.properties);
    for (const [name, supProp] of Object.entries(supProps)) {
      const subProp = subProps[name];
      if (!subProp) {
        if (supRequired.includes(name)) {
          return fail(`consumer constrains required property '${name}' but producer doesn't define it`);
        }
        continue; // optional property the producer may omit — fine
      }
      const r = isSubschema(subProp, supProp, resolve, seen);
      if (!r.ok) return fail(`property '${name}': ${r.reason}`);
    }
    if (sup.additionalProperties === false) {
      if (sub.additionalProperties !== false) return fail("consumer forbids additional properties but producer allows them");
      const allowed = new Set(Object.keys(supProps));
      for (const k of Object.keys(subProps)) {
        if (!allowed.has(k)) return fail(`producer property '${k}' not permitted by closed consumer`);
      }
    }
  }

  // array
  if (isArraySchema(sup) && sup.items && typeof sup.items === "object") {
    // A TUPLE consumer (`items` as an array of positional schemas) is not modeled; `items` is in
    // MODELED so it never trips the unmodeled-keyword check, which meant a tuple slot silently
    // accepted anything. Refuse it explicitly instead.
    if (Array.isArray(sup.items)) return fail("consumer uses tuple-form 'items' (not supported in v1)");
    if (!sub.items || typeof sub.items !== "object" || Array.isArray(sub.items)) {
      return fail("consumer constrains array items but producer doesn't");
    }
    const r = isSubschema(sub.items as Schema, sup.items as Schema, resolve, seen);
    if (!r.ok) return fail(`items: ${r.reason}`);
  }

  // numeric + length bounds: producer must be at least as strict (widen-only)
  const bounds = checkBounds(sub, sup);
  if (!bounds.ok) return bounds;

  return OK;
}

// --- literal values ---------------------------------------------------------

/**
 * Does a KNOWN value satisfy a consumer schema? The singleton-type half of the check, used when the
 * producer carries a `const` (an inline literal binding). Covers exactly the keywords `isSubschema`
 * models — anything else is already rejected upstream by the unmodeled-keyword check — and stays
 * ajv-free, which is what keeps this package's heavy dependency out of the checker's path.
 */
function valueSatisfies(value: unknown, sup: Schema, resolve: ResolveRef | undefined, seen: Set<string>, at: string): SubtypeResult {
  const where = at === "" ? "" : `${at}: `;
  const flat = flattenAllOf(deref(sup, resolve), resolve);
  if (isUniversal(flat)) return OK;

  const supVals = allowedValues(flat);
  if (supVals && !supVals.some((v) => deepEqual(v, value))) {
    return fail(`${where}value ${JSON.stringify(value)} not in consumer enum/const`);
  }

  const supTypes = typesOf(flat);
  if (supTypes) {
    const actual = jsonTypeOf(value);
    const ok = supTypes.includes(actual) || (actual === "integer" && supTypes.includes("number"));
    if (!ok) return fail(`${where}value is ${actual}, not allowed by consumer ${supTypes.join("|")}`);
  }

  if (typeof value === "number") {
    if (typeof flat.minimum === "number" && value < flat.minimum) return fail(`${where}${value} below consumer minimum ${flat.minimum}`);
    if (typeof flat.maximum === "number" && value > flat.maximum) return fail(`${where}${value} above consumer maximum ${flat.maximum}`);
  }
  if (typeof value === "string") {
    if (typeof flat.minLength === "number" && value.length < flat.minLength) return fail(`${where}string shorter than consumer minLength ${flat.minLength}`);
    if (typeof flat.maxLength === "number" && value.length > flat.maxLength) return fail(`${where}string longer than consumer maxLength ${flat.maxLength}`);
  }

  if (Array.isArray(value)) {
    if (typeof flat.minLength === "number" && value.length < flat.minLength) return fail(`${where}array shorter than consumer minLength ${flat.minLength}`);
    const items = flat.items;
    if (Array.isArray(items)) return fail(`${where}consumer uses tuple-form 'items' (not supported in v1)`);
    if (items && typeof items === "object") {
      for (const [i, el] of value.entries()) {
        const r = valueSatisfies(el, items as Schema, resolve, seen, `${at}[${i}]`);
        if (!r.ok) return r;
      }
    }
    return OK;
  }

  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const supProps = asSchemaMap(flat.properties);
    for (const r of asStringArray(flat.required)) {
      if (!(r in record)) return fail(`${where}consumer requires property '${r}' but the literal doesn't have it`);
    }
    if (flat.additionalProperties === false) {
      const allowed = new Set(Object.keys(supProps));
      for (const k of Object.keys(record)) {
        if (!allowed.has(k)) return fail(`${where}property '${k}' not permitted by closed consumer`);
      }
    }
    for (const [name, supProp] of Object.entries(supProps)) {
      if (!(name in record)) continue; // absent AND not required — fine
      const r = valueSatisfies(record[name], supProp, resolve, seen, at === "" ? name : `${at}.${name}`);
      if (!r.ok) return r;
    }
  }

  return OK;
}

/** A JSON value's schema `type`, with `integer` distinguished (it narrows `number`). */
function jsonTypeOf(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  if (typeof v === "number") return Number.isInteger(v) ? "integer" : "number";
  return typeof v; // string | boolean | object
}

// --- helpers ---------------------------------------------------------------

function deref(schema: Schema, resolve?: ResolveRef): Schema {
  if (typeof schema.$ref === "string" && resolve) {
    const target = resolve(schema.$ref);
    if (target) return target;
  }
  return schema;
}

/** Flatten `allOf` (an intersection) into one effective schema: properties merged, `required` unioned,
 *  `additionalProperties:false` if ANY member is closed; other keywords last-wins (sufficient for the
 *  schema-evolution case where members don't conflict). Recursive: a member may itself carry `allOf`. */
function flattenAllOf(schema: Schema, resolve?: ResolveRef): Schema {
  if (!Array.isArray(schema.allOf)) return schema;
  const { allOf, ...own } = schema;
  const merged: Schema = {};
  for (const member of [...(allOf as Schema[]), own as Schema]) {
    mergeInto(merged, flattenAllOf(deref(member, resolve), resolve));
  }
  return merged;
}

function mergeInto(target: Schema, src: Schema): void {
  for (const [k, v] of Object.entries(src)) {
    if (v === undefined || k === "allOf") continue;
    if (k === "properties") {
      target.properties = { ...asSchemaMap(target.properties), ...asSchemaMap(v) };
    } else if (k === "required") {
      target.required = [...new Set([...asStringArray(target.required), ...asStringArray(v)])];
    } else if (k === "additionalProperties") {
      target.additionalProperties = target.additionalProperties === false || v === false ? false : v;
    } else {
      target[k] = v; // type / bounds / enum — last-wins (members rarely conflict here)
    }
  }
}

const CONSTRAINING = ["type", "enum", "const", "properties", "required", "items", "minimum", "maximum", "minLength", "maxLength", "additionalProperties"];

function isUniversal(s: Schema): boolean {
  return ![...CONSTRAINING, X_TYPE].some((k) => k in s);
}

function typesOf(s: Schema): string[] | undefined {
  const t = s.type;
  if (t === undefined) return undefined;
  return Array.isArray(t) ? (t as string[]) : [t as string];
}

function allowedValues(s: Schema): unknown[] | undefined {
  if ("const" in s) return [s.const];
  if (Array.isArray(s.enum)) return s.enum as unknown[];
  return undefined;
}

function isObjectSchema(s: Schema): boolean {
  const t = typesOf(s);
  return (t ? t.includes("object") : false) || "properties" in s || "required" in s || "additionalProperties" in s;
}

function isArraySchema(s: Schema): boolean {
  const t = typesOf(s);
  return (t ? t.includes("array") : false) || "items" in s;
}

function checkBounds(sub: Schema, sup: Schema): SubtypeResult {
  if (typeof sup.minimum === "number" && !(typeof sub.minimum === "number" && sub.minimum >= sup.minimum)) {
    return fail(`producer minimum (${String(sub.minimum)}) below consumer minimum ${sup.minimum}`);
  }
  if (typeof sup.maximum === "number" && !(typeof sub.maximum === "number" && sub.maximum <= sup.maximum)) {
    return fail(`producer maximum (${String(sub.maximum)}) above consumer maximum ${sup.maximum}`);
  }
  if (typeof sup.minLength === "number" && !(typeof sub.minLength === "number" && sub.minLength >= sup.minLength)) {
    return fail(`producer minLength below consumer minLength ${sup.minLength}`);
  }
  if (typeof sup.maxLength === "number" && !(typeof sub.maxLength === "number" && sub.maxLength <= sup.maxLength)) {
    return fail(`producer maxLength above consumer maxLength ${sup.maxLength}`);
  }
  return OK;
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? (v.filter((x) => typeof x === "string") as string[]) : [];
}

function asSchemaMap(v: unknown): Record<string, Schema> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, Schema>) : {};
}

function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
