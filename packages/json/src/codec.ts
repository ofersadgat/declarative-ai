/**
 * Codecs, resolved by TYPE NAME (API.md, "Codecs and type names").
 *
 * A codec holds closures, so it cannot be content-addressed or stored — but a SCHEMA can. So a schema
 * NAMES its type with the `x-type` keyword and the codec is resolved by that name, globally, once per
 * type, forever:
 *
 * ```jsonc
 * { "type": "number", "x-type": "DateTime" }
 * ```
 *
 * ```ts
 * codecs.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });
 *
 * declare module "@declarative-ai/json" {
 *   interface TypeRegistry { DateTime: { value: Date; json: number } }
 * }
 * ```
 *
 * Registration is once per TYPE — not per op, per parameter, or per use. That is what makes rich types
 * work in the id family, where the op is stored and the closures cannot be.
 *
 * The shape is borrowed from the `$param` template mechanism (./template): a name inside the schema
 * document resolved through a map. The two coexist — `$param` binds type VARIABLES per call, `x-type`
 * binds type NAMES globally.
 *
 * **`x-type` is CONSTRAINING on the consumer side.** Every other `x-` keyword carries application
 * metadata and is ignored by the subtype checker; this one is the deliberate exception — a slot
 * declaring a type name accepts only producers declaring the same name, so a `DateTime` producer can
 * never silently hand an encoded epoch to a bare-number slot (or vice versa). See
 * `@declarative-ai/validate`'s `isSubschema`.
 */
import type { JsonValue, Jsonify, SchemaDocument } from "./json";
import { getOwn, setOwn } from "./ownProps";

/** The schema keyword that names a value's DECODED type. */
export const X_TYPE = "x-type";

/**
 * The global type-name → (decoded, wire) map, extended by DECLARATION MERGING from whichever package
 * owns the type:
 *
 * ```ts
 * declare module "@declarative-ai/json" {
 *   interface TypeRegistry { DateTime: { value: Date; json: number } }
 * }
 * ```
 *
 * Empty here by design: `json` knows the mechanism, never the types.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface TypeRegistry {}

/** A registered type name. */
export type TypeName = keyof TypeRegistry & string;

/** The DECODED type registered under `N`. */
export type DecodedOf<N extends TypeName> = TypeRegistry[N] extends { value: infer V } ? V : JsonValue;

/** The WIRE type registered under `N` — its `json` member, else `Jsonify` of the decoded type. */
export type WireOf<N extends TypeName> = TypeRegistry[N] extends { json: infer J }
  ? J
  : TypeRegistry[N] extends { value: infer V }
    ? Jsonify<V>
    : JsonValue;

/**
 * The two halves of a named type's serialization. `schema` is optional: it is the canonical DOCUMENT
 * for the type (handy for building slots), but resolution is by NAME — a codec is never looked up by
 * schema identity.
 */
export interface Codec<T = JsonValue, J = Jsonify<T>> {
  /** The canonical document for this type — including its `x-type` keyword. */
  schema?: SchemaDocument;
  encode(value: T): J;
  decode(json: J): T;
}

/**
 * Type-level interpretation of a schema document that ACCOUNTS FOR `x-type`: it walks the document and
 * maps every `{ "x-type": N }` node to `N`'s registered DECODED type, leaving ordinary nodes as their
 * JSON shape. It is the decode-side twin of {@link Jsonify}.
 *
 * Deliberately modest, and modest in the same way `ops`' `InferSchema` is: only compile-time LITERAL
 * documents can be interpreted, and only the shapes below (named leaf, object with `properties` +
 * `required`, array with `items`, `const`/`enum`, the primitive `type`s). Anything else degrades to
 * `JsonValue` — surfaced, never a silent `unknown`. Full inference over a literal document is
 * `InferSchema`'s job; this one exists to answer "what does this slot DECODE to".
 */
export type Decoded<S> = S extends { readonly [X_TYPE]: infer N }
  ? N extends TypeName
    ? DecodedOf<N>
    : JsonValue
  : S extends { readonly const: infer C }
    ? C
    : S extends { readonly enum: readonly (infer E)[] }
      ? E
      : S extends { readonly type: "object"; readonly properties: infer P }
        ? DecodedObject<P, RequiredNames<S>>
        : S extends { readonly type: "array"; readonly items: infer I }
          ? Decoded<I>[]
          : S extends { readonly type: "string" }
            ? string
            : S extends { readonly type: "number" | "integer" }
              ? number
              : S extends { readonly type: "boolean" }
                ? boolean
                : S extends { readonly type: "null" }
                  ? null
                  : JsonValue;

/** The `required` names of a literal object schema (none when the keyword is absent). */
type RequiredNames<S> = S extends { readonly required: readonly (infer R)[] } ? (R extends string ? R : never) : never;

/** Properties split by requiredness, each decoded. */
type DecodedObject<P, R extends string> = { [K in keyof P as K extends R ? K : never]: Decoded<P[K]> } & {
  [K in keyof P as K extends R ? never : K]?: Decoded<P[K]>;
};

// --- The registry -------------------------------------------------------------

/** A codec stored without its type parameters — what the runtime walkers below hold. */
type AnyCodec = Codec<never, never>;

/**
 * The process-global codec registry. One entry per TYPE NAME; re-registering the same name with a
 * different codec THROWS, because two codecs for one name would make a stored schema mean two
 * different things depending on load order.
 */
export class CodecRegistry {
  private readonly byName = new Map<string, AnyCodec>();

  /** Register a type's codec. Once, globally, per type. */
  register<N extends TypeName>(name: N, codec: Codec<DecodedOf<N>, WireOf<N>>): this {
    const existing = this.byName.get(name);
    if (existing !== undefined && existing !== (codec as unknown as AnyCodec)) {
      throw new Error(`codec for type '${name}' is already registered — a type name has exactly one codec`);
    }
    this.byName.set(name, codec as unknown as AnyCodec);
    return this;
  }

  has(name: string): boolean {
    return this.byName.has(name);
  }

  /** The codec for a name, or `undefined` — a slot naming an unregistered type stays raw JSON. */
  get(name: string): Codec<JsonValue, JsonValue> | undefined {
    return this.byName.get(name) as Codec<JsonValue, JsonValue> | undefined;
  }

  /** Every registered type name — diagnostics, and the "is this document interpretable" check. */
  names(): string[] {
    return [...this.byName.keys()];
  }
}

/** The process-global registry. Registration is once per type, forever (§3.3). */
export const codecs = new CodecRegistry();

/** The type name a schema node declares, if any. */
export function typeNameOf(schema: SchemaDocument | undefined): string | undefined {
  const n = schema?.[X_TYPE];
  return typeof n === "string" ? n : undefined;
}

// --- Runtime walks ------------------------------------------------------------
//
// The type-level `Decoded`/`Jsonify` pair has a runtime twin: given a schema document, walk a value
// and apply each named node's codec. This is the decode step that belongs at a call boundary — it is
// what `postProcess` on a structured call becomes once schemas can name rich types.

/** Read a nested schema node as a document (a non-object keyword value is not a schema). The key is
 *  often a VALUE's key, so the read must be own-property: `properties["__proto__"]` would otherwise
 *  return `Object.prototype` and be walked as if it were a schema. */
function nodeAt(schema: SchemaDocument | undefined, key: string): SchemaDocument | undefined {
  const v = schema === undefined ? undefined : getOwn(schema, key);
  return v !== null && v !== undefined && typeof v === "object" && !Array.isArray(v) ? (v as SchemaDocument) : undefined;
}

/**
 * The maximum schema-recursion depth the walks follow. Array/object descent is self-limiting — it
 * consumes a level of the (finite) VALUE at each step — but a COMPOSITION keyword (`allOf`/`anyOf`/
 * `oneOf`) recurses against a sub-schema on the SAME value, so a schema object hand-built to reference
 * itself (`s.allOf = [s]`) would recurse forever without ever shrinking the value. This bound turns
 * that into a visible passthrough (the value is returned raw at the cap) rather than a stack overflow,
 * matching the module's philosophy: an uninterpretable schema degrades to raw JSON, never throws.
 */
const MAX_SCHEMA_DEPTH = 100;

/**
 * A walk DIRECTION: the leaf transform (`step`, threaded so the shared {@link walk} recurses in the same
 * direction) plus how a `anyOf`/`oneOf` branch is SELECTED. Selection is direction-specific because the
 * value has a different shape on each side: decode sees the WIRE value (a leaf's `type` IS its runtime
 * JSON type, so a number picks the `{type:"number"}` branch), while encode sees the DECODED value (a
 * `Date` is a runtime object that matches no wire `type` — the branch is found by probing its codec).
 */
interface Direction {
  step(schema: SchemaDocument | undefined, value: JsonValue, depth: number): unknown;
  pickBranch(branches: (SchemaDocument | undefined)[], value: JsonValue): SchemaDocument | undefined;
}

/**
 * DECODE a wire value against a schema: every `x-type` node whose codec is registered is lifted to its
 * decoded type; everything else passes through structurally. Unregistered names are left as raw JSON —
 * a schema may legitimately name a type this process does not model.
 */
export function decodeWithSchema(schema: SchemaDocument | undefined, json: JsonValue): unknown {
  return DECODE.step(schema, json, 0);
}

const DECODE: Direction = {
  step(schema, json, depth) {
    const name = typeNameOf(schema);
    if (name !== undefined) {
      const codec = codecs.get(name);
      return codec ? codec.decode(json) : json;
    }
    return walk(schema, json, DECODE, depth);
  },
  // The wire value's JSON type IS the tag: the FIRST branch whose declared `type` admits it wins.
  pickBranch(branches, value) {
    return branches.find((b) => branchAccepts(b, jsonType(value)));
  },
};

/**
 * ENCODE a decoded value against a schema — the inverse of {@link decodeWithSchema}. Values under an
 * unregistered type name are passed through unchanged (they were never lifted in the first place).
 */
export function encodeWithSchema(schema: SchemaDocument | undefined, value: unknown): JsonValue {
  return ENCODE.step(schema, value as JsonValue, 0) as JsonValue;
}

const ENCODE: Direction = {
  step(schema, value, depth) {
    const name = typeNameOf(schema);
    if (name !== undefined) {
      const codec = codecs.get(name);
      return codec ? codec.encode(value) : value;
    }
    return walk(schema, value, ENCODE, depth);
  },
  // The value is DECODED, so its JSON type cannot tag a wire branch (a `Date` reads as `"object"`).
  // Probe instead: a coded branch matches when its codec ENCODES the value to the branch's wire `type`
  // (an epoch for `{type:"number","x-type":"DateTime"}`); a plain branch matches by JSON type, since a
  // plain value is already its own wire form. First match wins; none ⇒ raw (the caller leaves it be).
  pickBranch(branches, value) {
    for (const b of branches) {
      if (b === undefined) continue;
      const name = typeNameOf(b);
      const codec = name !== undefined ? codecs.get(name) : undefined;
      if (codec) {
        try {
          if (branchAccepts(b, jsonType(codec.encode(value) as JsonValue))) return b;
        } catch {
          // this value is not of the branch's type — try the next branch
        }
      } else if (branchAccepts(b, jsonType(value))) {
        return b;
      }
    }
    return undefined;
  },
};

/** The runtime JSON type of a value — the tag `anyOf`/`oneOf` selection matches against a branch's
 *  declared `type`. JSON has no `integer` at runtime, so a number answers `"number"` and the
 *  `integer`/`number` distinction is reconciled in {@link branchAccepts}. */
function jsonType(value: JsonValue): "string" | "number" | "boolean" | "null" | "array" | "object" {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  const t = typeof value;
  return t === "object" ? "object" : (t as "string" | "number" | "boolean");
}

/** Read a schema keyword whose value is an ARRAY of subschemas (`allOf`/`anyOf`/`oneOf`, or a tuple
 *  `items`/`prefixItems`). Each element is normalized to a document or `undefined` (a non-object entry
 *  — e.g. `true`/`false` — is not walkable and passes its value verbatim). */
function schemaList(schema: SchemaDocument | undefined, key: string): (SchemaDocument | undefined)[] | undefined {
  const v = schema === undefined ? undefined : getOwn(schema, key);
  if (!Array.isArray(v)) return undefined;
  return v.map((e) => (e !== null && typeof e === "object" && !Array.isArray(e) ? (e as SchemaDocument) : undefined));
}

/** Whether a union branch's declared `type` admits a value of JSON type `jt`. A branch with no `type`
 *  keyword cannot be discriminated by shape, so it is treated as non-matching — selection stays raw
 *  rather than guessing. `integer` accepts a `number` value (JSON has no integer at runtime). */
function branchAccepts(branch: SchemaDocument | undefined, jt: ReturnType<typeof jsonType>): boolean {
  if (branch === undefined) return false;
  const t = getOwn(branch, "type");
  const types = typeof t === "string" ? [t] : Array.isArray(t) ? t : undefined;
  if (types === undefined) return false;
  return types.some((dt) => dt === jt || (jt === "number" && dt === "integer"));
}

/**
 * The shared recursion of the two walks — arrays through `items`/tuple, objects through `properties`
 * (then `additionalProperties`), and the COMPOSITION keywords — parameterized by a {@link Direction} so
 * both sides share exactly this structure. `depth` guards the composition cycle vector.
 */
function walk(schema: SchemaDocument | undefined, value: JsonValue, dir: Direction, depth: number): unknown {
  if (schema === undefined || depth >= MAX_SCHEMA_DEPTH) return value;

  // Composition keywords constrain the SAME value against sub-schemas, independent of its JSON type, so
  // they are handled before the structural descent.
  //  - `allOf` INTERSECTS: thread the value through every branch in turn. A branch that doesn't carry
  //    the relevant structure is a structural passthrough (it returns the value unchanged), so composing
  //    them is order-independent for disjoint branches — each converts the members it names, leaving the
  //    rest (already-converted or plain) untouched.
  const allOf = schemaList(schema, "allOf");
  if (allOf) {
    let v = value;
    for (const branch of allOf) v = dir.step(branch, v, depth + 1) as JsonValue;
    return v;
  }
  //  - `anyOf`/`oneOf` MATCH ONE branch, selected direction-appropriately (see {@link Direction}). No
  //    branch applies ⇒ leave the value RAW — never a silent wrong lift.
  const union = schemaList(schema, "anyOf") ?? schemaList(schema, "oneOf");
  if (union) {
    const branch = dir.pickBranch(union, value);
    return branch === undefined ? value : dir.step(branch, value, depth + 1);
  }

  if (Array.isArray(value)) {
    // Tuple: `items: [A, B, …]` (draft-07) or `prefixItems: [A, B, …]` (2020-12). The i-th element uses
    // the i-th subschema; elements past the tuple use the "rest" schema — `additionalItems` for the
    // draft-07 form, the single-schema `items` for the 2020-12 form — or pass verbatim when absent.
    const prefix = schemaList(schema, "prefixItems");
    const itemsTuple = schemaList(schema, "items");
    const tuple = prefix ?? itemsTuple;
    if (tuple) {
      const rest = prefix ? nodeAt(schema, "items") : nodeAt(schema, "additionalItems");
      return value.map((v, i) => {
        const sub = i < tuple.length ? tuple[i] : rest;
        return sub === undefined ? v : dir.step(sub, v, depth + 1);
      });
    }
    // Single-schema `items`: every element converted against the one item schema.
    const items = nodeAt(schema, "items");
    return items === undefined ? value : value.map((v) => dir.step(items, v, depth + 1));
  }
  if (value !== null && typeof value === "object") {
    const props = nodeAt(schema, "properties");
    const additional = nodeAt(schema, "additionalProperties");
    if (props === undefined && additional === undefined) return value;
    const out: Record<string, unknown> = {};
    // `setOwn`, not `out[k] = …`: a wire object really can carry a `__proto__` key (`JSON.parse`
    // produces it as an own property) and plain assignment would silently drop it from the result.
    for (const [k, v] of Object.entries(value)) setOwn(out, k, dir.step(nodeAt(props, k) ?? additional, v, depth + 1));
    return out;
  }
  return value;
}
