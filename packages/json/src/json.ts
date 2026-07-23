/**
 * The JSON vocabulary the whole workspace converges on (API.md, "The JSON vocabulary").
 *
 * `unknown` is legal in exactly two positions: generic BOUNDS (the `RefFamily` slots in `ops`) and
 * external boundaries immediately narrowed inside a parsing function. Every other position uses these
 * types, a generic parameter, or a precise domain type.
 *
 * The load-bearing change over the ops redesign is that the typed schema no longer LIES about
 * serialization: `JsonSchema<T>`'s phantom carries the DECODED type `T`, while the schema DOCUMENT
 * describes `Jsonify<T>` — T's JSON projection. Encode is derivable from the type (the `toJSON()`
 * contract); DECODE is not derivable from any type, so it is a runtime function resolved by TYPE NAME
 * through the codec registry (see ./codec).
 */

/** Any JSON value — what a json-valued slot actually holds. Excludes undefined/functions/classes. */
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

/**
 * What may be SERIALIZED to JSON — `JsonValue` widened to tolerate `undefined` in optional
 * object members (which JSON serialization drops) and readonly arrays/objects. It is the
 * honest input type for canonicalization/hashing: precise domain records with optional
 * fields fit, while functions, symbols, and bigint — the values that would actually throw —
 * do not. Use `JsonValue` for values a program READS; use this only at a serialization edge.
 */
export type Serializable =
  | string
  | number
  | boolean
  | null
  | undefined
  | readonly Serializable[]
  | { readonly [key: string]: Serializable };

/**
 * Every FIELD of `T` is {@link Serializable} — the same check done property-wise instead of through an
 * index signature. Needed because TypeScript grants an implicit index signature to type aliases but NOT
 * to interfaces, so a nominal domain record (`LlmCallDefinition`, `WorkflowBundle`) fails a structural
 * `Serializable` test even when every one of its fields passes. Use as the bound on a serialization
 * entry point: `f<T extends Serializable | SerializableFields<T>>(value: T)`.
 */
export type SerializableFields<T> = { [K in keyof T]: Serializable };

/**
 * A JSON value read through a readonly lens — what `as const` literals produce. Every
 * `JsonValue` is assignable to it; use it where documents are consumed, not built.
 */
export type ReadonlyJsonValue =
  | string
  | number
  | boolean
  | null
  | readonly ReadonlyJsonValue[]
  | { readonly [key: string]: ReadonlyJsonValue };

// --- Jsonify: the wire projection of a decoded type ---------------------------

/** A tuple has a literal `length`; a plain array's is the wide `number`. */
type IsTuple<T extends readonly unknown[]> = number extends T["length"] ? false : true;

/**
 * The JSON projection of `T` — what `JSON.parse(JSON.stringify(value))` actually yields, at the type
 * level (§3.2):
 *
 *  - a type declaring `toJSON()` projects to ITS return type (recursively) — this is why encode is
 *    derivable from the type while decode is not;
 *  - `undefined`, functions, symbols, and bigint cannot be carried: object MEMBERS holding them are
 *    DROPPED, array ELEMENTS holding them become `null` (exactly `JSON.stringify`'s two behaviors);
 *  - tuples keep their arity; arrays stay arrays;
 *  - optionality and readonly-ness are preserved (a mapped type keeps its modifiers).
 *
 * Compile time works in `T`; the runtime check validates `Jsonify<T>`. You never ask a model to emit a
 * `DateTime` — you ask for an epoch, validate the epoch, and lift it with the type's registered codec.
 */
export type Jsonify<T> = [T] extends [JsonValue]
  ? T
  : T extends { toJSON(): infer R }
    ? Jsonify<R>
    : JsonifyNonJson<T>;

/**
 * The projection of a type that is NOT already JSON. Split out so the `[T] extends [JsonValue]`
 * short-circuit above stays one line: a type that is already a `JsonValue` projects to ITSELF, which is
 * both the right answer and the only way `Jsonify<JsonValue>` terminates — mapping `JsonValue`'s
 * recursive union-of-index-signature through {@link JsonifyObject} is what makes TypeScript give up
 * with "type instantiation is excessively deep" (TS2589), and `Jsonify<T>` is `Codec`'s DEFAULT wire
 * parameter, so that blow-up would be latent in public API. The test is non-distributive (`[T]`) so a
 * union is judged whole and short-circuits whole; anything with a `toJSON()` method is not assignable
 * to `JsonValue`, so the ordering cannot steal that branch.
 */
type JsonifyNonJson<T> = T extends string | number | boolean | null
  ? T
  : T extends undefined | bigint | symbol | ((...args: never[]) => unknown)
    ? never
    : T extends readonly unknown[]
      ? IsTuple<T> extends true
        ? { -readonly [K in keyof T]: JsonifyElement<T[K]> }
        : JsonifyElement<T[number]>[]
      : T extends object
        ? JsonifyObject<T>
        : never;

/** An ARRAY element: a value JSON cannot carry serializes as `null`. Distributive on purpose — each
 *  member of a union element type is judged on its own, so `(string | (() => void))[]` projects to
 *  `(string | null)[]` rather than quietly losing the un-serializable member. */
type JsonifyElement<T> = T extends unknown ? ([Jsonify<T>] extends [never] ? null : Jsonify<T>) : never;

/** An OBJECT projection: a member JSON cannot carry is omitted entirely. */
type JsonifyObject<T> = {
  [K in keyof T as [Jsonify<Exclude<T[K], undefined>>] extends [never] ? never : K]: Jsonify<Exclude<T[K], undefined>>;
};

// --- Schema documents ---------------------------------------------------------

declare const SchemaOutputType: unique symbol;

/**
 * A JSON Schema document carrying an optional PHANTOM type `T`. The phantom is the DECODED type; the
 * DOCUMENT describes `Jsonify<T>` (§3.2). Read it as "a schema *for* `T`, whose wire form is
 * `Jsonify<T>`" — the phantom deliberately does NOT carry `Jsonify<T>`, because TypeScript cannot infer
 * backwards through a conditional type and `JsonSchema<Jsonify<T>>` would make `T` uninferable at every
 * call site.
 *
 * The phantom is type-level only — no runtime cost, never serialized (the symbol key cannot collide
 * with schema keywords). The document surface is `ReadonlyJsonValue`-indexed so both plain values and
 * `as const` literals are assignable without casts.
 */
export type JsonSchema<T = JsonValue> = SchemaDocument & {
  readonly [SchemaOutputType]?: T;
};

/**
 * A schema DOCUMENT with no claim about the type it validates — what schema-reading code takes
 * (transformers, the subtype checker, a boundary validator): it inspects keywords and never the
 * phantom. Every `JsonSchema<T>` is assignable to it for any `T`, so a typed schema flows into
 * untyped schema machinery without a cast, while `JsonSchema<T>` stays the type-carrying form.
 *
 * The index has no `| undefined`, so a schema document NESTS inside another one (a `properties` entry
 * is itself a document). Reads still yield `| undefined` under `noUncheckedIndexedAccess`.
 */
export type SchemaDocument = { readonly [key: string]: ReadonlyJsonValue };

/**
 * The MUTABLE working form of a schema document — what schema TRANSFORMERS build and rewrite in
 * place (provider adaptation, union flattening, `$ref` inlining). Structurally a
 * {@link SchemaDocument} (assignable to it, and to `JsonSchema<T>` once branded) AND a `JsonValue`, so
 * a transformer's output flows straight back into the declaration vocabulary and nests inside another
 * node; it differs only in dropping the `readonly` index so the transform can assign into it. Reads
 * still yield `JsonValue | undefined` under `noUncheckedIndexedAccess`, so a missing keyword is never
 * silently treated as present.
 */
export type MutableSchema = { [key: string]: JsonValue };

/**
 * Extract the phantom type of a {@link JsonSchema}: `SchemaOutput<JsonSchema<T>> = T`, and `JsonValue`
 * for a document that carries no phantom.
 *
 * The two guards are not decoration. A document typed through an INDEX SIGNATURE (`SchemaDocument`
 * itself, `{ readonly [k: string]: JsonValue }`) matches the optional-phantom shape with nothing to
 * infer from, so `T` lands as `unknown`; a phantom explicitly bound to `undefined` would collapse to
 * `never`. Both are the "no type claimed" case and must read as `JsonValue` — `unknown` in an exported
 * position is exactly what §2.2/§3.4 bans.
 */
export type SchemaOutput<S> = S extends { readonly [SchemaOutputType]?: infer T }
  ? unknown extends T
    ? JsonValue
    : [Exclude<T, undefined>] extends [never]
      ? JsonValue
      : Exclude<T, undefined>
  : JsonValue;

/** Brand a plain schema document with a phantom type. Type-level only — the identity at runtime. */
export function typedSchema<T>(schema: SchemaDocument): JsonSchema<T> {
  return schema as JsonSchema<T>;
}

/**
 * Keywords whose value is DATA rather than a subschema. `{ "const": { "$ref": "…" } }` describes a
 * literal object that HAS a `$ref` MEMBER — the string is payload, not a reference, and following it
 * would make the loader fetch a document the schema never named.
 */
const DATA_KEYWORDS = new Set(["const", "enum", "default"]);

/** Collect every `$ref` string target reachable in a schema document. */
export function collectRefs(node: unknown, out = new Set<string>()): Set<string> {
  // The parameter is `unknown` — a caller may hand this a runtime object graph, not a parsed document —
  // so the walk carries a path set rather than trusting the input to be finite.
  refsInto(node, out, new Set<object>());
  return out;
}

function refsInto(node: unknown, out: Set<string>, seen: Set<object>): void {
  if (node === null || typeof node !== "object") return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) refsInto(x, out, seen);
    return;
  }
  for (const [k, v] of Object.entries(node)) {
    if (k === "$ref") {
      if (typeof v === "string") out.add(v);
      continue; // a `$ref` value is a URI, never a subschema to descend into
    }
    if (DATA_KEYWORDS.has(k)) continue;
    refsInto(v, out, seen);
  }
}

/** What a boundary check reports. */
export interface ValidationResult {
  ok: boolean;
  errors?: string;
}

/**
 * The MINIMAL structural validation seam, declared once at the bottom so every layer that needs a
 * boundary check names the SAME three-line interface and none of them learns about ajv.
 * `@declarative-ai/validate`'s `SchemaValidator` implements it; `exec` and `llm` consume it. Values are
 * JSON by construction at this boundary (a parsed output against a schema document).
 *
 * MAY resolve asynchronously: an inline-schema validator answers synchronously, but a STORE-BACKED one
 * (content-addressed `$ref`s resolved from an artifact store) has reads to do on a cold cache — that is
 * a fact of the id family, not an implementation detail to hide. Every consumer `await`s the result,
 * which costs a sync implementation nothing.
 */
export interface OutputValidator {
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult | Promise<ValidationResult>;
}

/**
 * The SYNC refinement, for consumers that validate mid-walk and cannot suspend (hw's slot validation,
 * the MCP input gate). Their schemas are inline documents, so a sync validator is not a compromise —
 * it is the inline family's truth. Every `SyncOutputValidator` IS an `OutputValidator`.
 */
export interface SyncOutputValidator {
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult;
}

/**
 * Narrow the maybe-async boundary validator to the SYNC seam — FAIL-CLOSED: an implementation that
 * actually suspends is refused with a naming reason rather than treated as a pass, because the sync
 * consumers (mid-walk slot validation, the MCP input gate) sit between an arbitrary payload and a host
 * impl. A genuinely sync validator passes through untouched.
 */
export function syncOnly(v: OutputValidator): SyncOutputValidator {
  return {
    validateValue: (schema, value) => {
      const r = v.validateValue(schema, value);
      return r instanceof Promise
        ? { ok: false, errors: "this consumer requires a synchronous validator (inline schemas); an async (store-backed) validator was injected" }
        : r;
    },
  };
}
