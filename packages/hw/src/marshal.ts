/**
 * The wire boundary: hw's JSON values on one side, a function's TypeScript values on the other
 * (SPEC §7.5.3).
 *
 * > **The JSON Schema is the wire type. Marshalling is the adapter between the wire and TypeScript.**
 *
 * Every static check — `isSubschema`, binding compatibility, expression inference (§6.2) — operates
 * on wire types and is untouched by this file. Nothing here widens or narrows a schema; it converts
 * VALUES at the moment a call is made and again when it returns.
 *
 * ## Why it is schema-driven
 *
 * A marshaller is declared **per leaf type**, and the traversal is derived from the schema's own
 * structure. So `Date` needs one entry and `Date[]`, `{ when: Date }` and `Date | null` need none —
 * which is the difference between a rule that composes and a table of special cases that will be
 * wrong the first time somebody nests a `Date` inside an array inside an object.
 *
 * The schema is also where the marker lives: a `Date` parameter's wire type is
 * `{"type":"string","format":"date-time"}`, and `format` is what says a conversion is due. That
 * keeps the marshalling table out of the type system — a wire schema is an ordinary JSON Schema that
 * an ordinary validator can check, and the adapter reads the same document.
 *
 * ## Why absence is the fast path
 *
 * Most signatures marshal nothing. `needsMarshalling` answers that once per schema and caches it, so
 * an ordinary call does not deep-walk its own arguments to discover there was nothing to do.
 */
import type { JsonValue, SchemaDocument } from "@declarative-ai/json";

export class MarshalError extends Error {}

/** One leaf type that does not travel as itself. */
export interface Marshaller {
  /** The `format` a wire schema carries to mean "this one". */
  readonly format: string;
  /** Wire → runtime, on the way IN to a function. */
  read(wire: JsonValue): unknown;
  /** Runtime → wire, on the way OUT of one. */
  write(value: unknown): JsonValue;
}

/**
 * The table.
 *
 * `Date` is the whole of it today, and the shape is what matters: adding a type means adding an
 * entry, never touching the traversal. Types with no entry AND no JSON form — `bigint`, `symbol`, a
 * function, a class whose identity is its methods — are refused at signature extraction rather than
 * silently mangled here, because a signature that does not describe the call is worse than one that
 * will not compile.
 */
export const MARSHALLERS: readonly Marshaller[] = [
  {
    format: "date-time",
    read: (wire) => {
      if (typeof wire !== "string") throw new MarshalError(`expected an ISO-8601 string for a Date, got ${typeOf(wire)}`);
      const date = new Date(wire);
      if (Number.isNaN(date.getTime())) throw new MarshalError(`'${wire}' is not a valid ISO-8601 date-time`);
      return date;
    },
    write: (value) => {
      if (!(value instanceof Date)) throw new MarshalError(`expected a Date, got ${typeOf(value)}`);
      if (Number.isNaN(value.getTime())) throw new MarshalError("an Invalid Date has no wire form");
      return value.toISOString();
    },
  },
];

const BY_FORMAT: ReadonlyMap<string, Marshaller> = new Map(MARSHALLERS.map((m) => [m.format, m]));

/** The marshaller a schema node names, if any. */
function marshallerFor(schema: SchemaDocument): Marshaller | undefined {
  const format = (schema as { format?: unknown }).format;
  return typeof format === "string" ? BY_FORMAT.get(format) : undefined;
}

/**
 * Does anything under this schema need converting? Cached, because the answer is a property of the
 * schema document and most calls convert nothing.
 */
const needs = new WeakMap<SchemaDocument, boolean>();
export function needsMarshalling(schema: SchemaDocument | undefined): boolean {
  if (schema === undefined || typeof schema !== "object") return false;
  const cached = needs.get(schema);
  if (cached !== undefined) return cached;
  // Seeded `false` before the walk so a self-referential schema terminates: a cycle that has come
  // back to itself contributes nothing beyond what its other branches already said.
  needs.set(schema, false);
  const answer = walkNeeds(schema);
  needs.set(schema, answer);
  return answer;
}

function walkNeeds(schema: SchemaDocument): boolean {
  if (marshallerFor(schema) !== undefined) return true;
  const properties = (schema as { properties?: Record<string, SchemaDocument> }).properties;
  if (properties !== undefined) {
    for (const value of Object.values(properties)) if (needsMarshalling(value)) return true;
  }
  const items = (schema as { items?: SchemaDocument | SchemaDocument[] }).items;
  if (Array.isArray(items)) {
    for (const item of items) if (needsMarshalling(item)) return true;
  } else if (items !== undefined && needsMarshalling(items)) return true;
  for (const key of ["anyOf", "oneOf", "allOf"] as const) {
    const branches = (schema as Record<string, unknown>)[key] as SchemaDocument[] | undefined;
    if (branches !== undefined) for (const branch of branches) if (needsMarshalling(branch)) return true;
  }
  const additional = (schema as { additionalProperties?: SchemaDocument | boolean }).additionalProperties;
  if (additional !== undefined && typeof additional === "object" && needsMarshalling(additional)) return true;
  return false;
}

/** Convert an argument on its way IN to a function: wire value, wire schema, runtime value out. */
export function marshalIn(wire: JsonValue, schema: SchemaDocument | undefined): unknown {
  if (!needsMarshalling(schema)) return wire;
  return convert(wire, schema!, "read");
}

/** Convert a return value on its way OUT: runtime value, wire schema, wire value out. */
export function marshalOut(value: unknown, schema: SchemaDocument | undefined): JsonValue {
  if (!needsMarshalling(schema)) return value as JsonValue;
  return convert(value, schema!, "write") as JsonValue;
}

/**
 * One traversal for both directions.
 *
 * The direction is a parameter rather than two near-identical walkers, because the structure being
 * walked is the same document and the only thing that differs at a leaf is which half of the
 * marshaller runs. Two copies of this would drift the first time a schema keyword was added to one.
 */
function convert(value: unknown, schema: SchemaDocument, direction: "read" | "write"): unknown {
  const marshaller = marshallerFor(schema);
  if (marshaller !== undefined) {
    // `null` passes through both ways: it is how an optional slot spells absence, and a marshaller
    // describes the type a value HAS rather than whether there is one.
    if (value === null || value === undefined) return value;
    return direction === "read" ? marshaller.read(value as JsonValue) : marshaller.write(value);
  }

  if (Array.isArray(value)) {
    const items = (schema as { items?: SchemaDocument | SchemaDocument[] }).items;
    if (items === undefined) return value;
    // Tuple form (`items` an array) positions its schemas; the list form applies one to every entry.
    return value.map((entry, i) => {
      const item = Array.isArray(items) ? items[i] : items;
      return item === undefined ? entry : convert(entry, item, direction);
    });
  }

  if (value !== null && typeof value === "object") {
    const properties = (schema as { properties?: Record<string, SchemaDocument> }).properties;
    const additional = (schema as { additionalProperties?: SchemaDocument | boolean }).additionalProperties;
    if (properties === undefined && (additional === undefined || typeof additional === "boolean")) return value;
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const member = properties?.[key] ?? (typeof additional === "object" ? additional : undefined);
      // `defineProperty` rather than assignment, for the same reason the object literal in `expr`
      // uses it: a `__proto__` key would otherwise invoke the inherited setter and re-parent the
      // result instead of storing a member.
      Object.defineProperty(out, key, {
        value: member === undefined ? entry : convert(entry, member, direction),
        writable: true,
        enumerable: true,
        configurable: true,
      });
    }
    return out;
  }

  return value;
}

function typeOf(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  return `a ${typeof value}`;
}
