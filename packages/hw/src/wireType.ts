/**
 * TypeScript type → wire schema (SPEC §7.5.2, §7.5.3).
 *
 * The half of signature extraction that decides what a parameter's JSON Schema is. It is the lossy
 * step, and how it fails is more important than what it converts.
 *
 * ## It fails CLOSED, in two different ways
 *
 * §6.2's binding compatibility is sound structural subtyping and deliberately conservative — "an
 * unmodeled keyword or a union rejects with a precise reason rather than passing silently". Feeding
 * it schemas machine-translated from a much richer type system is exactly how that strictness turns
 * into either false rejections or false confidence, so the two failures are kept apart:
 *
 *  - **Unrepresentable → an ERROR.** `bigint`, `symbol`, a function, a class whose identity is its
 *    methods. These have no wire form at all, so accepting one produces a signature that does not
 *    describe the call — worse than one that will not compile, because the lie survives.
 *  - **Unconstrained → UNTYPED, with a warning.** `any` and `unknown` become the universal schema,
 *    which §6.2 already defines as accepting anything. The warning names the parameter, because a
 *    slot that accepts anything and is then wired into a typed consumer is the failure §6.2 argues
 *    against for undeclared outputs.
 *
 * ## Marshalled types are not exceptions to that
 *
 * `Date` is representable *because a marshaller exists for it* — its wire type is
 * `{"type":"string","format":"date-time"}` and `marshal.ts` converts at the boundary. A type with
 * neither a JSON form nor a marshaller is refused, so the table is the whole of what makes
 * non-JSON types legal, and adding one means adding a marshaller rather than loosening a check.
 */
import type { JsonSchema, SchemaDocument } from "@declarative-ai/json";
import type * as TS from "typescript";

export class WireTypeError extends Error {}

/** What a conversion reports besides the schema. */
export interface WireTypeResult {
  schema: JsonSchema;
  /** Slots that fell back to the universal schema, and why — never silent. */
  warnings: readonly string[];
}

/** The universal schema: §6.2's "accepts anything". */
const ANY: JsonSchema = {} as JsonSchema;

interface Context {
  ts: typeof TS;
  checker: TS.TypeChecker;
  warnings: string[];
  /** Types currently being converted, so a recursive type terminates rather than recursing. */
  seen: Set<TS.Type>;
  where: string;
}

/**
 * Convert one type to its wire schema.
 *
 * `where` names the thing being converted — a parameter, a return — and appears in every warning and
 * error, because "cannot represent Map" without "of parameter `index`" is a message that sends the
 * reader looking through a whole file.
 */
export function typeToWireSchema(ts: typeof TS, checker: TS.TypeChecker, type: TS.Type, where: string): WireTypeResult {
  const context: Context = { ts, checker, warnings: [], seen: new Set(), where };
  const schema = convert(type, context);
  return { schema, warnings: context.warnings };
}

function convert(type: TS.Type, context: Context): JsonSchema {
  const { ts, checker } = context;
  const flags = type.flags;

  // `any` and `unknown` are DIFFERENT types and the same wire answer: nothing is known about the
  // value. They are separated only in the message, because `any` is usually an author who did not
  // annotate and `unknown` is usually one who meant it.
  if (flags & ts.TypeFlags.Any) {
    context.warnings.push(`${context.where} is 'any', so it is untyped — nothing wired into it will be checked`);
    return ANY;
  }
  if (flags & ts.TypeFlags.Unknown) {
    context.warnings.push(`${context.where} is 'unknown', so it is untyped — nothing wired into it will be checked`);
    return ANY;
  }

  // A type PARAMETER has no instantiation at extraction time, so its constraint is the most that can
  // be said and usually there is none. Generic functions are legal and not usefully typed.
  if (flags & ts.TypeFlags.TypeParameter) {
    context.warnings.push(`${context.where} is a generic type parameter, so it is untyped at this call`);
    return ANY;
  }

  // Literals first: they are narrower than the primitives they belong to, and `cond ? 'a' : 'b'`
  // inferring as an enum rather than widening to `string` is a property §6.2 leans on.
  if (type.isStringLiteral()) return { type: "string", const: type.value } as JsonSchema;
  if (type.isNumberLiteral()) return { type: "number", const: type.value } as JsonSchema;
  if (flags & ts.TypeFlags.BooleanLiteral) {
    return { type: "boolean", const: checker.typeToString(type) === "true" } as JsonSchema;
  }

  if (flags & ts.TypeFlags.String) return { type: "string" } as JsonSchema;
  if (flags & ts.TypeFlags.Number) return { type: "number" } as JsonSchema;
  if (flags & ts.TypeFlags.Boolean) return { type: "boolean" } as JsonSchema;
  if (flags & ts.TypeFlags.Null) return { type: "null" } as JsonSchema;
  // `undefined` and `void` are ABSENCE, not a value. They reach here only inside a union, where the
  // caller strips them into `optional`; standing alone they are a return type of nothing.
  if (flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Void)) return { type: "null" } as JsonSchema;

  if (flags & ts.TypeFlags.BigInt || flags & ts.TypeFlags.BigIntLiteral) {
    throw new WireTypeError(`${context.where} is a bigint, which has no JSON form`);
  }
  if (flags & (ts.TypeFlags.ESSymbol | ts.TypeFlags.UniqueESSymbol)) {
    throw new WireTypeError(`${context.where} is a symbol, which has no JSON form`);
  }
  if (flags & ts.TypeFlags.Never) {
    throw new WireTypeError(`${context.where} is 'never', so no value can ever satisfy it`);
  }

  if (type.isUnion()) return convertUnion(type, context);
  if (type.isIntersection()) {
    // An intersection of object types is a wider object, which the checker has already computed —
    // so the members are read off the intersection itself rather than merged by hand.
    return convertObject(type, context);
  }

  // An ENUM is a union of literals once resolved, so it arrives above; a bare enum type here is the
  // declaration rather than a value.
  if (flags & ts.TypeFlags.EnumLike) return convertUnion(type as TS.UnionType, context);

  if (flags & ts.TypeFlags.Object) return convertObjectLike(type, context);

  throw new WireTypeError(`${context.where} has type '${checker.typeToString(type)}', which has no wire form`);
}

/** `Date` and the rest of the marshalled table, recognised by NAME on the type's symbol. */
function marshalledSchema(type: TS.Type): JsonSchema | undefined {
  const name = type.getSymbol()?.getName();
  if (name === "Date") return { type: "string", format: "date-time" } as JsonSchema;
  return undefined;
}

function convertObjectLike(type: TS.Type, context: Context): JsonSchema {
  const { ts, checker } = context;

  const marshalled = marshalledSchema(type);
  if (marshalled !== undefined) return marshalled;

  if (checker.isArrayType(type)) {
    const element = checker.getTypeArguments(type as TS.TypeReference)[0];
    return { type: "array", items: element === undefined ? ANY : convert(element, context) } as JsonSchema;
  }
  if (checker.isTupleType(type)) {
    const members = checker.getTypeArguments(type as TS.TypeReference).map((t) => convert(t, context));
    return { type: "array", items: members as SchemaDocument[], minItems: members.length, maxItems: members.length } as JsonSchema;
  }

  // A CALLABLE is not data. Refused rather than converted to an empty object, which is what a naive
  // property walk would produce and which would then accept anything.
  if (type.getCallSignatures().length > 0 || type.getConstructSignatures().length > 0) {
    throw new WireTypeError(`${context.where} is a function type, which has no JSON form`);
  }

  const name = type.getSymbol()?.getName();
  if (name === "Map" || name === "Set" || name === "WeakMap" || name === "WeakSet") {
    throw new WireTypeError(
      `${context.where} is a ${name}, which has no JSON form — JSON.stringify(new ${name}()) is '{}'. ` +
        `Use an array or a record, or add a marshaller for it`,
    );
  }
  if (name === "Promise") {
    // A function may be async; its RETURN is what a caller sees, so a Promise reaching here means an
    // unawaited one nested inside a value, which is not data.
    throw new WireTypeError(`${context.where} is a Promise nested inside a value, which has no JSON form`);
  }

  return convertObject(type, context);
}

function convertObject(type: TS.Type, context: Context): JsonSchema {
  const { checker } = context;
  if (context.seen.has(type)) {
    // A recursive type: the cycle contributes nothing the outer pass has not already said, and the
    // universal schema is the honest answer rather than an infinite document. Same cut-off rule §6.2
    // applies to output inference.
    return ANY;
  }
  context.seen.add(type);
  try {
    const properties: Record<string, SchemaDocument> = {};
    const required: string[] = [];
    for (const symbol of checker.getPropertiesOfType(type)) {
      const declaration = symbol.valueDeclaration ?? symbol.declarations?.[0];
      const memberType = declaration
        ? checker.getTypeOfSymbolAtLocation(symbol, declaration)
        : checker.getDeclaredTypeOfSymbol(symbol);
      const name = symbol.getName();
      const inner = { ...context, where: `${context.where}.${name}` };
      const { optional, schema } = stripAbsence(memberType, inner);
      properties[name] = schema;
      if (!optional && (symbol.flags & context.ts.SymbolFlags.Optional) === 0) required.push(name);
    }
    const schema: Record<string, unknown> = { type: "object", properties };
    if (required.length > 0) schema.required = required;
    // Closed by default, which is what makes §6.2's object-width check mean anything: an open object
    // accepts extra members, and a producer that may emit them is not a subtype of one that may not.
    schema.additionalProperties = false;
    return schema as JsonSchema;
  } finally {
    context.seen.delete(type);
  }
}

/**
 * Split `T | undefined` into "optional" plus `T`.
 *
 * JSON Schema has no `undefined`: absence is `required`, and a member typed `T | undefined` is the
 * same wire shape as `T?`. Conflating them at the schema level would make every optional member a
 * union with `null`, which is a different claim.
 */
function stripAbsence(type: TS.Type, context: Context): { optional: boolean; schema: JsonSchema } {
  const { ts } = context;
  if (!type.isUnion()) return { optional: false, schema: convert(type, context) };
  const present = type.types.filter((t) => (t.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Void)) === 0);
  const optional = present.length !== type.types.length;
  if (present.length === 0) return { optional: true, schema: ANY };
  if (present.length === 1) return { optional, schema: convert(present[0]!, context) };
  return { optional, schema: convertUnionOf(present, context) };
}

/**
 * Convert a type that may be ABSENT — a parameter, or an optional member.
 *
 * The companion of {@link typeToWireSchema} for the one position where `T | undefined` means
 * something JSON Schema spells elsewhere: absence is `required`, not a union member.
 */
export function typeToWireSlot(
  ts: typeof TS,
  checker: TS.TypeChecker,
  type: TS.Type,
  where: string,
): WireTypeResult & { optional: boolean } {
  const context: Context = { ts, checker, warnings: [], seen: new Set(), where };
  const { optional, schema } = stripAbsence(type, context);
  return { optional, schema, warnings: context.warnings };
}

function convertUnion(type: TS.UnionType, context: Context): JsonSchema {
  return convertUnionOf(type.types, context);
}

function convertUnionOf(members: readonly TS.Type[], context: Context): JsonSchema {
  const { ts } = context;

  // `boolean` is `true | false` internally. Collapsing it back is not cosmetic: `anyOf` of two
  // consts is a strictly weaker thing for §6.2 to reason about than `{"type":"boolean"}`.
  const isBooleanPair =
    members.length === 2 && members.every((m) => (m.flags & ts.TypeFlags.BooleanLiteral) !== 0);
  if (isBooleanPair) return { type: "boolean" } as JsonSchema;

  const schemas = members.map((m) => convert(m, context));

  // A union of literals of ONE primitive type is an enum, which §6.2 understands directly — and
  // which is what makes `cond ? 'complete' : 'blocked'` satisfy an enum-constrained slot.
  const consts = schemas.map((s) => (s as { const?: unknown; type?: unknown }));
  const allConst = consts.every((s) => s.const !== undefined);
  const oneType = new Set(consts.map((s) => s.type)).size === 1;
  if (allConst && oneType && consts.length > 0) {
    return { type: consts[0]!.type as string, enum: consts.map((s) => s.const) } as JsonSchema;
  }

  // A union carrying a MARSHALLED member alongside anything indistinguishable from it on the wire
  // cannot be decided at a leaf: `Date | string` is `{"type":"string","format":"date-time"}` against
  // `{"type":"string"}`, and nothing in the value says which rule to apply. Refused rather than
  // guessed, per SPEC §7.5.3.
  const formats = schemas.filter((s) => (s as { format?: unknown }).format !== undefined);
  if (formats.length > 0 && schemas.length > formats.length) {
    const bare = schemas.find((s) => (s as { format?: unknown }).format === undefined) as { type?: unknown };
    const marshalledType = (formats[0] as { type?: unknown }).type;
    if (bare?.type === marshalledType) {
      throw new WireTypeError(
        `${context.where} is a union of a marshalled type and a plain '${String(marshalledType)}', ` +
          `which are indistinguishable on the wire — nothing at a leaf can decide which conversion applies`,
      );
    }
  }

  return { anyOf: schemas as SchemaDocument[] } as JsonSchema;
}
