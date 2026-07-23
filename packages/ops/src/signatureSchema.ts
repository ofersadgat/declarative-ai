/**
 * The `Signature` ⇄ schema bridge (API.md, "The `Signature` ⇄ schema bridge"; ported from findmyprompt
 * `src/engine/schema/signatureSchema.ts`). It lands in `ops` rather than `json` because a `Signature`
 * is part of the OP model — the pure schema/data transforms it composes with (`applyTemplate`,
 * `resolveTypes`) sit one layer down in `json`, with no model dependency at all.
 *
 * A `Signature` is an op's I/O contract: one input type `I`, one output type `O`. These are the
 * value-level witnesses `Schema<I>` / `Schema<O>` — a generic op instantiates its output template by
 * binding `$input := signatureInputSchema(sig)` and `$output := signatureOutputSchema(sig)`.
 *
 * Every function here takes a DEREF — how this family turns its `schema` member into a schema document.
 * The inline family's is the identity ({@link inlineDeref}); the id family's is a lookup into whatever
 * holds its artifacts. It is SYNC on purpose: an async deref would infect `resolveTypes` and every
 * consumer above it, and hydration is the caller's problem (the same rule §7.2 applies to blobs) — an
 * id-family caller pre-loads the schemas it needs and passes a map lookup.
 */
import type { JsonSchema, SchemaDocument } from "@declarative-ai/json";
import { resolveTypes } from "@declarative-ai/json";
import type { Parameter, RefFamily, Signature } from "./model";

/** How a family turns the `schema` a parameter carries into a schema document. */
export type SchemaDeref<F extends RefFamily> = (schema: F["schema"]) => JsonSchema;

/** The inline family carries the schema document itself, so its deref is the identity. */
export const inlineDeref: SchemaDeref<{ schema: JsonSchema } & RefFamily> = (schema) => schema;

/**
 * The schema of one parameter's type: the `schema` carried on the parameter when present (dereferenced by the
 * family), else the kind's broad type — `text` → a string, `blob` → a media string, untyped `json` →
 * the universal schema (accept any JSON).
 */
export function parameterSchema<F extends RefFamily>(parameter: Parameter<F>, deref: SchemaDeref<F>): JsonSchema {
  if (parameter.schema !== undefined) return deref(parameter.schema);
  if (parameter.kind === "text") return { type: "string" };
  if (parameter.kind === "blob") return { type: "string", contentEncoding: "base64" };
  return {}; // untyped json — the universal schema
}

/** `Schema<I>`: the input value's type under `signature`. */
export function signatureInputSchema<F extends RefFamily>(signature: Signature<F>, deref: SchemaDeref<F>): JsonSchema {
  return parameterSchema(signature.input, deref);
}

/** `Schema<O>`: the output value's type under `signature`. */
export function signatureOutputSchema<F extends RefFamily>(signature: Signature<F>, deref: SchemaDeref<F>): JsonSchema {
  return parameterSchema(signature.output, deref);
}

/**
 * Build a TYPE MAP from a `Signature` (`{ input: Schema<I>, output: Schema<O> }`) and resolve
 * `template` against it via the generic `resolveTypes`. A signature is just one way to produce the
 * map; a runner may build one directly. With NO signature the map is empty, so every type variable is
 * unbound and collapses to the universal `{}` ("any") — the generic floor: an undetermined variable is
 * `any`, never an error.
 */
export function bindSignatureTemplate<F extends RefFamily>(
  template: SchemaDocument,
  deref: SchemaDeref<F>,
  signature?: Signature<F>,
): JsonSchema {
  const typeMap: Record<string, SchemaDocument> = signature
    ? { input: signatureInputSchema(signature, deref), output: signatureOutputSchema(signature, deref) }
    : {};
  return resolveTypes(template, typeMap);
}

/**
 * Structural guard: is `v` a `Signature` VALUE? Used to discover the target signature carried in a
 * signature-dependent op's input, so the output template binds without a bespoke per-task handler.
 * Lenient on extra keys; strict on shape.
 *
 * Structural checking can only confirm the shape every family shares — a `kind` on each parameter and a
 * `name` on the output — so the caller names the family it is reading.
 */
export function asSignature<F extends RefFamily>(v: unknown): Signature<F> | undefined {
  if (!v || typeof v !== "object" || Array.isArray(v)) return undefined;
  const o = v as { input?: unknown; output?: unknown };
  const isParameter = (p: unknown): boolean => {
    if (!p || typeof p !== "object" || Array.isArray(p)) return false;
    const kind = (p as Parameter<F>).kind;
    return kind === "text" || kind === "json" || kind === "blob";
  };
  if (!isParameter(o.input)) return undefined;
  if (!isParameter(o.output) || typeof (o.output as { name?: unknown }).name !== "string") return undefined;
  return v as Signature<F>;
}

/**
 * The reflective meta-schema: what a `Signature` VALUE looks like. This is the witness that makes a
 * signature a first-class typed datum — so an op whose input is itself a `Signature` has a real schema
 * to bind as `Schema<I>` where `I = Signature`.
 *
 * The `schema` of each parameter is a `$param` HOLE — `input.schema` is the type variable `I`,
 * `output.schema` is `O` — the same holes a dependent row template carries. So a signature's
 * input/output type and its instances' input/output are literally one type variable in the stored
 * schema: "they match" is a fact of the data, not a render-time transform.
 */
const PARAMETER_KINDS = ["text", "json", "blob"] as const;

const INPUT_PARAMETER_SCHEMA: JsonSchema = {
  type: "object",
  title: "signature-parameter",
  properties: {
    kind: { type: "string", enum: [...PARAMETER_KINDS] },
    schema: { $param: "input" },
    index: { type: "number" },
  },
  required: ["kind"],
};

const OUTPUT_PARAMETER_SCHEMA: JsonSchema = {
  type: "object",
  title: "named-signature-parameter",
  properties: {
    name: { type: "string" },
    kind: { type: "string", enum: [...PARAMETER_KINDS] },
    schema: { $param: "output" },
    index: { type: "number" },
  },
  required: ["name", "kind"],
};

export const SIGNATURE_META_SCHEMA: JsonSchema = {
  type: "object",
  title: "signature",
  properties: {
    input: INPUT_PARAMETER_SCHEMA,
    output: OUTPUT_PARAMETER_SCHEMA,
  },
  required: ["input", "output"],
};
