/**
 * Schema INFERENCE from data (API.md, "Schema templates, inference, and select-typing"; ported from findmyprompt
 * `src/engine/schema/infer.ts`): determine a type variable's schema from the DATA that flows through
 * it. The rule is "as wide as possible within the kind" — two strings join to `string` (NOT an enum of
 * the literals), two objects to `object` — so the inferred type stays a large space an author can
 * narrow later with an explicit schema, rather than us over-constraining. `null`/`undefined` carry no
 * type information.
 *
 * The join policy is SWAPPABLE: `strict` (the default) errors when two values disagree in kind;
 * `widen` relaxes to the universal `{}` across kinds. `null`/`undefined` is set aside in both — it
 * unifies with anything rather than conflicting.
 */
import type { JsonSchema, ReadonlyJsonValue, SchemaDocument } from "./json";

export type JoinPolicy = "strict" | "widen";

/** The broad JSON-Schema type of one runtime value (its kind's top type); `null`/`undefined` → undefined. */
export function inferValueSchema(value: unknown): JsonSchema | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "string") return { type: "string" };
  if (typeof value === "number") return { type: "number" };
  if (typeof value === "boolean") return { type: "boolean" };
  if (Array.isArray(value)) return { type: "array" };
  if (typeof value === "object") return { type: "object" };
  return {}; // unreachable for JSON, but stay total
}

const kindOf = (s: SchemaDocument): ReadonlyJsonValue | undefined => s.type;

/**
 * Do two `type` keywords name the same kind? STRUCTURAL, not `===`: `type` is a `ReadonlyJsonValue`, so
 * reference equality reports two separately-built `{ type: ["string","null"] }` documents as a CONFLICT
 * — the array form is legal JSON Schema (a union of kinds) and two equal ones must unify. Order is not
 * significant in that union, so the arrays compare as sets of rendered members.
 */
function sameKind(a: ReadonlyJsonValue, b: ReadonlyJsonValue): boolean {
  if (Array.isArray(a) || Array.isArray(b)) {
    const norm = (v: ReadonlyJsonValue): string =>
      (Array.isArray(v) ? [...v] : [v]).map((x) => JSON.stringify(x)).sort().join(",");
    return norm(a) === norm(b);
  }
  return a === b;
}

/** Unify two slot schemas under the policy. Either side `undefined` (no info) yields the other. */
export function joinSchemas(
  a: JsonSchema | undefined,
  b: JsonSchema | undefined,
  policy: JoinPolicy = "strict",
): JsonSchema | undefined {
  if (a === undefined) return b;
  if (b === undefined) return a;
  const ka = kindOf(a);
  const kb = kindOf(b);
  // A schema with no `type` is the universal top `{}` — every value inhabits it, so it ABSORBS rather
  // than conflicting. `inferValueSchema`'s total fallback returns exactly that, and joining "anything"
  // with "string" is "anything": the widest sound answer, which is this module's whole policy.
  if (ka === undefined || kb === undefined) return {};
  if (sameKind(ka, kb)) {
    // Same kind: keep the broad kind type (e.g. two `string`s → `string`, two `object`s → `object`).
    return { type: ka };
  }
  if (policy === "widen") return {}; // cross-kind under widen → the universal top
  throw new Error(`type inference conflict: cannot unify ${JSON.stringify(a)} with ${JSON.stringify(b)}`);
}

/** Infer one variable's schema from a SET of values (e.g. every example's `input`): the join of each
 *  value's broad type, `null`/`undefined` skipped. `undefined` result = no value carried any type. */
export function inferFromValues(values: readonly unknown[], policy: JoinPolicy = "strict"): JsonSchema | undefined {
  let acc: JsonSchema | undefined;
  for (const v of values) acc = joinSchemas(acc, inferValueSchema(v), policy);
  return acc;
}
