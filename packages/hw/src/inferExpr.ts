/**
 * Expression TYPE INFERENCE (API.md, "Expression language & inference") — the reason the `{ expr }` binding leaf is not a
 * typing hole.
 *
 * The DSL is small (literals, member access, comparison/boolean/arithmetic operators, a
 * conditional) and every namespace it can touch carries a schema, so a result type is computable:
 * member access projects property schemas, each operator has a fixed signature (comparison →
 * boolean, logical → the join of its branches, `!` → boolean), and an unknown reference degrades
 * to the universal schema rather than failing.
 *
 * Semantically an expression IS a pure `FunctionOp` producer whose output schema is the inferred
 * type — so ordinary `isSubschema` binding checking applies to expr leaves with no special case,
 * and a guard is checked by requiring its inference to be `boolean` (strict: no truthiness
 * coercion). A declared `schema` on an expr leaf is an ASSERTION checked against the inferred
 * type, not the only source of typing.
 */
import type { JsonSchema, JsonValue } from "@declarative-ai/exec";
import type { Expr } from "./expr";

/** The universal schema — "any value" (what an unconstrained slot accepts). */
export const ANY_SCHEMA: JsonSchema = {};
const BOOLEAN: JsonSchema = { type: "boolean" };
const NUMBER: JsonSchema = { type: "number" };
const INTEGER: JsonSchema = { type: "integer" };
const STRING: JsonSchema = { type: "string" };
const NULL: JsonSchema = { type: "null" };

/** True when a schema constrains nothing — the inference result for an unknown reference. */
export function isUniversalSchema(s: JsonSchema | undefined): boolean {
  return s === undefined || Object.keys(s).length === 0;
}

/**
 * The typed namespaces an expression may read. Each maps a root name (`inputs`, `children`, …) to
 * a schema for that namespace, so member access is plain property projection. The validator builds
 * this from the state's declared slots; guard-only scalars (`run`, `limits`) get number schemas.
 */
export type ExprScope = Record<string, JsonSchema>;

export interface InferResult {
  schema: JsonSchema;
  /** Reference paths whose target could not be resolved in the scope — reported as errors by the
   *  validator (a typo'd reference is a mistake, not an `any`). */
  unresolved: string[][];
}

/** Infer the result type of a parsed expression against a typed scope. */
export function inferExpression(expr: Expr, scope: ExprScope): InferResult {
  const unresolved: string[][] = [];
  const schema = infer(expr, scope, unresolved);
  return { schema, unresolved };
}

function infer(expr: Expr, scope: ExprScope, unresolved: string[][]): JsonSchema {
  switch (expr.type) {
    case "lit":
      return literalSchema(expr.value);
    case "ident": {
      const s = scope[expr.name];
      if (s === undefined) {
        unresolved.push([expr.name]);
        return ANY_SCHEMA;
      }
      return s;
    }
    case "member": {
      const path = pathOf(expr);
      const base = infer(expr.obj, scope, unresolved);
      const projected = projectProperty(base, expr.prop);
      if (projected === undefined) {
        // Only report a MISSING property when the base was actually typed — projecting off an
        // already-universal schema is legitimately unknown, not a mistake.
        if (!isUniversalSchema(base) && path) unresolved.push(path);
        return ANY_SCHEMA;
      }
      return projected;
    }
    case "unary":
      infer(expr.arg, scope, unresolved);
      return BOOLEAN; // `!x` is boolean whatever `x` is
    case "binary":
      infer(expr.left, scope, unresolved);
      infer(expr.right, scope, unresolved);
      return BOOLEAN; // every modeled binary operator is a comparison
    case "logical": {
      // `a && b` / `a || b` yield one of their operand types (JS semantics), so the result is
      // their join — which for differing types widens to the universal schema.
      const l = infer(expr.left, scope, unresolved);
      const r = infer(expr.right, scope, unresolved);
      return joinSchemas(l, r);
    }
    case "cond": {
      infer(expr.test, scope, unresolved);
      const c = infer(expr.cons, scope, unresolved);
      const a = infer(expr.alt, scope, unresolved);
      return joinSchemas(c, a);
    }
  }
}

/** The dotted path of a pure member chain, or undefined when the base isn't an identifier chain. */
function pathOf(expr: Expr): string[] | undefined {
  if (expr.type === "ident") return [expr.name];
  if (expr.type === "member") {
    const base = pathOf(expr.obj);
    return base ? [...base, expr.prop] : undefined;
  }
  return undefined;
}

/**
 * A literal infers to its EXACT value (`const`), not just its type. This is what lets a conditional
 * over literals — `cond ? 'complete' : 'blocked'` — infer as the enum `["complete","blocked"]` and
 * so satisfy an enum-constrained consumer slot, instead of widening to `string` and being
 * conservatively rejected.
 */
function literalSchema(v: string | number | boolean | null): JsonSchema {
  if (v === null) return NULL;
  const type = typeof v === "string" ? "string" : typeof v === "number" ? (Number.isInteger(v) ? "integer" : "number") : "boolean";
  return { type, const: v };
}

/** Project one property's schema off an object schema. `undefined` = the schema doesn't declare it. */
function projectProperty(base: JsonSchema, prop: string): JsonSchema | undefined {
  // `.length` is the ONE property an array or a string exposes — the evaluator says so, and nothing
  // else about those types is readable in this DSL. Without a projection for it `inputs.items.length`
  // resolved to nothing, which the validator reports as a bad reference: `when: "inputs.items.length
  // > 0"` was unauthorable even though it evaluates perfectly well.
  if (prop === "length" && (base.type === "array" || base.type === "string")) return INTEGER;
  const props = base.properties;
  if (props !== null && typeof props === "object" && !Array.isArray(props)) {
    const p = (props as Record<string, JsonValue>)[prop];
    if (p !== undefined && p !== null && typeof p === "object" && !Array.isArray(p)) return p as JsonSchema;
    if (p !== undefined) return ANY_SCHEMA;
  }
  // An OPEN object (no `properties`, or `additionalProperties` allowed) may carry anything.
  const additional = base.additionalProperties;
  if (additional !== null && typeof additional === "object" && !Array.isArray(additional)) return additional as JsonSchema;
  if (base.type === "object" && props === undefined) return ANY_SCHEMA;
  return undefined;
}

/** The allowed values a schema pins down (`const`/`enum`), or undefined when it is not value-constrained. */
function allowedValues(s: JsonSchema): JsonValue[] | undefined {
  if ("const" in s && s.const !== undefined) return [s.const as JsonValue];
  if (Array.isArray(s.enum)) return s.enum as JsonValue[];
  return undefined;
}

/** The least schema accepting values of both branches. Two value-constrained schemas of the same
 *  type join to the UNION of their values (an enum), preserving the precision that lets an
 *  enum-constrained consumer accept a conditional over literals; otherwise the type is kept, and a
 *  genuine type mismatch widens to any. */
export function joinSchemas(a: JsonSchema, b: JsonSchema): JsonSchema {
  if (JSON.stringify(a) === JSON.stringify(b)) return a;
  if (a.type === b.type && a.type !== undefined) {
    const av = allowedValues(a);
    const bv = allowedValues(b);
    if (av && bv) {
      const values: JsonValue[] = [...av];
      for (const v of bv) if (!values.some((x) => JSON.stringify(x) === JSON.stringify(v))) values.push(v);
      return { type: a.type, enum: values };
    }
    return { type: a.type };
  }
  // `integer` ⊆ `number`, so a mixed numeric join is `number`.
  const numeric = new Set(["integer", "number"]);
  if (typeof a.type === "string" && typeof b.type === "string" && numeric.has(a.type) && numeric.has(b.type)) return NUMBER;
  return ANY_SCHEMA;
}

/** True iff the inferred type is exactly boolean — what a `when` guard must be (§7.2, strict). */
export function isBooleanSchema(s: JsonSchema): boolean {
  return s.type === "boolean";
}
