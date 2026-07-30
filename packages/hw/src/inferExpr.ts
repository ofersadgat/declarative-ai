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
import type { InlineFamily, JsonSchema, JsonValue, Ref } from "@declarative-ai/exec";
import type { Expr } from "./expr";
import { RESOLVER_REFS } from "./format";
import { pathOfRef } from "./lowerExpr";

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

/**
 * Infer the result type of a LOWERED expression — the same inference over the producer tree
 * (EXPRESSIONS.md §1) rather than over the AST.
 *
 * This is what keeps `{ expr }` from becoming a typing hole once it stops carrying a source string:
 * the inferred type IS the leaf's producer schema, so ordinary `isSubschema` binding checking still
 * applies with no special case, and a guard is still checked by requiring `boolean` strictly.
 *
 * The rules are the AST ones, reached structurally — every operator projects, joins or returns
 * boolean exactly as its resolver computes, because both read the same `projectProperty` and
 * `joinSchemas`. Anything that is not one of the operator resolvers (a literal leaf aside) infers to
 * the universal schema: an embedded operation carries its own contract, and guessing at one here
 * would be worse than admitting it is unknown.
 */
export function inferRef(ref: Ref<InlineFamily>, scope: ExprScope): InferResult {
  const unresolved: string[][] = [];
  const schema = inferOne(ref, scope, unresolved);
  return { schema, unresolved };
}

function inferOne(ref: Ref<InlineFamily>, scope: ExprScope, unresolved: string[][]): JsonSchema {
  if ("text" in ref) return literalSchema(ref.text);
  if ("json" in ref) {
    const v = ref.json;
    return v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean" ? literalSchema(v) : ANY_SCHEMA;
  }
  if (!("op" in ref)) return ANY_SCHEMA;
  const producer = ref.op;
  // A local child key names a producer whose schema is the CONSUMER's business, not the expression's.
  if (typeof producer === "string" || producer.kind !== "function") return ANY_SCHEMA;
  const operand = (name: string): JsonSchema => {
    const binding = producer.input[name]?.binding;
    return binding === undefined ? ANY_SCHEMA : inferOne(binding, scope, unresolved);
  };

  switch (producer.functionRef) {
    case RESOLVER_REFS.context: {
      const binding = producer.input.name?.binding;
      const name = binding !== undefined && "text" in binding ? binding.text : undefined;
      if (name === undefined) return ANY_SCHEMA;
      const s = scope[name];
      if (s === undefined) {
        unresolved.push([name]);
        return ANY_SCHEMA;
      }
      return s;
    }
    case RESOLVER_REFS.member: {
      const propBinding = producer.input.prop?.binding;
      const prop = propBinding !== undefined && "text" in propBinding ? propBinding.text : undefined;
      const base = operand("value");
      if (prop === undefined) return ANY_SCHEMA;
      const projected = projectProperty(base, prop);
      if (projected === undefined) {
        // Only report a MISSING property when the base was actually typed — projecting off an
        // already-universal schema is legitimately unknown, not a mistake.
        if (!isUniversalSchema(base)) {
          const path = pathOfRef(ref);
          if (path) unresolved.push(path);
        }
        return ANY_SCHEMA;
      }
      return projected;
    }
    case RESOLVER_REFS.not:
      operand("value");
      return BOOLEAN; // `!x` is boolean whatever `x` is
    case RESOLVER_REFS.eq:
    case RESOLVER_REFS.ne:
    case RESOLVER_REFS.strictEq:
    case RESOLVER_REFS.strictNe:
    case RESOLVER_REFS.lt:
    case RESOLVER_REFS.le:
    case RESOLVER_REFS.gt:
    case RESOLVER_REFS.ge:
      operand("left");
      operand("right");
      return BOOLEAN; // every modeled binary operator is a comparison
    case RESOLVER_REFS.and:
    case RESOLVER_REFS.or:
      // `a && b` / `a || b` yield one of their operand types (JS semantics), so the result is their
      // join — which for differing types widens to the universal schema.
      return joinSchemas(operand("left"), operand("right"));
    case RESOLVER_REFS.cond:
      operand("test");
      return joinSchemas(operand("then"), operand("else"));
    default:
      return ANY_SCHEMA;
  }
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
    case "apply": {
      // Every argument is inferred whatever the operation is, so a bad reference inside one is still
      // reported. The RESULT depends on which operation: the built-ins have known signatures, and
      // anything else is a reference to an operation document whose declared output is not in this
      // scope — unknown rather than wrong, until the loader has resolved it.
      const args = expr.args.map((a) => infer(a, scope, unresolved));
      switch (expr.op) {
        case RESOLVER_REFS.not:
          return BOOLEAN; // `!x` is boolean whatever `x` is
        case RESOLVER_REFS.and:
        case RESOLVER_REFS.or:
          // These yield one of their OPERAND types (JS semantics), so the result is their join —
          // which for differing types widens to the universal schema.
          return joinSchemas(args[0] ?? ANY_SCHEMA, args[1] ?? ANY_SCHEMA);
        case RESOLVER_REFS.cond:
          return joinSchemas(args[1] ?? ANY_SCHEMA, args[2] ?? ANY_SCHEMA);
        case RESOLVER_REFS.eq:
        case RESOLVER_REFS.ne:
        case RESOLVER_REFS.strictEq:
        case RESOLVER_REFS.strictNe:
        case RESOLVER_REFS.lt:
        case RESOLVER_REFS.le:
        case RESOLVER_REFS.gt:
        case RESOLVER_REFS.ge:
          return BOOLEAN; // every built-in comparison is one
        default:
          return ANY_SCHEMA;
      }
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
    // OWN properties only: an inherited hit (`constructor`, `toString`) is not a declared property,
    // and reporting it as one typed `inputs.o.constructor` as `ANY` instead of as a bad reference.
    const p = Object.hasOwn(props, prop) ? (props as Record<string, JsonValue>)[prop] : undefined;
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
