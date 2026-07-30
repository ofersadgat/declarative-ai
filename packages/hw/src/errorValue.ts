/**
 * Failures in the DATA plane (JaiRA EXPRESSIONS.md §5).
 *
 * A failure is a VALUE WITH A TYPE — always. Different kinds of failure are differently *shaped*
 * (a resolution failure, a policy denial and a provider error carry different detail), but none of
 * them is "not data". Whether one flows into a slot or terminates the operation is therefore not a
 * question about the failure's provenance; it is ordinary type checking against what the consumer
 * declared it accepts.
 *
 * What a failure has always been able to do is travel through the RESULT envelope — an impl resolves
 * with `Result<O, Failure>` rather than throwing, and `runFunction` never throws. What it could not
 * do is travel through the data plane: `resolveRef` returned `{ error }` and failed the state, and a
 * failed `{ result }` record was refused at dispatch with "there is no value to pass". So a workflow
 * could not look at a failure, only die of one.
 *
 * The rule:
 *
 *   **a failure flows into a slot whose declared type accepts its shape; otherwise the operation
 *   terminates with it.**
 *
 * Terminating is the implicit unwrap — the default a language with `Result` spells `?`. Declaring
 * the error type is how an author opts into handling it instead:
 *
 * ```jsonc
 * "schema": { "anyOf": [ { "type": "string" }, "$/types/failure" ] }
 * ```
 *
 * (A string where a schema belongs is a document reference, and `anyOf` items expand — so a shared
 * type library is how the union stays short.)
 */
import type { JsonSchema } from "@declarative-ai/exec";
import { isSubschema, type Schema } from "@declarative-ai/validate";
import { ERROR_CLASSES, ERROR_VALUE_SCHEMA, type Failure } from "@declarative-ai/json";

/** A failure as it appears in the data plane: wrapped, so nothing else can be mistaken for one. */
export interface ErrorValue<D = never> {
  error: Failure<D>;
}

/**
 * A binding that could not resolve, as a failure value.
 *
 * `permanent` because it is deterministic in the sense the classification means — "re-running cannot
 * help (bad input, bad config, unresolved op)". A consumer that declares it accepts one decides what
 * that means; a consumer that does not gets the unwrap, which is what every binding got before
 * failures reached the data plane.
 */
export function resolutionFailure(reason: string): ErrorValue {
  return { error: { classification: "permanent", reason } };
}

/**
 * The schema of ONE PARTICULAR failure — narrow enough that a slot declaring it handles a specific
 * kind accepts it, and a slot declaring a different kind does not.
 *
 * Without this, acceptance could only ever be checked against "any classified failure", so a slot
 * that carefully declared it handles `policy-denied` would refuse an actual policy denial (too
 * narrow to accept the general schema) — the shape distinction would exist in the type system and
 * do nothing at run time.
 */
export function errorValueSchemaFor(failure: Failure): JsonSchema {
  return {
    type: "object",
    properties: {
      error: {
        type: "object",
        properties: { classification: { const: failure.classification }, reason: { type: "string" } },
        required: ["classification", "reason"],
      },
    },
    required: ["error"],
  } as unknown as JsonSchema;
}

/** True for a value shaped like an {@link ErrorValue}. */
export function isErrorValue(v: unknown): v is ErrorValue {
  if (v === null || typeof v !== "object" || Array.isArray(v)) return false;
  const inner = (v as { error?: unknown }).error;
  if (inner === null || typeof inner !== "object" || Array.isArray(inner)) return false;
  const o = inner as { classification?: unknown; reason?: unknown };
  return typeof o.reason === "string" && typeof o.classification === "string" && (ERROR_CLASSES as readonly string[]).includes(o.classification);
}

/**
 * Whether a slot's declared type accepts a failure of the given shape.
 *
 * Two conditions, and both are needed:
 *
 *  1. the slot has a branch that **declares** an error — one requiring `error`; and
 *  2. the wrapped failure is an `isSubschema` of that branch.
 *
 * (2) alone — pure "does the value validate" — is far too permissive: JSON Schema objects are OPEN,
 * so `{ type: "object", properties: { plan } }` accepts `{ error: … }` quite happily, never having
 * said `plan` was required. Routing on that would push failures into precisely the slots nobody
 * thought about, so acceptance has to be DECLARED, not merely survivable.
 *
 * (1) alone would collapse every failure into one kind, so a slot handling a policy denial would
 * also swallow a provider timeout. Keeping the subschema check is what makes "differently shaped by
 * kind" real: a slot declaring the narrow kind refuses the wide one.
 *
 * Note what is NOT here any more: a special case for the unconstrained slot. `{}` has no branch
 * requiring `error`, so it declares no error — which is simply the ordinary reading of what the
 * author wrote, rather than an exception carved out of the rule.
 */
export function admitsError(schema: JsonSchema | undefined, failure: JsonSchema = ERROR_VALUE_SCHEMA as unknown as JsonSchema): boolean {
  if (schema === undefined) return false;
  return errorBranches(schema, 0).some((branch) => isSubschema(failure as Schema, branch as Schema).ok);
}

/**
 * The branches of a slot's schema that DECLARE an error — i.e. require the `error` key.
 *
 * Walked here rather than left to `isSubschema` because the checker does not prove
 * `X ⊆ (A | B)` from `X ⊆ B`: a union on the SUPERTYPE side is not decomposed. A union is the whole
 * spelling of "the value or the error", so the branches have to be visited explicitly.
 */
function errorBranches(schema: JsonSchema, depth: number): JsonSchema[] {
  if (depth > 8) return []; // a pathological schema is not worth chasing
  const out: JsonSchema[] = [];
  const required = (schema as { required?: unknown }).required;
  if (Array.isArray(required) && required.includes("error")) out.push(schema);
  for (const key of ["anyOf", "oneOf", "allOf"] as const) {
    const branches = (schema as Record<string, unknown>)[key];
    if (!Array.isArray(branches)) continue;
    for (const branch of branches) {
      if (branch !== null && typeof branch === "object" && !Array.isArray(branch)) {
        out.push(...errorBranches(branch as JsonSchema, depth + 1));
      }
    }
  }
  return out;
}
