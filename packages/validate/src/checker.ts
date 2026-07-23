/**
 * THE binding checker (API.md, "The binding checker"). findmyprompt's `checker.ts` and hw's `validate.ts`
 * were the same checker written twice: both walk an operation's input `Parameter`s and call
 * `isSubschema` on producer-vs-consumer. This is that checker, once, parameterized by the ref family
 * `F` with injectable resolution — the id family resolves through stores, the inline family passes
 * documents directly.
 *
 * What it proves, before anything runs and before anything costs money:
 *
 *  1. **Binding compatibility** — every producer's output schema is an `isSubschema` of the consuming
 *     slot's schema. Sound structural subtyping, conservative otherwise: an unmodeled keyword or a
 *     union REJECTS with a precise reason, never a silent pass.
 *  2. **Kind agreement** — a `prompt`/`function` slot (higher-order: the op DEFINITION is the value)
 *     must be fed an op of that kind; a data slot must be fed a data producer.
 *  3. **The positional-`index` invariant** — per op, `index` is all-set-or-all-unset and distinct,
 *     because it is the sort key for bare/tuple ingestion and a partial or duplicated set makes that
 *     ambiguous. (A map can be malformed this way; an ordered array could not.)
 *  4. **Signature<InlineFamily> conformance**, when one is supplied — the op produces the signature's output type.
 *
 * Everything family-specific — how a local producer name resolves, what a synthesized resolver
 * function's output type is, whether a producer is proven to have RUN on every path — arrives through
 * {@link CheckerHooks}. That is the whole difference between the two checkers this replaces.
 */
import type { JsonSchema, JsonValue, SchemaDocument } from "@declarative-ai/json";
import type { InlineFamily, Operation, Parameter, Ref, RefFamily, Signature } from "@declarative-ai/ops";
import { isSubschema, type ResolveRef, type Schema } from "./subtype";

export interface CheckIssue {
  /** Where in the operation, e.g. `operation.input.plan` or `children.critique.inputs.plan_doc`. */
  path: string;
  message: string;
}

export interface CheckResult {
  ok: boolean;
  errors: CheckIssue[];
}

/**
 * The family-specific knowledge the checker needs. Only `producer` is mandatory: everything else has a
 * sound default, so the inline family can pass almost nothing.
 */
export interface CheckerHooks<F extends RefFamily> {
  /** The schema DOCUMENT a slot's `schema` leaf denotes. Inline: it IS the document (the default).
   *  Id family: a store lookup, pre-loaded so this stays synchronous. */
  schemaOf?(slot: F["schema"] | undefined): JsonSchema | undefined;
  /** Resolve a producer reference to its operation. Inline: an embedded op, or a declared child looked
   *  up by local key. Id family: a store lookup by content id. */
  producer(ref: F["op"]): Operation<F> | undefined;
  /**
   * Family-specific typing of a producer operation, consulted BEFORE the generic
   * "its declared output schema is its type" rule. This is where hw types the resolver functions its
   * loader synthesizes (`expr.eval`, `select`, `scope.get`, …). Return `undefined` to fall through.
   */
  producerSchema?(op: Operation<F>, path: string, report: (message: string) => void): JsonSchema | undefined;
  /** `$ref` resolution for the subtype check (content-addressed schema graphs). */
  resolveRef?: ResolveRef;
  /**
   * Whether a producer named by `ref` is proven to have RUN on every path to this evaluation point.
   * Absent ⇒ not checked. A consuming slot that has opted out (a declared `default`, or `optional`)
   * passes `optOut` and skips the proof obligation.
   */
  reachable?(ref: F["op"]): boolean;
}

/** The schema an inline JSON literal satisfies — precise enough for the subschema check, so a literal
 *  also satisfies an `enum`/`const`-constrained consumer. */
export function schemaOfValue(v: JsonValue): JsonSchema {
  if (v === null) return { type: "null" };
  // `const` on the container forms too, so a literal array/object is a SINGLETON type the subtype
  // checker can decide against the consumer's `items`/bounds/`additionalProperties` directly. Without
  // it, a literal was compared schema-to-schema and had to restate constraints it never carries —
  // which rejected `{ json: ["a","b"] }` bound into an `array of string` slot.
  if (Array.isArray(v)) return { type: "array", const: v };
  switch (typeof v) {
    case "string":
      return { type: "string", const: v };
    case "number":
      return { type: Number.isInteger(v) ? "integer" : "number", const: v };
    case "boolean":
      return { type: "boolean", const: v };
    default: {
      // An object literal: describe it exactly (every key present and required), so it satisfies any
      // consumer that requires a subset of these properties.
      const properties: Record<string, JsonValue> = {};
      for (const [k, val] of Object.entries(v)) properties[k] = schemaOfValue(val) as JsonValue;
      // `const` alongside the structural description: the value form is what lets a closed or bounded
      // consumer be DECIDED, while `properties`/`required` keep the schema readable and still serve any
      // consumer that only looks at structure.
      return { type: "object", properties, required: Object.keys(v), const: v };
    }
  }
}

/**
 * A schema that constrains nothing — `{}`, or a document with no constraining keyword.
 *
 * ⚠️ NOT a pre-filter for the subschema check, and it used to be one. Its keyword list is the set this
 * checker knows how to READ, which is not the set `isSubschema` accepts: every consumer keyword the
 * subtype checker deliberately REJECTS (`anyOf`, `oneOf`, `not`, an unresolved `$ref`, `pattern`,
 * `minItems`, `uniqueItems`, …) is absent from that list, so such a consumer classified as "constrains
 * nothing" and the check short-circuited to a silent PASS — the exact opposite of the module's "never a
 * silent pass" contract. `isSubschema` decides universality itself (`subtype.ts`'s `isUniversal`, which
 * returns OK for a genuinely unconstrained consumer), so the binding path just calls it.
 *
 * Still exported: a caller that wants the cheap "is this slot worth describing" question — hw asks it
 * about inferred expression types — is asking something different from "does this producer fit".
 */
export function isUniversalSchema(s: SchemaDocument | undefined): boolean {
  if (!s) return true;
  return !["type", "enum", "const", "properties", "required", "items", "minimum", "maximum", "minLength", "maxLength", "additionalProperties", "x-type"].some(
    (k) => k in s,
  );
}

/** Check ONE operation's bindings. */
export function checkOperation<F extends RefFamily>(
  op: Operation<F>,
  hooks: CheckerHooks<F>,
  opts: { path?: string; signature?: Signature<InlineFamily>; seen?: Set<Operation<F>> } = {},
): CheckResult {
  const errors: CheckIssue[] = [];
  const seen = opts.seen ?? new Set<Operation<F>>();
  checkInto(op, hooks, opts.path ?? "operation", errors, seen);
  if (opts.signature) checkSignature(op, opts.signature, hooks, errors, opts.path ?? "operation");
  return { ok: errors.length === 0, errors };
}

/**
 * Check ONE binding against the schema of the slot it fills — the entry point hw uses directly for
 * child-input wiring, which is a binding that does not live on an operation.
 */
export function checkBinding<F extends RefFamily>(
  binding: Ref<F>,
  consumerSchema: JsonSchema | undefined,
  hooks: CheckerHooks<F>,
  path: string,
  opts: { optOut?: boolean; kind?: Parameter<F>["kind"] } = {},
): CheckIssue[] {
  const errors: CheckIssue[] = [];
  const producer = producerSchemaOf(binding, hooks, path, errors, opts.optOut === true, opts.kind);
  // Nothing to compare against is not a pass, it is silence: an unresolvable producer or an undeclared
  // slot type leaves nothing to decide. A DECLARED consumer schema always goes to `isSubschema`, even a
  // seemingly empty one — deciding "this constrains nothing" is that checker's job, and doing it here
  // with a different keyword list is how every rejection it models became a silent accept.
  if (producer === undefined || consumerSchema === undefined) return errors;
  const result = isSubschema(producer as Schema, consumerSchema as Schema, hooks.resolveRef);
  if (!result.ok) errors.push({ path, message: `wiring is not type-compatible with the slot: ${result.reason}` });
  return errors;
}

function checkInto<F extends RefFamily>(
  op: Operation<F>,
  hooks: CheckerHooks<F>,
  path: string,
  errors: CheckIssue[],
  seen: Set<Operation<F>>,
): void {
  if (seen.has(op)) return; // shared sub-graph already checked
  seen.add(op);

  checkIndexInvariant(op.input, errors, path);

  for (const [name, param] of Object.entries(op.input)) {
    const binding = param.binding as Ref<F> | undefined;
    if (binding === undefined) continue; // free slot — filled (and runtime-typed) at call time
    const slotPath = `${path}.input.${name}`;
    errors.push(...checkBinding(binding, schemaOf(hooks, param.schema), hooks, slotPath, { kind: param.kind }));
    // Recurse into an embedded producer so a nested graph is checked too.
    if (typeof binding === "object" && binding !== null && "op" in binding) {
      const inner = hooks.producer(binding.op as F["op"]);
      if (inner) checkInto(inner, hooks, slotPath, errors, seen);
    }
  }
}

/** Per-op `index` invariant: all-set-or-all-unset, and distinct when set. */
function checkIndexInvariant<F extends RefFamily>(
  input: { [name: string]: Parameter<F> },
  errors: CheckIssue[],
  path: string,
): void {
  const entries = Object.entries(input);
  const withIndex = entries.filter(([, p]) => typeof p.index === "number");
  if (withIndex.length === 0) return;
  if (withIndex.length !== entries.length) {
    errors.push({
      path,
      message: `input 'index' must be set on all slots or none (${withIndex.length}/${entries.length} set)`,
    });
  }
  const seen = new Set<number>();
  for (const [name, p] of withIndex) {
    if (seen.has(p.index!)) errors.push({ path, message: `duplicate input 'index' ${p.index} (slot '${name}')` });
    seen.add(p.index!);
  }
}

/**
 * Signature<InlineFamily> CONFORMANCE: the op must PRODUCE the signature's output type — kind, plus (for a data
 * kind) a subtype of `O`. The input side is a single type `I` the op consumes through its own free
 * slots, so input conformance is not a per-field match.
 */
function checkSignature<F extends RefFamily>(
  op: Operation<F>,
  signature: Signature<InlineFamily>,
  hooks: CheckerHooks<F>,
  errors: CheckIssue[],
  path: string,
): void {
  if (op.output.kind !== signature.output.kind) {
    errors.push({ path: `${path}.output`, message: `output kind '${op.output.kind}' != signature '${signature.output.kind}'` });
    return;
  }
  const slot = schemaOf(hooks, op.output.schema);
  const sig = signature.output.schema;
  // Same rule as `checkBinding`: a declared signature output goes to `isSubschema`, which is the one
  // place that decides whether a schema constrains anything.
  if (!slot || !sig) return;
  const r = isSubschema(slot as Schema, sig as Schema, hooks.resolveRef);
  if (!r.ok) errors.push({ path: `${path}.output`, message: `signature: ${r.reason}` });
}

function schemaOf<F extends RefFamily>(hooks: CheckerHooks<F>, slot: F["schema"] | undefined): JsonSchema | undefined {
  if (hooks.schemaOf) return hooks.schemaOf(slot);
  // Inline default: the leaf IS the document.
  return slot as JsonSchema | undefined;
}

/**
 * The output schema a binding produces, or `undefined` when nothing can be said (which is not an error
 * — it just means the subschema check is skipped). Reports kind and reachability problems as a side
 * effect.
 *
 * Exported because a family with SYNTHESIZED producers needs to recurse into it: hw's `select`
 * resolver types its output by projecting a property off the schema of the producer feeding its `value`
 * slot, which is this function applied one level down.
 */
export function producerSchemaOf<F extends RefFamily>(
  binding: Ref<F>,
  hooks: CheckerHooks<F>,
  path: string,
  errors: CheckIssue[],
  optOut: boolean,
  consumerKind?: Parameter<F>["kind"],
): JsonSchema | undefined {
  const err = (message: string): void => {
    errors.push({ path, message });
  };
  // A literal's type IS its value.
  if ("text" in binding) return { type: "string", const: binding.text as JsonValue };
  if ("json" in binding) return schemaOfValue(binding.json as JsonValue);
  if ("blob" in binding) return { type: "string", contentEncoding: "base64" };
  if ("result" in binding || "refs" in binding) return undefined; // resolved values; nothing static to say
  if (!("op" in binding)) return undefined;

  const producer = hooks.producer(binding.op as F["op"]);
  if (!producer) {
    err("references a producer that does not resolve");
    return undefined;
  }
  // Reachability: the edge's TYPE is always known; whether the producer has RUN by the time the edge
  // resolves is a control-flow property, and an unproven one is an error unless the slot opted out.
  if (!optOut && hooks.reachable && !hooks.reachable(binding.op as F["op"])) {
    err(
      "references a producer that is not proven to have run on every path to this point" +
        " — order it before this use, or declare a `default` on the consuming slot",
    );
  }
  // Higher-order: a `prompt`/`function` slot takes the op DEFINITION as its value, so the check is on
  // KIND, not on the producer's output type.
  if (consumerKind === "prompt" || consumerKind === "function") {
    if (producer.kind !== consumerKind) {
      err(`expects a ${consumerKind} op but the producer is a ${producer.kind} op`);
    }
    return undefined;
  }
  return hooks.producerSchema?.(producer, path, err) ?? schemaOf(hooks, producer.output.schema);
}
