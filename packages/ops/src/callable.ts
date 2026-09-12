/**
 * The CALLABLE type — what a `prompt`- or `function`-kind slot's `schema` holds (hw SPEC §4.1).
 *
 * A callable's type is its I/O contract, and that contract already has a shape: the `input` slot map
 * and the `output` slot an operation declares. So a callable slot's `schema` is that same shape rather
 * than a JSON Schema, and there is no second vocabulary for describing a function. It sits beside
 * {@link Signature} rather than being it: a signature is the LIVE declaration an op carries (slots
 * with bindings, in the ref family), while this is the plain-JSON TYPE a checker compares — stripped
 * of bindings, with a defaulted slot folded into `optional`, and self-describing through `kind`.
 *
 * `kind` is written INTO the schema, not only on the slot around it. An inferred type has no slot to
 * carry a kind on — `.inputs.reviewer` in an expression is a schema and nothing else — so the schema
 * says which of the two callables it types, and `kindFor` reads it back for a slot that left the kind
 * to be derived.
 */
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type { Operation, Parameter, RefFamily, RefKind } from "./model.js";
import { isRequiredSlot } from "./model.js";
import type { SchemaDeref } from "./signatureSchema.js";

export type CallableKind = "prompt" | "function";

/** One parameter of a callable type — a `ParameterDecl` with no binding and no authoring metadata. */
export interface CallableSlot {
  kind?: RefKind;
  schema?: JsonSchema;
  /** Whether a caller may omit it — `optional`, or a default the callee declared. */
  optional?: boolean;
  /** The parameter's POSITION, for positional binding. */
  index?: number;
}

export interface CallableSchema {
  kind: CallableKind;
  /** The parameters, by name. Absent ⇒ the callable accepts any arguments (untyped). */
  input?: Record<string, CallableSlot>;
  /** What it returns. Absent ⇒ the unconstrained type. */
  output?: { kind?: RefKind; schema?: JsonSchema };
}

const CALLABLE_KINDS: ReadonlySet<string> = new Set<CallableKind>(["prompt", "function"]);

export function isCallableKind(kind: unknown): kind is CallableKind {
  return typeof kind === "string" && CALLABLE_KINDS.has(kind);
}

/**
 * Structural guard: is this schema document a CALLABLE type?
 *
 * Decided by `kind` alone. A JSON Schema has no `kind` keyword, so the key cannot collide with one,
 * and an `input`/`output` pair without it is a plain object schema the author forgot to mark — which
 * {@link callableSchemaFor} repairs at load for a slot whose own `kind` says what it is.
 */
export function isCallableSchema(schema: unknown): schema is CallableSchema {
  if (schema === null || typeof schema !== "object" || Array.isArray(schema)) return false;
  return isCallableKind((schema as { kind?: unknown }).kind);
}

/**
 * The callable type an authored slot carries: its `schema` with the slot's `kind` stamped in.
 *
 * A slot writes `{ "kind": "function", "schema": { "input": …, "output": … } }` — the kind on the
 * slot, as every kind is written — and the schema becomes self-describing here so that everything
 * downstream (inference, the checker, a scope built over the slots) reads one form. A slot with no
 * schema is "some callable": typed by kind alone, accepting anything, returning the unconstrained type.
 */
export function callableSchemaFor(kind: CallableKind, schema: JsonSchema | undefined): CallableSchema {
  if (schema === undefined) return { kind };
  const s = schema as { input?: unknown; output?: unknown };
  const out: CallableSchema = { kind };
  if (s.input !== null && typeof s.input === "object" && !Array.isArray(s.input)) out.input = s.input as Record<string, CallableSlot>;
  if (s.output !== null && typeof s.output === "object" && !Array.isArray(s.output)) out.output = s.output as CallableSchema["output"];
  return out;
}

/** One slot of a live operation, as a callable type's parameter — bindings stripped, a default folded in. */
function slotOf<F extends RefFamily>(parameter: Parameter<F>, deref: SchemaDeref<F>): CallableSlot {
  const out: CallableSlot = {};
  if (parameter.kind !== undefined) out.kind = parameter.kind;
  const schema = parameter.schema !== undefined ? deref(parameter.schema) : undefined;
  if (schema !== undefined && Object.keys(schema).length > 0) out.schema = schema;
  if (!isRequiredSlot(parameter)) out.optional = true;
  if (parameter.index !== undefined) out.index = parameter.index;
  return out;
}

/**
 * The callable type of an OPERATION — what an uncalled reference to it infers to (hw SPEC §6.2).
 *
 * `map(xs, classify)` passes `classify` as a value, and that value's type is this: the parameters the
 * operation declares and what it returns, exactly as a call through the reference would be checked
 * against. Plain JSON, so it can sit in a scope and be compared by the ordinary rules.
 */
export function callableSchemaOf<F extends RefFamily>(op: Operation<F>, deref: SchemaDeref<F>): CallableSchema {
  const input: Record<string, CallableSlot> = {};
  for (const [name, parameter] of Object.entries(op.input)) input[name] = slotOf(parameter, deref);
  const output = slotOf(op.output, deref);
  const out: CallableSchema = { kind: op.kind, input };
  const outputType: CallableSchema["output"] = {};
  if (output.kind !== undefined) outputType.kind = output.kind;
  if (output.schema !== undefined) outputType.schema = output.schema;
  if (Object.keys(outputType).length > 0) out.output = outputType;
  return out;
}

/**
 * The order a callable binds POSITIONAL arguments in: by declared `index`, else declaration order —
 * the same rule an operation's slots follow, so a call through a value and a call by name agree.
 */
export function callablePositionalOrder(schema: CallableSchema): string[] {
  const entries = Object.entries(schema.input ?? {});
  const indexed = entries.filter(([, slot]) => slot.index !== undefined);
  if (indexed.length === 0) return entries.map(([name]) => name);
  return indexed.sort(([, a], [, b]) => (a.index ?? 0) - (b.index ?? 0)).map(([name]) => name);
}

/** The callable type as a plain JSON value — for placing it in a scope or a schema's `properties`. */
export function callableSchemaAsJson(schema: CallableSchema): JsonValue {
  return schema as unknown as JsonValue;
}
