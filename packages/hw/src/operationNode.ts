/**
 * The `operation.*` expression namespace (SPEC.md §6.1).
 *
 * A state's operation was not addressable. Its result landed straight in the state's declared output
 * slots, while a CHILD was addressable as `children.<key>.outputs` — and that asymmetry is why
 * engine metadata about the call (what it cost, which model actually served it, how many attempts it
 * took, which conversation position it ended at) had nowhere to live. It was reachable only by
 * reading the events journal, which is not a thing an expression can do.
 *
 * So the operation node gets its own namespace: `operation.*` for the state's own call, and
 * `children.<key>.operation.*` for a child's.
 *
 * ## It is a NAMESPACE, not a child
 *
 * Making the operation a child would have reused machinery that already exists — and would have
 * perturbed the instance tree, which is what `run.cursor`, `run.position` and `sequence` are all
 * defined against. "The operation" is not a position in the sequence, and pretending otherwise makes
 * three unrelated expressions ambiguous.
 *
 * ## `outputs` is what the CALL returned — `session` is one of them
 *
 * `operation.output` is the operation's own output namespace: the values the call produced, under
 * the names it produced them, plus `session` on a prompt op — the conversation position the call
 * ended at is simply one more thing a prompt operation outputs.
 *
 * This reverses an earlier split. `outputs` used to hold engine metadata ALONE, on the reasoning
 * that a state producing an output named `session` would otherwise shadow the engine's. That kept
 * the two apart at the cost of the operation's actual result having no address at all: it landed
 * directly in the state's declared slots, so there was no way to bind an output to it, rename it, or
 * put an expression between the call and the slot. Naming collisions are a smaller problem than an
 * unaddressable value, and they are the author's to resolve — `session` is a name they can see.
 *
 * The state's own `outputs.*` still exist and still mean "this state's outputs". What changes is
 * that they are now DOWNSTREAM of this namespace rather than the same thing wearing two names.
 *
 * ## Why the END position, and why there is no start marker
 *
 * You append AT a position but do not know where the call ended until the provider resolves, so the
 * end marker is the only value that can exist by the time an expression reads it — and it is the one
 * consumers want, since "append after me" and "fork after me" both mean *after*.
 *
 * There is deliberately no start marker. Recovery after an error does not need one: restart the
 * state, and DESIGN.md §1.6's instance-scoped resolution re-resolves the binding to the position the
 * failed attempt started from, which has since been appended to — so it forks, from exactly the
 * right place.
 *
 * ## Granularity
 *
 * Authored forking is PER-OPERATION. If one agentic call appends forty entries, a workflow cannot
 * branch at entry twenty; the store addresses finer positions so a human can scrub a transcript in
 * the UI, but the expression language exposes operation boundaries only.
 */
import type { JsonSchema, JsonValue } from "@declarative-ai/exec";

/** Terminal outcomes an operation can report — mirrors `children.<key>.outcome`. */
export const OPERATION_OUTCOMES = ["success", "error", "timeout", "canceled"] as const;

/**
 * The fields of `operation.*` that describe the CALL rather than carry its result — how it went,
 * what it cost, which model served it. Everything the call actually produced lives under `output`.
 *
 * Named as a set because a second reader needs it: the fan-out tally has to know that reading one of
 * these consumes no producer output, exactly as `children.<key>.outcome` does not (`fanout.ts`).
 * Counting them would force a materialization nothing needs.
 */
export const OPERATION_METADATA_FIELDS: ReadonlySet<string> = new Set(["outcome", "usage", "cost", "model"]);

/**
 * The one name under `operation.output` the ENGINE writes rather than the call returning it: the
 * conversation position a prompt call ended at.
 *
 * It shares the namespace with the call's own outputs (see the header), which is why the tally needs
 * it by name — `operation.output.<name>` is a read of the returned value, but this one particular
 * name is a session ref and can never be a stream.
 */
export const OPERATION_ENGINE_OUTPUT = "session";

/**
 * What `operation.*` exposes, as a typed union over the operation's kind.
 *
 * The common core is on every kind. `outputs.session` is PROMPT-only, and that is the point of
 * typing it rather than filling in `undefined`: `operation.output.session` written against a `ui`
 * gate is an authoring error the loader reports, not a binding that silently resolves to nothing and
 * fails somewhere downstream.
 */
export interface OperationNodeCore {
  outcome?: string;
  /** The measurement record, passed through rather than re-shaped, so a metric an executor starts
   *  reporting reaches expressions without a second mapping to keep in sync. */
  usage?: Record<string, JsonValue>;
  /** USD. Lifted out of `usage` because money is the field everyone asks for by name. */
  cost?: number;
  /** The model the call was actually made with. */
  model?: string;
}

export interface PromptOperationNode extends OperationNodeCore {
  /**
   * What the call RETURNED — plus `session` on a prompt op, the position it ENDED at.
   *
   * `session` shares the namespace with the call's own outputs rather than sitting apart from them,
   * because it IS one of a prompt operation's outputs. A state whose operation returns a value named
   * `session` shadows it; that is a collision the author can see and rename, which is a better
   * trade than the call's result having no address at all.
   */
  output?: JsonValue;
}

export type OperationNode = OperationNodeCore | PromptOperationNode;

const STRING: JsonValue = { type: "string" };
const NUMBER: JsonValue = { type: "number" };

/**
 * A session ref as a schema: one enumerable `id`, matching {@link SessionRef}.
 *
 * `additionalProperties: false` is load-bearing. It is what makes `operation.output.session.position`
 * — a plausible thing to reach for, given how the notation reads — a lint error rather than a
 * runtime `undefined`. A ref is opaque, and the schema says so.
 */
const SESSION_REF: JsonValue = { type: "object", properties: { id: STRING }, required: ["id"], additionalProperties: false };

/**
 * The schema for `operation.*` on a state, discriminated by what the state's operation IS.
 *
 * `undefined` for a state with no operation, so referring to `operation.anything` there is an
 * unresolved reference rather than an object of unknowns.
 */
export function operationNodeSchema(
  opKind: "prompt" | "function" | undefined,
  /**
   * The operation's declared output, when there is one — what typed names `outputs` carries.
   *
   * Passed in rather than derived here because only the caller knows it: the loader has the desugared
   * `op.output`, and this module is a schema builder with no view of the state. Absent ⇒ `outputs`
   * carries only what the engine itself provides, which on a prompt op is `session`.
   */
  opOutput?: { name: string; kind?: string; schema?: JsonValue },
): JsonSchema | undefined {
  if (opKind === undefined) return undefined;
  const properties: Record<string, JsonValue> = {
    outcome: { type: "string", enum: [...OPERATION_OUTCOMES] },
    usage: { type: "object" },
    cost: NUMBER,
    model: STRING,
  };
  // `provider` and `attempts` are absent on purpose, not forgotten. Neither reaches hw's seam today
  // — they are things the EXECUTOR knows and does not report — so declaring them would hand the lint
  // a field it could never resolve and every author a value that is always undefined. They arrive
  // with the executor-reported delta (DESIGN.md §1.6), alongside the code that fills them.
  const outputs = outputSchemaOf(opOutput, opKind === "prompt");
  if (outputs !== undefined) properties["output"] = outputs;
  // Closed, for the same reason `session` is: a typo in a metadata field name should be a lint
  // error, and every field this namespace has is listed right here.
  return { type: "object", properties, additionalProperties: false };
}

/**
 * The schema for `operation.output` — which IS the value the call returned.
 *
 * Not a record built around the return, but the return itself, whatever shape it has:
 *
 *  - an **object** ⇒ its properties are addressable, so `.operation.output.report` works;
 *  - an **array** ⇒ `.operation.output` is that array. An operation whose output is
 *    `{"items": {"type": "string"}}` returns a list of strings, and a slot binds the list;
 *  - a **blob or scalar** ⇒ `.operation.output` is the value.
 *
 * The earlier shape wrapped every return in a record keyed by the output's declared name, which only
 * ever read naturally for the object case and made `.operation.outputs.output` the spelling for a
 * list. The value is the value.
 *
 * `session` rides along as a property where the container can hold one — an object or an array, both
 * of which carry named properties. It is written UNDER anything the call returned by that name: the
 * author named theirs, and the engine's position is still reachable from a guard.
 */
function outputSchemaOf(
  opOutput: { name: string; kind?: string; schema?: JsonValue } | undefined,
  isPrompt: boolean,
): JsonValue | undefined {
  const base = opOutput?.schema;
  const carriesProperties = base === null || base === undefined || (typeof base === "object" && !Array.isArray(base));
  if (!isPrompt || !carriesProperties) return base === undefined ? undefined : base;

  const schema = (base ?? {}) as Record<string, JsonValue>;
  const declared = schema["properties"];
  const properties: Record<string, JsonValue> =
    declared !== null && typeof declared === "object" && !Array.isArray(declared)
      ? { ...(declared as Record<string, JsonValue>) }
      : {};
  if (properties["session"] === undefined) properties["session"] = SESSION_REF;
  // A declared shape keeps everything it said about itself — `type`, `items`, `required` — and gains
  // one property. An array schema carrying `properties` is unusual but exact here: the array IS the
  // value, and `session` is a property on it.
  return base === undefined
    ? { type: "object", properties, additionalProperties: false }
    : { ...schema, properties };
}

