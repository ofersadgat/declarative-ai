/**
 * The `operation.*` expression namespace (SESSIONS.md §8).
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
 * ## `outputs` is ENGINE-provided, not the state's produced slots
 *
 * `operation.outputs` holds what the ENGINE knows about the call, which today is exactly one thing:
 * the conversation position it ended at. The state's own produced values stay at `outputs.*`, where
 * they always were. Keeping them apart is what stops a state that produces an output named `session`
 * from shadowing the engine's.
 *
 * ## Why the END position, and why there is no start marker
 *
 * You append AT a position but do not know where the call ended until the provider resolves, so the
 * end marker is the only value that can exist by the time an expression reads it — and it is the one
 * consumers want, since "append after me" and "fork after me" both mean *after*.
 *
 * There is deliberately no start marker. Recovery after an error does not need one: restart the
 * state, and SESSIONS.md §4's instance-scoped resolution re-resolves the binding to the position the
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
 * What `operation.*` exposes, as a typed union over the operation's kind.
 *
 * The common core is on every kind. `outputs.session` is PROMPT-only, and that is the point of
 * typing it rather than filling in `undefined`: `operation.outputs.session` written against a `ui`
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
  /** Engine-provided call outputs. `session` is the position the call ENDED at. */
  outputs?: { session?: { id: string } };
}

export type OperationNode = OperationNodeCore | PromptOperationNode;

const STRING: JsonValue = { type: "string" };
const NUMBER: JsonValue = { type: "number" };

/**
 * A session ref as a schema: one enumerable `id`, matching {@link SessionRef}.
 *
 * `additionalProperties: false` is load-bearing. It is what makes `operation.outputs.session.position`
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
export function operationNodeSchema(opKind: "prompt" | "function" | undefined): JsonSchema | undefined {
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
  // with the executor-reported delta (SESSIONS.md §12), alongside the code that fills them.
  // Only a prompt operation has a conversation to end at. A function op — a `ui` gate, a delegated
  // adapter, a host helper — has none, so the slot is absent from its type and reaching for it is
  // caught at load time.
  if (opKind === "prompt") {
    properties["outputs"] = {
      type: "object",
      properties: { session: SESSION_REF },
      additionalProperties: false,
    };
  }
  // Closed, for the same reason `session` is: a typo in a metadata field name should be a lint
  // error, and every field this namespace has is listed right here.
  return { type: "object", properties, additionalProperties: false };
}
