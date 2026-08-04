/**
 * What an operation's `session` declaration means (DESIGN.md §1.6).
 *
 * A session is an append-only conversation stream, and a **session ref** names one AT
 * a position — which is what makes "continue from here" and "branch from here" the
 * same primitive. The ref is **opaque**: hw never parses it, and neither does any
 * executor. Only the session store interprets structure, which is what lets the
 * spelling change without touching a consumer.
 *
 * ## The two things `session` used to be
 *
 * One string used to key two unrelated concerns, and separating them is most of what
 * this module exists for:
 *
 *  - the **conversation** — which transcript this call joins;
 *  - the **resource bundle** (DESIGN §5.1) — the workspace its tools act within, the
 *    permission ledger, and the scope a `"session"` approval covers.
 *
 * They could share a key while a session was a name. They cannot once a session is a
 * POSITION, because a position changes on every call: a `"session"`-scoped approval
 * would then cover exactly one operation, and every fork would silently ask for its
 * own worktree — which DESIGN.md §5.1 explicitly rules out ("forks share a
 * worktree; JaiRA does not fork worktrees").
 *
 * So {@link SessionBinding} carries both, and the rules differ on purpose:
 *
 *  - the conversation follows §4 — a declared name joins that stream, `null` and
 *    absent each start a fresh one;
 *  - the resource key follows **what was DECLARED**, and is INHERITED from the
 *    enclosing instance when this operation declares nothing. A fork, compaction,
 *    resync, retry or loop iteration never changes which session was declared, so the
 *    worktree and the permission ledger survive all of them.
 *
 * That last point is the answer to "states get replayed and looped — how do you tell
 * the iterations apart?" You don't have to: the resource key was never derived from
 * the iteration.
 */
import type { JsonValue } from "@declarative-ai/exec";

/**
 * The value that flows through inputs, outputs and expressions.
 *
 * `id` is the only enumerable property, so the events journal and any serialized
 * inputs/outputs see `{ id }` and nothing else. Resolved forms carrying more attach it
 * non-enumerably (DESIGN.md §1.6).
 */
export interface SessionRef {
  readonly id: string;
}

/**
 * What an author (or an expression) may put in the `session` slot.
 *
 * `null` is the ONLY explicit "start fresh" marker, and it is deliberately not the
 * empty string: a prompt template interpolating a bad reference would otherwise
 * produce `""` and silently run an isolated conversation that looks like it worked.
 * `""` is an error — see {@link validateSessionDecl}.
 */
export type SessionDecl = string | null | SessionRef | SessionExpr;

/**
 * A `session` computed at run time — `{"expr": ".children.plan.operation.output.session"}`.
 *
 * The other three spellings are STATIC: the loader reads them off the merged document, and they
 * mean the same thing on every instance. That is right for a name and useless for a ref, because
 * the only way to obtain a ref is to read one an operation produced, and `operation.output.session`
 * does not exist until that operation has run.
 *
 * So the explicit spelling has to be a BINDING, resolved per instance against the consuming state's
 * own data. It is the one session form that is evaluated rather than read.
 */
export interface SessionExpr {
  readonly expr: string;
}

/** True for the `{ expr }` form — a session evaluated against the consuming instance. */
export function isSessionExpr(value: unknown): value is SessionExpr {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { expr?: unknown }).expr === "string"
  );
}

/**
 * Engine-minted session keys are prefixed, and an authored name may not be.
 *
 * A fresh conversation needs a key nothing else will collide with. Reserving one
 * character is cheaper than any scheme that tries to guess whether a collision was
 * meaningful, and it makes the failure a load-time message instead of two states
 * mysteriously sharing a transcript.
 */
export const RESERVED_SESSION_PREFIX = "#";

/**
 * The RUN's resource bundle: the workspace and permission ledger an operation gets
 * when neither it nor any ancestor declared a session.
 *
 * This is what `DEFAULT_SESSION` used to be, minus the part the append-only model removes.
 * It is no longer an implicit shared CONVERSATION — that is exactly the thing driving
 * unbounded context growth — but it remains the run's shared workspace, because a run
 * has one worktree and always did.
 */
export const RUN_RESOURCE_KEY = "default";

/** True for a `{ id }` ref rather than an authored name. */
export function isSessionRef(value: unknown): value is SessionRef {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { id?: unknown }).id === "string"
  );
}

/**
 * Reject a `session` declaration that cannot mean anything, with the reason.
 *
 * Returns the complaint, or `undefined` when the declaration is fine. Carried as data
 * rather than thrown so the validator reports it alongside every other authoring
 * error rather than aborting the load at the first one.
 */
export function validateSessionDecl(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (isSessionRef(value)) {
    return (value as SessionRef).id === ""
      ? "session ref has an empty id — the expression that produced it resolved to nothing"
      : undefined;
  }
  if (isSessionExpr(value)) {
    // Only the SHAPE is checkable here. What the expression resolves to is instance data, so a
    // string or a ref is accepted at run time and anything else fails there, named.
    return (value as SessionExpr).expr.trim() === "" ? "session expression is empty" : undefined;
  }
  if (typeof value !== "string") {
    return `session must be a name, a session ref, {"expr": …}, or null — got ${Array.isArray(value) ? "an array" : typeof value}`;
  }
  if (value === "") {
    // The failure this catches: `"session": "{{.inputs.thread}}"` where `thread` is
    // absent. An empty string would start an isolated conversation and report success.
    return "session is an empty string — write null to start a fresh session, or fix the reference that produced it";
  }
  if (value.startsWith(RESERVED_SESSION_PREFIX)) {
    return `session name '${value}' starts with '${RESERVED_SESSION_PREFIX}', which is reserved for engine-minted sessions`;
  }
  return undefined;
}

/**
 * What a `{ expr }` session evaluated TO, checked once the instance's data is in hand.
 *
 * The interesting case is absence. An expression over a child that has not run yet — or over a
 * failed one — resolves to nothing, and the tempting reading is "then start fresh". That is exactly
 * the silent isolation `""` is rejected for: the author asked to continue a specific conversation,
 * and quietly starting a different one produces a run that looks successful and has forgotten
 * everything. So absence is an ERROR, and `null` remains the only way to ask for fresh.
 *
 * A resolved STRING is treated as a name, matching the static spelling — an author computing a
 * conversation name is doing the same thing as writing one, and a ref is always an object.
 */
export function sessionFromExpr(expr: string, value: unknown): { session: SessionDecl } | { error: string } {
  if (value === undefined || value === null) {
    return { error: `session expression '${expr}' resolved to nothing — write null to start a fresh conversation, or wire it to an operation that has run` };
  }
  const complaint = validateSessionDecl(value);
  if (complaint !== undefined) return { error: `session expression '${expr}': ${complaint}` };
  // A nested `{ expr }` would be an expression producing an expression; nothing evaluates it.
  if (isSessionExpr(value)) return { error: `session expression '${expr}' resolved to another expression` };
  return { session: value as SessionDecl };
}

/** What one operation resolved to. Both halves are needed; neither implies the other. */
export interface SessionBinding {
  /**
   * The conversation this operation runs under. OPAQUE — hw passes it through and
   * never parses it.
   */
  id: string;
  /**
   * The resource bundle: workspace, permission ledger, and the scope a `"session"`
   * approval covers. Stable across forks, compactions, resyncs, retries and loop
   * iterations, because it is derived from what was DECLARED rather than from where
   * the conversation currently is.
   */
  resourceKey: string;
  /**
   * Always fork, rather than appending if the position is still the head
   * (DESIGN.md §1.6). Expressed where a session is CONSUMED, because a position
   * marker should not encode an intent about how a later caller will use it.
   */
  fork: boolean;
}

/** What the caller knows about the instance a binding is being resolved for. */
export interface SessionScope {
  /** Distinguishes one instance's fresh session from another's. */
  instanceId: number;
  /** The enclosing instance's resource key — inherited when this operation declares none. */
  inheritedResourceKey: string;
  /**
   * The position a named session currently sits at, as THIS INSTANCE sees it.
   *
   * Instance-scoped, never a global name → head map (DESIGN.md §1.6). The distinction
   * is what makes retry work: a restarted state re-resolves to the position it started
   * from, which has since been appended to by the failed attempt — so it forks, from
   * exactly the right place, instead of stacking the retry on top of the failure.
   *
   * Returning `undefined` means the name has no position yet, and the stream is
   * created at its root.
   */
  positionOf: (name: string) => string | undefined;
}

/**
 * Resolve one operation's session, per DESIGN.md §1.6.
 *
 * The four-way order in the document collapses to three cases here, because the
 * ENVIRONMENT MERGE has already run: an ancestor's `environment.session` reaches this
 * function as the operation's own `session`, and a nearer `null` has already
 * overridden it (which is why `null` has to survive that merge as a VALUE rather than
 * as absence — see `normalizeSynonyms`). So "absent" here means neither the operation
 * nor any ancestor declared one.
 */
export function resolveSession(declared: SessionDecl | undefined, fork: boolean, scope: SessionScope): SessionBinding {
  // A ref names an exact position. It does NOT name a resource bundle — it arrived
  // through data flow, from an operation that may live anywhere in the tree — so the
  // workspace and permissions stay the ones this instance already runs under.
  if (isSessionRef(declared)) {
    return { id: declared.id, resourceKey: scope.inheritedResourceKey, fork };
  }
  // A NAME joins a stream and names a resource bundle. The position comes from this
  // instance's own view of the name; the resource key is the name itself, which is
  // what keeps it stable while the position moves.
  if (typeof declared === "string") {
    return { id: scope.positionOf(declared) ?? declared, resourceKey: declared, fork };
  }
  // `null` (explicitly fresh) and absent (nothing declared anywhere) agree: a new
  // stream, private to this instance. They differ only in intent, and nothing
  // downstream acts on the difference.
  return { id: freshSessionKey(scope.instanceId), resourceKey: scope.inheritedResourceKey, fork };
}

/** The key a fresh, undeclared conversation gets. Unique per instance, and unauthorable. */
export function freshSessionKey(instanceId: number): string {
  return `${RESERVED_SESSION_PREFIX}i${instanceId}`;
}

/** Render a binding's session for an event payload — the enumerable `{ id }` and nothing else. */
export function sessionRefOf(binding: SessionBinding): SessionRef & JsonValue {
  return { id: binding.id } as SessionRef & JsonValue;
}
