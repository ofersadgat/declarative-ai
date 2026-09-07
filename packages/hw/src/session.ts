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
 *  - the resource key follows **what was DECLARED**, which since scoped names is the
 *    `(name, scope)` pair rather than the bare name, and is INHERITED from the
 *    enclosing instance when this operation declares nothing.
 *
 * ## A name has a SCOPE, and the scope is where the name was written
 *
 * A bare name used to live in one flat, run-global namespace, and nothing in the
 * spelling `"review"` said how far it reached. The answer was "the whole run, always",
 * which is wrong in three ways at once: a loop's second pass rejoined the first pass's
 * conversation, a subtree mounted twice collapsed onto one transcript and one worktree,
 * and two independently authored subtrees that both said `"main"` silently merged.
 *
 * So a name is now qualified by a SCOPE, and the scope is decided at the place that
 * WRITES the name rather than at each place that uses it — which is the whole point,
 * because it means a reader never has to visit a use site to learn what a name means.
 * `"review"` is sugar for `{ name: "review", in: <the state that wrote it> }`, and `in`
 * may be redirected to an enclosing scope (`parent`, `global`, an ancestor's id).
 *
 * Two consequences worth stating, because both are the opposite of what the flat
 * namespace did:
 *
 *  - **there is no lookup anywhere.** A declaration is canonicalized where it is
 *    written — see {@link normalizeSession}, which the loader runs before the
 *    environment merge — so it means the same thing wherever it merges to, and no use
 *    site resolves anything. That is why an inherited `environment.session` still
 *    scopes at the ROOT that wrote it: without normalizing at origin, a root setting
 *    one session for its subtree would hand every descendant a private one instead.
 *  - **two siblings that both write `"review"` do not share.** Different writers,
 *    different scopes. They share by naming a common one: `{ name: "review", in: "parent" }`.
 *
 * The scope resolves to an INSTANCE, not to a definition, and that is what makes the
 * §5.1 stability rule fall out instead of being configured: anchor above a loop and the
 * anchor instance is the same on every pass, so the worktree and the permission ledger
 * survive; anchor on the loop body and each pass gets its own, because that is what was
 * asked for. A retry is the same story — a retried state is a new instance, its ancestor
 * is not.
 *
 * ## Joining a session you cannot name
 *
 * `in` qualifies a NAME. A declaration that produces no name produces a session nothing
 * can refer to, which is a private conversation — that is `null`, and `{ in: … }` alone
 * is an error rather than a second spelling for it.
 *
 * `join` is the other half: take the declaration an ancestor wrote, whatever it is
 * called. It never takes a name, and its value chooses how much structural change should
 * break the author's assumption — `parent` asserts the immediate parent wrote one and
 * fails loudly when a wrapper composite is inserted between them, `nearest` accepts any
 * writer above and silently retargets, an ancestor's id pins that specific state, and
 * `global` pins the run root. Every one of them is resolved AT LOAD TIME to the concrete
 * `{ name, in }` it names, so `join` never survives into a run and no lookup happens
 * while the workflow is executing.
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
 * What `operation.output.session` publishes: where the call ENDED, and the conversation it ended in.
 *
 * Both are ordinary {@link SessionRef}s, and the difference between them is the whole of what an
 * author is choosing between when they wire one into a later state's `session`:
 *
 *  - the ref ITSELF is POSITIONED — the point immediately after this call's turn. Continuing from it
 *    appends when nothing else has spoken since, and BRANCHES when something has, so a later state
 *    picking up a conversation can never silently inherit turns it was never shown;
 *  - `end` is the same conversation with NO position, so it continues from wherever that conversation
 *    has got to by the time it is used. This is the "add to the end" case: three states appending in
 *    turn are one thread, not a fork per state.
 *
 * `end` is a real property, materialized when the node is published, rather than something the
 * consuming site interprets. That keeps it a plain ref everywhere downstream — it survives a spread,
 * a `JSON.parse(JSON.stringify(…))` and the events journal, and `resolveSession` needs no case for it.
 * A marker the store interpreted later would have to survive all three to stay correct, and losing it
 * would silently turn "continue" back into "branch".
 *
 * There is no `end.end`: `end` is already unpositioned, so `operationNodeSchema` types it as a
 * bare ref and a second hop is a lint error rather than a value that happens to be missing.
 */
export interface PublishedSession extends SessionRef {
  readonly end: SessionRef;
}

/** Publish a finished call's position: the point after it, and its conversation's moving end. */
export function publishedSession(position: string, conversation: string): PublishedSession {
  return { id: position, end: { id: conversation } };
}

/**
 * A `session` computed at run time — `{"expr": ".children.plan.operation.output.session"}`.
 *
 * The other spellings are STATIC: the loader reads them off the merged document, and they mean the
 * same thing on every instance. That is right for a name and useless for a ref, because the only way
 * to obtain a ref is to read one an operation produced, and `operation.output.session` does not
 * exist until that operation has run.
 *
 * So the explicit spelling has to be a BINDING, resolved per instance against the consuming state's
 * own data. It is the one session form that is evaluated rather than read.
 */
export interface SessionExpr {
  readonly expr: string;
  /** Always branch, rather than appending when the position is still the head. */
  readonly fork?: boolean;
  /**
   * The scope a name the expression RESOLVES TO is qualified by — the state that wrote the `expr`.
   *
   * An expression may resolve to a ref (which carries its own identity) or to a string, and a string
   * is a name like any other, so it needs a scope for the same reason a written one does. Filled in
   * by {@link normalizeSession} at the origin, which is what keeps a computed name and a written one
   * meaning the same thing in the same place.
   */
  readonly in?: string;
}

/** A session named relative to a scope: `"review"` is sugar for `{ name: "review", in: <writer> }`. */
export interface SessionName {
  readonly name: string;
  /**
   * Where the name lives. AUTHORED as a keyword (`parent`, `global`) or an ancestor's state id, and
   * NORMALIZED by the loader to a concrete state id, so everything after load sees one shape.
   */
  readonly in?: string;
  readonly fork?: boolean;
}

/**
 * Take the declaration an ancestor wrote, whatever it is called.
 *
 * Resolved away at load time — see the module header — so this form never reaches the engine.
 */
export interface SessionJoin {
  readonly join: string;
  readonly fork?: boolean;
}

/** A positioned ref, plus how to use it. */
export interface SessionAt {
  readonly id: string;
  readonly fork?: boolean;
}

/** What an author (or an expression) may put in the `session` slot. */
export type SessionDecl = string | null | SessionName | SessionJoin | SessionAt | SessionExpr;

/**
 * What survives {@link normalizeSession}: the same union with the sugar expanded, `join` resolved
 * away, and every `in` a concrete state id.
 */
export type NormalizedSession = null | (SessionName & { readonly in: string }) | SessionAt | SessionExpr;

/** The `in` values that name a scope structurally rather than by state id. */
export const SESSION_SCOPE_KEYWORDS = ["parent", "global", "document"] as const;
/** The `join` values that name a writer structurally rather than by state id. */
export const SESSION_JOIN_KEYWORDS = ["parent", "nearest", "global", "document"] as const;

/**
 * `document` is accepted by the grammar and refused by the resolver, on purpose.
 *
 * It was specified as "the file this declaration was written in", and this model has no such unit:
 * a state IS a file (`loader.ts` infers children from the directory listing, so `feature/plan/goals`
 * is a child of `feature/plan`), which makes "my document's root" resolve to the writer itself and
 * therefore mean nothing that `in` absent does not already mean.
 *
 * The unit the keyword was reaching for is real but different — the root of the MOUNTED SUBTREE, the
 * nearest ancestor reached by a cross-namespace reference (`$JAIRA/lib/review`) rather than by
 * directory inference. That is what makes a reusable subtree portable, and it is computable in the
 * walk. It is left unimplemented rather than silently given the name `document`, because redefining
 * an authored word to mean something adjacent is the class of ambiguity this whole change removes.
 */
const NO_DOCUMENT_SCOPE =
  "'document' names no scope in this model: a state is a file, so a document's root is the writer itself. " +
  "Name a mounted subtree's root state instead, or use 'global'";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** True for the `{ expr }` form — a session evaluated against the consuming instance. */
export function isSessionExpr(value: unknown): value is SessionExpr {
  return isRecord(value) && typeof (value as { expr?: unknown }).expr === "string";
}

/** True for a `{ id }` ref rather than an authored name. */
export function isSessionRef(value: unknown): value is SessionRef {
  return isRecord(value) && typeof (value as { id?: unknown }).id === "string";
}

/** True for the `{ name }` form, authored or normalized. */
export function isSessionName(value: unknown): value is SessionName {
  return isRecord(value) && typeof (value as { name?: unknown }).name === "string";
}

/** True for the `{ join }` form — resolved away at load, so nothing downstream should ever see one. */
export function isSessionJoin(value: unknown): value is SessionJoin {
  return isRecord(value) && typeof (value as { join?: unknown }).join === "string";
}

/** Whether this declaration asked to branch rather than append. Absent and `false` agree (§1.6). */
export function forkOf(declared: NormalizedSession | undefined): boolean {
  return declared !== null && declared !== undefined && declared.fork === true;
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

function reservedComplaint(name: string): string | undefined {
  return name.startsWith(RESERVED_SESSION_PREFIX)
    ? `session name '${name}' starts with '${RESERVED_SESSION_PREFIX}', which is reserved for engine-minted sessions`
    : undefined;
}

/**
 * Reject a `session` declaration that cannot mean anything, with the reason.
 *
 * Returns the complaint, or `undefined` when the declaration is fine. Carried as data
 * rather than thrown so the validator reports it alongside every other authoring
 * error rather than aborting the load at the first one.
 *
 * SHAPE only. Whether an `in` names a real ancestor, and whether a `join` finds a writer, are
 * questions about the TREE rather than about the value, so {@link normalizeSession} answers those
 * where the ancestry is in hand.
 */
export function validateSessionDecl(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") {
    if (value === "") {
      // The failure this catches: `"session": "{{.inputs.thread}}"` where `thread` is
      // absent. An empty string would start an isolated conversation and report success.
      return "session is an empty string — write null to start a fresh session, or fix the reference that produced it";
    }
    return reservedComplaint(value);
  }
  if (!isRecord(value)) {
    const got = Array.isArray(value) ? "an array" : typeof value;
    return `session must be a name, {"join": …}, a session ref, {"expr": …}, or null — got ${got}`;
  }
  const forms = (["name", "join", "id", "expr"] as const).filter((key) => value[key] !== undefined);
  if (forms.length === 0) {
    // The `{ in: … }` case, which reads as though it should mean something and cannot: `in` qualifies
    // a NAME, and a session with no name is one that nothing can refer to.
    return value["in"] !== undefined
      ? "session declares 'in' with no 'name' — a session with no name cannot be joined by anything; write null for a private conversation, or give it a name"
      : 'session must declare one of "name", "join", "id" or "expr"';
  }
  if (forms.length > 1) {
    const listed = forms.map((form) => `'${form}'`).join(" and ");
    return `session declares ${listed} — these are different ways to address a session, so exactly one is meaningful`;
  }
  if (value["fork"] !== undefined && typeof value["fork"] !== "boolean") return "session 'fork' must be true or false";

  const form = forms[0] as "name" | "join" | "id" | "expr";
  const raw = value[form];
  if (typeof raw !== "string") return `session '${form}' must be a string`;

  if (form === "id") {
    return raw === "" ? "session ref has an empty id — the expression that produced it resolved to nothing" : undefined;
  }
  if (form === "expr") {
    // Only the SHAPE is checkable here. What the expression resolves to is instance data, so a
    // string or a ref is accepted at run time and anything else fails there, named.
    return raw.trim() === "" ? "session expression is empty" : undefined;
  }
  if (form === "join") {
    if (value["in"] !== undefined) {
      return "session declares both 'join' and 'in' — 'join' takes another declaration's name and its scope, so there is nothing left for 'in' to qualify";
    }
    return raw === ""
      ? `session 'join' is empty — name an ancestor state, or one of ${SESSION_JOIN_KEYWORDS.join(", ")}`
      : undefined;
  }
  if (raw === "") {
    return "session name is empty — write null to start a fresh session, or fix the reference that produced it";
  }
  const scope = value["in"];
  if (scope !== undefined && (typeof scope !== "string" || scope === "")) {
    return `session 'in' must name an ancestor state, or one of ${SESSION_SCOPE_KEYWORDS.join(", ")}`;
  }
  return reservedComplaint(raw);
}

/** The state a declaration is being canonicalized FOR — see {@link SessionAncestor} for the two ids. */
export type SessionWriter = Pick<SessionAncestor, "id" | "source">;

/** One link of the ancestry a load-time resolution runs against — root first, writer excluded. */
export interface SessionAncestor {
  /**
   * What a scope RESOLVES TO: the id this state loads under on this path, which is its variant when
   * it is mounted more than once with different environments (§7.1a).
   *
   * Distinct from {@link source} because an author writes the canonical id and the engine compares
   * the variant. Resolving to the variant is also what keeps two mounts of one subtree apart: each
   * scopes its names to its own mount rather than to a shared definition.
   */
  readonly id: string;
  /** The state's CANONICAL id — what an author writes in an `in` or a `join`. */
  readonly source: string;
  /**
   * What this state WROTE, already normalized — absent when it declared nothing of its own.
   *
   * "Wrote" and "has" are deliberately different: a state that merely INHERITED a declaration is not
   * a writer, which is the whole of why `join: "parent"` can fail where `join: "nearest"` succeeds.
   * Collapsing the two would make `parent` resolve to the grandparent's declaration whenever the
   * parent had none of its own — which is exactly what `nearest` does, leaving `parent` with no job.
   */
  readonly declared?: NormalizedSession;
}

/** Resolve an `in` — a keyword or an ancestor's id — to the concrete state that scopes the name. */
function anchorOf(
  scope: string | undefined,
  writer: SessionWriter,
  ancestry: readonly SessionAncestor[],
): { id: string } | { error: string } {
  if (scope === undefined) return { id: writer.id };
  if (scope === "parent") {
    const parent = ancestry[ancestry.length - 1];
    return parent === undefined
      ? { error: `session 'in: "parent"' has no parent to scope to — '${writer.source}' is the root` }
      : { id: parent.id };
  }
  if (scope === "global") {
    // At the root the run's scope IS the writer, so unlike `parent` this is not an error there.
    const root = ancestry[0];
    return { id: root === undefined ? writer.id : root.id };
  }
  if (scope === "document") return { error: `session 'in: "document"': ${NO_DOCUMENT_SCOPE}` };
  if (scope === writer.source || scope === writer.id) return { id: writer.id };
  const named = ancestry.find((link) => link.source === scope || link.id === scope);
  return named === undefined
    ? { error: `session 'in: "${scope}"' does not name an ancestor of '${writer.source}' — a name can only be scoped to a state that encloses it` }
    : { id: named.id };
}

/**
 * Resolve a `join` to the declaration it names — at LOAD time, so nothing searches at run time.
 *
 * Every arm asks one question ("which ancestor WROTE a session?") and differs only in which answer
 * it will accept, which is the author choosing how much structural change should break them.
 */
function resolveJoin(
  declared: SessionJoin,
  ancestry: readonly SessionAncestor[],
): { session: NormalizedSession } | { error: string } {
  const { join } = declared;
  // `fork` is the JOINER's word about how to use the session, so it overrides what the writer said.
  // Everything else — the name, and the scope it lives in — is taken verbatim, which is the point:
  // the joiner does not know the name and must not have to.
  const taken = (link: SessionAncestor | undefined, complaint: string): { session: NormalizedSession } | { error: string } => {
    if (link?.declared === undefined) return { error: complaint };
    const session = link.declared;
    if (session === null || declared.fork === undefined) return { session };
    return { session: { ...session, fork: declared.fork } };
  };

  if (join === "nearest") {
    const writers = ancestry.filter((link) => link.declared !== undefined);
    return taken(writers[writers.length - 1], `session 'join: "nearest"' found no enclosing session — no ancestor declares one`);
  }
  if (join === "parent") {
    const parent = ancestry[ancestry.length - 1];
    return taken(
      parent,
      parent === undefined
        ? `session 'join: "parent"' has no parent to join — this state is the root`
        : `session 'join: "parent"' requires '${parent.source}' to declare a session and it declares none — it may only inherit one, which 'join: "nearest"' would follow`,
    );
  }
  if (join === "global") {
    const root = ancestry[0];
    return taken(
      root,
      root === undefined
        ? `session 'join: "global"' has no enclosing run root to join — this state is the root`
        : `session 'join: "global"' requires the run root '${root.source}' to declare a session, and it declares none`,
    );
  }
  if (join === "document") return { error: `session 'join: "document"': ${NO_DOCUMENT_SCOPE}` };
  const named = ancestry.find((link) => link.source === join || link.id === join);
  return taken(
    named,
    named === undefined
      ? `session 'join: "${join}"' does not name an ancestor — a session can only be joined from inside the state that declares it`
      : `session 'join: "${join}"' requires '${join}' to declare a session, and it declares none`,
  );
}

/**
 * Canonicalize one state's OWN declaration, at the place it was written (DESIGN.md §1.6).
 *
 * Run by the loader before the environment merge, which is what lets that merge stay a plain
 * nearest-wins overwrite: by the time a declaration travels, it already carries the scope it was
 * written in, so a root's `session` and a leaf's `session` stop being indistinguishable strings.
 *
 * `ancestry` is root-first and excludes `writer` itself. Errors are returned rather than thrown, so
 * the loader can report them beside every other authoring complaint.
 */
export function normalizeSession(
  declared: SessionDecl | undefined,
  writer: SessionWriter,
  ancestry: readonly SessionAncestor[],
): { session: NormalizedSession | undefined } | { error: string } {
  if (declared === undefined) return { session: undefined };
  const complaint = validateSessionDecl(declared);
  if (complaint !== undefined) return { error: complaint };
  if (declared === null) return { session: null };
  if (typeof declared === "string") return { session: { name: declared, in: writer.id } };
  if (isSessionJoin(declared)) return resolveJoin(declared, ancestry);
  if (isSessionRef(declared)) return { session: declared as SessionAt };
  if (isSessionExpr(declared)) {
    // A computed value may resolve to a NAME, and a name needs a scope for the same reason a written
    // one does. Anchored to the writer, so `{expr}` and the string it resolves to agree.
    const scope = anchorOf(declared.in, writer, ancestry);
    return "error" in scope ? scope : { session: { ...declared, in: scope.id } };
  }
  const named = declared as SessionName;
  const scope = anchorOf(named.in, writer, ancestry);
  return "error" in scope ? scope : { session: { ...named, in: scope.id } };
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
 * conversation name is doing the same thing as writing one — and it is scoped where the equivalent
 * written name would have been: at the state that wrote the expression.
 */
export function sessionFromExpr(expr: SessionExpr, value: unknown): { session: NormalizedSession } | { error: string } {
  const source = expr.expr;
  if (value === undefined || value === null) {
    return { error: `session expression '${source}' resolved to nothing — write null to start a fresh conversation, or wire it to an operation that has run` };
  }
  const complaint = validateSessionDecl(value);
  if (complaint !== undefined) return { error: `session expression '${source}': ${complaint}` };
  // A nested `{ expr }` would be an expression producing an expression; nothing evaluates it.
  if (isSessionExpr(value)) return { error: `session expression '${source}' resolved to another expression` };
  if (isSessionJoin(value)) {
    // `join` is answered against the DOCUMENT, and an expression is answered against instance data;
    // by the time one resolves there is no ancestry pass left to run it through.
    return { error: `session expression '${source}' resolved to a 'join', which is resolved at load time and cannot be computed` };
  }
  const fork = expr.fork ?? (value as { fork?: boolean }).fork;
  const withFork = <T extends object>(session: T): T => (fork === undefined ? session : { ...session, fork });
  if (typeof value === "string") {
    // The scope of a computed name is the scope of the state that wrote the expression, filled in by
    // `normalizeSession`. It is absent only if an `{expr}` reached here unnormalized.
    if (expr.in === undefined) return { error: `session expression '${source}' resolved to a name with no scope to qualify it` };
    return { session: withFork({ name: value, in: expr.in }) };
  }
  if (isSessionRef(value)) return { session: withFork({ id: (value as SessionRef).id }) };
  if (isSessionName(value)) {
    const named = value as SessionName;
    return named.in === undefined
      ? { error: `session expression '${source}' resolved to a name with no scope to qualify it` }
      : { session: withFork({ name: named.name, in: named.in }) };
  }
  return { error: `session expression '${source}' resolved to something that is not a session` };
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
  instanceId: string;
  /** The enclosing instance's resource key — inherited when this operation declares none. */
  inheritedResourceKey: string;
  /**
   * The position a named session currently sits at, as THIS INSTANCE sees it.
   *
   * Instance-scoped, never a global key → head map (DESIGN.md §1.6). The distinction
   * is what makes retry work: a restarted state re-resolves to the position it started
   * from, which has since been appended to by the failed attempt — so it forks, from
   * exactly the right place, instead of stacking the retry on top of the failure.
   *
   * Returning `undefined` means the key has no position yet, and the stream is
   * created at its root.
   */
  positionOf: (key: string) => string | undefined;
  /**
   * The instance a scope resolves to: the nearest ENCLOSING instance of `stateId`, or `undefined`
   * when no ancestor of this instance is that state.
   *
   * Instances rather than definitions is the whole of why a loop behaves correctly without a rule
   * about loops: a scope above the loop is one instance on every pass, and one inside it is a new
   * instance each time. "Nearest" matters for a subtree that mounts itself — the enclosing frame,
   * like a stack.
   */
  anchorOf: (stateId: string) => string | undefined;
}

/**
 * The separator between a session's name and the scope that qualifies it.
 *
 * NOT `@`, which would read more naturally and is already spoken for: a session POSITION is spelled
 * `<session>@<seq>` (SPEC §6.1), so an `@` here would put two different separators in one string.
 * Both position parsers in this repo happen to split on the LAST `@` and would survive it, but the
 * store is a pluggable seam (DESIGN §3.6) and one that split on the first would silently address the
 * wrong conversation. `#` is already the character this module reserves, so it costs no new grammar.
 */
const SCOPE_SEPARATOR = "#";

/**
 * The key a scoped name resolves to — legible on purpose.
 *
 * `workspaceFor(resourceKey)` is host-facing (`EngineConfig`), so a host deciding which worktree a
 * bundle gets should be able to recognize the name the author wrote rather than a hash of it. The
 * name leads for the same reason: the interesting half reads first.
 */
export function sessionKeyOf(name: string, anchor: string): string {
  return `${name}${SCOPE_SEPARATOR}${anchor}`;
}

/** The key a fresh, undeclared conversation gets. Unique per instance, and unauthorable. */
export function freshSessionKey(instanceId: string): string {
  return `${RESERVED_SESSION_PREFIX}i${instanceId}`;
}

/**
 * Resolve one operation's session, per DESIGN.md §1.6.
 *
 * The declaration reaching here is NORMALIZED (see {@link normalizeSession}) and post-merge: an
 * ancestor's `environment.session` arrives as this operation's own `session`, already carrying the
 * scope of the state that wrote it, and a nearer `null` has already overridden it (which is why
 * `null` has to survive that merge as a VALUE rather than as absence — see `splitExecEnvironment`).
 */
export function resolveSession(declared: NormalizedSession | undefined, scope: SessionScope): SessionBinding {
  const fork = forkOf(declared);
  if (declared !== null && declared !== undefined) {
    // A ref names an exact position. It does NOT name a resource bundle — it arrived through data
    // flow, from an operation that may live anywhere in the tree — so the workspace and permissions
    // stay the ones this instance already runs under.
    if ("id" in declared) return { id: declared.id, resourceKey: scope.inheritedResourceKey, fork };
    if ("name" in declared) {
      // A NAME joins a stream and names a resource bundle. Which stream, and which bundle, is the
      // `(name, scope)` PAIR — never the bare name, which is what used to make one word written in
      // two unrelated subtrees mean one transcript and one worktree.
      //
      // The fallback to the declared scope is unreachable for a declaration the loader normalized,
      // which checks that `in` names an ancestor. It matters only if an instance chain ever diverges
      // from the definition tree, and it degrades to the old run-global key rather than to a fresh
      // conversation, because silently isolating is the failure this module exists to prevent.
      const key = sessionKeyOf(declared.name, scope.anchorOf(declared.in) ?? declared.in);
      return { id: scope.positionOf(key) ?? key, resourceKey: key, fork };
    }
  }
  // `null` (explicitly fresh) and absent (nothing declared anywhere) agree: a new stream, private to
  // this instance. They differ only in intent, and nothing downstream acts on the difference.
  return { id: freshSessionKey(scope.instanceId), resourceKey: scope.inheritedResourceKey, fork };
}

/** Render a binding's session for an event payload — the enumerable `{ id }` and nothing else. */
export function sessionRefOf(binding: SessionBinding): SessionRef & JsonValue {
  return { id: binding.id } as SessionRef & JsonValue;
}
