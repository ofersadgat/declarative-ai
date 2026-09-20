/**
 * What an `environment.workspace` declaration means (NAMES.md §10, DESIGN.md §5.1).
 *
 * A workspace names the RESOURCE BUNDLE an operation runs in: the directory its tools act within,
 * the permission ledger, and the scope a `"session"` approval covers. It used to have no position of
 * its own — the bundle was keyed on the SESSION's name, on the reasoning that both were "what was
 * declared" and one declaration was cheaper than two. `session.ts` had already had to pull the two
 * apart once, when a session became a position; keying both on one name was the last of the
 * conflation, and it made two separate statements inexpressible:
 *
 *  - **many sessions in one workspace** — a planner and a critic that must not read each other's
 *    conversation, editing one tree. Naming two sessions asked for two worktrees;
 *  - **one session across several workspaces** — a conversation that carries on while the work moves
 *    to a different tree. Naming one session pinned one worktree.
 *
 * So `workspace` is a position like `session`, and everything about it that is not specific to a
 * workspace is `scope.ts`: a string is a SCOPED NAME, scoped where it is written or where a visible
 * `environment.names` entry says, redirected by `$in`, joined by `$join`, and resolved per instance
 * to the key `name#<instance address>`. That last part is the whole of why a loop behaves: a
 * workspace named above a loop is one worktree on every pass, and one named inside it is a fresh one
 * per pass, with no rule about loops anywhere.
 *
 * What a workspace IS — a worktree, a temp directory, a container — is the host's. The engine hands
 * `workspaceFor` the key and the name's configuration (`names.impl: { "from": "main" }`), and what
 * comes back is opaque to it. That is the provider half of "the position provides the value".
 *
 * Narrower than `session` on purpose. There is no `{ id }` and no `$expr`: a bundle is fixed when an
 * instance is CREATED, before any of its data exists to compute one from, and a workspace is not a
 * value that flows through outputs the way a session ref does.
 *
 * `null` is the one other spelling, and it means what it means on a session: a FRESH, PRIVATE one,
 * whatever the chain named — this instance's own bundle, shared with nothing but the subtree below
 * it, which inherits it like any other. It is not "the run's own". A state that wants the run's
 * bundle says so the way it says anything else it wants to share: by naming one scoped at the run
 * (`{ "$ref": "main", "$in": "global" }`), or by declaring nothing under a root that declares
 * nothing. Reading `null` as "the run's" would have made the one explicit way to ISOLATE a piece of
 * work silently share the tree everything else is editing.
 */
import { offeredKeywords, JOIN_KEYWORDS, RESERVED_NAME_PREFIX, SCOPE_KEYWORDS, anchorScope, joinedWriter, type ScopedName, type ScopeLink } from "./scope.js";

/** What an author may put in the `workspace` slot. */
export type WorkspaceDecl = string | null | { readonly $ref: string; readonly $in?: string } | { readonly $join: string };

/**
 * `null`, as it travels: a workspace with NO NAME, scoped at the state that wrote the `null`.
 *
 * It cannot travel as `null`. A declaration is inherited down the environment chain, and a bare
 * `null` arriving at a grandchild says nothing about who asked — so every state below would mint a
 * private bundle of its own, and "isolate this subtree" would shatter it into one worktree per leaf.
 * Stamping the writer on, exactly as a name has its scope stamped on, is what makes the fresh bundle
 * ONE bundle: every state below resolves to the nearest enclosing instance of the writer.
 */
export interface FreshWorkspace {
  readonly $in: string;
  readonly $ref?: undefined;
}

/** What survives {@link normalizeWorkspace}: a fresh private one, or a name — each with its scope resolved. */
export type NormalizedWorkspace = FreshWorkspace | ScopedName;

/** True for the normalized form of `null` — a bundle with no name. */
export function isFreshWorkspace(workspace: NormalizedWorkspace): workspace is FreshWorkspace {
  return workspace.$ref === undefined;
}

/**
 * The RUN's resource bundle: what an operation gets when neither it nor any ancestor named a
 * workspace. A run has one working tree and always did; naming a workspace is how a subtree asks
 * for another.
 */
export const RUN_RESOURCE_KEY = "default";

/**
 * The key a fresh, private bundle gets — `workspace: null`. Unauthorable, like a fresh session's.
 *
 * Built from the instance's ADDRESS rather than its id, unlike `freshSessionKey`, because this key
 * reaches the host and has to be the same in a resumed run: a worktree made for `#loop/build:1`
 * before a restart is the worktree that pass goes back to after it. A host tells the three kinds of
 * key apart by shape — `default`, `#<address>`, `name#<address>`.
 */
export function freshWorkspaceKey(address: string): string {
  return `${RESERVED_NAME_PREFIX}${address}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Reject a `workspace` declaration that cannot mean anything, with the reason — SHAPE only, like
 * `validateSessionDecl`: whether an `$in` names a real ancestor is a question about the tree.
 */
export function validateWorkspaceDecl(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const named = (name: string): string | undefined =>
    name === ""
      ? "workspace name is empty — write null for a fresh private workspace, or fix the reference that produced it"
      : name.startsWith(RESERVED_NAME_PREFIX)
        ? `workspace name '${name}' starts with '${RESERVED_NAME_PREFIX}', which is reserved for keys the engine mints`
        : undefined;
  if (typeof value === "string") return named(value);
  if (!isRecord(value)) return `workspace must be a name, {"$join": …}, or null — got ${Array.isArray(value) ? "an array" : typeof value}`;
  if (value.$expr !== undefined || value.id !== undefined) {
    return "workspace cannot be computed — a resource bundle is fixed when its instance is created, before there is any data to compute one from. Name it, and scope the name where it should live";
  }
  if (typeof value.$join === "string") {
    if (value.$ref !== undefined || value.$in !== undefined) return "workspace declares '$join' beside '$ref'/'$in' — '$join' takes another declaration's name and its scope whole";
    return value.$join === "" ? `workspace '$join' is empty — name an ancestor state, or one of ${offeredKeywords(JOIN_KEYWORDS)}` : undefined;
  }
  if (typeof value.$ref !== "string") return 'workspace must declare one of "$ref" or "$join"';
  if (value.$in !== undefined && (typeof value.$in !== "string" || value.$in === "")) {
    return `workspace '$in' must name an ancestor state, or one of ${offeredKeywords(SCOPE_KEYWORDS)}`;
  }
  return named(value.$ref);
}

/** One link of the ancestry a workspace `$join` runs against — what that state WROTE, if anything. */
export interface WorkspaceAncestor extends ScopeLink {
  readonly workspace?: NormalizedWorkspace;
}

/**
 * Canonicalize one state's OWN `workspace` declaration, at the place it was written.
 *
 * The same pass, for the same reason, as `normalizeSession`: the environment merge is a nearest-wins
 * overwrite, so the scope has to be stamped on while the writer is still known. Plain keys beside a
 * `$ref` are the NAME's configuration and are returned apart — they belong in the names table, with
 * every other writer's, not on the declaration that travels down the chain.
 */
export function normalizeWorkspace(
  declared: unknown,
  writer: ScopeLink,
  ancestry: readonly WorkspaceAncestor[],
  /** The scope of a visible `environment.names` entry for this name, when there is one (NAMES.md §3). */
  scopeOf: (name: string) => string | undefined = () => undefined,
): { workspace: NormalizedWorkspace | undefined; configuration?: Record<string, unknown> } | { error: string } {
  if (declared === undefined) return { workspace: undefined };
  const complaint = validateWorkspaceDecl(declared);
  if (complaint !== undefined) return { error: complaint };
  if (declared === null) return { workspace: { $in: writer.id } };
  if (typeof declared === "string") return { workspace: { $ref: declared, $in: scopeOf(declared) ?? writer.id } };
  const decl = declared as Record<string, unknown>;
  if (typeof decl.$join === "string") {
    const joined = joinedWriter(decl.$join, ancestry, (link) => link.workspace, "workspace");
    return "error" in joined ? joined : { workspace: joined.declared };
  }
  const { $ref: name, $in: redirect, ...configuration } = decl as { $ref: string; $in?: string } & Record<string, unknown>;
  const stray = Object.keys(configuration).find((key) => key.startsWith("$"));
  if (stray !== undefined) return { error: `'${stray}' means nothing on a workspace — it takes '$ref', '$in', '$join', and plain keys that configure the name` };
  const declaredAt = redirect === undefined ? scopeOf(name) : undefined;
  const scope = declaredAt !== undefined ? { id: declaredAt } : anchorScope(redirect, writer, ancestry, "workspace");
  if ("error" in scope) return scope;
  return { workspace: { $ref: name, $in: scope.id }, ...(Object.keys(configuration).length > 0 ? { configuration } : {}) };
}
