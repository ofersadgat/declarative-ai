/**
 * A scoped name: the pair `(name, scope)`, and the key it resolves to.
 *
 * This is the half of `session.ts` that was never about sessions. A name written into a position
 * is an IDENTITY, and the identity is the pair — the word and the state that scopes it — so two
 * uses that resolve to the same pair are the same thing and nothing else makes them so. `session`
 * was the first position to need that, and everything it needed turns out to be independent of
 * what the name is bound TO: a conversation, a workspace, a value. It lives here so the next
 * position reuses the mechanic instead of growing a second copy of it, and `session.ts` keeps
 * only what is a session's own — refs, positions, `fork`, the conversation/bundle split.
 *
 * ## Two halves, and no lookup between them
 *
 *  - **At load**, where a name is WRITTEN: {@link anchorScope} turns an authored `$in` (absent, a
 *    keyword, an ancestor's id) into the concrete state that scopes the name, and
 *    {@link joinedWriter} finds the ancestor whose declaration a `$join` takes. Both run at the
 *    writer, before the environment merge, because that merge is a nearest-wins overwrite:
 *    afterwards a root's `"planning"` and a leaf's are the same string and only the origin could
 *    still tell them apart. Normalizing there is what lets a declaration travel — it already
 *    carries the scope it was written in, so no use site ever resolves anything.
 *  - **At run**, where a name is USED: {@link keyOfScopedName} resolves the scoping STATE to its
 *    nearest enclosing INSTANCE and spells the pair as `name#<instance address>`. Instances rather
 *    than definitions is the whole of why loops and fan-out need no rule of their own — a scope
 *    above a loop is one instance on every pass, a scope inside it is a new one per pass, and an
 *    `each` element is `key[i]` (see {@link addressPath}).
 *
 * What is deliberately NOT here: the engine's instance chain. "The nearest enclosing instance of
 * this state" is asked through a callback, so this module knows about addresses and not about
 * instances, and a test can anchor a name without building an engine.
 */
import type { InstanceAddress } from "./ports.js";

/**
 * A name with the scope that qualifies it, as it exists after load: `$in` is a concrete state id.
 *
 * Spelled as it is AUTHORED — `{ "$ref": "review", "$in": "parent" }`, with `$in` resolved — rather
 * than as a private pair, because a normalized declaration travels through the environment merge and
 * is read back by expressions as `.environment.session`. One spelling means an author who reads what
 * the loader produced sees the thing they could have written.
 */
export interface ScopedName {
  readonly $ref: string;
  readonly $in: string;
}

/**
 * One state on the path from the root to a writer, by the two ids a scope can be said in.
 *
 * `id` is what a scope RESOLVES TO: the id this state loads under on this path, which is its variant
 * when it is mounted more than once with different environments (§7.1a). `source` is its CANONICAL
 * id — what an author writes in an `in` or a `join`. Distinct because an author writes the canonical
 * id and the engine compares the variant, and resolving to the variant is also what keeps two mounts
 * of one subtree apart: each scopes its names to its own mount rather than to a shared definition.
 */
export interface ScopeLink {
  readonly id: string;
  readonly source: string;
}

/** The `$in` values that name a scope structurally rather than by state id. */
export const SCOPE_KEYWORDS = ["parent", "global", "document"] as const;
/** The `$join` values that name a writer structurally rather than by state id. */
export const JOIN_KEYWORDS = ["parent", "nearest", "global", "document"] as const;

/**
 * The keywords a complaint OFFERS an author — every one the resolver will actually take. `document`
 * is recognized so that it can be refused with its own reason (below), which is no reason to suggest it.
 */
export function offeredKeywords(keywords: readonly string[]): string {
  return keywords.filter((keyword) => keyword !== "document").join(", ");
}

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

/**
 * Resolve an `$in` — absent, a keyword, or an ancestor's id — to the concrete state that scopes a name.
 *
 * `ancestry` is root-first and excludes `writer`. `what` is the position's noun (`"session"`), which
 * only the complaints use: the rule is the same for every position, and an author still needs to be
 * told WHICH declaration could not be scoped.
 */
export function anchorScope(
  scope: string | undefined,
  writer: ScopeLink,
  ancestry: readonly ScopeLink[],
  what: string,
): { id: string } | { error: string } {
  if (scope === undefined) return { id: writer.id };
  if (scope === "parent") {
    const parent = ancestry[ancestry.length - 1];
    return parent === undefined
      ? { error: `${what} '$in: "parent"' has no parent to scope to — '${writer.source}' is the root` }
      : { id: parent.id };
  }
  if (scope === "global") {
    // At the root the run's scope IS the writer, so unlike `parent` this is not an error there.
    const root = ancestry[0];
    return { id: root === undefined ? writer.id : root.id };
  }
  if (scope === "document") return { error: `${what} '$in: "document"': ${NO_DOCUMENT_SCOPE}` };
  if (scope === writer.source || scope === writer.id) return { id: writer.id };
  const named = ancestry.find((link) => link.source === scope || link.id === scope);
  return named === undefined
    ? { error: `${what} '$in: "${scope}"' does not name an ancestor of '${writer.source}' — a name can only be scoped to a state that encloses it` }
    : { id: named.id };
}

/**
 * Resolve a `$join` to the declaration it takes — at LOAD time, so nothing searches at run time.
 *
 * Every arm asks one question ("which ancestor WROTE one?") and differs only in which answer it will
 * accept, which is the author choosing how much structural change should break them: `parent` asserts
 * the immediate parent wrote one and fails loudly when a wrapper is inserted between them, `nearest`
 * accepts any writer above and silently retargets, an id pins that state, `global` pins the run root.
 *
 * `declaredOf` is what makes this position-independent: one ancestry serves every position, and each
 * asks it for its own declaration. It must answer what a link WROTE, never what it merely inherited —
 * collapsing the two would make `parent` resolve to the grandparent's declaration whenever the parent
 * had none of its own, which is exactly what `nearest` does, leaving `parent` with no job.
 */
export function joinedWriter<L extends ScopeLink, D>(
  join: string,
  ancestry: readonly L[],
  declaredOf: (link: L) => D | undefined,
  what: string,
): { declared: D } | { error: string } {
  const taken = (link: L | undefined, complaint: string): { declared: D } | { error: string } => {
    const declared = link === undefined ? undefined : declaredOf(link);
    return declared === undefined ? { error: complaint } : { declared };
  };

  if (join === "nearest") {
    const writers = ancestry.filter((link) => declaredOf(link) !== undefined);
    return taken(writers[writers.length - 1], `${what} '$join: "nearest"' found no enclosing ${what} — no ancestor declares one`);
  }
  if (join === "parent") {
    const parent = ancestry[ancestry.length - 1];
    return taken(
      parent,
      parent === undefined
        ? `${what} '$join: "parent"' has no parent to join — this state is the root`
        : `${what} '$join: "parent"' requires '${parent.source}' to declare a ${what} and it declares none — it may only inherit one, which '$join: "nearest"' would follow`,
    );
  }
  if (join === "global") {
    const root = ancestry[0];
    return taken(
      root,
      root === undefined
        ? `${what} '$join: "global"' has no enclosing run root to join — this state is the root`
        : `${what} '$join: "global"' requires the run root '${root.source}' to declare a ${what}, and it declares none`,
    );
  }
  if (join === "document") return { error: `${what} '$join: "document"': ${NO_DOCUMENT_SCOPE}` };
  const named = ancestry.find((link) => link.source === join || link.id === join);
  return taken(
    named,
    named === undefined
      ? `${what} '$join: "${join}"' does not name an ancestor — a ${what} can only be joined from inside the state that declares it`
      : `${what} '$join: "${join}"' requires '${join}' to declare a ${what}, and it declares none`,
  );
}

/**
 * Keys the engine mints itself are prefixed, and an authored name may not be.
 *
 * A key nothing authored can collide with is cheaper than any scheme that tries to guess whether a
 * collision was meaningful, and it makes the failure a load-time message instead of two states
 * mysteriously sharing what the name is bound to.
 */
export const RESERVED_NAME_PREFIX = "#";

/**
 * The separator between a name and the scope that qualifies it.
 *
 * NOT `@`, which would read more naturally and is already spoken for: a session POSITION is spelled
 * `<session>@<seq>` (SPEC §6.1), so an `@` here would put two different separators in one string.
 * Both position parsers in this repo happen to split on the LAST `@` and would survive it, but the
 * store is a pluggable seam (DESIGN §3.6) and one that split on the first would silently address the
 * wrong conversation. `#` is already the reserved character, so it costs no new grammar.
 */
const SCOPE_SEPARATOR = "#";

/**
 * The key a scoped name resolves to — legible on purpose.
 *
 * A key reaches the host (`workspaceFor(resourceKey)` on `EngineConfig`), so a host deciding which
 * worktree a bundle gets should be able to recognize the name the author wrote rather than a hash of
 * it. The name leads for the same reason: the interesting half reads first.
 */
export function scopedKeyOf(name: string, anchor: string): string {
  return `${name}${SCOPE_SEPARATOR}${anchor}`;
}

/**
 * Where a name's KEY travels when the name is read as a value — beside its configuration, under a
 * `$`-key, because the state system put it there and a reader should be able to see that it is not
 * something the author configured (NAMES.md §2).
 */
export const NAME_KEY = "$key";

/**
 * How an anchoring instance is NAMED inside a key — its address, as a path.
 *
 * Not the instance id, which would satisfy the semantics and fail everything else. A key reaches the
 * host, so it has to be two things a UUIDv7 is not: legible enough for a host to recognize which
 * bundle it is being asked about, and the same in two runs of one pinned definition — an id is minted
 * fresh per run, so a worktree keyed on one would be a different worktree every time the workflow
 * ran, and a resumed task would resolve a different key than the one it suspended under.
 *
 * An address is both, and it keeps the property the anchor exists for: an occurrence counts entries
 * under one key, so a loop's second pass through a scope is a different address, while a scope ABOVE
 * the loop has the same address on every pass. The root is `/`, and each step reads `key` or
 * `key:occurrence` — the occurrence is elided at 0 so the common case stays short.
 */
export function addressPath(address: InstanceAddress): string {
  if (address.length === 0) return "/";
  return address
    .map((step) => {
      const entry = step.occurrence === 0 ? step.childKey : `${step.childKey}:${step.occurrence}`;
      // An element of a fan-out (§6.2) is `key[i]`: the elements of one entry share the occurrence
      // and differ here, so a name scoped to the fanned-out state is one identity per element.
      return step.element === undefined ? entry : `${entry}[${step.element}]`;
    })
    .join("/");
}

/**
 * The key one USE of a scoped name resolves to, for the instance asking.
 *
 * `anchorOf` answers "the nearest ENCLOSING instance of this state, as an {@link addressPath}" — or
 * `undefined` when no ancestor of the asking instance is that state. "Nearest" matters for a subtree
 * that mounts itself: the enclosing frame, like a stack.
 *
 * The fallback to the declared scope is unreachable for a name the loader normalized, which checks
 * that `in` names an ancestor. It matters only if an instance chain ever diverges from the definition
 * tree, and it degrades to a run-global key rather than to a private one, because silently isolating
 * what the author asked to share is the failure scoped names exist to prevent.
 */
export function keyOfScopedName(scoped: ScopedName, anchorOf: (stateId: string) => string | undefined): string {
  return scopedKeyOf(scoped.$ref, anchorOf(scoped.$in) ?? scoped.$in);
}
