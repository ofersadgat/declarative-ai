/**
 * State PATH REFERENCES (§2.1): what an `id` or a `children[].state` is allowed to say.
 *
 * A state id has always been a path — the file's location under the workflow root, minus the
 * suffix. This makes that explicit and gives it the same shape as every other location JaiRA names
 * (artifact destinations, DESIGN §7.6): a bare path hangs off a per-field DEFAULT ROOT, and the
 * escape hatches are the ones a path already has.
 *
 * ```text
 *   feature/plan          bare        → <defaultRoot>/feature/plan
 *   ./goals               relative    → under the REFERRING state's own id (see below)
 *   ../shared/critique    relative    → up one from it
 *   $JAIRA/lib/review     root-var    → a configured root
 *   /opt/workflows/x      absolute    → itself
 *   file:/opt/workflows/x scheme      → itself
 * ```
 *
 * Two decisions worth stating, because both could reasonably have gone the other way:
 *
 *  - **`./` is relative to the referring state's id-as-a-directory, not to its file's directory.**
 *    `feature/plan.json` lives in `feature/`, but the tree convention puts its children in
 *    `feature/plan/`. Resolving `./goals` to `feature/goals` would therefore be useless in exactly
 *    the case relative refs exist for, so a state owns the namespace under its own id and `./goals`
 *    means `feature/plan/goals`.
 *  - **The canonical id keeps the BARE spelling whenever it resolves under the default root.** The
 *    id is an identity — it keys the snapshot hash, the event log, task rows, and `$STATE_ID`. If
 *    `feature/plan` canonicalized to an absolute host path, every stored snapshot would drift the
 *    day this shipped, and the same workflow would carry different ids on two machines. Out-of-tree
 *    refs, which have no bare spelling, canonicalize to an absolute POSIX path.
 */

export class StateRefError extends Error {}

export interface StateRefOptions {
  /**
   * Where a bare reference hangs off, as a resolved absolute path. Ids under it keep their bare
   * spelling. Absent ⇒ only bare references are accepted (the caller has no filesystem in hand).
   */
  defaultRoot?: string;
  /** Roots a `$VAR/…` reference may name, e.g. `{ JAIRA: "/p/.jaira", PROJECT: "/p" }`. */
  roots?: Readonly<Record<string, string>>;
  /** The canonical id of the state doing the referring — the base for `./` and `../`. */
  from?: string;
}

const SCHEME = /^([a-zA-Z][a-zA-Z0-9+.-]*):(\/\/)?(.*)$/;
const ROOT_VAR = /^\$([A-Z_][A-Z0-9_]*)(?:\/(.*))?$/;

/** Forward slashes, no trailing slash, no `.` segments; `..` is resolved where it is legal to. */
function normalizeSegments(path: string): string[] {
  const out: string[] = [];
  for (const segment of path.replace(/\\/g, "/").split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") {
      if (out.length === 0) return ["..", ...out]; // signal an escape; caller decides
      out.pop();
      continue;
    }
    out.push(segment);
  }
  return out;
}

function isAbsolutePath(path: string): boolean {
  return path.startsWith("/") || path.startsWith("\\") || /^[a-zA-Z]:[/\\]/.test(path);
}

/** Absolute paths canonicalize to POSIX separators so one id spells the same on either platform. */
function canonicalAbsolute(path: string): string {
  const posix = path.replace(/\\/g, "/");
  const match = /^([a-zA-Z]:)(\/.*)$/.exec(posix);
  const prefix = match ? match[1]! : "/";
  const body = match ? match[2]! : posix;
  const segments = normalizeSegments(body);
  if (segments[0] === "..") throw new StateRefError(`'${path}' climbs above the filesystem root`);
  return match ? `${prefix}/${segments.join("/")}` : `/${segments.join("/")}`;
}

/**
 * Resolve one authored reference to its canonical state id.
 *
 * `ref` is the string as written; the result is the id every other layer uses — bare when it lands
 * under `defaultRoot`, absolute POSIX otherwise.
 */
export function resolveStateRef(ref: string, options: StateRefOptions = {}): string {
  const trimmed = ref.trim();
  if (trimmed.length === 0) throw new StateRefError("state reference is empty");

  // A scheme, unless it is a bare Windows drive letter (`C:/…`), which is a path.
  const scheme = SCHEME.exec(trimmed);
  let body = trimmed;
  if (scheme && scheme[1]!.length > 1) {
    if (scheme[1]!.toLowerCase() !== "file") {
      throw new StateRefError(
        `unknown scheme '${scheme[1]!}:' in state reference '${ref}' — only 'file:' is supported (or no scheme)`,
      );
    }
    body = scheme[3] ?? "";
    if (!isAbsolutePath(body)) throw new StateRefError(`'file:' state reference '${ref}' must be an absolute path`);
    return canonicalAbsolute(body);
  }

  if (isAbsolutePath(body)) return canonicalAbsolute(body);

  const rootVar = ROOT_VAR.exec(body);
  if (rootVar) {
    const name = rootVar[1]!;
    const root = options.roots?.[name];
    if (root === undefined) {
      const known = Object.keys(options.roots ?? {});
      throw new StateRefError(
        `unknown root '$${name}' in state reference '${ref}'` +
          (known.length > 0 ? ` — known: ${known.map((r) => `$${r}`).join(", ")}` : ""),
      );
    }
    return underDefaultRoot(canonicalAbsolute(`${root}/${rootVar[2] ?? ""}`), options, ref);
  }

  // `./` and `../` resolve against the referring state's own id — see the module header.
  if (/^\.\.?(\/|$)/.test(body)) {
    if (options.from === undefined) {
      throw new StateRefError(`relative state reference '${ref}' has no referring state to resolve against`);
    }
    if (isAbsolutePath(options.from)) {
      return canonicalAbsolute(`${options.from}/${body}`);
    }
    const segments = normalizeSegments(`${options.from}/${body}`);
    if (segments[0] === "..") {
      throw new StateRefError(`relative state reference '${ref}' from '${options.from}' climbs above the workflow root`);
    }
    return segments.join("/");
  }

  const segments = normalizeSegments(body);
  if (segments.length === 0) throw new StateRefError(`state reference '${ref}' names no path`);
  if (segments[0] === "..") throw new StateRefError(`state reference '${ref}' climbs above the workflow root`);
  return segments.join("/");
}

/** Fold an absolute path back to its bare spelling when it lands under the default root. */
function underDefaultRoot(absolute: string, options: StateRefOptions, ref: string): string {
  const root = options.defaultRoot;
  if (root === undefined) return absolute;
  const canonicalRoot = canonicalAbsolute(root);
  if (absolute === canonicalRoot) throw new StateRefError(`state reference '${ref}' names the workflow root itself`);
  const prefix = canonicalRoot.endsWith("/") ? canonicalRoot : `${canonicalRoot}/`;
  return absolute.startsWith(prefix) ? absolute.slice(prefix.length) : absolute;
}

/** True when a canonical id lives under the default root (i.e. is a bare, workflow-relative path). */
export function isBareStateId(id: string): boolean {
  return !isAbsolutePath(id);
}

/**
 * The absolute file path a canonical id reads from, given the default root. Bare ids hang off the
 * root; absolute ids are themselves. The `.json` suffix is the caller's to add.
 */
export function stateFilePath(id: string, defaultRoot: string): string {
  return isBareStateId(id) ? canonicalAbsolute(`${defaultRoot}/${id}`) : id;
}
