/**
 * The one reference type (REFERENCES.md).
 *
 * A reference is a path in two halves — `<file-path>.<property-path>` — and the same grammar backs
 * all three of the things a workflow points at: naming a state to run, transcluding a document
 * node, and reading runtime data. What differs is the verb, not the syntax.
 *
 * ```text
 *   feature/plan                          a file, whole
 *   feature/plan.outputs.plan_doc         a property of that file
 *   .children.critique.outputs.outcome    a property of THIS file
 *   ./goals   ../shared/lint              relative to the referring state's own id
 *   $/types/markdown                      `$` is shorthand for `$JAIRA`
 *   /opt/workflows/review                 absolute
 * ```
 *
 * Two decisions worth stating, because both could reasonably have gone the other way:
 *
 *  - **`./` is relative to the referring state's id-as-a-directory, not to its file's directory.**
 *    `feature/plan.json` lives in `feature/`, but the tree convention puts its children in
 *    `feature/plan/`. Resolving `./goals` to `feature/goals` would be useless in exactly the case
 *    relative references exist for, so a state owns the namespace under its own id.
 *  - **The canonical id keeps the BARE spelling whenever it lands under the default root.** The id
 *    is an identity — it keys the snapshot hash, the event log, task rows, and `$STATE_ID`. If
 *    `feature/plan` canonicalized to an absolute host path, every stored snapshot would drift the
 *    day this shipped, and one workflow would carry different ids on two machines.
 *
 * Where the file path ENDS is decided by longest match against the directory listing, the way a
 * module resolver works — no fixed list of recognized extensions, any extension admitted, and "no
 * such file" reported where the reference is parsed rather than somewhere downstream.
 */
import { parse as parseYaml } from "yaml";

export class ReferenceError_ extends Error {}
export { ReferenceError_ as ReferenceError };

/** The filesystem a reference resolves against — injected, so a bundle can resolve in memory. */
export interface Vfs {
  /** Entry names (not paths) directly inside `dir`; empty when it does not exist. */
  list(dir: string): readonly string[];
  /** File contents as text, or `undefined` when it does not exist. */
  read(path: string): string | undefined;
}

export interface ReferenceOptions {
  /**
   * Where a bare reference hangs off — one root, or an ordered SEARCH PATH (EXPRESSIONS.md §4).
   *
   * With several, a bare reference is tried against each in turn and the first match wins, exactly
   * as a shell resolves a bare command name against `PATH`. Absolute and `file:` references are
   * themselves and `$VAR/…` names its own root, so neither consults it. A `./`/`../` reference is
   * anchored to the referring state FIRST and the result searched — the anchoring is what makes it
   * relative, and searching what follows is what lets a later layer's state point at an earlier
   * layer's override of its own child.
   *
   * EVERY entry produces bare ids (see {@link identityOf}), which is what makes the path a LAYERING
   * mechanism rather than only a convenience: the same bare id means "whichever layer supplies it".
   */
  defaultRoot?: string | readonly string[];
  /** Roots a `$VAR/…` reference may name — each names exactly one place, and never searches. */
  roots?: Readonly<Record<string, string>>;
  /**
   * The ordered LAYER ROOTS that a bare `$` searches — `[<project>/.jaira, ~/.jaira]` in JaiRA.
   *
   * `$` is the sigil for "resolve this against the layers", so `$/lib/review` finds the project's
   * copy if there is one and the shared copy otherwise. That is what lets the fragments a state is
   * assembled from — prompts, types, guards, operation documents — layer exactly as whole states
   * do; without it, an override model covers state files and nothing inside them.
   *
   * `defaultRoot` is derived from this in practice (`<root>/workflows`, `<root>/functions` per
   * root), but the two stay separate options because they answer different questions: this is where
   * a LAYER begins, that is where a BARE state id hangs off.
   *
   * Absent ⇒ `$` keeps its original meaning, an alias for `$JAIRA`.
   */
  rootPath?: readonly string[];
  /** The canonical id of the state doing the referring — the base for `./` and `../`. */
  from?: string;
  vfs?: Vfs;
  /** Collects non-fatal ambiguities (REFERENCES.md §9). */
  onWarn?: (message: string) => void;
  /**
   * What it means when a bare reference matches at more than one path entry.
   *
   * `"warn"` (the default) reports it: with nothing but `PATH` semantics, a file appearing at an
   * earlier entry silently changes what an existing reference means, anywhere in the workflow.
   *
   * `"override"` says the shadowing IS the design — an earlier entry is a project-local override of
   * a shared base layer — and stays quiet. It changes no resolution, only whether the overlap is
   * reported, because a caller that layers roots deliberately would otherwise get one warning per
   * overridden state and learn to ignore all of them.
   */
  shadowing?: "warn" | "override";
}

/** A parsed and located reference. */
export interface ResolvedReference {
  /** Absolute POSIX path of the target file; absent for a same-file (`.foo`) reference. */
  file?: string;
  /**
   * The canonical identity of the target: bare when it lands under ANY entry of the search path,
   * absolute otherwise, with a `.json`/`.yaml`/`.yml` suffix stripped so a state keeps today's id.
   */
  id?: string;
  /** Property path within the file. Empty ⇒ the file as a whole. */
  property: string[];
  /** True when the reference named no file — a property of the current one. */
  local: boolean;
}

/** Suffixes an extensionless reference probes, in order, and that a canonical id drops. */
export const DATA_EXTENSIONS = ["json", "yaml", "yml"] as const;

/**
 * True when a file deserializes to a VALUE rather than to text (REFERENCES.md §4.3).
 *
 * The same split `parseReferencedFile` makes, asked ahead of reading — which a caller needs when the
 * two produce indistinguishable JavaScript, as a `.md`'s text and a JSON string do.
 */
export function isDataFile(file: string | undefined): boolean {
  if (file === undefined) return true; // a property of the current document is already a value
  const ext = file.split("/").pop()?.split(".").pop()?.toLowerCase();
  return ext !== undefined && (DATA_EXTENSIONS as readonly string[]).includes(ext);
}

const SCHEME = /^([a-zA-Z][a-zA-Z0-9+.-]*):(\/\/)?(.*)$/;
const ROOT_VAR = /^\$([A-Z_][A-Z0-9_]*)?(?:\/(.*))?$/;

function isAbsolutePath(path: string): boolean {
  return path.startsWith("/") || path.startsWith("\\") || /^[a-zA-Z]:[/\\]/.test(path);
}

/** Forward slashes, `.` segments dropped, `..` applied; a leading `..` survives to signal escape. */
function normalizeSegments(path: string): string[] {
  const out: string[] = [];
  for (const segment of path.replace(/\\/g, "/").split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") {
      if (out.length === 0) return ["..", ...out];
      out.pop();
      continue;
    }
    out.push(segment);
  }
  return out;
}

/** Absolute paths canonicalize to POSIX separators so one target spells the same on either OS. */
function canonicalAbsolute(path: string): string {
  const posix = path.replace(/\\/g, "/");
  const drive = /^([a-zA-Z]:)(\/.*)$/.exec(posix);
  const segments = normalizeSegments(drive ? drive[2]! : posix);
  if (segments[0] === "..") throw new ReferenceError_(`'${path}' climbs above the filesystem root`);
  return drive ? `${drive[1]}/${segments.join("/")}` : `/${segments.join("/")}`;
}

/** True when a reference's head is a relative FILE path (`./x`, `../x`) rather than a property. */
function isRelativeFilePath(body: string): boolean {
  return /^\.\.?(\/|$)/.test(body);
}

/**
 * Characters the EXPRESSION grammar uses that a reference can never contain.
 *
 * Whitespace, the two quotes, parentheses, the comma, and every operator character. Deliberately a
 * blacklist rather than a whitelist of legal path characters: an unusual filename keeps working,
 * whereas a whitelist would silently reclassify one as an expression.
 */
const EXPRESSION_ONLY = /[\s()'",!?<>=&|]/;

/**
 * True when a string is spelled entirely within the REFERENCE grammar — a path and nothing else.
 *
 * This is the one predicate that decides which pass owns an authored string: expansion resolves a
 * path against the filesystem, and the loader lowers anything else as an expression. It has to be
 * shared, because two copies of "is this a path" drifting apart would move the boundary silently.
 *
 * It is decidable because the two grammars are disjoint outside the path characters themselves —
 * an expression that is *also* a well-formed path is precisely a bare dotted path, and that overlap
 * is settled by rule rather than by guessing: a bare name is a reference, and a leading dot is
 * runtime data (REFERENCES.md §5). So a string that passes here and names no file gets a reference
 * error, never a silent reinterpretation.
 */
export function isPathSpelling(reference: string): boolean {
  return reference.length > 0 && !EXPRESSION_ONLY.test(reference);
}

/**
 * True when a binding string reads THIS INSTANCE's data — a leading dot that is not `./` or `../`.
 *
 * The dot is the whole distinction, so it is tested in exactly one place. `./x` and `../x` are
 * relative FILE paths and belong to expansion; `.inputs.issue` is runtime and belongs to the
 * desugarer; `.inputs.n === 1` is neither a path nor a bare reference, so it lowers as an
 * expression like any other computation.
 */
export function isRuntimeReference(reference: string): boolean {
  return reference.startsWith(".") && !isRelativeFilePath(reference) && isPathSpelling(reference);
}

/**
 * Split a reference into the part naming a file and the part naming a property inside it.
 *
 * Everything before the last `/` is the directory; the remainder is matched against that
 * directory's listing, longest first. `user.address` in a directory holding `user.json` is the file
 * `user.json` and the property `address`; the same reference in a directory that also holds
 * `user.address.json` is that file, whole — deterministic, but ambiguous enough in intent to warn.
 */
function splitAtFile(
  absolutePrefix: string,
  options: ReferenceOptions,
  original: string,
): { file: string; property: string[] } {
  const cut = absolutePrefix.lastIndexOf("/");
  const dir = cut > 0 ? absolutePrefix.slice(0, cut) : "/";
  const tail = absolutePrefix.slice(cut + 1);
  if (tail === "") throw new ReferenceError_(`reference '${original}' names a directory, not a file`);

  const vfs = options.vfs;
  if (!vfs) {
    // No filesystem in hand: the whole tail is the file, which is what a caller resolving ids alone
    // (the "name a state" verb) means by it.
    return { file: `${dir}/${tail}`, property: [] };
  }
  const entries = new Set(vfs.list(dir));

  const parts = tail.split(".");
  const matches: Array<{ file: string; property: string[] }> = [];
  for (let take = parts.length; take >= 1; take--) {
    const prefix = parts.slice(0, take).join(".");
    const property = parts.slice(take);
    if (entries.has(prefix)) {
      matches.push({ file: `${dir}/${prefix}`, property });
      continue;
    }
    for (const ext of DATA_EXTENSIONS) {
      if (entries.has(`${prefix}.${ext}`)) {
        matches.push({ file: `${dir}/${prefix}.${ext}`, property });
        break;
      }
    }
  }
  const best = matches[0];
  if (!best) {
    throw new ReferenceError_(`reference '${original}' matches no file in '${dir}'`);
  }
  if (matches.length > 1 && options.onWarn) {
    options.onWarn(
      `reference '${original}' matches '${best.file}' but '${matches[1]!.file}' also matches — ` +
        `the longest match wins, so adding or removing a file here changes what this reference means`,
    );
  }
  // Two spellings of one state (`plan.json` and `plan.yaml`) are a stale-file hazard, not an error.
  if (options.onWarn) {
    const stem = best.file.replace(/\.(json|yaml|yml)$/i, "");
    const rivals = DATA_EXTENSIONS.map((e) => `${stem}.${e}`).filter((p) => p !== best.file && entries.has(p.slice(dir.length + 1)));
    if (rivals.length > 0) {
      options.onWarn(`'${best.file}' and '${rivals.join("', '")}' both define the same state; '${best.file}' wins`);
    }
  }
  return best;
}

/**
 * Warn when a bare reference matches at more than one entry of the search path.
 *
 * `PATH` semantics buy shadowing along with the convenience: a file added at an EARLIER entry
 * silently changes what an existing reference means, anywhere in the workflow. `splitAtFile` already
 * warns about a collision within one directory; this is the same hazard across roots, and without it
 * the path is the hazard without the `which -a`.
 *
 * Only meaningful with a filesystem in hand — with no `vfs` every root "matches" trivially, because
 * there is nothing to check existence against. Silent under `shadowing: "override"`, where an
 * earlier entry shadowing a later one is the caller's whole intent.
 */
function warnIfShadowed(
  path: readonly string[],
  matchedAt: number,
  body: string,
  options: ReferenceOptions,
  reference: string,
  chosen: string,
): void {
  const onWarn = options.onWarn;
  if (!onWarn || options.vfs === undefined || options.shadowing === "override") return;
  for (const root of path.slice(matchedAt + 1)) {
    try {
      // Quiet: the shadowed candidate's own within-directory ambiguities are not this warning's news.
      const other = splitAtFile(canonicalAbsolute(`${root}/${body}`), { ...options, onWarn: undefined }, reference);
      onWarn(
        `reference '${reference}' resolves to '${chosen}' but '${other.file}' also matches further along the path — ` +
          `the earlier entry wins, so adding or removing a file changes what this reference means`,
      );
    } catch {
      // No match at this root: nothing is being shadowed.
    }
  }
}

/**
 * Resolve a BARE body against the search path: first match wins, exactly as a shell resolves a bare
 * command name against `PATH`.
 *
 * Shared by the two spellings that produce a bare target — a bare reference, and a `./`/`../` one
 * whose referring state is itself bare. Both have to search, or a relative link inside a base-layer
 * state would be pinned to the base layer and never see the project's override of what it points at.
 */
function searchBare(body: string, options: ReferenceOptions, reference: string): ResolvedReference {
  const path = searchPath(options);
  if (path.length === 0) {
    // No root configured: the reference IS its own id, which is what an in-memory bundle means.
    const segments = normalizeSegments(body);
    if (segments[0] === "..") throw new ReferenceError_(`reference '${reference}' climbs above the root`);
    if (segments.length === 0) throw new ReferenceError_(`reference '${reference}' names no path`);
    return { file: segments.join("/"), id: segments.join("/"), property: [], local: false };
  }
  let lastError: unknown;
  for (const [index, root] of path.entries()) {
    let found;
    try {
      found = splitAtFile(canonicalAbsolute(`${root}/${body}`), options, reference);
    } catch (e) {
      lastError = e; // not here — try the next entry
      continue;
    }
    warnIfShadowed(path, index, body, options, reference, found.file);
    return { file: found.file, id: identityOf(found.file, options), property: found.property, local: false };
  }
  // With one root there is no "path" to speak of, so `splitAtFile`'s own message — which names the
  // file it looked for — says more than a list of one would.
  if (path.length === 1) throw lastError as Error;
  throw new ReferenceError_(`reference '${reference}' matches no file on the path (${path.join(", ")})`);
}

/**
 * Resolve a `$`-rooted body against the ordered LAYER ROOTS: first match wins.
 *
 * The same rule `searchBare` uses, over a different list. Kept separate because the two lists mean
 * different things — `rootPath` entries are layer roots (`…/.jaira`), `defaultRoot` entries are the
 * directories a bare state id hangs off (`…/.jaira/workflows`) — and conflating them would make
 * `$/workflows/x` and `x` two spellings that resolve differently for no reason a reader could see.
 *
 * A miss everywhere reports the roots tried, because "not found" without "looked here" is the least
 * useful message a path-based resolver can produce.
 */
function searchRoots(
  rootPath: readonly string[],
  rest: string,
  options: ReferenceOptions,
  reference: string,
): ResolvedReference {
  let lastError: unknown;
  for (const root of rootPath) {
    let found;
    try {
      found = splitAtFile(canonicalAbsolute(`${root}/${rest}`), options, reference);
    } catch (e) {
      lastError = e;
      continue;
    }
    return { file: found.file, id: identityOf(found.file, options), property: found.property, local: false };
  }
  if (rootPath.length === 1) throw lastError as Error;
  throw new ReferenceError_(`reference '${reference}' matches no file under any layer root (${rootPath.join(", ")})`);
}

/** The search path a bare reference is tried against, in order. */
function searchPath(options: ReferenceOptions): readonly string[] {
  const d = options.defaultRoot;
  if (d === undefined) return [];
  return typeof d === "string" ? [d] : d;
}

/**
 * Fold an absolute target back to its bare id when it lands under ANY entry of the search path.
 *
 * This is what makes the path a LAYERING mechanism. A canonical id keys the snapshot hash, the event
 * log, task rows and `$STATE_ID`; folding back from every entry means `feature/plan` names the same
 * state whether the file came from the project's own root or from a shared base root further along.
 * Two files at two entries therefore share one id — which is not a collision but an OVERRIDE, and
 * resolution has already picked the winner (first entry wins) before this is asked.
 *
 * The cost is real and worth stating: an id alone no longer says which file it came from, so a
 * workflow can mean different things in two projects. What keeps a RUN honest is that execution
 * reads a pinned snapshot of the resolved bundle rather than re-resolving the live path.
 *
 * The LONGEST matching root wins, so nested entries (`…/.jaira/workflows` inside `…/.jaira`) fold to
 * the most specific id rather than to whichever happens to be listed first.
 */
function identityOf(file: string, options: ReferenceOptions): string {
  const stripped = file.replace(/\.(json|yaml|yml)$/i, "");
  let best: string | undefined;
  for (const root of searchPath(options)) {
    const canonicalRoot = canonicalAbsolute(root);
    const prefix = canonicalRoot.endsWith("/") ? canonicalRoot : `${canonicalRoot}/`;
    if (!stripped.startsWith(prefix)) continue;
    const bare = stripped.slice(prefix.length);
    if (best === undefined || bare.length < best.length) best = bare;
  }
  return best ?? stripped;
}

/** Parse and locate one reference. */
export function resolveReference(reference: string, options: ReferenceOptions = {}): ResolvedReference {
  const trimmed = reference.trim();
  if (trimmed.length === 0) throw new ReferenceError_("reference is empty");

  // `.foo.bar` — a property of the current file. `./x` and `../x` are relative FILE paths.
  if (trimmed.startsWith(".") && !isRelativeFilePath(trimmed)) {
    const property = trimmed.slice(1).split(".");
    if (property.some((p) => p === "")) throw new ReferenceError_(`reference '${reference}' has an empty path segment`);
    return { property, local: true };
  }

  let body = trimmed;
  const scheme = SCHEME.exec(body);
  if (scheme && scheme[1]!.length > 1) {
    if (scheme[1]!.toLowerCase() !== "file") {
      throw new ReferenceError_(
        `unknown scheme '${scheme[1]!}:' in reference '${reference}' — only 'file:' is supported (or no scheme)`,
      );
    }
    body = scheme[3] ?? "";
    if (!isAbsolutePath(body)) throw new ReferenceError_(`'file:' reference '${reference}' must be absolute`);
  }

  // A trailing slash means a directory was named, which has no value — caught before normalization
  // strips it (REFERENCES.md §9), and before the bare branch searches the path.
  if (body.endsWith("/") || body.endsWith("\\")) {
    throw new ReferenceError_(`reference '${reference}' names a directory, not a file`);
  }

  let absolutePrefix: string;
  if (isAbsolutePath(body)) {
    absolutePrefix = canonicalAbsolute(body);
  } else if (body.startsWith("$")) {
    const rootVar = ROOT_VAR.exec(body);
    if (!rootVar) throw new ReferenceError_(`malformed root variable in reference '${reference}'`);
    const rest = rootVar[2] ?? "";
    // `$` alone SEARCHES `rootPath` — the ordered layer roots — so a shared fragment (a prompt, a
    // type, a guard, an operation document) layers exactly as a shared state does. Without this,
    // layering would cover whole state files and nothing they are assembled from, which is half a
    // feature. A NAMED root still names exactly one place: that is what `$JAIRA` is for.
    if (rootVar[1] === undefined && options.rootPath !== undefined && options.rootPath.length > 0) {
      return searchRoots(options.rootPath, rest, options, reference);
    }
    // `$` with no `rootPath` configured falls back to `$JAIRA`, which is what it has always meant.
    const name = rootVar[1] ?? "JAIRA";
    const root = options.roots?.[name];
    if (root === undefined) {
      const known = Object.keys(options.roots ?? {});
      throw new ReferenceError_(
        `unknown root '$${rootVar[1] ?? ""}' in reference '${reference}'` +
          (known.length > 0 ? ` — known: ${known.map((r) => `$${r}`).join(", ")}` : ""),
      );
    }
    absolutePrefix = canonicalAbsolute(`${root}/${rest}`);
  } else if (isRelativeFilePath(body)) {
    const from = options.from;
    if (from === undefined) {
      throw new ReferenceError_(`relative reference '${reference}' has no referring state to resolve against`);
    }
    const segments = normalizeSegments(`${from}/${body}`);
    if (segments[0] === "..") {
      throw new ReferenceError_(`relative reference '${reference}' from '${from}' climbs above the root`);
    }
    if (!isAbsolutePath(from)) {
      // The referring id is bare, so the target has a bare spelling too — and it is SEARCHED like any
      // other bare reference. Anchoring it to the primary root instead would mean a `./child` inside a
      // base-layer state could never reach the project's override of that child.
      if (segments.length === 0) throw new ReferenceError_(`reference '${reference}' names no path`);
      return searchBare(segments.join("/"), options, reference);
    }
    // An absolute referring id is out of tree and stays there: it has no bare spelling to search with.
    absolutePrefix = /^[a-zA-Z]:/.test(from.replace(/\\/g, "/"))
      ? `${segments[0]}/${segments.slice(1).join("/")}`
      : `/${segments.join("/")}`;
  } else {
    return searchBare(body, options, reference);
  }

  const { file, property } = splitAtFile(absolutePrefix, options, reference);
  return { file, id: identityOf(file, options), property, local: false };
}

// --- Loading a reference's value ---------------------------------------------

/** What a referenced file deserializes to, by extension (REFERENCES.md §4.3). */
export function parseReferencedFile(path: string, text: string): unknown {
  const ext = /\.([^./\\]+)$/.exec(path)?.[1]?.toLowerCase();
  if (ext === "json") {
    try {
      return JSON.parse(text) as unknown;
    } catch (e) {
      throw new ReferenceError_(`'${path}' is not valid JSON: ${(e as Error).message}`);
    }
  }
  if (ext === "yaml" || ext === "yml") {
    let value: unknown;
    try {
      // Duplicate keys are an authoring mistake, not last-wins.
      value = parseYaml(text, { uniqueKeys: true });
    } catch (e) {
      throw new ReferenceError_(`'${path}' is not valid YAML: ${(e as Error).message}`);
    }
    assertJsonRepresentable(value, path, new Set());
    return value;
  }
  // Anything else IS its text — which is what makes a prompt in a .md file work.
  return text;
}

/**
 * YAML is a superset of JSON, and the excess cannot survive the rest of the system: a non-string
 * key or a `Date` has no canonical form, and an anchor/alias cycle sends `canonicalize` into
 * unbounded recursion. Both are refused here, where the file name is still in hand.
 */
function assertJsonRepresentable(value: unknown, path: string, seen: Set<object>): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ReferenceError_(`'${path}': ${String(value)} is not representable in JSON`);
    return;
  }
  if (typeof value !== "object") {
    throw new ReferenceError_(`'${path}': ${typeof value} is not representable in JSON`);
  }
  if (seen.has(value)) throw new ReferenceError_(`'${path}' contains an alias cycle`);
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonRepresentable(item, path, seen);
  } else if (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null) {
    for (const [key, item] of Object.entries(value)) {
      if (typeof key !== "string") throw new ReferenceError_(`'${path}' has a non-string key`);
      assertJsonRepresentable(item, path, seen);
    }
  } else {
    throw new ReferenceError_(`'${path}' contains a ${value.constructor?.name ?? "non-plain"} value, which JSON cannot represent`);
  }
  seen.delete(value);
}

/** Walk a property path into a loaded value. */
export function selectProperty(value: unknown, property: readonly string[], reference: string): unknown {
  let current = value;
  for (const [index, key] of property.entries()) {
    if (typeof current === "string") {
      throw new ReferenceError_(`reference '${reference}' reads property '${key}' of text — text has no properties`);
    }
    if (current === null || typeof current !== "object") {
      throw new ReferenceError_(`reference '${reference}' has no '${property.slice(0, index + 1).join(".")}'`);
    }
    // OWN properties only. A transclusion is spliced into the document, so an inherited hit
    // (`$/types/user.constructor`) put a FUNCTION where a node was expected rather than reporting
    // that the file has no such property.
    current = Object.hasOwn(current, key) ? (current as Record<string, unknown>)[key] : undefined;
    if (current === undefined) {
      throw new ReferenceError_(`reference '${reference}' has no '${property.slice(0, index + 1).join(".")}'`);
    }
  }
  return current;
}
