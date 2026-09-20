/**
 * Reference expansion — the load-time pre-pass (REFERENCES.md §4, §8).
 *
 * A DOCUMENT reference is templating: the referenced node is spliced in and then behaves exactly as
 * if the author had typed it. That is what makes cross-file references well-defined without asking
 * "which instance?" — nothing has run yet, so there is no instance to be ambiguous about.
 *
 * Expansion runs before children inference, environment inheritance, desugaring and validation, so
 * every one of those passes sees a document as authored-in-full and needs no knowledge of
 * references at all.
 *
 * A RUNTIME reference (`.children.critique.output.outcome`) is left alone here. It sits in a
 * binding, it addresses this instance's data, and the desugarer lowers it like any other binding.
 *
 * ## The second reading of a variable string (NAMES.md §1)
 *
 * A mismatch marks a VARIABLE STRING, and a variable string is read two ways. Starting with `$` it
 * is a reference, resolved here, and it never walks scopes. Anything else is a SCOPED NAME first —
 * if an enclosing scope declares it (`environment.names`), or the position itself provides what a
 * name is bound to (`session`) — and a path off the default root only after that (§6's order).
 *
 * A scoped name is NOT resolved here, and cannot be: its identity is an instance's address and its
 * configuration is collected from the whole tree. Expansion's whole job for one is to recognize it,
 * leave it in the one canonical spelling — `{ "$ref": name, …overrides }` — and keep its hands off,
 * so the loader finds every use by shape without knowing which positions could have held one.
 */
import { mergeOperationFields } from "./merge.js";
import { bindingForDocument } from "./format.js";
import {
  isDataFile,
  isPathSpelling,
  isRuntimeReference,
  parseReferencedFile,
  ReferenceError,
  resolveReference,
  selectProperty,
  type ReferenceOptions,
  type ResolvedReference,
  type Vfs,
} from "./reference.js";
import { fieldShape, STATE_SHAPE, type Shape } from "./shape.js";

export interface ExpandOptions {
  vfs: Vfs;
  /** One root, or the ordered search path a bare reference is tried along (EXPRESSIONS.md §4). */
  defaultRoot?: string | readonly string[];
  roots?: Readonly<Record<string, string>>;
  /** The ordered layer roots a bare `$` searches — see `ReferenceOptions.rootPath`. */
  rootPath?: readonly string[];
  /** Canonical id of the file being expanded — the base for `./` and for a same-file reference. */
  from: string;
  onWarn?: (message: string) => void;
  /** Files pulled in by expansion, for the snapshot closure (§8.1). Absolute paths. */
  onRead?: (file: string) => void;
  /** Whether shadowing across path entries is reported — see `ReferenceOptions.shadowing`. */
  shadowing?: "warn" | "override";
  /**
   * The `names` entries ENCLOSING this state — what its ancestors' environments declared, by name.
   * With the state's own, which expansion reads off the document itself, these are the names a bare
   * string may mean before it means a file (NAMES.md §6). Only the keys are read.
   */
  enclosing?: Readonly<Record<string, unknown>>;
}

/** Expansion's own state: the options, plus the names visible at the node being walked. */
interface Walk extends ExpandOptions {
  visible: ReadonlySet<string>;
}

/** The key that spells a reference where a plain string is expected. */
export const REF_KEY = "$ref";

/**
 * Expand every document reference in one state file.
 *
 * The walk is guided by {@link STATE_SHAPE}: at each position it knows what type belongs there, and
 * a mismatch — a string where an object is expected, or a `{"$ref"}` where a string is — is what
 * marks a reference. Nothing else is treated as one, so a template that happens to contain a path
 * stays a template.
 */
export function expandReferences(document: unknown, options: ExpandOptions): unknown {
  const walk: Walk = { ...options, visible: new Set(Object.keys(options.enclosing ?? {})) };
  if (!isPlainObject(document) || document.environment === undefined) return expandNode(document, STATE_SHAPE, walk, [], new Set());
  // The state's OWN names are visible to its whole document, so they have to be known before any of
  // it is expanded.
  const own: Walk = { ...walk, visible: new Set([...walk.visible, ...peekNames(document.environment, walk)]) };
  return expandNode(document, STATE_SHAPE, own, [], new Set());
}

/**
 * The names an `environment` block declares, read WITHOUT expanding it.
 *
 * Expansion needs a block's own names before it can expand the block — a use beside the entry
 * (`"model": { "$ref": "plan" }` next to `names.plan`) must not be tried as a file — so this follows
 * only the top-level transclusion, of the block and of its `names`, and reads keys. Anything it
 * cannot read it skips: the real expansion is about to meet the same problem and report it properly.
 */
function peekNames(environment: unknown, walk: Walk): string[] {
  const follow = (value: unknown, at: Walk, depth: number): unknown => {
    const reference = typeof value === "string" ? value : isPlainObject(value) && typeof value[REF_KEY] === "string" ? value[REF_KEY] : undefined;
    if (reference === undefined || !reference.startsWith("$") || depth > 8) return value;
    try {
      const resolved = resolveReference(reference, refOptions(at));
      const target = follow(loadReferenced(resolved, reference, at, undefined), resolved.local ? at : { ...at, from: resolved.id ?? at.from }, depth + 1);
      return isPlainObject(value) && isPlainObject(target) ? { ...target, ...value } : target;
    } catch {
      return undefined;
    }
  };
  const block = follow(environment, walk, 0);
  const names = isPlainObject(block) ? follow(block.names, walk, 0) : undefined;
  return isPlainObject(names) ? Object.keys(names) : [];
}

/**
 * Is this variable string a SCOPED NAME rather than a reference (NAMES.md §1, §6)?
 *
 * `$…` never is — a reference starts at its root and does not walk scopes. Otherwise it is one when
 * the position provides (there is nothing else it could be), or when an enclosing scope declares it:
 * scopes are searched before the position's default root, so declaring a name shadows a file of the
 * same spelling, visibly, in the environment that declared it.
 *
 * "First USABLE match wins" (§6), and a name is usable only where a VALUE is read: it reads per
 * instance, so it cannot stand for structure the loader needs before any instance exists — a slot
 * map, a child mount, an operation. There a declared name is passed over and the string means what
 * it always did, a path off the default root. Reading it as the name would hand the loader
 * `{ "$ref" }` as though it were the slots themselves.
 */
function isScopedName(reference: string, shape: Shape, walk: Walk): boolean {
  if (reference.startsWith("$")) return false;
  if (shape.t === "name") return true;
  if (!walk.visible.has(reference)) return false;
  return (shape.t !== "object" && shape.t !== "array") || shape.value === true;
}

/** A bare word that is neither a usable name nor a file is, to its author, an undefined name. */
function asUndefinedName<T>(reference: string, path: readonly string[], walk: Walk, resolve: () => T): T {
  try {
    return resolve();
  } catch (e) {
    if (!(e instanceof ReferenceError) || /[/.$]/.test(reference)) throw e;
    throw new ReferenceError(
      walk.visible.has(reference)
        ? `${path.join(".")}: '${reference}' is a declared name, but this position holds structure rather than a value, so a name cannot stand here — a name goes where a value is read (a binding, an argument, a call setting). Read as a file instead: ${e.message}`
        : `${path.join(".")}: '${reference}' is not a name any enclosing scope declares (environment.names), and this position cannot provide one — ${e.message}`,
    );
  }
}

function refOptions(options: ExpandOptions): ReferenceOptions {
  return {
    ...(options.defaultRoot !== undefined ? { defaultRoot: options.defaultRoot } : {}),
    ...(options.roots !== undefined ? { roots: options.roots } : {}),
    ...(options.rootPath !== undefined ? { rootPath: options.rootPath } : {}),
    from: options.from,
    vfs: options.vfs,
    ...(options.onWarn !== undefined ? { onWarn: options.onWarn } : {}),
    ...(options.shadowing !== undefined ? { shadowing: options.shadowing } : {}),
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** A reference is LOCAL (same file) when it named no file — `.outputs.x`. */
function loadReferenced(resolved: ResolvedReference, reference: string, options: ExpandOptions, self: unknown): unknown {
  if (resolved.local) return selectProperty(self, resolved.property, reference);
  const file = resolved.file!;
  const text = options.vfs.read(file);
  if (text === undefined) throw new ReferenceError(`reference '${reference}' names '${file}', which does not exist`);
  options.onRead?.(file);
  return selectProperty(parseReferencedFile(file, text), resolved.property, reference);
}

/**
 * Resolve one reference and expand what it produced.
 *
 * The referenced node is expanded IN ITS OWN FILE's scope, so a fragment's `./` and `.foo` mean
 * what they mean where the fragment lives, not where it was pasted. Anything else would make a
 * fragment's meaning depend on its callers.
 */
function expandReferenced(
  reference: string,
  shape: Shape,
  options: Walk,
  path: string[],
  active: Set<string>,
  self: unknown,
): unknown {
  const resolved = resolveReference(reference, refOptions(options));
  const key = `${resolved.local ? options.from : resolved.file}#${resolved.property.join(".")}`;
  if (active.has(key)) {
    throw new ReferenceError(`reference cycle: ${[...active, key].join(" → ")}`);
  }
  const value = loadReferenced(resolved, reference, options, self);
  const nested: Walk = resolved.local ? options : { ...options, from: resolved.id ?? options.from };
  // Text is a leaf: there is nothing inside a prompt file to expand.
  if (typeof value === "string") return value;
  return expandNode(value, shape, nested, path, new Set([...active, key]));
}

/**
 * Resolve a path in BINDING position, and say what the target is rather than assuming.
 *
 * Three answers, decided by the target itself:
 *
 *  - a binding form (`{child}`, `{expr}`, `.inputs.x`, an operation document) is spliced as-is and
 *    behaves as though typed — including re-entering the desugarer, so a fragment may itself be a
 *    reference;
 *  - TEXT (a `.md`) is a text literal. This is the distinction that needs the file type to make:
 *    the text of a prompt fragment is a string, and so is a binding, and reading one as the other
 *    would parse a prompt as an expression;
 *  - anything else is data, wrapped as a JSON literal.
 *
 * Wrapping here rather than in the desugarer is deliberate. Only this pass knows a value arrived
 * from a FILE, and that is exactly what licenses reading an unrecognized object as data: an
 * unrecognized object typed inline is a misspelled binding tag, and should still be the error it
 * has always been.
 */
function expandBinding(reference: string, shape: Shape, options: Walk, path: string[], active: Set<string>): unknown {
  const resolved = expandReferenced(reference, shape, options, path, active, undefined);
  return bindingForDocument(resolved, isDataFile(resolveReference(reference, refOptions(options)).file));
}

function expandNode(value: unknown, shape: Shape, options: Walk, path: string[], active: Set<string>): unknown {
  // A reference-typed string is a path we RESOLVE elsewhere (naming a state), never transcluded.
  if (shape.t === "ref") return value;

  // Inside a JSON Schema, `$ref` belongs to JSON Schema — we never claim the key (§6). Our
  // references there are the bare-string form, which JSON Schema has no use for.
  if (shape.t === "schema") return expandSchema(value, options, path, active);

  if (shape.t === "names") return expandNames(value, options, path, active);

  if (typeof value === "string") {
    // A position that PROVIDES takes a name as written; only a `$…` string there is a reference.
    if (shape.t === "name") return isScopedName(value, shape, options) ? value : expandReferenced(value, shape, options, path, active, undefined);
    // A binding string is one of three things, and the two predicates below are the whole rule:
    //
    //  - a RUNTIME reference (`.inputs.issue`) — this instance's data, which the desugarer lowers;
    //  - an EXPRESSION (`add(.inputs.n, 1)`) — also the desugarer's, since resolving the names
    //    inside it is lowering's job, not expansion's;
    //  - a path — a document reference, resolved and spliced in here.
    //
    // A bare path lands in the third case, which is what makes `"binding": "lib/defaults"` mean the
    // node at that path. What it resolves TO decides how it then reads (an operation, a value,
    // another binding), and that dispatch is `desugarBinding`'s — it already is exactly that.
    if (shape.t === "binding") {
      if (isRuntimeReference(value) || !isPathSpelling(value)) return value;
      return expandBinding(value, shape, options, path, active);
    }
    // A string where an object or array belongs IS a reference (§3).
    if (shape.t === "object" || shape.t === "array") {
      // …unless an enclosing scope declares it and a name is USABLE here, in which case it is that
      // NAME (NAMES.md §6).
      if (isScopedName(value, shape, options)) return { [REF_KEY]: value };
      return asUndefinedName(value, path, options, () => expandReferenced(value, shape, options, path, active, undefined));
    }
    return value;
  }

  if (Array.isArray(value)) {
    const of = shape.t === "array" ? shape.of : { t: "any" as const };
    return value.map((item, i) => expandNode(item, of, options, [...path, String(i)], active));
  }

  if (!isPlainObject(value)) return value;

  // `{"$ref": …}` — the explicit form. Legal in any position, and the ONLY form that works where
  // the expected type is a string or unknown.
  if (typeof value[REF_KEY] === "string") {
    const reference = value[REF_KEY];
    const overrides = { ...value };
    delete overrides[REF_KEY];
    // A scoped name stays a use. Its sibling keys configure the NAME rather than override this
    // position, so they are expanded as the untyped block a `names` entry is, not by this shape.
    if (isScopedName(reference, shape, options)) {
      return { [REF_KEY]: reference, ...(expandNode(overrides, { t: "any" }, options, path, active) as Record<string, unknown>) };
    }
    const resolved = asUndefinedName(reference, path, options, () => expandReferenced(reference, shape, options, path, active, undefined));
    if (Object.keys(overrides).length === 0) return resolved;
    if (typeof resolved === "string" || !isPlainObject(resolved)) {
      throw new ReferenceError(
        `reference '${reference}' resolved to a ${typeof resolved} but sibling keys were given to override it`,
      );
    }
    // Sibling keys override, and their ORDER is ignored — see REFERENCES.md §4.1 for why that is
    // forced rather than chosen. The merge is the algebra `environment` inheritance already uses.
    const expandedOverrides = expandNode(overrides, shape, options, path, active) as Record<string, unknown>;
    return mergeOperationFields(resolved as never, expandedOverrides as never) as unknown;
  }

  // ALTERNATIVES (NAMES.md §6): one that references something and finds nothing is not USABLE.
  if (Array.isArray(value[ANY_KEY])) {
    const rest: Record<string, unknown> = { ...value };
    delete rest[ANY_KEY];
    return {
      ...(expandNode(rest, shape, options, path, active) as Record<string, unknown>),
      [ANY_KEY]: expandAlternatives(value[ANY_KEY], options, [...path, ANY_KEY], active),
    };
  }

  // An `environment` block's own names are visible inside it. At state level `expandReferences` has
  // already said so for the whole document; this is the child MOUNT's layer, whose names reach that
  // child and nothing beside it.
  const within: Walk =
    shape.t === "object" && shape.fields?.names?.t === "names" && value.names !== undefined
      ? { ...options, visible: new Set([...options.visible, ...peekNames(value, options)]) }
      : options;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const childShape = fieldShape(shape, key) ?? { t: "any" as const };
    out[key] = expandNode(child, childShape, within, [...path, key], active);
  }
  return out;
}

/** The key that lists alternatives. */
const ANY_KEY = "$any";

/**
 * Expand the alternatives of a `$any`, dropping the ones that are not USABLE (NAMES.md §6).
 *
 * "First usable alternative wins" needs a meaning for usable, and the only one a load can decide is
 * whether the alternative is THERE: one written as a reference that resolves to nothing — no file on
 * any layer — is skipped, with a warning, where anywhere else it would be a load error. That is the
 * point of listing alternatives across layers: `[{ "$ref": "$/roles.plan" }, { "model": "…" }]`
 * reads the project's role where there is one and falls through where there is not. Anything that
 * goes wrong INSIDE a target that was found — a cycle, a reference of its own that matches nothing —
 * is still an error: that is a mistake, not an absence. So is a list with nothing left in it.
 * `null` is a value, never an absence: it is kept.
 */
function expandAlternatives(alternatives: unknown[], options: Walk, path: string[], active: Set<string>): unknown[] {
  const usable: unknown[] = [];
  alternatives.forEach((alternative, i) => {
    const at = [...path, String(i)];
    // ABSENCE is asked of the alternative's OWN reference and of nothing below it: a role file that
    // is there and itself points at something missing is a mistake in that file, and skipping it
    // would turn a typo into a silent change of model.
    if (isPlainObject(alternative) && typeof alternative[REF_KEY] === "string" && alternative[REF_KEY].startsWith("$")) {
      const reference = alternative[REF_KEY];
      try {
        loadReferenced(resolveReference(reference, refOptions(options)), reference, options, undefined);
      } catch (e) {
        if (!(e instanceof ReferenceError)) throw e;
        options.onWarn?.(`${at.join(".")}: alternative skipped — ${e.message}`);
        return;
      }
    }
    usable.push(expandNode(alternative, { t: "any" }, options, at, active));
  });
  if (usable.length === 0) throw new ReferenceError(`${path.join(".")}: no alternative is usable — every one references something that is not there`);
  return usable;
}

/**
 * Expand a `names` block (NAMES.md §4).
 *
 * An entry is an untyped block, so inside one only the explicit `{ "$ref" }` form counts — with the
 * one reading that belongs to this position: a `$ref` to a NAME pastes that name's enclosing entry,
 * and the entry's own name always means the enclosing one. That is what lets `"impl": { "$ref":
 * "impl", "from": "dev" }` say "theirs, with this changed" without being a cycle.
 *
 * The paste itself is the loader's (`normalizeNames`), which holds the enclosing blocks; here the
 * entry is only kept out of file resolution, with its override keys expanded like any other block.
 */
function expandNames(value: unknown, options: Walk, path: string[], active: Set<string>): unknown {
  if (typeof value === "string") return expandReferenced(value, { t: "names" }, options, path, active, undefined);
  if (!isPlainObject(value)) return value;
  if (typeof value[REF_KEY] === "string") return expandNode(value, { t: "object", rest: { t: "any" } }, options, path, active);
  const out: Record<string, unknown> = {};
  for (const [name, entry] of Object.entries(value)) {
    const reference = typeof entry === "string" ? entry : isPlainObject(entry) && typeof entry[REF_KEY] === "string" ? entry[REF_KEY] : undefined;
    const at = [...path, name];
    if (reference === undefined || reference.startsWith("$") || (reference !== name && options.enclosing?.[reference] === undefined)) {
      out[name] = expandNode(entry, typeof entry === "string" ? { t: "object", rest: { t: "any" } } : { t: "any" }, options, at, active);
      continue;
    }
    const overrides = isPlainObject(entry) ? { ...entry } : {};
    delete overrides[REF_KEY];
    out[name] = { [REF_KEY]: reference, ...(expandNode(overrides, { t: "any" }, options, at, active) as Record<string, unknown>) };
  }
  return out;
}

/**
 * Expand inside a JSON Schema, leaving JSON Schema's own `$ref` untouched.
 *
 * A shared type library is the point of expanding here at all: `{"properties": {"doc":
 * "$/types/markdown"}}` composes, where linking whole schemas would not. A bare string is never
 * valid JSON Schema, so the two vocabularies cannot collide.
 */
function expandSchema(value: unknown, options: Walk, path: string[], active: Set<string>): unknown {
  if (typeof value === "string") {
    return expandReferenced(value, { t: "schema" }, options, path, active, undefined);
  }
  if (Array.isArray(value)) return value.map((item, i) => expandSchema(item, options, [...path, String(i)], active));
  if (!isPlainObject(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    // Only a keyword whose VALUE is itself a schema recurses. `"type": "string"` is a keyword value,
    // not a nested schema, and expanding it would read `string` as a reference to a file.
    if (key === REF_KEY || !SCHEMA_VALUED.has(key)) {
      out[key] = child;
      continue;
    }
    out[key] = SCHEMA_MAP_VALUED.has(key) && isPlainObject(child)
      ? Object.fromEntries(Object.entries(child).map(([n, sub]) => [n, expandSchema(sub, options, [...path, key, n], active)]))
      : expandSchema(child, options, [...path, key], active);
  }
  return out;
}

/**
 * JSON Schema keywords whose value is a schema (or a list of them) — the only places inside a
 * schema document where a nested schema, and therefore one of our references, can appear.
 */
const SCHEMA_VALUED: ReadonlySet<string> = new Set([
  "properties",
  "patternProperties",
  "additionalProperties",
  "items",
  "prefixItems",
  "contains",
  "propertyNames",
  "definitions",
  "$defs",
  "allOf",
  "anyOf",
  "oneOf",
  "not",
  "if",
  "then",
  "else",
]);

/** Of those, the ones whose value is a MAP of name → schema rather than a schema itself. */
const SCHEMA_MAP_VALUED: ReadonlySet<string> = new Set(["properties", "patternProperties", "definitions", "$defs"]);
