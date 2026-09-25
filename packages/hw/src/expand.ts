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
import { bindingForDocument, RESOLVER_REFS } from "./format.js";
import { canonicalOperation, ExprError, isSpread, parseExpression, pathOf, type Expr } from "./expr.js";
import { evaluateAt, holdsLambda } from "./loadExpr.js";
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
    // An explicit `$ref` EXPRESSION is evaluated as expansion will evaluate it, over raw documents —
    // its target's own references do not matter to which KEYS it has.
    if (reference !== undefined && isPlainObject(value) && !isPathSpelling(reference) && !reference.startsWith("...") && depth <= 8) {
      try {
        const target = evaluateAt(parseExpression(reference), new Map(), {
          reference: (spelling) => loadReferenced(resolveReference(spelling, refOptions(at)), spelling, at, undefined),
        });
        return isPlainObject(target) ? { ...target, ...value } : target;
      } catch {
        return undefined;
      }
    }
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
    const list: Shape = shape.t === "array" ? shape : { t: "array", of };
    const out: unknown[] = [];
    value.forEach((item, i) => {
      const at = [...path, String(i)];
      // `{ "$ref": "...<reference or expression>" }` as an ITEM splices the list it names into this
      // one (SPEC §5.4) — which is how a project's list says "mine, then the base layer's".
      const spread = spreadOf(item);
      if (spread === undefined) out.push(expandNode(item, of, options, at, active));
      else out.push(...expandSpread(spread, item as Record<string, unknown>, list, options, at, active));
    });
    return out;
  }

  if (!isPlainObject(value)) return value;

  // `{"$ref": …}` — the explicit form. Legal in any position, and the ONLY form that works where
  // the expected type is a string or unknown.
  if (typeof value[REF_KEY] === "string") {
    const reference = value[REF_KEY];
    const overrides = { ...value };
    delete overrides[REF_KEY];
    if (reference.startsWith(SPREAD)) {
      throw new ReferenceError(
        `${path.join(".")}: '$ref': '${reference}' — a leading '${SPREAD}' splices a list into the list it is an item of, and this '$ref' is not an item of a list`,
      );
    }
    // An EXPRESSION over documents (SPEC §5.4): evaluated now, at load, and its value
    // stands here exactly as a referenced node would — overridden by sibling keys the same way.
    if (!isPathSpelling(reference)) {
      const computed = expandExpression(reference, shape, options, path, active);
      if (Object.keys(overrides).length === 0) return computed;
      if (!isPlainObject(computed)) {
        throw new ReferenceError(`${path.join(".")}: '$ref' expression '${reference}' evaluated to ${describe(computed)} but sibling keys were given to override it`);
      }
      const expandedOverrides = expandNode(overrides, shape, options, path, active) as Record<string, unknown>;
      return mergeOperationFields(computed as never, expandedOverrides as never) as unknown;
    }
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

// --- `$ref` expressions (SPEC §5.4) ---------------------------------------------------

/** The prefix that makes a `$ref` item SPLICE its list into the list it stands in. */
const SPREAD = "...";

/** The body of a spread item — `{ "$ref": "...x" }` → `x` — or `undefined` for any other item. */
function spreadOf(item: unknown): string | undefined {
  if (!isPlainObject(item) || typeof item[REF_KEY] !== "string") return undefined;
  const reference = item[REF_KEY];
  return reference.startsWith(SPREAD) ? reference.slice(SPREAD.length).trim() : undefined;
}

/**
 * Expand one spread item to the items it splices in.
 *
 * The body is a reference or an expression, read as the LIST it stands in — so a reference to
 * another layer's `transitions` expands with the transitions shape, in its own file's scope, exactly
 * as `{ "$ref": "$BASE/x.transitions" }` in place of the whole list would. Whatever it produces must
 * be a list: splicing anything else has no meaning, and is refused rather than wrapped.
 */
function expandSpread(body: string, item: Record<string, unknown>, list: Shape, options: Walk, path: string[], active: Set<string>): unknown[] {
  const siblings = Object.keys(item).filter((k) => k !== REF_KEY);
  if (siblings.length > 0) {
    throw new ReferenceError(`${path.join(".")}: a spread '$ref' splices a list, so it takes no sibling keys to override it (found ${siblings.map((k) => `'${k}'`).join(", ")})`);
  }
  if (body.length === 0) throw new ReferenceError(`${path.join(".")}: '$ref': '${SPREAD}' names nothing to splice`);
  const value = isPathSpelling(body)
    ? asUndefinedName(body, path, options, () => expandReferenced(body, list, options, path, active, undefined))
    : expandExpression(body, list, options, path, active);
  if (!Array.isArray(value)) {
    throw new ReferenceError(`${path.join(".")}: '$ref': '${SPREAD}${body}' splices a list into this one, but it is ${describe(value)}`);
  }
  return value;
}

/**
 * Evaluate a `$ref` EXPRESSION at load, for a position of this shape.
 *
 * A reference inside it is resolved exactly as a path `$ref` is — found along the same roots, its
 * target expanded in its OWN file's scope, cycles refused — and read with the shape its VALUE will
 * have in the result (`referenceShapes`): the reference `filter` keeps items of is read as the list
 * this position holds; one read only to test something against is read as plain data.
 *
 * What comes back must fit the position: a list where a list belongs, an object where an object
 * does. A lambda cannot be a document value, so a result holding one is refused too.
 */
function expandExpression(source: string, shape: Shape, options: Walk, path: string[], active: Set<string>): unknown {
  const where = path.length > 0 ? `${path.join(".")}: ` : "";
  let parsed: Expr;
  try {
    parsed = parseExpression(source);
  } catch (e) {
    throw new ReferenceError(`${where}'$ref' '${source}' is neither a path nor an expression that parses: ${(e as Error).message}`);
  }
  const shapes = referenceShapes(parsed, shape);
  let value: unknown;
  try {
    value = evaluateAt(parsed, new Map(), {
      reference: (spelling) => expandReferenced(spelling, shapes.get(spelling) ?? { t: "any" }, options, path, active, undefined),
    });
  } catch (e) {
    const cause = e instanceof ExprError || e instanceof ReferenceError ? e.message : String(e);
    throw new ReferenceError(`${where}'$ref' expression '${source}' could not be evaluated: ${cause}`);
  }
  if (holdsLambda(value)) throw new ReferenceError(`${where}'$ref' expression '${source}' evaluated to a lambda, which is not a document value`);
  if (value === undefined) throw new ReferenceError(`${where}'$ref' expression '${source}' evaluated to nothing`);
  if (shape.t === "array" && !Array.isArray(value)) {
    throw new ReferenceError(`${where}'$ref' expression '${source}' must produce a list here, but produced ${describe(value)}`);
  }
  if (shape.t === "object" && !isPlainObject(value)) {
    throw new ReferenceError(`${where}'$ref' expression '${source}' must produce an object here, but produced ${describe(value)}`);
  }
  return value;
}

/**
 * The shape each REFERENCE in an expression is read with — the position's own shape for a reference
 * whose value is (a selection of) the result, and plain data for every other.
 *
 * Why it matters: expansion reads a node by what belongs where it lands (a string where an object
 * belongs is a reference). A list `filter` keeps items of lands HERE, so it is read as this position
 * is; a list of names the lambda only tests membership in lands nowhere, and reading it as a list of
 * transitions would resolve every name in it as a file.
 */
function referenceShapes(expr: Expr, position: Shape): Map<string, Shape> {
  const out = new Map<string, Shape>();
  const any: Shape = { t: "any" };
  const elementOf = (s: Shape): Shape => (s.t === "array" ? s.of : any);
  const listOf = (s: Shape): Shape => (s.t === "any" ? any : { t: "array", of: s });
  const visit = (e: Expr, at: Shape, bound: ReadonlySet<string>): void => {
    const note = (spelling: string): void => {
      if (!out.has(spelling) || out.get(spelling)!.t === "any") out.set(spelling, at);
    };
    switch (e.type) {
      case "ident":
        if (!bound.has(e.name)) note(e.name);
        return;
      case "member": {
        const p = pathOf(e);
        if (p !== undefined) {
          if (!bound.has(p[0]!)) note(p.join("."));
          return;
        }
        visit(e.obj, any, bound);
        return;
      }
      case "lit":
      case "self":
        return;
      case "array":
        for (const item of e.items) visit(item, elementOf(at), bound);
        return;
      case "object":
        for (const entry of e.entries) visit(entry.value, any, bound);
        return;
      case "lambda": {
        const inner = new Set([...bound, ...e.params]);
        visit(e.body, any, inner);
        return;
      }
      case "call": {
        if (e.callee.type === "member") {
          const args = [e.callee.obj, ...e.args.map((a) => (isSpread(a) ? a.value : a))];
          visitApplied(e.callee.prop, args, at, bound);
          return;
        }
        visit(e.callee, any, bound);
        for (const a of e.args) visit(isSpread(a) ? a.value : a, any, bound);
        return;
      }
      case "apply": {
        const args = e.args.map((a) => (isSpread(a) ? a.value : a));
        const dot = e.op.lastIndexOf(".");
        const name = dot > 0 ? e.op.slice(dot + 1) : e.op;
        // A dotted callee is a receiver call on its head (see `loadExpr.ts`): the head is argument 0.
        if (dot > 0 && SELECTING.has(canonicalOperation(name))) {
          const head = e.op.slice(0, dot);
          if (!bound.has(head.split(".")[0]!)) {
            if (!out.has(head) || out.get(head)!.t === "any") out.set(head, argumentShape(canonicalOperation(name), 0, at));
          }
          args.forEach((a, i) => visit(a, argumentShape(canonicalOperation(name), i + 1, at), bound));
          return;
        }
        visitApplied(e.op, args, at, bound);
        return;
      }
    }
  };
  const visitApplied = (op: string, args: readonly Expr[], at: Shape, bound: ReadonlySet<string>): void => {
    const name = canonicalOperation(op);
    args.forEach((a, i) => visit(a, argumentShape(name, i, at), bound));
  };
  const argumentShape = (op: string, i: number, at: Shape): Shape => {
    switch (op) {
      case RESOLVER_REFS.cond:
        return i === 0 ? any : at;
      case RESOLVER_REFS.and:
      case RESOLVER_REFS.or:
      case "coalesce":
      case "concat":
        return at;
      case "filter":
      case "slice":
      case "reverse":
      case "sort":
      case "sortBy":
      case "unique":
        return i === 0 ? at : any;
      case "append":
        return i === 0 ? at : elementOf(at);
      case "first":
      case "last":
      case "at":
      case "find":
        return i === 0 ? listOf(at) : any;
      default:
        return any;
    }
  };
  visit(expr, position, new Set());
  return out;
}

/** The operations whose result is (a selection of) their first argument — see `referenceShapes`. */
const SELECTING: ReadonlySet<string> = new Set(["filter", "slice", "reverse", "sort", "sortBy", "unique", "concat", "append", "first", "last", "at", "find", "coalesce"]);

function describe(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "nothing";
  if (Array.isArray(value)) return "a list";
  return typeof value === "object" ? "an object" : `a ${typeof value}`;
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
    if (isPlainObject(alternative) && typeof alternative[REF_KEY] === "string" && alternative[REF_KEY].startsWith("$") && isPathSpelling(alternative[REF_KEY])) {
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
    if (reference === undefined || reference.startsWith("$") || !isPathSpelling(reference) || (reference !== name && options.enclosing?.[reference] === undefined)) {
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
