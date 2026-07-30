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
 * A RUNTIME reference (`.children.critique.outputs.outcome`) is left alone here. It sits in a
 * binding, it addresses this instance's data, and the desugarer lowers it like any other binding.
 */
import { mergeOperationFields } from "./merge";
import {
  parseReferencedFile,
  ReferenceError,
  resolveReference,
  selectProperty,
  type ReferenceOptions,
  type ResolvedReference,
  type Vfs,
} from "./reference";
import { fieldShape, STATE_SHAPE, type Shape } from "./shape";

export interface ExpandOptions {
  vfs: Vfs;
  /** One root, or the ordered search path a bare reference is tried along (EXPRESSIONS.md §4). */
  defaultRoot?: string | readonly string[];
  roots?: Readonly<Record<string, string>>;
  /** Canonical id of the file being expanded — the base for `./` and for a same-file reference. */
  from: string;
  onWarn?: (message: string) => void;
  /** Files pulled in by expansion, for the snapshot closure (§8.1). Absolute paths. */
  onRead?: (file: string) => void;
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
  return expandNode(document, STATE_SHAPE, options, [], new Set());
}

function refOptions(options: ExpandOptions): ReferenceOptions {
  return {
    ...(options.defaultRoot !== undefined ? { defaultRoot: options.defaultRoot } : {}),
    ...(options.roots !== undefined ? { roots: options.roots } : {}),
    from: options.from,
    vfs: options.vfs,
    ...(options.onWarn !== undefined ? { onWarn: options.onWarn } : {}),
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
  options: ExpandOptions,
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
  const nested: ExpandOptions = resolved.local ? options : { ...options, from: resolved.id ?? options.from };
  // Text is a leaf: there is nothing inside a prompt file to expand.
  if (typeof value === "string") return value;
  return expandNode(value, shape, nested, path, new Set([...active, key]));
}

function expandNode(value: unknown, shape: Shape, options: ExpandOptions, path: string[], active: Set<string>): unknown {
  // A reference-typed string is a path we RESOLVE elsewhere (naming a state), never transcluded.
  if (shape.t === "ref") return value;

  // Inside a JSON Schema, `$ref` belongs to JSON Schema — we never claim the key (§6). Our
  // references there are the bare-string form, which JSON Schema has no use for.
  if (shape.t === "schema") return expandSchema(value, options, path, active);

  if (typeof value === "string") {
    // A leading-dot string in a binding is a RUNTIME reference: it addresses this instance's data,
    // and the desugarer lowers it. `./x` and `../x` are relative FILE paths, so they still resolve.
    if (shape.t === "binding") {
      return /^\.\.?(\/|$)/.test(value) || !value.startsWith(".")
        ? expandReferenced(value, shape, options, path, active, undefined)
        : value;
    }
    // A string where an object or array belongs IS a reference (§3).
    if (shape.t === "object" || shape.t === "array") {
      return expandReferenced(value, shape, options, path, active, undefined);
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
    const resolved = expandReferenced(reference, shape, options, path, active, undefined);
    const overrides = { ...value };
    delete overrides[REF_KEY];
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

  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const childShape = fieldShape(shape, key) ?? { t: "any" as const };
    out[key] = expandNode(child, childShape, options, [...path, key], active);
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
function expandSchema(value: unknown, options: ExpandOptions, path: string[], active: Set<string>): unknown {
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
