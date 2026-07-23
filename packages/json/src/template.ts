/**
 * Schema TEMPLATES — the *data* form of a generic type (API.md, "Schema templates, inference, and select-typing"; ported from
 * findmyprompt `src/engine/schema/template.ts`). A template is an ordinary JSON Schema document that
 * may contain PARAMETER HOLES: an object `{ "$param": "name" }` standing for a schema supplied later.
 * `applyTemplate` substitutes one binding per hole, turning a generic schema into a concrete one.
 *
 * This is parametric polymorphism for schemas — a generic op computing its concrete output schema per
 * call. Rather than a TS function computing output-schema-from-input, the type-function IS a stored
 * template document, applied by this one generic interpreter. That is what lets a generic op carry its
 * own type behaviour as data: it stores its output type as a template, and a binding instantiates it
 * (see `ops`' `bindSignatureTemplate`, the `Signature` bridge).
 *
 * A fully-applied template carries no `$param`, so the result is an ordinary schema for `isSubschema`
 * and Ajv alike.
 *
 * Substitution is SINGLE-PASS and VERBATIM: a binding that itself contains `$param` holes is spliced
 * in unchanged — its holes are NOT recursively filled from the same `bindings`. This is deliberate and
 * load-bearing: it makes application TOTAL (no substitution cycles) and gives dependent
 * self-application its one-level-deep structure, with the residual holes collapsing to "any".
 *
 * `$param` and `x-type` (./codec) are the same SHAPE — a name inside the document resolved through a
 * map — and they coexist: `$param` binds type VARIABLES per call, `x-type` binds type NAMES globally.
 */
import type { SchemaDocument } from "./json";
import { getOwn, setOwn } from "./ownProps";

export const PARAM_KEY = "$param";

/** A parameter hole: `{ $param: "input" }`. Detected by a string `$param` own-property. */
export function isParamHole(node: unknown): node is { [PARAM_KEY]: string } {
  return (
    !!node && typeof node === "object" && !Array.isArray(node) && typeof (node as Record<string, unknown>)[PARAM_KEY] === "string"
  );
}

/** Every parameter name a template references — the type-function's formal arguments. */
export function collectParams(node: unknown, out = new Set<string>()): Set<string> {
  if (isParamHole(node)) {
    out.add(node[PARAM_KEY]);
    return out;
  }
  if (Array.isArray(node)) {
    for (const x of node) collectParams(x, out);
  } else if (node && typeof node === "object") {
    for (const v of Object.values(node)) collectParams(v, out);
  }
  return out;
}

export interface ApplyOptions {
  /** A hole with no binding: throw (default), or — when `partial` — leave the hole in place so the
   *  template can be curried against a subset of its parameters. */
  partial?: boolean;
}

/**
 * Apply a schema template: replace each `{ $param: name }` hole with `bindings[name]`. Pure and
 * structural — the result is a fresh document with no aliasing into `bindings`. A hole whose parameter
 * has no binding throws unless `partial` is set. See the module note on single-pass/verbatim
 * semantics: a bound schema is spliced in as-is and is not re-substituted.
 */
export function applyTemplate(
  template: SchemaDocument,
  bindings: Record<string, SchemaDocument>,
  opts: ApplyOptions = {},
): SchemaDocument {
  return subst(template, bindings, opts) as SchemaDocument;
}

function subst(node: unknown, bindings: Record<string, SchemaDocument>, opts: ApplyOptions): unknown {
  if (isParamHole(node)) {
    const name = node[PARAM_KEY];
    // OWN binding only. `bindings[name]` resolves `"constructor"`/`"toString"` off `Object.prototype`
    // and would splice a FUNCTION into a schema document (which then dies in canonicalize or Ajv), and
    // reads `"__proto__"` as bound-to-`{}` instead of throwing. A parameter name is author data, so all
    // three are reachable — and "application is total" only holds if unbound really means unbound.
    const bound = getOwn(bindings, name);
    if (bound === undefined) {
      if (opts.partial) return { [PARAM_KEY]: name };
      throw new Error(`applyTemplate: no binding for parameter '${name}'`);
    }
    return clone(bound); // verbatim: do NOT recurse into the binding
  }
  if (Array.isArray(node)) return node.map((x) => subst(x, bindings, opts));
  if (node && typeof node === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node)) setOwn(out, k, subst(v, bindings, opts));
    return out;
  }
  return node; // primitive
}

function clone<T>(v: T): T {
  return v === null || typeof v !== "object" ? v : (JSON.parse(JSON.stringify(v)) as T);
}

/**
 * Collapse every UNRESOLVED `$param` hole to the universal schema `{}` ("any"). Run AFTER a partial
 * binding, where the residual holes are type variables the binding genuinely cannot determine — the
 * load-bearing case being a template applied to its OWN signature, where single-pass binding splices
 * the (still holey) signature in verbatim and leaves inner holes a level down. Those are NON-UNIFORM —
 * each one's type is only knowable once its own signature exists — so the widest sound type is "any".
 * Collapsing keeps the bound schema TOTAL (no `$param` ever reaches a provider or Ajv), maximally
 * permissive, and visible to the author to correct. A no-op on a hole-free document.
 */
export function collapseHoles(template: SchemaDocument): SchemaDocument {
  return collapse(template) as SchemaDocument;
}

/**
 * Step 2 of generic instantiation: resolve a (possibly generic) schema against a TYPE MAP — a plain
 * `{ typeName → schema }` mapping. Substitutes every bound `$param`, then collapses any variable still
 * UNBOUND to the universal `{}`. No `Signature` in the path — just names → schemas, which is what keeps
 * this module in `json`, below the op model. A hole-free schema passes through untouched.
 */
export function resolveTypes(template: SchemaDocument, typeMap: Record<string, SchemaDocument>): SchemaDocument {
  return collapseHoles(applyTemplate(template, typeMap, { partial: true }));
}

function collapse(node: unknown): unknown {
  if (isParamHole(node)) return {};
  if (Array.isArray(node)) return node.map(collapse);
  if (node && typeof node === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node)) setOwn(out, k, collapse(v));
    return out;
  }
  return node;
}
