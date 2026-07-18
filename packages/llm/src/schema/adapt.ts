import { findDiscriminators } from "../structured";
import { flattenForDepth } from "./flatten";
import type { AdaptNote, AdaptResult, KeywordRule, KeywordSupport, MaxDepthCountStrategy, ProviderSchemaProfile, SchemaNode } from "./profile";

/**
 * The generic, config-driven structured-output schema adapter (§5.1). Replaces the Anthropic-only
 * `patchSchemaForAnthropic`/`reconstructOutput` pair with one engine parameterized by a
 * `ProviderSchemaProfile`: it transforms the schema we send to the model, decides whether the call can
 * use strict constrained decoding (`fitsStrict`) or must fall back to an advisory hint, and returns a
 * `postProcess` that reverses every lossy transform on the way out.
 *
 * The reverse is RE-DERIVED from `(originalSchema, profile)` — never threaded as a recorded plan — the
 * same stateless pattern the old `reconstructOutput` used (it recomputes discriminators rather than
 * remembering them). Every forward transform here therefore has a deterministic inverse computable from
 * the original schema alone.
 */

// ---------------------------------------------------------------------------------------------------
// Meta-tag stripping (always-on, every profile)
// ---------------------------------------------------------------------------------------------------

/** Keywords that are NEVER useful to send to any provider: JSON Schema meta + our UI/type-identity tags
 *  (`$type`/`$param` are the schema→component registry keys, not validation keywords). */
const META_KEYWORDS = ["$schema", "$id", "$comment", "$type", "$param"] as const;

function isPlainObject(v: unknown): v is SchemaNode {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Deep clone with the meta/UI tags removed. The neutral starting point for every profile. */
function stripMetaDeep(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(stripMetaDeep);
  if (!isPlainObject(node)) return node;
  const out: SchemaNode = {};
  for (const [k, v] of Object.entries(node)) {
    if ((META_KEYWORDS as readonly string[]).includes(k)) continue;
    out[k] = stripMetaDeep(v);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------
// Node classification
// ---------------------------------------------------------------------------------------------------

const STRUCTURAL_KEYWORDS = [
  "type", "properties", "items", "prefixItems", "enum", "const",
  "oneOf", "anyOf", "allOf", "$ref", "additionalProperties", "required",
] as const;

/** An "any"/untyped node: `{}` or one carrying only annotations (title/description). It constrains
 *  nothing, so a closed-grammar (strict) decoder can't represent it. */
function isAnyNode(node: SchemaNode): boolean {
  return !STRUCTURAL_KEYWORDS.some((k) => k in node);
}

function isObjectNode(node: SchemaNode): boolean {
  return node.type === "object" || "properties" in node;
}

const ANY_ENCODED_DESC = "A JSON-encoded value (parse as JSON).";

// ---------------------------------------------------------------------------------------------------
// Root-array wrapping (for transports that require an OBJECT at the schema root)
// ---------------------------------------------------------------------------------------------------

/** The single property name a root array is wrapped under for object-root-only transports. */
const ROOT_ARRAY_KEY = "items";

function isArrayRootNode(node: SchemaNode): boolean {
  return node.type === "array" || (Array.isArray(node.type) && (node.type as unknown[]).includes("array"));
}

/** Wrap a root-array schema in a minimal single-property object. Annotations (`title`/`description`)
 *  are lifted onto the wrapper so the model still sees them at the root. The object-level policies
 *  (`additionalProperties`, required-all) are applied by `strictify`, which runs over the wrapper. */
function wrapRootArray(node: SchemaNode): SchemaNode {
  const wrapper: SchemaNode = { type: "object", properties: { [ROOT_ARRAY_KEY]: node }, required: [ROOT_ARRAY_KEY] };
  if (typeof node.title === "string") wrapper.title = node.title;
  if (typeof node.description === "string") wrapper.description = node.description;
  return wrapper;
}

/** Undo the wrap on the model's answer. Deliberately lenient: a bare array passes through (an advisory
 *  upstream that ignored the wrapper, or a transport that took the root array natively), and anything
 *  unrecognizable is returned as-is for the §4 Ajv boundary to reject. */
function unwrapRootArray(value: unknown): unknown {
  if (Array.isArray(value)) return value;
  if (isPlainObject(value) && Array.isArray(value[ROOT_ARRAY_KEY])) return value[ROOT_ARRAY_KEY];
  return value;
}

// ---------------------------------------------------------------------------------------------------
// Union flattening (the `unions: "flatten"` strategy — reused/ported from the Anthropic patch)
// ---------------------------------------------------------------------------------------------------

/**
 * Merge a `oneOf`/`anyOf`/`allOf` variant set into ONE object: the union of all variant properties,
 * with `required` = the union of every variant's required fields (deliberately over-strict — it's only
 * sent to COAX the model; `postProcess` restores correctness against the original union). `const`
 * discriminators that differ across variants collapse to an `enum` so the model may pick any.
 */
function mergeUnionVariants(variants: SchemaNode[]): SchemaNode | null {
  if (!variants.every((v) => v.type === "object" || "properties" in v)) return null;
  const mergedProps: SchemaNode = {};
  const requiredSets = variants.map((v) => new Set(Array.isArray(v.required) ? (v.required as string[]) : []));
  const allKeys = new Set<string>();
  for (const v of variants) {
    const props = isPlainObject(v.properties) ? v.properties : undefined;
    if (!props) continue;
    for (const key of Object.keys(props)) {
      allKeys.add(key);
      if (!(key in mergedProps)) {
        mergedProps[key] = props[key];
        continue;
      }
      const existing = mergedProps[key] as SchemaNode;
      const incoming = props[key] as SchemaNode;
      if (existing.const !== undefined && incoming.const !== undefined) {
        const vals = new Set<unknown>();
        if (Array.isArray(existing.enum)) (existing.enum as unknown[]).forEach((x) => vals.add(x));
        else vals.add(existing.const);
        vals.add(incoming.const);
        mergedProps[key] = { type: "string", enum: [...vals] };
      } else if (Array.isArray(existing.enum) && incoming.const !== undefined) {
        (existing.enum as unknown[]).push(incoming.const);
      }
    }
  }
  const required = [...allKeys].filter((k) => requiredSets.some((rs) => rs.has(k)));
  const merged: SchemaNode = { type: "object", properties: mergedProps };
  if (required.length > 0) merged.required = required;
  return merged;
}

// ---------------------------------------------------------------------------------------------------
// Keyword filtering
// ---------------------------------------------------------------------------------------------------

function keywordRule(keywords: KeywordSupport, kw: string): KeywordRule {
  if (Array.isArray(keywords)) return keywords.includes(kw); // whitelist: listed = supported, rest stripped
  const rule = (keywords as Record<string, KeywordRule>)[kw];
  return rule === undefined ? true : rule; // object: absent ⇒ allowed by default
}

/** Append a stripped constraint to the node's `description` (the `"describe"` strategy). */
function describeConstraint(node: SchemaNode, kw: string, value: unknown): void {
  const text = `${kw}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`;
  node.description = typeof node.description === "string" && node.description.length > 0
    ? `${node.description}\n${text}`
    : text;
}

/** Apply the profile's keyword rules to ONE node (in place). Handles strip / describe / `pos` (nested)
 *  / `allowedValues` whitelist. `oneOf`/`anyOf`/`allOf` are handled by the union pass, not here. */
function applyKeywordRules(node: SchemaNode, profile: ProviderSchemaProfile, isRoot: boolean): void {
  for (const kw of Object.keys(node)) {
    if (kw === "properties" || kw === "items" || kw === "oneOf" || kw === "anyOf" || kw === "allOf") continue;
    const rule = keywordRule(profile.keywords, kw);
    if (rule === true) continue;
    if (rule === false) {
      delete node[kw];
      continue;
    }
    if (rule === "nested-only") {
      if (isRoot) delete node[kw];
      continue;
    }
    if (rule === "describe") {
      describeConstraint(node, kw, node[kw]);
      delete node[kw];
      continue;
    }
    // object form
    if (rule.pos === "nested" && isRoot) {
      delete node[kw];
      continue;
    }
    if (rule.allowedValues) {
      // The keyword is supported only for whitelisted VALUES; an in-list value is kept, an out-of-list
      // value is stripped (or described, when `support: "describe"`).
      if (rule.allowedValues.includes(node[kw])) continue;
      if (rule.support === "describe") describeConstraint(node, kw, node[kw]);
      delete node[kw];
      continue;
    }
    if (rule.support === "strip") {
      delete node[kw];
    } else if (rule.support === "describe") {
      describeConstraint(node, kw, node[kw]);
      delete node[kw];
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// Optional handling (the capability-derived strategy ladder)
// ---------------------------------------------------------------------------------------------------

/** Make an already-cleaned child schema accept `null`, per the profile's null encoding. */
function makeNullable(schema: SchemaNode, how: ProviderSchemaProfile["nullable"]): SchemaNode {
  if (how === "nullable-flag") return { ...schema, nullable: true };
  if (how === "type-array" && typeof schema.type === "string") {
    return { ...schema, type: [schema.type, "null"] };
  }
  if (how === "type-array" && Array.isArray(schema.type)) {
    const types = schema.type as string[];
    return types.includes("null") ? schema : { ...schema, type: [...types, "null"] };
  }
  // anyOf-null, or a node with no concrete `type` to extend.
  return { anyOf: [schema, { type: "null" }] };
}

/** Force/relax `required` on an object node per `optionalSupport`. `"omit"` leaves it; `"nullable"`
 *  forces every property required and wraps the originally-optional ones nullable; `"none"` forces
 *  required with no null (the lossy floor). */
function applyOptional(node: SchemaNode, profile: ProviderSchemaProfile): void {
  if (profile.optionalSupport === "omit") return;
  const props = isPlainObject(node.properties) ? node.properties : undefined;
  if (!props) return;
  const keys = Object.keys(props);
  const required = new Set(Array.isArray(node.required) ? (node.required as string[]) : []);
  if (profile.optionalSupport === "nullable") {
    for (const k of keys) {
      if (!required.has(k)) props[k] = makeNullable(props[k] as SchemaNode, profile.nullable);
    }
  }
  node.required = keys; // force all required (both "nullable" and "none")
}

// ---------------------------------------------------------------------------------------------------
// Forward transform (the strict pass)
// ---------------------------------------------------------------------------------------------------

function strictify(input: SchemaNode, profile: ProviderSchemaProfile, isRoot: boolean): SchemaNode {
  // An "any" node has no children — encode it (or leave it for fitsStrict to reject).
  if (isAnyNode(input)) {
    if (profile.anyType === "encode-json-string") {
      const desc = typeof input.description === "string" ? `${input.description}\n${ANY_ENCODED_DESC}` : ANY_ENCODED_DESC;
      return { type: "string", description: desc };
    }
    return { ...input };
  }

  let node: SchemaNode = { ...input };

  // Unions first — flattening restructures the node into a plain object before anything else runs.
  if (profile.unions === "flatten") {
    for (const kw of ["oneOf", "anyOf", "allOf"] as const) {
      if (Array.isArray(node[kw])) {
        const merged = mergeUnionVariants(node[kw] as SchemaNode[]);
        if (merged) {
          delete node[kw];
          node = { ...node, ...merged };
        } else {
          // Non-object variants (e.g. a [{...},{type:null}] nullable union): keep the non-null branch.
          const variants = node[kw] as SchemaNode[];
          const nonNull = variants.find((v) => v.type !== "null");
          delete node[kw];
          if (nonNull) node = { ...node, ...nonNull };
        }
        break;
      }
    }
  } else {
    // anyOf mode: normalize oneOf → anyOf, leave the rest for the provider.
    if (Array.isArray(node.oneOf) && !node.anyOf) {
      node.anyOf = node.oneOf;
      delete node.oneOf;
    }
  }

  applyKeywordRules(node, profile, isRoot);

  // Recurse into children.
  if (isPlainObject(node.properties)) {
    const props = node.properties;
    for (const key of Object.keys(props)) props[key] = strictify(props[key] as SchemaNode, profile, false);
  }
  if (isPlainObject(node.items)) node.items = strictify(node.items as SchemaNode, profile, false);
  if (Array.isArray(node.anyOf)) node.anyOf = (node.anyOf as SchemaNode[]).map((v) => strictify(v, profile, false));

  // Object-level policies (run AFTER recursion so optional-wrapping sees cleaned children).
  if (isObjectNode(node)) {
    if (profile.additionalProperties === "force-false") {
      if (!("additionalProperties" in node)) node.additionalProperties = false;
    } else if (profile.additionalProperties === "strip") {
      delete node.additionalProperties;
    }
    applyOptional(node, profile);
  }

  if (Array.isArray(node.type) && profile.collapseTypeArrays) {
    const types = node.type as string[];
    node.type = types.find((t) => t !== "null") ?? types[0];
  }

  return node;
}

// ---------------------------------------------------------------------------------------------------
// Fit check (strict vs advisory)
// ---------------------------------------------------------------------------------------------------

interface SchemaStats {
  maxDepth: number;
  propertyCount: number;
  enumCount: number;
  nameEnumChars: number;
  hasAny: boolean;
  hasRef: boolean;
  rootUnion: boolean;
}

/**
 * Collect the stats `fitsStrict` gates on. `maxDepth` is counted per `strategy` (§5.1): which nodes add
 * a nesting level differs by transport (OpenAI counts only objects; the conservative default counts all).
 * A pure union node is TRANSPARENT under every mode — the matched variant occupies its position, so
 * `{anyOf:[A,B]}` is the depth of A|B, not one deeper.
 */
function collectStats(node: unknown, strategy: MaxDepthCountStrategy = "all"): SchemaStats {
  /** Does `n` add a nesting level under `strategy`? */
  const levels = (n: SchemaNode): boolean => {
    const isObject = n.type === "object" || isPlainObject(n.properties);
    const isArray = n.type === "array" || "items" in n;
    const isUnion = Array.isArray(n.oneOf) || Array.isArray(n.anyOf) || Array.isArray(n.allOf);
    if (isUnion && !isObject && !isArray) return false; // pure union — transparent
    if (strategy === "objects") return isObject;
    if (strategy === "containers") return isObject || isArray;
    return true; // "all" — every concrete node counts (objects, arrays, and the leaf)
  };

  const rec = (n: unknown, parentDepth: number, isRoot: boolean): SchemaStats => {
    const acc: SchemaStats = {
      maxDepth: parentDepth, propertyCount: 0, enumCount: 0, nameEnumChars: 0,
      hasAny: false, hasRef: false, rootUnion: false,
    };
    if (Array.isArray(n)) {
      for (const x of n) merge(acc, rec(x, parentDepth, false));
      return acc;
    }
    if (!isPlainObject(n)) return acc;

    const here = parentDepth + (levels(n) ? 1 : 0);
    acc.maxDepth = here;

    if ("$ref" in n) acc.hasRef = true;
    if (isAnyNode(n)) acc.hasAny = true;
    if (isRoot && (Array.isArray(n.oneOf) || Array.isArray(n.anyOf) || Array.isArray(n.allOf))) {
      acc.rootUnion = true;
    }
    if (Array.isArray(n.enum)) {
      acc.enumCount += n.enum.length;
      for (const e of n.enum) if (typeof e === "string") acc.nameEnumChars += e.length;
    }
    if (typeof n.const === "string") acc.nameEnumChars += n.const.length;

    if (isPlainObject(n.properties)) {
      const props = n.properties;
      const keys = Object.keys(props);
      acc.propertyCount += keys.length;
      for (const k of keys) {
        acc.nameEnumChars += k.length;
        merge(acc, rec(props[k], here, false));
      }
    }
    for (const kw of ["items", "additionalProperties"] as const) {
      if (isPlainObject(n[kw])) merge(acc, rec(n[kw], here, false));
      else if (Array.isArray(n[kw])) merge(acc, rec(n[kw], here, false));
    }
    // Union variants recurse at THIS node's depth: for a pure union `here === parentDepth`, so the
    // variant takes the union's position (transparent); the variant itself adds its own level.
    for (const kw of ["oneOf", "anyOf", "allOf"] as const) {
      if (Array.isArray(n[kw])) for (const v of n[kw] as unknown[]) merge(acc, rec(v, here, false));
    }
    return acc;
  };

  return rec(node, 0, true);
}

function merge(into: SchemaStats, from: SchemaStats): void {
  into.maxDepth = Math.max(into.maxDepth, from.maxDepth);
  into.propertyCount += from.propertyCount;
  into.enumCount += from.enumCount;
  into.nameEnumChars += from.nameEnumChars;
  into.hasAny ||= from.hasAny;
  into.hasRef ||= from.hasRef;
  into.rootUnion ||= from.rootUnion;
}

/** Decide whether the (already-transformed) schema fits the provider's strict bounds. Returns the
 *  blocking notes; an empty array means it fits and the call can use constrained decoding. */
function fitsStrict(schema: SchemaNode, profile: ProviderSchemaProfile): AdaptNote[] {
  const notes: AdaptNote[] = [];
  const stats = collectStats(schema, profile.maxDepthCountStrategy ?? "all");
  if (stats.hasAny && profile.anyType !== "encode-json-string") {
    notes.push({ code: "any-not-representable", detail: "schema contains an untyped/any node the strict decoder can't represent" });
  }
  if (stats.hasRef && (profile.refs !== "native" || !profile.recursion)) {
    notes.push({ code: "recursion-unsupported", detail: "schema uses $ref but the provider doesn't accept it under strict mode" });
  }
  if (stats.rootUnion && !profile.rootUnion) {
    notes.push({ code: "root-union-unsupported", detail: "a union (oneOf/anyOf/allOf) at the root is not allowed under strict mode" });
  }
  if (profile.maxDepth !== undefined && stats.maxDepth > profile.maxDepth) {
    notes.push({ code: "depth-exceeded", detail: `nesting depth ${stats.maxDepth} exceeds the max ${profile.maxDepth}` });
  }
  const lim = profile.limits;
  if (lim?.maxProperties !== undefined && stats.propertyCount > lim.maxProperties) {
    notes.push({ code: "properties-exceeded", detail: `property count ${stats.propertyCount} exceeds the max ${lim.maxProperties}` });
  }
  if (lim?.maxEnumValues !== undefined && stats.enumCount > lim.maxEnumValues) {
    notes.push({ code: "enums-exceeded", detail: `enum value count ${stats.enumCount} exceeds the max ${lim.maxEnumValues}` });
  }
  if (lim?.maxNameEnumChars !== undefined && stats.nameEnumChars > lim.maxNameEnumChars) {
    notes.push({ code: "name-enum-chars-exceeded", detail: `combined name+enum chars ${stats.nameEnumChars} exceeds the max ${lim.maxNameEnumChars}` });
  }
  return notes;
}

// ---------------------------------------------------------------------------------------------------
// Reverse (post-processing the model's answer back to the ORIGINAL schema)
// ---------------------------------------------------------------------------------------------------

function reverse(value: unknown, schema: unknown, profile: ProviderSchemaProfile): unknown {
  if (value == null || !isPlainObject(schema)) return value;

  // 1. Union reconstruction (flatten mode): the model returned every variant's props at once — pick the
  //    matched variant by its discriminator(s) and strip the rest, then recurse into that variant.
  if (profile.unions === "flatten") {
    for (const kw of ["oneOf", "anyOf"] as const) {
      if (!Array.isArray(schema[kw])) continue;
      const variants = schema[kw] as SchemaNode[];
      const discriminators = findDiscriminators(variants);
      if (discriminators.length === 0 || Array.isArray(value) || !isPlainObject(value)) break;
      const obj = value;
      if (discriminators.some((d) => obj[d] === undefined)) break;
      const matched = variants.find((v) => {
        const props = isPlainObject(v.properties) ? v.properties : undefined;
        return props ? discriminators.every((d) => (props[d] as SchemaNode | undefined)?.const === obj[d]) : false;
      });
      if (!matched) break;
      const props = (isPlainObject(matched.properties) ? matched.properties : {}) as SchemaNode;
      const trimmed: Record<string, unknown> = {};
      for (const key of Object.keys(props)) {
        if (key in obj) trimmed[key] = reverse(obj[key], props[key], profile);
      }
      return trimmed;
    }
  }

  // 2. Any-decode: the original node was untyped and we sent a string — parse it back.
  if (isAnyNode(schema) && profile.anyType === "encode-json-string") {
    if (typeof value === "string") {
      try {
        return JSON.parse(value);
      } catch {
        return value; // best-effort: a non-JSON string stays as-is
      }
    }
    return value;
  }

  // 3. Arrays.
  if (Array.isArray(value) && isPlainObject(schema.items)) {
    return value.map((item) => reverse(item, schema.items, profile));
  }

  // 4. Objects: recurse, and drop nullable-optional `null`s we forced into existence.
  if (isPlainObject(value) && isPlainObject(schema.properties)) {
    const props = schema.properties;
    const required = new Set(Array.isArray(schema.required) ? (schema.required as string[]) : []);
    const dropNulls = profile.optionalSupport === "nullable";
    const result: Record<string, unknown> = {};
    for (const [key, v] of Object.entries(value)) {
      const childSchema = props[key];
      if (dropNulls && childSchema !== undefined && !required.has(key) && v === null) continue;
      result[key] = childSchema !== undefined ? reverse(v, childSchema, profile) : v;
    }
    return result;
  }

  return value;
}

// ---------------------------------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------------------------------

/** Transform a BARE schema for the strict path (root-wrap if needed → strictify a clone) and check fit.
 *  `strictify` mutates nested `properties`/`items` in place, so it always gets a CLONE. */
function strictAttempt(bare: SchemaNode, needsRootWrap: boolean, profile: ProviderSchemaProfile): { transformed: SchemaNode; blocking: AdaptNote[] } {
  const wrapped = needsRootWrap ? wrapRootArray(bare) : bare;
  const transformed = strictify(structuredClone(wrapped), profile, true);
  return { transformed, blocking: fitsStrict(transformed, profile) };
}

/**
 * Adapt an ORIGINAL JSON Schema for one provider transport. Returns the schema to send, the
 * strict-vs-advisory decision, a `postProcess` that reverses every lossy transform, and diagnostic
 * notes. The original is never mutated. May THROW when `maxDepthStrategy` is `"error"` (or `"flatten"`
 * and depth is irreducible) — an intentional hard failure for callers that require strict decoding.
 */
export function adaptSchema(original: SchemaNode, profile: ProviderSchemaProfile): AdaptResult {
  const stripped = stripMetaDeep(original) as SchemaNode;

  // Root-array wrap: a transport that requires an OBJECT root gets the array under a single wrapper
  // property, removed again by `postProcess` (`unwrapRootArray` — lenient, so a bare-array answer from
  // an upstream that ignored the hint still passes). Applied on BOTH the strict and advisory paths:
  // even unconstrained OpenAI-dialect JSON modes can only emit an object root.
  const needsRootWrap = isArrayRootNode(stripped) && !profile.rootArray;
  const base = needsRootWrap ? wrapRootArray(stripped) : stripped;
  const unwrap = needsRootWrap ? unwrapRootArray : (v: unknown) => v;
  const wrapNotes: AdaptNote[] = needsRootWrap
    ? [{ code: "root-array-wrapped", detail: `root array wrapped in { ${ROOT_ARRAY_KEY} } (object-root-only transport)` }]
    : [];

  // The two terminal builders, sharing the wrap concern. `strictOf` unwraps FIRST, then reverses against
  // the (unwrapped) pre-strictify schema — the wrap is the outermost transform. `advisoryOf` discards the
  // lossy strict transform and ships the meta-stripped (still wrapped) base as the hint.
  const strictOf = (outgoing: SchemaNode, reverseSchema: SchemaNode, extra: AdaptNote[]): AdaptResult => ({
    outgoing,
    enforce: "strict",
    postProcess: (v) => reverse(unwrap(v), reverseSchema, profile),
    notes: [...wrapNotes, ...extra],
  });
  const advisoryOf = (notes: AdaptNote[]): AdaptResult => ({ outgoing: base, enforce: "advisory", postProcess: unwrap, notes: [...notes, ...wrapNotes] });
  // TEXT floor: no response_format at all — the schema is described in the prompt (by the executor) and
  // the JSON is parsed out of the plain-text completion. Ships the meta-stripped (still root-wrapped) base
  // as the shape hint; `unwrap` is the only reversal (the model was never asked to flatten unions etc.).
  const textOf = (notes: AdaptNote[]): AdaptResult => ({ outgoing: base, enforce: "text", postProcess: unwrap, notes: [...notes, ...wrapNotes] });

  // Tier the transport supports (§5.1). `false` → plain text; `"object"` → json_object advisory (never
  // strict — the decoder isn't schema-bound); `"schema"` → attempt strict below.
  if (profile.supportsStructuredOutput === false) {
    return textOf([{ code: "no-structured-output", detail: `${profile.id}: no json mode — plain-text completion` }]);
  }
  if (profile.supportsStructuredOutput === "object") {
    return advisoryOf([{ code: "no-structured-output", detail: `${profile.id}: json_object mode (schema is advisory)` }]);
  }

  // Transform for what the EFFECTIVE transport can't handle natively, then check fit. A profile whose
  // SDK already cleans the schema declares those capabilities as supported, so this is a near-identity
  // for it. Fits → strict + reversible post-process.
  const { transformed, blocking } = strictAttempt(stripped, needsRootWrap, profile);
  if (blocking.length === 0) return strictOf(transformed, original, []);

  // Doesn't fit. If DEPTH is (one of) the reason(s), `maxDepthStrategy` decides what to do about it; any
  // OTHER blocker (an `{}` any node, root union, property overflow) is out of this strategy's scope and
  // always falls to advisory below.
  if (blocking.some((b) => b.code === "depth-exceeded")) {
    const strategy = profile.maxDepthStrategy ?? "adapt";
    if (strategy === "error") {
      throw new Error(`adaptSchema: schema exceeds ${profile.id} max nesting depth (${profile.maxDepth}) and maxDepthStrategy="error"`);
    }
    if (strategy === "strict") {
      // Override our (conservative) depth estimate: force strict at FULL depth and let the provider judge —
      // UNLESS a non-depth blocker remains (forcing strict can't make a closed grammar represent that).
      if (blocking.every((b) => b.code === "depth-exceeded")) {
        return strictOf(transformed, original, [{ code: "depth-strict-forced", detail: `depth over cap ${profile.maxDepth}; forced strict (maxDepthStrategy="strict") — provider is the arbiter` }]);
      }
      return advisoryOf(blocking);
    }
    if (strategy === "flatten" || strategy === "flatten-or-adapt") {
      // Lossless object-key flatten on the PRE-strictify schema (original optionality intact), then
      // re-strictify + re-check. Reverse composes as: unwrap → reverse(against the flat schema) → unflatten.
      const { flat, unflatten } = flattenForDepth(stripped);
      const flatAttempt = strictAttempt(flat, needsRootWrap, profile);
      if (flatAttempt.blocking.length === 0) {
        return {
          outgoing: flatAttempt.transformed,
          enforce: "strict",
          postProcess: (v) => unflatten(reverse(unwrap(v), flat, profile)),
          notes: [...wrapNotes, { code: "depth-flattened", detail: "object chains flattened to dotted keys to fit the strict depth cap" }],
        };
      }
      // Flattening didn't reach strict. `"flatten"` errors only when DEPTH is still the blocker (the
      // depth-specific promise it makes); a residual NON-depth blocker falls to advisory like everything else.
      if (strategy === "flatten" && flatAttempt.blocking.some((b) => b.code === "depth-exceeded")) {
        throw new Error(`adaptSchema: schema still exceeds ${profile.id} max nesting depth (${profile.maxDepth}) after flattening and maxDepthStrategy="flatten"`);
      }
      return advisoryOf(blocking);
    }
    // "adapt" (and the default) → advisory.
    return advisoryOf(blocking);
  }

  return advisoryOf(blocking);
}

// Exposed for focused unit tests; not part of the public adapter contract.
export const __internal = { strictify, fitsStrict, reverse, collectStats, mergeUnionVariants, stripMetaDeep, isAnyNode, wrapRootArray, unwrapRootArray };
