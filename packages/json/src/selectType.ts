/**
 * Compile-time type-resolver for `select`/`project` operations (API.md, "Schema templates, inference, and select-typing"; ported from
 * findmyprompt `src/engine/schema/selectType.ts`) — the one irreducible custom piece of the schema
 * layer. A `select` op carries a JSONPath written as a *data* path (so the same literal also drives
 * runtime selection over a value); at CHECK time there is no data, so we walk the input JSON Schema in
 * parallel to *compute* the output schema:
 *   - object segment `.prop`  → descend `properties[prop]`;
 *   - wildcard `[*]`          → require `array`, descend `items`, and wrap the final result back in
 *                               `array` once per wildcard crossed;
 *   - index `[n]`             → require `array`, descend `items` (a single element).
 *
 * Static-literal paths only. No JSONPath library does this schema→schema transform. Throws on an
 * unsupported path segment or a structural mismatch (e.g. `.prop` on a non-object).
 */
import type { JsonSchema, SchemaDocument } from "./json";
import { getOwn } from "./ownProps";

export type PathSegment = { kind: "prop"; name: string } | { kind: "wildcard" } | { kind: "index"; index: number };

/** Parse the supported JSONPath subset: `$`, `.prop` / `['prop']`, `[*]`, `[n]`. */
export function parseJsonPath(path: string): PathSegment[] {
  let rest = path.trim();
  if (rest.startsWith("$")) rest = rest.slice(1);
  const segments: PathSegment[] = [];
  const re = /^(?:\.([A-Za-z_][\w-]*)|\[\s*(\*|\d+|'[^']*'|"[^"]*")\s*\])/;
  while (rest.length > 0) {
    const m = re.exec(rest);
    if (!m) throw new Error(`parseJsonPath: unsupported segment at "${rest}"`);
    if (m[1] !== undefined) {
      segments.push({ kind: "prop", name: m[1] });
    } else {
      const token = m[2]!;
      if (token === "*") segments.push({ kind: "wildcard" });
      else if (/^\d+$/.test(token)) segments.push({ kind: "index", index: Number(token) });
      else segments.push({ kind: "prop", name: token.slice(1, -1) }); // quoted property
    }
    rest = rest.slice(m[0].length);
  }
  return segments;
}

/** Read a nested keyword value as a schema document (a non-object value is not a schema). */
function asDocument(v: unknown): SchemaDocument | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as SchemaDocument) : undefined;
}

/** Compute the output schema of applying `path` to a value of `inputSchema`. */
export function resolveSelectOutputSchema(inputSchema: SchemaDocument, path: string): JsonSchema {
  const segments = parseJsonPath(path);
  let current: SchemaDocument = inputSchema;
  let arrayWraps = 0;

  for (const seg of segments) {
    if (seg.kind === "prop") {
      // Own-property read: the segment name comes from an author's path string, so `$.__proto__` would
      // otherwise resolve `Object.prototype` and pass for a schema instead of failing the check.
      const props = asDocument(current.properties);
      const next = asDocument(props === undefined ? undefined : getOwn(props, seg.name));
      if (!next) throw new Error(`resolveSelectOutputSchema: property '${seg.name}' not in schema`);
      current = next;
    } else {
      // wildcard or index — require an array, descend into items
      const items = asDocument(current.items);
      if (!items) throw new Error("resolveSelectOutputSchema: indexed/wildcard segment on a non-array schema");
      current = items;
      if (seg.kind === "wildcard") arrayWraps++;
    }
  }

  // Each wildcard crossed re-wraps the leaf in an array.
  let result: JsonSchema = current;
  for (let i = 0; i < arrayWraps; i++) result = { type: "array", items: result };
  return result;
}
