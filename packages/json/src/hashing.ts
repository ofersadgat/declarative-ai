import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import canonicalizeJCS from "canonicalize";
import type { Serializable, SerializableFields } from "./json";

/**
 * Content-addressing primitives, extracted from findmyprompt `src/engine/artifacts/`
 * (`hash.ts` + `canonicalize.ts`). Pure JS (no `node:crypto`) so hashing runs in any
 * runtime — Node, edge, and the Vercel Workflow runtime where Node modules are
 * forbidden. Digests are byte-identical to `node:crypto`'s.
 *
 * These live in `json` (API.md, "Hashing & identity") because canonical JSON serialization — RFC 8785 JCS
 * over `JsonValue` — is definitionally a json concern, and its consumers reach well past memoization:
 * the hw loader's snapshot hash, llm's schema cache keys, the ajv validator's inline-schema cache.
 * With hashing at the bottom, memoization is a few dozen dependency-free lines in `exec`.
 */

/**
 * Canonical JSON serialization — RFC 8785 (JCS): sorted object keys, normalized
 * numbers, no insignificant whitespace. Two values that are JSON-equal (modulo key
 * order) serialize identically, which is what makes content hashes stable and
 * `(operation, inputs)` memo keys correct.
 *
 * Throws on values JCS cannot serialize (`undefined`, functions, bigint, circular)
 * rather than silently hashing nothing — a non-serializable artifact is a bug.
 */
export function canonicalize<T extends Serializable | SerializableFields<T>>(value: T): string {
  assertCanonicalizable(value, "value", new Set<object>());
  const out = canonicalizeJCS(value as Parameters<typeof canonicalizeJCS>[0]);
  if (typeof out !== "string") {
    throw new Error("canonicalize: value is not JSON-serializable");
  }
  return out;
}

/**
 * The shape check the TYPE cannot make, run on the value itself.
 *
 * `SerializableFields<T>` admits any class whose DECLARED fields are serializable, while JCS reads only
 * OWN ENUMERABLE properties — so a class with `#private` fields and getters, an ordinary TS idiom,
 * canonicalizes to `{}` and EVERY instance hashes to sha256("{}"). Under a `(operation, inputs)` memo
 * key that is not a miss, it is a wrong CACHED result. `tsc` catches the nested case and reports
 * nothing at the top-level argument — the documented entry point — so the bound cannot be the check.
 *
 * Refused: a non-plain object (prototype neither `Object.prototype` nor `null`) that does not declare
 * `toJSON()`; functions/bigint/symbols, which JCS emits as literal `undefined` (invalid JSON) or drops;
 * and reference cycles, which JCS recurses into until the stack dies. `undefined` at a MEMBER position
 * is allowed — dropping it is exactly what `Serializable` documents — while a bare `undefined` value
 * still fails the string check above.
 */
function assertCanonicalizable(value: unknown, path: string, seen: Set<object>): void {
  if (value === null) return;
  const kind = typeof value;
  if (kind === "function" || kind === "bigint" || kind === "symbol") {
    throw new Error(`canonicalize: ${kind} at ${path} is not JSON-serializable`);
  }
  if (kind !== "object") return;
  const obj = value as object;
  if (seen.has(obj)) throw new Error(`canonicalize: circular reference at ${path}`);
  seen.add(obj);
  const toJSON = (obj as { toJSON?: unknown }).toJSON;
  if (typeof toJSON === "function") {
    // JCS serializes what `toJSON()` RETURNS, so that is the value that has to be serializable. This is
    // how a domain class opts IN to hashing — and the same contract `Jsonify<T>` projects through.
    assertCanonicalizable((toJSON as () => unknown).call(obj), `${path}.toJSON()`, seen);
  } else if (Array.isArray(obj)) {
    obj.forEach((v, i) => assertCanonicalizable(v, `${path}[${i}]`, seen));
  } else {
    const proto: unknown = Object.getPrototypeOf(obj);
    if (proto !== Object.prototype && proto !== null) {
      throw new Error(
        `canonicalize: ${(obj as { constructor?: { name?: string } }).constructor?.name || "non-plain object"} at ${path} is not a plain JSON object — ` +
          "only its own enumerable properties would be hashed (declare toJSON(), or hash a plain projection)",
      );
    }
    for (const [k, v] of Object.entries(obj)) assertCanonicalizable(v, `${path}.${k}`, seen);
  }
  seen.delete(obj); // path-scoped: an ALIASED node is fine, a node inside itself is not
}

const utf8 = new TextEncoder();

/** Pure-JS sha256 rendered as lowercase hex. */
export function sha256Hex(payload: string): string {
  return bytesToHex(sha256(utf8.encode(payload)));
}

/** Hash a JSON value's canonical form. */
export function hashCanonical<T extends Serializable | SerializableFields<T>>(value: T): string {
  return sha256Hex(canonicalize(value));
}
