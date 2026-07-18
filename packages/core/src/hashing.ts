import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import canonicalizeJCS from "canonicalize";

/**
 * Content-addressing primitives, extracted from findmyprompt `src/engine/artifacts/`
 * (`hash.ts` + `canonicalize.ts`). Pure JS (no `node:crypto`) so hashing runs in any
 * runtime — Node, edge, and the Vercel Workflow runtime where Node modules are
 * forbidden. Digests are byte-identical to `node:crypto`'s.
 */

/**
 * Canonical JSON serialization — RFC 8785 (JCS): sorted object keys, normalized
 * numbers, no insignificant whitespace. Two values that are JSON-equal (modulo key
 * order) serialize identically, which is what makes content hashes stable and
 * `(definition, inputs)` memo keys correct.
 *
 * Throws on values JCS cannot serialize (`undefined`, functions, bigint, circular)
 * rather than silently hashing nothing — a non-serializable artifact is a bug.
 */
export function canonicalize(value: unknown): string {
  const out = canonicalizeJCS(value as Parameters<typeof canonicalizeJCS>[0]);
  if (typeof out !== "string") {
    throw new Error("canonicalize: value is not JSON-serializable");
  }
  return out;
}

const utf8 = new TextEncoder();

/** Pure-JS sha256 rendered as lowercase hex. */
export function sha256Hex(payload: string): string {
  return bytesToHex(sha256(utf8.encode(payload)));
}

/** Hash a JSON value's canonical form. */
export function hashCanonical(value: unknown): string {
  return sha256Hex(canonicalize(value));
}

/**
 * The canonical memo key for one unit execution (DESIGN §3.4):
 *
 *   memoKey = sha256(canonicalize({ kind, definitionHash, inputs, workspaceTreeHash? }))
 *
 * `inputs` are carried as an object — JCS sorts keys, so the key is invariant to the
 * order callers assemble inputs in. `workspaceTreeHash` is REQUIRED for units whose
 * executor declares `mutatesWorkspace` (side-effecting runs are only memoizable
 * against a pinned workspace snapshot); it must be omitted for pure units.
 * Nondeterminism (draw indices, retry scopes) is the caller's concern via unhashed
 * scope tokens — this key deliberately has no place for them.
 */
export function memoKey(params: {
  kind: string;
  definitionHash: string;
  inputs: Record<string, unknown>;
  workspaceTreeHash?: string;
}): string {
  const { kind, definitionHash, inputs, workspaceTreeHash } = params;
  return sha256Hex(
    canonicalize({
      kind,
      definitionHash,
      inputs,
      ...(workspaceTreeHash !== undefined ? { workspaceTreeHash } : {}),
    }),
  );
}
