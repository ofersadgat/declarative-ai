/**
 * VERSIONS of a called state, side by side in one run (JaiRA decision 0029 §3).
 *
 * A run keeps the workflow it started with. When a state a script CALLS has changed since — a lesson
 * edited it — the engine may hand the next call to the new version instead: the new version's states
 * join the run's bundle under VERSIONED ids (`chat/turn@v1a2b3c4d5e6f`, and `…@v…/call_0` below it), so
 * the old version's states are untouched, every instance already run under them replays as it ran, and
 * the switch is an ordinary child entry the journal records (`version.adopted`, then
 * `instance.entered` naming the versioned id). A replay loads the same versions back into the bundle
 * first and re-attaches what was entered — nothing is decided twice.
 *
 * A version's identity is the content of its subtree with its own id taken out, so the same states under
 * a versioned name hash as they did under the plain one: a call never "adopts" what it is already running.
 */
import { createHash } from "node:crypto";
import type { LoadedState, StateDef, WorkflowBundle } from "./format.js";

/** `chat/turn@v1a2b3c4d5e6` → `chat/turn`; an unversioned id is its own base. */
export function baseStateIdOf(id: string): string {
  return id.replace(/@v[0-9a-f]+(?=\/|$)/, "");
}

/** The version a versioned id names (`chat/turn@v1a2b…` → `1a2b…`), or `undefined` for a plain id. */
export function versionOfStateId(id: string): string | undefined {
  return /@v([0-9a-f]+)(?=\/|$)/.exec(id)?.[1];
}

/** The id a version of `base` runs under. */
export function versionedStateId(base: string, version: string): string {
  return `${baseStateIdOf(base)}@v${version}`;
}

/** Every id of `bundle` in the subtree rooted at `root`: the root and everything under `root/`. */
export function subtreeIds(bundle: Pick<WorkflowBundle, "states">, root: string): string[] {
  return Object.keys(bundle.states)
    .filter((id) => id === root || id.startsWith(`${root}/`))
    .sort();
}

/** JSON with sorted keys and nothing a JSON document cannot hold — a stable text to hash. */
function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map((v) => (typeof v === "function" || v === undefined ? "null" : stableJson(v))).join(",")}]`;
  const entries = Object.keys(value as Record<string, unknown>)
    .sort()
    .filter((k) => {
      const v = (value as Record<string, unknown>)[k];
      return v !== undefined && typeof v !== "function";
    })
    .map((k) => `${JSON.stringify(k)}:${stableJson((value as Record<string, unknown>)[k])}`);
  return `{${entries.join(",")}}`;
}

/** Replace, anywhere in a state, a string that names an id of the subtree — and only those. */
function renamed<T>(value: T, ids: ReadonlyMap<string, string>): T {
  if (typeof value === "string") return (ids.get(value) ?? value) as T;
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((v) => renamed(v, ids)) as T;
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return value;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = renamed(v, ids);
  return out as T;
}

/**
 * The content hash of the subtree at `root` — its ids written relative to `root`, so a version's hash
 * does not depend on the name it runs under. 12 hex digits: a version's name, not a security boundary.
 */
export function versionHashOf(bundle: Pick<WorkflowBundle, "states">, root: string): string {
  const ids = subtreeIds(bundle, root);
  const relative = new Map(ids.map((id) => [id, `~${id.slice(root.length)}`]));
  // The LOADED states, the form a snapshot hashes — so a pinned version and the same files loaded fresh
  // agree. A running engine annotates the states it runs, so it hashes the states it calls BEFORE it
  // runs anything (`WorkflowEngine`), and a versioned id carries its hash in its name.
  const text = ids.map((id) => `${relative.get(id)}=${stableJson(renamed(bundle.states[id], relative))}`).join("\n");
  return createHash("sha256").update(text).digest("hex").slice(0, 12);
}

/**
 * A version's states under their versioned ids: the subtree at `bundle`'s `base`, every id in it — and
 * every reference to one (a child's `state`, a scoped name's `$in`) — moved under `versioned`.
 */
export function versionedStates(bundle: Pick<WorkflowBundle, "states">, base: string, versioned: string): Record<string, LoadedState> {
  const ids = subtreeIds(bundle, base);
  const map = new Map(ids.map((id) => [id, `${versioned}${id.slice(base.length)}`]));
  const out: Record<string, LoadedState> = {};
  for (const id of ids) out[map.get(id)!] = renamed(bundle.states[id]!, map);
  return out;
}

/** The same move for the states as authored, where the bundle kept them — what a version is named by. */
export function versionedSource(bundle: Pick<WorkflowBundle, "states" | "source">, base: string, versioned: string): Record<string, StateDef> {
  if (bundle.source === undefined) return {};
  const ids = subtreeIds(bundle, base);
  const map = new Map(ids.map((id) => [id, `${versioned}${id.slice(base.length)}`]));
  const out: Record<string, StateDef> = {};
  for (const id of ids) if (bundle.source[id] !== undefined) out[map.get(id)!] = renamed(bundle.source[id]!, map);
  return out;
}

/** What a run adopted: which state, under which versioned id, from which bundle (a host's snapshot). */
export interface AdoptedVersion {
  base: string;
  versioned: string;
  bundle: Pick<WorkflowBundle, "states" | "source">;
}

/** A bundle with versions a run adopted laid in beside its own states — what a replay must load. */
export function withVersions<B extends Pick<WorkflowBundle, "states" | "source">>(bundle: B, adopted: readonly AdoptedVersion[]): B {
  if (adopted.length === 0) return bundle;
  const states = { ...bundle.states };
  const source = bundle.source !== undefined ? { ...bundle.source } : undefined;
  for (const a of adopted) {
    Object.assign(states, versionedStates(a.bundle, baseStateIdOf(a.base), a.versioned));
    if (source !== undefined) Object.assign(source, versionedSource(a.bundle, baseStateIdOf(a.base), a.versioned));
  }
  return { ...bundle, states, ...(source !== undefined ? { source } : {}) };
}

/** A newer version a host offers for a called state — its bundle, rooted at that state's id. */
export interface OfferedVersion {
  bundle: WorkflowBundle;
  /** The host's name for it (a snapshot hash), recorded so a replay can load it back. */
  snapshot?: string;
}

/**
 * The host's answer to "is there a newer version of this called state than the one running?" —
 * `current` is the running version's hash ({@link versionHashOf}). `undefined` ⇒ keep running it.
 */
export type VersionSource = (stateId: string, current: string) => OfferedVersion | undefined | Promise<OfferedVersion | undefined>;
