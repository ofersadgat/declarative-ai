/**
 * Hashing, approval, and the freeze (SPEC §7.5.5).
 *
 * The threat model is a single user running their own code on their own machine. The risk is
 * therefore not privilege — it is **code changing without the user knowing**, which in a system where
 * an agent can write files includes code the user never wrote in the first place.
 *
 * ## One hash, three uses
 *
 * `moduleHash` is computed once per file and serves approval, the symbol-index cache, and the
 * transpile cache. A file is read and hashed once, and everything downstream keys on the answer.
 *
 * ## An unknown file is an unapproved file
 *
 * The property is deliberately stated in its strong form, because the weak one ("known files must
 * match") does not cover shadowing: a new file earlier on the search path changes which module a
 * specifier resolves to without modifying anything that already existed. Only "unknown means
 * unapproved" stops it.
 *
 * ## The freeze
 *
 * `freezeModules` resolves every reachable module, checks each hash against its approval, and stops
 * the run before anything executes if any file is unapproved or has changed. It is an ERROR and not a
 * prompt: a run is not the moment to be deciding what code to trust.
 *
 * What it returns is the **transpiled** output, which is what makes a frozen run actually frozen. A
 * stored hash can only detect drift and refuse; it cannot execute the version that was approved.
 * Storing the emitted code also removes the compiler from replay entirely, so a later toolchain
 * upgrade cannot change what a pinned run does.
 */
import { canonicalize, sha256Hex } from "@declarative-ai/exec";
import { prepareModules, type PrepareOptions } from "./moduleLoader.js";

export class IntegrityError extends Error {}

/**
 * The content hash of one user file — the single identity approval, indexing and emit all key on.
 *
 * Over the SOURCE rather than the emit, because what a person approves is what they can read. The
 * emit is derived from it and cached under it.
 */
export function moduleHash(source: string): string {
  return sha256Hex(source);
}

/**
 * True for a file under a `node_modules` directory, which is exempt from approval (SPEC §7.5.5).
 *
 * Keyed on the RESOLVED path and never on how the specifier was spelled, because the require path
 * (§7.5.4) lets a *bare* specifier reach a plain file in a search entry. A rule written as "bare
 * imports are exempt" would leave open exactly the hole approval closes.
 */
export function isVendored(file: string): boolean {
  return /(^|\/)node_modules(\/|$)/.test(file.replace(/\\/g, "/"));
}

/** Where a host keeps what the user has agreed to run. Read-only here; writing is the host's UI. */
export interface ApprovalStore {
  /** The approved content hash for this file, or `undefined` if it has never been approved. */
  approved(file: string): string | undefined;
}

/** An approval store over a plain map — what tests and an in-memory bundle want. */
export function approvalsOf(entries: Readonly<Record<string, string>>): ApprovalStore {
  return { approved: (file) => entries[file] };
}

/** A file awaiting a decision, with everything a diff needs. */
export interface PendingApproval {
  file: string;
  /** The hash as the file stands now. */
  hash: string;
  /** The source as it stands now — what a diff shows. */
  source: string;
  /**
   * The hash previously approved, when there was one.
   *
   * Its presence is the difference between the two questions a person is being asked. Absent is
   * "should this run at all"; present is "here is what changed" — and only the second can be
   * answered by looking at a diff, which is why the two are distinguished rather than merged into
   * one prompt.
   */
  previousHash?: string;
}

export interface DiscoverOptions extends Omit<PrepareOptions, "approved"> {
  approvals: ApprovalStore;
}

/**
 * What would need approving before `entries` could run.
 *
 * Runs the closure walk with NO approval gate, which is the only way to find out what a workflow
 * reaches — and is safe precisely because preparing does not execute anything (§7.5.4). The gate
 * applies at `freezeModules`, after a person has answered.
 */
export async function pendingApprovals(entries: readonly string[], options: DiscoverOptions): Promise<PendingApproval[]> {
  const prepared = await prepareModules(entries, { ...options, approved: undefined });
  const pending: PendingApproval[] = [];
  for (const file of prepared.emitted.keys()) {
    if (isVendored(file)) continue;
    const source = options.sources?.[file] ?? options.vfs.read(file);
    if (source === undefined) continue;
    const hash = moduleHash(source);
    const previousHash = options.approvals.approved(file);
    if (previousHash === hash) continue;
    pending.push({ file, hash, source, ...(previousHash !== undefined ? { previousHash } : {}) });
  }
  return pending;
}

/** A frozen definition's module half. */
export interface FrozenModules {
  /** Source content hash per file — what folds into the snapshot hash. */
  readonly hashes: ReadonlyMap<string, string>;
  /** Emitted CommonJS per file. What the snapshot directory holds, and what replay runs. */
  readonly emitted: ReadonlyMap<string, string>;
  /** The single value {@link foldModuleDigest} adds to a workflow's identity. */
  readonly digest: string;
}

/**
 * Check every reachable module against its approval and freeze the result.
 *
 * Throws before anything is executed if a file is unapproved or has changed since it was. Both
 * failures name every offending file rather than the first, because a person about to be asked for
 * approvals wants the list, not a sequence of one-at-a-time refusals.
 */
export async function freezeModules(entries: readonly string[], options: DiscoverOptions): Promise<FrozenModules> {
  const prepared = await prepareModules(entries, { ...options, approved: undefined });

  const hashes = new Map<string, string>();
  const unapproved: string[] = [];
  const changed: string[] = [];

  for (const file of [...prepared.emitted.keys()].sort()) {
    if (isVendored(file)) continue;
    const source = options.sources?.[file] ?? options.vfs.read(file);
    if (source === undefined) throw new IntegrityError(`'${file}' could not be read to verify it`);
    const hash = moduleHash(source);
    const approved = options.approvals.approved(file);
    if (approved === undefined) unapproved.push(file);
    else if (approved !== hash) changed.push(file);
    else hashes.set(file, hash);
  }

  if (unapproved.length > 0 || changed.length > 0) {
    const parts: string[] = [];
    if (unapproved.length > 0) parts.push(`never approved: ${unapproved.join(", ")}`);
    if (changed.length > 0) parts.push(`changed since approval: ${changed.join(", ")}`);
    throw new IntegrityError(`this workflow will not run — ${parts.join("; ")}`);
  }

  return { hashes, emitted: prepared.emitted, digest: moduleDigest(hashes) };
}

/**
 * One value standing for every module a workflow reaches.
 *
 * Over `(basename, hash)` pairs rather than absolute paths, and the choice matters both ways:
 *
 *  - **Not full paths**, because those are machine-specific (`/home/…` against `C:/Users/…`) and a
 *    snapshot that differs between two machines running the same workflow is not an identity.
 *  - **Not content alone**, because within one directory the winner of a symbol collision is decided
 *    by filename order (`moduleIndex`) — so a rename can change which code runs while every content
 *    hash stays the same. The basename is the machine-independent part that captures it.
 */
export function moduleDigest(hashes: ReadonlyMap<string, string>): string {
  const pairs = [...hashes]
    .map(([file, hash]) => [file.replace(/\\/g, "/").split("/").pop() ?? file, hash] as const)
    .sort((a, b) => (a[0] === b[0] ? (a[1] < b[1] ? -1 : 1) : a[0] < b[0] ? -1 : 1));
  return sha256Hex(canonicalize(pairs.map(([name, hash]) => [name, hash])));
}
