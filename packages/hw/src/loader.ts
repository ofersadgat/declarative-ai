/**
 * Bundle loading + snapshot hashing (SPEC §2.4, §12).
 *
 * A bundle is a map of state-ID → state-file JSON. State IDs are file paths relative
 * to the workflow root without suffix; `id` inside a file may be omitted and is
 * derived from the path (a present-but-mismatched `id` is a load error).
 *
 * The snapshot hash is the workflow-version identity (SPEC §12, JaiRA DESIGN §5.3):
 * `sha256(canonicalize(sorted [(stateId, contentHash(stateFile))]))` over the
 * transitive closure of states reachable from the root. Two bundles with the same
 * files hash identically regardless of load order or extra unreachable files.
 */
import { hashCanonical, sha256Hex, canonicalize } from "@ai-exec/core";
import type { ChildDecl, StateDef, WorkflowBundle } from "./format";

export class WorkflowLoadError extends Error {
  constructor(
    message: string,
    readonly stateId?: string,
  ) {
    super(stateId ? `${stateId}: ${message}` : message);
    this.name = "WorkflowLoadError";
  }
}

/** Strip a state-file name to its state ID: forward slashes, no extension. */
export function stateIdFromPath(relPath: string): string {
  return relPath.replace(/\\/g, "/").replace(/\.state\.json$|\.json$/i, "");
}

/**
 * Load a bundle from raw file contents (`stateId or relative path` → parsed JSON).
 * Restricts the bundle to the transitive closure reachable from `rootId` so the
 * snapshot hash never varies with unrelated files lying around the workflow dir.
 */
export function loadBundle(files: Record<string, unknown>, rootId: string): WorkflowBundle {
  const all = new Map<string, StateDef>();
  for (const [key, raw] of Object.entries(files)) {
    const id = stateIdFromPath(key);
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
      throw new WorkflowLoadError("state file is not a JSON object", id);
    }
    const def = raw as StateDef;
    if (def.id !== undefined && def.id !== id) {
      throw new WorkflowLoadError(`declared id '${def.id}' does not match path-derived id '${id}'`, id);
    }
    all.set(id, { ...def, id });
  }
  if (!all.has(rootId)) {
    throw new WorkflowLoadError(`root state '${rootId}' not found in bundle`);
  }
  // Transitive closure from the root.
  const states: Record<string, StateDef> = {};
  const queue = [rootId];
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (states[id]) continue;
    const def = all.get(id);
    if (!def) {
      // Missing children are a VALIDATION error (with context), not a load error —
      // keep loading so the validator can report all of them at once.
      continue;
    }
    states[id] = def;
    for (const child of Object.values(def.children ?? {})) {
      queue.push((child as ChildDecl).state);
    }
  }
  return { rootId, states };
}

/**
 * The snapshot hash — the bundle's version identity (SPEC §12). This value is the
 * `definitionHash` of a `hierarchical-workflow` ExecutionSpec and the memo-key
 * component for workflow executions (DESIGN §3.4).
 */
export function snapshotHash(bundle: WorkflowBundle): string {
  const entries = Object.entries(bundle.states)
    .map(([id, def]) => [id, hashCanonical(stripDerivedId(id, def))] as const)
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return sha256Hex(canonicalize({ rootId: bundle.rootId, states: entries }));
}

/** Hash the file as authored: a derived (previously absent) `id` must not change the hash. */
function stripDerivedId(id: string, def: StateDef): unknown {
  void id;
  const { id: _id, ...rest } = def;
  return rest;
}

/**
 * Node-only convenience: load every `*.json` under a directory as a bundle rooted at
 * `rootId`. Uses dynamic imports so the module stays edge-safe when unused.
 */
export async function loadBundleFromDir(dir: string, rootId: string): Promise<WorkflowBundle> {
  const { readdir, readFile } = await import("node:fs/promises");
  const { join, relative } = await import("node:path");
  const files: Record<string, unknown> = {};
  const walk = async (d: string): Promise<void> => {
    for (const entry of await readdir(d, { withFileTypes: true })) {
      const full = join(d, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) {
        const rel = relative(dir, full);
        files[rel] = JSON.parse(await readFile(full, "utf8")) as unknown;
      }
    }
  };
  await walk(dir);
  return loadBundle(files, rootId);
}
