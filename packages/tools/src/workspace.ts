/**
 * Workspace path resolution for fs-backed tools. A tool receives the current op's `Workspace` via
 * `ctx.workspace` (a Session-owned resource, DESIGN §5.1, "Sessions: the run-scoped resource bundle") and resolves every path
 * input relative to its `root`, refusing to escape it (SPEC §7.2 — "may not access files outside the
 * project").
 */
import path from "node:path";
import type { ExecServices, Workspace } from "@declarative-ai/exec";


/** The workspace for the current op, or a clear error when a workspace tool runs without one configured. */
export function requireWorkspace(ctx: ExecServices): Workspace {
  if (!ctx.workspace) throw new Error("this tool needs a workspace, but none is configured (ExecServices.workspace)");
  return ctx.workspace;
}

/**
 * Resolve a workspace-relative path to an absolute one, THROWING if it escapes the root (via `..`, an
 * absolute path, or a symlink-style prefix trick). The comparison is prefix-based on the normalized root
 * plus a path separator, so `/repo-evil` never counts as inside `/repo`.
 */
export function resolveInWorkspace(root: string, rel: string): string {
  const normRoot = path.resolve(root);
  const abs = path.resolve(normRoot, rel);
  if (abs !== normRoot && !abs.startsWith(normRoot + path.sep)) {
    throw new Error(`path '${rel}' escapes the workspace root`);
  }
  return abs;
}
