/**
 * The manifest must name every workspace package this one imports — including TYPE-ONLY imports.
 *
 * `runtime.ts` imported `Approver` from `@declarative-ai/permissions` without the dependency being
 * declared. It costs nothing at runtime (the import is erased) and nothing in this repo (a workspace
 * hoists everything into one `node_modules`), which is exactly why it survives review: it only breaks
 * on a STANDALONE install, where `tsc` cannot resolve the type and the package fails to build for the
 * one consumer who did not also install the rest of the monorepo.
 */
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

const manifest = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8")) as {
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
};

/** Every `@declarative-ai/*` specifier any source file names, `import type` included. */
function importedWorkspacePackages(dir: string): Set<string> {
  const found = new Set<string>();
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      for (const p of importedWorkspacePackages(path)) found.add(p);
    } else if (entry.name.endsWith(".ts")) {
      for (const m of readFileSync(path, "utf8").matchAll(/from\s+"(@declarative-ai\/[^"]+)"/g)) found.add(m[1]!);
    }
  }
  return found;
}

describe("package.json declares what the sources import", () => {
  it("names every workspace package reached from src/, type-only ones included", () => {
    const declared = new Set([...Object.keys(manifest.dependencies ?? {}), ...Object.keys(manifest.peerDependencies ?? {})]);
    const undeclared = [...importedWorkspacePackages(join(packageRoot, "src"))].filter((p) => !declared.has(p));
    expect(undeclared).toEqual([]);
  });
});
