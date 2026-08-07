/**
 * Making a NAMED agent binary launchable — the Windows half, which is where a bare command name stops
 * being launchable at all.
 *
 * Neither delegated transport spawns through a shell, and both are right not to: their argv carry a URL
 * with a bearer token, a JSON document, and whatever a project configured, none of which survive
 * `cmd.exe` quoting intact. But without a shell, Windows gives you nothing for free:
 *
 *  - **A bare name is not resolved.** `spawn("claude")` with `shell: false` does no PATH/PATHEXT lookup,
 *    so it fails as ENOENT even with `claude.exe` sitting on the PATH.
 *  - **A launcher script cannot be spawned at all.** An npm-installed CLI on Windows is a `.cmd` shim,
 *    and Node refuses to spawn one with `shell: false` — EINVAL, deliberately, since CVE-2024-27980
 *    (Node ≥ 20.12).
 *
 * So a caller who names a binary gets it RESOLVED here, to something that can actually be launched: a
 * real executable if one is on the PATH, and otherwise the package entry the shim itself delegates to.
 * That second step is a lookup, not a guess — the npm layout puts the entry at a fixed place relative to
 * the shim, which is exactly what the shim's own body names:
 *
 * ```
 * C:\Users\me\AppData\Roaming\npm\claude.cmd
 *   → "%dp0%\node_modules\@anthropic-ai\claude-code\bin\claude.exe"
 * ```
 *
 * Pure string work behind an injected {@link BinaryDeps.exists}, with no `node:path` and no `node:fs`:
 * this is win32-only by definition, so the platform's separator is known, and staying dependency-free
 * keeps the module edge-importable and the resolution testable against a fake filesystem — which is the
 * only way to test it at all from a machine that is not Windows.
 */

/** The filesystem and environment facts {@link resolveAgentBinary} needs, injected so it stays pure. */
export interface BinaryDeps {
  /** Does this absolute path name a file? The ONE effect this resolution has. */
  exists: (path: string) => boolean;
  /** Defaults to `win32` — the only platform this does anything on, so a caller that omits it is
   *  asking for the resolution rather than accidentally skipping it. */
  platform?: string;
  /** `PATH` entries to search when the command names no directory. */
  pathDirs?: readonly string[];
  /** Extensions Windows can spawn directly, most-preferred first. */
  binaryExtensions?: readonly string[];
  /** Extensions that name a LAUNCHER SCRIPT rather than a program — spawnable only through a shell. */
  launcherExtensions?: readonly string[];
}

/** What the resolution produced, and anything the caller should be told about it. */
export interface ResolvedBinary {
  /** What to launch. The caller's own string when nothing better could be found. */
  path: string;
  /** Why the result is not what the caller probably wanted. Absent on a clean resolution — a warning
   *  here is the difference between "this will fail with EINVAL" and an unexplained ENOENT. */
  warning?: string;
}

/** Where an npm-installed `claude` keeps its real entry, relative to the shim's own directory. The
 *  `.exe` is the current layout; `cli.js` is what older versions shipped, and the SDK still knows how to
 *  run one (it picks a JS runtime itself). Ordered most-preferred first. */
const PACKAGE_ENTRIES = ["node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe", "node_modules\\@anthropic-ai\\claude-code\\cli.js"] as const;

const DEFAULT_BINARY_EXTENSIONS = [".exe", ".com"] as const;
const DEFAULT_LAUNCHER_EXTENSIONS = [".cmd", ".bat", ".ps1"] as const;

/** The directory part of a path, `.` when there is none. Win32 separators only, deliberately. */
function dirOf(path: string): string {
  const cut = Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/"));
  return cut >= 0 ? path.slice(0, cut) : ".";
}

/**
 * Resolve a command NAME to something Windows can spawn without a shell.
 *
 * Non-win32 returns the command untouched: POSIX `spawn` resolves the PATH itself and there are no
 * launcher scripts to follow, so there is nothing here to do and doing it anyway would only be a way to
 * get it wrong.
 *
 * The order is what makes it correct rather than merely working:
 *
 *  1. **A real executable wins.** An `.exe` on the PATH is spawnable as it stands, and the native
 *     installer puts one there — so the common case costs one `exists` and changes nothing.
 *  2. **A launcher is FOLLOWED, never returned.** Handing back `claude.cmd` would trade an ENOENT for
 *     an EINVAL, which is strictly less informative.
 *  3. **Nothing found ⇒ the caller's own string, with a warning.** A spawn that fails naming the
 *     command the caller wrote is diagnosable; one that fails naming a path we invented is not.
 */
export function resolveAgentBinary(command: string, deps: BinaryDeps): ResolvedBinary {
  if ((deps.platform ?? "win32") !== "win32") return { path: command };
  const binary = deps.binaryExtensions ?? DEFAULT_BINARY_EXTENSIONS;
  const launchers = deps.launcherExtensions ?? DEFAULT_LAUNCHER_EXTENSIONS;
  const named = command.includes("/") || command.includes("\\");
  // A command naming a directory is searched where it points; a bare name is searched along the PATH.
  const bases = named ? [command] : (deps.pathDirs ?? []).map((dir) => `${dir}\\${command}`);
  const lower = command.toLowerCase();

  // Already spelled with an extension: take it at its word rather than appending a second one.
  const suffixed = [...binary, ...launchers].find((ext) => lower.endsWith(ext));
  const candidates = suffixed !== undefined ? bases : [...binary, ...launchers].flatMap((ext) => bases.map((base) => `${base}${ext}`));

  const hit = candidates.find((candidate) => deps.exists(candidate));
  if (hit === undefined) {
    return {
      path: command,
      warning:
        `'${command}' was not found on the PATH as an executable or as an npm launcher script. ` +
        `Spawning it without a shell will fail; pass an absolute \`binaryPath\` if it lives somewhere unusual.`,
    };
  }
  if (!launchers.some((ext) => hit.toLowerCase().endsWith(ext))) return { path: hit };

  // A launcher. Follow it to the package entry it delegates to — same directory, fixed npm layout.
  const dir = dirOf(hit);
  const entry = PACKAGE_ENTRIES.map((relative) => `${dir}\\${relative}`).find((candidate) => deps.exists(candidate));
  if (entry !== undefined) return { path: entry };
  return {
    path: command,
    warning:
      `'${command}' resolved to the launcher script '${hit}', which cannot be spawned without a shell (EINVAL), ` +
      `and no @anthropic-ai/claude-code package entry was found beside it. Passing '${command}' through unchanged.`,
  };
}

/**
 * Read the resolution facts off the real process, lazily.
 *
 * `node:fs` is imported here and nowhere else in this module, which is what keeps {@link resolveAgentBinary}
 * itself edge-importable and testable — and what lets a host with no filesystem inject its own `exists`
 * rather than being refused.
 */
export async function defaultBinaryDeps(): Promise<BinaryDeps> {
  const fs = await import("node:fs");
  const separator = process.platform === "win32" ? ";" : ":";
  return {
    platform: process.platform,
    pathDirs: (process.env["PATH"] ?? "").split(separator).filter((dir) => dir.length > 0),
    exists: (path) => fs.existsSync(path),
  };
}
