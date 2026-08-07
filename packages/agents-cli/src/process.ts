/**
 * The child-process SEAM every CLI-driven agent adapter runs through.
 *
 * It was private to `cliQuery.ts` while `claude` was the only CLI. It is shared now — `codexQuery.ts`
 * drives a different binary with a different protocol but the same process discipline — and JaiRA had
 * already been forced to clone it once (its `observedSpawn`, which wraps a spawn in job tracking).
 * Three copies of the two subtleties below is two too many.
 *
 * The two details that are copied deliberately and must not be "tidied":
 *
 *  - **stderr is IGNORED, not piped.** An unread pipe fills at ~64 KB and the child then blocks
 *    forever on write, so stdout stops and `exit` never settles. A CLI's diagnostics are not our
 *    channel — the exit code and its terminal message are.
 *  - **an `error` listener is attached.** A `ChildProcess` `'error'` event with no listener THROWS,
 *    which would take the host process down — and ENOENT on a missing binary is the likeliest
 *    first-run outcome. Capture it and let it surface through the exit code instead.
 */

/** A running child process, reduced to what an adapter needs. */
export interface AgentProcess {
  /** Newline-delimited JSON messages the agent wrote to stdout. */
  lines: AsyncIterable<string>;
  /** Terminate the process (wired to the caller's abort signal). */
  kill(): void;
  /** Resolves with the exit code once the process ends. */
  exit: Promise<number>;
  /**
   * Why the process could not be LAUNCHED, when that is what went wrong — read after `exit` settles.
   *
   * A failed spawn has no exit code, so it comes back as the sentinel `-1`, and `-1` means nothing to
   * anyone reading a failure. The real error says `spawn claude ENOENT`, which is the difference
   * between "the agent is not installed" and "the agent crashed". It was being captured and discarded.
   *
   * Optional so a fake process need not model it: absent ⇒ the exit code is the whole story.
   */
  launchFailure?: () => Error | undefined;
}

/** How a process is launched. `stdin` is the second way to hand an agent its instruction. */
export interface SpawnOptions {
  cwd?: string;
  /** The environment the child runs under. Absent ⇒ it inherits this process's, which is what a CLI
   *  needs by default (PATH, HOME, and whatever credential the binary reads). */
  env?: NodeJS.ProcessEnv;
  /**
   * Written to the child's stdin, which is then CLOSED.
   *
   * Present for the agents whose prompt does not belong in argv. A replayed conversation can run to
   * tens of kilobytes, and Windows caps a command line at ~32 KB — so an adapter that renders a
   * transcript into its prompt cannot pass it as an argument at all. Absent ⇒ stdin is `ignore`d,
   * which is what a CLI reading its prompt from argv wants (an inherited stdin would let it block
   * waiting for input that never comes).
   */
  stdin?: string;
}

/** The injectable process seam: launch the agent CLI with these argv and return its stream. */
export type SpawnProcess = (argv: string[], opts: SpawnOptions) => AgentProcess;

/** The filesystem facts {@link resolveProgram} needs, injected so it stays pure and testable. */
export interface ProgramDeps {
  exists: (path: string) => boolean;
  readText: (path: string) => string | undefined;
  platform?: string;
  /** `PATH` entries to search when the command names no directory. */
  pathDirs?: readonly string[];
  /** Extensions Windows can spawn directly, most-preferred first. */
  binaryExtensions?: readonly string[];
  /** The interpreter a resolved shim is run with. */
  node?: string;
}

/** The JS entry an npm `.cmd` shim delegates to, relative to the shim's own directory. */
const SHIM_ENTRY = /"%dp0%[\\/]([^"]+\.[cm]?js)"/i;

/** A JS file is a program only to an interpreter — Windows cannot spawn one directly. */
const JS_ENTRY = /\.[cm]?js$/i;

/**
 * Make a command NAME spawnable on Windows, without a shell.
 *
 * Node refuses to spawn a `.cmd`/`.bat` with `shell: false` — EINVAL, deliberately, since
 * CVE-2024-27980 — and `shell: true` is not an option here: these argv carry a URL with a bearer
 * token, a TOML inline table, and whatever a project configured, none of which survive `cmd.exe`
 * quoting intact. But an npm-installed CLI on Windows *is* a `.cmd` shim. `codex` is one; so is
 * `claude` when it comes from npm rather than from a native installer. Without this, spawning either
 * fails with an unexplained EINVAL/ENOENT.
 *
 * The shim is a stable, documented format whose last line names the JS entry it runs, so resolving it
 * to `node <entry>` is a lookup rather than a guess — and every argument stays literal, which is the
 * property that made "no shell" worth keeping.
 *
 * Two details are load-bearing:
 *
 *  - **A real executable wins**, and is returned as the caller WROTE it, because Windows appends
 *    PATHEXT itself. So a command backed by an `.exe` is untouched.
 *  - **An extensionless file is not "found".** npm installs a Git-Bash shell script beside every
 *    shim, and Windows cannot execute one — treating it as a hit is exactly how `codex` resolved to
 *    something unspawnable.
 *
 * Pure string work, no `node:path`: this is win32-only by definition, and staying dependency-free
 * keeps the module edge-importable and the function trivially testable.
 */
export function resolveProgram(command: string, deps: ProgramDeps): { file: string; prefix: string[] } {
  if ((deps.platform ?? "win32") !== "win32") return { file: command, prefix: [] };
  const binary = deps.binaryExtensions ?? [".exe", ".com"];
  const named = command.includes("/") || command.includes("\\");
  const dirs = named ? [""] : (deps.pathDirs ?? []);
  const candidates = (ext: string): string[] => dirs.map((dir) => (dir === "" ? `${command}${ext}` : `${dir}\\${command}${ext}`));

  const lower = command.toLowerCase();
  // A JS entry — what `resolveAgentBinary` falls back to for an older `@anthropic-ai/claude-code`, and
  // what a caller may pin directly. It is not a program: run it under the interpreter, exactly as a
  // resolved shim is, rather than handing Windows a file it cannot execute.
  if (JS_ENTRY.test(lower)) return { file: deps.node ?? "node", prefix: [command] };
  if (binary.some((ext) => lower.endsWith(ext))) return { file: command, prefix: [] };
  if (binary.some((ext) => candidates(ext).some((candidate) => deps.exists(candidate)))) return { file: command, prefix: [] };

  for (const candidate of [...candidates(".cmd"), ...candidates(".bat")]) {
    const entry = deps.readText(candidate)?.match(SHIM_ENTRY)?.[1];
    if (entry === undefined) continue;
    const cut = Math.max(candidate.lastIndexOf("\\"), candidate.lastIndexOf("/"));
    const dir = cut >= 0 ? candidate.slice(0, cut) : ".";
    return { file: deps.node ?? "node", prefix: [`${dir}\\${entry}`] };
  }
  // Nothing found: hand the name back and let the spawn fail, which at least names the command.
  return { file: command, prefix: [] };
}

/** The default `node:child_process` spawn, imported lazily so the module stays edge-importable. */
export async function defaultSpawn(): Promise<SpawnProcess> {
  const { spawn } = await import("node:child_process");
  const readline = await import("node:readline");
  const fs = await import("node:fs");
  // Resolved per launch rather than once: a host may install an agent mid-session, and this costs a
  // stat per spawn against a process that is about to take seconds.
  const program = (command: string): { file: string; prefix: string[] } =>
    resolveProgram(command, {
      platform: process.platform,
      pathDirs: (process.env["PATH"] ?? "").split(";").filter((d) => d.length > 0),
      node: process.execPath,
      exists: (p) => fs.existsSync(p),
      readText: (p) => (fs.existsSync(p) ? fs.readFileSync(p, "utf8") : undefined),
    });
  return (argv, opts) => {
    const [command, ...args] = argv;
    const { file, prefix } = program(command!);
    const child = spawn(file, [...prefix, ...args], {
      cwd: opts.cwd,
      // Spread only when the caller supplied one: `env: undefined` is what `child_process` reads as
      // "inherit", but stating it explicitly invites a later `{...opts.env}` that hands the child an
      // EMPTY environment and strips its PATH and credentials.
      ...(opts.env !== undefined ? { env: opts.env } : {}),
      stdio: [opts.stdin === undefined ? "ignore" : "pipe", "pipe", "ignore"],
      windowsHide: true,
    });
    const lines = readline.createInterface({ input: child.stdout!, crlfDelay: Infinity });

    let spawnError: Error | undefined;
    child.on("error", (e: Error) => {
      spawnError = e;
      lines.close();
    });

    if (opts.stdin !== undefined) {
      // An agent that fails to start, or that answers before reading its whole prompt, leaves this
      // write to a closed pipe — an EPIPE that arrives as an 'error' EVENT on the stream, which is
      // unhandled-throw territory just like the child's own. The run's real outcome is the exit code.
      child.stdin?.on("error", () => {});
      child.stdin?.end(opts.stdin);
    }

    return {
      lines,
      kill: () => void child.kill(),
      launchFailure: () => spawnError,
      exit: new Promise<number>((resolve) => {
        child.on("error", () => resolve(-1));
        child.on("close", (code) => resolve(spawnError ? -1 : (code ?? 0)));
      }),
    };
  };
}
