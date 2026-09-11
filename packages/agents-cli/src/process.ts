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
 *  - **stderr is piped and ALWAYS drained, into a bounded tail.** An unread pipe fills at ~64 KB
 *    and the child then blocks forever on write, so stdout stops and `exit` never settles — which is
 *    why it used to be `"ignore"`d outright. But a CLI that dies before its first stdout line says
 *    why on stderr and nowhere else: a run recorded as `agent CLI exited with code 1` had
 *    `Error: MCP tool mcp__dai__approve (passed via --permission-prompt-tool) not found` sitting
 *    in the pipe nobody read. So the pipe is consumed unconditionally (the deadlock does not care
 *    whether anybody is listening) and the last {@link STDERR_TAIL} characters are kept for the
 *    exit message. The data listener is attached synchronously at spawn, before anything awaits.
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
  /**
   * The last {@link STDERR_TAIL} characters the agent wrote to stderr — read after `exit` settles.
   *
   * What a CLI that died before its first stdout line has to say for itself. The exit code alone
   * names nothing: `exited with code 1` covers a missing permission tool, an unreachable MCP bridge,
   * a bad flag and an auth failure alike, and every one of them is spelled out on stderr. Optional so
   * a fake process need not model it: absent ⇒ the exit code is the whole story.
   */
  stderrTail?: () => string | undefined;
  /**
   * Write one more line to the agent's stdin, after it has started.
   *
   * The whole basis of STEERING. A `-p` subprocess is handed its prompt once and its stdin is closed,
   * so the only mid-run signal an adapter has is `kill()` — which ends the process, not the turn, and
   * throws away the partial answer along with it. With stdin open the CLI's control protocol is
   * reachable: an interrupt ends the TURN and a `result` still arrives, so "stop and tell me what you
   * found" can be answered with what was found.
   *
   * Optional because a fake process need not model it, and because its absence is the honest signal
   * that this transport cannot steer — `createCliAgentQuery` reads exactly that to decide whether to
   * offer `interrupt()` at all.
   */
  write?: (line: string) => void;
  /**
   * Close stdin — which is what ENDS the session when the input channel is a stream.
   *
   * Not a courtesy: under `--input-format stream-json` the CLI waits for another message rather than
   * exiting when a turn finishes, so without this a completed run never settles. Measured against
   * claude 2.1.246: closing stdin exits 0.
   */
  endInput?: () => void;
}

/** How much of the agent's stderr is kept for its exit message — the LAST characters, since the
 *  line that names the death is the last one written, and a chatty CLI's warnings come before it. */
export const STDERR_TAIL = 4096;

/**
 * A bounded tail of a text stream: feed it every chunk, read the last {@link STDERR_TAIL} characters.
 *
 * Shared with hosts that supply their own {@link SpawnProcess} (JaiRA's job-tracking spawn drains
 * stderr into its own store and needs the same tail for the same message), so the two copies of the
 * seam keep one definition of "what the process said".
 */
export function stderrTail(limit = STDERR_TAIL): { push(chunk: string): void; read(): string | undefined } {
  let kept = "";
  return {
    push(chunk) {
      kept = (kept + chunk).slice(-limit);
    },
    read() {
      const text = kept.trim();
      return text.length > 0 ? text : undefined;
    },
  };
}

/** The exit message for a run that ended with a nonzero code: the code, and what stderr said. */
export function exitMessage(what: string, code: number, tail: string | undefined): string {
  return tail === undefined ? `${what} exited with code ${code}` : `${what} exited with code ${code}: ${tail}`;
}

/** How a process is launched. `stdin` is the second way to hand an agent its instruction. */
export interface SpawnOptions {
  cwd?: string;
  /**
   * Keep stdin OPEN after the initial write, so the adapter can say more later.
   *
   * The difference between an agent you hand a prompt to and one you can steer. Closing stdin is how
   * a `stream-json` session ENDS, so an adapter that wants to interrupt or add a message must ask for
   * the channel to stay open and close it itself ({@link AgentProcess.endInput}).
   */
  keepInputOpen?: boolean;
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
      stdio: [opts.stdin === undefined && opts.keepInputOpen !== true ? "ignore" : "pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    const lines = readline.createInterface({ input: child.stdout!, crlfDelay: Infinity });
    // THE DRAIN, attached before anything can await the process — see the module comment. Consuming
    // the bytes is the requirement; keeping the tail is the point.
    const tail = stderrTail();
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => tail.push(chunk));
    child.stderr?.on("error", () => undefined);

    let spawnError: Error | undefined;
    child.on("error", (e: Error) => {
      spawnError = e;
      lines.close();
    });

    // An agent that fails to start, or that answers before reading its whole prompt, leaves a write
    // to a closed pipe — an EPIPE that arrives as an 'error' EVENT on the stream, which is
    // unhandled-throw territory just like the child's own. The run's real outcome is the exit code.
    child.stdin?.on("error", () => {});
    if (opts.stdin !== undefined) {
      // Written either way; only the CLOSE depends on whether the caller means to say more.
      if (opts.keepInputOpen === true) child.stdin?.write(opts.stdin);
      else child.stdin?.end(opts.stdin);
    }

    return {
      lines,
      kill: () => void child.kill(),
      launchFailure: () => spawnError,
      stderrTail: () => tail.read(),
      ...(child.stdin !== null
        ? {
            write: (line: string): void => void child.stdin?.write(line),
            endInput: (): void => void child.stdin?.end(),
          }
        : {}),
      exit: new Promise<number>((resolve) => {
        child.on("error", () => resolve(-1));
        child.on("close", (code) => resolve(spawnError ? -1 : (code ?? 0)));
      }),
    };
  };
}
