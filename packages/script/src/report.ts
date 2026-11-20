/**
 * REPORTING (SCRIPTS.md §17.4): a script run as a plain program tells an engine that will listen what
 * it does — every call, what it said, and each tool an agent wants to use — so the run is visible there
 * and gated there, though it is still a plain program: it runs its own calls, with the keys of its own
 * environment, reads `input()` from its own terminal, and is gone when its process is. Nothing about it
 * can be resumed.
 *
 * ## Finding an endpoint
 *
 * An engine that takes reports ADVERTISES itself ({@link advertiseEndpoint}): a JSON file
 * `{ "name", "socket", "token", "protocol", "pid" }` in {@link endpointsDir} — `$DAI_ENDPOINTS`, else this
 * user's directory in the system's temporary one — written when it starts listening and taken away when
 * it stops. One whose `pid` is no longer running is passed over, and so is every advert in a directory
 * that is not this user's alone.
 * `$DAI_SCRIPT_ENDPOINT` names one directly — the advert itself, or a file holding it — and
 * `DAI_SCRIPT_REPORT=0` turns reporting off.
 *
 * ## The wire
 *
 * Newline-delimited JSON on the advertised socket (a Unix socket, or a named pipe on Windows). A request
 * is `{ "id": n, "method": m, "params": p }` and is answered `{ "id": n, "result": r }` or
 * `{ "id": n, "error": { "message": s } }`; a notification has no `id` and no answer.
 *
 * | method         | kind         | params → result                                                        |
 * | -------------- | ------------ | ---------------------------------------------------------------------- |
 * | `hello`        | request      | `{ token, protocol, run }` → `{ run, name, version?, url? }` — begins it |
 * | `call.started` | notification | `{ call }` — a call is being made                                       |
 * | `call.settled` | request      | `{ id, outcome }` → `{}` — what it answered, and the conversation it added |
 * | `approve`      | request      | `{ id, toolName, input }` → `{ allow, reason? }` — an agent's tool use   |
 * | `note`         | notification | `{ kind, … }` — a phase, a log line, an input answered, a script called  |
 * | `end`          | request      | `{ value }` or `{ error }` → `{}`                                       |
 *
 * A session a call continues is named as the script holds it (`s1@2`): the endpoint keeps its own and
 * maps between them. A connection that closes before `end` is a run that ended without saying how.
 */
import { createConnection, type Socket } from "node:net";
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";

/** The protocol this client speaks. */
export const REPORT_PROTOCOL = 1;

/** What an engine advertises about where it takes reports. */
export interface EndpointAdvert {
  /** How the run is told it is being reported: "JaiRA". */
  name: string;
  /** A Unix socket's path, or a Windows named pipe's. */
  socket: string;
  /** Proves to the endpoint that this machine's user found it. */
  token: string;
  protocol: number;
  /** The advertising process — an advert whose process is gone is passed over. */
  pid?: number;
  /** The advertising engine's version, for a person reading the file. */
  version?: string;
}

/** The run, as `hello` describes it. */
export interface ReportedRunInfo {
  /** The script, absolute. */
  file: string;
  args: readonly unknown[];
  cwd: string;
  pid: number;
  /** The script's own `meta` — its name, and the environment (permission set) it declares. */
  meta?: { name?: string; description?: string; environment?: unknown };
}

/** One call, as it is begun. */
export interface ReportedCall {
  /** The run's own id for it — what its settlement and its approvals name. */
  id: string;
  kind: "llm" | "agent";
  model?: string;
  system?: string;
  prompt: string;
  /** The conversation it continues, as the script names it — `null` for a fresh one. */
  session?: string | null;
  schema?: JsonSchema;
  /** The directory an agent works in. */
  cwd?: string;
}

/** One call, as it ended. */
export interface ReportedOutcome {
  value?: unknown;
  error?: string;
  /** Where its conversation ended, as the script names it. */
  session?: string;
  /** What the call added to its conversation — its prompt, then what was said, provider-shaped. */
  messages?: JsonValue[];
  costUsd?: number;
  /** The agent's own session handle, when it ran on one. */
  providerSession?: string;
}

/** The decision about an agent's tool use. */
export interface ReportedApproval {
  allow: boolean;
  reason?: string;
}

/** A run being reported: what the host tells the endpoint as it goes. */
export interface ScriptReport {
  /** The endpoint's id for the run. */
  readonly run: string;
  /** Who is listening: "JaiRA". */
  readonly name: string;
  /** Where a person can watch it, when the endpoint says. */
  readonly url?: string;
  callStarted(call: ReportedCall): void;
  callSettled(id: string, outcome: ReportedOutcome): Promise<void>;
  approve(id: string, request: { toolName: string; input: unknown }): Promise<ReportedApproval>;
  note(kind: "phase" | "log" | "input" | "workflow", detail: Record<string, JsonValue>): void;
  end(outcome: { value: unknown } | { error: string }): Promise<void>;
  /** Let go of the connection — after `end`, or instead of it. */
  close(): void;
}

/**
 * Where endpoints advertise: `$DAI_ENDPOINTS`, else this user's directory in the system's temporary one —
 * `/tmp/declarative-ai-endpoints-<uid>` on macOS and Linux, `%TEMP%\declarative-ai-endpoints` on Windows.
 * An advert lasts only as long as its process, so it is kept where the system keeps what does not outlive
 * a session, not in the person's home. On macOS and Linux it is `/tmp` itself, as tmux keeps its sockets,
 * rather than `$TMPDIR` or `$XDG_RUNTIME_DIR`: a script run from a terminal, ssh or cron must look where an
 * engine opened from the desktop wrote, and those differ between them. `/tmp` is shared, so the directory
 * carries the user's id and is trusted only while it is theirs alone (`ownedPrivately`); Windows' `%TEMP%`
 * is already the user's own.
 */
export function endpointsDir(env: NodeJS.ProcessEnv = process.env): string {
  const named = env["DAI_ENDPOINTS"];
  if (named !== undefined && named.length > 0) return named;
  if (process.platform === "win32") return join(tmpdir(), "declarative-ai-endpoints");
  return join("/tmp", `declarative-ai-endpoints-${process.getuid?.() ?? "user"}`);
}

/**
 * Whether a directory is this user's alone. In a directory anyone else could write to, an advert could be
 * someone else's, and a run would send them its prompts, its answers and its tools to approve.
 */
function ownedPrivately(dir: string): boolean {
  if (process.platform === "win32") return true;
  try {
    const stat = lstatSync(dir);
    return stat.isDirectory() && stat.uid === process.getuid?.() && (stat.mode & 0o077) === 0;
  } catch {
    return false;
  }
}

/**
 * Advertise an endpoint as `<name>.json` in {@link endpointsDir}, making the directory — this user's only —
 * when it is not there, and refusing one that is not this user's alone. Returns the advert's path; the
 * engine removes it when it stops listening.
 *
 * Call it again while listening: it puts back an advert the system's cleaning of its temporary directory
 * took away (macOS and many Linuxes clear what has not changed in days), and a rewritten advert is never
 * that old.
 */
export function advertiseEndpoint(name: string, advert: EndpointAdvert, env: NodeJS.ProcessEnv = process.env): string {
  const dir = endpointsDir(env);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  if (!ownedPrivately(dir)) throw new Error(`${dir} is not this user's alone, so nothing is advertised there`);
  const file = join(dir, `${name}.json`);
  // Written beside it and renamed over it, so a run looking never reads half of one.
  const partial = `${file}.${process.pid}.partial`;
  writeFileSync(partial, `${JSON.stringify(advert, null, 2)}\n`, { mode: 0o600 });
  renameSync(partial, file);
  return file;
}

function pidAlive(pid: number | undefined): boolean {
  if (pid === undefined) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    // A process this user may not signal is still a process.
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
}

function readAdvert(text: string): EndpointAdvert | undefined {
  try {
    const advert = JSON.parse(text) as Partial<EndpointAdvert>;
    if (typeof advert.socket !== "string" || typeof advert.token !== "string" || typeof advert.protocol !== "number") return undefined;
    return { name: typeof advert.name === "string" ? advert.name : "an engine", socket: advert.socket, token: advert.token, protocol: advert.protocol, ...(typeof advert.pid === "number" ? { pid: advert.pid } : {}) };
  } catch {
    return undefined;
  }
}

/**
 * The endpoints a run may report to, in the order to try them: the one `$DAI_SCRIPT_ENDPOINT` names,
 * then the advertised ones whose process is running, newest first. None when `DAI_SCRIPT_REPORT=0`.
 */
export function findEndpoints(env: NodeJS.ProcessEnv = process.env): EndpointAdvert[] {
  if (env["DAI_SCRIPT_REPORT"] === "0") return [];
  const named = env["DAI_SCRIPT_ENDPOINT"];
  if (named !== undefined && named.length > 0) {
    const text = named.trimStart().startsWith("{") ? named : existsSync(named) ? readFileSync(named, "utf8") : undefined;
    const advert = text !== undefined ? readAdvert(text) : undefined;
    return advert !== undefined ? [advert] : [];
  }
  const dir = endpointsDir(env);
  if (!existsSync(dir) || !ownedPrivately(dir)) return [];
  const found: Array<{ advert: EndpointAdvert; at: number }> = [];
  for (const name of readdirSync(dir)) {
    if (!name.endsWith(".json")) continue;
    let text: string;
    try {
      text = readFileSync(join(dir, name), "utf8");
    } catch {
      continue;
    }
    const advert = readAdvert(text);
    if (advert === undefined || advert.protocol !== REPORT_PROTOCOL || !pidAlive(advert.pid)) continue;
    found.push({ advert, at: found.length });
  }
  return found.map((one) => one.advert);
}

/** A connection that speaks the wire: requests answered by id, notifications sent and forgotten. */
class Wire {
  private next = 0;
  private buffer = "";
  private readonly waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private closed: string | undefined;

  constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk: string) => {
      this.buffer += chunk;
      for (let at = this.buffer.indexOf("\n"); at >= 0; at = this.buffer.indexOf("\n")) {
        const line = this.buffer.slice(0, at);
        this.buffer = this.buffer.slice(at + 1);
        if (line.trim().length === 0) continue;
        let frame: { id?: unknown; result?: unknown; error?: { message?: unknown } };
        try {
          frame = JSON.parse(line) as typeof frame;
        } catch {
          continue;
        }
        if (typeof frame.id !== "number") continue;
        const pending = this.waiting.get(frame.id);
        if (pending === undefined) continue;
        this.waiting.delete(frame.id);
        if (frame.error !== undefined) pending.reject(new Error(typeof frame.error.message === "string" ? frame.error.message : "the endpoint refused"));
        else pending.resolve(frame.result);
      }
    });
    const end = (why: string): void => {
      if (this.closed !== undefined) return;
      this.closed = why;
      for (const pending of this.waiting.values()) pending.reject(new Error(`the endpoint went away: ${why}`));
      this.waiting.clear();
    };
    socket.on("close", () => end("closed"));
    socket.on("error", (e) => end(e.message));
  }

  request(method: string, params: unknown, timeoutMs?: number): Promise<unknown> {
    if (this.closed !== undefined) return Promise.reject(new Error(`the endpoint went away: ${this.closed}`));
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      const timer = timeoutMs !== undefined ? setTimeout(() => {
        this.waiting.delete(id);
        reject(new Error(`the endpoint did not answer '${method}' in ${timeoutMs} ms`));
      }, timeoutMs) : undefined;
      this.waiting.set(id, {
        resolve: (value) => (clearTimeout(timer), resolve(value)),
        reject: (error) => (clearTimeout(timer), reject(error)),
      });
      this.socket.write(`${JSON.stringify({ id, method, params })}\n`);
    });
  }

  notify(method: string, params: unknown): void {
    if (this.closed === undefined) this.socket.write(`${JSON.stringify({ method, params })}\n`);
  }

  get gone(): boolean {
    return this.closed !== undefined;
  }

  close(): void {
    this.socket.end();
  }
}

/**
 * Begin reporting a run to `advert`'s endpoint — `undefined` when it cannot be reached or will not take
 * it, and the run goes on unreported.
 */
export async function connectReport(advert: EndpointAdvert, run: ReportedRunInfo, options: { timeoutMs?: number } = {}): Promise<ScriptReport | undefined> {
  const timeoutMs = options.timeoutMs ?? 3000;
  const socket = await new Promise<Socket | undefined>((resolve) => {
    const s = createConnection(advert.socket);
    const timer = setTimeout(() => (s.destroy(), resolve(undefined)), timeoutMs);
    s.once("connect", () => (clearTimeout(timer), resolve(s)));
    s.once("error", () => (clearTimeout(timer), resolve(undefined)));
  });
  if (socket === undefined) return undefined;
  const wire = new Wire(socket);
  let hello: { run?: unknown; name?: unknown; url?: unknown };
  try {
    hello = (await wire.request("hello", { token: advert.token, protocol: REPORT_PROTOCOL, run }, timeoutMs)) as typeof hello;
  } catch {
    socket.destroy();
    return undefined;
  }
  if (typeof hello?.run !== "string") {
    socket.destroy();
    return undefined;
  }
  // A notification is fire-and-forget; a request that cannot be made is a report lost, never the run's
  // failure — the program goes on as a plain one.
  const settle = async (method: string, params: unknown): Promise<unknown> => {
    try {
      return await wire.request(method, params);
    } catch {
      return undefined;
    }
  };
  return {
    run: hello.run,
    name: typeof hello.name === "string" ? hello.name : advert.name,
    ...(typeof hello.url === "string" ? { url: hello.url } : {}),
    callStarted: (call) => wire.notify("call.started", { call }),
    callSettled: async (id, outcome) => void (await settle("call.settled", { id, outcome })),
    approve: async (id, request) => {
      // Gated by the endpoint: what it cannot answer is refused, never allowed by default.
      try {
        const answer = (await wire.request("approve", { id, toolName: request.toolName, input: request.input })) as { allow?: unknown; reason?: unknown };
        return answer?.allow === true ? { allow: true } : { allow: false, ...(typeof answer?.reason === "string" ? { reason: answer.reason } : {}) };
      } catch (e) {
        return { allow: false, reason: `${advert.name} could not be asked: ${(e as Error).message}` };
      }
    },
    note: (kind, detail) => wire.notify("note", { kind, ...detail }),
    end: async (outcome) => void (await settle("end", "error" in outcome ? { error: outcome.error } : { value: outcome.value ?? null })),
    close: () => wire.close(),
  };
}

/** Begin reporting to the first endpoint that takes the run, or `undefined`. */
export async function discoverReport(run: ReportedRunInfo, env: NodeJS.ProcessEnv = process.env): Promise<ScriptReport | undefined> {
  for (const advert of findEndpoints(env)) {
    const report = await connectReport(advert, run);
    if (report !== undefined) return report;
  }
  return undefined;
}
