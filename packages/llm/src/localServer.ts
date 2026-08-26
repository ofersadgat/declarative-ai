/**
 * MANAGED local servers (§5) — the lifecycle half of the `local` route.
 *
 * The route itself only needs a `baseURL`; whether anyone started the process behind it is a separate
 * question, and this module is the only place that answers it. Attaching to a server someone else runs
 * stays the default, because it is the common case and it needs no code at all.
 *
 * Three properties are load-bearing and easy to get wrong:
 *
 *  - **Probe before spawn.** A configured `serve` does NOT mean "start a server", it means "make sure
 *    one is there". If the endpoint already answers — a developer with `ollama serve` in another
 *    terminal, a second router in the same process — we adopt it. Spawning unconditionally would fail
 *    on the bound port and turn a working setup into an error.
 *  - **Close only what we spawned.** An adopted server belongs to whoever started it. Killing it on
 *    `close()` would take down the user's own process because a workflow happened to finish.
 *  - **Boot on first REQUEST, not on resolve.** `resolveModel` is synchronous and cannot await a
 *    process start, so readiness is awaited inside the provider's `fetch` seam. That also means a
 *    router configured with a managed server that is never CALLED never starts anything.
 */
import type { FetchFunction } from "@ai-sdk/provider-utils";
import { createLogger } from "@declarative-ai/log";

const log = createLogger("engine.providers.localServer");

/** A spawned server process, reduced to what the supervisor needs. */
export interface ServerProcess {
  /** Ask it to stop. */
  kill(): void;
  /** Resolves when it has exited. */
  exited: Promise<void>;
}

/** The injectable launch seam — supplied by the caller for a non-standard launcher, or in tests. */
export type SpawnServer = (spec: ManagedServerSpec) => ServerProcess | Promise<ServerProcess>;

/** How to START the server behind a {@link LocalServerConfig}, when we are the one responsible for it. */
export interface ManagedServerSpec {
  /** Executable to run — `ollama`, `llama-server`, `vllm`, … */
  command: string;
  args?: readonly string[];
  cwd?: string;
  /** Extra environment for the child, merged over the parent's. */
  env?: Record<string, string>;
  /**
   * URL polled until it answers, to decide BOTH "is one already running" and "has ours finished
   * booting". Defaults to `${baseURL}/models` — the OpenAI-compatible model list, which every server
   * this route targets implements and which needs no request body.
   */
  readyUrl?: string;
  /** How long to wait for readiness before giving up (and killing a process we started). Default 60s —
   *  a server that memory-maps a large model can take a while to answer its first request. */
  readyTimeoutMs?: number;
  /** Gap between readiness polls. Default 250ms. */
  pollIntervalMs?: number;
  /** How long to wait for a graceful exit before returning from `close()`. Default 5s. */
  stopTimeoutMs?: number;
  /** Launch seam. Defaults to `node:child_process.spawn`, imported lazily so this module stays
   *  importable where there is no child-process API. */
  spawn?: SpawnServer;
}

/** Default `spawn`, importing `node:child_process` lazily so importing this module never requires it. */
async function nodeSpawn(spec: ManagedServerSpec): Promise<ServerProcess> {
  const { spawn } = await import("node:child_process");
  const child = spawn(spec.command, [...(spec.args ?? [])], {
    ...(spec.cwd !== undefined ? { cwd: spec.cwd } : {}),
    env: { ...process.env, ...spec.env },
    // stdio is IGNORED, not piped. An unread pipe fills at ~64 KB and the child then blocks forever on
    // write — a server logging every request would wedge itself. Its diagnostics are not our channel.
    stdio: "ignore",
    detached: false,
  });
  const exited = new Promise<void>((resolve) => {
    child.once("exit", () => resolve());
    // A `ChildProcess` 'error' event with NO listener throws, which would take the host process down —
    // and ENOENT on a missing binary is the likeliest first-run outcome. Capture it and let the
    // readiness timeout report it instead.
    child.once("error", (err) => {
      log.warn("managed server failed to start", { command: spec.command, error: String(err) });
      resolve();
    });
  });
  return { kill: () => void child.kill(), exited };
}

/** True iff the endpoint answers at all. Any HTTP status counts — a 404 from a server that does not
 *  implement `/models` still proves something is listening, which is the question being asked. */
async function probe(url: string, fetchImpl: FetchFunction, timeoutMs: number): Promise<boolean> {
  try {
    const res = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) });
    void res.body?.cancel();
    return true;
  } catch {
    return false;
  }
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/**
 * Owns at most one server process for one {@link ManagedServerSpec}, and knows whether it owns it.
 *
 * `ensureReady` is idempotent and concurrency-safe: N simultaneous first calls share one boot. A FAILED
 * boot clears the memo so a later call retries — a server that lost a race for its port, or was simply
 * slow, should not poison the router for the life of the process — but the process we started is killed
 * first, so a retry cannot leave an orphan behind.
 */
export class ManagedServer {
  private ready: Promise<void> | undefined;
  private owned: ServerProcess | undefined;
  private readonly readyUrl: string;

  constructor(
    private readonly spec: ManagedServerSpec,
    baseURL: string,
    private readonly fetchImpl: FetchFunction = globalThis.fetch,
  ) {
    this.readyUrl = spec.readyUrl ?? `${baseURL.replace(/\/$/, "")}/models`;
  }

  ensureReady(): Promise<void> {
    this.ready ??= this.boot().catch((err: unknown) => {
      this.ready = undefined;
      throw err;
    });
    return this.ready;
  }

  private async boot(): Promise<void> {
    const pollMs = this.spec.pollIntervalMs ?? 250;
    // ADOPT an already-running server rather than fighting it for the port.
    if (await probe(this.readyUrl, this.fetchImpl, pollMs * 4)) {
      log.debug("adopted a server that was already running", { readyUrl: this.readyUrl });
      return;
    }

    const proc = await (this.spec.spawn ?? nodeSpawn)(this.spec);
    this.owned = proc;
    log.debug("spawned managed server", { command: this.spec.command, readyUrl: this.readyUrl });

    const deadline = Date.now() + (this.spec.readyTimeoutMs ?? 60_000);
    let exited = false;
    void proc.exited.then(() => {
      exited = true;
    });
    while (Date.now() < deadline) {
      if (await probe(this.readyUrl, this.fetchImpl, pollMs * 4)) return;
      if (exited) break; // it died; polling until the deadline would just delay the real message
      await sleep(pollMs);
    }
    // The reason is captured BEFORE the kill, because killing resolves `proc.exited` and would flip
    // `exited` itself — reporting every readiness TIMEOUT as "the process died", which sends the reader
    // hunting a crashing binary when the binary is fine and merely slow (or `readyUrl` is wrong).
    const reason = exited
      ? `managed server "${this.spec.command}" exited before it became ready (probed ${this.readyUrl})`
      : `managed server "${this.spec.command}" did not answer ${this.readyUrl} within ${this.spec.readyTimeoutMs ?? 60_000}ms`;
    // Kill before surfacing it: `ensureReady` clears its memo on rejection so a later call can retry,
    // and a retry that spawned a second process while the first was still running would leak one per
    // attempt.
    await this.close();
    throw new Error(reason);
  }

  /** Stop the process IF WE STARTED IT. An adopted server is left alone — it belongs to whoever ran it,
   *  and taking it down because a workflow finished would be a surprise the caller never asked for. */
  async close(): Promise<void> {
    const proc = this.owned;
    this.owned = undefined;
    this.ready = undefined;
    if (proc === undefined) return;
    proc.kill();
    await Promise.race([proc.exited, sleep(this.spec.stopTimeoutMs ?? 5_000)]);
  }
}

/** Wrap a fetch so every request through it waits for the server to be ready first. Memoized inside
 *  {@link ManagedServer}, so this costs one already-resolved promise per request after the first. */
export function readyGatedFetch(server: ManagedServer, fetchImpl: FetchFunction = globalThis.fetch): FetchFunction {
  return async (input, init) => {
    await server.ensureReady();
    return fetchImpl(input, init);
  };
}
