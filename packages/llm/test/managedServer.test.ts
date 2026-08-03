import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { describe, expect, it } from "vitest";
import { ManagedServer, type ManagedServerSpec, type ServerProcess } from "../src/localServer.js";
import { createModelRouter } from "../src/router.js";

/**
 * MANAGED local servers: probe-before-spawn, readiness polling, and close-only-what-we-started.
 *
 * The spawn seam is exercised two ways. Most cases use a fake process so the state machine can be
 * driven deterministically (a server that never becomes ready, one that dies during boot); the last
 * case spawns a REAL child — `node` itself, serving HTTP — because a fake can't prove that the default
 * launcher, the readiness poll and the kill actually compose.
 */

/** A fake process whose readiness and lifetime the test drives. */
function fakeProcess(): ServerProcess & { killed: boolean; die: () => void } {
  let resolveExit!: () => void;
  const exited = new Promise<void>((r) => (resolveExit = r));
  const proc = {
    killed: false,
    exited,
    kill: () => {
      proc.killed = true;
      resolveExit();
    },
    die: () => resolveExit(),
  };
  return proc;
}

/** A listening HTTP server that answers anything with 200. */
async function listening(): Promise<{ server: Server; baseURL: string }> {
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end("{}");
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return { server, baseURL: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1` };
}

const fast = (over: Partial<ManagedServerSpec> = {}): ManagedServerSpec => ({
  command: "should-not-run",
  pollIntervalMs: 10,
  readyTimeoutMs: 600,
  stopTimeoutMs: 200,
  ...over,
});

describe("ManagedServer", () => {
  it("ADOPTS a server that is already listening, and never spawns", async () => {
    // The case that makes `serve` safe to leave configured: a developer with `ollama serve` already
    // running in another terminal. Spawning unconditionally would fail on the bound port.
    const { server, baseURL } = await listening();
    try {
      let spawned = 0;
      const managed = new ManagedServer(fast({ spawn: () => (spawned++, fakeProcess()) }), baseURL);
      await managed.ensureReady();
      expect(spawned).toBe(0);
      // ...and closing must NOT stop it: it belongs to whoever started it.
      await managed.close();
      expect(server.listening).toBe(true);
    } finally {
      await new Promise<void>((r) => server.close(() => r()));
    }
  });

  it("spawns when nothing answers, then polls until it does", async () => {
    const { server, baseURL } = await listening();
    let hidden = true;
    try {
      // Probe against a URL that only starts answering once `hidden` flips — standing in for a server
      // that takes time to memory-map its weights before serving.
      const managed = new ManagedServer(
        fast({ spawn: () => fakeProcess() }),
        baseURL,
        (input, init) => (hidden ? Promise.reject(new Error("connection refused")) : fetch(input, init)),
      );
      const ready = managed.ensureReady();
      setTimeout(() => (hidden = false), 60);
      await expect(ready).resolves.toBeUndefined();
      await managed.close();
    } finally {
      await new Promise<void>((r) => server.close(() => r()));
    }
  });

  it("N concurrent first calls share ONE boot", async () => {
    const { server, baseURL } = await listening();
    try {
      let spawned = 0;
      const managed = new ManagedServer(fast({ spawn: () => (spawned++, fakeProcess()) }), baseURL, () =>
        Promise.reject(new Error("refused")),
      );
      // Nothing ever answers, so all five share the same failing boot rather than starting five servers.
      const results = await Promise.allSettled([1, 2, 3, 4, 5].map(() => managed.ensureReady()));
      expect(results.every((r) => r.status === "rejected")).toBe(true);
      expect(spawned).toBe(1);
    } finally {
      await new Promise<void>((r) => server.close(() => r()));
    }
  });

  it("kills the process it started when readiness times out, so a retry cannot orphan one", async () => {
    // `ensureReady` clears its memo on failure so a slow or unlucky boot can be retried. That is only
    // safe because the failed attempt's process is killed first — otherwise every retry leaks one.
    const procs: ReturnType<typeof fakeProcess>[] = [];
    const managed = new ManagedServer(
      fast({
        spawn: () => {
          const p = fakeProcess();
          procs.push(p);
          return p;
        },
      }),
      "http://127.0.0.1:1/v1",
      () => Promise.reject(new Error("refused")),
    );
    await expect(managed.ensureReady()).rejects.toThrow(/did not answer/);
    expect(procs).toHaveLength(1);
    expect(procs[0]!.killed).toBe(true);

    await expect(managed.ensureReady()).rejects.toThrow(/did not answer/); // retried...
    expect(procs).toHaveLength(2); // ...with a fresh process, the first already reaped
    expect(procs[1]!.killed).toBe(true);
  });

  it("reports a process that DIED during boot as such, without waiting out the timeout", async () => {
    const proc = fakeProcess();
    const started = Date.now();
    const managed = new ManagedServer(
      fast({ readyTimeoutMs: 5_000, spawn: () => proc }),
      "http://127.0.0.1:1/v1",
      () => Promise.reject(new Error("refused")),
    );
    setTimeout(() => proc.die(), 30);
    await expect(managed.ensureReady()).rejects.toThrow(/exited before it became ready/);
    expect(Date.now() - started).toBeLessThan(4_000); // did not sit out the 5s deadline
  });

  it("spawns a REAL child with the default launcher, waits for it, and kills it", async () => {
    // `node` as the server binary: cross-platform, already installed, and it proves the real
    // child_process path — not just the state machine around it.
    const port = 34_517;
    const script = `require("http").createServer((q,s)=>{s.writeHead(200,{"content-type":"application/json"});s.end("{}")}).listen(${port},"127.0.0.1")`;
    const managed = new ManagedServer(
      { command: process.execPath, args: ["-e", script], pollIntervalMs: 50, readyTimeoutMs: 20_000 },
      `http://127.0.0.1:${port}/v1`,
    );
    await managed.ensureReady();
    expect((await fetch(`http://127.0.0.1:${port}/v1/models`)).ok).toBe(true);
    await managed.close();
    // The port is free again, which is the observable proof the child is gone.
    await expect(fetch(`http://127.0.0.1:${port}/v1/models`, { signal: AbortSignal.timeout(2_000) })).rejects.toThrow();
  }, 30_000);
});

describe("createModelRouter — managed lifecycle", () => {
  it("boots on the first REQUEST, not on resolveModel", async () => {
    // `resolveModel` is synchronous and cannot await a process start. Gating the transport instead also
    // means a router configured with a server it never calls starts nothing at all.
    let spawned = 0;
    const router = createModelRouter({
      skipDispatcher: true,
      local: {
        baseURL: "http://127.0.0.1:1/v1",
        fetch: () => Promise.reject(new Error("refused")),
        serve: fast({ spawn: () => (spawned++, fakeProcess()) }),
      },
    });
    router.resolveModel("local/m");
    router.resolveModel("local/m", { strictStructuredOutput: true });
    expect(spawned).toBe(0); // resolving is free
    await router.close?.();
  });

  it("shares ONE process across clients that differ only by the structured-output flag", async () => {
    // Clients are keyed on the per-call strict flag; supervisors must be keyed on the endpoint, or the
    // first strict call after an advisory one would start a second server on a bound port.
    const { server, baseURL } = await listening();
    try {
      let spawned = 0;
      const router = createModelRouter({
        skipDispatcher: true,
        local: { baseURL, serve: fast({ spawn: () => (spawned++, fakeProcess()) }) },
      });
      const advisory = router.resolveModel("local/m") as unknown as { config: { fetch?: typeof fetch } };
      const strict = router.resolveModel("local/m", { strictStructuredOutput: true }) as unknown as { config: { fetch?: typeof fetch } };
      // Drive both transports; the endpoint is already listening, so both adopt the same supervisor.
      await advisory.config.fetch!(`${baseURL}/models`);
      await strict.config.fetch!(`${baseURL}/models`);
      expect(spawned).toBe(0); // adopted, not spawned — and crucially, adopted ONCE
      await router.close?.();
      expect(server.listening).toBe(true);
    } finally {
      await new Promise<void>((r) => server.close(() => r()));
    }
  });

  it("close() is idempotent and safe with nothing managed", async () => {
    const router = createModelRouter({ skipDispatcher: true, local: { baseURL: "http://127.0.0.1:1/v1" } });
    await expect(router.close?.()).resolves.toBeUndefined();
    await expect(router.close?.()).resolves.toBeUndefined();
  });
});
