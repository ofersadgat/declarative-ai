/**
 * A PERSISTENT bridge: one listener, on a worker thread, serving every run — the host's side.
 *
 * `defaultStartMcpBridge` is one listener per run on the caller's own loop. This is the alternative
 * for a host whose loop is busy: the socket and the MCP handshake live in `mcpBridgeWorker.ts`, on a
 * thread that stays responsive while the host folds records and writes snapshots, and each run is a
 * REGISTRATION on that thread rather than a server of its own. What differentiates runs is what
 * already did — the per-run secret in the URL path: the worker matches a request's path against its
 * registered runs and serves that run's tools, so two concurrent agents share a port and nothing else.
 *
 * What stays here, on the host's thread, is everything that touches the run: the spec (its approver,
 * its tool impls, its validator) never crosses the port. The worker forwards each `tools/call` as a
 * message and this module answers it through the same `handleToolCall` the in-process bridge uses —
 * so the wire contract, the reserved-name rule and the argument gate have exactly one implementation.
 *
 * `start` IS a {@link StartMcpBridge}, so a transport takes it through the seam it already has, and
 * the bridges it returns carry `ready` — the signal the adapter holds the prompt for. The worker is
 * spawned on first use, `unref`'d so it never keeps a process alive, and respawned if it dies; a
 * registration in flight when it dies is refused, which the adapter reports as a bridge that could
 * not start.
 */
import { Worker } from "node:worker_threads";
import { textResult, type McpToolResult } from "./deps.js";
import type { McpBridge, McpBridgeSpec, StartMcpBridge } from "./mcpBridge.js";
import type { FromWorker, ToWorker } from "./mcpBridgeWorker.js";
import { bridgePath, handleToolCall, newBridgeToken, toolDescriptors } from "./mcpProtocol.js";

export interface McpBridgeHostOptions {
  /**
   * The worker's entry file. Default: `mcpBridgeWorker.js` beside this module — right when this
   * package is loaded from its own `dist`, wrong once a bundler has folded this module into
   * something else, which is why a bundled host emits the worker as its own entry and names it here.
   */
  workerFile?: string | URL;
}

export interface McpBridgeHost {
  /** Register a run: what a transport's `startBridge` option takes. */
  start: StartMcpBridge;
  /** Stop the worker. Every registered run's bridge is gone with it; `start` refuses afterwards. */
  close(): Promise<void>;
}

interface Run {
  spec: McpBridgeSpec;
  markReady: () => void;
}

export function createMcpBridgeHost(options: McpBridgeHostOptions = {}): McpBridgeHost {
  const runs = new Map<string, Run>();
  const registering = new Map<string, { resolve: (port: number) => void; reject: (e: Error) => void }>();
  let worker: Worker | undefined;
  let closed = false;

  /**
   * Whether the worker keeps the process alive: only while a run is registered or registering.
   *
   * Idle, it is `unref`'d, so a host that is done does not wait on a bridge with nothing to serve.
   * But a run in flight must hold the process — the moment between `start` and the CLI's spawn has
   * no other handle open, and a plain `unref` there let a script exit mid-await ("unsettled
   * top-level await") with no run at all.
   */
  const retain = (): void => {
    if (worker === undefined) return;
    if (runs.size > 0 || registering.size > 0) worker.ref();
    else worker.unref();
  };

  const forget = (which: Worker, reason: Error): void => {
    if (worker !== which) return;
    worker = undefined;
    for (const [token, waiting] of registering) {
      registering.delete(token);
      waiting.reject(reason);
    }
    // A run whose worker died has no bridge any more; its adapter is not left waiting on a handshake
    // that cannot come — the CLI's own failure to reach the port is what it reports.
    for (const run of runs.values()) run.markReady();
    runs.clear();
  };

  const answer = (message: FromWorker): void => {
    switch (message.type) {
      case "registered":
        registering.get(message.token)?.resolve(message.port);
        registering.delete(message.token);
        retain();
        return;
      case "failed":
        registering.get(message.token)?.reject(new Error(message.message));
        registering.delete(message.token);
        retain();
        return;
      case "ready":
        runs.get(message.token)?.markReady();
        return;
      case "call":
        void answerCall(message.token, message.name, message.args).then((result) => worker?.postMessage({ type: "result", id: message.id, result } satisfies ToWorker));
        return;
      case "closed":
        return;
    }
  };

  const answerCall = async (token: string, name: string, args: unknown): Promise<McpToolResult> => {
    const run = runs.get(token);
    // A call for a run that has closed — its bridge was torn down between the agent's request and
    // this answer. Not a tool result the agent can act on, and above all not an allow.
    if (run === undefined) return textResult("the run this bridge served has ended", true);
    try {
      return await handleToolCall(run.spec, name, args);
    } catch (e) {
      return textResult(`tool '${name}' failed: ${e instanceof Error ? e.message : String(e)}`, true);
    }
  };

  const spawn = (): Worker => {
    if (worker !== undefined) return worker;
    const file = options.workerFile ?? new URL("./mcpBridgeWorker.js", import.meta.url);
    const w = new Worker(file);
    w.on("message", answer);
    w.on("error", (e) => forget(w, e));
    w.on("exit", (code) => forget(w, new Error(`the MCP bridge worker exited (${code})`)));
    worker = w;
    retain();
    return w;
  };

  const start: StartMcpBridge = async (spec) => {
    if (closed) throw new Error("the MCP bridge host is closed");
    // Computed HERE, before anything is registered: `toolDescriptors` is where a host tool named
    // `approve` is refused, and the in-process bridge refuses it at the same point.
    const descriptors = toolDescriptors(spec);
    const token = newBridgeToken();
    const path = bridgePath(token);
    let markReady: () => void = () => undefined;
    const ready = new Promise<void>((resolve) => {
      markReady = resolve;
    });
    runs.set(token, { spec, markReady });
    let port: number;
    try {
      port = await new Promise<number>((resolve, reject) => {
        registering.set(token, { resolve, reject });
        const w = spawn();
        retain();
        w.postMessage({ type: "register", token, path, descriptors } satisfies ToWorker);
      });
    } catch (e) {
      runs.delete(token);
      retain();
      throw e;
    }
    return {
      url: `http://127.0.0.1:${port}${path}`,
      ready,
      close: async () => {
        runs.delete(token);
        markReady();
        worker?.postMessage({ type: "unregister", token } satisfies ToWorker);
        retain();
      },
    };
  };

  return {
    start,
    close: async () => {
      closed = true;
      const w = worker;
      if (w === undefined) return;
      worker = undefined;
      for (const run of runs.values()) run.markReady();
      runs.clear();
      // Held for the length of the close: an idle worker is `unref`'d, and a process with nothing
      // else pending would otherwise exit before this settles.
      w.ref();
      const gone = new Promise<void>((resolve) => w.once("exit", () => resolve()));
      w.postMessage({ type: "close" } satisfies ToWorker);
      // `close` lets the listener drain; `terminate` is the floor under it.
      await Promise.race([gone, new Promise<void>((resolve) => setTimeout(resolve, 1000).unref())]);
      await w.terminate();
    },
  };
}
