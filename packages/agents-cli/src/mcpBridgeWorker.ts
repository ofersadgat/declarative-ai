/**
 * The bridge WORKER — one loopback listener for every run, on an event loop of its own.
 *
 * `defaultStartMcpBridge` stands a listener up per run, in the host's own thread. That is the right
 * shape for a library default, and it was measured to be the wrong place for a busy host: the CLI
 * never waits for an `--mcp-config` server, so a handshake that the host's loop is too congested to
 * answer before the model's first tool call is a run that dies on `mcp__dai__approve … not found`.
 * In the app that ran this, the same handshake took 1–3.7 s mid-task and 40 ms when idle; in a bare
 * process it takes ~130 ms. The delay is the host's loop, not the bridge — so the bridge moves off it.
 *
 * This file is that thread. It owns the socket and the MCP transport and answers `initialize` and
 * `tools/list` itself, from descriptors the host handed it at registration. What it cannot answer it
 * forwards over the port: a `tools/call` — a permission decision, or a host-implemented tool — is a
 * message to the parent, which runs the real impl (`handleToolCall`, against the run's spec, which
 * never leaves the main thread) and posts the result back. The lifetimes are the run's: a run is
 * REGISTERED under its own path token and UNREGISTERED at close; the listener outlives them all.
 *
 * ## Why this file imports nothing of its own
 *
 * It is loaded by path — `new Worker(file)` — so it is not part of the bundle graph the host is in.
 * A bundler is told to emit it as its own entry; a test points at this source directly, which Node
 * runs as-is (type stripping) only if every import resolves without a `.js`→`.ts` rewrite. So the
 * two small pieces of `mcpProtocol` it needs are restated here rather than imported: the
 * constant-time compare (a path check that leaks its match prefix through timing is a check a local
 * process can walk), and the SDK specifiers. Everything else — the path FORMAT, the descriptors, the
 * dispatch — arrives over the port, so the protocol keeps one home.
 *
 * ## Messages
 *
 * parent → worker: `register {token, path, descriptors}`, `unregister {token}`, `result {id, result}`,
 * `close`. worker → parent: `registered {token, port}` | `failed {token, message}`, `ready {token}`
 * (the run's first `tools/list` was served), `call {id, token, name, args}`, `closed`.
 */
import { createServer, type IncomingMessage, type Server as HttpServer, type ServerResponse } from "node:http";
import { parentPort } from "node:worker_threads";

// --- The wire, in both directions -------------------------------------------------

export interface RegisterMessage {
  type: "register";
  token: string;
  /** The URL path this run answers on — computed by the host with `bridgePath`, compared here. */
  path: string;
  /** The `tools/list` entries, as the host computed them with `toolDescriptors`. */
  descriptors: unknown[];
}
export interface UnregisterMessage {
  type: "unregister";
  token: string;
}
export interface ResultMessage {
  type: "result";
  id: number;
  /** An `McpToolResult` — what `handleToolCall` answered. */
  result: unknown;
}
export interface CloseMessage {
  type: "close";
}
export type ToWorker = RegisterMessage | UnregisterMessage | ResultMessage | CloseMessage;

export interface RegisteredMessage {
  type: "registered";
  token: string;
  port: number;
}
export interface FailedMessage {
  type: "failed";
  token: string;
  message: string;
}
export interface ReadyMessage {
  type: "ready";
  token: string;
}
export interface CallMessage {
  type: "call";
  id: number;
  token: string;
  name: string;
  args: unknown;
}
export interface ClosedMessage {
  type: "closed";
}
export type FromWorker = RegisteredMessage | FailedMessage | ReadyMessage | CallMessage | ClosedMessage;

// --- Restated on purpose (see the header) -------------------------------------------

const SDK_SERVER = "@modelcontextprotocol/sdk/server/index.js";
const SDK_TYPES = "@modelcontextprotocol/sdk/types.js";
const SDK_HTTP = "@modelcontextprotocol/sdk/server/streamableHttp.js";

/** Length-independent, early-exit-free comparison — the same one `mcpProtocol` uses, for the same
 *  reason. */
function secretEquals(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// --- The minimal SDK surface, cast to rather than imported --------------------------

interface SdkServer {
  setRequestHandler(schema: unknown, handler: (request: { params: { name: string; arguments?: unknown } }) => unknown): void;
  connect(transport: unknown): Promise<void>;
  close(): Promise<void>;
}
interface SdkModules {
  Server: new (info: { name: string; version: string }, options: { capabilities: { tools: Record<string, never> } }) => SdkServer;
  ListToolsRequestSchema: unknown;
  CallToolRequestSchema: unknown;
  StreamableHTTPServerTransport: new (options: { sessionIdGenerator: undefined }) => {
    handleRequest(req: unknown, res: unknown): Promise<void>;
    close(): Promise<void>;
  };
}

// --- State ------------------------------------------------------------------------

interface Run {
  path: string;
  descriptors: unknown[];
  /** `ready` is posted ONCE per run: the CLI may list tools again later, and that is not news. */
  readySent: boolean;
}

const runs = new Map<string, Run>();
const pending = new Map<number, (result: unknown) => void>();
let calls = 0;
let listener: HttpServer | undefined;
let listening: Promise<number> | undefined;

const port = parentPort;
if (port === null) throw new Error("mcpBridgeWorker must run as a worker thread");
const post = (message: FromWorker): void => port.postMessage(message);

/**
 * Which registered run a request is for, or none.
 *
 * The same two checks `isAuthorizedBridgeRequest` makes — any `Origin` refuses, and the path must
 * equal a run's — done over EVERY registered run without an early exit, so the answer's timing says
 * nothing about which token came close.
 */
function runFor(req: IncomingMessage): { token: string; run: Run } | undefined {
  if (req.headers.origin !== undefined || req.url === undefined) return undefined;
  const path = req.url.split("?")[0] ?? "";
  let found: { token: string; run: Run } | undefined;
  for (const [token, run] of runs) if (secretEquals(path, run.path)) found = { token, run };
  return found;
}

/** Forward a `tools/call` to the parent and wait for what its impl answered. */
function forward(token: string, name: string, args: unknown): Promise<unknown> {
  return new Promise((resolve) => {
    const id = ++calls;
    pending.set(id, resolve);
    post({ type: "call", id, token, name, args });
  });
}

async function loadSdk(): Promise<SdkModules> {
  try {
    // Variable specifiers: the SDK is an optional peer, kept off the build-time graph.
    const [server, types, http] = await Promise.all([
      import(/* @vite-ignore */ SDK_SERVER) as Promise<Pick<SdkModules, "Server">>,
      import(/* @vite-ignore */ SDK_TYPES) as Promise<Pick<SdkModules, "ListToolsRequestSchema" | "CallToolRequestSchema">>,
      import(/* @vite-ignore */ SDK_HTTP) as Promise<Pick<SdkModules, "StreamableHTTPServerTransport">>,
    ]);
    return {
      Server: server.Server,
      ListToolsRequestSchema: types.ListToolsRequestSchema,
      CallToolRequestSchema: types.CallToolRequestSchema,
      StreamableHTTPServerTransport: http.StreamableHTTPServerTransport,
    };
  } catch {
    throw new Error("@modelcontextprotocol/sdk is not installed — the bridge worker cannot serve without it");
  }
}

/** Bind the one listener, on first use. Bound explicitly to 127.0.0.1 on an ephemeral port, kept
 *  for the life of the worker. */
function listen(): Promise<number> {
  listening ??= (async () => {
    const sdk = await loadSdk();
    const server = createServer((req, res) => {
      void handle(sdk, req, res);
    });
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const address = server.address();
    if (address === null || typeof address === "string") {
      server.close();
      throw new Error("the MCP bridge worker could not determine its own port");
    }
    listener = server;
    return address.port;
  })();
  return listening;
}

async function handle(sdk: SdkModules, req: IncomingMessage, res: ServerResponse): Promise<void> {
  // FIRST, before a Server is built or a byte of body is read: an unauthenticated caller learns only
  // that there is nothing at this path.
  const hit = runFor(req);
  if (hit === undefined) {
    res.writeHead(404).end();
    return;
  }
  const { token, run } = hit;
  // A fresh server AND transport per request — stateless mode forbids reusing either.
  const server = new sdk.Server({ name: "declarative-ai", version: "0.1.0" }, { capabilities: { tools: {} } });
  server.setRequestHandler(sdk.ListToolsRequestSchema, () => {
    if (!run.readySent) {
      run.readySent = true;
      post({ type: "ready", token });
    }
    return { tools: run.descriptors };
  });
  server.setRequestHandler(sdk.CallToolRequestSchema, (request) => forward(token, request.params.name, request.params.arguments));
  const transport = new sdk.StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => {
    void transport.close();
    void server.close();
  });
  try {
    await server.connect(transport);
    await transport.handleRequest(req, res);
  } catch {
    if (!res.headersSent) res.writeHead(500).end();
  }
}

port.on("message", (message: ToWorker) => {
  switch (message.type) {
    case "register": {
      runs.set(message.token, { path: message.path, descriptors: message.descriptors, readySent: false });
      listen().then(
        (p) => post({ type: "registered", token: message.token, port: p }),
        (e: unknown) => {
          runs.delete(message.token);
          post({ type: "failed", token: message.token, message: e instanceof Error ? e.message : String(e) });
        },
      );
      return;
    }
    case "unregister":
      runs.delete(message.token);
      return;
    case "result": {
      const resolve = pending.get(message.id);
      pending.delete(message.id);
      resolve?.(message.result);
      return;
    }
    case "close": {
      runs.clear();
      const server = listener;
      listener = undefined;
      if (server === undefined) {
        post({ type: "closed" });
        return;
      }
      server.close(() => post({ type: "closed" }));
      // Any keep-alive connection an agent left open would hold the listener open otherwise.
      server.closeAllConnections();
      return;
    }
  }
});
