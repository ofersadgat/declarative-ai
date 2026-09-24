/**
 * A host's own MCP servers, CONNECTED — the client half of the bridge's proxy.
 *
 * Why a proxy at all: a transport with no permission callback (codex) calls an MCP server it is handed
 * ITSELF, so a call to one of its tools never crosses anything the host controls, and nobody can be
 * asked about it. Handing codex the server anyway left the host two answers, both wrong: approve every
 * tool on codex's side (an `ask` never asks), or take away every tool a person would have to be asked
 * about (an `ask` is a `deny`). So the bridge stands in for the server instead. This module connects to
 * the REAL server — the MCP SDK's `Client`, over stdio for a command, over streamable HTTP falling back
 * to SSE for an address — lists its tools, and forwards a call once the bridge's gate has allowed it;
 * the bridge serves the listing under the server's own name, so the agent sees the same
 * `mcp__<server>__<tool>` it always did (see `handleServerToolCall` in `./mcpProtocol`).
 *
 * It runs where the SPEC is — the host's thread — never in the bridge worker: the connection is a
 * resource of the run (a child process, for stdio), and the worker holds nothing of a run but its
 * path and its descriptors.
 *
 * The SDK is the same OPTIONAL peer the bridge already loads, reached by variable specifiers for the
 * same reason: this package type-checks, and its tests run, without it.
 */
import type { JsonValue, McpServerSpec, McpToolDescriptor } from "./deps.js";
import type { McpProxy } from "./mcpProtocol.js";

/** Connect to ONE server and list its tools — the seam a test stands in for. `signal` aborts a
 *  connection still being made. */
export type ConnectMcpServer = (name: string, spec: McpServerSpec, options: ConnectMcpOptions) => Promise<McpProxy>;

export interface ConnectMcpOptions {
  signal: AbortSignal;
  /** How long one forwarded call may take before the proxy gives up on it, in ms. */
  callTimeoutMs: number;
}

/** The minimal SDK client surface used here — cast to, never imported. */
interface SdkClient {
  connect(transport: unknown, options?: { signal?: AbortSignal }): Promise<void>;
  listTools(params?: { cursor?: string }, options?: { signal?: AbortSignal }): Promise<{ tools: Array<Record<string, unknown>>; nextCursor?: string }>;
  request(request: { method: string; params: Record<string, unknown> }, schema: unknown, options?: { timeout?: number; maxTotalTimeout?: number }): Promise<unknown>;
  close(): Promise<void>;
}
interface SdkClientModules {
  Client: new (info: { name: string; version: string }, options: { capabilities: Record<string, never> }) => SdkClient;
  StdioClientTransport: new (params: { command: string; args?: string[]; env?: Record<string, string>; cwd?: string; stderr?: "pipe" }) => {
    stderr: NodeJS.ReadableStream | null;
  };
  StreamableHTTPClientTransport: new (url: URL, options?: { requestInit?: { headers?: Record<string, string> } }) => unknown;
  SSEClientTransport: new (url: URL, options?: { requestInit?: { headers?: Record<string, string> } }) => unknown;
  CallToolResultSchema: unknown;
}

const SDK_CLIENT = "@modelcontextprotocol/sdk/client/index.js";
const SDK_STDIO = "@modelcontextprotocol/sdk/client/stdio.js";
const SDK_HTTP = "@modelcontextprotocol/sdk/client/streamableHttp.js";
const SDK_SSE = "@modelcontextprotocol/sdk/client/sse.js";
const SDK_TYPES = "@modelcontextprotocol/sdk/types.js";

async function loadClientSdk(): Promise<SdkClientModules> {
  try {
    // Variable specifiers: the SDK is an optional peer, kept off the build-time graph.
    const [client, stdio, http, sse, types] = await Promise.all([
      import(/* @vite-ignore */ SDK_CLIENT),
      import(/* @vite-ignore */ SDK_STDIO),
      import(/* @vite-ignore */ SDK_HTTP),
      import(/* @vite-ignore */ SDK_SSE),
      import(/* @vite-ignore */ SDK_TYPES),
    ]);
    return {
      Client: client.Client,
      StdioClientTransport: stdio.StdioClientTransport,
      StreamableHTTPClientTransport: http.StreamableHTTPClientTransport,
      SSEClientTransport: sse.SSEClientTransport,
      CallToolResultSchema: types.CallToolResultSchema,
    };
  } catch {
    throw new Error("@modelcontextprotocol/sdk is not installed — it is required to proxy an MCP server through the bridge");
  }
}

/** A page cap on `tools/list`: a server that always answers with a cursor would otherwise hold the
 *  connection open until the startup deadline. */
const MAX_TOOL_PAGES = 50;

/** How much of a stdio server's stderr is kept to say why it failed. */
const STDERR_TAIL = 2000;

/** One listed tool as the bridge serves it: the four fields an agent reads, VERBATIM. */
function descriptorOf(tool: Record<string, unknown>): McpToolDescriptor {
  return {
    name: String(tool["name"]),
    ...(typeof tool["description"] === "string" ? { description: tool["description"] } : {}),
    inputSchema: (tool["inputSchema"] ?? { type: "object" }) as JsonValue,
    ...(tool["annotations"] !== undefined ? { annotations: tool["annotations"] as JsonValue } : {}),
  };
}

/**
 * The server's own error, as the agent should see it. The SDK's `McpError` prefixes its message with
 * `MCP error <code>: `; the bridge's MCP server would print that prefix again when it serializes the
 * error, so it is taken off here and the code and data carried beside it, where the serializer reads
 * them (`code`, `message`, `data`).
 */
function serverError(e: unknown): Error {
  if (!(e instanceof Error)) return new Error(String(e));
  const code = (e as { code?: unknown }).code;
  if (typeof code !== "number") return e;
  const data = (e as { data?: unknown }).data;
  return Object.assign(new Error(e.message.replace(/^MCP error -?\d+: /, "")), { code, ...(data !== undefined ? { data } : {}) });
}

/** Connect a client over one transport and list every page of its tools. Closes the client on failure. */
async function connectOver(sdk: SdkClientModules, transport: unknown, signal: AbortSignal): Promise<{ client: SdkClient; tools: McpToolDescriptor[] }> {
  const client = new sdk.Client({ name: "declarative-ai-bridge", version: "0.1.0" }, { capabilities: {} });
  try {
    await client.connect(transport, { signal });
    const tools: McpToolDescriptor[] = [];
    let cursor: string | undefined;
    for (let page = 0; page < MAX_TOOL_PAGES; page++) {
      const answer = await client.listTools(cursor !== undefined ? { cursor } : undefined, { signal });
      tools.push(...answer.tools.map(descriptorOf));
      cursor = typeof answer.nextCursor === "string" && answer.nextCursor.length > 0 ? answer.nextCursor : undefined;
      if (cursor === undefined) break;
    }
    return { client, tools };
  } catch (e) {
    await client.close().catch(() => undefined);
    throw e;
  }
}

/**
 * The real connection: the MCP SDK's client, over stdio for a command (the server's environment is the
 * SDK's safe default one plus the spec's own, which the SDK merges), or over streamable HTTP falling
 * back to SSE for an older server.
 *
 * A call is forwarded as a raw `tools/call` request rather than through `callTool`, which would check a
 * result's structured content against the tool's output schema and refuse it on the server's behalf:
 * the result comes back UNCHANGED, and judging it is the agent's business.
 */
export const defaultConnectMcpServer: ConnectMcpServer = async (_name, spec, options) => {
  const sdk = await loadClientSdk();
  let stderr = "";
  let connected: { client: SdkClient; tools: McpToolDescriptor[] };
  if ("command" in spec) {
    const transport = new sdk.StdioClientTransport({
      command: spec.command,
      ...(spec.args !== undefined ? { args: spec.args } : {}),
      ...(spec.env !== undefined ? { env: spec.env } : {}),
      ...(spec.cwd !== undefined ? { cwd: spec.cwd } : {}),
      // Piped and READ: a pipe nobody drains fills, and a server blocked writing to it stops answering.
      // What it said is kept to name the failure when it does not start.
      stderr: "pipe",
    });
    transport.stderr?.on("data", (chunk: Buffer | string) => {
      stderr = (stderr + chunk.toString()).slice(-STDERR_TAIL);
    });
    try {
      connected = await connectOver(sdk, transport, options.signal);
    } catch (e) {
      const said = stderr.trim();
      throw new Error(`${e instanceof Error ? e.message : String(e)}${said.length > 0 ? ` — it said: ${said}` : ""}`);
    }
  } else {
    const init = { requestInit: { headers: spec.headers ?? {} } };
    try {
      connected = await connectOver(sdk, new sdk.StreamableHTTPClientTransport(new URL(spec.url), init), options.signal);
    } catch (first) {
      if (options.signal.aborted) throw first;
      // An older server speaks only SSE. If it fails too, the first failure is the one worth reading — a
      // streamable server that refused says why, an SSE retry would not.
      try {
        connected = await connectOver(sdk, new sdk.SSEClientTransport(new URL(spec.url), init), options.signal);
      } catch {
        throw first;
      }
    }
  }
  const { client, tools } = connected;
  return {
    tools,
    call: async (tool, args) => {
      try {
        // `maxTotalTimeout` too: a server that keeps reporting progress must not hold a call past the bound.
        return await client.request({ method: "tools/call", params: { name: tool, arguments: args } }, sdk.CallToolResultSchema, {
          timeout: options.callTimeoutMs,
          maxTotalTimeout: options.callTimeoutMs,
        });
      } catch (e) {
        throw serverError(e);
      }
    },
    close: () => client.close().catch(() => undefined),
  };
};

/**
 * Connect every server side by side, each under the one deadline, and list its tools.
 *
 * All or nothing — the transport marks each server `required`, so a run the agent would start without
 * one of them is a run the host did not configure: the first server that cannot start or list fails
 * the whole connection, NAMED, and every server that did connect is closed again before the throw.
 */
export async function connectMcpServers(
  servers: Record<string, McpServerSpec>,
  connect: ConnectMcpServer,
  options: { timeoutMs: number; callTimeoutMs: number },
): Promise<Record<string, McpProxy>> {
  const controller = new AbortController();
  const seconds = (ms: number): string => (ms >= 1000 ? `${Math.round(ms / 1000)} s` : `${ms} ms`);
  const settled = await Promise.allSettled(
    Object.entries(servers).map(async ([name, spec]): Promise<[string, McpProxy]> => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const deadline = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`it did not list its tools within ${seconds(options.timeoutMs)}`)), options.timeoutMs);
      });
      const connecting = connect(name, spec, { signal: controller.signal, callTimeoutMs: options.callTimeoutMs });
      try {
        return [name, await Promise.race([connecting, deadline])];
      } catch (e) {
        // A connection that lands after its deadline is closed, not leaked.
        void connecting.then((late) => late.close(), () => undefined);
        throw new Error(`the MCP server '${name}' could not start: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }
    }),
  );
  const proxies = Object.fromEntries(settled.flatMap((s) => (s.status === "fulfilled" ? [s.value] : [])));
  const failed = settled.find((s): s is PromiseRejectedResult => s.status === "rejected");
  if (failed !== undefined) {
    controller.abort();
    await closeMcpProxies(proxies);
    throw failed.reason;
  }
  return proxies;
}

/** Close every connection — ended, never thrown out of: a server that fails to close is gone either way. */
export async function closeMcpProxies(proxies: Record<string, McpProxy>): Promise<void> {
  await Promise.all(Object.values(proxies).map((proxy) => proxy.close().catch(() => undefined)));
}
