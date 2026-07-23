/**
 * The MCP server this package hosts so a `claude` SUBPROCESS can call back into us — for a permission
 * decision before each gated tool-use, and to run host-implemented tools.
 *
 * All the wire shapes live in `./mcpProtocol`, which is pure and fully tested. This module is only the
 * transport: an HTTP listener on loopback, and the SDK plumbing that routes a request into
 * `handleToolCall`. It is the one part that cannot be exercised without the SDK installed, so it is
 * kept as small as it can be and sits behind the injectable {@link StartMcpBridge} seam — the same
 * pattern `agents-api` uses for the Agent SDK.
 *
 * The SDK is an OPTIONAL peer dependency, loaded through a variable specifier so this package
 * type-checks and its tests run without it.
 *
 * Two SDK behaviours the implementation below depends on, both verified against 1.29.0 rather than
 * assumed:
 *
 *  - **Stateless mode requires a NEW `Server` AND a new transport per request.** Reusing either throws
 *    ("Stateless transport cannot be reused across requests" / "Already connected to a transport"),
 *    and the symptom is the SECOND request failing — which is the CLI's `notifications/initialized`,
 *    so it looks like `connect()` broke rather than reuse.
 *  - **The low-level `Server` does not validate tool arguments at all.** It advertises our JSON Schema
 *    to the client and then hands whatever arrives straight to the handler. The high-level
 *    `registerTool` does validate, but it takes Zod schemas and ours are JSON Schema documents written
 *    by workflow authors. So validation is ours to do, and it lives in `handleToolCall`: an injected
 *    tool's arguments are checked against its own `inputSchema` through the `OutputValidator` seam
 *    (`@declarative-ai/json`'s three lines — deliberately not `@declarative-ai/validate`, so ajv stays
 *    off the agent path) whenever the caller supplies one. `agents-api`'s adapter supplies `ctx.validator`
 *    automatically, so the engine-driven path is checked without anyone opting in.
 */
import type { AgentPermissionDecision, AgentToolRequest, InjectedTool, SyncOutputValidator } from "./deps";
import { bridgePath, handleToolCall, isAuthorizedBridgeRequest, newBridgeToken, toolDescriptors } from "./mcpProtocol";

/** What a bridge exposes to the CLI. */
export interface McpBridge {
  /** The `--mcp-config` URL the agent connects to. Carries the run's secret in its path — treat it as a
   *  credential, not as an address: anything that learns it can drive every tool this bridge serves. */
  url: string;
  close(): Promise<void>;
}

/** What the bridge serves. */
export interface McpBridgeSpec {
  /** Host-implemented tools the agent may call. */
  tools?: Record<string, InjectedTool>;
  /** The approver a permission ask routes to. Absent ⇒ no approval tool is exposed. */
  approve?: (req: AgentToolRequest) => Promise<AgentPermissionDecision>;
  /** Checks an injected tool's arguments against its own `inputSchema` before the impl sees them.
   *  Absent ⇒ unvalidated, as everywhere else a seam is optional — so the caller that owns the tools is
   *  the one that decides. */
  validator?: SyncOutputValidator;
}

/** The injectable seam: stand up a server for this spec. Tests inject a fake; the default is below. */
export type StartMcpBridge = (spec: McpBridgeSpec) => Promise<McpBridge>;

/** The minimal SDK surface used here — cast to, never imported, so the missing optional dependency
 *  stays off the type graph. */
interface SdkServer {
  setRequestHandler(schema: unknown, handler: (request: { params: { name: string; arguments?: unknown } }) => unknown): void;
  connect(transport: unknown): Promise<void>;
  close(): Promise<void>;
}
interface SdkServerModule {
  Server: new (info: { name: string; version: string }, options: { capabilities: { tools: Record<string, never> } }) => SdkServer;
}
interface SdkTypesModule {
  ListToolsRequestSchema: unknown;
  CallToolRequestSchema: unknown;
}
interface SdkHttpModule {
  StreamableHTTPServerTransport: new (options: { sessionIdGenerator: undefined }) => {
    handleRequest(req: unknown, res: unknown): Promise<void>;
    close(): Promise<void>;
  };
}

const SDK_SERVER = "@modelcontextprotocol/sdk/server/index.js";
const SDK_TYPES = "@modelcontextprotocol/sdk/types.js";
const SDK_HTTP = "@modelcontextprotocol/sdk/server/streamableHttp.js";

/** The message a caller sees when the optional dependency is missing. Exported so the refusal path can
 *  name the remedy without duplicating the string. */
export const SDK_MISSING =
  `@modelcontextprotocol/sdk is not installed — it is required to route a CLI agent's tool permissions back to your approver. ` +
  `Install it, or inject a \`startBridge\` seam, or run the agent with no approver and an up-front permission posture.`;

/**
 * The real bridge: a loopback HTTP listener speaking stateless streamable-HTTP MCP.
 *
 * Bound EXPLICITLY to 127.0.0.1 on an ephemeral port. Loopback because the agent is a child process on
 * this machine; ephemeral because a fixed port would collide between concurrent runs.
 *
 * Loopback is NOT the authorization story, and treating it as one made this an unauthenticated server
 * that executes host tools: every other process on the machine is also on loopback, an ephemeral port is
 * a short search, and `tools/call` reaches the impl without the approver being consulted (approval is
 * the CLI's job, on the far side of this port). So every request must carry the run's
 * {@link newBridgeToken} in its path and must be free of an `Origin` header, checked BEFORE anything is
 * dispatched; a failure is a bare 404 that says nothing about which check it failed or that a server is
 * even here.
 */
export const defaultStartMcpBridge: StartMcpBridge = async (spec) => {
  let serverModule: SdkServerModule;
  let typesModule: SdkTypesModule;
  let httpModule: SdkHttpModule;
  try {
    // Variable specifiers: TS will not resolve (or require) these at build time — they are optional.
    serverModule = (await import(/* @vite-ignore */ SDK_SERVER)) as unknown as SdkServerModule;
    typesModule = (await import(/* @vite-ignore */ SDK_TYPES)) as unknown as SdkTypesModule;
    httpModule = (await import(/* @vite-ignore */ SDK_HTTP)) as unknown as SdkHttpModule;
  } catch {
    throw new Error(SDK_MISSING);
  }
  const http = await import("node:http");

  const descriptors = toolDescriptors(spec);
  const buildServer = (): SdkServer => {
    const server = new serverModule.Server({ name: "declarative-ai", version: "0.1.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(typesModule.ListToolsRequestSchema, () => ({ tools: descriptors }));
    server.setRequestHandler(typesModule.CallToolRequestSchema, (request) =>
      handleToolCall(spec, request.params.name, request.params.arguments),
    );
    return server;
  };

  // One secret per bridge, i.e. per run: it dies with the listener, so a leaked URL from a finished run
  // authorizes nothing.
  const token = newBridgeToken();

  const listener = http.createServer((req, res) => {
    void (async () => {
      // FIRST, before a Server is built or a byte of body is read. An unauthenticated caller learns only
      // that there is nothing at this path.
      if (!isAuthorizedBridgeRequest({ url: req.url, origin: req.headers.origin }, token)) {
        res.writeHead(404).end();
        return;
      }
      // A fresh server AND transport per request — stateless mode forbids reusing either.
      const server = buildServer();
      const transport = new httpModule.StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
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
    })();
  });

  await new Promise<void>((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(0, "127.0.0.1", resolve);
  });
  const address = listener.address();
  if (address === null || typeof address === "string") {
    listener.close();
    throw new Error("the MCP bridge could not determine its own port");
  }

  return {
    url: `http://127.0.0.1:${address.port}${bridgePath(token)}`,
    close: () =>
      new Promise<void>((resolve) => {
        listener.close(() => resolve());
        // Any keep-alive connection the agent left open would hold the process otherwise.
        listener.closeAllConnections?.();
      }),
  };
};
