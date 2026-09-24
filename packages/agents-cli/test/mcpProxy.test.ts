/**
 * The bridge's PROXY, against a REAL MCP server — a small stdio server started with the SDK's own
 * server half, the way a host's configured server is — and a real bridge, both the per-run listener and
 * the worker-thread host.
 *
 * What is pinned: the real server's tools reach the agent verbatim under the server's own path; a call
 * the gate allows is forwarded and its result comes back unchanged; one it refuses is answered
 * `PermissionDenied` and the real tool NEVER runs (the fixture's tool leaves a file behind, so "never
 * ran" is observable); one the gate parks waits there, with the tool not run, until a person answers;
 * and the real server's own error reaches the agent as that error.
 */
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import type { AgentPermissionDecision, AgentToolRequest } from "@declarative-ai/agents-api";
import { defaultStartMcpBridge, type McpBridge, type McpBridgeSpec } from "../src/mcpBridge.js";
import { createMcpBridgeHost, type McpBridgeHost } from "../src/mcpBridgeHost.js";
import { bridgeServerPath, type McpProxy } from "../src/mcpProtocol.js";
import { closeMcpProxies, connectMcpServers, defaultConnectMcpServer } from "../src/mcpProxy.js";

const sdkInstalled = await import("@modelcontextprotocol/sdk/client/index.js").then(
  () => true,
  () => false,
);

const workerFile = new URL("../src/mcpBridgeWorker.ts", import.meta.url);

/** One JSON-RPC call over streamable HTTP. */
async function rpc(url: string, method: string, params?: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params: params ?? {} }),
  });
  const text = await response.text();
  const data = text
    .split("\n")
    .find((l) => l.startsWith("data:"))
    ?.slice("data:".length);
  return JSON.parse((data ?? text).trim()) as Record<string, unknown>;
}

const INITIALIZE = { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "test", version: "1.0.0" } };

async function callTool(url: string, name: string, args: unknown): Promise<Record<string, unknown>> {
  await rpc(url, "initialize", INITIALIZE);
  return rpc(url, "tools/call", { name, arguments: args });
}

let root = "";
let serverFile = "";

beforeAll(() => {
  if (!sdkInstalled) return;
  const require = createRequire(import.meta.url);
  const at = (path: string): string => JSON.stringify(require.resolve(`@modelcontextprotocol/sdk/${path}`));
  root = mkdtempSync(join(tmpdir(), "mcp-proxy-"));
  serverFile = join(root, "server.cjs");
  writeFileSync(
    serverFile,
    [
      `const { writeFileSync } = require("node:fs");`,
      `const { Server } = require(${at("server/index.js")});`,
      `const { StdioServerTransport } = require(${at("server/stdio.js")});`,
      `const { ListToolsRequestSchema, CallToolRequestSchema } = require(${at("types.js")});`,
      `const server = new Server({ name: "fixture", version: "1" }, { capabilities: { tools: {} } });`,
      `server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [`,
      `  { name: "echo", description: "Say it back", inputSchema: { type: "object", properties: { text: { type: "string" } } }, annotations: { readOnlyHint: true, title: "Echo" } },`,
      `  { name: "touch", description: "Leave a file", inputSchema: { type: "object", properties: { path: { type: "string" } } }, annotations: { destructiveHint: true } },`,
      `  { name: "bad", description: "Always refuses", inputSchema: { type: "object" } },`,
      `] }));`,
      `server.setRequestHandler(CallToolRequestSchema, async (request) => {`,
      `  const args = request.params.arguments || {};`,
      `  if (request.params.name === "echo") return { content: [{ type: "text", text: "echo:" + args.text + ":" + (process.env.PROXY_ENV || "") }], structuredContent: { echoed: args.text } };`,
      `  if (request.params.name === "touch") { writeFileSync(args.path, "ran"); return { content: [{ type: "text", text: "touched" }] }; }`,
      `  throw Object.assign(new Error("no such node"), { code: -32602, data: { node: "1:2" } });`,
      `});`,
      `server.connect(new StdioServerTransport());`,
    ].join("\n"),
  );
});

afterAll(() => {
  if (root !== "") rmSync(root, { recursive: true, force: true });
});

const fixture = () => ({ fixture: { command: process.execPath, args: [serverFile], env: { PROXY_ENV: "seen" } } });

/** A gate whose answers the test decides, and which records what it was asked. */
function gate() {
  const asked: AgentToolRequest[] = [];
  let answer: (req: AgentToolRequest) => Promise<AgentPermissionDecision> = async () => ({ allow: true });
  return {
    asked,
    answer: (next: typeof answer) => {
      answer = next;
    },
    serverToolGate: (req: AgentToolRequest) => {
      asked.push(req);
      return answer(req);
    },
  };
}

describe.skipIf(!sdkInstalled)("connecting a real server", () => {
  it("lists its tools VERBATIM — annotations included — and forwards a call, the result unchanged", async () => {
    const proxies = await connectMcpServers(fixture(), defaultConnectMcpServer, { timeoutMs: 20_000, callTimeoutMs: 20_000 });
    try {
      expect(proxies["fixture"]!.tools).toEqual([
        { name: "echo", description: "Say it back", inputSchema: { type: "object", properties: { text: { type: "string" } } }, annotations: { readOnlyHint: true, title: "Echo" } },
        { name: "touch", description: "Leave a file", inputSchema: { type: "object", properties: { path: { type: "string" } } }, annotations: { destructiveHint: true } },
        { name: "bad", description: "Always refuses", inputSchema: { type: "object" } },
      ]);
      // The server's environment is the spec's, on top of the SDK's safe default.
      expect(await proxies["fixture"]!.call("echo", { text: "hi" })).toEqual({ content: [{ type: "text", text: "echo:hi:seen" }], structuredContent: { echoed: "hi" } });
      // The server's own error, its code and data carried and its message as the server said it.
      await expect(proxies["fixture"]!.call("bad", {})).rejects.toMatchObject({ message: "no such node", code: -32602, data: { node: "1:2" } });
    } finally {
      await closeMcpProxies(proxies);
    }
  });

  it("fails naming a server that cannot start", async () => {
    await expect(
      connectMcpServers({ ...fixture(), missing: { command: join(root, "no-such-binary") } }, defaultConnectMcpServer, { timeoutMs: 20_000, callTimeoutMs: 1000 }),
    ).rejects.toThrow(/^the MCP server 'missing' could not start: /);
  });
});

/** The same four behaviours, through whichever bridge serves the run. */
function throughTheBridge(label: string, start: () => (spec: McpBridgeSpec) => Promise<McpBridge>) {
  describe.skipIf(!sdkInstalled)(`a proxied server, through ${label}`, () => {
    let proxies: Record<string, McpProxy> = {};
    let bridge: McpBridge | undefined;
    afterEach(async () => {
      await bridge?.close();
      bridge = undefined;
      await closeMcpProxies(proxies);
      proxies = {};
    });
    const serve = async (g: ReturnType<typeof gate>): Promise<string> => {
      proxies = await connectMcpServers(fixture(), defaultConnectMcpServer, { timeoutMs: 20_000, callTimeoutMs: 20_000 });
      bridge = await start()({ servers: proxies, serverToolGate: g.serverToolGate });
      return bridgeServerPath(bridge.url, "fixture");
    };

    it("serves the real server's tools at its own path, and forwards what the gate ALLOWS", async () => {
      const g = gate();
      const url = await serve(g);
      await rpc(url, "initialize", INITIALIZE);
      const listed = (await rpc(url, "tools/list")).result as { tools: Array<{ name: string; annotations?: unknown }> };
      expect(listed.tools.map((t) => t.name)).toEqual(["echo", "touch", "bad"]);
      expect(listed.tools[0]!.annotations).toEqual({ readOnlyHint: true, title: "Echo" });
      expect((await callTool(url, "echo", { text: "hi" })).result).toEqual({ content: [{ type: "text", text: "echo:hi:seen" }], structuredContent: { echoed: "hi" } });
      expect(g.asked).toEqual([{ toolName: "mcp__fixture__echo", input: { text: "hi" } }]);
      // The bridge's own path serves nothing of the server's.
      await rpc(bridge!.url, "initialize", INITIALIZE);
      expect(((await rpc(bridge!.url, "tools/list")).result as { tools: unknown[] }).tools).toEqual([]);
    });

    it("answers a DENY as PermissionDenied, and the real tool never runs", async () => {
      const g = gate();
      g.answer(async () => ({ allow: false, reason: "not in this state's permission set" }));
      const url = await serve(g);
      const marker = join(root, `denied-${label.replace(/\W/g, "")}`);
      const answered = (await callTool(url, "touch", { path: marker })).result as { content: Array<{ text: string }>; isError?: boolean };
      expect(JSON.parse(answered.content[0]!.text)).toEqual({ denied: true, tool: "mcp__fixture__touch", reason: "not in this state's permission set" });
      expect(existsSync(marker)).toBe(false);
    });

    it("PARKS an ask on the approver — the tool waits, unrun, until a person answers", async () => {
      const g = gate();
      let decide: (d: AgentPermissionDecision) => void = () => undefined;
      let reached: () => void = () => undefined;
      const parked = new Promise<void>((resolve) => (reached = resolve));
      g.answer(
        () =>
          new Promise<AgentPermissionDecision>((resolve) => {
            decide = resolve;
            reached();
          }),
      );
      const url = await serve(g);
      const marker = join(root, `asked-${label.replace(/\W/g, "")}`);
      await rpc(url, "initialize", INITIALIZE);
      const call = rpc(url, "tools/call", { name: "touch", arguments: { path: marker } });
      await parked;
      await new Promise((r) => setTimeout(r, 100));
      expect(existsSync(marker)).toBe(false);
      decide({ allow: true });
      expect(((await call).result as { content: Array<{ text: string }> }).content[0]!.text).toBe("touched");
      expect(existsSync(marker)).toBe(true);
    });

    it("hands the real server's ERROR to the agent as that error", async () => {
      const url = await serve(gate());
      expect((await callTool(url, "bad", {})).error).toEqual({ code: -32602, message: "no such node", data: { node: "1:2" } });
    });
  });
}

throughTheBridge("the per-run bridge", () => defaultStartMcpBridge);

let host: McpBridgeHost | undefined;
afterAll(async () => {
  await host?.close();
});
throughTheBridge("the worker-thread host", () => {
  host ??= createMcpBridgeHost({ workerFile });
  return host.start;
});
