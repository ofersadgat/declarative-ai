/**
 * The persistent bridge host, driven over REAL HTTP against the real SDK — on a worker thread.
 *
 * `mcpBridge.test.ts` pins the per-run listener; this pins the shape that replaces it in a busy host:
 * one listener for every run, a run being a registration under its own path secret, calls routed to
 * the run's own spec on the host thread, and `ready` per run. The worker is loaded by PATH, so the
 * tests point at its source and Node runs it as-is — the file imports nothing of its own for exactly
 * this reason (see its header).
 */
import { afterEach, describe, expect, it } from "vitest";
import { createMcpBridgeHost, type McpBridgeHost } from "../src/mcpBridgeHost.js";
import { APPROVAL_TOOL } from "../src/mcpProtocol.js";
import { defaultStartMcpBridge, type McpBridge } from "../src/mcpBridge.js";

const sdkInstalled = await import("@modelcontextprotocol/sdk/server/index.js").then(
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

async function callTool(url: string, name: string, args: unknown): Promise<{ content: Array<{ text: string }>; isError?: boolean }> {
  await rpc(url, "initialize", INITIALIZE);
  const response = await rpc(url, "tools/call", { name, arguments: args });
  return response.result as { content: Array<{ text: string }>; isError?: boolean };
}

const allow = { approve: async () => ({ allow: true as const }) };

describe.skipIf(!sdkInstalled)("ready on the per-run bridge — the handshake's end, which is what the prompt waits for", () => {
  let bridge: McpBridge | undefined;
  afterEach(async () => {
    await bridge?.close();
    bridge = undefined;
  });

  it("settles on the agent's tools/list, not on connect", async () => {
    bridge = await defaultStartMcpBridge(allow);
    let settled = false;
    void bridge.ready!.then(() => {
      settled = true;
    });
    await rpc(bridge.url, "initialize", INITIALIZE);
    await new Promise((r) => setTimeout(r, 5));
    expect(settled).toBe(false);
    await rpc(bridge.url, "tools/list");
    await bridge.ready;
    expect(settled).toBe(true);
  });

  it("is released by close, so a waiter never holds a bridge that has gone", async () => {
    const closing = await defaultStartMcpBridge(allow);
    await closing.close();
    await expect(closing.ready).resolves.toBeUndefined();
  });
});

describe.skipIf(!sdkInstalled)("the persistent bridge host — one worker-thread listener for every run", () => {
  let host: McpBridgeHost | undefined;
  afterEach(async () => {
    await host?.close();
    host = undefined;
  });

  it("serves two runs on ONE port, each under its own secret, routing calls to its own spec", async () => {
    host = createMcpBridgeHost({ workerFile });
    const a = await host.start(allow);
    const b = await host.start({ approve: async () => ({ allow: false, reason: "b says no" }) });
    expect(new URL(a.url).port).toBe(new URL(b.url).port);
    expect(a.url).not.toBe(b.url);
    const fromA = await callTool(a.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "ls" } });
    const fromB = await callTool(b.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "ls" } });
    expect(JSON.parse(fromA.content[0]!.text)).toMatchObject({ behavior: "allow" });
    expect(JSON.parse(fromB.content[0]!.text)).toMatchObject({ behavior: "deny", message: "b says no" });
    await a.close();
    await b.close();
  });

  it("runs an injected tool on the HOST thread — the impl never crosses the port", async () => {
    host = createMcpBridgeHost({ workerFile });
    const seen: unknown[] = [];
    const run = await host.start({
      tools: {
        add: {
          inputSchema: { type: "object" } as never,
          run: (i) => {
            seen.push(i);
            return { sum: (i.a as number) + (i.b as number) };
          },
        },
      },
    });
    const result = await callTool(run.url, "add", { a: 2, b: 3 });
    expect(JSON.parse(result.content[0]!.text)).toEqual({ sum: 5 });
    expect(seen).toEqual([{ a: 2, b: 3 }]);
    await run.close();
  });

  it("advertises each run's own tools, and reports ready once a run's list has been served", async () => {
    host = createMcpBridgeHost({ workerFile });
    const run = await host.start({ ...allow, tools: { read_file: { inputSchema: { type: "object" } as never, run: () => null } } });
    let settled = false;
    void run.ready!.then(() => {
      settled = true;
    });
    await rpc(run.url, "initialize", INITIALIZE);
    await new Promise((r) => setTimeout(r, 5));
    expect(settled).toBe(false);
    const listed = await rpc(run.url, "tools/list");
    expect((listed.result as { tools: Array<{ name: string }> }).tools.map((t) => t.name).sort()).toEqual([APPROVAL_TOOL, "read_file"]);
    await run.ready;
    await run.close();
  });

  it("stops answering a closed run's path while the listener goes on serving the others", async () => {
    host = createMcpBridgeHost({ workerFile });
    const a = await host.start(allow);
    const b = await host.start(allow);
    await a.close();
    const response = await fetch(a.url, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
      body: "{}",
    });
    expect(response.status).toBe(404);
    const fromB = await callTool(b.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "ls" } });
    expect(JSON.parse(fromB.content[0]!.text)).toMatchObject({ behavior: "allow" });
    await b.close();
  });

  it("refuses an Origin-bearing request and an unknown path with the same bare 404", async () => {
    host = createMcpBridgeHost({ workerFile });
    const run = await host.start(allow);
    const withOrigin = await fetch(run.url, { method: "POST", headers: { origin: "https://evil.example", "content-type": "application/json" }, body: "{}" });
    expect(withOrigin.status).toBe(404);
    const elsewhere = await fetch(run.url.replace("/mcp/", "/mcp/0"), { method: "POST" });
    expect(elsewhere.status).toBe(404);
    await run.close();
  });

  it("refuses a host tool named `approve` before anything is registered", async () => {
    host = createMcpBridgeHost({ workerFile });
    await expect(host.start({ ...allow, tools: { approve: { inputSchema: {} as never, run: () => "ran" } } })).rejects.toThrow(/reserved/);
  });

  it("refuses to start a run once closed, and closing releases every run's ready", async () => {
    host = createMcpBridgeHost({ workerFile });
    const run = await host.start(allow);
    await host.close();
    await expect(run.ready).resolves.toBeUndefined();
    await expect(host.start(allow)).rejects.toThrow(/closed/);
    host = undefined;
  });

  it("refuses a run — rather than hanging it — when the worker file cannot be loaded", async () => {
    host = createMcpBridgeHost({ workerFile: new URL("./no-such-worker.js", import.meta.url) });
    await expect(host.start(allow)).rejects.toThrow();
  });
});
