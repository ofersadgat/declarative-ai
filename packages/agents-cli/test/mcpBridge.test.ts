/**
 * The MCP bridge, driven over REAL HTTP against the real SDK.
 *
 * `mcpProtocol.test.ts` pins the wire shapes; this pins the transport — that a `claude` subprocess
 * pointed at our `--mcp-config` URL can actually complete a handshake, discover the approval tool, and
 * get a decision back. Without it, the one part of the permission path that cannot be unit-tested is
 * also the part that has to work for any of it to mean anything.
 *
 * The SDK is an optional PEER dependency for consumers and a DEV dependency here, precisely so this
 * test can exist. It is skipped rather than failed when the SDK is absent, so the suite still runs in
 * an install that declined it.
 */
import { afterEach, describe, expect, it } from "vitest";
import { defaultStartMcpBridge, type McpBridge } from "../src/mcpBridge";
import { APPROVAL_TOOL, mcpConfigJson } from "../src/mcpProtocol";

const sdkInstalled = await import("@modelcontextprotocol/sdk/server/index.js").then(
  () => true,
  () => false,
);

/** One JSON-RPC call over streamable HTTP. The `Accept` header matters: a plain `application/json`
 *  Accept is rejected by the transport. */
async function rpc(url: string, method: string, params?: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params: params ?? {} }),
  });
  const text = await response.text();
  // Streamable HTTP answers as SSE (`event: message` / `data: {…}`), so pull the payload off the
  // `data:` line; a plain JSON body is returned as-is.
  const data = text
    .split("\n")
    .find((l) => l.startsWith("data:"))
    ?.slice("data:".length);
  return JSON.parse((data ?? text).trim()) as Record<string, unknown>;
}

const INITIALIZE = { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "test", version: "1.0.0" } };

/** Complete the handshake, then make one call — the sequence a real client performs. */
async function callTool(url: string, name: string, args: unknown): Promise<{ content: Array<{ text: string }>; isError?: boolean }> {
  await rpc(url, "initialize", INITIALIZE);
  const response = await rpc(url, "tools/call", { name, arguments: args });
  return response.result as { content: Array<{ text: string }>; isError?: boolean };
}

let bridge: McpBridge | undefined;
afterEach(async () => {
  await bridge?.close();
  bridge = undefined;
});

describe.skipIf(!sdkInstalled)("the MCP bridge over real HTTP", () => {
  it("serves a loopback URL bearing the run's secret, and completes a handshake", async () => {
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    // Loopback and an ephemeral port, plus a 256-bit per-run path secret — the port is not the
    // authorization, it is only the address.
    expect(bridge.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/mcp\/[0-9a-f]{64}$/);
    const initialized = await rpc(bridge.url, "initialize", INITIALIZE);
    expect((initialized.result as { serverInfo: { name: string } }).serverInfo.name).toBe("declarative-ai");
  });

  it("advertises the approval tool with its schema, and an injected tool's schema VERBATIM", async () => {
    const inputSchema = { type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false } as never;
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }), tools: { read_file: { inputSchema, run: () => null } } });
    await rpc(bridge.url, "initialize", INITIALIZE);
    const listed = await rpc(bridge.url, "tools/list");
    const tools = (listed.result as { tools: Array<{ name: string; inputSchema: unknown }> }).tools;
    expect(tools.map((t) => t.name).sort()).toEqual([APPROVAL_TOOL, "read_file"]);
    // The whole reason for the low-level server: our authored JSON Schema reaches the agent unchanged,
    // with no Zod conversion in the middle.
    expect(tools.find((t) => t.name === "read_file")!.inputSchema).toEqual(inputSchema);
  });

  it("returns an ALLOW that echoes updatedInput, which is what the CLI requires", async () => {
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    const result = await callTool(bridge.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "ls" }, tool_use_id: "tu_1" });
    expect(JSON.parse(result.content[0]!.text)).toEqual({ behavior: "allow", updatedInput: { command: "ls" } });
  });

  it("returns a DENY carrying the approver's reason", async () => {
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: false, reason: "outside the workspace" }) });
    const result = await callTool(bridge.url, APPROVAL_TOOL, { tool_name: "Write", input: { path: "/etc/passwd" } });
    expect(JSON.parse(result.content[0]!.text)).toMatchObject({ behavior: "deny", message: "outside the workspace" });
  });

  it("passes the real tool name and input to the approver", async () => {
    let seen: unknown;
    bridge = await defaultStartMcpBridge({
      approve: async (req) => {
        seen = req;
        return { allow: true };
      },
    });
    await callTool(bridge.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "rm -rf /" } });
    expect(seen).toEqual({ toolName: "Bash", input: { command: "rm -rf /" } });
  });

  it("runs an injected tool and returns its value", async () => {
    bridge = await defaultStartMcpBridge({
      tools: { add: { inputSchema: { type: "object" } as never, run: (i) => ({ sum: (i.a as number) + (i.b as number) }) } },
    });
    const result = await callTool(bridge.url, "add", { a: 2, b: 3 });
    expect(JSON.parse(result.content[0]!.text)).toEqual({ sum: 5 });
  });

  it("serves MORE THAN ONE request — stateless mode needs a fresh server and transport each time", async () => {
    // Getting this wrong fails on the SECOND request, which in a real run is the CLI's
    // `notifications/initialized` — so it looks like connect() broke rather than reuse.
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    for (const command of ["ls", "pwd", "whoami"]) {
      const result = await callTool(bridge.url, APPROVAL_TOOL, { tool_name: "Bash", input: { command } });
      expect(JSON.parse(result.content[0]!.text)).toEqual({ behavior: "allow", updatedInput: { command } });
    }
  });

  it("DENIES a malformed permission ask over the wire, rather than erroring into an allow", async () => {
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    const result = await callTool(bridge.url, APPROVAL_TOOL, { nonsense: true });
    expect(JSON.parse(result.content[0]!.text).behavior).toBe("deny");
  });

  it("404s anything that is not the MCP path", async () => {
    bridge = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    const response = await fetch(bridge.url.replace("/mcp", "/elsewhere"), { method: "POST" });
    expect(response.status).toBe(404);
  });

  it("refuses to expose a host tool named `approve` — the name belongs to the permission gate", async () => {
    await expect(
      defaultStartMcpBridge({ approve: async () => ({ allow: true }), tools: { approve: { inputSchema: {} as never, run: () => "ran" } } }),
    ).rejects.toThrow(/reserved/);
  });
});

/**
 * THE attack, over real HTTP: an unrelated local process that has found the port.
 *
 * The bridge executes host tools (`@declarative-ai/tools` ships `run_command` and `write_file`) and the
 * approver lives on the FAR side of this port — the CLI is what asks it — so an unauthenticated
 * `tools/call` ran whatever the workflow wired, with an approver that denies everything, for the
 * lifetime of the run.
 */
describe.skipIf(!sdkInstalled)("an unauthenticated caller on loopback", () => {
  let ran: unknown[] = [];
  const spec = () => ({
    approve: async () => ({ allow: false as const, reason: "denied by policy" }),
    tools: {
      run_shell: {
        inputSchema: {} as never,
        run: (i: Record<string, unknown>) => {
          ran.push(i);
          return "executed";
        },
      },
    },
  });

  const post = (url: string, body: unknown, headers: Record<string, string> = {}) =>
    fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json, text/event-stream", ...headers },
      body: JSON.stringify(body),
    });

  const call = { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "run_shell", arguments: { cmd: "rm -rf /" } } };

  it.each([
    ["the bare /mcp path the bridge used to answer on", (url: string) => url.replace(/\/mcp\/[0-9a-f]+$/, "/mcp")],
    ["a guessed token", (url: string) => url.replace(/[0-9a-f]{64}$/, "0".repeat(64))],
    ["a truncated token", (url: string) => url.slice(0, -1)],
  ])("is refused at %s — and the tool does not run", async (_case, mangle) => {
    ran = [];
    bridge = await defaultStartMcpBridge(spec());
    const response = await post(mangle(bridge.url), call);
    expect(response.status).toBe(404);
    // The whole point: not "the call was denied" but "the impl was never reached".
    expect(ran).toEqual([]);
    // And the refusal says nothing — no server name, no hint that a token is what was missing.
    expect(await response.text()).toBe("");
  });

  it("is refused even WITH the right token when it carries an Origin (DNS rebinding)", async () => {
    ran = [];
    bridge = await defaultStartMcpBridge(spec());
    const response = await post(bridge.url, call, { origin: "https://evil.example" });
    expect(response.status).toBe(404);
    expect(ran).toEqual([]);
  });

  it("but the REAL path still works — the fix must not break the run it protects", async () => {
    ran = [];
    bridge = await defaultStartMcpBridge(spec());
    const result = await callTool(bridge.url, "run_shell", { cmd: "ls" });
    expect(result.content[0]!.text).toBe("executed");
    expect(ran).toEqual([{ cmd: "ls" }]);
  });

  it("hands the secret to the CLI through the `--mcp-config` it already consumes", async () => {
    // There is no second channel to keep in sync: the URL in the config document IS the credential.
    bridge = await defaultStartMcpBridge(spec());
    const config = JSON.parse(mcpConfigJson(bridge.url)) as { mcpServers: { dai: { url: string } } };
    expect(config.mcpServers.dai.url).toBe(bridge.url);
    const result = await callTool(config.mcpServers.dai.url, "run_shell", { cmd: "pwd" });
    expect(result.content[0]!.text).toBe("executed");
  });

  it("stops answering once closed — a leaked listener would outlive the run it gated", async () => {
    const closing = await defaultStartMcpBridge({ approve: async () => ({ allow: true }) });
    const url = closing.url;
    await closing.close();
    await expect(rpc(url, "initialize", INITIALIZE)).rejects.toThrow();
  });
});
