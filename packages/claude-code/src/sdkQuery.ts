/**
 * The default {@link AgentQuery} — a THIN wrapper over `@anthropic-ai/claude-agent-sdk`'s `query()`. It is
 * loaded lazily (a variable module specifier, so this package type-checks and its fake-driven tests run
 * WITHOUT the SDK installed) and the SDK is an OPTIONAL peer dependency.
 *
 * ⚠️ UNVERIFIED against a live SDK build: the field names below (`type: "result"`, `.result`/`.text`,
 * `.total_cost_usd`, the `canUseTool` request/return shape) reflect the documented API and MUST be
 * confirmed against the installed `@anthropic-ai/claude-agent-sdk` version — adjust the two mapping spots if
 * they differ. All adapter LOGIC is tested via the injectable seam; only this boundary mapping is untested
 * here.
 */
import type { AgentQuery, AgentStreamMessage } from "./seam";

/** The minimal surface of the SDK we call — cast to, never imported (keeps the missing dep off the type graph). */
interface SdkModule {
  query(arg: { prompt: string; options?: Record<string, unknown> }): AsyncIterable<Record<string, unknown>>;
  /** In-process ("SDK") MCP server factory + tool builder — how custom tools are injected. */
  createSdkMcpServer?: (config: { name: string; tools: unknown[] }) => unknown;
  tool?: (name: string, description: string, inputSchema: unknown, handler: (input: Record<string, unknown>) => unknown) => unknown;
}

const SDK_SPECIFIER = "@anthropic-ai/claude-agent-sdk";

/** The in-process MCP server name our injected tools are exposed under (agent sees `mcp__dai__<tool>`). */
const MCP_SERVER = "dai";

export const sdkAgentQuery: AgentQuery = async function* (opts): AsyncIterable<AgentStreamMessage> {
  let sdk: SdkModule;
  try {
    // Variable specifier: TS won't resolve (or require) the module at build time — it's optional.
    sdk = (await import(/* @vite-ignore */ SDK_SPECIFIER)) as unknown as SdkModule;
  } catch {
    throw new Error(`${SDK_SPECIFIER} is not installed — install it, or inject a \`query\` seam into createClaudeCodeRuntime`);
  }

  const controller = new AbortController();
  if (opts.abortSignal) {
    if (opts.abortSignal.aborted) controller.abort();
    else opts.abortSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  // ⚠️ VERIFY: the SDK's `canUseTool` request/return shape.
  const canUseTool = opts.canUseTool
    ? async (req: { toolName?: string; toolInput?: Record<string, unknown> }) => {
        const decision = await opts.canUseTool!(
          { toolName: String(req.toolName ?? ""), input: req.toolInput ?? {} },
          { signal: controller.signal },
        );
        return decision.allow ? { allow: true } : { allow: false, reason: decision.reason ?? "denied" };
      }
    : undefined;

  // ⚠️ VERIFY: `createSdkMcpServer` / `tool` names + the `mcp__<server>__<tool>` allow-list convention.
  // Inject our tools as an in-process MCP server so the agent calls our impls (not its native built-ins).
  let mcpServers: Record<string, unknown> | undefined;
  let allowedTools = opts.allowedTools;
  const injected = opts.mcpTools ? Object.entries(opts.mcpTools) : [];
  if (injected.length > 0 && sdk.createSdkMcpServer && sdk.tool) {
    const tools = injected.map(([name, t]) => sdk.tool!(name, t.description ?? "", t.inputSchema, (input) => t.run(input)));
    mcpServers = { [MCP_SERVER]: sdk.createSdkMcpServer({ name: MCP_SERVER, tools }) };
    // The allow-list is the caller's NATIVE names PLUS our injected tools' MCP-qualified names.
    allowedTools = [...(opts.allowedTools ?? []), ...injected.map(([name]) => `mcp__${MCP_SERVER}__${name}`)];
  }

  const q = sdk.query({
    prompt: opts.prompt,
    options: {
      ...(opts.cwd !== undefined ? { cwd: opts.cwd } : {}),
      ...(allowedTools !== undefined ? { allowedTools } : {}),
      ...(mcpServers !== undefined ? { mcpServers } : {}),
      ...(opts.permissionMode !== undefined ? { permissionMode: opts.permissionMode } : {}),
      ...(canUseTool ? { canUseTool } : {}),
      abortController: controller,
    },
  });

  for await (const msg of q) {
    // ⚠️ VERIFY: the terminal message discriminator + text/cost field names.
    if (msg["type"] === "result") {
      const text = typeof msg["result"] === "string" ? (msg["result"] as string) : typeof msg["text"] === "string" ? (msg["text"] as string) : "";
      const costUsd = typeof msg["total_cost_usd"] === "number" ? (msg["total_cost_usd"] as number) : undefined;
      yield { type: "result", result: { text, costUsd } };
    } else {
      yield { type: "other" };
    }
  }
};
