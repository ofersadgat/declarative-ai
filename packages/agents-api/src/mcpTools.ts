/**
 * Serving OUR tools to a delegated agent over MCP — the TRANSPORT-INDEPENDENT half.
 *
 * Both delegated transports inject host tools the same way: the agent is told about an MCP server named
 * `dai`, addresses its tools as `mcp__dai__<name>`, and every call arrives as untyped JSON that has to be
 * checked against the tool's own `inputSchema` before an impl sees it. What differs is only how the
 * server is REACHED — an in-process `Server` handed to the Agent SDK (`sdkQuery.ts`), or a loopback HTTP
 * listener a subprocess connects to (`agents-cli`'s `mcpBridge.ts`). This module is the part that does
 * not differ, and it lives here because `agents-cli` depends on this package rather than the reverse.
 *
 * It is pure: no MCP SDK import, no transport, no `node:` anything. That is what makes the dispatch
 * assertable without either optional dependency installed — the same reason `sdkOptions` and `cliArgv`
 * are pure functions.
 *
 * ⚠️ The JSON Schema travels VERBATIM. The MCP SDK's high-level `registerTool` (and the Agent SDK's own
 * `tool()` helper, which wraps it) accepts only ZOD schemas — a JSON Schema document reaches it and
 * throws `inputSchema must be a Zod schema or raw shape`. Our schemas are JSON Schema documents written
 * by workflow authors, so both transports drive the LOW-LEVEL `Server`, which advertises whatever it is
 * given and validates nothing. Validation is therefore ours, and it is {@link runInjectedTool}'s.
 */
import type { JsonValue, SchemaDocument, SyncOutputValidator } from "@declarative-ai/exec";
import type { InjectedTool } from "./seam.js";

/** The MCP server name our injected tools are exposed under; the agent sees `mcp__dai__<tool>`. */
export const MCP_SERVER_NAME = "dai";

/** An MCP tool's fully-qualified name, as an agent addresses it. */
export function mcpToolName(tool: string, server: string = MCP_SERVER_NAME): string {
  return `mcp__${server}__${tool}`;
}

/** An MCP `tools/list` entry. Raw JSON Schema travels verbatim — the low-level server does not
 *  re-serialize it, which is why our authored schemas need no conversion. */
export interface McpToolDescriptor {
  name: string;
  description?: string;
  inputSchema: JsonValue;
}

/** One MCP tool result. */
export interface McpToolResult {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
}

/** The one-line result shape both transports answer with. */
export function textResult(text: string, isError?: boolean): McpToolResult {
  return { content: [{ type: "text", text }], ...(isError === true ? { isError: true } : {}) };
}

/** The `tools/list` entries for a set of injected tools. */
export function injectedToolDescriptors(tools: Record<string, InjectedTool> | undefined): McpToolDescriptor[] {
  return Object.entries(tools ?? {}).map(([name, tool]) => ({
    name,
    ...(tool.description !== undefined ? { description: tool.description } : {}),
    inputSchema: tool.inputSchema as unknown as JsonValue,
  }));
}

/**
 * Check one tool's arguments against its OWN declared schema. Returns the failure text, or `undefined`
 * when there is nothing to check (no validator injected, or a tool that declares no schema — an absent
 * schema constrains nothing, so there is no obligation to enforce).
 *
 * A validator that THROWS counts as a refusal, not as a pass: this is the last thing standing between an
 * arbitrary payload and a host impl, so it fails closed.
 */
export function validateToolInput(
  validator: SyncOutputValidator | undefined,
  tool: InjectedTool,
  input: Record<string, JsonValue>,
): string | undefined {
  if (!validator) return undefined;
  const schema = tool.inputSchema as SchemaDocument | undefined;
  if (schema === undefined || typeof schema !== "object") return undefined;
  try {
    const result = validator.validateValue(schema, input);
    return result.ok ? undefined : (result.errors ?? "does not match the tool's declared input schema");
  } catch (e) {
    return `input could not be validated: ${e instanceof Error ? e.message : String(e)}`;
  }
}

/** What {@link runInjectedTool} serves. */
export interface InjectedToolSpec {
  tools?: Record<string, InjectedTool>;
  /** Checks an injected tool's arguments against its own `inputSchema` before the impl sees them.
   *  Absent ⇒ unvalidated, so the caller that owns the tools is the one that decides. */
  validator?: SyncOutputValidator;
}

/**
 * Run ONE injected tool call, whatever the transport carrying it.
 *
 * A tool that THROWS becomes an `isError` result rather than a transport fault: a tool failure is
 * something the AGENT reads and reacts to (DESIGN §5.1, "Functions and tools"), not a failure of the run.
 */
export async function runInjectedTool(spec: InjectedToolSpec, name: string, args: unknown): Promise<McpToolResult> {
  const tool = spec.tools?.[name];
  if (!tool) return textResult(`no tool '${name}' is available`, true);
  const input = args !== null && typeof args === "object" && !Array.isArray(args) ? (args as Record<string, JsonValue>) : {};
  const invalid = validateToolInput(spec.validator, tool, input);
  if (invalid) return textResult(`tool '${name}' input is invalid: ${invalid}`, true);
  try {
    const value = await tool.run(input);
    return textResult(typeof value === "string" ? value : JSON.stringify(value ?? null));
  } catch (e) {
    return textResult(`tool '${name}' failed: ${e instanceof Error ? e.message : String(e)}`, true);
  }
}
