/**
 * The CLI's permission + tool-injection PROTOCOL, as pure functions.
 *
 * A `claude` subprocess cannot call back into our process directly, but it can be pointed at an MCP
 * server we host: `--mcp-config` declares it, `--permission-prompt-tool` names one of its tools, and the
 * CLI then asks that tool for a decision before each gated tool-use. That is the CLI's equivalent of the
 * SDK adapter's `canUseTool`, and it is what lets `policyEnforcement` be honest here.
 *
 * Everything wire-shaped lives in this module, with no dependency on the MCP SDK or on `node:http`, so
 * the contract below is unit-testable on its own. `./mcpBridge` is the thin part that actually serves it.
 *
 * ## Provenance of the shapes below
 *
 * There is no public prose specification for the permission-prompt tool's request/response contract.
 * The shapes here were read off the SHIPPING CLI implementation (v2.1.215) — its own Zod schemas and
 * call sites — not inferred from documentation or from the Agent SDK's `.d.ts`. Two consequences worth
 * keeping in mind:
 *
 *  - `updatedInput` is REQUIRED on an allow. The Agent SDK's TypeScript `PermissionResult` type marks
 *    it optional; the MCP wire path does NOT, and a bare `{"behavior":"allow"}` fails the CLI's parse.
 *    So an allow always echoes the original input back. This is the detail most likely to be wrong if
 *    it is ever "simplified".
 *  - `message` is REQUIRED on a deny.
 *
 * Being version-pinned observation rather than a published contract, this is the first thing to check
 * if a future CLI rejects our decisions.
 */
import type { AgentPermissionDecision, AgentToolRequest, InjectedTool, JsonValue, OutputValidator, SchemaDocument } from "./deps";

/** The MCP server name our tools are exposed under; the agent sees `mcp__dai__<tool>`. */
export const MCP_SERVER_NAME = "dai";

/** The tool the CLI asks for a permission decision. */
export const APPROVAL_TOOL = "approve";

/** An MCP tool's fully-qualified name, as the CLI addresses it. */
export function mcpToolName(tool: string, server: string = MCP_SERVER_NAME): string {
  return `mcp__${server}__${tool}`;
}

/** What `--permission-prompt-tool` is given. */
export const PERMISSION_PROMPT_TOOL = mcpToolName(APPROVAL_TOOL);

/**
 * The approval tool's declared input schema. The CLI does not check that this MATCHES what it sends,
 * but it does refuse a permission-prompt tool with no input schema at all — and declaring the real
 * fields keeps the tool coherent to anything that reads it.
 */
export const APPROVAL_INPUT_SCHEMA = {
  type: "object",
  properties: {
    tool_name: { type: "string", description: "The name of the tool requesting permission" },
    input: { type: "object", description: "The input for the tool" },
    tool_use_id: { type: "string", description: "The unique tool use request ID" },
  },
  required: ["tool_name", "input"],
} as const;

/** The `--mcp-config` document declaring our server. `type` is REQUIRED: the CLI reads an entry with a
 *  `url` but no `type` as a stdio server and errors out. The URL carries the run's {@link newBridgeToken}
 *  in its path, so handing the CLI this document is also how the secret reaches it — there is no second
 *  channel to keep in sync. */
export function mcpConfigJson(url: string, server: string = MCP_SERVER_NAME): string {
  return JSON.stringify({ mcpServers: { [server]: { type: "http", url } } });
}

// --- Bridge authentication ----------------------------------------------------

/**
 * The per-run BEARER SECRET for the bridge.
 *
 * The bridge is a loopback HTTP server that executes host tools and answers permission questions, so
 * "only reachable from this machine" is not an authorization story: every other process on the box is
 * also on loopback, and an ephemeral port is a two-byte search. Without this, any local process that
 * finds the port drives `ctx.tools` — `run_command`, `write_file` — for the lifetime of the run, and the
 * approver is never consulted because approval is the CLI's job on the far side of the port.
 *
 * 256 bits from the platform CSPRNG. NOT a counter, a pid, or a timestamp: those are guessable by
 * exactly the local attacker this defends against. `globalThis.crypto` rather than `node:crypto` keeps
 * this module free of node imports, like the rest of the protocol.
 */
export function newBridgeToken(): string {
  const bytes = new Uint8Array(32);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The URL path a token authorizes. The secret rides the PATH (not a header) because that is what
 *  `--mcp-config`'s `url` carries end to end — the CLI reproduces it without knowing it is a secret. */
export function bridgePath(token: string): string {
  return `/mcp/${token}`;
}

/** Length-independent, early-exit-free comparison — a token check that leaks its match prefix through
 *  timing is a token check an adjacent local process can walk. */
function secretEquals(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * May this request be dispatched at all? Answered BEFORE any MCP handling, so an unauthenticated caller
 * never reaches `handleToolCall` — not the approval tool, and above all not a host tool impl.
 *
 * Two independent checks:
 *  - the path must carry the run's token, which is the actual authentication;
 *  - a request carrying ANY `Origin` is refused, whatever the value. The CLI never sends one; a browser
 *    always does. That closes DNS rebinding, where a page resolves a hostname to 127.0.0.1 and posts to
 *    the port — it cannot read the reply cross-origin, but a tool call has already RUN by then.
 */
export function isAuthorizedBridgeRequest(req: { url?: string; origin?: string | string[] }, token: string): boolean {
  if (req.origin !== undefined) return false;
  if (req.url === undefined) return false;
  // Compare the PATH only: a query string is not part of the secret, and `?` is not a token character.
  const path = req.url.split("?")[0] ?? "";
  return secretEquals(path, bridgePath(token));
}

/** A permission request as the CLI sends it — snake_case, and `tool_use_id` optional. */
export interface ApprovalRequest {
  toolName: string;
  input: Record<string, JsonValue>;
  toolUseId?: string;
}

/**
 * Read the CLI's approval arguments. Returns `undefined` for anything that is not a well-formed
 * request: a malformed permission ask must DENY, never fall through to an allow, so the caller treats
 * an unreadable request as a refusal rather than guessing at a tool name.
 */
export function parseApprovalRequest(args: unknown): ApprovalRequest | undefined {
  if (args === null || typeof args !== "object" || Array.isArray(args)) return undefined;
  const bag = args as Record<string, unknown>;
  const toolName = bag["tool_name"];
  if (typeof toolName !== "string" || toolName.length === 0) return undefined;
  const rawInput = bag["input"];
  const input = rawInput !== null && typeof rawInput === "object" && !Array.isArray(rawInput) ? (rawInput as Record<string, JsonValue>) : {};
  const toolUseId = bag["tool_use_id"];
  return { toolName, input, ...(typeof toolUseId === "string" ? { toolUseId } : {}) };
}

/**
 * The decision payload, as the JSON string the CLI expects inside `content[0].text`.
 *
 * `input` is echoed into `updatedInput` on an allow because the CLI's schema requires it — we never
 * MODIFY the agent's tool input, but we must restate it. Nothing here rewrites what the agent asked
 * for; an approver that wanted to would be a different feature with a different seam.
 */
export function approvalResponseText(decision: AgentPermissionDecision, input: Record<string, JsonValue>): string {
  if (decision.allow) return JSON.stringify({ behavior: "allow", updatedInput: input });
  return JSON.stringify({ behavior: "deny", message: decision.reason ?? "denied by permission policy" });
}

/** The deny payload for a request we could not even read. */
export function malformedApprovalResponseText(): string {
  return JSON.stringify({ behavior: "deny", message: "the permission request could not be read; denying" });
}

/** An MCP `tools/list` entry. Raw JSON Schema travels verbatim — the low-level server does not
 *  re-serialize it, which is why our authored schemas need no conversion. */
export interface McpToolDescriptor {
  name: string;
  description?: string;
  inputSchema: JsonValue;
}

/**
 * `approve` is RESERVED on this server, whether or not an approver is wired.
 *
 * The name is the permission protocol's, not an authoring namespace's, and a collision is ambiguous in
 * both directions: with an approver, the gate shadows a host tool of that name and the agent silently
 * gets a permission verdict where it asked for a tool result; without one, a host tool would be exposed
 * to the agent under precisely the name `--permission-prompt-tool` addresses. Neither is a thing to
 * guess at, so a colliding registration is refused LOUDLY — at bridge start, which `cliQuery` turns into
 * a refused run rather than a downgraded one.
 */
export function assertNoReservedToolNames(tools: Record<string, InjectedTool> | undefined): void {
  if (tools && APPROVAL_TOOL in tools) {
    throw new Error(
      `an injected tool may not be named '${APPROVAL_TOOL}': that name is reserved for the CLI's permission-prompt tool (${PERMISSION_PROMPT_TOOL}) — rename the tool`,
    );
  }
}

/** Every tool our server exposes: the approval gate (when an approver is wired) plus each injected
 *  host tool. Throws on a reserved-name collision (see {@link assertNoReservedToolNames}). */
export function toolDescriptors(spec: { tools?: Record<string, InjectedTool>; approve?: unknown }): McpToolDescriptor[] {
  assertNoReservedToolNames(spec.tools);
  const out: McpToolDescriptor[] = [];
  if (spec.approve) {
    out.push({
      name: APPROVAL_TOOL,
      description: "Decide whether a tool call is permitted.",
      inputSchema: APPROVAL_INPUT_SCHEMA as unknown as JsonValue,
    });
  }
  for (const [name, tool] of Object.entries(spec.tools ?? {})) {
    out.push({
      name,
      ...(tool.description !== undefined ? { description: tool.description } : {}),
      inputSchema: tool.inputSchema as unknown as JsonValue,
    });
  }
  return out;
}

/**
 * The MCP-qualified names an injected tool set is addressed by — the logical names alone name nothing
 * the agent can call.
 *
 * `cliArgv` deliberately does NOT feed these to `--allowedTools`: that flag is the CLI's PRE-APPROVAL
 * list, and pre-approving our own bridge-served tools is exactly what stopped `--permission-prompt-tool`
 * from ever being asked about them. This helper stays exported for a caller that means to pre-approve
 * them anyway (it can pass the result in `allowedTools`) — an explicit, visible opt-out of the gate
 * rather than a silent one.
 */
export function injectedToolAllowEntries(tools: Record<string, InjectedTool> | undefined): string[] {
  return Object.keys(tools ?? {}).map((name) => mcpToolName(name));
}

/** One MCP tool result. */
export interface McpToolResult {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
}

const textResult = (text: string, isError?: boolean): McpToolResult => ({
  content: [{ type: "text", text }],
  ...(isError === true ? { isError: true } : {}),
});

/**
 * Serve one `tools/call`. This is the whole server behaviour, independent of transport — which is what
 * makes the permission path testable without spawning anything.
 *
 * A tool that THROWS becomes an `isError` result rather than a transport fault: a tool failure is
 * something the AGENT reads and reacts to (DESIGN §5.1, "Functions and tools"), not a failure of the run.
 */
export async function handleToolCall(
  spec: {
    tools?: Record<string, InjectedTool>;
    approve?: (req: AgentToolRequest) => Promise<AgentPermissionDecision>;
    validator?: OutputValidator;
  },
  name: string,
  args: unknown,
): Promise<McpToolResult> {
  // The reserved name is handled here and NEVER falls through to `spec.tools` — a host tool called
  // `approve` must not become callable just because no approver happens to be wired (`toolDescriptors`
  // refuses that registration up front; this is the same rule at the dispatch point).
  if (name === APPROVAL_TOOL) {
    if (!spec.approve) return textResult(`no tool '${name}' is available`, true);
    const request = parseApprovalRequest(args);
    if (!request) return textResult(malformedApprovalResponseText());
    let decision: AgentPermissionDecision;
    try {
      decision = await spec.approve({ toolName: request.toolName, input: request.input });
    } catch {
      // An approver that throws is not consent. Deny.
      return textResult(malformedApprovalResponseText());
    }
    return textResult(approvalResponseText(decision, request.input));
  }

  const tool = spec.tools?.[name];
  if (!tool) return textResult(`no tool '${name}' is available`, true);
  const input = args !== null && typeof args === "object" && !Array.isArray(args) ? (args as Record<string, JsonValue>) : {};
  // The low-level MCP `Server` advertises our schema and then hands the handler whatever arrived —
  // it validates NOTHING. So the boundary check is ours, through `json`'s three-line `OutputValidator`
  // seam (no ajv on the agent path; the caller injects whatever implements it). A malformed call must
  // fail as a tool error the agent reads, rather than reaching an impl that trusted its declared schema.
  const invalid = validateToolInput(spec.validator, tool, input);
  if (invalid) return textResult(`tool '${name}' input is invalid: ${invalid}`, true);
  try {
    const value = await tool.run(input);
    return textResult(typeof value === "string" ? value : JSON.stringify(value ?? null));
  } catch (e) {
    return textResult(`tool '${name}' failed: ${e instanceof Error ? e.message : String(e)}`, true);
  }
}

/**
 * Check one tool's arguments against its OWN declared schema. Returns the failure text, or `undefined`
 * when there is nothing to check (no validator injected, or a tool that declares no schema — an absent
 * schema constrains nothing, so there is no obligation to enforce).
 *
 * A validator that THROWS counts as a refusal, not as a pass: this is the last thing standing between an
 * arbitrary payload and a host impl, so it fails closed like every other gate in this module.
 */
function validateToolInput(
  validator: OutputValidator | undefined,
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
