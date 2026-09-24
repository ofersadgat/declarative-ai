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
 * call sites — not inferred from documentation or from the Agent SDK's `.d.ts`.
 *
 * ✅ RE-CONFIRMED against `claude 2.1.142` by running it, through a real loopback bridge answering a
 * forced `permissions.ask` on `Bash`. All three observations still hold, and each was checked by
 * answering the WRONG way and watching what broke:
 *
 *  - `updatedInput` is REQUIRED on an allow. The Agent SDK's TypeScript `PermissionResult` type marks
 *    it optional; the MCP wire path does NOT. A bare `{"behavior":"allow"}` does not deny — it fails
 *    the CLI's parse, the tool call comes back as a harness ERROR, and the agent works around it (it
 *    reached for `PowerShell` when `Bash` "errored"). So an allow always echoes the original input
 *    back. This is the detail most likely to be wrong if it is ever "simplified".
 *  - `message` is REQUIRED on a deny. A bare `{"behavior":"deny"}` is likewise a validation error
 *    rather than a refusal — the agent retried and then reported the harness as broken, which is not
 *    what a denied tool should look like to it.
 *  - `type` is REQUIRED on the `--mcp-config` entry. Without it the server is SILENTLY DROPPED
 *    (`system/init` reports `mcp_servers: []`); with it the same document registers the server.
 *
 * One thing that changed and is worth knowing: `--permission-prompt-tool` no longer appears in
 * `claude --help` on 2.1.142. It is still accepted and still drives the callback — the live runs above
 * are through it — but an undocumented flag is one version from disappearing, so a run where nothing
 * ever reaches the approver is the symptom to look for.
 *
 * Being version-pinned observation rather than a published contract, this is the first thing to check
 * if a future CLI rejects our decisions.
 */
import {
  injectedToolCallOf,
  injectedToolDescriptors,
  MCP_SERVER_NAME,
  mcpToolName,
  runInjectedTool,
  sdkServerEntries,
  textResult,
  type AgentPermissionDecision,
  type AgentToolRequest,
  type InjectedTool,
  type JsonValue,
  type McpServerSpec,
  type McpToolDescriptor,
  type McpToolResult,
  type SyncOutputValidator,
} from "./deps.js";

// The tool vocabulary is `agents-api`'s (see `./deps`), re-exported here so this module stays the one
// place a CLI-side caller reads the protocol from.
export { MCP_SERVER_NAME, mcpToolName, type McpToolDescriptor, type McpToolResult };

/** The tool the CLI asks for a permission decision. */
export const APPROVAL_TOOL = "approve";

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

/**
 * The ONE `--mcp-config` document a run is handed: our server, when a bridge serves this run, and the
 * host's own servers beside it. `type` is REQUIRED on every entry: the CLI reads an entry with a `url`
 * but no `type` as a stdio server and errors out. The bridge's URL carries the run's
 * {@link newBridgeToken} in its path, so handing the CLI this document is also how the secret reaches
 * it — there is no second channel to keep in sync.
 *
 * One document rather than one per party. The CLI does accumulate a repeated `--mcp-config` (a host
 * measured two documents against claude 2.1.142 and `system/init` listed both), but a second flag is a
 * second thing a host had to smuggle through `extraArgs`, and one document is what the agent is handed
 * on every other transport. The host's entries are {@link sdkServerEntries}' — the shapes the SDK
 * sibling hands the same binary — so the two claude transports cannot drift apart.
 */
export function mcpConfigJson(url: string | undefined, servers?: Record<string, McpServerSpec>): string {
  return JSON.stringify({ mcpServers: { ...sdkServerEntries(servers), ...(url !== undefined ? { [MCP_SERVER_NAME]: { type: "http", url } } : {}) } });
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

/**
 * Where the bridge serves one PROXIED server — the run's path with the server's name after it, so the
 * same secret authorizes it and a server name (already refused unless it is `[A-Za-z0-9_-]`, see
 * `mcpServerNameRefusal`) needs no escaping. Given the bridge's URL, it is that URL's too: the agent is
 * pointed at `<bridge url>/<server>`.
 */
export function bridgeServerPath(base: string, server: string): string {
  return `${base}/${server}`;
}

/** Length-independent, early-exit-free comparison — a token check that leaks its match prefix through
 *  timing is a token check an adjacent local process can walk. */
function secretEquals(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** Which server of a run a request is for: ours (`server` absent), or one the bridge proxies. */
export interface BridgeTarget {
  server?: string;
}

/**
 * May this request be dispatched at all, and to which server? Answered BEFORE any MCP handling, so an
 * unauthenticated caller never reaches `handleToolCall` — not the approval tool, not a host tool impl,
 * and not a proxied server. `undefined` refuses.
 *
 * Two independent checks:
 *  - the path must be the run's own ({@link bridgePath}) or one of its proxied servers'
 *    ({@link bridgeServerPath}) — the token in it is the actual authentication. Every candidate is
 *    compared, with no early exit, so the answer's timing says nothing about which came close;
 *  - a request carrying ANY `Origin` is refused, whatever the value. The CLI never sends one; a browser
 *    always does. That closes DNS rebinding, where a page resolves a hostname to 127.0.0.1 and posts to
 *    the port — it cannot read the reply cross-origin, but a tool call has already RUN by then.
 */
export function bridgeTargetOf(req: { url?: string; origin?: string | string[] }, token: string, servers: readonly string[] = []): BridgeTarget | undefined {
  if (req.origin !== undefined) return undefined;
  if (req.url === undefined) return undefined;
  // Compare the PATH only: a query string is not part of the secret, and `?` is not a token character.
  const path = req.url.split("?")[0] ?? "";
  const base = bridgePath(token);
  let found: BridgeTarget | undefined = secretEquals(path, base) ? {} : undefined;
  for (const server of servers) if (secretEquals(path, bridgeServerPath(base, server))) found = { server };
  return found;
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
 * MODIFY the agent's tool input, but we must restate it. The one exception is a decision that CARRIES
 * a replacement: an answered `AskUserQuestion` returns the human's choices as `{...input, answers}`,
 * which is the documented way the answer reaches the agent. Anything else rewriting arguments would
 * be a different feature with a different seam.
 */
export function approvalResponseText(decision: AgentPermissionDecision, input: Record<string, JsonValue>): string {
  if (decision.allow) return JSON.stringify({ behavior: "allow", updatedInput: decision.updatedInput ?? input });
  return JSON.stringify({ behavior: "deny", message: decision.reason ?? "denied by permission policy" });
}

/** The deny payload for a request we could not even read. */
export function malformedApprovalResponseText(): string {
  return JSON.stringify({ behavior: "deny", message: "the permission request could not be read; denying" });
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
  out.push(...injectedToolDescriptors(spec.tools));
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

/**
 * Serve one `tools/call`. This is the whole server behaviour, independent of transport — which is what
 * makes the permission path testable without spawning anything.
 *
 * The APPROVAL branch is this package's; everything else is {@link runInjectedTool}, which both
 * delegated transports share. A tool that THROWS becomes an `isError` result rather than a transport
 * fault: a tool failure is something the AGENT reads and reacts to (DESIGN §5.1, "Functions and
 * tools"), not a failure of the run.
 */
export async function handleToolCall(
  spec: {
    tools?: Record<string, InjectedTool>;
    approve?: (req: AgentToolRequest) => Promise<AgentPermissionDecision>;
    validator?: SyncOutputValidator;
  },
  name: string,
  args: unknown,
  /** The request's `_meta`, as the CLI sent it — where it names the call's tool-use id. */
  meta?: unknown,
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
      decision = await spec.approve({ toolName: request.toolName, input: request.input, ...(request.toolUseId !== undefined ? { toolUseId: request.toolUseId } : {}) });
    } catch {
      // An approver that throws is not consent. Deny.
      return textResult(malformedApprovalResponseText());
    }
    return textResult(approvalResponseText(decision, request.input));
  }

  // The low-level MCP `Server` advertises our schema and then hands the handler whatever arrived — it
  // validates NOTHING — so the boundary check on an injected tool's arguments lives in `runInjectedTool`.
  return runInjectedTool(spec, name, args, injectedToolCallOf(meta));
}

// --- Proxied servers ------------------------------------------------------------

/**
 * One of the host's own MCP servers, CONNECTED — what a bridge serves under the server's name on behalf
 * of an agent that cannot be asked about a call (codex). `./mcpProxy` builds one over the MCP SDK's
 * client; a test builds one by hand.
 */
export interface McpProxy {
  /** The server's tools as it listed them when it was connected — name, description, input schema and
   *  annotations verbatim. Listed up front because codex fixes its per-tool approval table from them. */
  tools: McpToolDescriptor[];
  /**
   * Forward one call and answer with what the server answered — its result VERBATIM (whatever content
   * parts, structured content or `isError` it carries), or a throw carrying its JSON-RPC `code`,
   * `message` and `data`, which the bridge's MCP server hands the agent as the same error.
   */
  call(tool: string, args: Record<string, JsonValue>): Promise<unknown>;
  /** Close the connection — for a stdio server, end the process it started. */
  close(): Promise<void>;
}

/** What a bridge needs to serve proxied servers: the connections, and the gate each call crosses. */
export interface ProxiedServers {
  servers?: Record<string, McpProxy>;
  /** Put to every call before it is forwarded — see `AgentQueryOptions.serverToolGate`. Absent ⇒ the
   *  host published no gate, and a call is forwarded as a served tool with no gate is run. */
  serverToolGate?: (req: AgentToolRequest) => Promise<AgentPermissionDecision>;
}

/**
 * The answer a gated call that was REFUSED gets: `PermissionDenied`, the shape the executor answers a
 * refused injected tool with, so the agent reads one refusal whichever server the tool was on. Not an
 * `isError` for the same reason: it is the policy's answer, not the tool's failure.
 */
export function deniedResult(tool: string, reason: string | undefined): McpToolResult {
  return textResult(JSON.stringify({ denied: true, tool, ...(reason !== undefined ? { reason } : {}) }));
}

/**
 * Serve one `tools/call` for a PROXIED server — the gate first, then the real server.
 *
 * The subject the gate is asked about is the name the agent calls the tool by, `mcp__<server>__<tool>`,
 * which is also the name claude's permission callback reports for the same call — so a mode written
 * for that tool answers the same way on either transport. The input is the call's own object, handed
 * to the gate and then forwarded, so a host that keeps something against it (JaiRA notes the MCP call
 * it really is) finds it on the way through.
 *
 * A tool the server did not list is refused without asking anybody or forwarding anything: the agent
 * was never offered it. A gate that throws is not consent — the call is refused.
 */
export async function handleServerToolCall(spec: ProxiedServers, server: string, name: string, args: unknown): Promise<unknown> {
  const proxy = spec.servers?.[server];
  if (proxy === undefined || !proxy.tools.some((tool) => tool.name === name)) return textResult(`no tool '${name}' is available on '${server}'`, true);
  const subject = mcpToolName(name, server);
  const input = args !== null && typeof args === "object" && !Array.isArray(args) ? (args as Record<string, JsonValue>) : {};
  if (spec.serverToolGate !== undefined) {
    let decision: AgentPermissionDecision;
    try {
      decision = await spec.serverToolGate({ toolName: subject, input });
    } catch {
      return deniedResult(subject, "the permission check failed; denying");
    }
    if (!decision.allow) return deniedResult(subject, decision.reason);
  }
  return proxy.call(name, input);
}
