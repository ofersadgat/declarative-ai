/**
 * The default {@link AgentQuery} — a THIN wrapper over `@anthropic-ai/claude-agent-sdk`'s `query()`. It is
 * loaded lazily (a variable module specifier, so this package type-checks and its fake-driven tests run
 * WITHOUT the SDK installed) and the SDK is an OPTIONAL peer dependency.
 *
 * ✅ VERIFIED against `@anthropic-ai/claude-agent-sdk 0.3.223` — its own `sdk.d.ts`, plus a live
 * seed/resume/fork run through this file:
 *
 *  - `Options.resume?: string` ("Session ID to resume. Loads the conversation history…") and
 *    `Options.forkSession?: boolean` ("When true, resumed sessions will fork to a new session ID rather
 *    than continuing the previous session. **Use with `resume`**") — which is why the two are emitted
 *    together below rather than independently;
 *  - `SDKResultSuccess` carries `result: string`, `total_cost_usd: number`, `session_id: string`, and
 *    `is_error: boolean`. Note the last one is `boolean`, not `false`: the SDK's own type says a
 *    `subtype: "success"` message may report a FAILED run, which is exactly the case that used to reach
 *    callers as the agent's answer.
 *
 * ⚠️ Still unverified: the `canUseTool` request/return shape and the `createSdkMcpServer`/`tool`
 * factories, which a session run does not exercise.
 *
 * Both mapping spots are PURE FUNCTIONS rather than inline expressions, so they stay assertable with
 * the SDK ABSENT — which is the normal state here, and how the session fields came to be dropped in the
 * first place: an optional peer dependency means no test in this package reaches the end-to-end path.
 */
import type { FunctionInputs } from "@declarative-ai/exec";
import type { AgentQuery, AgentQueryOptions, AgentStreamMessage } from "./seam.js";

/** The minimal surface of the SDK we call — cast to, never imported (keeps the missing dep off the type graph). */
interface SdkModule {
  query(arg: { prompt: string; options?: Record<string, unknown> }): AsyncIterable<Record<string, unknown>>;
  /** In-process ("SDK") MCP server factory + tool builder — how custom tools are injected. */
  createSdkMcpServer?: (config: { name: string; tools: unknown[] }) => unknown;
  tool?: (name: string, description: string, inputSchema: unknown, handler: (input: FunctionInputs) => unknown) => unknown;
}

const SDK_SPECIFIER = "@anthropic-ai/claude-agent-sdk";

/** The in-process MCP server name our injected tools are exposed under (agent sees `mcp__dai__<tool>`). */
const MCP_SERVER = "dai";

/**
 * The serializable half of the SDK's `options` — everything that does NOT need the SDK module in hand.
 *
 * Split out for the same reason `cliArgv` is: it makes the boundary mapping directly assertable. The
 * package is an OPTIONAL peer dependency, so a test that drives `sdkAgentQuery` end-to-end cannot run
 * here at all — which is precisely how this mapping came to silently drop the session fields, with the
 * adapter declaring native resume and fork while requesting neither. What stays inside the generator is
 * only what the SDK module itself constructs: the in-process MCP server, the approval callback, and the
 * abort controller.
 */
export function sdkOptions(opts: AgentQueryOptions): Record<string, unknown> {
  return {
    ...(opts.cwd !== undefined ? { cwd: opts.cwd } : {}),
    // The model, when the caller named one. Absent ⇒ the SDK's own default, which is the ordinary
    // case; the prefix that selected this transport is already stripped by the executor.
    ...(opts.model !== undefined ? { model: opts.model } : {}),
    // Injected tools deliberately stay OFF `allowedTools`: that list PRE-APPROVES, so naming them there
    // would open both gates at once — the agent would never put them to `canUseTool` → our approver, the
    // very thing the CLI sibling guards against (`cliQuery.ts`). They are gated through `canUseTool`; a
    // caller that really means to pre-approve them can pass their `mcp__dai__<name>` entries in
    // `allowedTools` itself, which is at least visible.
    ...(opts.allowedTools !== undefined ? { allowedTools: opts.allowedTools } : {}),
    // The deny floor MUST reach the agent: dropping it would let a `deny`d tool run while the workflow
    // believes it is blocked (`seam.ts` — "an adapter that cannot honour it must refuse, not drop it").
    // The SDK honours `disallowedTools`; a future SDK that cannot must refuse here, not silently omit it.
    ...(opts.disallowedTools !== undefined ? { disallowedTools: opts.disallowedTools } : {}),
    ...(opts.permissionMode !== undefined ? { permissionMode: opts.permissionMode } : {}),
    // The SESSION. `resume` continues the conversation the SDK already holds; `resume` + `forkSession`
    // branches it into a new id seeded with the parent's history, leaving the parent untouched. The
    // pair is what makes this transport's declared `sessionResume`/`sessionFork` true rather than
    // aspirational: without them the session layer skips replay — reading zero messages on the
    // strength of that declaration — and every call still starts a cold conversation.
    //
    // `forkSession` is nested inside `resume` deliberately: it BRANCHES a conversation, so it means
    // nothing without one to branch, and the executor only ever sets the two together.
    ...(opts.resume !== undefined ? { resume: opts.resume, ...(opts.forkSession === true ? { forkSession: true } : {}) } : {}),
  };
}

/**
 * The terminal `result` message, normalized — the other half of the boundary mapping, and pure for the
 * same reason.
 *
 * ⚠️ VERIFY: the discriminator and the text/cost field names. `session_id` and `is_error` ARE verified —
 * against the CLI this SDK drives as a subprocess (`claude 2.1.142`, see `cliQuery.ts`), which emits the
 * same stream-json message on the same wire.
 */
export function readSdkResult(msg: Record<string, unknown>): AgentStreamMessage {
  if (msg["type"] !== "result") return { type: "other" };
  const text = typeof msg["result"] === "string" ? (msg["result"] as string) : typeof msg["text"] === "string" ? (msg["text"] as string) : "";
  // A failed run is reported IN the result message, not as an exception: `is_error: true` arrives
  // alongside `subtype: "success"`, carrying the failure text where the answer would be. Yielding it as
  // a result would hand the caller `Not logged in · Please run /login` as the agent's answer, with
  // `finishReason: "stop"` — indistinguishable from a run that succeeded and found nothing.
  if (msg["is_error"] === true) return { type: "other", error: text.length > 0 ? text : "the agent SDK reported a failed run" };
  const costUsd = typeof msg["total_cost_usd"] === "number" ? (msg["total_cost_usd"] as number) : undefined;
  // The id this run ENDED in — a new one after a fork. Dropping it leaves the next call with no handle
  // to resume, which is the half of the session story that fails silently: the transport resumes
  // correctly and still starts fresh, because nothing recorded where it got to.
  const sessionId = typeof msg["session_id"] === "string" ? (msg["session_id"] as string) : undefined;
  return { type: "result", result: { text, costUsd, ...(sessionId !== undefined ? { sessionId } : {}) } };
}

export const sdkAgentQuery: AgentQuery = async function* (opts): AsyncIterable<AgentStreamMessage> {
  let sdk: SdkModule;
  try {
    // Variable specifier: TS won't resolve (or require) the module at build time — it's optional.
    sdk = (await import(/* @vite-ignore */ SDK_SPECIFIER)) as unknown as SdkModule;
  } catch {
    throw new Error(`${SDK_SPECIFIER} is not installed — install it, or inject a \`query\` seam into createClaudeCodeFunction`);
  }

  const controller = new AbortController();
  if (opts.abortSignal) {
    if (opts.abortSignal.aborted) controller.abort();
    else opts.abortSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  // ⚠️ VERIFY: the SDK's `canUseTool` request/return shape.
  const canUseTool = opts.canUseTool
    ? async (req: { toolName?: string; toolInput?: FunctionInputs }) => {
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
  const injected = opts.mcpTools ? Object.entries(opts.mcpTools) : [];
  if (injected.length > 0 && sdk.createSdkMcpServer && sdk.tool) {
    const tools = injected.map(([name, t]) => sdk.tool!(name, t.description ?? "", t.inputSchema, (input: FunctionInputs) => t.run(input)));
    mcpServers = { [MCP_SERVER]: sdk.createSdkMcpServer({ name: MCP_SERVER, tools }) };
  }
  const q = sdk.query({
    prompt: opts.prompt,
    options: {
      ...sdkOptions(opts),
      ...(mcpServers !== undefined ? { mcpServers } : {}),
      ...(canUseTool ? { canUseTool } : {}),
      abortController: controller,
    },
  });

  for await (const msg of q) {
    const next = readSdkResult(msg);
    yield next;
    // A run the SDK itself reported as failed is terminal: the stream has nothing further to say, and
    // continuing would leave the consumer waiting on a result message that is never coming.
    if (next.error !== undefined) return;
  }
};
