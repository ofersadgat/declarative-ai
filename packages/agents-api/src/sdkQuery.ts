/**
 * The default {@link AgentQuery} — a THIN wrapper over `@anthropic-ai/claude-agent-sdk`'s `query()`. It is
 * loaded lazily (a variable module specifier, so this package type-checks and its fake-driven tests run
 * WITHOUT the SDK installed) and the SDK is an OPTIONAL peer dependency.
 *
 * ✅ VERIFIED against `@anthropic-ai/claude-agent-sdk 0.3.223` — its own `sdk.d.ts`, a live probe of the
 * two factories, and a seed/resume/fork run through this file. Nothing here is documentation-shaped any
 * more; what the checks turned up:
 *
 *  - `Options.resume?: string` ("Session ID to resume. Loads the conversation history…") and
 *    `Options.forkSession?: boolean` ("When true, resumed sessions will fork to a new session ID rather
 *    than continuing the previous session. **Use with `resume`**") — which is why the two are emitted
 *    together below rather than independently;
 *  - `SDKResultSuccess` carries `result: string`, `total_cost_usd: number`, `session_id: string`, and
 *    `is_error: boolean`. Note the last one is `boolean`, not `false`: the SDK's own type says a
 *    `subtype: "success"` message may report a FAILED run, which is exactly the case that used to reach
 *    callers as the agent's answer.
 *  - **`canUseTool` takes POSITIONAL arguments and answers in `behavior` vocabulary.** The real type is
 *    `(toolName, input, options) => Promise<PermissionResult | null>` with
 *    `PermissionResult = {behavior:"allow", updatedInput?} | {behavior:"deny", message}`. This file used
 *    to pass one object and answer `{allow: boolean}` — so every approval read an `undefined` tool name
 *    and every verdict was unparseable. See {@link sdkPermissionCallback}.
 *  - **`tool()` + `createSdkMcpServer` reject a JSON Schema.** A live probe: `tool()` accepts the
 *    document and stores it unchanged, then `createSdkMcpServer` throws `inputSchema must be a Zod
 *    schema or raw shape, received an unrecognized object`. Our schemas are JSON Schema documents
 *    written by workflow authors, so the factories are not usable here at all. What IS usable is what
 *    they PRODUCE: `{type:"sdk", name, instance}`, where `instance` is only ever `.connect(transport)`ed.
 *    So this file builds the low-level MCP `Server` itself and hands that over — see
 *    {@link sdkMcpServer}. Injected tools were previously dead on this transport: every run with one
 *    threw before the agent started.
 *
 * The mapping spots are PURE FUNCTIONS rather than inline expressions, so they stay assertable with the
 * SDK ABSENT — which is the normal state here, and how the session fields came to be dropped in the
 * first place: an optional peer dependency means no test in this package reaches the end-to-end path.
 */
import type { FunctionInputs, JsonValue } from "@declarative-ai/exec";
import { defaultBinaryDeps, resolveAgentBinary, type BinaryDeps } from "./binary.js";
import { injectedToolDescriptors, MCP_SERVER_NAME, runInjectedTool } from "./mcpTools.js";
import { readAgentMessage } from "./streamMessages.js";
import type { AgentPermissionCallback, AgentQuery, AgentQueryOptions, AgentRun, AgentStreamMessage, InjectedTool } from "./seam.js";

/** The minimal surface of the SDK we call — cast to, never imported (keeps the missing dep off the type graph). */
interface SdkModule {
  query(arg: { prompt: string | AsyncIterable<unknown>; options?: Record<string, unknown> }): SdkQuery;
}

/** The SDK's `Query` — an async generator that also carries the control requests. */
interface SdkQuery extends AsyncIterable<Record<string, unknown>> {
  interrupt?(): Promise<unknown>;
  setPermissionMode?(mode: string): Promise<void>;
  setModel?(model?: string): Promise<void>;
}

const SDK_SPECIFIER = "@anthropic-ai/claude-agent-sdk";

/** The MCP SDK modules the in-process tool server is built from. Optional, like the Agent SDK itself:
 *  a run with no injected tools must not require them. */
const MCP_SERVER_MODULE = "@modelcontextprotocol/sdk/server/index.js";
const MCP_TYPES_MODULE = "@modelcontextprotocol/sdk/types.js";

/** The message a caller sees when the MCP SDK is missing but tools were asked for. Loud, per the rule
 *  a transport follows for anything it cannot honour: dropping the tools would run the agent on its own
 *  built-ins while the workflow believes it is calling our impls. */
export const MCP_SDK_MISSING =
  `@modelcontextprotocol/sdk is not installed — it is required to serve host tools to an in-process agent. ` +
  `Install it, or run the agent with \`injectTools: false\` so it uses its own built-ins.`;

/** The minimal MCP surface used here — cast to, never imported. */
export interface McpToolServer {
  setRequestHandler(schema: unknown, handler: (request: { params: { name: string; arguments?: unknown } }) => unknown): void;
}
interface McpServerModule {
  Server: new (info: { name: string; version: string }, options: { capabilities: { tools: Record<string, never> } }) => McpToolServer;
}
interface McpTypesModule {
  ListToolsRequestSchema: unknown;
  CallToolRequestSchema: unknown;
}

/**
 * What an MCP tool server is BUILT from — the optional dependency, behind a seam.
 *
 * Injected for the same reason every other boundary in this package is: `@modelcontextprotocol/sdk` is
 * an optional peer, so a test cannot assert the wiring by reaching into a real `Server` (its handlers
 * are private) and must not have to install one to find out whether `tools/list` advertises our
 * descriptors and `tools/call` reaches our impls. That wiring is the whole of what this file adds — the
 * bug it replaced was a factory that threw before the agent started — so it is the part that most needs
 * to be assertable.
 */
export interface McpServerDeps {
  create(): McpToolServer;
  listSchema: unknown;
  callSchema: unknown;
}

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
/**
 * What a delegated agent LOADS, decided rather than inherited.
 *
 * Both halves were unset, so both were whatever the SDK happened to default to — and one of those
 * defaults was actively wrong:
 *
 *  - **`systemPrompt`.** Omitting it does NOT mean "use Claude Code's prompt". The SDK's own code reads
 *    `if (systemPrompt === undefined) prompt = ""` — an EMPTY system prompt. So this transport was
 *    running a bare model that happened to have Claude Code's tools, while the CLI sibling (which
 *    passes no `--system-prompt` and therefore gets the preset) ran the real thing. Two transports
 *    documented as interchangeable were not. The preset is what a delegated agent is delegated FOR, so
 *    it is now requested explicitly.
 *  - **`settingSources`.** Omitted means "load user, project and local" — the operator's personal
 *    `~/.claude/settings.json` included. That makes a workflow's behaviour depend on whose machine it
 *    ran on, which is not reproducible, and it is a real hole rather than a tidiness complaint: a
 *    personal `permissions.allow` entry PRE-APPROVES tools, so a run the workflow believes is gated by
 *    `ctx.approve` silently is not. `project` alone keeps the thing that belongs to the repository —
 *    `.claude/settings.json` and, through it, CLAUDE.md, without which a coding agent is materially
 *    worse at the project it was pointed at — and drops the two that belong to a person.
 *
 * Both are overridable through `providerOptions.claudeCode`, because a host that WANTS the operator's
 * settings (a local developer tool, as opposed to a server) is making a legitimate choice — just not
 * one it should make by accident.
 */
export const DEFAULT_SETTING_SOURCES = ["project"] as const;

/** The `providerOptions.claudeCode` keys this transport understands. Anything else is REFUSED: a
 *  setting silently ignored is the failure the whole escape hatch would otherwise introduce. */
export const CLAUDE_CODE_OPTION_KEYS = [
  "settingSources",
  "systemPrompt",
  "appendSystemPrompt",
  "settings",
  "maxBudgetUsd",
  "extraArgs",
  "fastMode",
  "ultracode",
] as const;

/** What this run asks for that the CLAUDE transports cannot honour, or `undefined` to proceed. Shared
 *  by both, since they drive the same binary and differ only in how they reach it. */
export function claudeOptionsRefusal(opts: AgentQueryOptions): string | undefined {
  const unknown = Object.keys(opts.providerOptions ?? {}).filter((k) => !(CLAUDE_CODE_OPTION_KEYS as readonly string[]).includes(k));
  if (unknown.length > 0) {
    return (
      `providerOptions.claudeCode carries unknown key(s): ${unknown.join(", ")}. ` +
      `This transport understands ${CLAUDE_CODE_OPTION_KEYS.join(", ")} — a setting it does not recognise would be silently ignored`
    );
  }
  return undefined;
}

/** The `Settings` bag, folded from the explicit key plus the two conveniences the escape hatch names
 *  at top level. Absent when nothing asked for anything. */
function claudeSettings(providerOptions: Record<string, JsonValue> | undefined): Record<string, JsonValue> | undefined {
  const explicit = providerOptions?.["settings"];
  const settings: Record<string, JsonValue> = { ...(explicit !== null && typeof explicit === "object" && !Array.isArray(explicit) ? explicit : {}) };
  // `fastMode` and `ultracode` are `Settings` keys rather than `query()` options, so they travel in
  // this bag. Accepted at top level because that is how a caller thinks of them.
  for (const key of ["fastMode", "ultracode"] as const) {
    if (providerOptions?.[key] !== undefined) settings[key] = providerOptions[key]!;
  }
  return Object.keys(settings).length > 0 ? settings : undefined;
}

/** The `systemPrompt` option: a caller's own prompt REPLACES the preset, an append rides ON it, and
 *  neither ⇒ the preset itself (see {@link DEFAULT_SETTING_SOURCES} for why that is stated, not left). */
function claudeSystemPrompt(providerOptions: Record<string, JsonValue> | undefined): unknown {
  const custom = providerOptions?.["systemPrompt"];
  if (typeof custom === "string") return custom;
  const append = providerOptions?.["appendSystemPrompt"];
  return { type: "preset", preset: "claude_code", ...(typeof append === "string" ? { append } : {}) };
}

export function sdkOptions(opts: AgentQueryOptions, binaryPath: string | undefined = opts.binaryPath): Record<string, unknown> {
  const po = opts.providerOptions;
  const settings = claudeSettings(po);
  const sources = po?.["settingSources"];
  return {
    ...(opts.cwd !== undefined ? { cwd: opts.cwd } : {}),
    // WHICH binary answers. Absent ⇒ the SDK's own bundled executable, which is the right default and
    // the reason this went unnoticed: the defect only bites a host that pins its own build, and then it
    // bites on Windows, where the SDK spawns without a shell and a bare name resolves to nothing.
    // `binaryPath` is passed already-resolved (see `resolveAgentBinary`) so what arrives here is
    // launchable rather than merely named.
    ...(binaryPath !== undefined ? { pathToClaudeCodeExecutable: binaryPath } : {}),
    // The whole environment, or nothing: the SDK documents an omitted `env` as "inherits process.env",
    // so passing a partial bag would silently strip PATH and the credentials off the subprocess.
    ...(opts.env !== undefined ? { env: opts.env } : {}),
    // The model, when the caller named one. Absent ⇒ the SDK's own default, which is the ordinary
    // case; the prefix that selected this transport is already stripped by the executor.
    ...(opts.model !== undefined ? { model: opts.model } : {}),
    // Injected tools deliberately stay OFF `allowedTools`: that list PRE-APPROVES, so naming them there
    // would open both gates at once — the agent would never put them to `canUseTool` → our approver, the
    // very thing the CLI sibling guards against (`cliQuery.ts`). They are gated through `canUseTool`; a
    // caller that really means to pre-approve them can pass their `mcp__dai__<name>` entries in
    // `allowedTools` itself, which is at least visible.
    //
    // ✅ An EMPTY list means "pre-approve nothing", not "allow nothing" — checked against the binary the
    // SDK drives, where a run with `--allowedTools ""` still used its native `Read` freely. So this and
    // `cliArgv`'s omit-when-empty agree, and neither silently disarms the agent.
    ...(opts.allowedTools !== undefined ? { allowedTools: opts.allowedTools } : {}),
    // The deny floor MUST reach the agent: dropping it would let a `deny`d tool run while the workflow
    // believes it is blocked (`seam.ts` — "an adapter that cannot honour it must refuse, not drop it").
    // The SDK honours `disallowedTools`; a future SDK that cannot must refuse here, not silently omit it.
    ...(opts.disallowedTools !== undefined ? { disallowedTools: opts.disallowedTools } : {}),
    ...(opts.permissionMode !== undefined ? { permissionMode: opts.permissionMode } : {}),
    // STREAMING, and the reason `DELEGATED_CAPS.streaming: true` is now true rather than aspirational.
    // Without it the SDK yields whole assistant turns and the first thing a caller sees is the finished
    // answer — for a run that can take minutes, that is indistinguishable from a hung process.
    includePartialMessages: true,
    // How hard to think. `effort` takes the neutral level verbatim — the SDK's vocabulary is
    // `low | medium | high | xhigh | max`, which is why `xhigh` is now in `ReasoningSpec` rather than
    // being smuggled through `providerOptions`. A budget is the newer `thinking` option rather than the
    // deprecated `maxThinkingTokens`, whose meaning collapsed to on/off on recent models.
    ...(opts.reasoning?.effort !== undefined ? { effort: opts.reasoning.effort } : {}),
    ...(opts.reasoning?.budgetTokens !== undefined ? { thinking: { type: "enabled", budgetTokens: opts.reasoning.budgetTokens } } : {}),
    // The agent's OWN loop bound: one step is one model→tool→model turn, which is exactly what
    // `maxTurns` counts.
    ...(opts.maxSteps !== undefined ? { maxTurns: opts.maxSteps } : {}),
    // `toolChoice: "none"` — answer from what you already know. `[]` is the SDK's documented "disable
    // all built-in tools"; `auto` is the default and says nothing.
    ...(opts.toolChoice === "none" ? { tools: [] } : {}),
    // DECIDED, not inherited — see {@link DEFAULT_SETTING_SOURCES} for both, and for why omitting
    // `systemPrompt` was running this transport with no system prompt at all.
    settingSources: Array.isArray(sources) ? sources : [...DEFAULT_SETTING_SOURCES],
    systemPrompt: claudeSystemPrompt(po),
    ...(settings !== undefined ? { settings } : {}),
    ...(typeof po?.["maxBudgetUsd"] === "number" ? { maxBudgetUsd: po["maxBudgetUsd"] } : {}),
    ...(po?.["extraArgs"] !== undefined ? { extraArgs: po["extraArgs"] } : {}),
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
 * The stream mapping, kept under its old name.
 *
 * It is now {@link readAgentMessage}, shared with the CLI sibling — the two transports carry the SAME
 * messages, because the SDK drives that binary as a subprocess and hands its `stream-json` lines
 * through with their field names untouched. Two mappings were two chances to drop the same field, and
 * both dropped every field: everything but the terminal `result` collapsed to `{type: "other"}`.
 */
export const readSdkResult: (msg: Record<string, unknown>) => AgentStreamMessage = readAgentMessage;

/**
 * Our approver, in the SDK's own calling convention.
 *
 * Pure and exported because it is the shape that was WRONG and could not be caught: the SDK is an
 * optional peer dependency, so nothing here ever called it, and a callback that reads `req.toolName`
 * off a positional `toolName` argument produces `""` for every tool while answering in a vocabulary
 * (`{allow}`) the SDK does not parse. Both halves fail silently — the agent proceeds under its own
 * defaults while the workflow believes its approver is in force.
 *
 * `updatedInput` echoes the agent's own input back on an allow. We never REWRITE what the agent asked
 * for; the field is how the SDK's wire path spells "and use these arguments", and the CLI it drives
 * requires it (see `agents-cli`'s `mcpProtocol.ts`, where a bare allow was observed failing the parse).
 */
export function sdkPermissionCallback(
  approve: AgentPermissionCallback,
  signal: AbortSignal,
): (toolName: string, input: Record<string, unknown>, options: { signal?: AbortSignal }) => Promise<Record<string, unknown>> {
  return async (toolName, input, options) => {
    const decision = await approve({ toolName, input: (input ?? {}) as FunctionInputs }, { signal: options?.signal ?? signal });
    return decision.allow ? { behavior: "allow", updatedInput: input ?? {} } : { behavior: "deny", message: decision.reason ?? "denied" };
  };
}

/**
 * The in-process MCP server our injected tools are served from, as the SDK's `mcpServers` entry.
 *
 * Built from the LOW-LEVEL `Server` rather than through `createSdkMcpServer`/`tool`, because those
 * convert `inputSchema` through Zod and ours are JSON Schema documents — see this module's header for
 * the probe. The entry shape is exactly what `createSdkMcpServer` returns (`{type:"sdk", name,
 * instance}`) and the SDK does one thing with `instance`: `connect()` it to an in-memory transport. So
 * this is the same contract reached without the schema conversion.
 *
 * Returns `undefined` when there is nothing to serve. THROWS when there are tools and no MCP SDK: the
 * caller turns that into a refused run rather than a run with the tools quietly missing.
 */
export async function sdkMcpServer(
  tools: Record<string, InjectedTool>,
  spec: { validator?: AgentQueryOptions["validator"]; deps?: McpServerDeps } = {},
  server: string = MCP_SERVER_NAME,
): Promise<Record<string, unknown> | undefined> {
  if (Object.keys(tools).length === 0) return undefined;
  const deps = spec.deps ?? (await defaultMcpServerDeps());
  const descriptors = injectedToolDescriptors(tools);
  const instance = deps.create();
  instance.setRequestHandler(deps.listSchema, () => ({ tools: descriptors }));
  instance.setRequestHandler(deps.callSchema, (request) =>
    runInjectedTool({ tools, ...(spec.validator !== undefined ? { validator: spec.validator } : {}) }, request.params.name, request.params.arguments),
  );
  return { [server]: { type: "sdk", name: server, instance } };
}

/** Build the server from the real optional dependency. THROWS when it is missing: the caller turns
 *  that into a refused run rather than one with the tools quietly absent. */
export async function defaultMcpServerDeps(): Promise<McpServerDeps> {
  let serverModule: McpServerModule;
  let typesModule: McpTypesModule;
  try {
    // Variable specifiers: TS will not resolve (or require) these at build time — they are optional.
    serverModule = (await import(/* @vite-ignore */ MCP_SERVER_MODULE)) as unknown as McpServerModule;
    typesModule = (await import(/* @vite-ignore */ MCP_TYPES_MODULE)) as unknown as McpTypesModule;
  } catch {
    throw new Error(MCP_SDK_MISSING);
  }
  return {
    create: () => new serverModule.Server({ name: "declarative-ai", version: "0.1.0" }, { capabilities: { tools: {} } }),
    listSchema: typesModule.ListToolsRequestSchema,
    callSchema: typesModule.CallToolRequestSchema,
  };
}

/** Settings for the default SDK query — the seams a test stands in for. */
export interface SdkAgentOptions {
  /** The filesystem/environment facts a named `binaryPath` is resolved against. Default:
   *  {@link defaultBinaryDeps}, read off the real process. Tests inject a fake filesystem. */
  binaryDeps?: BinaryDeps;
  /** Where a resolution warning goes. Default: `console.warn`. A warning is not a refusal — the run
   *  still proceeds with what the caller wrote, because a resolution we could not complete is not the
   *  same claim as a binary that is definitely absent. */
  warn?: (message: string) => void;
}

/**
 * An input QUEUE the SDK reads as the run's prompt.
 *
 * Streaming input is what buys the control channel: the SDK's `interrupt` / `setPermissionMode` /
 * `setModel` are documented as "only supported when streaming input/output is used", and a `prompt:
 * string` is not that. So the initial instruction becomes the first message on this queue, `send()`
 * pushes further ones, and the queue closes when the run ends.
 *
 * The closure owns it — nothing outside manages its lifetime, which is the property that lets the whole
 * control surface be four optional methods rather than a resource a caller has to release.
 */
export class InputQueue {
  private readonly buffer: unknown[] = [];
  private waiter: ((v: IteratorResult<unknown>) => void) | undefined;
  private closed = false;

  push(text: string): void {
    if (this.closed) return;
    const message = { type: "user", message: { role: "user", content: text }, parent_tool_use_id: null };
    const waiter = this.waiter;
    if (waiter) {
      this.waiter = undefined;
      waiter({ value: message, done: false });
    } else this.buffer.push(message);
  }

  close(): void {
    this.closed = true;
    const waiter = this.waiter;
    if (waiter) {
      this.waiter = undefined;
      waiter({ value: undefined, done: true });
    }
  }

  iterate(): AsyncIterable<unknown> {
    const self = this;
    return {
      [Symbol.asyncIterator]: (): AsyncIterator<unknown> => ({
        next: (): Promise<IteratorResult<unknown>> => {
          const buffered = self.buffer.shift();
          if (buffered !== undefined) return Promise.resolve({ value: buffered, done: false });
          if (self.closed) return Promise.resolve({ value: undefined, done: true });
          return new Promise((resolve) => (self.waiter = resolve));
        },
      }),
    };
  }
}

/** Build an SDK-driven agent query. The bare {@link sdkAgentQuery} is this with every default taken. */
export function createSdkAgentQuery(config: SdkAgentOptions = {}): AgentQuery {
  return (opts) => runSdkQuery(opts, config);
}

export const sdkAgentQuery: AgentQuery = (opts) => runSdkQuery(opts, {});

/**
 * One live run: the message stream, plus the control requests the SDK's `Query` object carries.
 *
 * The query object is created INSIDE the generator (it needs an awaited dynamic import), so the control
 * methods close over a slot rather than over the object. That is not a workaround, it is the honest
 * shape: a control request before the query exists has nothing to reach, and answering it as a no-op is
 * exactly right — there is no turn under way to interrupt.
 */
function runSdkQuery(opts: AgentQueryOptions, config: SdkAgentOptions): AgentRun {
  const input = new InputQueue();
  let query: SdkQuery | undefined;
  let finished = false;

  const stream = sdkMessages(opts, config, input, (q) => (query = q), () => (finished = true));

  return {
    [Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](),
    // IDEMPOTENT, and a no-op once the run is over: an interrupt racing the stream's own end is the
    // ordinary case (a user presses Stop as the answer lands), and it must not become an error or a
    // second settle. The turn ends, a `result` still arrives, and the call SUCCEEDS.
    interrupt: async () => {
      if (finished) return;
      await query?.interrupt?.();
    },
    send: async (text: string) => {
      if (finished) return;
      input.push(text);
    },
    setPermissionMode: async (mode) => {
      if (finished) return;
      await query?.setPermissionMode?.(mode);
    },
    setModel: async (model: string) => {
      if (finished) return;
      await query?.setModel?.(model);
    },
  };
}

async function* sdkMessages(
  opts: AgentQueryOptions,
  config: SdkAgentOptions,
  input: InputQueue,
  hold: (q: SdkQuery) => void,
  done: () => void,
): AsyncIterable<AgentStreamMessage> {
  let sdk: SdkModule;
  try {
    // Variable specifier: TS won't resolve (or require) the module at build time — it's optional.
    sdk = (await import(/* @vite-ignore */ SDK_SPECIFIER)) as unknown as SdkModule;
  } catch {
    throw new Error(`${SDK_SPECIFIER} is not installed — install it, or inject a \`query\` seam into createClaudeCodeFunction`);
  }

  // REFUSE BEFORE STARTING, as every transport here does: a setting silently ignored leaves the caller
  // believing in a configuration that was never applied.
  const refusal = claudeOptionsRefusal(opts);
  if (refusal !== undefined) {
    yield { type: "other", error: refusal };
    return;
  }

  const controller = new AbortController();
  if (opts.abortSignal) {
    if (opts.abortSignal.aborted) controller.abort();
    else opts.abortSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  const canUseTool = opts.canUseTool ? sdkPermissionCallback(opts.canUseTool, controller.signal) : undefined;

  // Inject our tools as an in-process MCP server so the agent calls our impls (not its native built-ins).
  // A failure to build one is REFUSED rather than dropped — running without the tools would leave the
  // agent answering from its own built-ins while the caller believes its impls are in play.
  let mcpServers: Record<string, unknown> | undefined;
  try {
    mcpServers = await sdkMcpServer(opts.mcpTools ?? {}, { ...(opts.validator !== undefined ? { validator: opts.validator } : {}) });
  } catch (e) {
    yield { type: "other", error: e instanceof Error ? e.message : String(e) };
    return;
  }

  // Resolved BEFORE the query is built, and only when the caller named one: absent means "use your own
  // executable", which is what the SDK does best and what a resolution here would take away.
  let binaryPath: string | undefined;
  if (opts.binaryPath !== undefined) {
    const resolved = resolveAgentBinary(opts.binaryPath, config.binaryDeps ?? (await defaultBinaryDeps()));
    binaryPath = resolved.path;
    if (resolved.warning !== undefined) (config.warn ?? ((m: string) => console.warn(m)))(resolved.warning);
  }

  // The instruction is the FIRST message on the input queue rather than a `prompt` string, because the
  // control requests exist only in streaming-input mode. Pushed before `query()` so it is already
  // buffered when the SDK first reads.
  input.push(opts.prompt);
  const q = sdk.query({
    prompt: input.iterate(),
    options: {
      ...sdkOptions(opts, binaryPath),
      ...(mcpServers !== undefined ? { mcpServers } : {}),
      ...(canUseTool ? { canUseTool } : {}),
      abortController: controller,
    },
  });
  hold(q);

  try {
    for await (const msg of q) {
      const next = readSdkResult(msg as Record<string, JsonValue>);
      yield next;
      // A run the SDK itself reported as failed is terminal: the stream has nothing further to say, and
      // continuing would leave the consumer waiting on a result message that is never coming.
      if (next.error !== undefined) return;
      // The turn is OVER. In streaming-input mode the SDK waits for more input rather than ending, so
      // without this a one-turn call never returns — it parks on a queue only a `send()` could feed.
      if (next.type === "result") return;
    }
  } finally {
    // Whatever ended it — the result, an error, a consumer finalizing early — the input queue closes
    // and the control methods go quiet. An open queue would hold the subprocess waiting for a message
    // that is never coming.
    done();
    input.close();
  }
}
