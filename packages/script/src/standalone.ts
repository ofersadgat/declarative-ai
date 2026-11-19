/**
 * The STANDALONE host: what `@declarative-ai/hw/script`'s hooks call when no engine is running them
 * (SCRIPTS.md §17). One file runs two ways — durably, compiled to states under the engine, and as a
 * plain program under `node` — and this is the second way.
 *
 * Nothing here is recorded or replayed. A process that dies does not resume; that is the difference
 * between the two modes, and the price of needing nothing but `node`. What the engine would have made a
 * state is simply done:
 *
 *  - `llm()` calls the model through `@declarative-ai/llm`'s router, with the conversation it continues
 *    held in memory (a session ref is `<stream>@<seq>`, as the engine's are);
 *  - `agent()` runs a coding agent through `@declarative-ai/agents-cli` (Claude Code by default);
 *  - `input()` reads the terminal, and resolves `null` at end of input;
 *  - `workflow()` imports the named script and calls it — an imported workflow is a function call here;
 *  - `phase()` and `log()` write to stderr, `now()`/`random()` are the clock and the coin.
 */
import { createInterface } from "node:readline";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { AgentOptions, InputOptions, LlmCall, ScriptHost } from "@declarative-ai/hw/script";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import { createModelRouter, executeLlmCall, type ModelRouter } from "@declarative-ai/llm";
import type { AgentQuery, AgentQueryOptions } from "@declarative-ai/agents-api";
import { createCliAgentQuery } from "@declarative-ai/agents-cli";

/** One turn of a conversation held in memory. */
export interface StandaloneTurn {
  role: "user" | "assistant";
  content: string;
}

export interface StandaloneOptions {
  /** The model an `llm()` that names none runs on (route-prefixed). Default: `$DAI_MODEL`. */
  model?: string;
  /** Reaches the models. Default: `createModelRouter()`, keys from the environment. */
  router?: ModelRouter;
  /**
   * Makes one model call — the seam a test or another transport replaces. Default: `executeLlmCall`
   * over `router`. Resolves the call's value (text, or the parsed structure) and what it cost.
   */
  execute?: (definition: StandaloneLlmDefinition) => Promise<{ value: unknown; costUsd?: number } | { error: string; costUsd?: number }>;
  /** Runs `agent()`. Default: the Claude Code CLI on the PATH. */
  agentQuery?: AgentQuery;
  /** The directory agents work in and `workflow()` names resolve from. Default: `process.cwd()`. */
  cwd?: string;
  /** Reads the person's next message; `null` ends the input. Default: a line from stdin. */
  read?: (prompt: string) => Promise<string | null>;
  /** Where `phase()` and `log()` write. Default: stderr. */
  write?: (line: string) => void;
  /**
   * Loads a script by file URL — what `workflow()` and the runner import with. Default: `import()`.
   * `dai-script` passes tsx's `tsImport` where Node cannot run TypeScript itself, which loads a `.ts`
   * as ES modules whatever the folder's `package.json` says.
   */
  importer?: (url: string) => Promise<unknown>;
}

/** Conversations held in memory, addressed the way the engine's are: `<stream>@<seq>`. */
export class StandaloneConversations {
  private readonly positions = new Map<string, { turns: StandaloneTurn[]; agentSession?: string }>();
  private readonly heads = new Map<string, number>();
  private streams = 0;

  /** Where a call that names `ref` starts: its history, and the stream it appends to. */
  start(ref: unknown): { stream: string; seq: number; turns: StandaloneTurn[]; agentSession?: string } {
    if (ref === null || ref === undefined) return { stream: `s${++this.streams}`, seq: 0, turns: [] };
    const id = typeof ref === "object" && typeof (ref as { id?: unknown }).id === "string" ? (ref as { id: string }).id : undefined;
    if (id === undefined) throw new TypeError("a session is a ref — { id } — as a call that asked withSession handed it back");
    const at = this.positions.get(id);
    if (at === undefined) throw new Error(`no conversation is at '${id}' in this process — a standalone run keeps its conversations in memory`);
    const split = id.lastIndexOf("@");
    const stream = id.slice(0, split);
    const seq = Number(id.slice(split + 1));
    // Continuing a stream's head appends to it; continuing an earlier position branches, as the engine's do.
    if (this.heads.get(stream) === seq) return { stream, seq, turns: at.turns, ...(at.agentSession !== undefined ? { agentSession: at.agentSession } : {}) };
    return { stream: `s${++this.streams}`, seq: 0, turns: at.turns, ...(at.agentSession !== undefined ? { agentSession: at.agentSession } : {}) };
  }

  /** Record where a call ended, and return the ref of that position. */
  append(from: { stream: string; seq: number }, turns: StandaloneTurn[], agentSession?: string): string {
    const seq = from.seq + 1;
    const id = `${from.stream}@${seq}`;
    this.positions.set(id, { turns, ...(agentSession !== undefined ? { agentSession } : {}) });
    this.heads.set(from.stream, seq);
    return id;
  }

  /** The turns at a position — what a conversation said up to there. */
  turnsAt(id: string): StandaloneTurn[] | undefined {
    return this.positions.get(id)?.turns;
  }
}

/** What the standalone host hands a model call: the model, the conversation so far and this turn, the shape. */
export interface StandaloneLlmDefinition {
  model: string;
  system?: string;
  messages: StandaloneTurn[];
  schema?: JsonSchema;
  [knob: string]: unknown;
}

/** The default model call: `executeLlmCall` through a model router. */
function routerExecute(router: () => ModelRouter): NonNullable<StandaloneOptions["execute"]> {
  return async (definition) => {
    const result = await executeLlmCall(definition as never, { modelRouter: router() });
    const costUsd = (result.metrics as { costUsd?: number }).costUsd ?? 0;
    if ("error" in result && result.error !== undefined) return { error: (result.error as { reason?: string }).reason ?? String(result.error), costUsd };
    return { value: (result.value as { value?: unknown }).value, costUsd };
  };
}

/** A failed call, thrown into the script with the reason it failed. */
export class StandaloneCallError extends Error {}

/**
 * A loaded script's default export — through a CommonJS interop wrapper too (`default.default`), which
 * is how tsx hands back a `.ts` it loaded where the folder does not say `"type": "module"`.
 */
export function defaultExportOf(mod: unknown): unknown {
  const d = (mod as { default?: unknown } | null)?.default;
  if (typeof d === "function") return d;
  const inner = (d as { default?: unknown } | null | undefined)?.default;
  return typeof inner === "function" ? inner : d;
}

/** Strip a route prefix an agent binary would not understand: `claude-code/sonnet` → `sonnet`. */
function agentModel(model: string | undefined): string | undefined {
  if (model === undefined) return undefined;
  const slash = model.indexOf("/");
  const name = slash < 0 ? model : model.slice(slash + 1);
  return name === "default" ? undefined : name;
}

/** Render a prompt import's `{{.inputs.x}}` holes from its inputs. */
function renderTemplate(template: string, inputs: Record<string, unknown> | undefined): string {
  return template.replace(/\{\{\s*\.inputs\.([A-Za-z_$][\w$]*)\s*\}\}/g, (_m, key: string) => {
    const value = inputs?.[key];
    return typeof value === "string" ? value : JSON.stringify(value ?? "");
  });
}

/** Parse a parameter list off a function's source — how a named `workflow()` call binds positionally. */
function parameterNames(fn: (...args: unknown[]) => unknown): string[] {
  const source = Function.prototype.toString.call(fn);
  const open = source.indexOf("(");
  const close = source.indexOf(")", open);
  if (open < 0 || close < 0) return [];
  return source
    .slice(open + 1, close)
    .split(",")
    .map((p) => p.trim().replace(/[=:].*$/s, "").replace(/^\.\.\./, "").trim())
    .filter((p) => /^[A-Za-z_$][\w$]*$/.test(p));
}

/** The default `read`: one line from stdin, `null` at end of input. */
function stdinReader(): (prompt: string) => Promise<string | null> {
  let rl: ReturnType<typeof createInterface> | undefined;
  let closed = false;
  const lines: string[] = [];
  const waiting: Array<(line: string | null) => void> = [];
  const ensure = (): void => {
    if (rl !== undefined) return;
    rl = createInterface({ input: process.stdin, terminal: false });
    rl.on("line", (line) => {
      const next = waiting.shift();
      if (next !== undefined) next(line);
      else lines.push(line);
    });
    rl.on("close", () => {
      closed = true;
      for (const w of waiting.splice(0)) w(null);
    });
  };
  return (prompt) => {
    ensure();
    if (prompt.length > 0) process.stderr.write(prompt.endsWith(" ") ? prompt : `${prompt} `);
    const queued = lines.shift();
    if (queued !== undefined) return Promise.resolve(queued);
    if (closed) return Promise.resolve(null);
    return new Promise((resolve) => waiting.push(resolve));
  };
}

/** `"4h"`, `"30m"`, `"90s"` or a number of seconds, as milliseconds. */
export function idleMs(idle: number | string | undefined): number | undefined {
  if (idle === undefined) return undefined;
  if (typeof idle === "number") return idle * 1000;
  const m = /^(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?$/.exec(idle.trim());
  if (m === null) throw new TypeError(`input({ idle }): '${idle}' is not a duration — write seconds, or "30m", "4h"`);
  const n = Number(m[1]);
  const unit = m[2] ?? "s";
  return n * { ms: 1, s: 1000, m: 60_000, h: 3_600_000, d: 86_400_000 }[unit]!;
}

/** Build the standalone host. */
export function createStandaloneHost(options: StandaloneOptions = {}): ScriptHost & { conversations: StandaloneConversations } {
  const conversations = new StandaloneConversations();
  const cwd = options.cwd ?? process.cwd();
  const write = options.write ?? ((line: string) => void process.stderr.write(`${line}\n`));
  let router = options.router;
  const execute = options.execute ?? routerExecute(() => (router ??= createModelRouter()));
  let agentQuery = options.agentQuery;
  let read = options.read;
  const importer = options.importer ?? ((url: string) => import(url));
  /** The version of each script `workflow()` last ran, by file — to say when a newer one takes over. */
  const versions = new Map<string, string>();
  let spent = 0;

  const llm = async (call: LlmCall, typed: JsonSchema | undefined): Promise<unknown> => {
    const config = (call.config ?? {}) as Record<string, unknown>;
    const model = (typeof call.model === "string" ? call.model : undefined) ?? (typeof config.model === "string" ? config.model : undefined) ?? options.model ?? process.env.DAI_MODEL;
    if (model === undefined) throw new StandaloneCallError("llm() names no model — pass { model }, or set DAI_MODEL (route-prefixed: anthropic/claude-sonnet-5)");
    const prompt = typeof call.template === "string" ? renderTemplate(call.template, call.inputs) : call.prompt;
    const schema = (call.output?.schema ?? typed) as JsonSchema | undefined;
    const text = schema === undefined || JSON.stringify(schema) === '{"type":"string"}';
    // Structured output wants an object at the root: anything else goes over the wire as `{ value }`.
    const wrapped = !text && (schema as { type?: unknown }).type !== "object";
    const sent = text ? undefined : wrapped ? ({ type: "object", properties: { value: schema }, required: ["value"] } as JsonSchema) : schema;
    const from = conversations.start(call.session);
    const { model: _m, ...knobs } = config;
    const def: StandaloneLlmDefinition = {
      ...knobs,
      model,
      ...(typeof call.system === "string" ? { system: call.system } : {}),
      messages: [...from.turns, { role: "user" as const, content: prompt }],
      ...(sent !== undefined ? { schema: sent } : {}),
    };
    const result = await execute(def);
    spent += result.costUsd ?? 0;
    if ("error" in result) {
      if ("failureValue" in call) return call.failureValue;
      throw new StandaloneCallError(result.error);
    }
    const raw = result.value;
    const value = wrapped && raw !== null && typeof raw === "object" ? (raw as { value?: unknown }).value : raw;
    const said = typeof raw === "string" ? raw : JSON.stringify(raw);
    const id = conversations.append(from, [...from.turns, { role: "user", content: prompt }, { role: "assistant", content: said }]);
    return call.withSession === true ? { value, session: { id } } : value;
  };

  const agent = async (prompt: string, opts: AgentOptions, typed: JsonSchema | undefined): Promise<unknown> => {
    agentQuery ??= createCliAgentQuery();
    const output = opts.output as { schema?: JsonSchema } | undefined;
    const schema = (opts.schema ?? output?.schema ?? typed) as JsonSchema | undefined;
    const from = conversations.start(opts.session);
    const configured = (opts.config as { model?: unknown } | undefined)?.model;
    const model = agentModel(typeof opts.model === "string" ? opts.model : typeof configured === "string" ? configured : undefined);
    const query: AgentQueryOptions = {
      prompt,
      cwd,
      ...(model !== undefined ? { model } : {}),
      ...(schema !== undefined && JSON.stringify(schema) !== '{"type":"string"}' ? { schema } : {}),
      ...(from.agentSession !== undefined ? { resume: from.agentSession } : {}),
    } as AgentQueryOptions;
    let answer: unknown = null;
    let failed: string | undefined;
    let providerSession: string | undefined;
    try {
      for await (const message of agentQuery(query)) {
        const sid = (message as { sessionId?: unknown }).sessionId;
        if (typeof sid === "string") providerSession = sid;
        if (message.error !== undefined) failed = message.error;
        if (message.type === "result" && message.result !== undefined) {
          spent += message.result.costUsd ?? 0;
          answer = schema !== undefined && JSON.stringify(schema) !== '{"type":"string"}' ? (message.result.structured ?? null) : message.result.text;
          if (schema !== undefined && message.result.structured === undefined) failed ??= "the agent did not produce the answer's shape";
        }
      }
    } catch (e) {
      failed = (e as Error).message;
    }
    // Claude's null contract: an agent() that failed resolves to its failureValue, `null` by default.
    if (failed !== undefined) {
      write(`agent(): ${failed}`);
      return "failureValue" in opts ? opts.failureValue : null;
    }
    const said = typeof answer === "string" ? answer : JSON.stringify(answer);
    const id = conversations.append(from, [...from.turns, { role: "user", content: prompt }, { role: "assistant", content: said }], providerSession);
    return opts.withSession === true ? { value: answer, session: { id } } : answer;
  };

  const input = async (opts: InputOptions): Promise<unknown> => {
    read ??= stdinReader();
    const line = read(opts.prompt ?? ">");
    const ms = idleMs(opts.idle);
    const answer = ms === undefined ? await line : await Promise.race([line, new Promise<"idle">((r) => setTimeout(() => r("idle"), ms).unref?.())]);
    if (answer === "idle") return { idle: true };
    if (answer === null) return null;
    return { text: answer };
  };

  const workflow = async (ref: unknown, args: unknown): Promise<unknown> => {
    const named = typeof ref === "string" ? ref : typeof ref === "object" && ref !== null && typeof (ref as { scriptPath?: unknown }).scriptPath === "string" ? (ref as { scriptPath: string }).scriptPath : undefined;
    if (named === undefined) throw new StandaloneCallError("workflow() names a script by its path or its state id");
    const candidates = /\.(ts|mts|js|mjs)$/.test(named)
      ? [isAbsolute(named) ? named : resolve(cwd, named)]
      : [".jaira/workflows", "workflows", "."].flatMap((dir) => [".ts", ".js", ".mjs"].map((ext) => resolve(cwd, dir, `${named}${ext}`)));
    const file = candidates.find((c) => existsSync(c));
    if (file === undefined) throw new StandaloneCallError(`workflow('${named}'): no script found — a standalone run can call scripts, not JSON states (looked in ${candidates.join(", ")})`);
    // Non-durable HANDOFF (decision 0029 §3): a call is a boundary, so each call imports the script as
    // it stands now — named by its content, so an unchanged file is the module already loaded and an
    // edited one is a fresh import Node's module cache cannot answer with the old code.
    const version = createHash("sha256").update(readFileSync(file, "utf8")).digest("hex").slice(0, 12);
    const was = versions.get(file);
    if (was !== undefined && was !== version) write(`▸ ${named}: running a newer version (${version})`);
    versions.set(file, version);
    const main = defaultExportOf(await importer(`${pathToFileURL(file).href}?v=${version}`));
    if (typeof main !== "function") throw new StandaloneCallError(`workflow('${named}'): ${file} has no default export to call`);
    const fn = main as (...a: unknown[]) => unknown;
    if (Array.isArray(args)) return fn(...args);
    if (args !== null && typeof args === "object") {
      const names = parameterNames(fn);
      const keys = Object.keys(args);
      if (names.length > 0 && keys.every((k) => names.includes(k))) return fn(...names.map((n) => (args as Record<string, unknown>)[n]));
    }
    return args === undefined ? fn() : fn(args);
  };

  return {
    conversations,
    llm,
    agent,
    input,
    workflow,
    call: async (ref) => {
      throw new StandaloneCallError(`'${ref}' is a registered function — it exists only where an engine registered it; run this script durably to call it`);
    },
    recorded: async (_name, fn) => fn(),
    phase: (title) => write(`▸ ${title}`),
    log: (message) => write(message),
    now: () => Date.now(),
    random: () => Math.random(),
    budget: {
      total: null,
      spent: () => spent,
      remaining: () => Number.POSITIVE_INFINITY,
    },
  };
}

export type { JsonValue };
