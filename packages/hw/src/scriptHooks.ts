/**
 * The hooks a workflow script calls (SCRIPTS.md §6, §9) — `llm`, `agent`, `input`, `parallel`,
 * `pipeline`, `phase`, `log`, `workflow`, `now`, `random`, `budget`.
 *
 * ## One module, reached two ways
 *
 * A script names them as GLOBALS (Claude's spelling) or imports them from
 * `@declarative-ai/hw/script` (hw's). The compiler rewrites a global reference to the import, so both
 * spellings arrive here, and a `"function"`-mode module (SCRIPTS.md §11) reaches the same exports by
 * the same import.
 *
 * ## The CALLER is found, not passed
 *
 * The code calling a hook is user code: it may call it from a helper declared at module level, from a
 * callback three closures deep, or from a function module another state called. None of those can be
 * handed an argument naming the instance it runs for. So the engine RUNS the code inside an
 * `AsyncLocalStorage` scope holding a {@link ScriptHost} for the instance, and each hook reads the host
 * off the current async context. Two instances running concurrently each see their own host, because
 * each await chain carries its own store.
 *
 * `parallel` and `pipeline` are plain JavaScript over promises and need no host — they are Claude's
 * semantics exactly, and nothing about them is recorded: only the calls inside them are.
 *
 * ## Outside any engine: the standalone host
 *
 * A script is also a plain program (SCRIPTS.md §17): `@declarative-ai/hw/script` is an export of this
 * package, so `node my-workflow.ts` resolves it, and a call made with no engine's host in scope goes
 * to the STANDALONE host, when one is installed ({@link installStandaloneHost} —
 * `@declarative-ai/script` installs one that calls models and agents directly). The engine's host
 * always wins inside a run, so one file runs durably under the engine and non-durably under `node`.
 */
import type { JsonSchema } from "@declarative-ai/json";
import type { SessionRef } from "./session.js";

/** The specifier a script or function module imports the hooks from. */
export const HOOK_MODULE = "@declarative-ai/hw/script";

/** Everything `llm()` takes — the prompt operation of SPEC §7.1 with its environment (SCRIPTS.md §9). */
export interface LlmCall {
  prompt: string;
  system?: string;
  model?: string;
  /** Knobs merged flat over the call — `temperature`, `reasoning`, `configRef`, … */
  config?: Record<string, unknown>;
  output?: { schema?: JsonSchema; kind?: "text" | "json" };
  tools?: string[];
  session?: unknown;
  workspace?: unknown;
  permissions?: unknown;
  conversation?: unknown;
  /** What the call returns if it fails, in place of throwing. */
  failureValue?: unknown;
  /** A display label for the call's record. Not part of its identity. */
  label?: string;
  /**
   * Hand back where the call's conversation ENDED as well as its value: the call resolves to
   * `{ value, session }`, and `session` is the ref the next call continues from (`session: turn.session`)
   * or forks (SCRIPTS.md §9). Not part of the operation — it changes only what the script is handed.
   */
  withSession?: boolean;
  /**
   * A prompt IMPORT's template, rendered with `inputs` — `{{.inputs.x}}` holes and all — where a plain
   * `prompt` is text (SCRIPTS.md §10).
   */
  template?: string;
  inputs?: Record<string, unknown>;
  [knob: string]: unknown;
}

/** Claude's `agent()` options, and every `LlmCall` field besides (SCRIPTS.md §9). */
export interface AgentOptions extends Partial<Omit<LlmCall, "prompt">> {
  schema?: JsonSchema;
  effort?: string;
  isolation?: "worktree";
  agentType?: string;
  phase?: string;
}

/** What `llm({ …, withSession: true })` resolves to. */
export interface WithSession<T> {
  value: T;
  /** Where the call's conversation ended; `null` when the executor kept no conversation. */
  session: SessionRef | null;
}

/** What `input()` takes (SCRIPTS.md §17). */
export interface InputOptions {
  /** What the person is being asked — shown where they answer. */
  prompt?: string;
  /**
   * How long to wait for an answer before resolving `{ idle: true }` instead: seconds, or a string
   * like `"4h"`, `"30m"`. Absent ⇒ wait indefinitely.
   */
  idle?: number | string;
  /** A display label for the call's record. */
  label?: string;
  [option: string]: unknown;
}

/**
 * What `input()` resolves to: the person's message — its text, any files, and the settings they
 * chose for it — or `{ idle: true }` when `idle` elapsed, or `null` when they ended the run.
 */
export interface InputMessage {
  text: string;
  files?: unknown[];
  /** The per-message settings the person chose (model, reasoning, tools, …): the program decides. */
  settings?: Record<string, unknown>;
  idle?: false;
}
export type InputAnswer = InputMessage | { idle: true } | null;

/** The function a durable `input()` dispatches: a host registers the gate under this name. */
export const INPUT_FUNCTION = "input";

/** What one instance's script run can do — implemented by the engine, per instance. */
export interface ScriptHost {
  /** One `llm()` call: dispatched (or replayed) at its own site. Throws on failure. */
  llm(call: LlmCall, schema: JsonSchema | undefined): Promise<unknown>;
  /** One `agent()` call — `llm()` with agent defaults, `null` on failure. */
  agent(prompt: string, options: AgentOptions, schema: JsonSchema | undefined): Promise<unknown>;
  /** One `input()`: wait for a person's next message (SCRIPTS.md §17). */
  input?(options: InputOptions): Promise<unknown>;
  /** A called state (SCRIPTS.md §10). */
  workflow(ref: unknown, args: unknown): Promise<unknown>;
  /** A registered function a script imported from `$REGISTRY` (SCRIPTS.md §10). */
  call(ref: string, args: readonly unknown[]): Promise<unknown>;
  /** Code imported `with { as: "operation" }`: run once, recorded, read back on a replay. */
  recorded(name: string, fn: () => unknown): Promise<unknown>;
  phase(title: string): void;
  log(message: string): void;
  /** Journaled: recorded the first time, replayed after. */
  now(): number;
  random(): number;
  readonly budget: { total: number | null; spent(): number; remaining(): number };
}

/** The async-context store the engine runs script code in. Loaded once, on first use. */
interface Store {
  run<T>(host: ScriptHost, fn: () => T): T;
  getStore(): ScriptHost | undefined;
}
let store: Store | undefined;

/**
 * Load `AsyncLocalStorage` — the one await the hooks need, taken by the engine before it runs any
 * script code. Dynamic so this module stays importable where `node:async_hooks` is not.
 */
export async function loadScriptContext(): Promise<void> {
  if (store !== undefined) return;
  const { AsyncLocalStorage } = await import("node:async_hooks");
  store = new AsyncLocalStorage<ScriptHost>() as unknown as Store;
}

/** Whether {@link loadScriptContext} has completed. */
export function hasScriptContext(): boolean {
  return store !== undefined;
}

/** Run `fn` with `host` as the current script host. {@link loadScriptContext} must have completed. */
export function runWithScriptHost<T>(host: ScriptHost, fn: () => T): T {
  if (store === undefined) throw new Error("the script context is not loaded — await loadScriptContext() first");
  return store.run(host, fn);
}

/**
 * The standalone host lives on `globalThis`, under a registered symbol, so every copy of this module
 * a process loads (the built one, a source one a test imports) sees the same host.
 */
const STANDALONE = Symbol.for("@declarative-ai/hw/script#standalone");
type WithStandalone = { [STANDALONE]?: ScriptHost };

/**
 * Install the host a hook uses when no engine's host is in scope — what makes a script a plain program
 * (`node my-workflow.ts`). `undefined` removes it. Returns the host it replaced.
 */
export function installStandaloneHost(standalone: ScriptHost | undefined): ScriptHost | undefined {
  const g = globalThis as WithStandalone;
  const previous = g[STANDALONE];
  if (standalone === undefined) delete g[STANDALONE];
  else g[STANDALONE] = standalone;
  return previous;
}

/** The standalone host, if one is installed. */
export function standaloneHost(): ScriptHost | undefined {
  return (globalThis as WithStandalone)[STANDALONE];
}

function host(hook: string): ScriptHost {
  const current = store?.getStore() ?? standaloneHost();
  if (current === undefined) {
    throw new Error(
      `${hook}() was called outside a workflow run — it can only run inside a script or a function a workflow called, or under a standalone host (run the file with dai-script, or install one with installStandaloneHost)`,
    );
  }
  return current;
}

/** `llm("…")`, `llm("…", { … })` and `llm({ prompt, … })` are one call. */
function normalizeCall(first: unknown, second: unknown): LlmCall {
  if (typeof first === "string") return { ...((second as Partial<LlmCall> | undefined) ?? {}), prompt: first };
  if (first !== null && typeof first === "object" && typeof (first as LlmCall).prompt === "string") return first as LlmCall;
  throw new TypeError("llm() takes a prompt string, or an object with a 'prompt'");
}

interface Typed<F> {
  (...args: unknown[]): unknown;
  /**
   * The same hook with its OUTPUT CONTRACT fixed — what `llm<T>(…)` compiles to (SCRIPTS.md §9): the
   * type argument is gone at run time, and this carries the schema it was converted to.
   */
  withOutput(schema: JsonSchema): F;
}

function typedLlm(schema: JsonSchema | undefined): (first: unknown, second?: unknown) => Promise<unknown> {
  return (first, second) => host("llm").llm(normalizeCall(first, second), schema);
}
function typedAgent(schema: JsonSchema | undefined): (prompt: unknown, options?: unknown) => Promise<unknown> {
  return (prompt, options) => {
    if (typeof prompt !== "string") return Promise.reject(new TypeError("agent() takes a prompt string"));
    return host("agent").agent(prompt, (options as AgentOptions | undefined) ?? {}, schema);
  };
}

export const llm = Object.assign(typedLlm(undefined), { withOutput: (schema: JsonSchema) => typedLlm(schema) }) as unknown as Typed<ReturnType<typeof typedLlm>>;
export const agent = Object.assign(typedAgent(undefined), { withOutput: (schema: JsonSchema) => typedAgent(schema) }) as unknown as Typed<ReturnType<typeof typedAgent>>;

/** Claude's `parallel`: a barrier, and a thunk that throws resolves to `null`. */
export async function parallel(thunks: ReadonlyArray<() => unknown>): Promise<unknown[]> {
  if (!Array.isArray(thunks)) throw new TypeError("parallel() takes an array of functions");
  if (thunks.length > MAX_ITEMS) throw new RangeError(`parallel() takes at most ${MAX_ITEMS} items; got ${thunks.length}`);
  return Promise.all(thunks.map(async (thunk) => {
    try {
      return await thunk();
    } catch {
      return null;
    }
  }));
}

/**
 * Claude's `pipeline`: every item through every stage with no barrier between stages; each stage gets
 * `(previous, item, index)`, and a stage that throws drops that item to `null`.
 */
export async function pipeline(items: readonly unknown[], ...stages: Array<(prev: unknown, item: unknown, index: number) => unknown>): Promise<unknown[]> {
  if (!Array.isArray(items)) throw new TypeError("pipeline() takes an array of items");
  if (items.length > MAX_ITEMS) throw new RangeError(`pipeline() takes at most ${MAX_ITEMS} items; got ${items.length}`);
  return Promise.all(items.map(async (item, index) => {
    let value: unknown = item;
    try {
      for (const stage of stages) value = await stage(value, item, index);
      return value;
    } catch {
      return null;
    }
  }));
}

/** Claude's item cap for `parallel`/`pipeline` — an explicit error, never a silent truncation. */
const MAX_ITEMS = 4096;

/**
 * Wait for the person's next message (SCRIPTS.md §17). Under the engine it is a gate the run parks
 * at; under the standalone host it reads the terminal. Resolves `null` when the person ends the run.
 */
export function input(options?: InputOptions | string): Promise<unknown> {
  const h = host("input");
  if (h.input === undefined) return Promise.reject(new Error("this host cannot ask a person for input"));
  return h.input(typeof options === "string" ? { prompt: options } : (options ?? {}));
}

export function phase(title: string): void {
  host("phase").phase(String(title));
}
export function log(message: string): void {
  host("log").log(String(message));
}
export function workflow(ref: unknown, args?: unknown): Promise<unknown> {
  return host("workflow").workflow(ref, args);
}
/** A registered function, by name — what `import { f } from "$REGISTRY"` compiles to. */
export function call(ref: string, ...args: unknown[]): Promise<unknown> {
  return host("call").call(ref, args);
}
/** A recorded call of imported code — what `with { as: "operation" }` compiles to. */
export function recorded(name: string, fn: () => unknown): Promise<unknown> {
  return host("recorded").recorded(name, fn);
}
export function now(): number {
  return host("now").now();
}
export function random(): number {
  return host("random").random();
}

/** Claude's `budget` — read through the current host, so it is the running instance's. */
export const budget = {
  get total(): number | null {
    return host("budget").budget.total;
  },
  spent: (): number => host("budget").budget.spent(),
  remaining: (): number => host("budget").budget.remaining(),
};

/** The module a `require(HOOK_MODULE)` returns. */
export const hookModule: Readonly<Record<string, unknown>> = Object.freeze({
  llm,
  agent,
  input,
  parallel,
  pipeline,
  phase,
  log,
  workflow,
  call,
  recorded,
  now,
  random,
  budget,
});

/**
 * The hooks' TYPES, for the checker a script or a function module is read under (SCRIPTS.md §13):
 * the globals a Claude-style script uses, and the module an hw-style one imports.
 */
export const HOOK_GLOBALS_PATH = "/__hw_script__/globals.d.ts";
export const HOOK_MODULE_PATH = "/__hw_script__/index.d.ts";

const HOOK_TYPES = `
interface HwLlmCall {
  prompt: string;
  system?: string;
  model?: string;
  config?: Record<string, unknown>;
  output?: { schema?: object; kind?: "text" | "json" };
  tools?: string[];
  session?: unknown;
  workspace?: unknown;
  permissions?: unknown;
  conversation?: unknown;
  failureValue?: unknown;
  label?: string;
  withSession?: boolean;
  [knob: string]: unknown;
}
interface HwSessionRef { id: string }
interface HwInputOptions { prompt?: string; idle?: number | string; label?: string; [option: string]: unknown }
interface HwMessage { text: string; files?: unknown[]; settings?: Record<string, unknown>; idle?: false }
interface HwAgentOptions extends Partial<Omit<HwLlmCall, "prompt">> {
  schema?: object;
  effort?: string;
  isolation?: "worktree";
  agentType?: string;
  phase?: string;
}
interface HwBudget { readonly total: number | null; spent(): number; remaining(): number }
`;

const HOOK_SIGNATURES = (prefix: string): string => `
${prefix} function llm<T = string>(call: HwLlmCall & { withSession: true }): Promise<{ value: T; session: HwSessionRef | null }>;
${prefix} function llm<T = string>(prompt: string, call: Partial<HwLlmCall> & { withSession: true }): Promise<{ value: T; session: HwSessionRef | null }>;
${prefix} function llm<T = string>(call: HwLlmCall): Promise<T>;
${prefix} function llm<T = string>(prompt: string, call?: Partial<HwLlmCall>): Promise<T>;
${prefix} function agent<T = string>(prompt: string, options?: HwAgentOptions): Promise<T | null>;
${prefix} function input(options?: HwInputOptions | string): Promise<HwMessage | { idle: true } | null>;
${prefix} function parallel<T>(thunks: ReadonlyArray<() => Promise<T> | T>): Promise<Array<T | null>>;
${prefix} function pipeline(items: readonly any[], ...stages: Array<(prev: any, item: any, index: number) => any>): Promise<any[]>;
${prefix} function phase(title: string): void;
${prefix} function log(message: string): void;
${prefix} function workflow(ref: string | { scriptPath: string }, args?: unknown): Promise<any>;
${prefix} function now(): number;
${prefix} function random(): number;
${prefix} const budget: HwBudget;
`;

export const HOOK_GLOBALS_SOURCE = `${HOOK_TYPES}${HOOK_SIGNATURES("declare")}\ndeclare const args: any;\n`;
export const HOOK_MODULE_SOURCE = `${HOOK_TYPES}${HOOK_SIGNATURES("export declare")}`;
