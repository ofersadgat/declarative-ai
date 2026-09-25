/**
 * Running a compiled script's code (SCRIPTS.md §6.2, §9, §11) — the engine's half of `$script`.
 *
 * The engine owns dispatch, recording and replay; this module owns the pieces that are only about
 * scripts, so `engine.ts` holds the integration and not the detail:
 *
 *  - {@link ScriptSegments} prepares a segment's module — the code a compiled state runs, plus the
 *    import closure it reaches — through the same loader, resolver and approval gate a function module
 *    goes through (SPEC §7.5.4, §7.5.5).
 *  - {@link lowerLlmCall} turns an `llm()` call into the prompt operation it is, over the environment
 *    the state resolves in.
 *  - {@link ScriptRunState} is one run of a segment: the ordinal of every call it makes, what it has
 *    spent, and the replay scheduler that hands back recorded answers in the order they first settled.
 */
import type { Failure, InlineFamily, JsonValue, Operation } from "@declarative-ai/exec";
import type { JsonSchema } from "@declarative-ai/json";
import { isSubschema, type Schema } from "@declarative-ai/validate";
import type { ExecEnvironmentDecl, OperationFields } from "./format.js";
import { desugarOperation, splitExecEnvironment } from "./loader.js";
import type { ScriptCallRecord } from "./load.js";
import { prepareModules } from "./moduleLoader.js";
import type { Vfs } from "./reference.js";
import type { AgentOptions, LlmCall } from "./scriptHooks.js";
import { segmentPath } from "./scriptCompile.js";

/** Where a script's imports resolve, and what an `agent()` call defaults to — the host's seam. */
export interface ScriptModuleOptions {
  /** The filesystem imports are read from. Absent ⇒ a script may import nothing. */
  vfs?: Vfs;
  requirePath?: readonly string[];
  roots?: Readonly<Record<string, string>>;
  /** Whether an imported file may run (SPEC §7.5.5). The segment's own code is part of the definition. */
  approved?: (file: string) => boolean;
  /**
   * The layer every `agent()` call starts from, under its own options (SCRIPTS.md §9) — the tools and
   * configuration "an agent" means on this host. Policy, so the host's to say.
   */
  agent?: OperationFields;
  /** The most calls one instance's code may make — Claude's runaway backstop. Default 1000. */
  maxCalls?: number;
}

/** A phase's code, loaded: run it from an entry with its variables. */
export type SegmentFunction = (vars: Record<string, unknown>, entry: number) => Promise<unknown>;

const NO_FILES: Vfs = { list: () => [], read: () => undefined };

export class ScriptSegments {
  /** Transpiled code by content hash, shared across every load. */
  private readonly transpiled = new Map<string, string>();

  constructor(private readonly options: ScriptModuleOptions = {}) {}

  /**
   * Prepare and run a segment's module, returning its entry function. The module is EXECUTED afresh
   * each call — a phase's module-level code is part of the phase — while its transpilation is cached.
   */
  async load(code: string, file: string): Promise<SegmentFunction> {
    const path = segmentPath(file, code);
    const prepared = await prepareModules([path], {
      vfs: this.options.vfs ?? NO_FILES,
      requirePath: this.options.requirePath ?? [],
      ...(this.options.roots !== undefined ? { roots: this.options.roots } : {}),
      // The segment is part of the workflow's definition, hashed with it; only what it IMPORTS is a
      // file the approval gate is asked about.
      approved: (f) => f === path || (this.options.approved?.(f) ?? true),
      cache: this.transpiled,
      sources: { [path]: code },
    });
    const namespace = prepared.execute(path) as { default?: unknown };
    if (typeof namespace.default !== "function") throw new Error(`the compiled code of '${file}' exports no segment`);
    return namespace.default as SegmentFunction;
  }

  get maxCalls(): number {
    return this.options.maxCalls ?? 1000;
  }

  get agentDefaults(): OperationFields | undefined {
    return this.options.agent;
  }
}

/** An `llm()` call as the operation it dispatches, and what its answer needs on the way back. */
export interface LoweredCall {
  op: Operation<InlineFamily>;
  env: ExecEnvironmentDecl;
  /** The typed return was wrapped as `{ value }` for the wire, and is unwrapped on the way back. */
  unwrap: boolean;
  hasFailureValue: boolean;
  failureValue?: unknown;
  label?: string;
  phase?: string;
  /** The call asked for a private workspace — `agent({ isolation: "worktree" })`. */
  isolated: boolean;
}

/** Fields of an `llm()` call that are not operation knobs. */
const CALL_OWN = new Set(["prompt", "system", "config", "output", "failureValue", "label", "phase"]);

/**
 * Lower one `llm()` call (SCRIPTS.md §9) over the defaults its state resolves in.
 *
 * The prompt goes in VERBATIM: it is bound as an input and the template is `{{.inputs.prompt}}`, and
 * a render does not re-render what it substitutes — so braces in a prompt are text, as they are in
 * the JavaScript string that built it. A call gets a FRESH conversation unless it names one (§10).
 */
export function lowerLlmCall(stateId: string, defaults: OperationFields | undefined, call: LlmCall, typed: JsonSchema | undefined): LoweredCall {
  if (typeof call.prompt !== "string") throw new TypeError("llm() needs a string 'prompt'");
  const fields: Record<string, unknown> = { ...(defaults ?? {}) };
  delete fields.session;
  fields.session = null;
  let isolated = false;
  for (const [key, value] of Object.entries(call)) {
    if (CALL_OWN.has(key) || value === undefined) continue;
    if (key === "workspace" && value === null) {
      isolated = true;
      continue;
    }
    fields[key] = value;
  }
  if (call.config !== undefined) Object.assign(fields, call.config);

  const written = call.output?.schema;
  if (typed !== undefined && written !== undefined) {
    const a = isSubschema(typed as Schema, written as Schema);
    const b = isSubschema(written as Schema, typed as Schema);
    if (!a.ok || !b.ok) {
      throw new TypeError(`llm(): the type argument and output.schema describe different values — ${(!a.ok ? a.reason : b.reason) ?? "they disagree"}`);
    }
  }
  const schema = written ?? typed;
  const text = schema === undefined && call.output?.kind !== "json";
  const object = schema !== undefined && (schema as { type?: unknown }).type === "object";
  const output = text
    ? { text: { kind: "text" } }
    : object || schema === undefined
      ? { value: { kind: "json", ...(schema !== undefined ? { schema } : {}) } }
      : { value: { schema } };
  const { op: opFields, env } = splitExecEnvironment(fields as OperationFields);
  const op = desugarOperation(
    {
      ...opFields,
      kind: "prompt",
      prompt: "{{.inputs.prompt}}",
      ...(call.system !== undefined ? { system: call.system } : {}),
      input: { prompt: { kind: "text", binding: { text: call.prompt } } },
      output: output as unknown as OperationFields["output"],
    } as OperationFields,
    stateId,
  );
  return {
    op,
    env,
    unwrap: !text && !object && schema !== undefined,
    hasFailureValue: "failureValue" in call,
    ...("failureValue" in call ? { failureValue: call.failureValue } : {}),
    ...(typeof call.label === "string" ? { label: call.label } : {}),
    ...(typeof call.phase === "string" ? { phase: call.phase } : {}),
    isolated,
  };
}

/**
 * Claude's `agent(prompt, opts)` as the `llm()` call it is (SCRIPTS.md §9): the host's agent layer,
 * then the options lowered, with `null` on failure.
 */
export function agentCall(prompt: string, options: AgentOptions, hostDefaults: OperationFields | undefined): LlmCall {
  const { schema, model, effort, isolation, agentType, phase, label, config, output, ...rest } = options;
  const call: LlmCall = {
    ...((hostDefaults ?? {}) as Record<string, unknown>),
    ...rest,
    prompt,
    failureValue: "failureValue" in options ? options.failureValue : null,
  };
  const knobs: Record<string, unknown> = { ...(config ?? {}) };
  if (model !== undefined) knobs.model = model;
  if (effort !== undefined) knobs.reasoning = { effort };
  if (agentType !== undefined) knobs.configRef = agentType;
  if (Object.keys(knobs).length > 0) call.config = { ...(call.config ?? {}), ...knobs };
  if (isolation === "worktree") call.workspace = null;
  if (schema !== undefined || output !== undefined) call.output = { ...(output ?? {}), ...(schema !== undefined ? { schema } : {}) };
  if (typeof label === "string") call.label = label;
  if (phase !== undefined) call.phase = phase;
  return call;
}

/**
 * One run of a segment's code: the ORDINAL of every call it makes, what it has spent, and the replay
 * of what an earlier run of the same code already recorded.
 *
 * A call's site is `<content hash>#<k>`, k counting the identical calls this run has made before it.
 * The ordinal is what keeps three skeptics asked one prompt three draws rather than one memo hit —
 * and it restarts with the run, so a replay of the same code assigns the same sites.
 */
export class ScriptRunState {
  private readonly ordinals = new Map<string, number>();
  spent = 0;
  calls = 0;
  /** The phase `phase()` last named, in `"state"` mode — what a call is grouped under by default. */
  phase: string | undefined;
  private readonly recorded: Map<string, { record: ScriptCallRecord; order: number }>;
  private readonly waiting: Array<{ order: number; release: () => void }> = [];
  private scheduled = false;

  constructor(
    records: readonly ScriptCallRecord[] | undefined,
    /** Prepended to every site — a function call's own, so its calls never share a site with the state's. */
    private readonly prefix = "",
  ) {
    this.recorded = new Map((records ?? []).map((record, order) => [record.site, { record, order }]));
  }

  /** The next site for a call with this content key. */
  site(key: string): string {
    const k = this.ordinals.get(key) ?? 0;
    this.ordinals.set(key, k + 1);
    return `${this.prefix}${key}#${k}`;
  }

  /** What an earlier run recorded at this site, if anything. */
  recordAt(site: string): ScriptCallRecord | undefined {
    return this.recorded.get(site)?.record;
  }

  /**
   * Wait for a recorded answer's TURN (SCRIPTS.md §11): whenever the code is quiescent, the
   * earliest-settled recorded answer it is waiting on is released — so a race it ran the first time
   * resolves the same way, and a call it issues after one answer is issued before the next arrives.
   */
  release(site: string): Promise<void> {
    const order = this.recorded.get(site)?.order ?? Number.MAX_SAFE_INTEGER;
    return new Promise((release) => {
      this.waiting.push({ order, release });
      this.schedule();
    });
  }

  private schedule(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    setImmediate(() => {
      this.scheduled = false;
      this.waiting.sort((a, b) => a.order - b.order);
      this.waiting.shift()?.release();
      if (this.waiting.length > 0) this.schedule();
    });
  }
}

/** A failed call, thrown into the script: its classification travels with it. */
export class ScriptCallError extends Error {
  constructor(readonly failure: Failure) {
    super(failure.reason);
  }
}

/** The value a recorded or live answer resolves to in the script. */
export function answerOf(lowered: LoweredCall, value: unknown): unknown {
  if (!lowered.unwrap) return value;
  return value !== null && typeof value === "object" ? (value as { value?: unknown }).value : value;
}

export type { JsonValue };
