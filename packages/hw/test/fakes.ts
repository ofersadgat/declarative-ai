/**
 * Test doubles for the capability model: a scripted PROMPT `Executor` (dispatching on the op's
 * configured model name) and scripted registry entries. The fakes stand in for
 * `@declarative-ai/promptop`'s executor and for host UI functions — the engine only ever sees the
 * `@declarative-ai/exec` contracts, which is exactly what lets hw be tested with no LLM in the graph.
 */
import {
  failureOf,
  hostFunction,
  isOk,
  newCapabilityRegistry,
  type Capabilities,
  type CapabilityRegistry,
  type ExecHandle,
  type ExecResult,
  type ExecServices,
  type Executor,
  type Failure,
  type FunctionInputs,
  type FunctionResult,
  type HostCapabilities,
  type InlineFamily,
  type JsonValue,
  type Operation,
  type PromptOp,
  type ResolvedValue,
  type Tool,
} from "@declarative-ai/exec";
import { mergeWorkflowMetrics, type WorkflowMetrics } from "../src/ports";

const CAPS: Capabilities = {
  structuredOutput: true,
  sessionResume: false,
  streaming: false,
  interactive: false,
  readOnly: true,
  mutatesWorkspace: false,
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

async function* empty(): AsyncGenerator<never> {}

/** A recorded prompt-op invocation: the resolved op plus the services it ran with. The old
 *  `PromptOpEnvironment` is gone — its three fields are `ExecServices` fields now (§4.1). */
export interface FakeCall {
  op: PromptOp<InlineFamily>;
  ctx: ExecServices;
  /** The op's configured model — what a state's `operation.config.model` selected. */
  name: string;
}

export type Script = (call: FakeCall) => ExecResult<ResolvedValue, WorkflowMetrics> | Promise<ExecResult<ResolvedValue, WorkflowMetrics>>;

export const ok = (value: JsonValue, costUsd = 0.01): ExecResult<ResolvedValue, WorkflowMetrics> => ({
  value,
  // No `rawText`: the model's raw text is `LlmOutput` payload and stops at the prompt executor. What
  // an execution returns is the op's output-parameter value.
  metrics: { durationMs: 1, costUsd, costSource: "table" },
});

/** A promise you resolve from the test body. */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (v: T) => void;
}
export function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

/** A fake prompt `Executor` backed by ONE script, sharing a call log. */
export class FakePromptExecutor implements Executor<ExecServices, WorkflowMetrics> {
  readonly metrics = { merge: mergeWorkflowMetrics };
  readonly calls: FakeCall[] = [];
  readonly capabilities = CAPS;
  constructor(private readonly script: Script) {}

  start(operation: Operation<InlineFamily>, ctx: ExecServices): ExecHandle<ResolvedValue, WorkflowMetrics> {
    const op = operation as PromptOp<InlineFamily>;
    const config = op.config !== null && typeof op.config === "object" && !Array.isArray(op.config) ? op.config : {};
    const call: FakeCall = { op, ctx, name: typeof config.model === "string" ? config.model : "" };
    this.calls.push(call);
    const outcome = (async (): Promise<ExecResult<ResolvedValue, WorkflowMetrics>> => {
      const canceled = new Promise<ExecResult<ResolvedValue, WorkflowMetrics>>((resolve) => {
        const onAbort = (): void => resolve({ error: { classification: "canceled", reason: "aborted" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } });
        if (ctx.abortSignal?.aborted) onAbort();
        else ctx.abortSignal?.addEventListener("abort", onAbort, { once: true });
      });
      try {
        return await Promise.race([Promise.resolve(this.script(call)), canceled]);
      } catch (e) {
        return { error: { classification: "permanent", reason: (e as Error).message }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } };
      }
    })();
    return { events: empty(), result: outcome, cancel: async () => {} };
  }
}

/** The model a recorded call dispatched to (the fixture's `operation.config.model`). */
export function modelOf(call: FakeCall): string {
  return call.name;
}

export function promptOf(call: FakeCall): string {
  return call.op.user ?? "";
}

/** The rendered template tail — the prompt with any conversation-history preamble
 *  stripped, so scripts can dispatch on content without matching history echoes. */
export function promptTail(call: FakeCall): string {
  const p = promptOf(call);
  const marker = "</conversation-history>";
  const at = p.lastIndexOf(marker);
  return at < 0 ? p.trim() : p.slice(at + marker.length).trim();
}

/** The tools the call was given, by name. */
export function toolNamesOf(call: FakeCall): string[] {
  return Object.keys(call.ctx.tools ?? {});
}

/** The default capabilities a scripted (interactive) host function declares. Required and TOTAL, per
 *  variant — there is no "registered but uncharacterized" entry any more (§2). */
export const INTERACTIVE: HostCapabilities = { interactive: true, readOnly: true, memoizable: false };

/** A scripted host function: returns queued responses FIFO; records its invocations. Its impl RESOLVES
 *  a `FunctionResult<ResolvedValue, WorkflowMetrics>` (§4.2), so an exhausted queue is a CLASSIFIED failure rather than a thrown exception the
 *  engine has to guess a classification for. */
export class ScriptedFunction {
  readonly calls: FunctionInputs[] = [];
  constructor(
    private readonly queue: JsonValue[],
    readonly capabilities: HostCapabilities = INTERACTIVE,
  ) {}

  readonly run = async (inputs: FunctionInputs): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => {
    this.calls.push(inputs);
    const next = this.queue.shift();
    if (next === undefined) return { error: failureOf(new Error("scripted function responses exhausted")) };
    return { value: next };
  };

  /** Register under `name` as a host entry. */
  register(registry: CapabilityRegistry<WorkflowMetrics>, name: string): this {
    registry.functions.set(name, hostFunction(this.run, this.capabilities));
    return this;
  }
}

/** A fresh capability registry with the one discriminated function map. */
export function newRegistry(): CapabilityRegistry<WorkflowMetrics> {
  return newCapabilityRegistry<WorkflowMetrics>();
}

/** A function that always fails — the search-context stand-in for a refused human gate. Errors as DATA
 *  (§4.2), which is the contract; {@link throwingFunction} is the impl that ignores it. */
export const rejectingFunction = async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({
  error: failureOf(new Error("interactive function not allowed in this context")),
});

/** A function that THROWS rather than resolving a `FunctionResult<ResolvedValue, WorkflowMetrics>`. Nothing at registration forces an impl
 *  through `liftThrowing`, so this is a shape the engine must survive: the workflow degrades per SPEC
 *  §3.3 instead of the exception escaping `engine.run()`. */
export const throwingFunction = async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => {
  throw new Error("interactive function not allowed in this context");
};

/** A trivial tool for permission/gating tests. */
export function fakeTool(name: string, readOnly = false): Tool {
  return {
    description: name,
    inputSchema: { type: "object" },
    readOnly,
    run: () => ({ ok: true }),
  };
}

/** Read a result's failure, or `undefined` when it succeeded — `error` is not a property of the union. */
export function errorOf<O, M extends { durationMs: number }>(r: ExecResult<O, M>): Failure | undefined {
  return isOk(r) ? undefined : r.error;
}
