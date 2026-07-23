import { describe, expect, it } from "vitest";
import type { ExecServices, InlineFamily, Operation, Tool } from "@declarative-ai/exec";
import { PromptExecutor, createPromptExecutor } from "../src/executor";
import { fakeRunner, okOutcome, promptOp, errorOf } from "./fakes";

describe("PromptExecutor (core) — outcome mapping", () => {
  it("declares the prompt capabilities as one TOTAL record (no separate ExecutorCapabilities)", () => {
    expect(new PromptExecutor().capabilities).toEqual({
      structuredOutput: true,
      sessionResume: false,
      streaming: true,
      interactive: false,
      readOnly: true,
      mutatesWorkspace: false,
      policyEnforcement: "none",
      memoizable: true,
      runtime: "edge-safe",
    });
  });

  it("projects an LlmCallResult onto the op's OUTPUT PARAMETER — the model payload stops here", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    const out = await createPromptExecutor({ runner }).start(promptOp(), {}).result;
    expect(errorOf(out)).toBeUndefined();
    // What execution returns is the value of the output parameter, and only that.
    expect(out.value).toEqual({ answer: "4" });
    // `rawText`, `thinking`, `toolCalls`, and `finishReason` are `LlmOutput` — provider payload. They
    // rode on the execution result before this refactor; nothing downstream ever read them.
    expect("rawText" in out).toBe(false);
    expect("finishReason" in out).toBe(false);
    expect("thinking" in out).toBe(false);
    // Metrics DO cross, because spend and timing are what execution aggregates.
    expect(out.metrics.costUsd).toBe(0.001);
    expect(typeof out.metrics.startMs).toBe("number");
  });

  it("carries a FAILED call's real cost through the projection — a failed call still costs money", async () => {
    // The provider generated and billed, then the output failed validation. The spend is real and has
    // to survive the LlmCallResult → ExecResult projection, or every budget above this under-charges.
    const { runner } = fakeRunner([
      okOutcome({ error: { classification: "api-retriable", reason: "validation failed" }, metrics: { costUsd: 0.004, costSource: "provider" } }),
    ]);
    const out = await createPromptExecutor({ runner }).start(promptOp(), {}).result;
    expect(errorOf(out)?.classification).toBe("api-retriable");
    expect(out.metrics.costUsd).toBe(0.004);
    expect(out.metrics.costSource).toBe("provider");
    // And the partial payload survives alongside it, which is what makes a repair turn possible.
    expect(out.value).toEqual({ answer: "4" });
  });

  it("never rejects for a call failure — the error rides on the outcome", async () => {
    const { runner } = fakeRunner([okOutcome({ error: { classification: "permanent", reason: "boom" } })]);
    const out = await createPromptExecutor({ runner }).start(promptOp(), {}).result;
    expect(errorOf(out)).toEqual({ classification: "permanent", reason: "boom" });
  });

  it("normalizes a thrown runner error into a permanent failure outcome", async () => {
    const runner = async (): Promise<never> => {
      throw new Error("wiring fault");
    };
    const out = await createPromptExecutor({ runner }).start(promptOp(), {}).result;
    expect(errorOf(out)).toMatchObject({ classification: "permanent", reason: "wiring fault" });
  });

  it("refuses a FUNCTION op — the prompt executor is not a general dispatcher", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    const fn: Operation<InlineFamily> = { kind: "function", functionRef: "f", input: {}, output: { name: "output", kind: "json" } };
    const out = await createPromptExecutor({ runner }).start(fn, {}).result;
    expect(errorOf(out)?.reason).toMatch(/handed a function operation/);
  });

  it("hands ctx.validator to the call environment", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const validator = { validateValue: () => ({ ok: true }) };
    await createPromptExecutor({ runner }).start(promptOp(), { validator }).result;
    expect(calls[0]!.env.validator).toBe(validator);
  });

  it("passes ctx.timeoutMs through as the per-call budget (the field PromptOpEnvironment used to carry)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    await createPromptExecutor({ runner }).start(promptOp(), { timeoutMs: 1234 }).result;
    expect(calls[0]!.timeoutMs).toBe(1234);
  });
});

describe("core — loud failure on unconsumed wrapper fields", () => {
  it("refuses a ctx.deadline the bare core cannot honor (compose withDeadline)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const out = await createPromptExecutor({ runner }).start(promptOp(), { deadline: { maxDurationMs: 1000 }, stepStartMs: 0 }).result;
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/compose withDeadline/);
    expect(calls).toHaveLength(0);
  });

  it("refuses a sessionId no session layer consumed (compose withSession)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const out = await createPromptExecutor({ runner }).start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/compose withSession/);
    expect(calls).toHaveLength(0);
  });
});

describe("core — cancel", () => {
  it("cancel() aborts the in-flight call and the outcome resolves with classification 'canceled'", async () => {
    let seen: AbortSignal | undefined;
    const runner = async (_def: unknown, env: { abortSignal?: AbortSignal }): Promise<ReturnType<typeof okOutcome>> => {
      seen = env.abortSignal;
      await new Promise<void>((r) => env.abortSignal?.addEventListener("abort", () => r(), { once: true }));
      return okOutcome({ error: { classification: "network-retriable", reason: "aborted" } });
    };
    const handle = createPromptExecutor({ runner: runner as never }).start(promptOp(), {});
    await new Promise((r) => setTimeout(r, 0));
    void handle.cancel();
    const out = await handle.result;
    expect(seen?.aborted).toBe(true);
    expect(errorOf(out)?.classification).toBe("canceled");
  });

  it("a pre-aborted ctx.abortSignal maps the failure to 'canceled'", async () => {
    const controller = new AbortController();
    controller.abort();
    const { runner } = fakeRunner([okOutcome({ error: { classification: "network-retriable", reason: "stopped" } })]);
    const out = await createPromptExecutor({ runner }).start(promptOp(), { abortSignal: controller.signal }).result;
    expect(errorOf(out)?.classification).toBe("canceled");
  });
});

describe("core — tools and blob outputs", () => {
  it("declares ctx.tools on the definition AND adapts their impls into the call environment", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const tool: Tool = {
      description: "adds",
      inputSchema: { type: "object", properties: { a: { type: "number" } } },
      readOnly: true,
      run: (input: Record<string, unknown>) => ({ sum: Number(input.a) + 1 }),
    };
    const ctx: ExecServices = { tools: { add: tool } };
    await createPromptExecutor({ runner }).start(promptOp(), ctx).result;
    expect(calls[0]!.def.tools).toEqual([{ name: "add", description: "adds", inputSchema: tool.inputSchema }]);
    const exec = calls[0]!.env.toolExecutors!.add!;
    expect(await exec({ a: 1 }, {} as never)).toEqual({ sum: 2 });
  });

  it("tools given at CONSTRUCTION are declared AND executable — the loop must not silently degrade", async () => {
    // The declarations came from `ctx.tools ?? options.tools` while the executors came from `ctx.tools`
    // alone, so a construction-time tool was announced to the model with nothing able to run it:
    // `call.ts`'s `executable` check goes false, `stopWhen` is never set, and the bounded tool LOOP
    // becomes a single turn that returns an unexecuted tool call.
    const { runner, calls } = fakeRunner([okOutcome()]);
    const tool: Tool = {
      description: "adds",
      inputSchema: { type: "object", properties: { a: { type: "number" } } },
      readOnly: true,
      run: (input: Record<string, unknown>) => ({ sum: Number(input.a) + 1 }),
    };
    await createPromptExecutor({ runner, tools: { add: tool } }).start(promptOp(), {}).result;
    expect(calls[0]!.def.tools).toEqual([{ name: "add", description: "adds", inputSchema: tool.inputSchema }]);
    expect(calls[0]!.env.toolExecutors).toBeDefined();
    expect(await calls[0]!.env.toolExecutors!.add!({ a: 1 }, {} as never)).toEqual({ sum: 2 });
  });

  it("a generated FILE lands in a blob output slot — not in a parallel artifacts channel (§7.1)", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const { runner, calls } = fakeRunner([okOutcome({ parsed: undefined, files: [{ mediaType: "image/png", bytes }] })]);
    const op = promptOp({ output: { name: "output", kind: "blob", schema: { type: "string", contentMediaType: "image/png" } } });
    const out = await createPromptExecutor({ runner }).start(op, {}).result;
    expect(out.value).toBe(bytes);
    // …and the call that produced them was NOT a structured-JSON call: forwarding the blob parameter's
    // `{type:"string",contentMediaType:…}` as the output contract made the provider's normal finish read
    // as "structured output was empty/absent" even though the bytes arrived and projected correctly.
    expect(calls[0]!.def.schema).toBeUndefined();
  });

  it("a TEXT-kind output goes through the TEXT path and projects a string (§5.2)", async () => {
    const { runner, calls } = fakeRunner([okOutcome({ parsed: "four", rawText: "four" })]);
    const op = promptOp({ output: { name: "output", kind: "text", schema: { type: "string" } } });
    const out = await createPromptExecutor({ runner }).start(op, {}).result;
    expect(calls[0]!.def.schema).toBeUndefined();
    expect(errorOf(out)).toBeUndefined();
    expect(out.value).toBe("four");
  });
});
