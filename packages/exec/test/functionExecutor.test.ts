/**
 * `FunctionExecutor` — the function half, now that it is a first-class executor rather than a private
 * method on the dispatcher.
 *
 * Most of its behaviour is already covered through `OperationExecutor` in `wrappers.test.ts`, which is
 * the point: the split is behaviour-preserving. What is asserted here is what only became true once it
 * stood alone — that it can be constructed, wrapped and reached on its own, and that it says so
 * plainly when it is wired somewhere the dispatcher belonged.
 */
import { describe, expect, it } from "vitest";
import { FunctionExecutor, createFunctionExecutor, FUNCTION_CAPABILITIES } from "../src/functionExecutor.js";
import { OperationExecutor } from "../src/operationExecutor.js";
import { isOk, promptOp, pureFunction, runtimeFunction, type FunctionInputs, type FunctionRegistry } from "@declarative-ai/ops";
import type { ExecMetrics, ExecServices } from "../src/contract.js";
import { finishedHandle } from "../src/handles.js";

const registry = (): FunctionRegistry<ExecServices, ExecMetrics> => new Map();

const callOp = (functionRef: string, text = "in") => ({
  kind: "function" as const,
  functionRef,
  input: { value: { kind: "text" as const, binding: { text } } },
  output: { name: "out", kind: "text" as const },
});

describe("FunctionExecutor", () => {
  it("runs a registered function on its own — no dispatcher required", async () => {
    const functions = registry();
    functions.set("shout", pureFunction((i: FunctionInputs) => ({ value: String(i.value).toUpperCase() })) as never);
    const result = await new FunctionExecutor({ functions }).start(callOp("shout", "hi"), {}).result;
    expect(isOk(result) && result.value).toBe("HI");
  });

  it("names the missing function rather than failing anonymously", async () => {
    const result = await new FunctionExecutor({ functions: registry() }).start(callOp("nope"), {}).result;
    expect(!isOk(result) && result.error.reason).toBe("no function 'nope' is registered");
    expect(!isOk(result) && result.error.classification).toBe("permanent");
  });

  it("reports the ENTRY's capabilities per op, not one static record for the whole registry", async () => {
    // Which is what a wrapper gating on capabilities reads; a registry-wide record would make every
    // entry look like the most permissive one in it.
    const functions = registry();
    functions.set("agent", runtimeFunction(async () => ({ value: "x", metrics: { durationMs: 0 } }), { ...FUNCTION_CAPABILITIES, memoizable: false }) as never);
    const exec = new FunctionExecutor({ functions });
    expect(exec.capabilitiesFor(callOp("agent")).memoizable).toBe(false);
    expect(exec.capabilitiesFor(callOp("unknown")).memoizable).toBe(true); // the fallback record
  });

  it("refuses a PROMPT op as the wiring mistake it is", async () => {
    // Reachable only by installing this where the dispatcher belonged. Saying "no function registered"
    // would send the reader looking for a registry entry that was never supposed to exist.
    const op = promptOp({ user: "hi", config: { model: "anthropic/x" }, output: { name: "a", schema: { type: "string" } } });
    const result = await new FunctionExecutor({ functions: registry() }).start(op, {}).result;
    expect(!isOk(result) && result.error.reason).toContain("dispatch by kind is OperationExecutor's job");
  });

  it("checks cancellation BEFORE running anything", async () => {
    const functions = registry();
    let ran = false;
    functions.set("work", pureFunction(() => ((ran = true), { value: "done" })) as never);
    const result = await new FunctionExecutor({ functions }).start(callOp("work"), { abortSignal: AbortSignal.abort() }).result;
    expect(ran).toBe(false);
    expect(!isOk(result) && result.error.classification).toBe("canceled");
  });

  it("createFunctionExecutor mirrors the constructor", () => {
    expect(createFunctionExecutor({ functions: registry() })).toBeInstanceOf(FunctionExecutor);
  });
});

describe("OperationExecutor takes either a registry or a built function executor", () => {
  it("accepts a bare registry, so every existing call site keeps working", async () => {
    const functions = registry();
    functions.set("shout", pureFunction((i: FunctionInputs) => ({ value: String(i.value).toUpperCase() })) as never);
    const result = await new OperationExecutor({ functions }).start(callOp("shout", "hi"), {}).result;
    expect(isOk(result) && result.value).toBe("HI");
  });

  it("accepts a function executor, so that half can be wrapped or subclassed like the prompt half", async () => {
    // The asymmetry this removes: the prompt slot was always an injected object, and the function
    // slot was a private method nobody could reach.
    const functions = registry();
    functions.set("shout", pureFunction(() => ({ value: "inner" })) as never);
    class Loud extends FunctionExecutor {
      override start() {
        return finishedHandle({ value: "SUBCLASSED", metrics: { durationMs: 0 } }) as never;
      }
    }
    const result = await new OperationExecutor({ functions: new Loud({ functions }) }).start(callOp("shout"), {}).result;
    expect(isOk(result) && result.value).toBe("SUBCLASSED");
  });

  it("still refuses a prompt op with no prompt executor wired in", async () => {
    const op = promptOp({ user: "hi", config: { model: "anthropic/x" }, output: { name: "a", schema: { type: "string" } } });
    const result = await new OperationExecutor({ functions: registry() }).start(op, {}).result;
    expect(!isOk(result) && result.error.reason).toContain("no prompt executor is wired in");
  });
});
