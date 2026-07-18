import { describe, expect, it } from "vitest";
import type { CallEstimate, ExecServices, ExecutionSpec, RateLimiter } from "@ai-exec/core";
import { DEFAULT_FLOOR_MS, DEFAULT_SAFETY_MARGIN_MS } from "@ai-exec/services";
import type { CallOutcome } from "../src/generate";
import type { CallRunner, CallRunnerDeps, LlmCallDefinition } from "../src/executor";
import { createLlmCallExecutor, LlmCallExecutor } from "../src/executor";
import type { StructuredCallParams } from "../src/llmStep";

const DEF: LlmCallDefinition = {
  modelId: "claude-haiku-4-5",
  prompt: "What is 2+2?",
  system: "Answer tersely.",
  maxOutputTokens: 100,
  timeoutMs: 30_000,
};

function specOf(overrides: Partial<ExecutionSpec> = {}, def: LlmCallDefinition = DEF): ExecutionSpec {
  return {
    kind: "llm-call",
    definition: def,
    definitionHash: "test-hash",
    inputs: {},
    outputSchema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
    ...overrides,
  };
}

function okOutcome(partial: Partial<CallOutcome> = {}): CallOutcome {
  return {
    value: { answer: "4" },
    rawText: '{"answer":"4"}',
    finishReason: "stop",
    metrics: { inputTokens: 10, outputTokens: 5, cost: 0.001, costSource: "table", durationMs: 20 },
    ...partial,
  };
}

function validationFailure(errors = "data/answer must be string"): CallOutcome {
  return {
    value: { answer: 7 },
    rawText: '{"answer":7}',
    finishReason: "stop",
    metrics: { inputTokens: 10, outputTokens: 5, cost: 0.001, costSource: "table", durationMs: 20 },
    error: {
      classification: "api-retriable",
      reason: `post-reconstruction validation failed: output failed schema validation: ${errors}`,
    },
  };
}

/** A fake runner that records every invocation and replays a scripted outcome sequence. */
function fakeRunner(script: CallOutcome[]): { runner: CallRunner; calls: { params: StructuredCallParams; deps: CallRunnerDeps }[] } {
  const calls: { params: StructuredCallParams; deps: CallRunnerDeps }[] = [];
  const runner: CallRunner = async (params, deps) => {
    calls.push({ params, deps });
    const next = script[Math.min(calls.length - 1, script.length - 1)];
    return next!;
  };
  return { runner, calls };
}

describe("LlmCallExecutor — outcome mapping", () => {
  it("declares the llm-call capabilities", () => {
    const ex = new LlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner });
    expect(ex.kind).toBe("llm-call");
    expect(ex.capabilities).toEqual({
      structuredOutput: true,
      sessionResume: false,
      streaming: true,
      interactive: false,
      mutatesWorkspace: false,
      policyEnforcement: "none",
      memoizable: true,
      runtime: "edge-safe",
    });
  });

  it("maps a successful CallOutcome onto the core Outcome (value/rawText/finishReason/metrics + startMs)", async () => {
    const { runner, calls } = fakeRunner([
      okOutcome({ thinking: [{ type: "reasoning", text: "let me think", textOffset: 0 }] }),
    ]);
    const handle = createLlmCallExecutor({ runner }).start(specOf(), {});
    const outcome = await handle.outcome;

    expect(outcome.error).toBeUndefined();
    expect(outcome.value).toEqual({ answer: "4" });
    expect(outcome.rawText).toBe('{"answer":"4"}');
    expect(outcome.finishReason).toBe("stop");
    expect(outcome.thinking).toEqual([{ type: "reasoning", text: "let me think", textOffset: 0 }]);
    expect(outcome.metrics.inputTokens).toBe(10);
    expect(outcome.metrics.outputTokens).toBe(5);
    expect(outcome.metrics.cost).toBeCloseTo(0.001);
    expect(outcome.metrics.durationMs).toBe(20);
    expect(typeof outcome.metrics.startMs).toBe("number");

    // The definition + spec.outputSchema reached the runner as StructuredCallParams.
    expect(calls).toHaveLength(1);
    expect(calls[0]!.params.modelId).toBe("claude-haiku-4-5");
    expect(calls[0]!.params.prompt).toBe("What is 2+2?");
    expect(calls[0]!.params.schema).toEqual(specOf().outputSchema);
    expect(calls[0]!.params.timeoutMs).toBe(30_000);
  });

  it("never rejects for a call failure — the error rides on the outcome", async () => {
    const failed: CallOutcome = {
      rawText: "partial",
      finishReason: "error",
      metrics: { durationMs: 5 },
      error: { classification: "network-retriable", reason: "socket hang up" },
    };
    const handle = createLlmCallExecutor({ runner: fakeRunner([failed]).runner }).start(specOf(), {});
    const outcome = await handle.outcome;
    expect(outcome.error).toEqual({ classification: "network-retriable", reason: "socket hang up" });
    expect(outcome.rawText).toBe("partial");
  });

  it("normalizes a thrown runner error into a permanent failure outcome", async () => {
    const runner: CallRunner = async () => {
      throw new Error("wiring exploded");
    };
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), {}).outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toContain("wiring exploded");
  });

  it("returns an empty completed event stream (v1)", async () => {
    const handle = createLlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner }).start(specOf(), {});
    const events = [];
    for await (const e of handle.events) events.push(e);
    expect(events).toEqual([]);
    await handle.outcome;
  });
});

describe("LlmCallExecutor — repair loop", () => {
  it("repairs a validation failure up to spec.repairTurns, augmenting the prompt with the errors", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), validationFailure(), okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf({ repairTurns: 2 }), {}).outcome;

    expect(calls).toHaveLength(3);
    expect(calls[0]!.params.prompt).toBe("What is 2+2?");
    expect(calls[1]!.params.prompt).toContain("Your previous output failed schema validation:");
    expect(calls[1]!.params.prompt).toContain("data/answer must be string");
    expect(calls[1]!.params.prompt).toContain("Return ONLY corrected JSON matching the schema.");
    expect(calls[2]!.params.prompt).toContain("Your previous output failed schema validation:");

    // Final outcome is the success, with metrics ACCUMULATED across all three attempts.
    expect(outcome.error).toBeUndefined();
    expect(outcome.value).toEqual({ answer: "4" });
    expect(outcome.metrics.inputTokens).toBe(30);
    expect(outcome.metrics.outputTokens).toBe(15);
    expect(outcome.metrics.cost).toBeCloseTo(0.003);
    expect(outcome.metrics.durationMs).toBe(60);
  });

  it("stops after repairTurns extra calls and returns the last validation failure", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), validationFailure(), validationFailure()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf({ repairTurns: 1 }), {}).outcome;
    expect(calls).toHaveLength(2);
    expect(outcome.error?.classification).toBe("api-retriable");
    expect(outcome.error?.reason).toContain("validation");
    // The failed parse's value is preserved (§4 preserve-on-error).
    expect(outcome.value).toEqual({ answer: 7 });
  });

  it("defaults repairTurns to 0 — a validation failure is terminal", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), {}).outcome;
    expect(calls).toHaveLength(1);
    expect(outcome.error?.reason).toContain("validation");
  });

  it("does NOT repair a non-validation api-retriable failure", async () => {
    const overloaded: CallOutcome = {
      rawText: "",
      finishReason: "error",
      metrics: { durationMs: 5 },
      error: { classification: "api-retriable", reason: "529 overloaded" },
    };
    const { runner, calls } = fakeRunner([overloaded, okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf({ repairTurns: 3 }), {}).outcome;
    expect(calls).toHaveLength(1);
    expect(outcome.error?.reason).toBe("529 overloaded");
  });
});

describe("LlmCallExecutor — cancel", () => {
  it("cancel() aborts the in-flight call and the outcome resolves with classification 'canceled'", async () => {
    let sawAbort = false;
    const runner: CallRunner = async (_params, deps) => {
      await new Promise<void>((resolve) => {
        if (deps.abortSignal?.aborted) return resolve();
        deps.abortSignal?.addEventListener("abort", () => resolve(), { once: true });
      });
      sawAbort = true;
      return {
        rawText: "",
        finishReason: "aborted",
        metrics: { durationMs: 3 },
        error: { classification: "network-retriable", reason: "deadline-abort: stream cut off before completion" },
      };
    };
    const handle = createLlmCallExecutor({ runner }).start(specOf(), {});
    await handle.cancel();
    const outcome = await handle.outcome;
    expect(sawAbort).toBe(true);
    expect(outcome.error?.classification).toBe("canceled");
  });

  it("a pre-aborted spec.abortSignal maps the failure to 'canceled'", async () => {
    const ac = new AbortController();
    ac.abort();
    const aborted: CallOutcome = {
      rawText: "",
      finishReason: "aborted",
      metrics: { durationMs: 1 },
      error: { classification: "network-retriable", reason: "deadline-abort: stream cut off before completion" },
    };
    const outcome = await createLlmCallExecutor({ runner: fakeRunner([aborted]).runner })
      .start(specOf({ abortSignal: ac.signal }), {})
      .outcome;
    expect(outcome.error?.classification).toBe("canceled");
  });
});

describe("LlmCallExecutor — deadline", () => {
  it("fails fast with a 'deadline-floor' reason when the window is below the floor", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const ctx: ExecServices = {
      clock: { now: () => now },
      deadline: { maxDurationMs: DEFAULT_SAFETY_MARGIN_MS + DEFAULT_FLOOR_MS - 1 }, // remaining = floor - 1
      stepStartMs: now,
    };
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), ctx).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("deadline");
    expect(outcome.error?.reason.startsWith("deadline-floor")).toBe(true);
    expect(outcome.metrics.durationMs).toBe(0);
  });

  it("clamps the per-call timeout to the remaining window when a deadline is set", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const ctx: ExecServices = {
      clock: { now: () => now },
      deadline: { maxDurationMs: 20_000, safetyMarginMs: 4_000 }, // remaining = 16s < def's 30s
      stepStartMs: now,
    };
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), ctx).outcome;
    expect(outcome.error).toBeUndefined();
    expect(calls[0]!.params.timeoutMs).toBe(16_000);
  });

  it("uses spec.limits.timeoutMs when the definition names none", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const def: LlmCallDefinition = { ...DEF, timeoutMs: undefined };
    const outcome = await createLlmCallExecutor({ runner })
      .start(specOf({ limits: { timeoutMs: 12_345 } }, def), {})
      .outcome;
    expect(outcome.error).toBeUndefined();
    expect(calls[0]!.params.timeoutMs).toBe(12_345);
  });
});

describe("LlmCallExecutor — rate limiter pass-through", () => {
  it("schedules the call through ctx.rateLimiter with the token estimate and reports the outcome", async () => {
    const scheduled: CallEstimate[] = [];
    const reported: { rateLimited?: boolean; modelId?: string }[] = [];
    const rateLimiter: RateLimiter = {
      async schedule(est, run) {
        scheduled.push(est);
        return run();
      },
      reportOutcome(o) {
        reported.push(o);
      },
    };
    const { runner, calls } = fakeRunner([okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), { rateLimiter }).outcome;

    expect(outcome.error).toBeUndefined();
    expect(calls).toHaveLength(1);
    expect(scheduled).toHaveLength(1);
    // chars/4 over prompt+system; output = the configured cap.
    const inputChars = DEF.prompt.length + DEF.system!.length;
    expect(scheduled[0]).toEqual({
      inputTokens: Math.ceil(inputChars / 4),
      outputTokens: 100,
      modelId: "claude-haiku-4-5",
    });
    expect(reported).toEqual([{ rateLimited: undefined, modelId: "claude-haiku-4-5" }]);
  });

  it("reports rateLimited: true on a 429 outcome", async () => {
    const limited: CallOutcome = {
      rawText: "",
      finishReason: "error",
      metrics: { durationMs: 2 },
      error: { classification: "network-retriable", reason: "429 rate limited", rateLimited: true },
    };
    const reported: { rateLimited?: boolean; modelId?: string }[] = [];
    const rateLimiter: RateLimiter = {
      async schedule(_est, run) {
        return run();
      },
      reportOutcome(o) {
        reported.push(o);
      },
    };
    await createLlmCallExecutor({ runner: fakeRunner([limited]).runner }).start(specOf(), { rateLimiter }).outcome;
    expect(reported).toEqual([{ rateLimited: true, modelId: "claude-haiku-4-5" }]);
  });
});

describe("LlmCallExecutor — validator pass-through", () => {
  it("hands ctx.validator to the runner deps", async () => {
    const validator = { validateValue: () => ({ ok: true }) };
    const { runner, calls } = fakeRunner([okOutcome()]);
    await createLlmCallExecutor({ runner }).start(specOf(), { validator }).outcome;
    expect(calls[0]!.deps.validator).toBe(validator);
  });
});
