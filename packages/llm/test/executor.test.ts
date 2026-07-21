import { describe, expect, it } from "vitest";
import type { CallEstimate, ExecEvent, ExecServices, Executor, Outcome, RateLimiter } from "@declarative-ai/core";
import { composeExecutors, MapSessionStore } from "@declarative-ai/core";
import { DEFAULT_FLOOR_MS, DEFAULT_SAFETY_MARGIN_MS } from "@declarative-ai/services";
import type { CallOutcome } from "../src/generate";
import type { CallRunner, LlmCallDefinition } from "../src/executor";
import { createLlmCallExecutor, LlmCallExecutor } from "../src/executor";
import { withDeadline, withMemoize, withRateLimit, withRepair, withRetry, withSession } from "../src/wrappers";
import { DEF, fakeRunner, memoCache, okOutcome, specOf, validationFailure } from "./fakes";

describe("LlmCallExecutor (core) — outcome mapping", () => {
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
    expect(outcome.metrics.durationMs).toBe(20);
    expect(typeof outcome.metrics.startMs).toBe("number");

    // The definition + spec.outputSchema reached the runner as StructuredCallParams.
    expect(calls).toHaveLength(1);
    expect(calls[0]!.params.model).toBe("anthropic/claude-haiku-4-5");
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
    const outcome = await createLlmCallExecutor({ runner: fakeRunner([failed]).runner }).start(specOf(), {}).outcome;
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

  it("hands ctx.validator to the runner deps", async () => {
    const validator = { validateValue: () => ({ ok: true }) };
    const { runner, calls } = fakeRunner([okOutcome()]);
    await createLlmCallExecutor({ runner }).start(specOf(), { validator }).outcome;
    expect(calls[0]!.deps.validator).toBe(validator);
  });

  it("returns an empty completed event stream (v1)", async () => {
    const handle = createLlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner }).start(specOf(), {});
    const events = [];
    for await (const e of handle.events) events.push(e);
    expect(events).toEqual([]);
    await handle.outcome;
  });
});

describe("core — per-call time budget", () => {
  it("uses spec.limits.timeoutMs as the budget when the definition names none", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const def: LlmCallDefinition = { ...DEF, timeoutMs: undefined };
    await createLlmCallExecutor({ runner }).start(specOf({ limits: { timeoutMs: 12_345 } }, def), {}).outcome;
    expect(calls[0]!.params.timeoutMs).toBe(12_345);
  });

  it("a spec.limits.timeoutMs ABOVE the 10-minute default extends the budget (not clamped)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const def: LlmCallDefinition = { ...DEF, timeoutMs: undefined };
    await createLlmCallExecutor({ runner }).start(specOf({ limits: { timeoutMs: 900_000 } }, def), {}).outcome;
    expect(calls[0]!.params.timeoutMs).toBe(900_000);
  });

  it("refuses (permanent) when the definition's timeoutMs exceeds spec.limits.timeoutMs — never silently clamps", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner }).start(specOf({ limits: { timeoutMs: 10_000 } }), {}).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/timeoutMs \(30000ms\) exceeds spec.limits.timeoutMs \(10000ms\)/);
  });
});

describe("core — loud failure on unconsumed wrapper fields", () => {
  it("refuses a ctx.deadline the bare core cannot honor (compose withDeadline)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const ctx: ExecServices = { deadline: { maxDurationMs: 60_000 }, stepStartMs: 0 };
    const outcome = await createLlmCallExecutor({ runner }).start(specOf(), ctx).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/compose withDeadline/);
  });

  it("refuses a declaration sessionId no session layer consumed (compose withSession)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const outcome = await createLlmCallExecutor({ runner })
      .start(specOf({}, { ...DEF, sessionId: "s1" }), {})
      .outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/compose withSession/);
  });
});

describe("core — cancel", () => {
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

describe("withRepair", () => {
  const repairing = (runner: CallRunner, turns: number) => composeExecutors(new LlmCallExecutor({ runner }), withRepair({ turns }));

  it("repairs a validation failure up to the constructed turns, augmenting the prompt with the errors", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), validationFailure(), okOutcome()]);
    const outcome = await repairing(runner, 2).start(specOf(), {}).outcome;

    expect(calls).toHaveLength(3);
    expect(calls[0]!.params.prompt).toBe("What is 2+2?");
    expect(calls[1]!.params.prompt).toContain("Your previous output failed schema validation:");
    expect(calls[1]!.params.prompt).toContain("data/answer must be string");
    expect(calls[2]!.params.prompt).toContain("Your previous output failed schema validation:");

    // Final outcome is the success, with metrics ACCUMULATED across all three attempts.
    expect(outcome.error).toBeUndefined();
    expect(outcome.value).toEqual({ answer: "4" });
    expect(outcome.metrics.inputTokens).toBe(30);
    expect(outcome.metrics.outputTokens).toBe(15);
    expect(outcome.metrics.cost).toBeCloseTo(0.003);
    expect(outcome.metrics.durationMs).toBe(60);
  });

  it("stops after the constructed turns and returns the last validation failure (value preserved)", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), validationFailure(), validationFailure()]);
    const outcome = await repairing(runner, 1).start(specOf(), {}).outcome;
    expect(calls).toHaveLength(2);
    expect(outcome.error?.classification).toBe("api-retriable");
    expect(outcome.value).toEqual({ answer: 7 });
  });

  it("withRepair(0) — a validation failure is terminal", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), okOutcome()]);
    const outcome = await repairing(runner, 0).start(specOf(), {}).outcome;
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
    const outcome = await repairing(runner, 3).start(specOf(), {}).outcome;
    expect(calls).toHaveLength(1);
    expect(outcome.error?.reason).toBe("529 overloaded");
  });

  it("sends an augmented definition on the repair turn (identity reflects what is sent)", async () => {
    const seen: LlmCallDefinition[] = [];
    const inner = new LlmCallExecutor({ runner: fakeRunner([validationFailure(), okOutcome()]).runner });
    const spy: Executor = {
      kind: inner.kind,
      capabilities: inner.capabilities,
      start: (spec, ctx) => {
        seen.push(spec.definition as LlmCallDefinition);
        return inner.start(spec, ctx);
      },
    };
    await composeExecutors(spy, withRepair({ turns: 1 })).start(specOf(), {}).outcome;
    expect(seen).toHaveLength(2);
    // The repair turn carries the concrete validation errors appended to the prompt, so the sent
    // definition differs from the original — what an inner memoize would (correctly) re-key on.
    expect(seen[0]!.prompt).not.toContain("failed schema validation");
    expect(seen[1]!.prompt).toContain("failed schema validation");
  });
});

describe("withRetry — unified re-attempt policy", () => {
  const NO_WAIT = (_ms: number) => Promise.resolve(); // inject a no-op sleep so backoff never really waits
  const transientFailure = (): CallOutcome => ({
    rawText: "",
    finishReason: "error",
    metrics: { durationMs: 3, cost: 0.001 },
    error: { classification: "network-retriable", reason: "429 rate limited" },
  });

  it("retries a transient (network-retriable) failure up to the cap, then succeeds (metrics accumulate)", async () => {
    const { runner, calls } = fakeRunner([transientFailure(), transientFailure(), okOutcome()]);
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRetry({ transient: { cap: 3, waitMs: NO_WAIT } }));
    const outcome = await exec.start(specOf(), {}).outcome;
    expect(calls).toHaveLength(3);
    expect(calls.every((c) => c.params.prompt === "What is 2+2?")).toBe(true); // same request re-sent
    expect(outcome.error).toBeUndefined();
    expect(outcome.metrics.durationMs).toBe(3 + 3 + 20); // folded across all attempts
  });

  it("stops at the transient cap and returns the last failure", async () => {
    const { runner, calls } = fakeRunner([transientFailure(), transientFailure(), transientFailure()]);
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRetry({ transient: { cap: 1, waitMs: NO_WAIT } }));
    const outcome = await exec.start(specOf(), {}).outcome;
    expect(calls).toHaveLength(2); // initial + 1 retry
    expect(outcome.error?.classification).toBe("network-retriable");
  });

  it("validation.feedback:true reshapes the former withRepair (errors appended to the prompt)", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), okOutcome()]);
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRetry({ validation: { turns: 1, feedback: true } }));
    const outcome = await exec.start(specOf(), {}).outcome;
    expect(calls).toHaveLength(2);
    expect(calls[1]!.params.prompt).toContain("failed schema validation");
    expect(outcome.error).toBeUndefined();
  });

  it("validation.feedback:false is a blind re-roll (same prompt, no error hint)", async () => {
    const { runner, calls } = fakeRunner([validationFailure(), okOutcome()]);
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRetry({ validation: { turns: 1, feedback: false } }));
    const outcome = await exec.start(specOf(), {}).outcome;
    expect(calls).toHaveLength(2);
    expect(calls[1]!.params.prompt).toBe("What is 2+2?"); // unchanged — no feedback appended
    expect(outcome.error).toBeUndefined();
  });

  it("does not retry a permanent failure on either axis", async () => {
    const permanent: CallOutcome = { rawText: "", finishReason: "error", metrics: { durationMs: 1 }, error: { classification: "permanent", reason: "bad request" } };
    const { runner, calls } = fakeRunner([permanent, okOutcome()]);
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRetry({ transient: 3, validation: { turns: 3, feedback: true } }));
    const outcome = await exec.start(specOf(), {}).outcome;
    expect(calls).toHaveLength(1);
    expect(outcome.error?.reason).toBe("bad request");
  });
});

describe("withDeadline", () => {
  const withDl = (runner: CallRunner) => composeExecutors(new LlmCallExecutor({ runner }), withDeadline());

  it("fails fast with a 'deadline-floor' reason when the window is below the floor", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const ctx: ExecServices = {
      clock: { now: () => now },
      deadline: { maxDurationMs: DEFAULT_SAFETY_MARGIN_MS + DEFAULT_FLOOR_MS - 1 }, // remaining = floor - 1
      stepStartMs: now,
    };
    const outcome = await withDl(runner).start(specOf(), ctx).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("deadline");
    expect(outcome.error?.reason.startsWith("deadline-floor")).toBe(true);
    expect(outcome.metrics.durationMs).toBe(0);
  });

  it("clamps the per-call timeout (and the definition's own budget) to the remaining window", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const ctx: ExecServices = {
      clock: { now: () => now },
      deadline: { maxDurationMs: 20_000, safetyMarginMs: 4_000 }, // remaining = 16s < def's 30s
      stepStartMs: now,
    };
    const outcome = await withDl(runner).start(specOf(), ctx).outcome;
    expect(outcome.error).toBeUndefined();
    expect(calls[0]!.params.timeoutMs).toBe(16_000);
  });

  it("consumes ctx.deadline — the bare core inside never sees it (no loud-failure refusal)", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    const ctx: ExecServices = { clock: { now: () => 0 }, deadline: { maxDurationMs: 600_000 }, stepStartMs: 0 };
    const outcome = await withDl(runner).start(specOf(), ctx).outcome;
    expect(outcome.error).toBeUndefined();
  });

  it("takes the DeadlineConfig at CONSTRUCTION — then start needs only stepStartMs (no ctx.deadline)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const cfg = { maxDurationMs: DEFAULT_SAFETY_MARGIN_MS + DEFAULT_FLOOR_MS - 1 }; // remaining = floor - 1
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withDeadline({ deadline: cfg }));
    const outcome = await exec.start(specOf(), { clock: { now: () => now }, stepStartMs: now }).outcome;
    expect(calls).toHaveLength(0); // the construction cfg drove the fail-fast — no ctx.deadline supplied
    expect(outcome.error?.classification).toBe("deadline");
  });

  it("applies the inner executor directly when passed (form-1 direct nesting)", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const now = 1_000_000;
    const cfg = { maxDurationMs: DEFAULT_SAFETY_MARGIN_MS + DEFAULT_FLOOR_MS - 1 };
    const exec = withDeadline({ deadline: cfg }, new LlmCallExecutor({ runner })); // direct: returns an Executor
    const outcome = await exec.start(specOf(), { clock: { now: () => now }, stepStartMs: now }).outcome;
    expect(calls).toHaveLength(0);
    expect(outcome.error?.classification).toBe("deadline");
  });
});

describe("withRateLimit", () => {
  const limited = (runner: CallRunner, limiter: RateLimiter) => composeExecutors(new LlmCallExecutor({ runner }), withRateLimit({ limiter }));

  it("schedules the call through the limiter with the token estimate and reports the outcome", async () => {
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
    const outcome = await limited(runner, rateLimiter).start(specOf(), {}).outcome;

    expect(outcome.error).toBeUndefined();
    expect(calls).toHaveLength(1);
    // chars/4 over the extracted prompt text (system + prompt, joined); output = the configured cap.
    const inputChars = `${DEF.system}\n${DEF.prompt}`.length;
    expect(scheduled[0]).toEqual({
      inputTokens: Math.ceil(inputChars / 4),
      outputTokens: 100,
      modelId: "anthropic/claude-haiku-4-5",
    });
    expect(reported).toEqual([{ rateLimited: undefined, modelId: "anthropic/claude-haiku-4-5" }]);
  });

  it("reports rateLimited: true on a 429 outcome", async () => {
    const limitedOutcome: CallOutcome = {
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
    await limited(fakeRunner([limitedOutcome]).runner, rateLimiter).start(specOf(), {}).outcome;
    expect(reported).toEqual([{ rateLimited: true, modelId: "anthropic/claude-haiku-4-5" }]);
  });

  it("a limiter fault is normalized into a permanent-failure outcome — handle.outcome never rejects", async () => {
    const rateLimiter: RateLimiter = {
      async schedule() {
        throw new Error("bucket exploded");
      },
      reportOutcome() {},
    };
    const outcome = await limited(fakeRunner([okOutcome()]).runner, rateLimiter).start(specOf(), {}).outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toContain("bucket exploded");
  });

  it("cancel while QUEUED prevents the call from ever starting (canceled outcome, nothing reported)", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const reported: unknown[] = [];
    const rateLimiter: RateLimiter = {
      async schedule(_est, run) {
        await gate; // the call sits queued until the test releases the slot
        return run();
      },
      reportOutcome(o) {
        reported.push(o);
      },
    };
    const { runner, calls } = fakeRunner([okOutcome()]);
    const handle = limited(runner, rateLimiter).start(specOf(), {});
    const canceling = handle.cancel(); // lands while queued — before any inner call exists
    release();
    await canceling;
    const outcome = await handle.outcome;
    expect(calls).toHaveLength(0); // the provider call never started
    expect(outcome.error?.classification).toBe("canceled");
    expect(reported).toEqual([]); // a call that never ran feeds no AIMD signal
  });
});

describe("wrappers — event forwarding", () => {
  it("forwards the inner handle's event stream through a wrapper (not swallowed)", async () => {
    const inner: Executor = {
      kind: "llm-call",
      capabilities: new LlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner }).capabilities,
      start: () => ({
        events: (async function* () {
          yield { type: "progress", message: "step-1" } as ExecEvent;
        })(),
        outcome: Promise.resolve(okOutcome() as Outcome),
        cancel: async () => {},
      }),
    };
    const handle = composeExecutors(inner, withMemoize({ cache: memoCache() })).start(specOf(), {});
    const events: ExecEvent[] = [];
    for await (const e of handle.events) events.push(e);
    expect(events).toEqual([{ type: "progress", message: "step-1" }]);
    await handle.outcome;
  });
});

describe("withMemoize + composition", () => {
  it("memoize (outermost) caches the final result; a second identical call skips the inner", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const cache = memoCache();
    // memoize OUTERMOST of repair: caches the final (post-repair) result under the declaration's key.
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withRepair({ turns: 0 }), withMemoize({ cache }));
    const o1 = await exec.start(specOf(), {}).outcome;
    const o2 = await exec.start(specOf(), {}).outcome;
    expect(o1.value).toEqual({ answer: "4" });
    expect(o2.value).toEqual({ answer: "4" });
    expect(calls).toHaveLength(1); // second served from cache
  });

  it("does not cache a failure", async () => {
    const failed: CallOutcome = {
      rawText: "",
      finishReason: "error",
      metrics: { durationMs: 1 },
      error: { classification: "permanent", reason: "boom" },
    };
    const { runner, calls } = fakeRunner([failed, okOutcome()]);
    const cache = memoCache();
    const exec = composeExecutors(new LlmCallExecutor({ runner }), withMemoize({ cache }));
    const o1 = await exec.start(specOf(), {}).outcome;
    const o2 = await exec.start(specOf(), {}).outcome;
    expect(o1.error).toBeDefined();
    expect(o2.value).toEqual({ answer: "4" });
    expect(calls).toHaveLength(2); // failure not cached → re-run
  });

  it("REFUSES (throws at composition time) to wrap a session layer", () => {
    const core = new LlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner });
    expect(() => composeExecutors(core, withSession({ sessions: new MapSessionStore() }), withMemoize({ cache: memoCache() }))).toThrow(
      /must not wrap a session layer/,
    );
  });
});
