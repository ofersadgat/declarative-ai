import { describe, expect, it } from "vitest";
import type { BudgetMeter, BudgetMetrics, BudgetReservation, CallEstimate, Capabilities, ExecMetrics, ExecServices, Executor, MetricsAlgebra, Operation, InlineFamily, RateLimiter, SessionState } from "@declarative-ai/exec";
import { EXEC_METRICS_ALGEBRA, MapMemoCache, RUNTIME_CAPABILITIES, compose, withMemoize, wrapHandle } from "@declarative-ai/exec";
import type { ModelMessage } from "ai";
import { createPromptExecutor } from "../src/executor";
import { withBudget, withRateLimit, withSession } from "../src/wrappers";
import { fakeRunner, okOutcome, promptOp, transcripts, errorOf } from "./fakes";

/** An inner executor whose STATIC record says `memoizable`, but whose PER-OP record says the opposite —
 *  exactly the shape `OperationExecutor` has (static `FUNCTION_CAPABILITIES`, per-op the registry entry).
 *  It counts `start` calls so a memoize above the wrapper can be caught caching when it must not. */
type WideMetrics = ExecMetrics & BudgetMetrics;
function countingInner(perOp: Partial<Capabilities>): { inner: Executor<ExecServices, WideMetrics>; starts: () => number } {
  let n = 0;
  const algebra: MetricsAlgebra<WideMetrics> = { merge: (_a, b) => b };
  const inner: Executor<ExecServices, WideMetrics> = {
    capabilities: { ...RUNTIME_CAPABILITIES, memoizable: true },
    capabilitiesFor: () => ({ ...RUNTIME_CAPABILITIES, memoizable: true, ...perOp }),
    metrics: algebra,
    start: () => wrapHandle(async () => ({ value: n++ as unknown as never, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" as const } })),
  };
  return { inner, starts: () => n };
}


describe("withRateLimit", () => {
  it("schedules the call through the limiter with the token estimate and reports the outcome", async () => {
    const seen: { est: CallEstimate[]; reported: unknown[] } = { est: [], reported: [] };
    const limiter: RateLimiter = {
      schedule: async (est, run) => {
        seen.est.push(est);
        return run();
      },
      reportOutcome: (o) => void seen.reported.push(o),
    };
    const { runner } = fakeRunner([okOutcome()]);
    const stack = withRateLimit({ limiter }, createPromptExecutor({ runner }));
    const out = await stack.start(promptOp(), {}).result;
    expect(errorOf(out)).toBeUndefined();
    expect(seen.est[0]!.modelId).toBe("anthropic/claude-haiku-4-5");
    expect(seen.est[0]!.outputTokens).toBe(100); // the op's declared ceiling
    expect(seen.est[0]!.inputTokens).toBeGreaterThan(0);
    expect(seen.reported).toEqual([{ rateLimited: undefined, modelId: "anthropic/claude-haiku-4-5" }]);
  });

  it("reports rateLimited: true on a 429 outcome (the AIMD signal)", async () => {
    const reported: unknown[] = [];
    const limiter: RateLimiter = { schedule: (_e, run) => run(), reportOutcome: (o) => void reported.push(o) };
    const { runner } = fakeRunner([okOutcome({ error: { classification: "network-retriable", reason: "429", rateLimited: true } })]);
    await withRateLimit({ limiter }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(reported).toEqual([{ rateLimited: true, modelId: "anthropic/claude-haiku-4-5" }]);
  });

  it("a limiter fault is normalized into a permanent-failure outcome — the handle never rejects", async () => {
    const limiter: RateLimiter = {
      schedule: () => {
        throw new Error("limiter exploded");
      },
      reportOutcome: () => {},
    };
    const { runner } = fakeRunner([okOutcome()]);
    const out = await withRateLimit({ limiter }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(errorOf(out)).toMatchObject({ classification: "permanent", reason: "limiter exploded" });
  });

  it("cancel while QUEUED prevents the call from ever starting", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const limiter: RateLimiter = {
      schedule: async (_e, run) => {
        await gate;
        return run();
      },
      reportOutcome: () => {},
    };
    const { runner, calls } = fakeRunner([okOutcome()]);
    const handle = withRateLimit({ limiter }, createPromptExecutor({ runner })).start(promptOp(), {});
    const canceling = handle.cancel();
    release();
    const out = await handle.result;
    await canceling;
    expect(calls).toHaveLength(0);
    expect(errorOf(out)?.classification).toBe("canceled");
  });

  it("estimates against the RESOLVED config — a model and ceiling supplied via `defaults`", async () => {
    // The estimate used to read `op.config` directly, BEFORE `resolveConfig` merges defaults ← preset ←
    // inline. With the model coming from `defaults` the limiter saw `modelId: undefined` (so per-model
    // AIMD silently degraded to a single global signal) and a `defaults`-supplied `maxOutputTokens` was
    // replaced by the estimator's 512 fallback.
    const seen: { est: CallEstimate[]; reported: unknown[] } = { est: [], reported: [] };
    const limiter: RateLimiter = {
      schedule: (est, run) => {
        seen.est.push(est);
        return run();
      },
      reportOutcome: (o) => void seen.reported.push(o),
    };
    const defaults = { model: "anthropic/claude-haiku-4-5", maxOutputTokens: 4000 };
    const { runner } = fakeRunner([okOutcome()]);
    const stack = withRateLimit({ limiter, defaults }, createPromptExecutor({ runner, defaults }));
    await stack.start(promptOp({ config: {} }), {}).result;
    expect(seen.est[0]!.modelId).toBe("anthropic/claude-haiku-4-5");
    expect(seen.est[0]!.outputTokens).toBe(4000);
    expect(seen.reported).toEqual([{ rateLimited: undefined, modelId: "anthropic/claude-haiku-4-5" }]);
  });

  it("counts the FULL message set that will be sent, not just system + user", async () => {
    // `withSession` threads the transcript in as config-layer `messages`; pricing `op.system + op.user`
    // made a 20k-char conversation declare ~7 input tokens, so the limiter under-declared by three
    // orders of magnitude on any multi-turn call.
    const { store: sessions, seam } = transcripts();
    await sessions.put("chat-1", { messages: [{ role: "user", content: "x".repeat(8000) }] });
    const est: CallEstimate[] = [];
    const limiter: RateLimiter = {
      schedule: (e, run) => {
        est.push(e);
        return run();
      },
      reportOutcome: () => {},
    };
    const { runner, calls } = fakeRunner([okOutcome()]);
    const stack = withSession({ sessions: seam }, withRateLimit({ limiter }, createPromptExecutor({ runner })));
    await stack.start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    // The transcript really is on the wire (chars/4 ⇒ ≳2000 tokens), and the estimate says so.
    expect(JSON.stringify(calls[0]!.def.messages).length).toBeGreaterThan(8000);
    expect(est[0]!.inputTokens).toBeGreaterThan(2000);
  });

  it("passes a FUNCTION op straight through — it has nothing to price", async () => {
    let scheduled = 0;
    const limiter: RateLimiter = {
      schedule: (_e, run) => {
        scheduled++;
        return run();
      },
      reportOutcome: () => {},
    };
    const inner: Executor = {
      capabilities: createPromptExecutor().capabilities,
      metrics: EXEC_METRICS_ALGEBRA,
      start: () => ({ events: (async function* () {})(), result: Promise.resolve({ value: null, metrics: { durationMs: 0 } }), cancel: async () => {} }),
    };
    await withRateLimit({ limiter }, inner).start(
      { kind: "function", functionRef: "f", input: {}, output: { name: "output", kind: "json" } },
      {},
    ).result;
    expect(scheduled).toBe(0);
  });
});

describe("withBudget — per-call reserve → settle", () => {
  const meterOf = (script: Array<number | null>, available = 0): { meter: BudgetMeter; settled: number[]; reserved: number[] } => {
    const settled: number[] = [];
    const reserved: number[] = [];
    let i = 0;
    const meter: BudgetMeter = {
      reserve: async (est) => {
        reserved.push(est);
        const next = script[Math.min(i++, script.length - 1)];
        if (next === null) return null;
        const res: BudgetReservation = { ledgerId: `L${next}`, settle: async (c) => void settled.push(c) };
        return res;
      },
      availableCostUsd: async () => available,
    };
    return { meter, settled, reserved };
  };
  const pricing = {
    estimateCostUsd: (_m: string, i: number, o: number) => (i + o) / 1000,
    affordableOutputTokens: () => 256,
  };

  it("reserves before the call and settles the ACTUAL cost after", async () => {
    const { meter, settled } = meterOf([1]);
    const { runner } = fakeRunner([okOutcome()]);
    const out = await withBudget({ meter, pricing }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(settled).toEqual([0.001]); // the call's real cost, not the estimate
    // The ledger row id is NOT stamped onto metrics: a metrics record reports what the work measured,
    // and the wrapper did not measure a ledger row. It stays on the BudgetReservation that owns it.
    expect("ledgerId" in out.metrics).toBe(false);
  });

  it("reads the meter from ctx.meter when none is given at construction", async () => {
    const { meter, settled } = meterOf([1]);
    const { runner } = fakeRunner([okOutcome()]);
    await withBudget({ pricing }, createPromptExecutor({ runner })).start(promptOp(), { meter }).result;
    expect(settled).toEqual([0.001]);
  });

  it("clamps maxOutputTokens to the affordable ceiling and retries the reserve once", async () => {
    const { meter, reserved } = meterOf([null, 2]);
    const { runner, calls } = fakeRunner([okOutcome()]);
    const op = promptOp({}, { maxOutputTokens: 4000 }); // more than the wallet affords
    const out = await withBudget({ meter, pricing }, createPromptExecutor({ runner })).start(op, {}).result;
    expect(errorOf(out)).toBeUndefined();
    expect(reserved).toHaveLength(2);
    // The clamp is a real edit to the OP that is sent, so an inner memoize keys on what was actually
    // run — the same reason `withDeadline` lowers a real field rather than passing a side-channel.
    expect(calls[0]!.def.maxOutputTokens).toBe(256);
  });

  it("refuses with an out-of-credits outcome (no call, no settle) when even the clamped reserve won't fit", async () => {
    const { meter, settled } = meterOf([null]);
    const { runner, calls } = fakeRunner([okOutcome()]);
    const out = await withBudget({ meter, pricing }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(errorOf(out)?.classification).toBe("out-of-credits");
    expect(calls).toHaveLength(0);
    expect(settled).toEqual([]);
  });

  it("settles a FAILED call at its REAL cost — a failed call still costs money", async () => {
    const { meter, settled } = meterOf([1]);
    // The case that matters: the provider generated and BILLED, then the output failed validation (or
    // was truncated, or 5xx'd mid-stream). That is spend. Settling it at $0 would silently forgive a
    // real charge and let the wallet drift — the reserve would be released without the money moving.
    const { runner } = fakeRunner([
      okOutcome({ error: { classification: "api-retriable", reason: "validation failed" }, metrics: { costUsd: 0.004 } }),
    ]);
    await withBudget({ meter, pricing }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(settled).toEqual([0.004]);
  });

  it("settles $0 only when nothing was sent — a pre-call refusal is genuinely free", async () => {
    const { meter, settled } = meterOf([1]);
    const { runner } = fakeRunner([okOutcome({ error: { classification: "permanent", reason: "x" }, metrics: { costUsd: 0, costSource: "table" } })]);
    await withBudget({ meter, pricing }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(settled).toEqual([0]);
  });

  it("meters a call whose model comes from `defaults` — it used to run entirely UNMETERED", async () => {
    // `configOf(op)` read the op's inline fragment, so a `defaults`-supplied model made `model ===
    // undefined` and the wrapper returned the inner handle straight out: no reserve, no settle, no
    // out-of-credits gate. A complete no-op wrapper, silently, for the whole run.
    const { meter, reserved, settled } = meterOf([1]);
    const { runner } = fakeRunner([okOutcome()]);
    const defaults = { model: "anthropic/claude-haiku-4-5", maxOutputTokens: 100 };
    const stack = withBudget({ meter, pricing, defaults }, createPromptExecutor({ runner, defaults }));
    const out = await stack.start(promptOp({ config: {} }), {}).result;
    expect(errorOf(out)).toBeUndefined();
    expect(reserved).toHaveLength(1);
    expect(settled).toEqual([0.001]);
  });

  it("reserves against the FULL transcript, not just system + user", async () => {
    const { meter, reserved } = meterOf([1]);
    const { store: sessions, seam } = transcripts();
    await sessions.put("chat-1", { messages: [{ role: "user", content: "y".repeat(5000) }] });
    const { runner } = fakeRunner([okOutcome()]);
    const stack = withSession({ sessions: seam }, withBudget({ meter, pricing }, createPromptExecutor({ runner })));
    await stack.start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    // `pricing` charges (input + output)/1000. Blind to the transcript the reserve was priced on ~107
    // tokens (~$0.107); the 5000 chars actually sent are ~1250 input tokens, so a correct reserve is >$1.
    expect(reserved[0]!).toBeGreaterThan(1);
  });

  it("is a pure passthrough when no meter is available", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const out = await withBudget({ pricing }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(errorOf(out)).toBeUndefined();
    expect(calls).toHaveLength(1);
  });
});

describe("withSession — client-managed conversation", () => {
  it("no session id → passthrough (op unchanged, store untouched)", async () => {
    const { store: sessions, seam } = transcripts();
    const { runner, calls } = fakeRunner([okOutcome()]);
    await withSession({ sessions: seam }, createPromptExecutor({ runner })).start(promptOp(), {}).result;
    expect(calls[0]!.def.prompt).toBe("What is 2+2?");
    expect(await sessions.get("chat-1")).toBeUndefined();
  });

  it("a fresh session seeds the transcript with the turn + the assistant reply", async () => {
    const { store: sessions, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    // No `session` field on the result: it used to echo back the caller's own logical key, which
    // nothing read. What the wrapper actually DOES is write the transcript, so that is what is asserted.
    await withSession({ sessions: seam }, createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    expect(await sessions.get("chat-1")).toEqual({
      messages: [
        { role: "user", content: "What is 2+2?" },
        { role: "assistant", content: '{"answer":"4"}' },
      ],
    });
  });

  it("resuming prepends the stored transcript, and STRIPS the session fields the core would refuse", async () => {
    const { store: sessions, seam } = transcripts();
    await sessions.put("chat-1", { messages: [{ role: "user", content: "earlier" }] });
    const { runner, calls } = fakeRunner([okOutcome()]);
    await withSession({ sessions: seam }, createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    expect(calls[0]!.def.messages).toEqual([
      { role: "user", content: "earlier" },
      { role: "user", content: "What is 2+2?" },
    ]);
    expect(calls[0]!.def.sessionId).toBeUndefined();
  });

  it("the fold PRESERVES other stored SessionState fields", async () => {
    const { store: sessions, seam } = transcripts();
    const prior: SessionState<ModelMessage> = { messages: [], providerSessionId: "keep-me" };
    await sessions.put("chat-1", prior);
    const { runner } = fakeRunner([okOutcome()]);
    await withSession({ sessions: seam }, createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1" }), {}).result;
    expect((await sessions.get("chat-1"))!.providerSessionId).toBe("keep-me");
  });

  it("REFUSES providerSessionId — provider-side resume is not supported yet", async () => {
    const { store: sessions, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    const out = await withSession({ sessions: seam }, createPromptExecutor({ runner })).start(promptOp({}, { providerSessionId: "p1" }), {}).result;
    expect(errorOf(out)?.reason).toMatch(/provider-side session resume is not supported/);
  });

  it("REFUSES a sessionId with no SessionStore available — the seam is REQUIRED at start when unconstructed", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    // Composing without a store makes `sessions` part of what `.start` demands; passing `undefined`
    // is the only way to reach the runtime refusal, which is itself the point of the typed requirement.
    const out = await withSession(createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1" }), { sessions: undefined as never })
      .result;
    expect(errorOf(out)?.reason).toMatch(/no SessionStore is available/);
  });

  it("falls back to the run-scoped ctx.sessions store when none was constructed", async () => {
    const { store: sessions, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    await withSession(createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1" }), { sessions: seam }).result;
    expect((await sessions.get("chat-1"))!.messages).toHaveLength(2);
  });

  it("declares sessionResume capability", () => {
    const { runner } = fakeRunner([okOutcome()]);
    expect(compose(createPromptExecutor({ runner })).with(withSession()).capabilities.sessionResume).toBe(true);
  });
});

describe("per-entry capabilities forwarding (so a withMemoize ABOVE these wrappers gates on the real entry)", () => {
  const limiter: RateLimiter = { schedule: (_e, run) => run(), reportOutcome: () => {} };
  const op = promptOp() as unknown as Operation<InlineFamily>;

  it("withRateLimit forwards the per-op capability record", () => {
    const { inner } = countingInner({ memoizable: false });
    const rl = withRateLimit({ limiter }, inner);
    expect(rl.capabilitiesFor).toBeDefined();
    expect(rl.capabilitiesFor!(op).memoizable).toBe(false);
  });

  it("withBudget forwards the per-op capability record", () => {
    const { inner } = countingInner({ memoizable: false });
    const b = withBudget({}, inner);
    expect(b.capabilitiesFor).toBeDefined();
    expect(b.capabilitiesFor!(op).memoizable).toBe(false);
  });

  it("withSession folds sessionResume into the per-op record while preserving inner fields", () => {
    const { inner } = countingInner({ memoizable: false });
    const s = withSession(inner);
    // sessionResume forced true (mirrors its static capabilities), the inner memoizable:false preserved.
    expect(s.capabilitiesFor!(op)).toMatchObject({ memoizable: false, sessionResume: true });
  });

  it("end to end: a memoize above withRateLimit does NOT cache a memoizable:false entry", async () => {
    const { inner, starts } = countingInner({ memoizable: false });
    const stack = withMemoize({ cache: new MapMemoCache() }, withRateLimit({ limiter }, inner));
    await stack.start(op, {}).result;
    await stack.start(op, {}).result;
    expect(starts()).toBe(2); // ran twice — not served from cache
  });

  it("end to end: a memoize above withRateLimit DOES cache a memoizable:true entry (control)", async () => {
    const { inner, starts } = countingInner({ memoizable: true });
    const stack = withMemoize({ cache: new MapMemoCache() }, withRateLimit({ limiter }, inner));
    await stack.start(op, {}).result;
    await stack.start(op, {}).result;
    expect(starts()).toBe(1); // second call served from cache
  });
});
