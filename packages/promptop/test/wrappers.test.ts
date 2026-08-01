import { describe, expect, it } from "vitest";
import type { BudgetMeter, BudgetMetrics, BudgetReservation, CallEstimate, Capabilities, ExecMetrics, ExecServices, Executor, MapSessionStore, MetricsAlgebra, Operation, InlineFamily, RateLimiter, SessionStore } from "@declarative-ai/exec";
import { EXEC_METRICS_ALGEBRA, MapMemoCache, RUNTIME_CAPABILITIES, compose, withMemoize, wrapHandle } from "@declarative-ai/exec";
import type { ModelMessage } from "ai";
import { createPromptExecutor } from "../src/executor";
import { withBudget, withRateLimit, withSession } from "../src/wrappers";
import { fakeRunner, okOutcome, promptOp, sessionStack, transcripts, errorOf } from "./fakes";

/**
 * Put a stream in a known state and return the ref for its HEAD.
 *
 * Streams are append-only, so seeding one means appending to it — and the caller needs the resulting
 * position back, because continuing from a stale one is a fork, not a continuation.
 */
async function seed(store: SessionStore, id: string, ...messages: ModelMessage[]): Promise<string> {
  const rec = store as unknown as MapSessionStore<ModelMessage>;
  const at = await rec.resolve({ ref: `${id}@0` });
  rec.open({ id: `seed:${id}`, session: at.at });
  rec.close(`seed:${id}`, { result: { value: { messages } } });
  return `${at.at.id}@${at.at.seq + 1}`;
}

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
    // Pricing `op.system + op.user` made a 20k-char conversation declare ~7 input tokens, so the
    // limiter under-declared by three orders of magnitude on any multi-turn call. The transcript used
    // to be visible because `withSession` inlined it into the op's config; it no longer does, so this
    // wrapper reads the resolved session directly — otherwise the same bug comes straight back.
    const { seam } = transcripts();
    const head = await seed(seam, "chat-1", { role: "user", content: "x".repeat(8000) });
    const est: CallEstimate[] = [];
    const limiter: RateLimiter = {
      schedule: (e, run) => {
        est.push(e);
        return run();
      },
      reportOutcome: () => {},
    };
    const { runner, calls } = fakeRunner([okOutcome()]);
    const stack = sessionStack(seam, withRateLimit({ limiter }, createPromptExecutor({ runner, record: true })));
    await stack.start(promptOp({}, { sessionId: head }), {}).result;
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
    const { seam } = transcripts();
    const head = await seed(seam, "chat-1", { role: "user", content: "y".repeat(5000) });
    const { runner } = fakeRunner([okOutcome()]);
    const stack = sessionStack(seam, withBudget({ meter, pricing }, createPromptExecutor({ runner, record: true })));
    await stack.start(promptOp({}, { sessionId: head }), {}).result;
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

/**
 * `withSession` — the wrapper keeps the POLICY, the executor owns the MECHANISM (SESSIONS.md §6).
 *
 * What these pin, in the order the model depends on them: the reservation is taken before the call
 * and released whatever happens; the transcript is what the EXECUTOR reported rather than something
 * the wrapper synthesized; a failed call still folds; and the outcome carries the EFFECTIVE position,
 * which is the only channel a caller has for learning that its call was forked.
 */
describe("withSession — append-only conversation", () => {
  /** Everything a session stream ends up holding, flattened for assertion. */
  const contentsOf = (store: MapSessionStore<ModelMessage>, ref: string): ModelMessage[] => store.messages(`${ref}@99`);

  it("no session named anywhere → passthrough, store untouched", async () => {
    const { store, seam } = transcripts();
    const { runner, calls } = fakeRunner([okOutcome()]);
    await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp(), {}).result;
    expect(calls[0]!.def.prompt).toBe("What is 2+2?");
    expect(contentsOf(store, "s_chat-1")).toEqual([]);
  });

  it("folds the turn the call SENT plus what the provider said it appended", async () => {
    const { store, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    expect(contentsOf(store, "chat-1")).toEqual([
      { role: "user", content: "What is 2+2?" },
      { role: "assistant", content: '{"answer":"4"}' },
    ]);
  });

  it("mirrors the provider's log VERBATIM — tool calls, results and provider options survive", async () => {
    // The old wrapper synthesized one stringified assistant turn from the output value, which threw
    // away everything in between. A reasoning part's signature has to come back byte-identical.
    const { store, seam } = transcripts();
    const appended: ModelMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "think", providerOptions: { anthropic: { signature: "sig-abc" } } },
          { type: "tool-call", toolCallId: "t1", toolName: "lookup", input: { q: "2+2" } },
        ],
      },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "t1", toolName: "lookup", output: { type: "json", value: 4 } }] },
      { role: "assistant", content: [{ type: "text", text: '{"answer":"4"}' }] },
    ];
    const { runner } = fakeRunner([okOutcome({ messages: appended })]);
    await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    expect(contentsOf(store, "chat-1").slice(1)).toEqual(appended);
  });

  it("replays the stored stream, and STRIPS the session fields the core would refuse", async () => {
    const { store, seam } = transcripts();
    const { runner, calls } = fakeRunner([okOutcome(), okOutcome()]);
    const stack = sessionStack(seam, createPromptExecutor({ runner, record: true }));
    await stack.start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    await stack.start(promptOp({}, { sessionId: "chat-1@1" }), {}).result;
    // The second call carried the first exchange on the wire — replay, because the Messages API and
    // everything through the AI SDK are stateless.
    expect(calls[1]!.def.messages).toEqual([
      { role: "user", content: "What is 2+2?" },
      { role: "assistant", content: '{"answer":"4"}' },
      { role: "user", content: "What is 2+2?" },
    ]);
    expect(calls[1]!.def.sessionId).toBeUndefined();
    expect(contentsOf(store, "chat-1")).toHaveLength(4);
  });

  it("FOLDS ON FAILURE — turns the provider appended before it failed still exist remotely", async () => {
    // Not recording them means the next append-by-handle meets a head we do not mirror, which is
    // divergence on the very next call.
    const { store, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome({ error: { classification: "permanent", reason: "model exploded" } })]);
    const out = await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp({}, { sessionId: "chat-1@0" }), {})
      .result;
    expect(errorOf(out)?.reason).toMatch(/model exploded/);
    expect(contentsOf(store, "chat-1")).toHaveLength(2);
  });

  it("reports the EFFECTIVE position on the outcome, not the key it was handed", async () => {
    const { seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    const out = await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp({}, { sessionId: "chat-1@0" }), {})
      .result;
    // Two entries went in, so the call ENDED at position 2 — "append after me" needs that, and the
    // caller has no other way to learn it.
    expect(out.metrics.sessionRef).toBe("chat-1@1");
  });

  it("FORKS when the position is no longer the head, leaving the original untouched", async () => {
    const { store, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome(), okOutcome()]);
    const stack = sessionStack(seam, createPromptExecutor({ runner, record: true }));
    await stack.start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    // The same position again: it has since been appended to, so forking is the only answer that
    // means anything — and nothing had to ask for it.
    const forked = await stack.start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    expect(forked.metrics.sessionRef).not.toBe("chat-1@1");
    expect(contentsOf(store, "chat-1")).toHaveLength(2);
  });

  it("`fork: true` branches even when the position IS still the head", async () => {
    // Deliberate divergence — fan variants out of one point — cannot be inferred from stream state.
    const { store, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome(), okOutcome()]);
    const stack = sessionStack(seam, createPromptExecutor({ runner, record: true }));
    await stack.start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    const forked = await stack.start(promptOp({}, { sessionId: "chat-1@1", fork: true }), {}).result;
    expect(forked.metrics.sessionRef).not.toMatch(/^chat-1@/);
    expect(contentsOf(store, "chat-1")).toHaveLength(2);
  });

  it("hands the executor a session whose ONLY enumerable property is `id`", async () => {
    // Everything else is non-enumerable, so the journal and any serialized inputs/outputs see `{ id }`.
    const { seam } = transcripts();
    let seen: ExecServices["session"];
    const { runner } = fakeRunner([okOutcome()]);
    const spy: Executor<ExecServices, ExecMetrics> = {
      capabilities: { ...RUNTIME_CAPABILITIES },
      metrics: EXEC_METRICS_ALGEBRA,
      start: (op, ctx) => {
        seen = ctx.session;
        return createPromptExecutor({ runner }).start(op, ctx) as never;
      },
    };
    await sessionStack(seam, spy).start(promptOp({}, { sessionId: "chat-1@0" }), {}).result;
    expect(Object.keys(seen!)).toEqual(["id"]);
    expect(JSON.parse(JSON.stringify(seen))).toEqual({ id: "chat-1@0" });
    expect(seen!.mode).toBe("append");
  });

  it("ACCEPTS providerSessionId — a stateful adapter can be told to resume", async () => {
    // It used to be refused outright, because the wrapper rewrote the config with inline messages and
    // there was no way to say "resume this handle" instead of "here is the transcript again".
    const { seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    const out = await sessionStack(seam, createPromptExecutor({ runner, record: true })).start(promptOp({}, { providerSessionId: "p1" }), {})
      .result;
    expect(errorOf(out)).toBeUndefined();
  });

  it("REFUSES a session with no SessionStore available — the seam is REQUIRED at start when unconstructed", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    // Composing without a store makes `sessions` part of what `.start` demands; passing `undefined`
    // is the only way to reach the runtime refusal, which is itself the point of the typed requirement.
    const out = await withSession(createPromptExecutor({ runner })).start(promptOp({}, { sessionId: "chat-1@0" }), {
      sessions: undefined as never,
    }).result;
    expect(errorOf(out)?.reason).toMatch(/no SessionStore is available/);
  });

  it("falls back to the run-scoped ctx.sessions store when none was constructed", async () => {
    const { store, seam } = transcripts();
    const { runner } = fakeRunner([okOutcome()]);
    // Both seams from ctx rather than construction — the store AND the record sink, since recording
    // is what a session append IS.
    await sessionStack(seam, createPromptExecutor({ runner, record: true }) as never).start(promptOp({}, { sessionId: "chat-1@0" }), {
      sessions: seam,
      records: seam as never,
    }).result;
    expect(contentsOf(store, "chat-1")).toHaveLength(2);
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

describe("withBudget — post-charge mode (computeCost)", () => {
  type Reuse = { reuse?: { originalCostUsd: number; owners: string[] } };
  /** An inner executor resolving a fixed result — a memo layer serving a hit, say. */
  const stubInner = (metrics: WideMetrics & Reuse, value: unknown = "cached"): Executor<ExecServices, WideMetrics> => ({
    capabilities: { ...RUNTIME_CAPABILITIES },
    metrics: { merge: (_a, b) => b },
    start: () => wrapHandle(async () => ({ value: value as never, metrics })),
  });
  const chargeReuse = (me: string) => (_op: Operation<InlineFamily>, r: { metrics: WideMetrics & Reuse }): number => {
    const reuse = r.metrics.reuse;
    return reuse && !reuse.owners.includes(me) ? reuse.originalCostUsd : 0;
  };
  const debitMeter = (): { meter: BudgetMeter; debits: number[]; reserved: number[] } => {
    const debits: number[] = [];
    const reserved: number[] = [];
    return {
      meter: {
        reserve: async (est) => {
          reserved.push(est);
          return { settle: async () => undefined };
        },
        availableCostUsd: async () => 100,
        debit: async (c) => void debits.push(c),
      },
      debits,
      reserved,
    };
  };

  it("debits the computed charge and folds it into the reported costUsd — no pre-call reserve", async () => {
    const { meter, debits, reserved } = debitMeter();
    const hit: WideMetrics & Reuse = { durationMs: 0, costUsd: 0, costSource: "table", reuse: { originalCostUsd: 0.02, owners: ["alice"] } };
    const out = await withBudget({ meter, computeCost: chargeReuse("bob") }, stubInner(hit)).start(promptOp(), {}).result;
    expect(debits).toEqual([0.02]);
    expect(reserved).toEqual([]); // a computed charge prices the RESULT; there is nothing to reserve
    expect(out.metrics.costUsd).toBeCloseTo(0.02, 12);
  });

  it("a zero charge (an owner's own hit, or a real call billed by the inner instance) touches nothing", async () => {
    const { meter, debits, reserved } = debitMeter();
    const hit: WideMetrics & Reuse = { durationMs: 0, costUsd: 0, costSource: "table", reuse: { originalCostUsd: 0.02, owners: ["alice"] } };
    const out = await withBudget({ meter, computeCost: chargeReuse("alice") }, stubInner(hit)).start(promptOp(), {}).result;
    expect(debits).toEqual([]);
    expect(reserved).toEqual([]);
    expect(out.metrics.costUsd).toBe(0);
  });

  it("falls back to an immediate reserve→settle for a meter without a debit channel", async () => {
    const settled: number[] = [];
    const reserved: number[] = [];
    const meter: BudgetMeter = {
      reserve: async (est) => {
        reserved.push(est);
        return { settle: async (c) => void settled.push(c) };
      },
      availableCostUsd: async () => 100,
    };
    const hit: WideMetrics & Reuse = { durationMs: 0, costUsd: 0, costSource: "table", reuse: { originalCostUsd: 0.05, owners: [] } };
    const out = await withBudget({ meter, computeCost: chargeReuse("bob") }, stubInner(hit)).start(promptOp(), {}).result;
    expect(reserved).toEqual([0.05]);
    expect(settled).toEqual([0.05]);
    expect(out.metrics.costUsd).toBeCloseTo(0.05, 12);
  });

  it("charges FUNCTION ops too — reuse is not a prompt-only concern", async () => {
    const { meter, debits } = debitMeter();
    const fnOp: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "combine",
      input: {},
      output: { name: "out", kind: "json" },
    };
    const hit: WideMetrics & Reuse = { durationMs: 0, costUsd: 0, costSource: "table", reuse: { originalCostUsd: 0.01, owners: [] } };
    const out = await withBudget({ meter, computeCost: chargeReuse("bob") }, stubInner(hit)).start(fnOp, {}).result;
    expect(debits).toEqual([0.01]);
    expect(out.metrics.costUsd).toBeCloseTo(0.01, 12);
  });
});
