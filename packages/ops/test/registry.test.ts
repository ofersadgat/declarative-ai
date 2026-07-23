import { describe, expect, it } from "vitest";
import { isOk } from "@declarative-ai/json";
import {
  HOST_CAPABILITIES,
  RUNTIME_CAPABILITIES,
  failureOf,
  hostFunction,
  isStreaming,
  liftThrowing,
  pureFunction,
  runFunction,
  runtimeFunction,
  type FunctionInputs,
  type FunctionRegistry,
} from "../src/registry";

interface FakeCtx {
  label: string;
}

/** The registry IS a map — `new Map()`, not a bespoke container. */
type Registry = FunctionRegistry<FakeCtx, { durationMs: number; cost?: number }>;
const newRegistry = (): Registry => new Map();

describe("the function registry is a Map (API.md, \"The function registry\")", () => {
  it("holds pure entries whose impl resolves a value result", () => {
    const r = newRegistry();
    r.set("identity", pureFunction((inputs: FunctionInputs) => ({ value: inputs.value ?? null })));
    const entry = r.get("identity");
    expect(entry?.kind).toBe("pure");
    expect(entry!.kind === "pure" && entry!.impl({ value: 42 })).toEqual({ value: 42 });
    expect(r.has("identity")).toBe(true);
    expect(r.get("missing")).toBeUndefined();
    expect(r.has("missing")).toBe(false);
  });

  it("chains, because Map.set returns the map — the one thing the old class existed to provide", () => {
    const r = newRegistry()
      .set("a", pureFunction(() => ({ value: 1 })))
      .set("b", pureFunction(() => ({ value: 2 })));
    expect([...r.keys()].sort()).toEqual(["a", "b"]);
  });

  it("holds host and runtime entries in ONE map, discriminated by kind", async () => {
    const r = newRegistry();
    r.set("greet", hostFunction(async (inputs: FunctionInputs, ctx: FakeCtx) => ({ value: `${ctx.label}:${String(inputs.name)}` }), HOST_CAPABILITIES));
    r.set("agent", runtimeFunction(async () => ({ value: "done" }), RUNTIME_CAPABILITIES));
    expect(r.get("greet")?.kind).toBe("host");
    expect(r.get("agent")?.kind).toBe("runtime");
    expect([...r.keys()].sort()).toEqual(["agent", "greet"]);
    await expect(runFunction(r.get("greet")!, { name: "ada" }, { label: "hi" })).resolves.toEqual({ value: "hi:ada" });
  });

  it("carries REQUIRED, total capabilities per variant — no silent undefined gate", () => {
    const r = newRegistry();
    r.set(
      "agent",
      runtimeFunction(async () => ({ value: null }), { ...RUNTIME_CAPABILITIES, policyEnforcement: "callback", mutatesWorkspace: true }),
    );
    const entry = r.get("agent");
    expect(entry?.kind).toBe("runtime");
    // The whole point of the union: reading `policyEnforcement` yields a decision, never `undefined`.
    expect(entry!.kind === "runtime" && entry!.capabilities.policyEnforcement).toBe("callback");
    expect(entry!.kind === "runtime" && entry!.capabilities.memoizable).toBe(false);
  });

  it("tracks streaming registration per ref", () => {
    const r = newRegistry();
    r.set("plain", hostFunction(async () => ({ value: null }), HOST_CAPABILITIES));
    r.set("streamy", hostFunction(async () => ({ value: null }), HOST_CAPABILITIES, { stream: true }));
    expect(isStreaming(r.get("plain"))).toBe(false);
    expect(isStreaming(r.get("streamy"))).toBe(true);
    expect(isStreaming(r.get("missing"))).toBe(false);
  });

  it("later registration overrides (swappable entries)", () => {
    const r = newRegistry();
    r.set("f", pureFunction(() => ({ value: 1 })));
    r.set("f", pureFunction(() => ({ value: 2 })));
    const entry = r.get("f");
    expect(entry!.kind === "pure" && entry!.impl({})).toEqual({ value: 2 });
  });
});

describe("errors as data (DESIGN §3.3)", () => {
  it("classifies a thrown 429 as retriable instead of blanket-permanent", async () => {
    const impl = liftThrowing(async () => {
      throw Object.assign(new Error("rate limited"), { status: 429 });
    }, "function 'flaky'");
    const result = await impl({}, undefined);
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.classification).toBe("network-retriable");
    expect(!isOk(result) && result.error.rateLimited).toBe(true);
    expect(!isOk(result) && result.error.reason).toContain("function 'flaky'");
  });

  it("maps an abort to `canceled`, not a failure of the unit", () => {
    const failure = failureOf(Object.assign(new Error("stopped"), { name: "AbortError" }));
    expect(failure.classification).toBe("canceled");
  });

  it("keeps a deterministic error permanent", async () => {
    const impl = liftThrowing(() => {
      throw new Error("bad input");
    });
    const result = await impl({}, undefined);
    expect(!isOk(result) && result.error.classification).toBe("permanent");
    expect(!isOk(result) && result.error.reason).toBe("bad input");
  });

  it("lets a FAILED result still carry its partial value", () => {
    // The failure branch permits `value` — a truncated generation is diagnosable, not empty.
    const partial: { error: { classification: "permanent"; reason: string }; value?: string } = {
      error: { classification: "permanent", reason: "cut off" },
      value: "half an ans",
    };
    expect(isOk(partial)).toBe(false);
    expect(partial.value).toBe("half an ans");
  });
});

describe("runFunction never throws (DESIGN §3.3)", () => {
  // Nothing at registration forces an impl through `liftThrowing` — the entry constructors take a raw
  // impl — so the `catch` fallback has to live at the one place every dispatch path goes through.
  // Otherwise a throwing impl rejects out of the caller's error handling entirely.
  it("classifies a throwing async impl instead of rejecting", async () => {
    const r = newRegistry();
    r.set(
      "boom",
      hostFunction((async () => {
        throw Object.assign(new Error("upstream 429"), { statusCode: 429 });
      }) as never, HOST_CAPABILITIES),
    );
    const result = await runFunction(r.get("boom")!, {}, { label: "ctx" });
    expect(isOk(result)).toBe(false);
    // The classification survives, which is the whole point: a 429 inside an impl stays retriable.
    expect(!isOk(result) && result.error.classification).toBe("network-retriable");
  });

  it("classifies a SYNC throw from a pure impl", async () => {
    const r = newRegistry();
    r.set(
      "boom",
      pureFunction(() => {
        throw new Error("bad glue");
      }),
    );
    const result = await runFunction(r.get("boom")!, {}, { label: "ctx" });
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toBe("bad glue");
  });

  it("maps a thrown AbortError to `canceled`, not a unit failure", async () => {
    const r = newRegistry();
    r.set(
      "boom",
      runtimeFunction((async () => {
        throw Object.assign(new Error("stopped"), { name: "AbortError" });
      }) as never, RUNTIME_CAPABILITIES),
    );
    const result = await runFunction(r.get("boom")!, {}, { label: "ctx" });
    expect(!isOk(result) && result.error.classification).toBe("canceled");
  });

  it("passes an impl's reported metrics through untouched", async () => {
    const r = newRegistry();
    r.set("agent", hostFunction((async () => ({ value: "text", metrics: { durationMs: 5, cost: 0.02 } })) as never, HOST_CAPABILITIES));
    const result = await runFunction(r.get("agent")!, {}, { label: "ctx" });
    expect(result.metrics).toEqual({ durationMs: 5, cost: 0.02 });
  });
});
