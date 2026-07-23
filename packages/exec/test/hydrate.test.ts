import { describe, expect, it } from "vitest";
import type { ExecResult, ExecServices, Executor, Failure, InlineFamily, Operation, ResolvedValue } from "../src/index";
import { EXEC_METRICS_ALGEBRA, MapMemoCache, RUNTIME_CAPABILITIES, compose, isOk, withHydration, withMemoize } from "../src/index";

const errorOf = <O,>(r: ExecResult<O>): Failure | undefined => (isOk(r) ? undefined : r.error);

const CAPS = { ...RUNTIME_CAPABILITIES, memoizable: true, runtime: "edge-safe" as const };

/** The id family stand-in: an op that is a cheap handle — its identity is free, its content is not. */
interface IdOp {
  id: string;
  userTextId: string;
}

/** The "store": what hydration reads on a miss. */
const TEXTS: Record<string, string> = { t1: "what is 2+2?", t2: "what is 3+3?" };

function inlineOp(user: string): Operation<InlineFamily> {
  return { kind: "prompt", user, config: { model: "m" }, input: {}, output: { name: "output", kind: "json" } };
}

/** An inline leaf recording the RESOLVED ops it was handed. */
function leaf(): { core: Executor; seen: Operation<InlineFamily>[] } {
  const seen: Operation<InlineFamily>[] = [];
  const core: Executor = {
    capabilities: CAPS,
    metrics: EXEC_METRICS_ALGEBRA,
    start(op) {
      seen.push(op);
      return {
        events: (async function* () {})(),
        result: Promise.resolve({ value: `ran:${(op as { user?: string }).user}` as ResolvedValue, metrics: { durationMs: 1 } }),
        cancel: async () => {},
      };
    },
  };
  return { core, seen };
}

describe("withHydration — the family-transition wrapper", () => {
  const hydrate = (loads: string[]) => (op: IdOp): Operation<InlineFamily> => {
    loads.push(op.userTextId);
    const text = TEXTS[op.userTextId];
    if (text === undefined) throw new Error(`no such text artifact: ${op.userTextId}`);
    return inlineOp(text);
  };

  it("resolves the family op and dispatches the RESOLVED inline op to the stack below", async () => {
    const { core, seen } = leaf();
    const loads: string[] = [];
    const exec = compose(core).with(withHydration<IdOp>(hydrate(loads)));
    const out = await exec.start({ id: "op1", userTextId: "t1" }, {}).result;
    expect(errorOf(out)).toBeUndefined();
    expect(out.value).toBe("ran:what is 2+2?");
    expect(loads).toEqual(["t1"]);
    expect(seen[0]?.kind).toBe("prompt");
  });

  it("a memoize ABOVE the transition keys on the cheap id — a hit never hydrates", async () => {
    const { core } = leaf();
    const loads: string[] = [];
    const exec = compose(core)
      .with(withHydration<IdOp>(hydrate(loads)))
      .with(withMemoize<ExecServices, IdOp>({ cache: new MapMemoCache(), identify: (op) => op.id }));
    const first = await exec.start({ id: "op1", userTextId: "t1" }, {}).result;
    const second = await exec.start({ id: "op1", userTextId: "t1" }, {}).result;
    expect(first.value).toBe("ran:what is 2+2?");
    expect(second.value).toBe("ran:what is 2+2?");
    expect(loads).toEqual(["t1"]); // hydrated ONCE — the hit short-circuited every store read
    const third = await exec.start({ id: "op2", userTextId: "t2" }, {}).result;
    expect(third.value).toBe("ran:what is 3+3?");
    expect(loads).toEqual(["t1", "t2"]);
  });

  it("a hydration fault is a classified failure through the never-throws handle, not a rejection", async () => {
    const { core, seen } = leaf();
    const exec = compose(core).with(withHydration<IdOp>(hydrate([])));
    const out = await exec.start({ id: "opX", userTextId: "missing" }, {}).result;
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/hydration: no such text artifact/);
    expect(seen).toHaveLength(0); // nothing was dispatched
  });

  it("forwards the family's per-op capability record so gates above can read it without hydrating", async () => {
    const { core } = leaf();
    const exec = withHydration<IdOp>(hydrate([]), {
      capabilitiesFor: (op) => ({ ...CAPS, memoizable: op.id !== "volatile" }),
    })(core);
    expect(exec.capabilitiesFor?.({ id: "volatile", userTextId: "t1" })?.memoizable).toBe(false);
    expect(exec.capabilitiesFor?.({ id: "op1", userTextId: "t1" })?.memoizable).toBe(true);
  });
});
