import { compose, MapSessionStore } from "@declarative-ai/core";
import { describe, expect, it } from "vitest";
import { LlmCallExecutor } from "../src/executor";
import { withDeadline, withMemoize, withRepair, withSession } from "../src/wrappers";
import { fakeRunner, memoCache, okOutcome, specOf } from "./fakes";

const core = () => new LlmCallExecutor({ runner: fakeRunner([okOutcome()]).runner });

describe("composition — two forms (#3) + typed requirements (#4)", () => {
  it("form 2 (inside-out builder) runs the stack", async () => {
    const exec = compose(core()).with(withRepair({ turns: 1 })).with(withMemoize({ cache: memoCache() }));
    const out = await exec.start(specOf(), {}).outcome;
    expect(out.error).toBeUndefined();
    expect(out.value).toEqual({ answer: "4" });
  });

  it("form 1 (direct nesting) and form 2 (builder) nest identically", async () => {
    const f1 = withMemoize({ cache: memoCache() }, withRepair({ turns: 1 }, core())); // withMemoize({cache}, withRepair({turns}, core))
    const f2 = compose(core()).with(withRepair({ turns: 1 })).with(withMemoize({ cache: memoCache() })); // core.with(a).with(b)
    const [o1, o2] = [await f1.start(specOf(), {}).outcome, await f2.start(specOf(), {}).outcome];
    expect(o1.value).toEqual(o2.value);
  });

  it("the builder is an Executor (drops into the same call sites)", () => {
    const exec = compose(core()).with(withRepair({ turns: 1 }));
    expect(exec.kind).toBe("llm-call");
    expect(exec.capabilities.structuredOutput).toBe(true);
  });

  // Compile-time gate (#4): the stack's `start` requires EXACTLY the env its wrappers add. These lines are
  // type-checked (the test project includes test/**) but never executed — a never-invoked closure.
  it("start() requires exactly the added requirements (compile-time)", () => {
    const typeOnly = (): void => {
      const dl = compose(core()).with(withDeadline());
      dl.start(specOf(), { deadline: { maxDurationMs: 1 }, stepStartMs: 0 }); // ✓ both supplied
      // @ts-expect-error withDeadline ADDED deadline+stepStartMs to the requirement — {} is missing them
      dl.start(specOf(), {});

      const sess = compose(core()).with(withSession()); // store-less → requires ctx.sessions
      sess.start(specOf(), { sessions: new MapSessionStore() }); // ✓
      // @ts-expect-error store-less withSession ADDED `sessions` to the requirement
      sess.start(specOf(), {});

      // Request #2: providing a seam at CONSTRUCTION removes it from start's requirement — for ANY subset,
      // because the config object mirrors the ctx seams and start requires `Omit<seams, keyof provided>`.
      const dlDeadline = compose(core()).with(withDeadline({ deadline: { maxDurationMs: 60_000 } }));
      dlDeadline.start(specOf(), { stepStartMs: 0 }); // ✓ deadline from config → needs only stepStartMs
      // @ts-expect-error stepStartMs is still required (per-execution origin)
      dlDeadline.start(specOf(), {});

      // stepStartMs alone → start needs only `deadline` (positional args could NOT express this):
      const dlStep = compose(core()).with(withDeadline({ stepStartMs: 0 }));
      dlStep.start(specOf(), { deadline: { maxDurationMs: 1 } }); // ✓
      // @ts-expect-error deadline is still required
      dlStep.start(specOf(), {});

      // both seams provided → start needs neither:
      const dlBoth = compose(core()).with(withDeadline({ deadline: { maxDurationMs: 1 }, stepStartMs: 0 }));
      dlBoth.start(specOf(), {}); // ✓

      // Requirements ACCUMULATE across the chain: deadline+stepStartMs AND sessions all required.
      const both = compose(core()).with(withDeadline()).with(withSession());
      both.start(specOf(), { deadline: { maxDurationMs: 1 }, stepStartMs: 0, sessions: new MapSessionStore() }); // ✓
      // @ts-expect-error missing `sessions` — proves withDeadline's requirement did not erase withSession's
      both.start(specOf(), { deadline: { maxDurationMs: 1 }, stepStartMs: 0 });
    };
    expect(typeof typeOnly).toBe("function");
  });
});
