/**
 * A parent's abort signal must not accumulate a listener per child it ever mounted.
 *
 * Cancelling a parent cancels its children, and the wiring for that is one `abort` listener per
 * child, added to the parent's signal at mount. `{ once: true }` looks like it cleans up and does
 * not: it unregisters when the listener FIRES, and on any run that is not cancelled it never fires.
 * So the listener — and the `AbortController` it closes over — is held for as long as the parent is,
 * one per child ENTRY. A sequence pays one per member; a loop pays one per iteration.
 *
 * That is a real leak with a loud symptom and a useless message. Node warns at ten
 * (`MaxListenersExceededWarning: Possible EventTarget memory leak detected. 11 abort listeners added
 * to [AbortSignal]`), which names a symptom that is true of every signal in the process and points
 * at nothing. Downstream it surfaced on a workflow whose middle phase loops draft → critique →
 * confidence, and on any RESUME, which re-walks a recorded tree and mounts every child again in one
 * burst.
 *
 * The count is asserted directly rather than by listening for the warning. A warning fires at a
 * threshold, so a test written around it passes for every leak smaller than ten and says nothing
 * about the shape of the bug; the invariant is that a finished child leaves nothing behind, and that
 * holds at one child as well as at fifty.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type ResolvedValue } from "@declarative-ai/exec";
import { newRegistry, ok } from "./fakes.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { WorkflowMetrics } from "../src/ports.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };
const result = (v: unknown) => ok(v as never) as ExecResult<ResolvedValue, WorkflowMetrics>;

const num = { schema: { type: "number" } } as const;

/**
 * `a` then `b`, with `b` sending control back to `a` — twelve passes of each.
 *
 * Twelve because ten is where Node starts warning: a fixture that stopped at nine would leave the
 * regression invisible in exactly the range that produced the bug report.
 */
function loopFiles(): Record<string, StateDef> {
  return {
    root: {
      label: "Root",
      limits: { max_iterations: 11 },
      children: {
        a: { state: "root/a" },
        b: { state: "root/b", transitions: [{ to: "a", when: ".run.iteration < .limits.max_iterations" }] },
      },
    } as StateDef,
    "root/a": { label: "a", outputs: { n: num }, operation: { kind: "function", function: "tick" } } as StateDef,
    "root/b": { label: "b", outputs: { n: num }, operation: { kind: "function", function: "tick" } } as StateDef,
  };
}

describe("the abort wiring between a parent and its children", () => {
  it("leaves no listener behind for a child that has finished", async () => {
    let calls = 0;
    const registry = newRegistry();
    registry.functions.set(
      "tick",
      hostFunction(async () => {
        calls += 1;
        return result({ n: calls });
      }, HOST),
    );

    // INSTRUMENTED, because the signal that leaks is not reachable from out here. A mount subscribes
    // to the parent INSTANCE's signal — an `AbortController` the engine makes for itself — and the
    // caller's signal only ever gets the one listener that composes it in. Counting what the caller
    // can see therefore reports a clean run for every possible version of this code, which is what
    // the first draft of this test did.
    //
    // So the prototype is wrapped and every add and remove is paired up per signal. What that
    // measures is the invariant itself — the most `abort` listeners any single signal held at one
    // time — rather than a proxy for it.
    const live = new Map<AbortSignal, number>();
    let peak = 0;
    const add = AbortSignal.prototype.addEventListener;
    const remove = AbortSignal.prototype.removeEventListener;
    AbortSignal.prototype.addEventListener = function (this: AbortSignal, type: string, ...rest: never[]) {
      if (type === "abort") {
        const n = (live.get(this) ?? 0) + 1;
        live.set(this, n);
        if (n > peak) peak = n;
      }
      return (add as (...a: unknown[]) => void).call(this, type, ...rest);
    } as typeof add;
    AbortSignal.prototype.removeEventListener = function (this: AbortSignal, type: string, ...rest: never[]) {
      if (type === "abort") live.set(this, Math.max(0, (live.get(this) ?? 0) - 1));
      return (remove as (...a: unknown[]) => void).call(this, type, ...rest);
    } as typeof remove;

    let outcome;
    try {
      const engine = new WorkflowEngine({ bundle: loadBundle(loopFiles(), "root"), registry });
      outcome = await engine.run({ inputs: {} });
    } finally {
      AbortSignal.prototype.addEventListener = add;
      AbortSignal.prototype.removeEventListener = remove;
    }

    expect(outcome.outcome).toBe("success");
    // 24 child mounts: twelve passes of two children. Every one of them added a listener, and before
    // the fix every one of them kept it — the root's signal peaked at 24.
    expect(calls).toBe(24);

    // The invariant: a parent holds listeners for the children RUNNING under it, not for the ones it
    // has ever had. This fixture runs its two children one after another, so the honest ceiling is
    // small and constant; what it must not be is a function of how many times the loop went round.
    expect(peak).toBeLessThan(5);
  });

  it("still cancels a running child through its parent", async () => {
    // The listener exists for a reason, and a cleanup that removed it too eagerly would pass the
    // test above while quietly breaking cancellation — so the guarantee is asserted beside it.
    const registry = newRegistry();
    let sawAbort = false;
    registry.functions.set(
      "block",
      hostFunction(async (_inputs: Record<string, unknown>, ctx?: { abortSignal?: AbortSignal }) => {
        await new Promise<void>((resolve) => {
          if (ctx?.abortSignal?.aborted === true) return resolve();
          ctx?.abortSignal?.addEventListener("abort", () => resolve(), { once: true });
          setTimeout(resolve, 2000);
        });
        sawAbort = ctx?.abortSignal?.aborted === true;
        return result({ n: 1 });
      }, HOST),
    );

    const files: Record<string, StateDef> = {
      root: { label: "Root", children: { a: { state: "root/a" } } } as StateDef,
      "root/a": { label: "a", outputs: { n: num }, operation: { kind: "function", function: "block" } } as StateDef,
    };

    const outer = new AbortController();
    const engine = new WorkflowEngine({ bundle: loadBundle(files, "root"), registry });
    const running = engine.run({ inputs: {}, abortSignal: outer.signal });
    // Long enough for the child to be mounted and its operation dispatched.
    await new Promise((r) => setTimeout(r, 50));
    outer.abort();
    await running;

    expect(sawAbort).toBe(true);
  });
});
