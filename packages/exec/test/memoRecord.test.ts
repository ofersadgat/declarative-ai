/**
 * A memo HIT is indistinguishable from the call it replaces — including its PAYLOAD.
 *
 * `ctx.returnRecord` changes what an execution hands back (the whole `LlmOutput` beside the op's
 * output value) without changing what it computes. That makes it a request about the ANSWER's shape,
 * not part of the question — so the two obvious implementations are both wrong:
 *
 *  - **Key on the flag** and the same work is stored twice, once per shape.
 *  - **Cache what this caller asked for** and the entry cannot serve the other kind: a caller that
 *    wants the payload gets one without it, `withRecord` falls back to the projected value, and it
 *    stores a position holding no conversation — the original empty-transcript bug, re-entering
 *    through the cache.
 *
 * So the wrapper NORMALIZES on the way in (always ask for the payload) and SHAPES on the way out
 * (hand back only what this caller requested). One entry, either shape, no observable difference.
 */
import { describe, expect, it } from "vitest";
import { MapMemoCache, withMemoize } from "../src/index.js";
import type { ExecServices, Executor, Operation, InlineFamily } from "../src/index.js";

const PAYLOAD = { value: { answer: "4" }, finishReason: "stop", messages: [{ role: "assistant", content: "4" }] };

/** A core that reports the payload only when asked — exactly as a prompt executor does. */
function core(): { executor: Executor; runs: () => number; asked: () => boolean[] } {
  let runs = 0;
  const asked: boolean[] = [];
  return {
    runs: () => runs,
    asked: () => asked,
    executor: {
      capabilities: { memoizable: true } as never,
      metrics: { merge: (a) => a, empty: () => ({ durationMs: 0 }) },
      start: (_op, ctx: ExecServices) => {
        runs++;
        asked.push(ctx.returnRecord === true);
        return {
          events: (async function* () {})(),
          cancel: async () => {},
          result: Promise.resolve({
            value: { answer: "4" },
            metrics: { durationMs: 1 },
            ...(ctx.returnRecord === true ? { record: PAYLOAD } : {}),
          } as never),
        };
      },
    } as Executor,
  };
}

const op = { kind: "prompt", user: "q", config: {}, input: {}, output: { name: "o", kind: "json" } } as unknown as Operation<InlineFamily>;

describe("withMemoize and the record channel", () => {
  it("asks for the payload even when its caller did not, so ONE entry serves both", async () => {
    const { executor, runs, asked } = core();
    const memo = withMemoize({ cache: new MapMemoCache() }, executor);

    // Caller 1 wants only the value.
    const plain = await memo.start(op, {}).result;
    expect(plain.value).toEqual({ answer: "4" });
    expect(plain).not.toHaveProperty("record");

    // Caller 2 wants the payload — and gets it, from the SAME entry.
    const full = await memo.start(op, { returnRecord: true }).result;
    expect(full.value).toEqual({ answer: "4" });
    expect((full as { record?: unknown }).record).toEqual(PAYLOAD);

    // One execution, and it was asked for everything.
    expect(runs()).toBe(1);
    expect(asked()).toEqual([true]);
  });

  it("does not leak the payload to a caller that did not ask", async () => {
    const { executor } = core();
    const memo = withMemoize({ cache: new MapMemoCache() }, executor);
    await memo.start(op, { returnRecord: true }).result; // fill the cache WITH a payload
    const plain = await memo.start(op, {}).result;
    // A hit must be indistinguishable from the call it replaces. The call would not have reported a
    // payload to this caller, so neither does the hit.
    expect(plain).not.toHaveProperty("record");
  });
});
