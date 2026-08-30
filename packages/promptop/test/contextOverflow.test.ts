/**
 * A conversation that cannot fit the model is refused BEFORE it is sent.
 *
 * The provider's own answer is `Prompt is too long`, with nothing about which conversation, how far
 * over, or what to do — and it arrives after the request has been built and paid for. One measured run
 * ended that way holding three passes of finished work: 2.5M characters against a model that could not
 * take a fraction of it, and the only thing the run could report was the provider's rejection.
 *
 * `contextLength` is the fact that makes a better answer possible, and it lives in the model catalog —
 * which is to say on the executor's side of the line, not the store's. That is the whole reason this
 * check is here rather than where a conversation is resolved.
 */
import { describe, expect, it } from "vitest";
import { createPromptExecutor } from "../src/executor.js";
import { fakeRunner, okOutcome, promptOp, errorOf } from "./fakes.js";

const big = (chars: number): string => "x".repeat(chars);

describe("a call too large for its model", () => {
  it("refuses unsent, naming the model, the limit, and what was assembled", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const stack = createPromptExecutor({ runner, contextLengthFor: () => 1_000 });
    // ~25k tokens estimated against a 1k limit — impossible by any tokenizer.
    const out = await stack.start(promptOp({ user: big(100_000) }), {}).result;

    const failure = errorOf(out);
    expect(failure?.reason).toMatch(/does not fit/);
    expect(failure?.reason).toMatch(/1,000-token context/);
    // UNSENT: the point is not to pay for the rejection.
    expect(calls).toHaveLength(0);
  });

  it("sends anything near the line and lets the provider judge", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    // ~1.25k tokens against a 1.2k limit: over, but inside the margin the estimate cannot resolve.
    const stack = createPromptExecutor({ runner, contextLengthFor: () => 1_200 });
    expect(errorOf(await stack.start(promptOp({ user: big(5_000) }), {}).result)).toBeUndefined();
    expect(calls).toHaveLength(1);
  });

  it("refuses nothing for a model whose limit is unknown", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const stack = createPromptExecutor({ runner, contextLengthFor: () => undefined });
    expect(errorOf(await stack.start(promptOp({ user: big(500_000) }), {}).result)).toBeUndefined();
    expect(calls).toHaveLength(1);
  });
});
