import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backgroundAgentsOf, laterResult, TurnHold } from "../src/turnHold.js";

const tasks = (...types: string[]): Record<string, unknown> => ({ type: "system", subtype: "background_tasks_changed", tasks: types.map((task_type, i) => ({ task_id: `t${i}`, task_type })) });
const init = { type: "system", subtype: "init" };

describe("what the process says is running", () => {
  it("counts agents, not shells: a dev server would hold a run for ever", () => {
    expect(backgroundAgentsOf(tasks("local_agent", "local_bash"))).toBe(1);
    expect(backgroundAgentsOf(tasks("local_bash"))).toBe(0);
    expect(backgroundAgentsOf(tasks())).toBe(0);
    expect(backgroundAgentsOf(init)).toBeUndefined();
  });
});

describe("whether a result ends the run", () => {
  beforeEach(() => void vi.useFakeTimers());
  afterEach(() => void vi.useRealTimers());

  it("does, with nothing behind it", () => {
    expect(new TurnHold().finished()).toBe(false);
  });

  it("does not while an agent runs, and the turn its end begins is the last", () => {
    const hold = new TurnHold();
    const end = vi.fn();
    hold.endWith(end);
    hold.heard(tasks("local_agent"));
    expect(hold.finished()).toBe(true);
    // ✅ measured: the list empties and `init` follows in the same tick.
    hold.heard(tasks());
    hold.heard(init);
    vi.advanceTimersByTime(60_000);
    expect(end).not.toHaveBeenCalled();
    expect(hold.finished()).toBe(false);
  });

  it("ends the run when what held it goes and no turn begins", () => {
    const hold = new TurnHold(1000);
    const end = vi.fn();
    hold.endWith(end);
    hold.heard(tasks("local_agent"));
    hold.finished();
    hold.heard(tasks());
    vi.advanceTimersByTime(999);
    expect(end).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(end).toHaveBeenCalledOnce();
  });

  it("waits for the turn a message written during the last one begins", () => {
    const hold = new TurnHold(1000);
    const end = vi.fn();
    hold.endWith(end);
    hold.sent();
    expect(hold.finished()).toBe(true);
    hold.heard(init);
    vi.advanceTimersByTime(5000);
    expect(end).not.toHaveBeenCalled();
    expect(hold.finished()).toBe(false);
  });

  it("does not hold a run told to stop, and says when there is no turn to interrupt", () => {
    const under = new TurnHold();
    under.heard(tasks("local_agent"));
    expect(under.stop()).toBe(false);
    expect(under.finished()).toBe(false);
    const between = new TurnHold();
    between.heard(tasks("local_agent"));
    between.finished();
    expect(between.stop()).toBe(true);
  });
});

describe("the run's result, over its turns", () => {
  it("takes the last word and the cumulative cost, and adds each turn's usage", () => {
    const first = { text: "LAUNCHED", costUsd: 0.04, usage: { inputTokens: 100, outputTokens: 259 } };
    const second = { text: "FINISHED", costUsd: 0.06, usage: { inputTokens: 40, outputTokens: 292 } };
    expect(laterResult(first, second)).toEqual({ text: "FINISHED", costUsd: 0.06, usage: { inputTokens: 140, outputTokens: 551 } });
  });

  it("keeps a value the first turn was asked for, and an answer a stopped turn left empty", () => {
    const later = laterResult({ text: "the answer", structured: { ok: true } }, { text: "" });
    expect(later).toEqual({ text: "the answer", structured: { ok: true } });
  });
});
