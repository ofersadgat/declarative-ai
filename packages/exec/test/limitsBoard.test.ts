/** The limits board: per-account state, the two refresh rules, and the reporter a session layer hands down. */
import { describe, expect, it } from "vitest";
import {
  boardReporter,
  createLimitsBoard,
  limitStatusAt,
  mergeLimitReadings,
  remainingPercent,
  tightestWindow,
  weeklyWindow,
  type LimitReading,
} from "../src/index.js";

const complete = (at: string, five: number, week: number, resetsFive = "2026-09-24T14:05:00.000Z"): LimitReading => ({
  route: "claude-cli",
  plan: "max",
  windows: [
    { id: "five_hour", label: "5-hour", minutes: 300, usedPercent: five, resetsAt: resetsFive },
    { id: "seven_day", label: "Weekly", minutes: 10080, usedPercent: week, resetsAt: "2026-09-28T09:00:00.000Z" },
    { id: "seven_day_opus", label: "Weekly · Opus", minutes: 10080, usedPercent: 71, resetsAt: "2026-09-28T09:00:00.000Z", model: "opus" },
  ],
  status: "ok",
  source: "query",
  at,
  complete: true,
});

describe("merging readings", () => {
  it("keeps a figure a sparse update did not state, and takes its verdict", () => {
    const sparse: LimitReading = {
      route: "claude-cli",
      plan: null,
      windows: [{ id: "five_hour", label: "5-hour", minutes: 300, usedPercent: null, resetsAt: "2026-09-24T14:05:00.000Z", status: "exhausted" }],
      status: "exhausted",
      source: "stream",
      at: "2026-09-24T12:00:00.000Z",
    };
    const merged = mergeLimitReadings(complete("2026-09-24T11:00:00.000Z", 62, 33), sparse);
    expect(merged.windows).toHaveLength(3);
    expect(merged.windows[0]).toMatchObject({ id: "five_hour", usedPercent: 62, status: "exhausted" });
    expect(merged.plan).toBe("max");
    expect(merged.status).toBe("exhausted");
  });

  it("forgets a spent window once its reset has passed", () => {
    const r = complete("2026-09-24T11:00:00.000Z", 100, 33, "2026-09-24T14:05:00.000Z");
    expect(limitStatusAt(r, Date.parse("2026-09-24T14:00:00Z"))).toBe("exhausted");
    expect(limitStatusAt(r, Date.parse("2026-09-24T14:06:00Z"))).toBe("ok");
  });

  it("finds the window that decides the next call on a model, and what is left of it", () => {
    const r = complete("2026-09-24T11:00:00.000Z", 62, 33);
    const now = Date.parse("2026-09-24T12:00:00Z");
    expect(tightestWindow(r, "claude-sonnet-5", now)!.id).toBe("five_hour");
    expect(tightestWindow(r, "claude-opus-5-5", now)!.id).toBe("seven_day_opus");
    expect(remainingPercent(r, "claude-opus-5-5", now)).toBe(29);
    expect(weeklyWindow(r)!.id).toBe("seven_day");
  });
});

describe("the board", () => {
  it("holds a state per account, merged, and tells subscribers", () => {
    const board = createLimitsBoard();
    const seen: string[] = [];
    const sub = board.subscribe((account) => seen.push(account));
    board.report("anthropic:me", complete("2026-09-24T11:00:00.000Z", 62, 33));
    expect(board.state("anthropic:me").reading!.windows).toHaveLength(3);
    expect(board.state("nobody")).toEqual({ reading: null, updatedAt: null, lastSentAt: null, refreshing: false });
    expect(seen).toContain("anthropic:me");
    sub.dispose();
    board.close();
  });

  it("rule 1: refreshes when sending has gone on too long with nothing heard", async () => {
    let t = 0;
    let asked = 0;
    const board = createLimitsBoard({ now: () => t, refreshAfterSendingMs: 1000 });
    board.offerRefresh("a", async () => {
      asked += 1;
      return complete(new Date(t).toISOString(), 10, 10);
    });
    board.sent("a");
    t = 500;
    board.sent("a");
    expect(asked).toBe(0);
    t = 1600;
    board.sent("a");
    await board.refresh("a", { olderThanMs: Number.MAX_SAFE_INTEGER });
    expect(asked).toBe(1);
    board.close();
  });

  it("rule 2 on demand: fresh() refreshes a state that is too old and leaves a fresh one", async () => {
    let t = Date.parse("2026-09-24T11:00:00Z");
    let asked = 0;
    const board = createLimitsBoard({ now: () => t, refreshAfterMs: 60_000 });
    board.offerRefresh("a", async () => {
      asked += 1;
      return complete(new Date(t).toISOString(), 20, 20);
    });
    await board.fresh("a");
    expect(asked).toBe(1);
    await board.fresh("a");
    expect(asked).toBe(1);
    t += 120_000;
    await board.fresh("a");
    expect(asked).toBe(2);
    board.close();
  });

  it("a refresh that throws leaves the state as it was", async () => {
    const board = createLimitsBoard();
    board.report("a", complete("2026-09-24T11:00:00.000Z", 62, 33));
    board.offerRefresh("a", async () => {
      throw new Error("offline");
    });
    const s = await board.refresh("a");
    expect(s.reading!.windows[0]!.usedPercent).toBe(62);
    expect(s.refreshing).toBe(false);
    board.close();
  });

  it("the reporter maps routes to accounts and drops what maps nowhere", () => {
    const board = createLimitsBoard();
    const usage = boardReporter(board, (route) => (route.startsWith("claude") ? "anthropic:me" : undefined));
    usage.sent("claude-code");
    usage.limits({ ...complete("2026-09-24T11:00:00.000Z", 1, 1), route: "claude-cli" });
    usage.limits({ ...complete("2026-09-24T11:00:00.000Z", 1, 1), route: "somewhere" });
    expect([...board.all().keys()]).toEqual(["anthropic:me"]);
    expect(board.state("anthropic:me").lastSentAt).not.toBeNull();
    board.close();
  });
});
