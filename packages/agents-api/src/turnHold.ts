/**
 * Whether a finished turn is the end of the run — one rule, for the two transports over Claude's binary.
 *
 * A `result` ends a TURN, and both transports used to end the process on it by closing its input. A
 * process with a background agent running then kills that agent: `task_updated {"status":"killed"}`,
 * `task_notification {"status":"stopped"}`, and nothing ever tells the model. The turn had said "when it
 * reports, I'll merge its branch"; it never reported (2026-10-30, a subagent an hour into building a
 * runtime, killed at the process's own ten-minute bound).
 *
 * ✅ MEASURED (claude 2.1.287, stream-json in and out, input left open): the agent goes on after the
 * `result`, and when it ends the process says so — `background_tasks_changed`, `task_notification
 * {"status":"completed"}`, a fresh `system/init` — and takes a new turn of its own, which ends in a
 * second `result`. So the run ends at the last `result`, with nothing left behind it. `total_cost_usd`
 * is cumulative across those results; `usage` is each turn's own.
 *
 * Only AGENTS hold a run. A background shell is on the same list (`local_bash`), and may be a dev
 * server nobody will ever stop.
 */
import type { AgentResult, AgentTokenCounts } from "./seam.js";

/** How long, after what held the run has gone, a new turn has to begin before the run ends without one. */
export const TURN_SETTLE_MS = 5_000;

/** How many background agents a line says are running, or `undefined` for a line that does not say. */
export function backgroundAgentsOf(msg: Record<string, unknown>): number | undefined {
  if (msg["type"] !== "system" || msg["subtype"] !== "background_tasks_changed" || !Array.isArray(msg["tasks"])) return undefined;
  return (msg["tasks"] as Array<{ task_type?: unknown }>).filter((task) => typeof task?.task_type === "string" && task.task_type.includes("agent")).length;
}

/**
 * The run's result so far, with a later turn's on top. The last word is the run's, and `costUsd` is
 * cumulative, so both are simply the later one's; `usage` is each turn's own, so it adds up; and a value
 * a turn was asked for stays, since the turn a background agent's end began was not asked for one.
 */
export function laterResult(was: AgentResult | undefined, next: AgentResult): AgentResult {
  if (was === undefined) return next;
  const usage = addUsage(was.usage, next.usage);
  return {
    ...was,
    ...next,
    text: next.text.length > 0 ? next.text : was.text,
    ...(usage !== undefined ? { usage } : {}),
  };
}

function addUsage(a: AgentTokenCounts | undefined, b: AgentTokenCounts | undefined): AgentTokenCounts | undefined {
  if (a === undefined || b === undefined) return a ?? b;
  const sum: AgentTokenCounts = { ...a };
  for (const [key, value] of Object.entries(b) as Array<[keyof AgentTokenCounts, number]>) sum[key] = (sum[key] ?? 0) + value;
  return sum;
}

/** Follows one run's stream and says, at each `result`, whether the process goes on. */
export class TurnHold {
  private agents = 0;
  /** Messages written into the run that no turn has begun for. */
  private unread = 0;
  /** A `result` went by and the process goes on: no turn is under way. */
  private idle = false;
  private stopping = false;
  private over = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private end: (() => void) | undefined;

  constructor(private readonly settleMs: number = TURN_SETTLE_MS) {}

  /** What ends the process when a run being held turns out to have nothing more coming. */
  endWith(end: () => void): void {
    this.end = end;
  }

  /** Whether a background agent is running now: closing the input would only leave it to the process's bound. */
  get running(): boolean {
    return this.agents > 0;
  }

  /** Whether the process has been told to end, so a message written now would reach nothing. */
  get closed(): boolean {
    return this.over;
  }

  /** Every line of the stream, before anything else reads it. */
  heard(msg: Record<string, unknown>): void {
    const agents = backgroundAgentsOf(msg);
    if (agents !== undefined) {
      this.agents = agents;
      if (agents > 0) clearTimeout(this.timer);
      else if (this.idle) this.settle();
    }
    if (msg["type"] === "system" && msg["subtype"] === "init") {
      clearTimeout(this.timer);
      this.idle = false;
      this.unread = Math.max(0, this.unread - 1);
    }
  }

  /** A `result` arrived. True when the process goes on, and must not be closed. */
  finished(): boolean {
    if (this.stopping || (this.agents === 0 && this.unread === 0)) return false;
    this.idle = true;
    if (this.agents === 0) this.settle();
    return true;
  }

  /** A message was written into the run. */
  sent(): void {
    this.unread += 1;
  }

  /** The run was told to stop. True when no turn is under way, so that an interrupt would have nothing to end. */
  stop(): boolean {
    this.stopping = true;
    return this.idle;
  }

  /** The process is being closed. */
  close(): void {
    this.over = true;
    clearTimeout(this.timer);
  }

  private settle(): void {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.end?.(), this.settleMs);
  }
}
