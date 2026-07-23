import { describe, expect, it } from "vitest";
import {
  PermissionLedger,
  planExitTool,
  isPermissionDenied,
  withPermission,
  type Approver,
  type PermissionDecision,
  type ProfilePredicate,
  type SmartApprover,
} from "../src/permissions";
import type { ExecServices, FunctionInputs, Tool } from "@declarative-ai/exec";

const CTX: ExecServices = {};

/** A tool that records every input it actually executed with. `readOnly` marks it in-scope for read-only/plan. */
function recordingTool(readOnly = false): Tool & { runs: unknown[] } {
  const runs: unknown[] = [];
  return {
    runs,
    description: "echo",
    inputSchema: { type: "object" },
    readOnly,
    run: (input: FunctionInputs) => {
      runs.push(input);
      return { ok: true };
    },
  };
}

/** An approver that returns a fixed decision and counts how many times it was asked. */
function scriptedApprover(decision: PermissionDecision): Approver & { asked: number } {
  const fn = ((_req: unknown) => {
    fn.asked++;
    return decision;
  }) as Approver & { asked: number };
  fn.asked = 0;
  return fn;
}

describe("PermissionLedger — scope-chain resolution", () => {
  it("defaults to ask; baseline default and per-tool baseline layer under overrides", () => {
    const ledger = new PermissionLedger({ baseline: { default: "deny", tools: { read_file: "allow" } } });
    expect(ledger.resolve("bash", "s1")).toBe("deny"); // baseline default
    expect(ledger.resolve("read_file", "s1")).toBe("allow"); // per-tool baseline
    expect(new PermissionLedger().resolve("bash", "s1")).toBe("ask"); // nothing set anywhere
  });

  it("narrower scope shadows broader: session > run > process > baseline", () => {
    const proc = new Map<string, "allow" | "deny" | "ask">();
    const ledger = new PermissionLedger({ baseline: { default: "ask" }, process: proc });
    ledger.apply("bash", { decision: "allow", scope: "always" }, "s1"); // process layer
    expect(ledger.resolve("bash", "s1")).toBe("allow");
    ledger.apply("bash", { decision: "deny", scope: "workflow-run" }, "s1"); // run shadows process
    expect(ledger.resolve("bash", "s1")).toBe("deny");
    ledger.apply("bash", { decision: "allow", scope: "session" }, "s1"); // session shadows run
    expect(ledger.resolve("bash", "s1")).toBe("allow");
    expect(ledger.resolve("bash", "s2")).toBe("deny"); // a different session still sees the run layer
  });

  it("`always` writes the host-owned process map, so it can outlive the run", () => {
    const proc = new Map<string, "allow" | "deny" | "ask">();
    const ledger = new PermissionLedger({ process: proc });
    ledger.apply("bash", { decision: "allow", scope: "always" }, "s1");
    expect(proc.get("bash")).toBe("allow"); // a fresh ledger sharing this process map would inherit it
  });

  it("`once` records nothing", () => {
    const ledger = new PermissionLedger();
    ledger.apply("bash", { decision: "allow", scope: "once" }, "s1");
    expect(ledger.resolve("bash", "s1")).toBe("ask"); // unchanged
  });
});

describe("withPermission — the tool wrapper", () => {
  it("allow runs the tool; deny returns a PermissionDenied without running it", async () => {
    const allowTool = recordingTool();
    const denyTool = recordingTool();
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { tools: { a: "allow", d: "deny" } } });

    const a = withPermission(allowTool, { ledger, sessionId: "s1", toolName: "a", approve });
    const d = withPermission(denyTool, { ledger, sessionId: "s1", toolName: "d", approve });

    expect(await a.run({ x: 1 }, CTX)).toEqual({ ok: true });
    expect(allowTool.runs).toEqual([{ x: 1 }]);

    const denied = await d.run({ y: 2 }, CTX);
    expect(isPermissionDenied(denied)).toBe(true);
    expect(denyTool.runs).toHaveLength(0); // never executed
    expect(approve.asked).toBe(0); // neither allow nor deny asks a human
  });

  it("ask invokes the approver, applies the decision, then allows/denies accordingly", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger(); // everything defaults to ask
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve });

    const out = await wrapped.run({ cmd: "ls" }, CTX);
    expect(out).toEqual({ ok: true });
    expect(approve.asked).toBe(1);
    expect(tool.runs).toEqual([{ cmd: "ls" }]);
  });

  it("`always this session` persists: the second call does not ask again", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "allow", scope: "session" });
    const ledger = new PermissionLedger();
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve });

    await wrapped.run({ n: 1 }, CTX);
    await wrapped.run({ n: 2 }, CTX);

    expect(approve.asked).toBe(1); // asked once; the session-scoped allow covered the second call
    expect(tool.runs).toEqual([{ n: 1 }, { n: 2 }]);
    expect(ledger.resolve("bash", "s1")).toBe("allow");
  });

  it("an ask that denies returns PermissionDenied and does not run the tool", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "deny", scope: "workflow-run" });
    const ledger = new PermissionLedger();
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve });

    const out = await wrapped.run({ cmd: "rm -rf /" }, CTX);
    expect(isPermissionDenied(out)).toBe(true);
    expect(tool.runs).toHaveLength(0);
    expect(ledger.resolve("bash", "s1")).toBe("deny"); // recorded run-wide
  });
});

describe("profile axis (read-only / plan / full)", () => {
  it("read-only profile denies a mutating tool outright, regardless of an allow mode", async () => {
    const write = recordingTool(false); // mutating
    const read = recordingTool(true); // read-only
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { default: "allow", profile: "read-only" } });

    const w = withPermission(write, { ledger, sessionId: "s1", toolName: "write", approve });
    const r = withPermission(read, { ledger, sessionId: "s1", toolName: "read", approve });

    expect(isPermissionDenied(await w.run({}, CTX))).toBe(true); // out of profile
    expect(write.runs).toHaveLength(0);
    expect(await r.run({}, CTX)).toEqual({ ok: true }); // read-only tool is in scope
  });

  it("plan mode: a mutating tool is blocked until planExitTool flips the session to full", async () => {
    const write = recordingTool(false);
    const approveTool = scriptedApprover({ decision: "allow", scope: "once" });
    const approvePlan = scriptedApprover({ decision: "allow", scope: "session" });
    const ledger = new PermissionLedger({ baseline: { default: "allow", profile: "plan" } });

    const w = withPermission(write, { ledger, sessionId: "s1", toolName: "write", approve: approveTool });
    expect(isPermissionDenied(await w.run({ a: 1 }, CTX))).toBe(true); // plan ⇒ mutating tool blocked

    const exit = planExitTool({ ledger, sessionId: "s1", approve: approvePlan });
    expect(await exit.run({ plan: "do the thing" }, CTX)).toEqual({ approved: true });
    expect(ledger.resolveProfile("s1")).toBe("full");

    expect(await w.run({ a: 2 }, CTX)).toEqual({ ok: true }); // now executes
    expect(write.runs).toEqual([{ a: 2 }]);
  });

  it("plan exit denied leaves the profile at plan", async () => {
    const ledger = new PermissionLedger({ baseline: { profile: "plan" } });
    const exit = planExitTool({ ledger, sessionId: "s1", approve: scriptedApprover({ decision: "deny", scope: "session" }) });
    expect(await exit.run({ plan: "nope" }, CTX)).toEqual({ approved: false });
    expect(ledger.resolveProfile("s1")).toBe("plan");
  });
});

describe("smart mode (arg-inspecting policy)", () => {
  // allow the read tool; for anything else, ask when the input is flagged dangerous, else deny.
  const policy: SmartApprover = (req) => (req.tool === "read" ? "allow" : (req.input as { danger?: boolean }).danger ? "ask" : "deny");

  it("smart 'allow' runs the tool without asking a human", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "deny", scope: "once" }); // would deny if consulted
    const ledger = new PermissionLedger({ baseline: { default: "smart" } });
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "read", approve, smart: policy });
    expect(await wrapped.run({}, CTX)).toEqual({ ok: true });
    expect(approve.asked).toBe(0);
    expect(tool.runs).toHaveLength(1);
  });

  it("smart 'deny' blocks without asking a human", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { default: "smart" } });
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve, smart: policy });
    expect(isPermissionDenied(await wrapped.run({}, CTX))).toBe(true);
    expect(approve.asked).toBe(0);
    expect(tool.runs).toHaveLength(0);
  });

  it("smart 'ask' escalates to the human gate", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { default: "smart" } });
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve, smart: policy });
    expect(await wrapped.run({ danger: true }, CTX)).toEqual({ ok: true });
    expect(approve.asked).toBe(1); // the uncertain case was escalated to the human
  });

  it("smart with no policy supplied falls back to asking the human", async () => {
    const tool = recordingTool();
    const approve = scriptedApprover({ decision: "deny", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { default: "smart" } });
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "bash", approve }); // no smart
    expect(isPermissionDenied(await wrapped.run({}, CTX))).toBe(true);
    expect(approve.asked).toBe(1);
  });
});

describe("custom profiles", () => {
  // a "search" profile: only the grep/glob tools are in scope, by name.
  const profiles: Record<string, ProfilePredicate> = { search: (t) => t.name === "grep" || t.name === "glob" };

  it("admits only the tools the custom profile's predicate allows", async () => {
    const grep = recordingTool(true);
    const write = recordingTool(false);
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = new PermissionLedger({ baseline: { default: "allow", profile: "search" } });
    const g = withPermission(grep, { ledger, sessionId: "s1", toolName: "grep", approve, profiles });
    const w = withPermission(write, { ledger, sessionId: "s1", toolName: "write", approve, profiles });

    expect(await g.run({}, CTX)).toEqual({ ok: true }); // in the search profile
    expect(isPermissionDenied(await w.run({}, CTX))).toBe(true); // out of the search profile
  });

  it("an unknown custom profile admits nothing (safe default)", async () => {
    const tool = recordingTool(true);
    const ledger = new PermissionLedger({ baseline: { default: "allow", profile: "mystery" } });
    const wrapped = withPermission(tool, { ledger, sessionId: "s1", toolName: "grep", approve: scriptedApprover({ decision: "allow", scope: "once" }) });
    expect(isPermissionDenied(await wrapped.run({}, CTX))).toBe(true); // no predicate for "mystery"
  });
});
