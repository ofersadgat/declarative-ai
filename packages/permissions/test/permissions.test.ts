import { describe, expect, it } from "vitest";
import {
  createToolGate,
  PermissionLedger,
  planExitTool,
  isPermissionDenied,
  withPermission,
  type Approver,
  type PermissionDecision,
  type ProfilePredicate,
  type SmartApprover,
} from "../src/permissions.js";
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

describe("the delegated gate — the same decision, reached by the other route", () => {
  /**
   * A delegated agent cannot be handed wrapped tools: it runs its own loop and calls its own
   * built-ins, which nobody registered and nothing can wrap. So it answers its native permission
   * callback through {@link createToolGate} instead — and the point of these is that the two routes
   * reach ONE implementation.
   *
   * Before the gate existed the delegated side called the human approver directly, which collapsed
   * three of the four modes without failing: `smart` never ran its policy, `allow` asked anyway, and
   * the session profile was never consulted at all.
   */
  const gateFor = (opts: Partial<Parameters<typeof createToolGate>[0]> = {}) => {
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const ledger = opts.ledger ?? new PermissionLedger({});
    return {
      approve,
      ledger,
      gate: createToolGate({ ledger, sessionId: "s1", approve, ...opts }),
    };
  };

  it("runs the SMART policy instead of asking a human", async () => {
    let inspected: FunctionInputs | undefined;
    const smart: SmartApprover = ({ input }) => {
      inspected = input;
      return (input as { danger?: boolean }).danger === true ? "deny" : "allow";
    };
    const { gate, approve } = gateFor({
      tools: { bash: { readOnly: false } },
      authored: { tools: { bash: "smart" } },
      smart: { bash: smart },
    });

    expect(await gate.check({ name: "bash" }, { danger: false })).toEqual({ allow: true });
    expect(inspected).toEqual({ danger: false });
    expect(await gate.check({ name: "bash" }, { danger: true })).toMatchObject({ allow: false });
    // The whole point: a `smart` tool decided twice and the human was never involved.
    expect(approve.asked).toBe(0);
  });

  it("does not ask about a tool the caller set to `allow`", async () => {
    const { gate, approve } = gateFor({ tools: { bash: { readOnly: false } }, authored: { tools: { bash: "allow" } } });
    expect(await gate.check({ name: "bash" }, {})).toEqual({ allow: true });
    expect(approve.asked).toBe(0);
  });

  it("refuses a `deny` without asking, and asks about an `ask`", async () => {
    const { gate, approve } = gateFor({
      tools: { bash: { readOnly: false }, read_file: { readOnly: true } },
      authored: { tools: { bash: "deny", read_file: "ask" } },
    });
    expect(await gate.check({ name: "bash" }, {})).toMatchObject({ allow: false });
    expect(approve.asked).toBe(0);
    expect(await gate.check({ name: "read_file" }, {})).toEqual({ allow: true });
    expect(approve.asked).toBe(1);
  });

  it("applies the session PROFILE, which the direct-approver path skipped entirely", async () => {
    // A `read-only` state could previously be talked into a write by one distracted click, because
    // nothing between the agent and the human knew the profile existed.
    const ledger = new PermissionLedger({ baseline: { profile: "read-only" } });
    const { gate, approve } = gateFor({
      ledger,
      tools: { bash: { readOnly: false }, read_file: { readOnly: true } },
      authored: { tools: { bash: "allow", read_file: "allow" } },
    });
    expect(await gate.check({ name: "bash" }, {})).toMatchObject({ allow: false, reason: expect.stringContaining("read-only") });
    expect(approve.asked).toBe(0); // refused outright — a profile needs no human
    expect(await gate.check({ name: "read_file" }, {})).toEqual({ allow: true });
  });

  it("honours the STATE's authored mode over the workflow-wide baseline", async () => {
    // The reason the gate exists at all: raw tools used to mean the state's own `permissions` block
    // reached the adapter through no channel whatsoever.
    const ledger = new PermissionLedger({ baseline: { tools: { bash: "allow" } } });
    const { gate, approve } = gateFor({ ledger, tools: { bash: { readOnly: false } }, authored: { tools: { bash: "deny" } } });
    expect(await gate.check({ name: "bash" }, {})).toMatchObject({ allow: false });
    expect(approve.asked).toBe(0);
  });

  describe("modeOf — what up-front configuration may assume", () => {
    it("reports `allow` only for a tool that really is pre-approvable", () => {
      const { gate } = gateFor({
        tools: { a: { readOnly: false }, b: { readOnly: false }, c: { readOnly: false }, d: { readOnly: false } },
        authored: { tools: { a: "allow", b: "ask", c: "smart", d: "deny" } },
      });
      // `allowedTools` carries exactly the first. Carrying all four is what made an authored `ask`
      // never ask — and pre-approving `smart` would decide the call before its policy ran.
      expect(gate.modeOf({ name: "a" })).toBe("allow");
      expect(gate.modeOf({ name: "b" })).toBe("ask");
      expect(gate.modeOf({ name: "c" })).toBe("smart");
      expect(gate.modeOf({ name: "d" })).toBe("deny");
    });

    it("ESCALATES a tool it cannot classify under a narrowing profile", async () => {
      // An agent's own `Bash` is not a tool we registered, so there is no `readOnly` to judge it by.
      // Denying would refuse its `Read` under `read-only` — the one thing that profile plainly
      // permits — and allowing would let it write under a profile that forbids writing.
      const ledger = new PermissionLedger({ baseline: { profile: "read-only", default: "allow" } });
      const { gate, approve } = gateFor({ ledger });
      expect(gate.modeOf({ name: "Bash" })).toBe("ask");
      expect(await gate.check({ name: "Bash" }, {})).toEqual({ allow: true });
      expect(approve.asked).toBe(1);
    });

    it("leaves an unclassifiable tool alone under `full`, which excludes nothing", () => {
      const ledger = new PermissionLedger({ baseline: { default: "allow" } });
      const { gate } = gateFor({ ledger });
      expect(gate.modeOf({ name: "Bash" })).toBe("allow");
    });

    it("still honours an explicit `deny` for a tool it cannot classify", () => {
      const ledger = new PermissionLedger({ baseline: { profile: "read-only", tools: { Bash: "deny" } } });
      const { gate } = gateFor({ ledger });
      expect(gate.modeOf({ name: "Bash" })).toBe("deny");
    });
  });

  it("reads OWN entries only, so a tool named `constructor` cannot resolve to a prototype member", () => {
    const { gate } = gateFor({ authored: { tools: {} }, smart: {} });
    expect(gate.modeOf({ name: "constructor" })).toBe("ask");
    expect(gate.modeOf({ name: "toString" })).toBe("ask");
  });
});

describe("preGated — a tool already wrapped, so the gate must not gate it twice", () => {
  /**
   * For a host that cannot know, when it wires the tools, which executor will answer. Wrapping is the
   * safe default — an unwrapped tool reaching a `policyEnforcement: "none"` executor is ungated — but
   * if the route turns out to be a delegated agent, that agent asks about the very call the wrapper is
   * about to ask about again. One `ask`, two prompts.
   */
  it("reports `allow` to CONFIGURATION, so the adapter pre-approves and its callback never fires", () => {
    const ledger = new PermissionLedger({});
    const gate = createToolGate({
      ledger,
      sessionId: "s1",
      approve: scriptedApprover({ decision: "allow", scope: "once" }),
      tools: { bash: { readOnly: false } },
      authored: { tools: { bash: "ask" } },
      preGated: ["bash"],
    });
    expect(gate.modeOf({ name: "bash" })).toBe("allow");
    // And only for the named one — everything else resolves normally.
    expect(gate.modeOf({ name: "Write" })).toBe("ask");
  });

  it("still answers `check` truthfully — pre-gated says WHERE the decision is made, not that there is none", async () => {
    // An adapter that asks anyway must not be told `allow` for a tool the policy denies.
    const ledger = new PermissionLedger({});
    const approve = scriptedApprover({ decision: "allow", scope: "once" });
    const gate = createToolGate({
      ledger,
      sessionId: "s1",
      approve,
      tools: { bash: { readOnly: false } },
      authored: { tools: { bash: "deny" } },
      preGated: ["bash"],
    });
    expect(await gate.check({ name: "bash" }, {})).toMatchObject({ allow: false });
    expect(approve.asked).toBe(0);
  });
});
