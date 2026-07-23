/**
 * Tool-call permissions (DESIGN §5.1, "Permissions: two orthogonal axes"). An agent's tool call is authorized by a MODE —
 * `allow` / `deny` / `ask` — resolved through a scope chain of in-memory overlays plus the workflow-authored
 * baseline. On `ask`, an interactive approval collects a human {@link PermissionDecision}; its `scope`
 * decides how long the choice persists, which is applied by writing the resolved mode into the matching
 * overlay layer. Everything here is in-memory: durable, cross-run policy belongs in the workflow definition
 * (the authored baseline), never in this decision path.
 *
 * This module is the MODE mechanism + scope chain. The orthogonal PROFILE axis (`read-only`/`plan`/`full`)
 * and the engine wiring layer on top of it.
 */
import type { FunctionInputs, JsonValue } from "@declarative-ai/ops";
import type { ExecServices, Tool } from "@declarative-ai/exec";

/**
 * How an in-scope tool call is authorized. `smart` defers to a bound {@link SmartApprover} that inspects the
 * tool + args and returns `allow`/`deny` directly, or `ask` to escalate to the human gate — so arg-pattern
 * policies (allow `git status`, ask `git push`) need no special primitive.
 */
export type PermissionMode = "allow" | "deny" | "ask" | "smart";

/** A smart approver's verdict for one call: authorize directly, or escalate to the human (`ask`). */
export type SmartVerdict = "allow" | "deny" | "ask";

/** A per-tool `smart`-mode policy: inspect the call and decide (or escalate). No human interaction itself. */
export type SmartApprover = (req: PermissionRequest) => SmartVerdict | Promise<SmartVerdict>;

/**
 * Which effects are in scope for a runtime operation (the orthogonal axis to {@link PermissionMode}):
 * `full` — every tool; `read-only` — only tools declaring `readOnly`; `plan` — read-only until
 * a human-gated exit rebinds the session's profile to `full` (see {@link planExitTool}). Any OTHER string is
 * a CUSTOM profile, resolved through a host-supplied {@link ProfilePredicate} map (the `(string & {})` keeps
 * the built-in literals as autocomplete hints while admitting custom names).
 */
export type PermissionProfile = "read-only" | "plan" | "full" | (string & {});

/** A custom profile: given a tool, is it in scope? Registered by name (e.g. a `search` profile). */
export type ProfilePredicate = (tool: { name: string; readOnly: boolean }) => boolean;

/**
 * Whether a tool is in scope under a profile: `full` admits all; `read-only`/`plan` admit only read-only
 * tools; any other name resolves through `custom` (an unknown custom profile admits nothing — safe default).
 */
export function inProfile(
  profile: PermissionProfile,
  tool: { name: string; readOnly: boolean },
  custom?: Record<string, ProfilePredicate>,
): boolean {
  if (profile === "full") return true;
  if (profile === "read-only" || profile === "plan") return tool.readOnly;
  const pred = custom?.[profile];
  return pred ? pred(tool) : false;
}

/**
 * How long a human's authorization persists — all IN-MEMORY, widening by containment (a call ⊂ session ⊂
 * workflow run ⊂ host process). `once` is not stored. Narrower shadows broader at resolve time.
 */
export type PermissionScope = "once" | "session" | "workflow-run" | "always";

/** A human's answer to an `ask`. `scope` selects which overlay layer the resolved mode is written to. */
export interface PermissionDecision {
  decision: "allow" | "deny";
  scope: PermissionScope;
}

/** The request handed to an approver when a tool call resolves to `ask`. */
export interface PermissionRequest {
  tool: string;
  /** The tool input as the model produced it. `FunctionInputs`, not `JsonValue`: a `blob` slot carries
   *  bytes (DESIGN §3.7), and an approver must be able to see what it is authorizing. */
  input: FunctionInputs;
  sessionId: string;
}

/** Collects a human decision for an `ask` — backed by an interactive `HostFunction` in the real system,
 *  a plain callback in tests. */
export type Approver = (req: PermissionRequest) => PermissionDecision | Promise<PermissionDecision>;

/**
 * The workflow-authored, durable baseline (the one non-ephemeral layer): a per-tool mode and a `default`
 * for unlisted tools. Authored as a workflow default merged with a per-state override; unset ⇒ `ask`.
 */
export interface PermissionBaseline {
  default?: PermissionMode;
  tools?: Record<string, PermissionMode>;
  /** The starting profile for a session (unset ⇒ `full`). A `plan` baseline is what makes a state plan-mode. */
  profile?: PermissionProfile;
}

/**
 * The COMPILED safety policy carried on `ctx.policy` (API.md, "ExecPolicy — the compiled policy on ctx.policy"): a real type, not an
 * opaque blob. It is the authored baseline plus the host-supplied resolution machinery an executor needs
 * to enforce it, and how it is enforced follows the executor's `policyEnforcement` capability:
 *   `"callback"` — a composed runtime wraps each tool with {@link withPermission} and gates per call;
 *   `"config"`   — a delegated adapter translates the policy into its agent's own permission config
 *                  (mode + allowed/denied tool names) and routes its native prompt back through `approve`;
 *   `"none"`     — the unit takes no tool calls, so nothing to enforce.
 */
export interface ExecPolicy {
  /** The authored, durable baseline: per-tool modes, the `default` for unlisted tools, starting profile. */
  baseline?: PermissionBaseline;
  /** Custom profile predicates by name — consulted when the session's profile isn't a built-in. */
  profiles?: Record<string, ProfilePredicate>;
  /** Per-tool `smart`-mode policies: inspect the call and decide, or escalate to the human gate. */
  smart?: Record<string, SmartApprover>;
  /** DELEGATED adapters only: the black-box agent's OWN tools this operation may use, by native name
   *  (a `Tool | NativeToolRef` rename binding's `native` side) — an allow-list, not an impl set. */
  nativeTools?: string[];
}

/** The result an agent's tool loop sees when a call is refused — the model reads it and continues.
 *  A type alias, not an interface, so it is structurally a `JsonValue` (it travels back to the model
 *  as the tool's JSON result). */
export type PermissionDenied = {
  denied: true;
  tool: string;
  reason: string;
};

export function isPermissionDenied(v: JsonValue): v is PermissionDenied & JsonValue {
  return typeof v === "object" && v !== null && !Array.isArray(v) && (v as { denied?: unknown }).denied === true;
}

/**
 * The in-memory permission overlays for ONE workflow run, plus the authored baseline. Owns the `run` layer
 * (all sessions in this run) and per-session layers; the `process` layer (spanning multiple runs, gone on
 * restart) is injected by the host so it can outlive any single run. Resolution walks
 * session → workflow-run → process → baseline → default(`ask`); a decision is applied at its `scope`'s layer.
 */
export class PermissionLedger {
  private readonly baseline: PermissionBaseline;
  private readonly process: Map<string, PermissionMode>;
  private readonly run = new Map<string, PermissionMode>();
  private readonly sessions = new Map<string, Map<string, PermissionMode>>();
  /** Per-session profile overrides — a `plan`→`full` exit writes here (profile is per-agent = per-session). */
  private readonly sessionProfiles = new Map<string, PermissionProfile>();

  constructor(opts: { baseline?: PermissionBaseline; process?: Map<string, PermissionMode> } = {}) {
    this.baseline = opts.baseline ?? {};
    // Host-owned so an `always` decision survives across runs in the same process (DESIGN §5.1, "Persistence granularity — a scope chain").
    this.process = opts.process ?? new Map();
  }

  /** Effective mode for `tool` in `sessionId`, most-specific layer first: session → run → process →
   *  `fallback` (the per-STATE authored mode, shadowing the workflow-wide baseline) → baseline → `ask`. */
  resolve(tool: string, sessionId: string, fallback?: PermissionMode): PermissionMode {
    return (
      this.sessions.get(sessionId)?.get(tool) ??
      this.run.get(tool) ??
      this.process.get(tool) ??
      fallback ??
      this.baseline.tools?.[tool] ??
      this.baseline.default ??
      "ask"
    );
  }

  /** The session's effective profile (its override ?? the authored baseline ?? `full`). */
  resolveProfile(sessionId: string): PermissionProfile {
    return this.sessionProfiles.get(sessionId) ?? this.baseline.profile ?? "full";
  }

  /** Set the session's profile — e.g. a plan-mode exit rebinding `plan` → `full` (see {@link planExitTool}). */
  setProfile(sessionId: string, profile: PermissionProfile): void {
    this.sessionProfiles.set(sessionId, profile);
  }

  /** Seed the authored profile ONCE — set only if the session has no override yet, so a later
   *  {@link setProfile} (e.g. a plan exit) is never clobbered by re-entering the state. */
  seedProfile(sessionId: string, profile: PermissionProfile): void {
    if (!this.sessionProfiles.has(sessionId)) this.sessionProfiles.set(sessionId, profile);
  }

  /** Record a decision at the layer its `scope` names (`once` writes nothing — it governs this call only). */
  apply(tool: string, decision: PermissionDecision, sessionId: string): void {
    const mode: PermissionMode = decision.decision === "allow" ? "allow" : "deny";
    switch (decision.scope) {
      case "once":
        return;
      case "session": {
        let m = this.sessions.get(sessionId);
        if (!m) this.sessions.set(sessionId, (m = new Map()));
        m.set(tool, mode);
        return;
      }
      case "workflow-run":
        this.run.set(tool, mode);
        return;
      case "always":
        this.process.set(tool, mode);
        return;
    }
  }
}

/**
 * Wrap a {@link Tool} so every call is gated by the permission ledger for `(sessionId, toolName)`:
 * `allow` runs it, `deny` returns a {@link PermissionDenied} result, `ask` invokes the approver and applies
 * the returned decision before allowing or denying. The wrapped tool is itself a `Tool` (same
 * `description`/`inputSchema`), so it drops into a runtime's tool set unchanged.
 */
export function withPermission(
  tool: Tool,
  opts: {
    ledger: PermissionLedger;
    sessionId: string;
    toolName: string;
    approve: Approver;
    authoredMode?: PermissionMode;
    /** The `smart`-mode policy for this tool. When `smart` resolves and none is supplied, it escalates to `ask`. */
    smart?: SmartApprover;
    /** Custom profile predicates by name — consulted when the session's profile isn't a built-in. */
    profiles?: Record<string, ProfilePredicate>;
  },
): Tool {
  const { ledger, sessionId, toolName, approve, authoredMode, smart, profiles } = opts;
  const deny = (reason: string): PermissionDenied => ({ denied: true, tool: toolName, reason });
  return {
    description: tool.description,
    inputSchema: tool.inputSchema,
    readOnly: tool.readOnly,
    async run(input: FunctionInputs, ctx: ExecServices): Promise<JsonValue> {
      // Profile gate first: an out-of-scope tool is refused regardless of mode (a mutating tool under
      // `read-only`/`plan`, or one a custom profile's predicate excludes).
      const profile = ledger.resolveProfile(sessionId);
      if (!inProfile(profile, { name: toolName, readOnly: tool.readOnly }, profiles)) {
        return deny(`tool '${toolName}' is out of the '${profile}' profile`);
      }
      let mode = ledger.resolve(toolName, sessionId, authoredMode);
      if (mode === "smart") {
        // The smart policy decides directly, or returns `ask` to escalate to the human gate below.
        mode = smart ? await smart({ tool: toolName, input, sessionId }) : "ask";
      }
      if (mode === "ask") {
        const decision = await approve({ tool: toolName, input, sessionId });
        ledger.apply(toolName, decision, sessionId);
        mode = decision.decision === "allow" ? "allow" : "deny";
      }
      if (mode === "deny") return deny(`tool '${toolName}' denied by permission policy`);
      return tool.run(input, ctx);
    },
  };
}

/**
 * The plan-mode exit gate (Claude Code's `ExitPlanMode`): a read-only tool the agent calls with its plan;
 * on human approval it rebinds the session's profile `plan` → `full`, so subsequent tool calls may mutate.
 * It carries its own approval (distinct from per-tool-call permission), so it is registered directly — NOT
 * wrapped by {@link withPermission} — and is `readOnly` so it stays callable while the profile is `plan`.
 */
export function planExitTool(opts: { ledger: PermissionLedger; sessionId: string; approve: Approver }): Tool {
  const { ledger, sessionId, approve } = opts;
  return {
    description: "Present the plan and request approval to leave plan mode and begin execution.",
    inputSchema: { type: "object", properties: { plan: { type: "string" } }, required: ["plan"] },
    readOnly: true,
    async run(input: FunctionInputs): Promise<JsonValue> {
      const decision = await approve({ tool: "exit_plan", input, sessionId });
      if (decision.decision === "allow") {
        ledger.setProfile(sessionId, "full");
        return { approved: true };
      }
      return { approved: false };
    },
  };
}
