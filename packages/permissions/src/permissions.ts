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
 * A profile as a MAP over named tools, rather than a predicate over `readOnly`.
 *
 * The predicate form answers for the tools the host registered and has nothing to say about the ones
 * it did not — which is the whole of a real failure: a delegated agent arrives with a dozen built-ins
 * nobody modelled, the scope check can only report `unknown`, and every such call is escalated to a
 * human, including the ones the profile plainly permits. Twenty ungoverned reads under `read-only`,
 * with no fault anywhere in the mechanism.
 *
 * A table says something about every tool it names, and {@link ProfileTable.other} says something
 * about every tool it does not. Between them nothing is left unclassifiable, so nothing has to be
 * escalated for want of an opinion.
 *
 * Modes rather than booleans, deliberately: "out of scope" was always a `deny` in disguise, and
 * expressing it as one lets a profile say `ask` — "this profile permits reading, but tell me" —
 * which the boolean form could not.
 */
export interface ProfileTable {
  /** Per-tool, by name. */
  tools?: Record<string, PermissionMode>;
  /** What a tool this table does not name resolves to. */
  default?: PermissionMode;
  /**
   * What a tool the HOST does not register resolves to — see {@link ToolGateOptions.tools}.
   *
   * Distinct from {@link ProfileTable.default}, and the distinction is the point: "I have not decided
   * about `write_file`" and "I have never heard of this tool" are different questions, and only one
   * of them deserves `deny`. With one field they had one answer.
   */
  other?: PermissionMode;
}

/**
 * One authored scope: a PLACE, and what may happen there.
 *
 * The DATA shape lives here so a workflow can author it and the loader can carry it; the RESOLUTION
 * — glob matching, specificity, inheritance, what an unmatched path means — deliberately does not.
 * That belongs to the host, which is why the decision path takes a callback
 * ({@link ToolDecisionOptions.scopeOf}) rather than this array: a glob grammar in this package would
 * be a second place for "is this path inside the sandbox" to be answered differently.
 */
export interface ScopeDecl {
  /** A path glob. Absolute, or relative to whatever the host resolves against. */
  path?: string;
  /** A URL glob, for tools that reach the network. */
  url?: string;
  /** Modes for the tools this scope names. */
  tools?: Record<string, PermissionMode>;
  /** What a tool this scope does not name resolves to here. */
  default?: PermissionMode;
}

/** Either form. A predicate is the older, narrower statement; a table is the complete one. */
export type ProfileRule = ProfilePredicate | ProfileTable;

const isTable = (rule: ProfileRule | undefined): rule is ProfileTable => typeof rule === "object" && rule !== null;

/**
 * The mode a table gives one tool — its own entry, then `default`, then `other` for a name the host
 * does not register. `undefined` when the table declines to say anything at all.
 */
export function profileTableMode(
  table: ProfileTable,
  tool: string,
  registered: boolean,
): PermissionMode | undefined {
  const own = table.tools !== undefined && Object.hasOwn(table.tools, tool) ? table.tools[tool] : undefined;
  if (own !== undefined) return own;
  return registered ? table.default : (table.other ?? table.default);
}

/**
 * Whether a tool is in scope under a profile: `full` admits all; `read-only`/`plan` admit only read-only
 * tools; any other name resolves through `custom` (an unknown custom profile admits nothing — safe default).
 *
 * A table-form profile is in scope for anything it does not resolve to `deny`, which is what makes the
 * two forms interchangeable to a caller that only asks this question.
 */
export function inProfile(
  profile: PermissionProfile,
  tool: { name: string; readOnly: boolean },
  custom?: Record<string, ProfileRule>,
): boolean {
  if (profile === "full") return true;
  if (profile === "read-only" || profile === "plan") return tool.readOnly;
  const rule = custom?.[profile];
  if (isTable(rule)) return profileTableMode(rule, tool.name, true) !== "deny";
  return rule ? rule(tool) : false;
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

// --- Mid-run user questions -----------------------------------------------------

/** One choice a {@link UserQuestion} offers. */
export interface UserQuestionOption {
  label: string;
  /** What choosing this means — trade-offs, implications. */
  description?: string;
}

/**
 * One question a running agent asks the person driving it.
 *
 * NOT a permission. An approval asks "may this call run?"; this asks "which way do you want it?" —
 * the call itself IS the question, and putting it through the approval gate produces the absurdity
 * of a human approving the act of being asked. Its own vocabulary keeps the two channels apart.
 */
export interface UserQuestion {
  /** The complete question, e.g. "Which library should we use for date formatting?" */
  question: string;
  /** Short chip/tag label (e.g. "Library"). */
  header?: string;
  options: UserQuestionOption[];
  /** True ⇒ several options may be chosen; the answer is then a list of labels. */
  multiSelect?: boolean;
}

/** A batch of questions asked together, with the approval-scope key they arrived under. */
export interface UserQuestionRequest {
  questions: UserQuestion[];
  sessionId: string;
}

/**
 * The chosen answers, keyed by the QUESTION TEXT — a single label, a free-text answer, or (for a
 * multi-select) a list of labels. The question text is the key because it is the one field both
 * ends hold verbatim; an index would silently misalign the moment anything filtered the list.
 */
export type UserAnswers = Record<string, string | readonly string[]>;

/**
 * Put a running agent's questions to the person driving it.
 *
 * Resolves to the answers, or to `undefined` when nobody will answer — the question was dismissed,
 * or the run has no interactive surface at all. `undefined` is a real answer ("decide yourself"),
 * not an error: an agent that asks and hears nothing should proceed on its own judgment, which is
 * exactly what the adapters tell it.
 */
export type AskUser = (req: UserQuestionRequest) => Promise<UserAnswers | undefined>;

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
  /**
   * Profiles by name — a predicate, or a {@link ProfileTable}.
   *
   * A name here SHADOWS the built-in of the same name, which is how a host replaces `read-only` with
   * a table that has an opinion about the agent's own built-ins rather than escalating them.
   */
  profiles?: Record<string, ProfileRule>;
  /** Per-tool `smart`-mode policies: inspect the call and decide, or escalate to the human gate. */
  smart?: Record<string, SmartApprover>;
  /**
   * The host's own per-CALL narrowing — see {@link ToolDecisionOptions.scopeOf}.
   *
   * On the policy rather than passed at each gate because it is a property of the project's posture,
   * not of one operation: a sandbox bounds everything running under it. A host tool may read it off
   * `ctx.policy` too, which is what lets a tool that ENUMERATES (a glob, a grep) drop results it
   * would not have been allowed to open — a refusal at the call is not enough when the call's own
   * answer is the listing.
   */
  scopeOf?: ScopeNarrowing;
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

/** Everything the one decision needs. Shared by {@link withPermission} and {@link createToolGate}. */
export interface ToolDecisionOptions {
  ledger: PermissionLedger;
  sessionId: string;
  approve: Approver;
  /** The per-STATE authored mode for this tool, shadowing the workflow-wide baseline. */
  authoredMode?: PermissionMode;
  /** The `smart`-mode policy for this tool. When `smart` resolves and none is supplied, it escalates to `ask`. */
  smart?: SmartApprover;
  /** Custom profiles by name — a predicate, or a {@link ProfileTable}. See {@link ProfileRule}. */
  profiles?: Record<string, ProfileRule>;
  /** True when the HOST registered this tool — what tells `default` from `other` in a profile table. */
  registered?: boolean;
  /** The operation's own permission block, passed through to {@link ToolDecisionOptions.scopeOf}. */
  authored?: { scopes?: ScopeDecl[] } | undefined;
  /**
   * What a caller's own narrowing says about THIS call, given its input.
   *
   * The seam for a permission that depends on more than the tool's name — a path, a URL, whatever the
   * caller models. It is a callback rather than a table because this package should not learn what a
   * tool's arguments mean: which argument of `bash` is a path, and whether it falls inside somebody's
   * sandbox, is a question only the host can answer, and a glob grammar in here would be a second
   * place for it to be answered differently.
   *
   * The verdict NARROWS. It is folded in with {@link strictestMode}, so a caller's `deny` refuses a
   * call the profile would have allowed and a caller's `allow` never rescues one the profile refused.
   * `undefined` ⇒ the caller has nothing to say about this call.
   */
  scopeOf?: ScopeNarrowing;
}

/**
 * A host's per-call narrowing.
 *
 * `authored` is the operation's own permission block, so a host that layers a project-wide FLOOR
 * under a per-state table can see both at the moment of decision — the engine builds a gate per
 * state and passes that state's block through. Without it a narrowing could only ever be global,
 * and "this state may write under app/" would have nowhere to live.
 */
export type ScopeNarrowing = (
  tool: { name: string; readOnly?: boolean },
  input: FunctionInputs,
  authored?: { scopes?: ScopeDecl[] } | undefined,
) => PermissionMode | undefined;

/** How restrictive each mode is — `deny` ▸ `ask` ▸ `smart` ▸ `allow`. */
const MODE_RANK: Record<PermissionMode, number> = { allow: 0, smart: 1, ask: 2, deny: 3 };

/**
 * The more restrictive of two modes — how two independent narrowings compose.
 *
 * `smart` sits under `ask` because it MAY decide without a human and is not guaranteed to, so it
 * cannot dominate an explicit `ask`.
 */
export function strictestMode(a: PermissionMode, b: PermissionMode): PermissionMode {
  return MODE_RANK[a] >= MODE_RANK[b] ? a : b;
}

/** Why a call was refused, or that it may proceed. */
export type ToolDecision = { allow: true } | { allow: false; reason: string };

/**
 * Decide ONE tool call — profile, then mode, then `smart`, then the human.
 *
 * THE single implementation, and it is single on purpose. The two `policyEnforcement` styles reach it
 * by different routes — a composed runtime through {@link withPermission} wrapping each tool, a
 * delegated one through {@link createToolGate} answering its native permission callback — and before
 * this existed only the first route had one. The delegated side called `approve` directly, which made
 * `smart` unreachable (its policy was never consulted) and `allow` indistinguishable from `ask` (the
 * human was asked either way). Both were silent: the modes were configured, displayed, and ignored.
 *
 * `readOnly` is optional because the delegated caller does not always know it — an agent's own
 * built-in is not a tool we registered. See {@link ToolGate.modeOf} for what an unknown one resolves
 * to and why that is `ask` rather than a guess in either direction.
 */
export async function decideToolCall(
  tool: { name: string; readOnly?: boolean },
  input: FunctionInputs,
  opts: ToolDecisionOptions,
): Promise<ToolDecision> {
  const { ledger, sessionId, approve, authoredMode, smart, profiles, registered } = opts;
  const profile = ledger.resolveProfile(sessionId);

  /**
   * What the profile says, as a MODE.
   *
   * A table answers for every tool including the ones the host never registered, which is what
   * retires the `unknown ⇒ ask` escalation below for callers that supply one. A predicate — or a
   * built-in profile — still answers in/out, and `unknown` still escalates, because a predicate
   * genuinely cannot classify a tool whose `readOnly` nobody knows.
   */
  const rule = profiles?.[profile];
  const table = isTable(rule) ? profileTableMode(rule, tool.name, registered ?? tool.readOnly !== undefined) : undefined;
  if (table === undefined) {
    // Profile gate first: an out-of-scope tool is refused regardless of mode (a mutating tool under
    // `read-only`/`plan`, or one a custom profile's predicate excludes).
    const scoped = profileScopeOf(profile, tool, profiles);
    if (scoped === "out") return { allow: false, reason: `tool '${tool.name}' is out of the '${profile}' profile` };
    if (scoped === "unknown") {
      // An UNCLASSIFIABLE tool under a narrowing profile is escalated rather than resolved either
      // way. It must not slip through on an `allow` the profile would have refused — and a profile
      // TABLE is how a caller stops having to pay this, by having an opinion about the name.
      const resolved = ledger.resolve(tool.name, sessionId, authoredMode);
      if (resolved !== "deny") return finish("ask", tool, input, opts);
    }
  } else if (table === "deny") {
    return { allow: false, reason: `tool '${tool.name}' is denied by the '${profile}' profile` };
  }

  // The caller's own narrowing — a path scope, a URL allow-list, whatever it models. Composed as a
  // narrowing rather than an override: it may refuse what the profile allowed, never the reverse.
  const narrowed = opts.scopeOf?.(tool, input, opts.authored);
  let mode = ledger.resolve(tool.name, sessionId, authoredMode);
  if (table !== undefined) mode = strictestMode(mode, table);
  if (narrowed !== undefined) mode = strictestMode(mode, narrowed);
  if (mode === "deny") return { allow: false, reason: `tool '${tool.name}' denied by permission policy` };
  return finish(mode, tool, input, opts);
}

/** The tail of {@link decideToolCall}: `smart`, then the human, then the verdict. */
async function finish(
  start: PermissionMode,
  tool: { name: string; readOnly?: boolean },
  input: FunctionInputs,
  opts: ToolDecisionOptions,
): Promise<ToolDecision> {
  const { ledger, sessionId, approve, smart } = opts;
  let mode = start;
  if (mode === "smart") {
    // The smart policy decides directly, or returns `ask` to escalate to the human gate below.
    mode = smart ? await smart({ tool: tool.name, input, sessionId }) : "ask";
  }
  if (mode === "ask") {
    const decision = await approve({ tool: tool.name, input, sessionId });
    ledger.apply(tool.name, decision, sessionId);
    mode = decision.decision === "allow" ? "allow" : "deny";
  }
  return mode === "deny" ? { allow: false, reason: `tool '${tool.name}' denied by permission policy` } : { allow: true };
}

/**
 * Whether a profile admits a tool — `unknown` when the tool cannot be classified at all.
 *
 * The third answer exists for the delegated case only. `full` excludes nothing, so an unclassifiable
 * tool is fine there; under any NARROWING profile the predicate needs a `readOnly` we may not have.
 */
function profileScopeOf(
  profile: PermissionProfile,
  tool: { name: string; readOnly?: boolean },
  profiles?: Record<string, ProfileRule>,
): "in" | "out" | "unknown" {
  if (profile === "full") return "in";
  if (tool.readOnly === undefined) return "unknown";
  return inProfile(profile, { name: tool.name, readOnly: tool.readOnly }, profiles) ? "in" : "out";
}

/**
 * Wrap a {@link Tool} so every call is gated by the permission ledger for `(sessionId, toolName)`:
 * `allow` runs it, `deny` returns a {@link PermissionDenied} result, `ask` invokes the approver and applies
 * the returned decision before allowing or denying. The wrapped tool is itself a `Tool` (same
 * `description`/`inputSchema`), so it drops into a runtime's tool set unchanged.
 */
export function withPermission(
  tool: Tool,
  opts: Omit<ToolDecisionOptions, "authoredMode"> & { toolName: string; authoredMode?: PermissionMode },
): Tool {
  const { toolName, ...decision } = opts;
  return {
    description: tool.description,
    inputSchema: tool.inputSchema,
    readOnly: tool.readOnly,
    async run(input: FunctionInputs, ctx: ExecServices): Promise<JsonValue> {
      const verdict = await decideToolCall({ name: toolName, readOnly: tool.readOnly }, input, decision);
      if (!verdict.allow) return { denied: true, tool: toolName, reason: verdict.reason } satisfies PermissionDenied;
      return tool.run(input, ctx);
    },
  };
}

/**
 * The permission gate a DELEGATED agent drives its own loop through.
 *
 * A delegated adapter cannot be given wrapped tools — it runs its own loop and calls its own
 * built-ins, which we never registered and cannot wrap. What it has instead is a native permission
 * callback and an up-front configuration step, and this is what both of those consult so that they
 * consult the same thing {@link withPermission} does.
 */
export interface ToolGate {
  /**
   * The session's effective profile.
   *
   * Published because one transport can act on the profile and on nothing else: `codex exec` has no
   * per-tool gate at all, so its whole enforcement is the up-front `--sandbox`, and a `plan` profile
   * is the one thing that maps onto it exactly — a planning turn must not write. Without this the
   * sandbox came from a statically-configured option and a state that authored `plan` ran with
   * whatever the adapter had been constructed with.
   */
  readonly profile: PermissionProfile;
  /** Decide one call, input in hand. Everything resolves here: profile, mode, `smart`, the human. */
  check(tool: { name: string; readOnly?: boolean }, input: FunctionInputs): Promise<ToolDecision>;
  /**
   * The mode a tool resolves to WITHOUT its input — what up-front configuration is allowed to assume.
   *
   * Three callers care, and the distinctions are the whole reason this is separate from `check`:
   *
   *  - `deny` ⇒ never offer the tool at all. A floor needs no input to apply.
   *  - `allow` ⇒ safe to PRE-APPROVE, so the agent is not interrupted for it. This is what
   *    `allowedTools` should carry, and carrying every tool regardless of mode is what made an
   *    authored `ask` never ask.
   *  - anything else (`ask`, `smart`) ⇒ must go through {@link check}. `smart` in particular INSPECTS
   *    the input, so pre-approving it would decide the call before the thing that decides it ran.
   *
   * A tool the gate cannot classify — an agent's own built-in, with no `readOnly` we know — resolves
   * to `ask` under any narrowing profile rather than to `allow` or `deny`. Denying would refuse an
   * agent its own `Read` under `read-only`, which is the one thing that profile plainly permits;
   * allowing would let it write under a profile that forbids writing. Escalating puts the one
   * question we cannot answer to somebody who can.
   */
  modeOf(tool: { name: string; readOnly?: boolean }): PermissionMode;
}

/** What a gate governs: the tools we registered, plus the authored block that shadows the baseline. */
export interface ToolGateOptions {
  ledger: PermissionLedger;
  sessionId: string;
  approve: Approver;
  /** The tools we REGISTERED, by name — this is where a `readOnly` is known. */
  tools?: Record<string, { readOnly: boolean }>;
  /**
   * The operation's own `environment.permissions`, which shadows the workflow-wide baseline.
   *
   * `other` is what a tool the host does NOT register resolves to — an agent's own built-in, a tool
   * from somebody else's MCP server. Separate from `default` for the reason {@link ProfileTable.other}
   * gives: with one field, "I have not decided about this tool" and "I have never heard of it" had
   * one answer, and only one of them deserves `deny`.
   */
  authored?: {
    default?: PermissionMode;
    other?: PermissionMode;
    tools?: Record<string, PermissionMode>;
    /** WHERE each tool may act — carried to {@link ToolDecisionOptions.scopeOf}, never resolved here. */
    scopes?: ScopeDecl[];
  };
  smart?: Record<string, SmartApprover>;
  profiles?: Record<string, ProfileRule>;
  /** Passed to every decision — see {@link ToolDecisionOptions.scopeOf}. */
  scopeOf?: ToolDecisionOptions["scopeOf"];
  /**
   * Tools ALREADY wrapped with {@link withPermission}, which this gate must therefore not gate again.
   *
   * For a host that cannot know, at wiring time, whether the executor it is handing tools to will
   * enforce by callback or by wrapping. The safe move is to wrap — an unwrapped tool reaching a
   * `policyEnforcement: "none"` executor is ungated — and the cost is that a delegated adapter then
   * asks about the same call the wrapper is about to ask about again. One `ask`, two prompts.
   *
   * Naming them here resolves it at the right end: they report `allow` to CONFIGURATION, so the
   * adapter pre-approves them and its callback never fires, and the wrapper underneath makes the real
   * decision when the tool actually runs. Nothing is loosened — the gate that matters is the one
   * closest to the call.
   */
  preGated?: readonly string[];
}

/** Build the {@link ToolGate} for one delegated call. */
export function createToolGate(opts: ToolGateOptions): ToolGate {
  // OWN entries only, throughout. These maps are keyed by TOOL NAME, so a tool called `constructor`
  // would otherwise resolve its mode — and its smart rule — to a prototype member.
  const own = <T>(map: Record<string, T> | undefined, name: string): T | undefined =>
    map !== undefined && Object.hasOwn(map, name) ? map[name] : undefined;
  // `other` for a name the host never registered; `default` for one it did. Two questions, and the
  // gap between them is where an agent's built-ins used to fall.
  const authoredMode = (name: string): PermissionMode | undefined =>
    own(opts.authored?.tools, name) ??
    (own(opts.tools, name) !== undefined ? opts.authored?.default : (opts.authored?.other ?? opts.authored?.default));
  /** The registered tool's `readOnly` when we have it; the caller's claim otherwise. */
  const known = (tool: { name: string; readOnly?: boolean }): { name: string; readOnly?: boolean } => {
    const registered = own(opts.tools, tool.name);
    return registered !== undefined ? { name: tool.name, readOnly: registered.readOnly } : tool;
  };
  const decisionFor = (name: string): ToolDecisionOptions => ({
    ledger: opts.ledger,
    sessionId: opts.sessionId,
    approve: opts.approve,
    ...(authoredMode(name) !== undefined ? { authoredMode: authoredMode(name) } : {}),
    ...(own(opts.smart, name) !== undefined ? { smart: own(opts.smart, name) } : {}),
    ...(opts.profiles !== undefined ? { profiles: opts.profiles } : {}),
    registered: own(opts.tools, name) !== undefined,
    ...(opts.authored !== undefined ? { authored: opts.authored } : {}),
    ...(opts.scopeOf !== undefined ? { scopeOf: opts.scopeOf } : {}),
  });
  const preGated = new Set(opts.preGated ?? []);
  return {
    // A GETTER: a plan-mode exit rebinds the session's profile mid-run, and a snapshot taken when the
    // gate was built would go on reporting `plan` after the door had been opened.
    get profile(): PermissionProfile {
      return opts.ledger.resolveProfile(opts.sessionId);
    },
    // NOT short-circuited for a pre-gated tool. `preGated` is a statement about where the decision is
    // made, not that there is none — an adapter that asks anyway must still get the true answer.
    check: (tool, input) => decideToolCall(known(tool), input, decisionFor(tool.name)),
    modeOf: (tool) => {
      const it = known(tool);
      // Pre-approved AT CONFIGURATION so the adapter's callback does not fire — see `preGated`. The
      // wrapper underneath decides the call for real when the tool runs.
      if (preGated.has(it.name)) return "allow";
      const profile = opts.ledger.resolveProfile(opts.sessionId);
      const mode = opts.ledger.resolve(it.name, opts.sessionId, authoredMode(it.name));
      const rule = opts.profiles?.[profile];
      if (isTable(rule)) {
        // A table has an opinion about every name, so there is nothing to escalate for want of one.
        const said = profileTableMode(rule, it.name, own(opts.tools, it.name) !== undefined);
        return said === undefined ? mode : strictestMode(mode, said);
      }
      const scoped = profileScopeOf(profile, it, opts.profiles);
      if (scoped === "out") return "deny";
      // Same escalation as `decideToolCall`, and it has to be here too: this is the answer that
      // decides whether the tool is PRE-APPROVED, so resolving it to `allow` would skip the gate
      // entirely for exactly the tool we could not classify. A profile TABLE is how a caller stops
      // paying this — see above.
      return scoped === "unknown" && mode !== "deny" ? "ask" : mode;
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
