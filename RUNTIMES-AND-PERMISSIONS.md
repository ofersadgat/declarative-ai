# Runtimes, Tools, Sessions & Permissions

> **Status: LANDED — implemented and green across 6 packages** (typecheck clean, 455 tests). Everything in
> this design is built except the real Claude Agent SDK verification (its boundary mapping needs the SDK
> installed) and the `opencode` adapter. Successor design to `HW-REDESIGN.md`, which unified operation blocks
> into `runtime` / `function` and introduced the `CapabilityRegistry`. This doc specifies what a `runtime`
> actually *is* across its several kinds, promotes **tools** to a registerable concept, names the run-scoped
> resource bundle a **Session**, and defines the **permission/approval** model. One scope-chain primitive
> resolves all of it. See **§8 Implementation status** for the details.

## Why

`HW-REDESIGN.md` landed a single `runtime` operation and a `RuntimeOp` normalized shape. But `RuntimeOp` is
prompt-shaped (`prompt` / `system` / `config` / `outputSchema`) — it reads like a plain LLM call wearing a
generic name. That hides real structure:

- There are **several kinds of runtime** — an `llm` call, the Claude Code CLI, the Claude Agent **SDK**,
  opencode, a command-line tool — with genuinely different qualities (do they own their loop? operate on a
  workspace? emit visible tool calls?). Piling them into one undifferentiated `RuntimeOp` prevents reuse.
- "Give the agent a workspace" isn't a runtime primitive — a workspace is reached **through tools** (read,
  write, bash) that happen to act on it. So the missing concept isn't *workspace*, it's a **registerable set
  of tools**. There is preliminary tool support (`ToolDefinition` + `ToolExecutor`) but no registration.
- There is no **permission / approval** model, and no coherent story for the run-scoped state a working
  agent accumulates (conversation, workspace, evolving authorizations).

This doc resolves all three, and the resolution is uniform: everything is name→thing lookup through a
**scope chain** (a run-scoped environment layered over the static declaration registry).

## 1. Runtimes: composed vs delegated

A **runtime** is an adapter (static, registered in `registry.runtimes`) that knows how to talk to one
backend. Two strategies, and a workflow author **chooses per state** which they want:

| Quality | **Composed** (`llm`) | **Delegated** (`claude-code`, `claude-code-sdk`, `opencode`) |
| --- | --- | --- |
| Owns the multi-turn loop | No — we drive a bounded loop (`llmStep`) | Yes — the agent loops internally |
| Operates on a workspace | Only via the tools we give it | Yes (a cwd + permissions) |
| Emits tool calls we can see | Yes (each in-loop call) | Mostly no (black box) |
| Tools | Our impls, run in-loop | Its built-ins, selected/permitted by name (or ours, MCP-injected) |

Two consequences worth stating outright:

- **CLI is a transport, not a category.** A command-line invocation is *sometimes* a plain function (`git
  status`, a linter) and *sometimes* the launch mechanism for a delegated agent (claude-code and opencode
  are binaries). "CLI" never appears in the taxonomy.
- **claude-code (CLI) ≠ claude-code-sdk.** Same underlying agent, two integration surfaces — two adapters.
- **An `llm` with reasoning + tools + a step budget *is* an agent.** The machinery already exists —
  `llmStep` runs a bounded tool loop (`toolExecutors` + `maxSteps`). So "make the `llm` runtime approach
  Claude Code" is mostly *wiring existing pieces + supplying tools*, not building a new loop. The goal is to
  get the composed option as close to the delegated agents as possible while still letting the user pick —
  e.g. one LLM for coding, a second for review, a third for general state-transition Q&A.

## 2. Functions and tools

- **Function** — named host code, `inputs → output` (sync or async). Can be *anything*. Graph-invokable as
  a state `function` operation.
- **Tool** — a Function **plus** the call-metadata an LLM needs to decide to call it: a `description`, an
  `inputSchema`, invocation semantics. So `Tool ⊂ Function`. This is already the shape today, just split
  across two types: `ToolExecutor` (the function) + `ToolDefinition` (name/description/schema). We unify the
  context object so **one impl can be surfaced either way** — as a graph function or as an agent tool.
- Because a tool *is* a function, a tool's body may itself invoke another runtime or sub-workflow. **Tools
  can be anything, just like functions** — they are not restricted to leaf host code. A function that calls
  a tool (or vice versa) is the bridge between the two surfaces.

### The Claude-Code-parity tool library

We build a library of function-tools matching Claude Code's toolset — `read_file`, `write`, `edit`, `bash`,
`grep`, `glob`, … These are the impls that turn the composed `llm` runtime into a coding agent, and they are
**where the workspace lives**. Handing an agent "a workspace" is really handing it these tools; the
workspace itself is a resource they close over (see Sessions).

## 3. Registry vs environment — one scope chain

Two containers of the same *shape* (typed, name-keyed facets reusing `Registry<T>`) but different
**lifetime and mutability**:

- **Static base — `CapabilityRegistry`**: `{ runtimes, functions, skills, tools }`. Declaration-time,
  host-provided, **shared across many runs**, effectively immutable during a run.
- **Run overlay — the environment**: the mutable, run-scoped state, keyed by `sessionId`.

Name resolution walks **overlay → base** (a scope chain, local-shadows-global). This is what makes runtime
rebinds safe: a rebind writes to the overlay, shadowing the base, so it is automatically run-scoped and
never leaks into another run or mutates shared host config. This lines up with the existing
declaration/environment split — the registry is *declaration*, the overlay is *environment*.

### Sessions: the run-scoped resource bundle

The run-scoped identity is a **Session**, keyed by `sessionId`. It owns
`{ conversation, workspace, permission bindings, tool renames }`. States that share a `sessionId` share the
whole bundle — the same lever as sharing a conversation today, generalized from "transcript" to "transcript
+ workspace + permissions + tools." Notes:

- **`Session ⊋ transcript`.** The conversation is one facet (`session.conversation`); the Session is the
  bundle. (This renames the earlier working term "Agent" — an "agent" was only ever the informal name for
  "everything owned under one `sessionId`," so the key *is* the concept. "Session" also avoids colliding
  with LLM-agent-the-behavior and the retired `agent` operation keyword.)
- **Sharing model**: a state's `runtime` block names a `sessionId`; same id ⇒ shared Session; absent ⇒ a
  fresh per-state Session. Explicit-id-to-share, fresh-by-default.
- **Workspace is not runtime-owned and not always shared.** It is a Session-owned resource: default-shared
  within a subtree (a review agent reads what a coding agent wrote — that sharing is the point), override to
  isolate (parallel fan-out in worktrees). Different runtimes sharing a workspace is common and good.

### Tool renames are just overlay bindings

The portable-vocabulary goal: the **logical tool name** is the vocabulary — a state says an agent may use
`read_file`, `bash`, and *the same name means the same thing everywhere*. Per-runtime differences are
handled by binding the logical name in the (Session-scoped) overlay — a **rename is not a separate map
type**, it is one more name→binding:

- Composed (`llm`): `read_file → our-fn` (run in-loop).
- Delegated with a native tool: `read_file → NativeRef("Read")` — a redirect reference the adapter
  translates into the agent's alias/allowlist (we cannot execute the agent's built-in ourselves).
- Delegated, our behavior: `read_file → our-fn` **injected via MCP**, so `bash` on claude-code is identical
  to `bash` on the `llm` runtime.

So the overlay value is `Function | NativeRef`. `native` vs `mcp` is a **delegated-only** concern; the `llm`
runtime is always our-impl-in-loop, and MCP is purely the transport for pushing our tools into a black box.

## 4. Permissions

### Two orthogonal axes: profile × mode

- **Profile** (direction / which effects are in scope): `read-only`, `plan`, `full`. Out-of-profile tools
  are denied outright. Any other profile name is a **custom** profile, resolved through a host-supplied
  `ProfilePredicate` (`EngineConfig.permissions.profiles`) — e.g. a `search` profile admitting only
  `grep`/`glob`. An unknown custom name admits nothing (safe default).
- **Mode** (how in-scope calls are authorized): `allow`, `deny`, `ask`, `smart`.

This generalizes Claude Code's flat mode list rather than copying it: `default` = full×ask,
`acceptEdits` = full×smart(allow-edits), `plan` = read-only×ask + exit gate, `bypassPermissions` = full×allow.

### Plan mode

Claude Code's plan mode = a **read-only profile** + a designated **exit gate** (`ExitPlanMode`) that
surfaces a plan, asks the human, and on approval flips the profile to execute. In this model that is not a
special case: `plan` is the read-only profile plus a human-gated transition that **rebinds** the profile
`read-only → full` — the exact approval-rebind mechanism below. A delegated runtime in `plan` sets the
agent's native plan mode where it has one; its exit gate routes back through our approval UI.

### Enforcement — the permission wrapper + interactive approval

Approval reuses the interactive-`function` mechanism (the same one that renders `human_review` UI). A
**wrapper around tool execution** decides each call:

```
tool call → resolve request_permission_<tool> (Session overlay → base) → run that approver function:
  allow → run the tool
  deny  → return a "denied" tool-result to the model (it keeps going; not terminal)
  ask   → invoke the interactive approval function (renders UI, awaits the human)
```

Key points:

- `request_permission_<tool>` is **wrapper-internal**, never exposed to the model (a model must never
  self-authorize). It reuses the name→function machinery only as the resolution mechanism.
- The approval function **returns a decision as data** — `{ decision, scope }` — and the **engine applies
  the rebind**. The function stays a pure input-collector, and the mode change becomes an observable
  `EngineEvent` rather than a hidden side-effect.
- **`smart` = a bound `SmartApprover` that inspects the tool + args** and returns `allow`/`deny` directly,
  or `ask` to escalate to the human gate — e.g. allow `git status`, ask `git push`. Supplied per tool via
  `EngineConfig.permissions.smart` (name → policy); a `smart` mode with no policy escalates to `ask`.
- A state whose tools are in `ask` mode is **non-memoizable** (its behavior depends on live human input) —
  consistent with interactive functions already being impure.

### Persistence granularity — a scope chain, all in-memory

When a human authorizes, they pick how long it sticks. **Every one of these is ephemeral/in-memory** —
nothing here crosses the durability boundary:

| Granularity | Written to | Lifetime |
| --- | --- | --- |
| `once` | nowhere (one-shot answer) | this call |
| `always this session` | Session overlay (`sessionId`) | this Session |
| `always this workflow run` | run overlay (`runId`) | this run, all its Sessions |
| `always` | process overlay | the host/executor process, across runs — gone on restart |

Resolution walks **most-specific → least**, so a narrower decision shadows a broader one:

```
session → workflow-run → process → definition-authored baseline → hard default (ask)
```

The first three are ephemeral overlays (live human decisions). The **process overlay lives in the host**,
above the per-run engine, and is threaded into each run's ctx. **Durable, cross-run policy is not a runtime
decision** — it belongs in the workflow definition (see below). "Remember this forever" = edit the
definition, not a permission click.

### The authored baseline (the one declaration layer)

Beneath the ephemeral overlays sits the durable layer: the **definition-authored** profile + per-tool modes.
Authored at **two levels — a workflow default with per-state override**. This is the only permission state
that is *declaration* rather than *environment*, and it is where a decision that should survive across runs
lives.

### Delegated approval fidelity

Delegated agents (claude-code-sdk / opencode) route their loop's tool calls back through **our** approval
function via the SDK's permission hook (a `canUseTool`-style callback), so the ask-UX is uniform across
runtimes. Fallback where an SDK offers no hook: set the agent's native permission mode, or refuse `ask`-mode
for that runtime. We design our-UI-first.

## 5. The unifying picture

One primitive — name→binding resolved through a scope chain — appears at every layer:

- **Capability resolution**: overlay → `CapabilityRegistry`.
- **Tool renames**: a Session-overlay binding (`logical → Function | NativeRef`).
- **Permissions**: `request_permission_<tool>` resolved session → run → process → authored baseline.

And it lands cleanly on declaration/environment: the **definition** sets durable baselines (authored policy,
cross-run intent); the **environment** accumulates live overrides at three in-memory lifetimes.

## 6. Cleanup this enables / requires

- **✅ Removed the dead `InteractionPort` seam from core** — `InteractionPort`, `ExecutionSpec.interaction`,
  the `interaction_request` `ExecEvent`, and the `interactive`-capability comment referencing
  `spec.interaction`. hw no longer reads any of it (interactive behavior is now `registry.functions` with
  `capabilities.interactive`). Clean break — JaiRA (the one consumer that still referenced it) migrates
  itself.
- **✅ Corrected HW-REDESIGN.md's stale "keeper" note** — it claimed the `ui.*` expression namespace stays
  `ui` because `function` is a reserved word. That namespace was renamed to `function.*`; the expr lexer
  treats `function` as an ordinary identifier (only `true`/`false`/`null` are reserved), so the note was
  wrong on both counts.
- **Resync DESIGN.md / API.md** (partial) — they still describe some pre-redesign surface; the new
  runtime-tools + permissions API is documented in API.md.

## 7. Deferred / open

- **A broader cross-run durable policy store** — explicitly *out* of the permission path; durability is the
  definition's job.
- **Delegated-hook coupling** — each SDK's permission callback is a distinct integration surface; the
  adapter phase absorbs that per runtime.

## Suggested build order

1. Unify `Function` / `Tool` (shared ctx) and add a `tools` facet + the Session overlay (scope-chain
   resolver).
2. Wire the `llm` runtime to pull registered tools into the existing `llmStep` loop.
3. Build the Claude-Code-parity tool library (most of the labor).
4. Permission wrapper: profile × mode, the approval-rebind seam, the four granularities.
5. Delegated adapters — `claude-code-sdk` first (programmatic is easier to drive than the CLI), then the
   native-permission-hook routing.

## 8. Implementation status

**Landed** (framework core; 410 tests, typecheck clean):

- **`Tool = HostFunction + metadata`** and the **`tools` registry facet** — `@declarative-ai/core`
  `contract.ts` (`Tool`, `NativeToolRef`, `CapabilityRegistry.tools`, `MapCapabilityRegistry`).
- **`llm` runtime executes registered tools in a bounded loop** — a state's `runtime.tools: string[]`
  resolves through `registry.tools`; the engine hands executables via `RuntimeOp.tools`; `createLlmRuntime`
  declares them and forwards `ctx.tools`; the llm-call executor adapts them into the existing `llmStep`
  loop. Unregistered tool ⇒ permanent op failure.
- **Permission model** — `@declarative-ai/core` `permissions.ts`: `PermissionMode` (allow/deny/ask/smart) ×
  `PermissionProfile` (read-only/plan/full), the `PermissionLedger` scope chain (session → workflow-run →
  process → authored baseline → `ask`), `withPermission` tool wrapper, `planExitTool` (the plan→full gate),
  and `SmartApprover` (arg-inspecting `smart`-mode policy that escalates to `ask` when uncertain). Custom
  profiles via `ProfilePredicate` (a named in-scope predicate, e.g. a `search` profile). Wired into the
  engine: `EngineConfig.permissions { approve, baseline, process, smart, profiles }` and per-state
  `runtime.permissions { profile, default, tools }` + `runtime.session`. Tools are permission-wrapped only
  when an approver is supplied; otherwise they run unguarded.
- **Workspace resource** — `Workspace { root }` in `@declarative-ai/core` (`stores.ts`), threaded as
  `ExecServices.workspace` and forwarded by the engine to each runtime's ctx.
- **Per-session Session overlay** — a state's `runtime.session` is the sharing key for its owned resources.
  **Conversation:** the transcript lives in the run's `SessionStore` as `SessionState.messages`, keyed per
  session — ONE store shared with the llm `withSession` path (both read `ctx.sessions`), so the built-in
  `conversationMode` preamble and a runtime's own session continuity never diverge. States sharing a session
  thread history; a distinct session isolates it.
  **Workspace:** `EngineConfig.workspaceFor(sessionId)` resolves each session's workspace (falling back to
  the run-level `services.workspace`), so a fan-out can isolate per-worktree. **Permissions:** already
  `sessionId`-keyed in the `PermissionLedger`. Absent `runtime.session` ⇒ the run's **default** session, so
  a plain workflow is one shared session (preserving SPEC §4.7 whole-run `full_history`).
- **Workspace tool library** — `@declarative-ai/tools`: `read_file` / `list_dir` / `grep` / `glob`
  (read-only), `write_file` / `edit_file` (mutating), and `run_command` (shell), operating on
  `ctx.workspace` with a path-escape guard (`resolveInWorkspace`, SPEC §7.2). `grep`/`glob` are pure-fs (no
  ripgrep/shell dep — cross-platform) with result caps and a default ignore set; `run_command` runs via
  Node's shell with a timeout + output caps and is `readOnly: false` so the `read-only`/`plan` profiles
  block it. `allTools` is the full set. Together these make the composed `llm` runtime a coding agent.
- **Delegated `claude-code` runtime adapter** — `@declarative-ai/claude-code` `createClaudeCodeRuntime`: a
  `Runtime` that maps a `RuntimeOp` + ctx onto an injectable `AgentQuery` seam (prompt, `cwd` from
  `ctx.workspace`, `allowedTools` from `runtime.tools`, `permissionMode`), routes the agent's native tool
  callback back through `ctx.approve`, and collects the stream into an `Outcome`. The approver is threaded to
  runtimes via `ExecServices.approve` (engine `childServices`). Fully tested against a fake query — the real
  `@anthropic-ai/claude-agent-sdk` import is a lazy, optional-peer default (`sdkQuery.ts`), **isolated and
  marked to verify against the installed SDK** (its message/callback field names are documented, not
  build-checked).
- **MCP tool injection** — the delegated adapter injects `runtime.tools` (OUR impls, closing over the
  runtime ctx) into the agent as an in-process MCP server, so a `read_file`/`bash` on `claude-code` runs the
  SAME code as on the composed `llm` runtime (the portable-vocabulary payoff). The engine hands a delegated
  runtime (`policyEnforcement: "callback"`) **RAW** tools — no engine-side `withPermission` wrap — so
  authorization flows only through the agent's `canUseTool` → `ctx.approve` and injected tools aren't
  double-gated. `injectTools: false` opts into native allow-list mode instead.
- **Tool renames (`NativeToolRef`)** — `createClaudeCodeRuntime({ nativeTools })` resolves a logical tool to
  the agent's NATIVE built-in `ref.native` (aliased into the allow-list) instead of injecting ours — so a
  run can use the agent's own `Read` for `read_file` while still MCP-injecting our `bash`. Tools with no
  entry default to injection. The allow-list handed to the agent is the native names plus the MCP-qualified
  names of the injected ones.
- **`InteractionPort` removed from core** (§6, first bullet) — clean break; JaiRA migrates itself.
- **HW-REDESIGN.md `ui.*` "keeper" note corrected** (§6, second bullet).

**Deferred** (external deps / large surface — the remaining §6/§7 items):

- **Verify + wire the real Claude Agent SDK** — install `@anthropic-ai/claude-agent-sdk` and confirm the
  `sdkQuery.ts` boundary mapping against the live shapes (result/`canUseTool` fields AND the
  `createSdkMcpServer`/`tool` MCP-injection API + `mcp__<server>__<tool>` allow-list convention), then add
  the **`opencode`** adapter.
- **Full API/SPEC/DESIGN doc resync** beyond the additions noted here.
