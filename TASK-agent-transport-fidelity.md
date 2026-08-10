# Task: agent-transport fidelity

Bring the delegated-agent transports (`@declarative-ai/agents-api`, `@declarative-ai/agents-cli`) up to
the losslessness and control standard the AI SDK path already meets.

## Why

`llmConfig.ts` states the goal on the way IN — a stored config "transforms losslessly into a real call."
`output.ts` implements it on the way OUT — `thinking` segments with `providerMetadata` preserved intact,
`toolCalls`, `toolResults`, verbatim `messages` ("the provider's own log rather than a reconstruction"),
eight token counts plus `rawUsage` kept "so `costUsd` is always recomputable."

The agent boundary honours neither. `sdkQuery` collapses every non-`result` message to `{ type: "other" }`;
`AgentExecutor.invoke` fabricates `finishReason: "stop"` and synthesizes a one-turn message log; `runAgent`
reads three fields off the lowered `LlmCallDefinition` and drops the rest. The information is all there in
the SDK stream — it is thrown away at the mapping, not missing at the source.

Most of this is filling in types that already exist. Work in order; each task should typecheck and leave
the suite green on its own.

## Conventions to hold to

- Fakes only. No test touches a provider, a network, or a real binary — inject the `AgentQuery`,
  `SpawnProcess`, and `StartMcpBridge` seams as the existing tests do.
- `npm run typecheck` covers every package individually; `npm test` is vitest across `packages/*/test`.
- Loud failure over silent degradation. A transport that cannot honour something REFUSES rather than
  dropping it — the rule `AgentQueryOptions.model` and `disallowedTools` already state.
- The `Result` envelope never throws for a unit failure. Classify, don't flatten.
- New shapes carry the rationale in the doc comment, in the register of the surrounding code.

---

## Task 0 — verify the two boundary mappings (do this first)

`sdkQuery.ts` and `cliQuery.ts` both carry ⚠️ UNVERIFIED banners: the field names reflect documented API
and have never been checked against a live build. Everything below builds on those mappings, so confirm
them before widening anything.

- Install `@anthropic-ai/claude-agent-sdk` and check: the `query()` options names used, the `canUseTool`
  request/return shape, the terminal message discriminator, `result`/`text`, `total_cost_usd`,
  `createSdkMcpServer`/`tool`, and the `mcp__<server>__<tool>` convention.
- Check the CLI flags in `cliArgv` against the installed `claude --help`, and re-confirm
  `mcpProtocol.ts`'s pinned observations (required `updatedInput` on allow, required `message` on deny,
  required `type` on the `--mcp-config` entry) still hold on the current version.
- Record the version each was verified against, replace the ⚠️ banners with a "verified against X" note,
  and fix whatever diverged.

**Done when:** no ⚠️ UNVERIFIED banner remains in either package, and each names the version it was
checked against.

---

## Task 1 — `binaryPath` and `env` on the query seam

A live defect, not just a gap. `cliQuery` spawns bare `"claude"`; `sdkQuery` never sets
`pathToClaudeCodeExecutable`. On Windows the SDK spawns without a shell and without PATH/PATHEXT
resolution, so a bare command name fails as "native binary not found" and an npm `claude.cmd` shim fails
with `spawn EINVAL` (Node ≥ 20.12).

- Add `binaryPath?: string` and `env?: NodeJS.ProcessEnv` to `AgentQueryOptions`.
- `sdkQuery` passes `binaryPath` as `pathToClaudeCodeExecutable`; `cliQuery` uses it as the command.
- Add Windows resolution: resolve against PATH/PATHEXT, and when the result is a launcher script
  (`.cmd`/`.bat`/`.ps1`), follow it to the real package entry —
  `node_modules/@anthropic-ai/claude-code/bin/claude.exe`, falling back to `cli.js` for older versions.
  Warn and pass the original through when neither is found. Behind an injectable file-existence check so
  tests run against a fake filesystem.
- Plumb both from `AgentExecutorOptions` through `runAgent`.

Multi-account isolation (`CLAUDE_CONFIG_DIR`, continuation-group keys) is explicitly **out of scope** —
`env` just needs to exist and be forwarded.

**Done when:** a Windows-platform test with a faked `.cmd` shim resolves to the package entry, and the
resolved path reaches both transports.

---

## Task 2 — lossless output

Widen `AgentStreamMessage` to carry the SDK's structured messages, and fill the `LlmOutput` /
`LlmMetrics` fields that already exist:

| SDK source | Target |
| --- | --- |
| `assistant` content: thinking blocks | `LlmOutput.thinking` — `type: "reasoning"`, signature preserved in `providerMetadata` |
| `assistant` content: `tool_use` | `LlmOutput.toolCalls` |
| `user` messages carrying `tool_result` | `LlmOutput.toolResults` |
| `result.usage` / `modelUsage` | `TokenCounts` fields + `rawUsage` |
| `result.subtype` | real `finishReason` — stop hardcoding `"stop"` |
| the assistant/user turns themselves | `messages`, verbatim |

Delete the synthesized `messages: [{ role: "assistant", content: result.text }]` and its comment; the
agent does hand back its log, we were just discarding it.

Two further pieces:

- **Streaming.** Pass `includePartialMessages: true` and map `stream_event` text deltas onto the
  `ExecEvent.output_partial` variant that already exists. `DELEGATED_CAPS.streaming: true` is currently
  aspirational — this is what makes it true.
- **Passthrough.** The `system` subtypes (`init`, `compact_boundary`, `hook_started`, `task_progress`,
  `api_retry`, `rate_limit_event`, `permission_denied`, …) have no neutral home and should not get one.
  Add one `{ type: "provider_event"; payload: JsonValue }` variant to `ExecEvent` and forward them
  opaquely. Same precedent as `rawUsage` and `providerMetadata`: open by nature, JSON by construction.
  `exec` must not learn Claude vocabulary.

On the CLI path the same information arrives as `stream-json` lines — map both transports to the same
normalized shape.

**Done when:** a fake stream carrying thinking, tool calls, tool results, usage and deltas produces an
`LlmOutput` with every field populated, and `output_partial` events reach a `for await` over the handle.

---

## Task 3 — lossless input

`resolveConfig`, `PromptExecutorOptions.defaults`, and `op.config` already merge a partial
`LlmConfiguration` into `definition`. `runAgent` reads `model`, `providerSessionId`, and `messages` off
it and drops everything else.

- Forward the neutral fields the transports can honour: `reasoning`, `maxOutputTokens`, `stopSequences`,
  `tools`/`toolChoice`/`maxSteps`. Anything a transport cannot honour must **refuse**, not drop.
- Add `xhigh` to `ReasoningSpec.effort` — Claude Code has a level `low | medium | high` cannot name — and
  map `reasoning` onto the SDK's `effort`.
- Route transport-specific settings through `providerOptions`, which is already documented as "the full
  escape hatch… passed through to the provider verbatim" and is where Anthropic's `thinking` already
  rides:

  ```ts
  providerOptions: { claudeCode: { fastMode: true, ultracode: true, settingSources: ["user","project","local"], extraArgs: {…} } }
  ```

  Neutral core stays strict-parsed; nothing Claude-specific leaks into `LlmConfiguration`.

Note for whoever picks this up: `settingSources` and the `systemPrompt` preset are currently unset, so we
inherit whatever the SDK defaults to. Decide deliberately what a delegated agent should load, and write
the decision down.

**Done when:** a config fragment carrying reasoning, token limits and `providerOptions.claudeCode`
arrives at the query seam intact, and an unhonourable field produces a loud refusal with a test.

---

## Task 4 — control channel

`AgentQuery` is a one-shot: `(opts) => AsyncIterable<message>`, created and consumed inside the private
`runAgent`, so nothing outside can steer a live turn.

**The seam.** Return an interruptable iterable — an `AsyncIterable<AgentStreamMessage>` with optional
methods, so every existing `for await` keeps working unchanged:

```ts
interface AgentRun extends AsyncIterable<AgentStreamMessage> {
  interrupt?(): Promise<void>;
  send?(text: string): Promise<void>;
  setPermissionMode?(mode: AgentPermissionMode): Promise<void>;
  setModel?(model: string): Promise<void>;
}
```

The `sdkQuery` closure holds the query object and owns the `AsyncIterable<SDKUserMessage>` input queue —
no external lifetime management needed. Methods are **optional, never throwing stubs**: absent means
unsupported, `if (run.interrupt)` is the runtime check, and codex leaves them undefined (SIGINT to a
subprocess is a kill, not a graceful turn end). Mirror it with a `sessionSteering` capability so a caller
can decide before offering UI.

**Exposure.** The control surface has to reach a caller. Pick one:

- `ExecHandle.control?` — plumbed through `invoke` → `start`, forwarded by `wrapHandle` from the current
  inner handle along the path `cancel()` already takes via `ctl.started`. This settles the retry question
  for free (control targets the current attempt). Principled; touches the core contract and wrappers.
- A `ctx` sink (`onAgentStarted?: (run) => void`) called when the run is created. Zero change to
  `ExecHandle`; less principled; reversible.

Prefer the first unless it fights the wrapper generics — then take the second and leave a note.

**⚠️ Do not route `interrupt()` through the abort controller.** `handles.ts` deliberately unifies
`cancel()` and `ctx.abortSignal` as one event that settles the handle with a `canceled` failure.
Interrupt is a third thing: the turn ends, a `result` message still arrives, and the call **succeeds**.
Wiring it to abort throws away an answer the agent produced. Keep them separate and make `interrupt()`
idempotent, so a call racing the stream ending is a no-op rather than a second settle.

**Done when:** a fake run can be interrupted mid-stream and the handle settles with the agent's partial
answer as a SUCCESS, not a cancellation; and a transport without `interrupt` is visible as such before
the call rather than by throwing during it.

---

## Task 5 — the `injectTools` gaps

Two problems, both in `runAgent`'s tool split.

1. **Natives are not disabled.** `denied` is built only from policy `deny` entries, so with
   `injectTools: true` and no denies the agent has both its own `Read`/`Write`/`Bash` and
   `mcp__dai__read_file` — and its system prompt steers it hard toward the natives. If that is what a live
   run shows, injection is adding duplicates the model ignores rather than achieving the portable-vocabulary
   goal it exists for. **Verify against a live run first**, then, if confirmed, add the replaced natives to
   `disallowedTools` so injection actually displaces them.
2. **"Natives plus extras" is not expressible.** `injectTools: false` routes everything native, leaving no
   way to say "the agent's own tools for everything, plus these additional MCP tools" — which is what a
   host wanting to expose its own capability (a preview pane, a build runner) to an otherwise-stock agent
   needs. Separate "replace `ctx.tools`" from "add these tools" so both compose.

While here: confirm what an empty `allowedTools: []` means to the SDK. `cliArgv` omits the flag when
empty; `sdkQuery` passes `[]` through. If the SDK reads that as "pre-approve nothing" the two paths agree;
if it reads it as "allow nothing", they do not.

**Done when:** injection demonstrably displaces the natives it replaces, and a run can add MCP tools
without giving up the agent's built-ins.

---

## Task 6 — bound the event queue

`EventQueue.push` drops events when closed but buffers without limit when open and undrained. A long agent
turn with no attached consumer grows the buffer unboundedly. Add a bound with an explicit policy — drop
oldest and mark the gap, so a late consumer learns it missed events rather than silently receiving a
truncated stream.

Fan-out stays the consumer's job; the single-consumer contract and `SINGLE_CONSUMER_REASON` are correct
as they stand.

---

## Explicitly out of scope

- **Multi-account isolation** — `CLAUDE_CONFIG_DIR` per instance, continuation-group keys. Deferred.
- **Provider probe** — is the binary installed, what version, is it authenticated, which models and
  slash commands does it support. Wanted eventually; not part of this task.
- **Session handle caching.** Reusing a live query across turns instead of `resume`. Deferred until the
  above lands, and until the saving is measured (process spawn + agent init + bridge setup). When it is
  picked up: it belongs beside `ResidencyManager` as a `withAgentSession` wrapper, not inside
  `AgentExecutor` — and the cache key is the full frozen config, not the session id, because `ctx.tools`,
  the policy baseline, and the permission mode are all baked in at creation. The captured-`ctx` hazard
  (`canUseTool` and every injected tool close over one call's `approve`, scope, and signal) needs a
  mutable current-call slot before any of it is safe.
