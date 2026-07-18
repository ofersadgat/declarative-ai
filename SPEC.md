# Hierarchical Workflow Specification

> **Provenance and scope.** This file is the canonical specification of the
> hierarchical-workflow formalism implemented by `@ai-exec/hw`, migrated from
> `JaiRA/SPEC.md` (which remains the product spec for the JaiRA app).
> Normative for this library: §2.3–§2.4 (states, state IDs), §3 (state machine
> semantics), §4 (inputs/outputs/params/artifacts/conversations), §5 (state
> file format), §6 (transition expressions), §7.3 and §9 (worked examples,
> used as golden tests), §8 (UI states, realized here via the
> `InteractionPort`), §10.1–§10.4 (statuses, run records, durability, async
> children), §12 (versioning → snapshot hashing). Sections about tasks,
> boards, Git isolation, safety policy, and MVP scope describe the JaiRA
> product and are context for the library, not requirements on it.

## 1. Purpose

JaiRA is a local-first, single-user project management and agent orchestration app.
It combines a Jira-style Kanban interface with hierarchical, declarative state
machines that can run local AI coding agents, collect human input, produce
artifacts, and move tasks through project-defined workflows.

The central design goal is to make AI-assisted work observable, resumable,
auditable, and configurable without letting agents directly own workflow control.

## 2. Core Concepts

### 2.1 Project

A project is the root unit of configuration and execution. It contains:

- Workflow state files.
- Tasks.
- Artifact references.
- Conversation artifacts.
- Execution history.
- Runtime configuration for local agents.
- Safety policy.

For the MVP, a JaiRA project maps to one local project directory. Workflow
state files live in a reserved, engine-owned directory (`.jaira/workflows/`)
inside the project. Agents have no read or write access to `.jaira/`.

### 2.2 Task

A task is a unit of work shown on a board. A task has exactly one active state at
a time within its current workflow level.

A task may:

- Move through workflow states.
- Spawn subtasks.
- Enter child workflows.
- Produce artifacts.
- Accumulate conversation history.
- Wait for user input.
- Wait for external events.

Subtasks are separate tasks. Substates are child states inside a workflow tree.
These are distinct concepts.

### 2.3 Workflow State

A workflow state is defined by one state file. State files form a tree.

Each state may define:

- Inputs.
- Outputs.
- Parameters.
- Child states.
- Default child order.
- Agent execution behavior.
- UI behavior.
- Transition rules.
- Iteration limits.
- Safety requirements.

A state only knows about its own declared children. It does not know siblings,
parents, or arbitrary external states.

### 2.4 State ID

A state ID is the state file path relative to the workflow root, without the file
suffix.

Examples:

```text
feature
feature/plan
feature/plan/critique
feature/plan/critique/address_weaknesses
```

### 2.5 Board

A board is a visual projection of the active states of tasks.

At any level, columns correspond to visible child states. Double-clicking a task
that is inside a state with a child workflow opens the sub-board for that state.

Example:

```text
feature
  plan
    goals
    context
    critique
  design
  implement
  review
```

At the feature board level, the task appears in `plan`, `design`, `implement`,
or `review`.

When the task is in `plan`, double-clicking opens the planning sub-board with
columns such as `goals`, `context`, and `critique`.

## 3. Hierarchical State Machine Semantics

### 3.1 Tree Structure

The workflow definition is a tree. Runtime movement is controlled by transitions
within each node's local scope.

A state can transition to:

- One of its own child states.
- A typed termination outcome.

A state cannot transition directly to a sibling, parent, or ancestor. If a child
needs to influence its parent, it terminates with structured outputs. The parent
then decides what to do.

### 3.2 Operations

A state does its work by running operations. Operation kinds:

- `ui`: a built-in UI component that collects structured data from the user.
- `agent`: a local agent run driven by the state's prompt.
- `skill`: a named reusable skill invoked through the agent runtime.
- Child states, entered by sequence order or by explicit transition.

A state may declare any combination of these. Operations run one at a time in a
fixed priority order: `ui`, then `agent`, then `skill`, then child states in
`sequence` order. Async child states are the only exception to one-at-a-time
execution.

### 3.3 Evaluation Loop

1. Entering a state creates a new state instance. Declared inputs are resolved
   and validated; validation failure blocks the state.
2. The engine runs the highest-priority operation that has not yet run in this
   instance.
3. When an operation completes, its outputs are validated and the state's
   transitions are evaluated in declared order. The first transition whose
   `when` expression is true is taken.
4. A taken transition either enters a child state or terminates the state.
5. If no transition matches, the engine runs the next operation in priority
   order.
6. If no operations remain and no transition matches: if any children are
   still running, the state waits, and each child completion triggers another
   round of transition evaluation. Only when all children have finished and no
   transition applies does the state terminate with `terminate.success`.

If an operation fails unrecoverably, or a child terminates with `error` or
`timeout` and no transition handles it, the state terminates with
`terminate.error` instead of continuing.

Transition order matters: child-completion conditions should be declared before
child-entry conditions, so that the evaluation that runs after a child
completes does not immediately re-enter it.

Taking a transition to a child that appears in the `sequence` resets the
sequence cursor to that child and clears the recorded results of that child and
every later child in the sequence; default ordering then resumes from there.
For example, with sequence `[a, b, c, d]`, a transition from `d` back to `b`
clears `b`, `c`, and `d`; only `a` retains its results. Children outside the
sequence keep their results.

Starting an async child does not trigger transition evaluation; evaluation runs
when the child completes. A transition whose `when` expression references
outputs of a child that has started but not yet finished is skipped for that
evaluation round and becomes eligible again when the child resolves
(Section 10.4).

### 3.4 State Instances

Entering a state — including re-entering it via a transition — creates a fresh
instance. Outputs and child results belong to the instance. References such as
`children.<id>.outputs` resolve to the most recent instance of that child
within the current parent instance, and evaluate to `undefined` if the child
has not run.

`run.iteration` is the count of transitions taken so far within the current
state instance, starting at 0. It is the standard guard for cycles.

### 3.5 Active Path

A task has an active path through the state tree.

Example:

```text
feature/plan/critique
```

When `critique` terminates, control returns to the active `plan` instance,
which validates the child's outputs and continues its own evaluation loop.

### 3.6 Termination

Termination is a typed return from a child state to its parent. The termination
outcome describes how the state finished, not what it decided:

```text
terminate.success
terminate.error
terminate.canceled
terminate.timeout
```

Domain-level results such as "approved" or "needs changes" are ordinary
schema-validated outputs. Parents branch on `children.<id>.outputs.*` for
decisions and on `children.<id>.outcome` for failure handling.

A child termination includes:

- A termination outcome.
- Validated outputs.
- Produced artifacts.
- Execution metadata.

Output validation failure or an unrecoverable operation failure terminates the
state with `terminate.error`.

### 3.7 Parent-Owned Control

Parents own:

- Child ordering.
- Child input wiring.
- Child result handling.
- Output remapping.
- Parent-level transitions.

Children do not directly mutate parent state.

A parent's declared outputs are resolved when the parent terminates.

## 4. Inputs, Outputs, Parameters, and Artifacts

### 4.1 Required By Default

Inputs and outputs are required by default.

A field is optional only when:

- It declares `optional: true`.
- It is an input with a `default` value.

Schema validation failure blocks the state.

### 4.2 Inputs

Inputs are named values available to a state. Inputs may come from:

- Task fields.
- Parent-provided values.
- Previous child outputs.
- Artifacts.
- Conversation artifacts.
- Literal values.
- Parameters.

A child state receives only its declared inputs. It does not know where those
inputs came from.

Good:

```json
{
  "critique": {
    "state": "feature/plan/critique",
    "inputs": {
      "plan_doc": "children.context.outputs.plan_doc",
      "goals": "children.goals.outputs.goals"
    }
  }
}
```

Bad:

```text
The critique state directly reads ../context.outputs.plan_doc.
```

Wiring values are expressions evaluated against the parent's context. A bare
string is a reference; literal values must be wrapped:

```json
{ "severity_threshold": { "value": "significant" } }
```

### 4.3 Outputs

Outputs are named values produced by a state. An output may declare a `from`
expression, evaluated against the state's context when the state terminates.

Outputs should be schema-validated when they are used for:

- Transitions.
- UI decisions.
- Parent control flow.
- Automation.

### 4.4 Generic Passthrough Outputs

Some wrapper or delegation states need to return whatever a child state produces.
This is supported explicitly with passthrough outputs.

Example:

```json
{
  "outputs": {
    "child_outputs": {
      "type": "passthrough",
      "from": "children.critique.outputs"
    }
  }
}
```

Rule:

```text
Generic outputs may be stored or passed upward. Transition-relevant outputs
should be explicitly typed.
```

### 4.5 Parameters

Parameters configure reusable states.

Example:

```json
{
  "params": {
    "severity_threshold": {
      "type": "string",
      "enum": ["minor", "significant", "critical"],
      "default": "significant"
    },
    "max_findings": {
      "type": "number",
      "default": 10
    }
  }
}
```

Parameters can be optional when they have defaults.

### 4.6 Artifacts

Artifacts are durable work products. Examples:

- Markdown plans.
- Design documents.
- Patches.
- Test reports.
- Review notes.
- Conversation summaries.
- Full conversation logs.

Git is responsible for artifact versioning. JaiRA records artifact references
and workflow metadata, but artifact history is delegated to the project Git
repository.

### 4.7 Conversations

Conversations are artifacts. A state may choose how to use prior conversation
context.

Supported conversation modes:

```text
full_history
summary
fresh
selected_artifacts
```

The initial default is `full_history`, but the schema must support all modes so
projects can move toward more controlled context selection over time.

## 5. State File Format

State files are declarative JSON for the MVP.

The implementation may later support JSON5, YAML, or editor-friendly syntaxes,
but the MVP should define canonical JSON semantics first.

### 5.1 Top-Level Fields

```text
id
label
description
params
inputs
outputs
agent
skill
ui
children
sequence
transitions
limits
```

### 5.2 Field Summary

`id`
: State ID, equal to relative path without suffix.

`label`
: Human-readable state name.

`description`
: Short explanation of the state purpose.

`params`
: Reusable state configuration.

`inputs`
: Schema for values the state receives.

`outputs`
: Schema for values the state emits.

`agent`
: Local AI agent execution configuration.

`skill`
: Named reusable skill invocation, executed through the agent runtime.

`ui`
: Built-in user interface component configuration.

`children`
: Declared child states, their input wiring, and async flags.

`sequence`
: Default order for child execution.

`transitions`
: Local transition rules.

`limits`
: Iteration and timeout limits.

## 6. Transition Expressions

Expressions use a small language with JavaScript evaluation semantics:
equality, comparison, and truthiness behave exactly as in JavaScript. One
deviation: property access on `undefined` or missing values yields `undefined`
instead of throwing (implicit optional chaining), so `children.x.outputs.y` is
safely `undefined` when `x` has never started.

References to a child that has started but not yet finished are pending, not
`undefined`: a transition using a pending reference is skipped for that
evaluation round, and input wiring using one waits for it to resolve
(Section 10.4).

The same language is used for transition conditions, input wiring references,
and output `from` expressions.

The expression language should support:

- Literals.
- Identifiers.
- Property access.
- Parentheses.
- Unary `!`.
- Binary comparison operators.
- Boolean operators.
- Conditional (ternary) expressions.
- Numeric comparisons.
- String comparisons.
- `.length` for arrays and strings.

Supported operators:

```text
==
!=
===
!==
<
<=
>
>=
&&
||
!
?:
```

The expression language must be pure. It must not support:

- Arbitrary JavaScript execution.
- Function calls.
- Imports.
- Mutation.
- Loops.
- Filesystem access.
- Network access.
- Async execution.

### 6.1 Expression Context

Expressions may read from a controlled context.

Suggested namespaces:

```text
inputs.*
outputs.*
params.*
ui.*
children.<id>.outputs.*
children.<id>.outcome
run.*
limits.*
artifacts.*
conversations.*
```

## 7. Agent Execution

### 7.1 Local Agents

The MVP supports local code agents only.

Example providers:

- `claude_code`
- `codex`
- `opencode`
- Project-specific local command adapters.

Agent providers should be modeled as runtime adapters with capabilities, not as
interchangeable strings.

### 7.2 Agent Responsibilities

Agents may:

- Inspect the project.
- Modify files within the project directory.
- Run allowed commands.
- Produce artifacts.
- Produce structured outputs matching the state schema.
- Propose transitions when allowed by the state.

Agents may not:

- Bypass human approval gates.
- Mutate workflow state directly.
- Manipulate Git history.
- Access files outside the project directory.
- Execute blocked commands.

### 7.3 Agent State Example

```json
{
  "id": "feature/plan/critique",
  "label": "Critique Plan",
  "description": "Review the current plan for significant weaknesses.",
  "inputs": {
    "plan_doc": {
      "type": "artifact",
      "format": "markdown"
    },
    "severity_threshold": {
      "type": "string",
      "enum": ["minor", "significant", "critical"],
      "default": "significant"
    }
  },
  "outputs": {
    "outcome": {
      "type": "string",
      "enum": ["clean", "needs_changes", "blocked"]
    },
    "weaknesses": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "critique_report": {
      "type": "artifact",
      "format": "markdown"
    },
    "human_decision": {
      "type": "string",
      "enum": ["approve", "request_changes", "block"],
      "optional": true,
      "from": "children.human_review.outputs.decision"
    }
  },
  "agent": {
    "provider": "claude_code",
    "conversation": {
      "mode": "full_history"
    },
    "prompt": {
      "template": "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema."
    }
  },
  "children": {
    "address_weaknesses": {
      "state": "feature/plan/critique/address_weaknesses",
      "inputs": {
        "plan_doc": "inputs.plan_doc",
        "weaknesses": "outputs.weaknesses",
        "critique_report": "outputs.critique_report"
      }
    },
    "human_review": {
      "state": "feature/plan/critique/human_review",
      "inputs": {
        "plan_doc": "inputs.plan_doc",
        "critique_report": "outputs.critique_report"
      }
    }
  },
  "transitions": [
    {
      "to": "terminate.success",
      "when": "children.human_review.outcome === 'success'"
    },
    {
      "to": "terminate.success",
      "when": "children.address_weaknesses.outcome === 'success'"
    },
    {
      "to": "terminate.success",
      "when": "outputs.outcome === 'clean'"
    },
    {
      "to": "human_review",
      "when": "outputs.outcome === 'blocked'"
    },
    {
      "to": "address_weaknesses",
      "when": "outputs.outcome === 'needs_changes'"
    }
  ]
}
```

Execution walk-through: the agent operation runs first. `clean` terminates
immediately, before any child runs. `needs_changes` runs one fix pass and then
terminates so the parent can decide whether to re-plan. `blocked` collects a
human decision, surfaced through the `human_decision` output. The
child-completion transitions are declared first so the evaluation that runs
after a child completes does not re-enter it. The retry loop lives in the
parent (Section 9), which re-runs the whole planning pass and gets a fresh
critique instance each time.

## 8. UI States

Human interaction is modeled as UI output, not as a special human runtime.

A state may declare a built-in UI component. The component displays state inputs
and returns structured data. Transition logic remains in the state file.

### 8.1 MVP Built-In Components

Suggested MVP components:

- `choose_option`
- `review_artifact`
- `edit_markdown`
- `fill_form`
- `confirm_action`

### 8.2 UI State Example

```json
{
  "id": "feature/plan/critique/human_review",
  "label": "Human Review",
  "inputs": {
    "plan_doc": {
      "type": "artifact",
      "format": "markdown"
    },
    "critique_report": {
      "type": "artifact",
      "format": "markdown"
    }
  },
  "outputs": {
    "decision": {
      "type": "string",
      "enum": ["approve", "request_changes", "block"]
    },
    "comments": {
      "type": "string",
      "format": "markdown",
      "optional": true
    }
  },
  "ui": {
    "component": "choose_option",
    "prompt": "Review the critique result.",
    "options": ["approve", "request_changes", "block"]
  }
}
```

This state declares no transitions: once the component completes, no
operations remain, so the state terminates with `terminate.success` and its
validated outputs. The parent branches on `outputs.decision`.

## 9. Parent State Example

```json
{
  "id": "feature/plan",
  "label": "Planning",
  "inputs": {
    "issue": {
      "type": "artifact",
      "format": "markdown"
    }
  },
  "outputs": {
    "outcome": {
      "type": "string",
      "enum": ["complete", "blocked"],
      "from": "children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'"
    },
    "plan_doc": {
      "type": "artifact",
      "format": "markdown",
      "from": "children.context.outputs.plan_doc"
    },
    "critique": {
      "type": "passthrough",
      "from": "children.critique.outputs"
    }
  },
  "children": {
    "goals": {
      "state": "feature/plan/goals",
      "inputs": {
        "issue": "inputs.issue"
      }
    },
    "context": {
      "state": "feature/plan/context",
      "inputs": {
        "issue": "inputs.issue",
        "goals": "children.goals.outputs.goals"
      }
    },
    "critique": {
      "state": "feature/plan/critique",
      "inputs": {
        "plan_doc": "children.context.outputs.plan_doc",
        "severity_threshold": {
          "value": "significant"
        }
      }
    }
  },
  "sequence": ["goals", "context", "critique"],
  "transitions": [
    {
      "to": "terminate.success",
      "when": "children.critique.outputs.outcome === 'clean'"
    },
    {
      "to": "goals",
      "when": "children.critique.outputs.outcome === 'needs_changes' && run.iteration < limits.max_iterations"
    },
    {
      "to": "terminate.success",
      "when": "children.critique.outcome === 'success'"
    }
  ],
  "limits": {
    "max_iterations": 3
  }
}
```

The re-plan loop lives here: `needs_changes` transitions back to `goals`,
which resets the sequence and clears the recorded results of `goals`,
`context`, and `critique`, so the next pass runs fresh instances (and the
first two transitions evaluate to false until `critique` runs again). When the
iteration limit is reached, or critique reports `blocked`, the final
transition fires and the `outcome` output resolves to `blocked`.

## 10. Execution Lifecycle

### 10.1 State Run Status

A state run may be in one of these statuses:

```text
queued
running
waiting_for_user
waiting_for_event
sleeping
blocked
failed
completed
canceled
```

### 10.2 State Run Record

Each state run records:

- Task ID.
- State ID.
- Workflow version.
- State file content hash.
- Input artifact references.
- Output artifact references.
- Conversation references.
- Agent provider and configuration.
- Commands requested.
- Commands executed.
- Commands blocked.
- UI data returned.
- Transition decisions.
- Validation errors.
- Start time.
- End time.
- Token or cost metadata when available.

### 10.3 Durable Execution

The engine must persist enough information to resume after app restart.

At minimum, the persisted data must answer:

- Which task was active?
- Which state path was active?
- Was an agent running?
- Was user input pending?
- Which artifacts had been produced?
- Did a transition already occur?
- Which workflow version was in use?

### 10.4 Async Child States

A child entry may declare `"async": true`. When the sequence cursor reaches an
async child, or a transition targets it, the child starts and the engine
immediately continues with the next operation instead of waiting. Multiple
async children may run concurrently.

Waiting is dataflow-driven; no explicit join construct is needed:

- A child whose input wiring references outputs of an unresolved async child
  does not start until those outputs resolve.
- A transition whose `when` expression references outputs of an unresolved
  async child is skipped until that child resolves.
- Each child completion triggers another round of transition evaluation.
- If no transitions apply and no children are left to run, still-running async
  children are waited for. The state terminates only when all children have
  finished and no transition applies.
- Terminating a state cancels any still-running descendant states.

Example — fan-out reviews with a dataflow join:

```json
{
  "children": {
    "claude_review": {
      "state": "review/agent_review",
      "async": true
    },
    "codex_review": {
      "state": "review/agent_review",
      "async": true
    },
    "synthesize": {
      "state": "review/synthesize",
      "inputs": {
        "review_a": "children.claude_review.outputs.report",
        "review_b": "children.codex_review.outputs.report"
      }
    }
  },
  "sequence": ["claude_review", "codex_review", "synthesize"]
}
```

Both reviews start without blocking. The sequence cursor reaches `synthesize`
immediately, but its inputs reference both review outputs, so it waits for
both to resolve before starting. With no transitions declared, the state
terminates with `terminate.success` once all three children finish.

### 10.5 Branch Isolation and Concurrent Tasks

Tasks may run concurrently. Isolation is Git-based:

- A task may be bound to a Git branch. The engine materializes concurrently
  active branches as separate worktrees, so concurrent tasks never share a
  working tree.
- A task may spawn subtasks that share its branch, for collaborative patterns
  such as multiple agents reviewing the same change. Coordinating writes
  within a shared branch is the workflow author's responsibility in the MVP.
- Async children of a single state run in the task's own branch context.

## 11. Safety Policy

### 11.1 Required MVP Restrictions

Agents are limited to their task's worktree of the project directory.

Agents have no read or write access to the engine-owned `.jaira/` directory,
including workflow state files and run records.

Agents cannot bypass human approval gates.

Agents cannot destroy Git history.

Agents cannot execute denied commands.

### 11.2 Git Operations

Agents may use Git constructively: commit, create branches, switch branches,
and inspect history.

The policy layer blocks operations that destroy history or discard work,
equivalent to:

```text
git push --force
git push --mirror
git reset --hard
git rebase
git filter-branch
git filter-repo
git gc / git prune
git reflog expire / git reflog delete
rm -rf .git
```

The actual implementation should block by parsed command intent, not only by raw
string matching.

### 11.3 Commands Requiring Approval

The MVP should require explicit user approval for:

- Pushes.
- Merges.
- Deployments.
- Package publishing.
- Commands that access network resources.
- Commands that modify global configuration.
- Commands that access secrets.
- Commands that install or execute remote scripts.

### 11.4 Approval Gates

Approval gates are ordinary UI states; there is no separate policy mechanism.
The engine guarantees that a UI component's outputs can only be produced by a
real user interaction in the app, and agents have no channel to write another
state's outputs. A transition guarded on a UI state's outputs is therefore
user-controlled by construction.

An agent may include a recommendation in its artifacts or outputs, but a
recommendation never satisfies a gate.

## 12. Workflow Versioning

Each task run pins the workflow version it started with.

The MVP can represent a workflow version as:

- The Git commit hash when available.
- The hash of all referenced state files.
- The project-local workflow version ID.

If state files change while a task is active, the task continues using its pinned
version unless the user explicitly migrates it. The engine executes a task from
its pinned snapshot of the state files, not from the current on-disk copies.

## 13. History and Pruning

JaiRA stores state run history by default.

Stored history includes:

- State runs.
- Transitions.
- Agent outputs.
- UI outputs.
- Artifact references.
- Conversation references.

Users may prune historical records.

Pruning should preserve task correctness. A task cannot prune data required to
resume its current active state.

## 14. MVP Scope

### 14.1 Included

- Single local project.
- Single user.
- One file per state.
- Declarative JSON state files.
- Hierarchical state execution.
- Default child ordering.
- Local child-only transitions.
- Async child states.
- Typed termination outcomes.
- Schema-validated inputs and outputs.
- Generic passthrough outputs.
- Built-in UI components.
- Local agent runtime abstraction.
- Conversation artifacts.
- Artifact references.
- Durable state run records.
- Kanban board and sub-board navigation.
- Branch-per-task isolation via Git worktrees.
- Safety policy for local agents.
- Blocking on schema validation failure.

### 14.2 Deferred

- Multi-user collaboration.
- Remote hosted agents.
- GitHub or GitLab integration.
- Deployment automation.
- Arbitrary custom UI components.
- Arbitrary webhooks.
- General DAG execution beyond async child states.
- Cross-project workflow libraries.
- Rich workflow migration tooling.

## 15. Open Design Questions

These questions are not required for the first implementation, but should be
resolved before expanding beyond the MVP:

1. Should artifacts be stored in a reserved JaiRA directory, or can users choose
   project-specific artifact paths?
2. Should the state file extension be `.json`, `.state.json`, or something else?
3. Should state files be allowed to omit `id` and derive it from the file path?
4. What is the minimum built-in UI component set for a useful MVP?
5. Which local agent provider should be implemented first?
6. How should agent adapters report partial progress?
7. Should workflow validation happen continuously, on save, or only when a task
   starts?
8. What is the first task database format: SQLite, plain files, or embedded app
   storage?
9. How should subtasks relate to parent task completion?
10. Should spawned subtasks block parent completion by default?
11. How are skills defined and registered (project-level skill library
    format), and what parameters does a skill invocation take?
12. How does a task get bound to a branch: at creation, by a state, or by the
    user?

## 16. Design Principles

1. State files are declarative.
2. States only know their own children.
3. Children return structured outputs; parents decide what those outputs mean.
4. UI components produce data; state files own transition logic.
5. Agents produce artifacts and proposals; the engine owns workflow control.
6. Transition expressions are pure and limited.
7. Schema validation is mandatory for state outputs.
8. Generic passthrough is explicit.
9. Cycles require iteration limits.
10. Workflow versions are pinned per task run.
11. Human approval gates cannot be bypassed.
12. Agent filesystem access is scoped to the task's worktree.
13. Destructive Git operations are blocked; constructive Git use is allowed.
14. Execution history is stored by default and can be pruned safely.
15. Operations run in fixed priority order: UI, agent, skill, children.
16. Entering a state creates a fresh instance; results never leak across
    instances.
17. Workflow definitions are engine-owned; agents cannot read or modify them.
