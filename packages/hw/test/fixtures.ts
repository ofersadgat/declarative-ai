/**
 * Shared fixtures: the SPEC's worked examples (§7.3 critique, §8.2 human review, §9 planning parent)
 * plus the minimal supporting states they reference. Used by loader/validator tests and as the engine
 * golden tests' workflow.
 *
 * Written in the post-ops-redesign format (DESIGN §3.1): one `operation` per state, slots
 * carrying JSON Schemas, and wiring as authored BINDING SUGAR (`{ child, output }`, `{ input }`,
 * `{ expr }`) that the loader lowers to base `Ref` cases.
 */
import type { StateDef } from "../src/format.js";

// Slot builders, not shared constants: each call returns a FRESH object, so a test that mutates one
// state's slot (to provoke a validation error) can't leak that mutation into every other fixture.
/** An artifact-typed slot (SPEC §4.6) — a durable work product carried inline. */
/** An ARTIFACT slot is a BLOB-kind slot (DESIGN §3.7): the bespoke `x-artifact: true` marker
 *  is gone, and the kind is derived from JSON Schema's own `contentMediaType` (DESIGN §3.7). */
const artifact = (format: string) => ({ kind: "blob", schema: { type: "string", contentMediaType: format } }) as const;
const str = () => ({ schema: { type: "string" } }) as const;
const strArray = () => ({ schema: { type: "array", items: { type: "string" } } }) as const;
/** What a CALL returns where the state stores it as an artifact: a string, no `kind`. The kind is
 *  the SLOT's business — `.operation.output` says what came back, the state says what it is. */
const markdownText = (format: string) => ({ schema: { type: "string", contentMediaType: format } }) as const;

export const PLAN_ID = "feature/plan";

export function specPlanningFiles(): Record<string, StateDef> {
  return {
    "feature/plan": {
      label: "Planning",
      // SPEC §4.7's "threads across states" is DECLARED now, not implicit. An undeclared operation
      // gets its own stream (DESIGN.md §1.6) — an implicit process-wide transcript is what drove
      // unbounded context growth — so a workflow whose states are meant to share a conversation says
      // so once, at the root, and the environment chain carries it down.
      environment: { session: "planning" },
      inputs: { issue: artifact("markdown") },
      outputs: {
        outcome: {
          schema: { type: "string", enum: ["complete", "blocked"] },
          binding: { expr: ".children.critique.output.outcome === 'clean' ? 'complete' : 'blocked'" },
        },
        plan_doc: { ...artifact("markdown"), binding: ".children.context.output.plan_doc" },
        // A "passthrough" output: the whole child result as ONE value. `*` is required now that a
        // bare `{ child }` selects the slot's own name instead.
        critique: { binding: ".children.critique.output" },
      },
      children: {
        goals: { state: "feature/plan/goals", inputs: { issue: ".inputs.issue" } },
        context: {
          state: "feature/plan/context",
          inputs: { issue: ".inputs.issue", goals: ".children.goals.output.goals" },
        },
        critique: {
          state: "feature/plan/critique",
          inputs: {
            plan_doc: ".children.context.output.plan_doc",
            severity_threshold: { text: "significant" },
          },
          // Every rule here is about `critique`, so all three live on its mount (SPEC §3.3) and this
          // state declares no `transitions` of its own. On the mount they are not merely false until
          // `critique` runs again after a re-plan — they are not evaluated until it finishes.
          transitions: [
            { to: "terminate.success", when: ".children.critique.output.outcome === 'clean'" },
            {
              to: "goals",
              when: ".children.critique.output.outcome === 'needs_changes' && .run.iteration < .limits.max_iterations",
            },
            { to: "terminate.success", when: ".children.critique.outcome === 'success'" },
          ],
        },
      },
      sequence: ["goals", "context", "critique"],
      limits: { max_iterations: 3 },
    },
    "feature/plan/goals": {
      label: "Goals",
      inputs: { issue: artifact("markdown") },
      outputs: { goals: { ...strArray(), binding: ".operation.output.goals" } },
      operation: {
        kind: "prompt",
        prompt: "Extract goals from {{.inputs.issue}}.",
        model: "planner",
        output: { goals: strArray() },
      },
    },
    "feature/plan/context": {
      label: "Context",
      inputs: { issue: artifact("markdown"), goals: strArray() },
      outputs: { plan_doc: { ...artifact("markdown"), binding: ".operation.output.plan_doc" } },
      operation: {
        kind: "prompt",
        prompt: "Write the plan for {{.inputs.issue}}.",
        model: "planner",
        output: { plan_doc: markdownText("markdown") },
      },
    },
    "feature/plan/critique": {
      label: "Critique Plan",
      description: "Review the current plan for significant weaknesses.",
      inputs: {
        plan_doc: artifact("markdown"),
        severity_threshold: { schema: { type: "string", enum: ["minor", "significant", "critical"] }, default: "significant" },
      },
      outputs: {
        outcome: {
          schema: { type: "string", enum: ["clean", "needs_changes", "blocked"] },
          binding: ".operation.output.outcome",
        },
        weaknesses: { ...strArray(), binding: ".operation.output.weaknesses" },
        critique_report: { ...artifact("markdown"), binding: ".operation.output.critique_report" },
        human_decision: {
          schema: { type: "string", enum: ["approve", "request_changes", "block"] },
          optional: true,
          binding: ".children.human_review.output.decision",
        },
      },
      environment: { conversation: { mode: "full_history" } },
      operation: {
        kind: "prompt",
        model: "critic",
        prompt: "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema.",
        // What the CALL returns. `human_decision` is not here: it comes from a child when the state
        // terminates, so asking the model for it would be a contract the model cannot meet.
        output: {
          outcome: { schema: { type: "string", enum: ["clean", "needs_changes", "blocked"] } },
          weaknesses: strArray(),
          critique_report: markdownText("markdown"),
        },
      },
      children: {
        address_weaknesses: {
          state: "feature/plan/critique/address_weaknesses",
          inputs: {
            plan_doc: ".inputs.plan_doc",
            weaknesses: { expr: ".operation.output.weaknesses" },
            critique_report: { expr: ".operation.output.critique_report" },
          },
          transitions: [{ to: "terminate.success", when: ".children.address_weaknesses.outcome === 'success'" }],
        },
        human_review: {
          state: "feature/plan/critique/human_review",
          inputs: { plan_doc: ".inputs.plan_doc", critique_report: { expr: ".operation.output.critique_report" } },
          transitions: [{ to: "terminate.success", when: ".children.human_review.outcome === 'success'" }],
        },
      },
      // What is left is what the state's OWN operation decides — terminate on `clean`, enter one child
      // or the other otherwise. The two child-completion rules moved to the mounts they are about, so
      // nothing depends any more on their being declared ahead of the entry rules (SPEC §3.3).
      transitions: [
        { to: "terminate.success", when: ".operation.output.outcome === 'clean'" },
        { to: "human_review", when: ".operation.output.outcome === 'blocked'" },
        { to: "address_weaknesses", when: ".operation.output.outcome === 'needs_changes'" },
      ],
    },
    "feature/plan/critique/address_weaknesses": {
      label: "Address Weaknesses",
      inputs: { plan_doc: artifact("markdown"), weaknesses: strArray(), critique_report: artifact("markdown") },
      outputs: { resolution: { ...str(), binding: ".operation.output.resolution" } },
      operation: {
        kind: "prompt",
        prompt: "Fix the listed weaknesses.",
        model: "fixer",
        output: { resolution: str() },
      },
    },
    "feature/plan/critique/human_review": {
      label: "Human Review",
      inputs: { plan_doc: artifact("markdown"), critique_report: artifact("markdown") },
      outputs: {
        decision: {
          schema: { type: "string", enum: ["approve", "request_changes", "block"] },
          binding: ".operation.output.decision",
        },
        comments: {
          schema: { type: "string", format: "markdown" },
          optional: true,
          binding: ".operation.output.comments",
        },
      },
      // An interactive host function — a plain FunctionOp like any other (§3), with its authored
      // surface bound to named input slots via `args`.
      operation: {
        kind: "function",
        function: "choose_option",
        args: { prompt: "Review the critique result.", options: ["approve", "request_changes", "block"] },
        output: {
          decision: { schema: { type: "string", enum: ["approve", "request_changes", "block"] } },
          comments: { schema: { type: "string", format: "markdown" }, optional: true },
        },
      },
    },
  };
}

/** SPEC §10.4 — fan-out reviews with a dataflow join. */
export const FANOUT_ID = "review";

export function specFanoutFiles(): Record<string, StateDef> {
  return {
    review: {
      label: "Fan-out Review",
      inputs: { change: str() },
      outputs: { summary: { ...str(), binding: ".children.synthesize.output.summary" } },
      children: {
        claude_review: { state: "review/agent_review", async: true, inputs: { change: ".inputs.change" } },
        codex_review: { state: "review/agent_review", async: true, inputs: { change: ".inputs.change" } },
        synthesize: {
          state: "review/synthesize",
          inputs: {
            review_a: ".children.claude_review.output.report",
            review_b: ".children.codex_review.output.report",
          },
        },
      },
      sequence: ["claude_review", "codex_review", "synthesize"],
    },
    "review/agent_review": {
      label: "Agent Review",
      inputs: { change: str() },
      outputs: { report: str() },
      operation: { kind: "prompt", prompt: "Review {{.inputs.change}}.", model: "reviewer" },
    },
    "review/synthesize": {
      label: "Synthesize",
      inputs: { review_a: str(), review_b: str() },
      outputs: { summary: str() },
      operation: {
        kind: "prompt",
        prompt: "Combine {{.inputs.review_a}} and {{.inputs.review_b}}.",
        model: "synthesizer",
      },
    },
  };
}
