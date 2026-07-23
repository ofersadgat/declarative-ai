/**
 * Shared fixtures: the SPEC's worked examples (§7.3 critique, §8.2 human review, §9 planning parent)
 * plus the minimal supporting states they reference. Used by loader/validator tests and as the engine
 * golden tests' workflow.
 *
 * Written in the post-ops-redesign format (DESIGN §3.1): one `operation` per state, slots
 * carrying JSON Schemas, and wiring as authored BINDING SUGAR (`{ child, output }`, `{ input }`,
 * `{ expr }`) that the loader lowers to base `Ref` cases.
 */
import type { StateDef } from "../src/format";

// Slot builders, not shared constants: each call returns a FRESH object, so a test that mutates one
// state's slot (to provoke a validation error) can't leak that mutation into every other fixture.
/** An artifact-typed slot (SPEC §4.6) — a durable work product carried inline. */
/** An ARTIFACT slot is a BLOB-kind slot (DESIGN §3.7): the bespoke `x-artifact: true` marker
 *  is gone, and the kind is derived from JSON Schema's own `contentMediaType` (DESIGN §3.7). */
const artifact = (format: string) => ({ kind: "blob", schema: { type: "string", contentMediaType: format } }) as const;
const str = () => ({ schema: { type: "string" } }) as const;
const strArray = () => ({ schema: { type: "array", items: { type: "string" } } }) as const;

export const PLAN_ID = "feature/plan";

export function specPlanningFiles(): Record<string, StateDef> {
  return {
    "feature/plan": {
      label: "Planning",
      inputs: { issue: artifact("markdown") },
      outputs: {
        outcome: {
          schema: { type: "string", enum: ["complete", "blocked"] },
          binding: { expr: "children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'" },
        },
        plan_doc: { ...artifact("markdown"), binding: { child: "context", output: "plan_doc" } },
        // A "passthrough" output is just an unconstrained slot bound to a producer.
        critique: { binding: { child: "critique" } },
      },
      children: {
        goals: { state: "feature/plan/goals", inputs: { issue: { input: "issue" } } },
        context: {
          state: "feature/plan/context",
          inputs: { issue: { input: "issue" }, goals: { child: "goals", output: "goals" } },
        },
        critique: {
          state: "feature/plan/critique",
          inputs: {
            plan_doc: { child: "context", output: "plan_doc" },
            severity_threshold: { text: "significant" },
          },
        },
      },
      sequence: ["goals", "context", "critique"],
      transitions: [
        { to: "terminate.success", when: "children.critique.outputs.outcome === 'clean'" },
        {
          to: "goals",
          when: "children.critique.outputs.outcome === 'needs_changes' && run.iteration < limits.max_iterations",
        },
        { to: "terminate.success", when: "children.critique.outcome === 'success'" },
      ],
      limits: { max_iterations: 3 },
    },
    "feature/plan/goals": {
      label: "Goals",
      inputs: { issue: artifact("markdown") },
      outputs: { goals: strArray() },
      operation: { kind: "prompt", prompt: { template: "Extract goals from {{inputs.issue}}." }, config: { model: "planner" } },
    },
    "feature/plan/context": {
      label: "Context",
      inputs: { issue: artifact("markdown"), goals: strArray() },
      outputs: { plan_doc: artifact("markdown") },
      operation: { kind: "prompt", prompt: { template: "Write the plan for {{inputs.issue}}." }, config: { model: "planner" } },
    },
    "feature/plan/critique": {
      label: "Critique Plan",
      description: "Review the current plan for significant weaknesses.",
      inputs: {
        plan_doc: artifact("markdown"),
        severity_threshold: { schema: { type: "string", enum: ["minor", "significant", "critical"] }, default: "significant" },
      },
      outputs: {
        outcome: { schema: { type: "string", enum: ["clean", "needs_changes", "blocked"] } },
        weaknesses: strArray(),
        critique_report: artifact("markdown"),
        human_decision: {
          schema: { type: "string", enum: ["approve", "request_changes", "block"] },
          optional: true,
          binding: { child: "human_review", output: "decision" },
        },
      },
      environment: { conversation: { mode: "full_history" } },
      operation: {
        kind: "prompt",
        config: { model: "critic" },
        prompt: {
          template:
            "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema.",
        },
      },
      children: {
        address_weaknesses: {
          state: "feature/plan/critique/address_weaknesses",
          inputs: {
            plan_doc: { input: "plan_doc" },
            weaknesses: { expr: "outputs.weaknesses" },
            critique_report: { expr: "outputs.critique_report" },
          },
        },
        human_review: {
          state: "feature/plan/critique/human_review",
          inputs: { plan_doc: { input: "plan_doc" }, critique_report: { expr: "outputs.critique_report" } },
        },
      },
      transitions: [
        { to: "terminate.success", when: "children.human_review.outcome === 'success'" },
        { to: "terminate.success", when: "children.address_weaknesses.outcome === 'success'" },
        { to: "terminate.success", when: "outputs.outcome === 'clean'" },
        { to: "human_review", when: "outputs.outcome === 'blocked'" },
        { to: "address_weaknesses", when: "outputs.outcome === 'needs_changes'" },
      ],
    },
    "feature/plan/critique/address_weaknesses": {
      label: "Address Weaknesses",
      inputs: { plan_doc: artifact("markdown"), weaknesses: strArray(), critique_report: artifact("markdown") },
      outputs: { resolution: str() },
      operation: { kind: "prompt", prompt: { template: "Fix the listed weaknesses." }, config: { model: "fixer" } },
    },
    "feature/plan/critique/human_review": {
      label: "Human Review",
      inputs: { plan_doc: artifact("markdown"), critique_report: artifact("markdown") },
      outputs: {
        decision: { schema: { type: "string", enum: ["approve", "request_changes", "block"] } },
        comments: { schema: { type: "string", format: "markdown" }, optional: true },
      },
      // An interactive host function — a plain FunctionOp like any other (§3), with its authored
      // surface bound as the `config` input.
      operation: {
        kind: "function",
        function: "choose_option",
        config: { prompt: "Review the critique result.", options: ["approve", "request_changes", "block"] },
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
      outputs: { summary: { ...str(), binding: { child: "synthesize", output: "summary" } } },
      children: {
        claude_review: { state: "review/agent_review", async: true, inputs: { change: { input: "change" } } },
        codex_review: { state: "review/agent_review", async: true, inputs: { change: { input: "change" } } },
        synthesize: {
          state: "review/synthesize",
          inputs: {
            review_a: { child: "claude_review", output: "report" },
            review_b: { child: "codex_review", output: "report" },
          },
        },
      },
      sequence: ["claude_review", "codex_review", "synthesize"],
    },
    "review/agent_review": {
      label: "Agent Review",
      inputs: { change: str() },
      outputs: { report: str() },
      operation: { kind: "prompt", prompt: { template: "Review {{inputs.change}}." }, config: { model: "reviewer" } },
    },
    "review/synthesize": {
      label: "Synthesize",
      inputs: { review_a: str(), review_b: str() },
      outputs: { summary: str() },
      operation: {
        kind: "prompt",
        prompt: { template: "Combine {{inputs.review_a}} and {{inputs.review_b}}." },
        config: { model: "synthesizer" },
      },
    },
  };
}
