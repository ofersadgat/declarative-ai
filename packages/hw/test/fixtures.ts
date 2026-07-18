/**
 * Shared fixtures: the SPEC's worked examples (§7.3 critique, §8.2 human review,
 * §9 planning parent) plus the minimal supporting states they reference. Used by
 * loader/validator tests and as the engine golden tests' workflow.
 */
import type { StateDef } from "../src/format";

export const PLAN_ID = "feature/plan";

export function specPlanningFiles(): Record<string, StateDef> {
  return {
    "feature/plan": {
      label: "Planning",
      inputs: { issue: { type: "artifact", format: "markdown" } },
      outputs: {
        outcome: {
          type: "string",
          enum: ["complete", "blocked"],
          from: "children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'",
        },
        plan_doc: { type: "artifact", format: "markdown", from: "children.context.outputs.plan_doc" },
        critique: { type: "passthrough", from: "children.critique.outputs" },
      },
      children: {
        goals: { state: "feature/plan/goals", inputs: { issue: "inputs.issue" } },
        context: {
          state: "feature/plan/context",
          inputs: { issue: "inputs.issue", goals: "children.goals.outputs.goals" },
        },
        critique: {
          state: "feature/plan/critique",
          inputs: {
            plan_doc: "children.context.outputs.plan_doc",
            severity_threshold: { value: "significant" },
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
      inputs: { issue: { type: "artifact", format: "markdown" } },
      outputs: { goals: { type: "array", items: { type: "string" } } },
      agent: { provider: "planner", prompt: { template: "Extract goals from {{inputs.issue}}." } },
    },
    "feature/plan/context": {
      label: "Context",
      inputs: {
        issue: { type: "artifact", format: "markdown" },
        goals: { type: "array", items: { type: "string" } },
      },
      outputs: { plan_doc: { type: "artifact", format: "markdown" } },
      agent: { provider: "planner", prompt: { template: "Write the plan for {{inputs.issue}}." } },
    },
    "feature/plan/critique": {
      label: "Critique Plan",
      description: "Review the current plan for significant weaknesses.",
      inputs: {
        plan_doc: { type: "artifact", format: "markdown" },
        severity_threshold: {
          type: "string",
          enum: ["minor", "significant", "critical"],
          default: "significant",
        },
      },
      outputs: {
        outcome: { type: "string", enum: ["clean", "needs_changes", "blocked"] },
        weaknesses: { type: "array", items: { type: "string" } },
        critique_report: { type: "artifact", format: "markdown" },
        human_decision: {
          type: "string",
          enum: ["approve", "request_changes", "block"],
          optional: true,
          from: "children.human_review.outputs.decision",
        },
      },
      agent: {
        provider: "critic",
        conversation: { mode: "full_history" },
        prompt: {
          template:
            "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema.",
        },
      },
      children: {
        address_weaknesses: {
          state: "feature/plan/critique/address_weaknesses",
          inputs: {
            plan_doc: "inputs.plan_doc",
            weaknesses: "outputs.weaknesses",
            critique_report: "outputs.critique_report",
          },
        },
        human_review: {
          state: "feature/plan/critique/human_review",
          inputs: { plan_doc: "inputs.plan_doc", critique_report: "outputs.critique_report" },
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
      inputs: {
        plan_doc: { type: "artifact", format: "markdown" },
        weaknesses: { type: "array", items: { type: "string" } },
        critique_report: { type: "artifact", format: "markdown" },
      },
      outputs: { resolution: { type: "string" } },
      agent: { provider: "fixer", prompt: { template: "Fix the listed weaknesses." } },
    },
    "feature/plan/critique/human_review": {
      label: "Human Review",
      inputs: {
        plan_doc: { type: "artifact", format: "markdown" },
        critique_report: { type: "artifact", format: "markdown" },
      },
      outputs: {
        decision: { type: "string", enum: ["approve", "request_changes", "block"] },
        comments: { type: "string", format: "markdown", optional: true },
      },
      ui: {
        component: "choose_option",
        prompt: "Review the critique result.",
        options: ["approve", "request_changes", "block"],
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
      inputs: { change: { type: "string" } },
      outputs: { summary: { type: "string", from: "children.synthesize.outputs.summary" } },
      children: {
        claude_review: {
          state: "review/agent_review",
          async: true,
          inputs: { change: "inputs.change" },
        },
        codex_review: {
          state: "review/agent_review",
          async: true,
          inputs: { change: "inputs.change" },
        },
        synthesize: {
          state: "review/synthesize",
          inputs: {
            review_a: "children.claude_review.outputs.report",
            review_b: "children.codex_review.outputs.report",
          },
        },
      },
      sequence: ["claude_review", "codex_review", "synthesize"],
    },
    "review/agent_review": {
      label: "Agent Review",
      inputs: { change: { type: "string" } },
      outputs: { report: { type: "string" } },
      agent: { provider: "reviewer", prompt: { template: "Review {{inputs.change}}." } },
    },
    "review/synthesize": {
      label: "Synthesize",
      inputs: { review_a: { type: "string" }, review_b: { type: "string" } },
      outputs: { summary: { type: "string" } },
      agent: { provider: "synthesizer", prompt: { template: "Combine {{inputs.review_a}} and {{inputs.review_b}}." } },
    },
  };
}
