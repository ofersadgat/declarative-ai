/**
 * @declarative-ai/permissions — the tool-call permission model (DESIGN §5.1, "Permissions: two orthogonal axes").
 *
 * Its only consumers are the workflow engine and the delegated-agent adapters, so it is its own
 * package rather than 265 lines sitting in a core everything depends on. It DECLARES its own seams on
 * `ExecServices` (DESIGN §3.2) — `exec` therefore does not know that permissions exist.
 */
import type { Approver, ExecPolicy, ToolGate } from "./permissions.js";

export * from "./permissions.js";

declare module "@declarative-ai/exec" {
  interface ExecServices {
    /** The compiled safety policy for the operation in flight — enforced per the executing entry's
     *  `policyEnforcement` capability. */
    policy?: ExecPolicy;
    /** The human tool-call approver. The engine wraps a COMPOSED runtime's tools itself, but a
     *  DELEGATED runtime that drives its own loop reads this to route its native permission callback
     *  back through our approval UI. Absent ⇒ no interactive gate. */
    approve?: Approver;
    /**
     * The full permission gate for a DELEGATED call — profile, mode, `smart`, then the human.
     *
     * Distinct from {@link approve}, which is only the last of those four. An adapter that answered
     * its native callback with `approve` alone put every tool to a human: `smart` never ran its
     * policy, and an authored `allow` was asked about anyway. This is what it should consult instead;
     * `approve` remains for an adapter that has not been taught about it, and as the escalation the
     * gate itself performs.
     *
     * Absent ⇒ fall back to `approve`. Both absent ⇒ no gate, and a delegated agent's own tools run
     * ungated — which is a wiring mistake, not a mode.
     */
    gate?: ToolGate;
  }
}
