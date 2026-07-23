/**
 * @declarative-ai/permissions — the tool-call permission model (DESIGN §5.1, "Permissions: two orthogonal axes").
 *
 * Its only consumers are the workflow engine and the delegated-agent adapters, so it is its own
 * package rather than 265 lines sitting in a core everything depends on. It DECLARES its own seams on
 * `ExecServices` (DESIGN §3.2) — `exec` therefore does not know that permissions exist.
 */
import type { Approver, ExecPolicy } from "./permissions";

export * from "./permissions";

declare module "@declarative-ai/exec" {
  interface ExecServices {
    /** The compiled safety policy for the operation in flight — enforced per the executing entry's
     *  `policyEnforcement` capability. */
    policy?: ExecPolicy;
    /** The human tool-call approver. The engine wraps a COMPOSED runtime's tools itself, but a
     *  DELEGATED runtime that drives its own loop reads this to route its native permission callback
     *  back through our approval UI. Absent ⇒ no interactive gate. */
    approve?: Approver;
  }
}
