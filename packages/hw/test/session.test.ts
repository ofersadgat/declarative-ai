/**
 * What an operation's `session` declaration means (DESIGN.md §1.6).
 *
 * Two properties carry the whole step, and they are easy to conflate:
 *
 *  - the CONVERSATION follows the declaration — a name joins that stream, `null` and absent each
 *    start a fresh one, and absent no longer means a shared "default";
 *  - the RESOURCE BUNDLE (workspace, permission ledger, approval scope) does NOT. It is inherited
 *    from the enclosing instance, so it survives a fork, a retry and a loop iteration — none of
 *    which change what the author declared.
 */
import { describe, expect, it } from "vitest";
import {
  RUN_RESOURCE_KEY,
  freshSessionKey,
  isSessionRef,
  resolveSession,
  validateSessionDecl,
  type SessionScope,
} from "../src/session";
import { mergeOperationFields, normalizeSynonyms } from "../src/merge";
import { loadBundle } from "../src/loader";
import { validateBundle } from "../src/validate";
import type { OperationFields, StateDef } from "../src/format";

const scope = (overrides: Partial<SessionScope> = {}): SessionScope => ({
  instanceId: 7,
  inheritedResourceKey: "enclosing",
  positionOf: () => undefined,
  ...overrides,
});

describe("what may be declared", () => {
  it("accepts a name, a ref and null", () => {
    expect(validateSessionDecl("planning")).toBeUndefined();
    expect(validateSessionDecl({ id: "ses_abc@14" })).toBeUndefined();
    expect(validateSessionDecl(null)).toBeUndefined();
    expect(validateSessionDecl(undefined)).toBeUndefined();
  });

  it("rejects the empty string, and says to write null instead", () => {
    // The failure this exists for: `"session": "{{inputs.thread}}"` with `thread` absent. An empty
    // string would start an isolated conversation and report success.
    expect(validateSessionDecl("")).toMatch(/write null/);
  });

  it("rejects a ref whose expression resolved to nothing", () => {
    expect(validateSessionDecl({ id: "" })).toMatch(/resolved to nothing/);
  });

  it("rejects a name in the reserved space engine-minted sessions live in", () => {
    expect(validateSessionDecl("#i3")).toMatch(/reserved/);
  });

  it("rejects a shape that is neither", () => {
    expect(validateSessionDecl(42)).toMatch(/must be a name, a session ref, or null/);
    expect(validateSessionDecl(["planning"])).toMatch(/an array/);
  });

  it("tells a ref from a name", () => {
    expect(isSessionRef({ id: "x" })).toBe(true);
    expect(isSessionRef("x")).toBe(false);
    expect(isSessionRef(null)).toBe(false);
    expect(isSessionRef({})).toBe(false);
  });
});

describe("resolution (§4)", () => {
  it("a NAME joins that stream and names the resource bundle", () => {
    const binding = resolveSession("planning", false, scope());
    expect(binding.id).toBe("planning");
    // The name is what stays put while the conversation moves — which is what makes a worktree and
    // its approvals survive a fork.
    expect(binding.resourceKey).toBe("planning");
  });

  it("a REF names an exact position but NOT a resource bundle", () => {
    // It arrived through data flow from an operation that may live anywhere in the tree, so it says
    // nothing about which workspace this state should act in.
    const binding = resolveSession({ id: "ses_abc@14" }, false, scope());
    expect(binding.id).toBe("ses_abc@14");
    expect(binding.resourceKey).toBe("enclosing");
  });

  it("null starts a fresh stream and keeps the enclosing bundle", () => {
    const binding = resolveSession(null, false, scope());
    expect(binding.id).toBe(freshSessionKey(7));
    expect(binding.resourceKey).toBe("enclosing");
  });

  it("absent agrees with null — it does NOT fall back to a shared default", () => {
    // The behaviour change: an implicit process-wide transcript is what drove unbounded context
    // growth, so an undeclared operation gets its own stream.
    expect(resolveSession(undefined, false, scope()).id).toBe(freshSessionKey(7));
    expect(resolveSession(undefined, false, scope()).id).not.toBe(RUN_RESOURCE_KEY);
  });

  it("gives two instances two different fresh streams", () => {
    expect(resolveSession(null, false, scope({ instanceId: 1 })).id).not.toBe(
      resolveSession(null, false, scope({ instanceId: 2 })).id,
    );
  });

  it("reads a named stream's position from THIS instance, not from a global head", () => {
    // The instance-scoped invariant (§4). A restarted state re-resolves to the position it started
    // from, so the append the failed attempt made turns the retry into a fork from the right place
    // instead of stacking it on top of the failure.
    const atFourteen = resolveSession("planning", false, scope({ positionOf: () => "ses_abc@14" }));
    const atTwenty = resolveSession("planning", false, scope({ positionOf: () => "ses_abc@20" }));
    expect(atFourteen.id).toBe("ses_abc@14");
    expect(atTwenty.id).toBe("ses_abc@20");
    // ...and the resource bundle is the same on both, because the declaration did not change.
    expect(atFourteen.resourceKey).toBe(atTwenty.resourceKey);
  });

  it("carries `fork` through from the consumption site", () => {
    expect(resolveSession("planning", true, scope()).fork).toBe(true);
    expect(resolveSession("planning", false, scope()).fork).toBe(false);
  });
});

describe("the environment merge", () => {
  it("lets a nearer null BEAT an inherited name", () => {
    // The case most merges get wrong: `null` has to survive as a VALUE, not as absence, or the
    // operation quietly joins the conversation its author was explicitly opting out of.
    const merged = mergeOperationFields({ session: "planning" }, { session: null });
    expect(merged.session).toBeNull();
    expect("session" in merged).toBe(true);
  });

  it("lets an absent key inherit", () => {
    expect(mergeOperationFields({ session: "planning" }, { model: "sonnet" }).session).toBe("planning");
  });

  it("normalizes sessionId to session BEFORE merging, so a child's null overrides", () => {
    // Left as two keys, `sessionId: null` would sit alongside the ancestor's `session: "planning"`
    // rather than overriding it.
    expect(mergeOperationFields({ session: "planning" }, { sessionId: null }).session).toBeNull();
    expect(normalizeSynonyms({ sessionId: null } as OperationFields).session).toBeNull();
    expect(normalizeSynonyms({ sessionId: "planning" } as OperationFields).session).toBe("planning");
  });

  it("still refuses a document declaring both, unless they agree", () => {
    expect(() => normalizeSynonyms({ session: "a", sessionId: "b" })).toThrow(/one of them has to go/);
    expect(() => normalizeSynonyms({ session: "a", sessionId: "a" })).not.toThrow();
    // Refs compare by id, so the same position written twice is not a conflict.
    expect(() => normalizeSynonyms({ session: { id: "x@1" }, sessionId: { id: "x@1" } })).not.toThrow();
    expect(() => normalizeSynonyms({ session: { id: "x@1" }, sessionId: { id: "x@2" } })).toThrow();
  });

  it("survives the split into the loaded execution environment", () => {
    const files: Record<string, StateDef> = {
      root: {
        environment: { session: "planning" },
        children: { leaf: { state: "leaf" } },
      },
      leaf: {
        environment: { session: null },
        operation: { kind: "prompt", prompt: "hi", model: "m" },
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    // `null` must reach the loaded state; dropped at the split it would silently restore "planning".
    expect(loadBundle(files, "root").states["leaf"]?.environment?.session).toBeNull();
  });
});

describe("the load-time lint", () => {
  const bundleWith = (session: unknown): ReturnType<typeof validateBundle> => {
    const files: Record<string, StateDef> = {
      root: {
        environment: { session } as never,
        operation: { kind: "prompt", prompt: "hi", model: "m" },
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    return validateBundle(loadBundle(files, "root"));
  };

  it("reports an empty session as an authoring error", () => {
    const issues = bundleWith("").errors;
    expect(issues.some((i) => i.path === "operation.session" && /write null/.test(i.message))).toBe(true);
  });

  it("says nothing about a well-formed one", () => {
    expect(bundleWith("planning").errors.filter((i) => i.path === "operation.session")).toEqual([]);
    expect(bundleWith(null).errors.filter((i) => i.path === "operation.session")).toEqual([]);
  });

  it("catches an inherited empty session, not just a locally written one", () => {
    const files: Record<string, StateDef> = {
      root: { environment: { session: "" } as never, children: { leaf: { state: "leaf" } } },
      leaf: {
        operation: { kind: "prompt", prompt: "hi", model: "m" },
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    const errors = validateBundle(loadBundle(files, "root")).errors;
    expect(errors.some((i) => i.stateId === "leaf" && i.path === "operation.session")).toBe(true);
  });
});
