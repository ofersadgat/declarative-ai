/**
 * What an operation's `session` declaration means (DESIGN.md §1.6).
 *
 * Three properties carry the whole thing, and they are easy to conflate:
 *
 *  - the CONVERSATION follows the declaration — a name joins that stream, `null` and absent each
 *    start a fresh one, and absent no longer means a shared "default";
 *  - the RESOURCE BUNDLE (workspace, permission ledger, approval scope) follows what was DECLARED,
 *    so it survives a fork, a retry and a loop iteration — none of which change the declaration;
 *  - a NAME is qualified by the SCOPE it was written in. That is what stops one word written in two
 *    unrelated subtrees from meaning one transcript and one worktree.
 */
import { describe, expect, it } from "vitest";
import {
  RUN_RESOURCE_KEY,
  freshSessionKey,
  isSessionRef,
  normalizeSession,
  resolveSession,
  sessionFromExpr,
  sessionKeyOf,
  validateSessionDecl,
  type SessionAncestor,
  type SessionScope,
} from "../src/session.js";
import { mergeOperationFields, refuseSynonyms } from "../src/merge.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import type { OperationFields, StateDef } from "../src/format.js";

const scope = (overrides: Partial<SessionScope> = {}): SessionScope => ({
  instanceId: "7",
  inheritedResourceKey: "enclosing",
  positionOf: () => undefined,
  // Every state is its own anchor unless a test says otherwise, which is what a bare name does.
  anchorOf: (stateId) => stateId,
  ...overrides,
});

/** A writer with no ancestors — the root case, and the simplest thing to normalize against. */
const writer = (id: string): { id: string; source: string } => ({ id, source: id });
const ancestor = (id: string, declared?: SessionAncestor["declared"]): SessionAncestor => ({
  id,
  source: id,
  ...(declared !== undefined ? { declared } : {}),
});

describe("what may be declared", () => {
  it("accepts a name, a ref, an expression and null", () => {
    expect(validateSessionDecl("planning")).toBeUndefined();
    expect(validateSessionDecl({ id: "ses_abc@14" })).toBeUndefined();
    expect(validateSessionDecl({ expr: ".inputs.thread" })).toBeUndefined();
    expect(validateSessionDecl(null)).toBeUndefined();
    expect(validateSessionDecl(undefined)).toBeUndefined();
  });

  it("accepts the scoped and joined spellings", () => {
    expect(validateSessionDecl({ name: "review", in: "parent" })).toBeUndefined();
    expect(validateSessionDecl({ name: "review", in: "batch", fork: true })).toBeUndefined();
    expect(validateSessionDecl({ join: "nearest" })).toBeUndefined();
  });

  it("rejects the empty string, and says to write null instead", () => {
    // The failure this exists for: `"session": "{{.inputs.thread}}"` with `thread` absent. An empty
    // string would start an isolated conversation and report success.
    expect(validateSessionDecl("")).toMatch(/write null/);
  });

  it("rejects a ref whose expression resolved to nothing", () => {
    expect(validateSessionDecl({ id: "" })).toMatch(/resolved to nothing/);
  });

  it("rejects a name in the reserved space engine-minted sessions live in", () => {
    expect(validateSessionDecl("#i3")).toMatch(/reserved/);
    expect(validateSessionDecl({ name: "#i3" })).toMatch(/reserved/);
  });

  /**
   * `in` qualifies a NAME. With no name there is nothing to qualify, and the session it would
   * produce is one nothing else can refer to — which is what `null` already says. Accepting it as a
   * second spelling for "private" is what would make `{ in: "parent" }` look like a way for siblings
   * to rendezvous, so it is refused with the alternative named.
   */
  it("rejects `in` with no name, and points at null", () => {
    expect(validateSessionDecl({ in: "parent" })).toMatch(/cannot be joined by anything/);
    expect(validateSessionDecl({ in: "parent" })).toMatch(/write null/);
  });

  it("rejects two addressing modes at once", () => {
    // A ref is an absolute address and a name is one relative to a scope; a declaration carrying
    // both is not a pair of coordinates, it is two different answers.
    expect(validateSessionDecl({ name: "review", id: "ses_1" })).toMatch(/exactly one is meaningful/);
    expect(validateSessionDecl({ join: "parent", in: "batch" })).toMatch(/nothing left for 'in' to qualify/);
  });

  it("rejects an empty object, and a shape that is neither", () => {
    expect(validateSessionDecl({})).toMatch(/must declare one of/);
    expect(validateSessionDecl(42)).toMatch(/must be a name/);
    expect(validateSessionDecl(["planning"])).toMatch(/an array/);
  });

  it("tells a ref from a name", () => {
    expect(isSessionRef({ id: "x" })).toBe(true);
    expect(isSessionRef("x")).toBe(false);
    expect(isSessionRef(null)).toBe(false);
    expect(isSessionRef({})).toBe(false);
  });
});

/**
 * Normalization is where the scope gets stamped on, and it runs at the place the declaration was
 * WRITTEN. That ordering is the whole mechanism: after the environment merge a root's `"planning"`
 * and a leaf's are the same string, so nothing downstream could tell them apart.
 */
describe("normalizing a declaration at its origin", () => {
  it("expands a bare name to the writer's own scope", () => {
    expect(normalizeSession("review", writer("leaf"), [])).toEqual({ session: { name: "review", in: "leaf" } });
  });

  it("scopes to the parent on request, so siblings can share one name", () => {
    const ancestry = [ancestor("root"), ancestor("batch")];
    expect(normalizeSession({ name: "review", in: "parent" }, writer("coder"), ancestry)).toEqual({
      session: { name: "review", in: "batch" },
    });
    expect(normalizeSession({ name: "review", in: "parent" }, writer("reviewer"), ancestry)).toEqual({
      session: { name: "review", in: "batch" },
    });
  });

  it("scopes to a named ancestor, and refuses one that is not an ancestor", () => {
    const ancestry = [ancestor("root"), ancestor("batch")];
    expect(normalizeSession({ name: "review", in: "root" }, writer("coder"), ancestry)).toEqual({
      session: { name: "review", in: "root" },
    });
    expect(normalizeSession({ name: "review", in: "elsewhere" }, writer("coder"), ancestry)).toEqual({
      error: expect.stringContaining("does not name an ancestor") as unknown as string,
    });
  });

  it("refuses `in: parent` at the root, where there is no parent", () => {
    const outcome = normalizeSession({ name: "review", in: "parent" }, writer("root"), []);
    expect(outcome).toHaveProperty("error");
    expect((outcome as { error: string }).error).toMatch(/is the root/);
  });

  /**
   * The behaviour change with teeth. Two siblings writing one word do NOT share, because they are
   * two writers and therefore two scopes. It reads as a surprise exactly once, and the alternative
   * — the old flat namespace — is the surprise that cannot be seen at all.
   */
  it("gives two siblings that write one name two different sessions", () => {
    const ancestry = [ancestor("batch")];
    const left = normalizeSession("review", writer("coder"), ancestry);
    const right = normalizeSession("review", writer("reviewer"), ancestry);
    expect(left).not.toEqual(right);
  });

  it("refuses `document`, which names no unit in this model", () => {
    // A state IS a file here (children are inferred from the directory listing), so a document's
    // root is the writer itself and the keyword would mean nothing `in` absent does not.
    const outcome = normalizeSession({ name: "review", in: "document" }, writer("leaf"), [ancestor("root")]);
    expect(outcome).toHaveProperty("error");
    expect((outcome as { error: string }).error).toMatch(/names no scope in this model/);
  });
});

/**
 * `join` takes another declaration's name and scope, for a state that wants in without knowing what
 * the conversation is called. Each keyword is a different assertion about WHERE the writer is, which
 * is the author choosing how much structural change should break them.
 */
describe("joining a session you cannot name", () => {
  const root = ancestor("root", { name: "thread", in: "root" });
  const wrapper = ancestor("wrapper");
  const batch = ancestor("batch", { name: "work", in: "batch" });

  it("`nearest` takes the closest writer", () => {
    expect(normalizeSession({ join: "nearest" }, writer("leaf"), [root, batch])).toEqual({
      session: { name: "work", in: "batch" },
    });
  });

  it("`nearest` skips a state that only INHERITED a session", () => {
    // "Wrote" and "has" are different: after the merge every state appears to have one, and only the
    // loader still knows who actually declared it.
    expect(normalizeSession({ join: "nearest" }, writer("leaf"), [root, wrapper])).toEqual({
      session: { name: "thread", in: "root" },
    });
  });

  /**
   * The point of `parent` over `nearest`: it ASSERTS that the immediate parent declares one. Insert
   * a wrapper composite and `nearest` silently retargets to the grandparent — the child joins a
   * different transcript with nothing to notice — while `parent` turns the tree edit into an error.
   */
  it("`parent` fails where `nearest` would silently retarget", () => {
    expect(normalizeSession({ join: "parent" }, writer("leaf"), [root, batch])).toEqual({
      session: { name: "work", in: "batch" },
    });
    const throughWrapper = normalizeSession({ join: "parent" }, writer("leaf"), [root, wrapper]);
    expect(throughWrapper).toHaveProperty("error");
    expect((throughWrapper as { error: string }).error).toMatch(/'join: "nearest"' would follow/);
  });

  it("`global` pins the run root", () => {
    expect(normalizeSession({ join: "global" }, writer("leaf"), [root, batch])).toEqual({
      session: { name: "thread", in: "root" },
    });
  });

  it("names an ancestor directly, pinning that specific writer", () => {
    expect(normalizeSession({ join: "root" }, writer("leaf"), [root, batch])).toEqual({
      session: { name: "thread", in: "root" },
    });
    expect(normalizeSession({ join: "wrapper" }, writer("leaf"), [root, wrapper])).toEqual({
      error: expect.stringContaining("declares none") as unknown as string,
    });
  });

  it("reports when nothing above declares anything at all", () => {
    const outcome = normalizeSession({ join: "nearest" }, writer("leaf"), [wrapper]);
    expect(outcome).toHaveProperty("error");
    expect((outcome as { error: string }).error).toMatch(/no ancestor declares one/);
  });

  it("takes the writer's name but the JOINER's fork", () => {
    // `fork` is a statement about how this call will use the session, so it is the one thing the
    // joiner keeps; the name and the scope are taken verbatim, which is the whole point.
    expect(normalizeSession({ join: "nearest", fork: true }, writer("leaf"), [batch])).toEqual({
      session: { name: "work", in: "batch", fork: true },
    });
  });

  it("joins a parent's explicitly FRESH session", () => {
    // `null` is a declaration, not an absence: the parent said "a fresh thread for me", and a child
    // joining it is saying "I am in that thread".
    expect(normalizeSession({ join: "parent" }, writer("leaf"), [ancestor("batch", null)])).toEqual({ session: null });
  });
});

describe("resolution (§4)", () => {
  it("a NAME keys both the stream and the resource bundle on the (name, scope) PAIR", () => {
    const binding = resolveSession({ name: "planning", in: "root" }, scope());
    expect(binding.id).toBe(sessionKeyOf("planning", "root"));
    // The pair is what stays put while the conversation moves — which is what makes a worktree and
    // its approvals survive a fork.
    expect(binding.resourceKey).toBe(sessionKeyOf("planning", "root"));
  });

  it("keeps one name written in two scopes apart", () => {
    const left = resolveSession({ name: "main", in: "featureA" }, scope());
    const right = resolveSession({ name: "main", in: "featureB" }, scope());
    expect(left.id).not.toBe(right.id);
    expect(left.resourceKey).not.toBe(right.resourceKey);
  });

  it("resolves a scope to the enclosing INSTANCE, so a loop's passes differ", () => {
    // Anchor inside the loop and each pass is a new instance, so each gets its own conversation and
    // its own bundle. Anchor above it and every pass finds the same instance — which is how §5.1's
    // stability rule falls out instead of being a rule about loops.
    const pass = (instance: string): string =>
      resolveSession({ name: "review", in: "body" }, scope({ anchorOf: () => instance })).resourceKey;
    expect(pass("i1")).not.toBe(pass("i2"));
    const stable = (): string =>
      resolveSession({ name: "review", in: "root" }, scope({ anchorOf: () => "rootInstance" })).resourceKey;
    expect(stable()).toBe(stable());
  });

  it("a REF names an exact position but NOT a resource bundle", () => {
    // It arrived through data flow from an operation that may live anywhere in the tree, so it says
    // nothing about which workspace this state should act in.
    const binding = resolveSession({ id: "ses_abc@14" }, scope());
    expect(binding.id).toBe("ses_abc@14");
    expect(binding.resourceKey).toBe("enclosing");
  });

  it("null starts a fresh stream and keeps the enclosing bundle", () => {
    const binding = resolveSession(null, scope());
    expect(binding.id).toBe(freshSessionKey("7"));
    expect(binding.resourceKey).toBe("enclosing");
  });

  it("absent agrees with null — it does NOT fall back to a shared default", () => {
    // The behaviour change: an implicit process-wide transcript is what drove unbounded context
    // growth, so an undeclared operation gets its own stream.
    expect(resolveSession(undefined, scope()).id).toBe(freshSessionKey("7"));
    expect(resolveSession(undefined, scope()).id).not.toBe(RUN_RESOURCE_KEY);
  });

  it("gives two instances two different fresh streams", () => {
    expect(resolveSession(null, scope({ instanceId: "1" })).id).not.toBe(resolveSession(null, scope({ instanceId: "2" })).id);
  });

  it("reads a named stream's position from THIS instance, not from a global head", () => {
    // The instance-scoped invariant (§4). A restarted state re-resolves to the position it started
    // from, so the append the failed attempt made turns the retry into a fork from the right place
    // instead of stacking it on top of the failure.
    const named = { name: "planning", in: "root" } as const;
    const atFourteen = resolveSession(named, scope({ positionOf: () => "ses_abc@14" }));
    const atTwenty = resolveSession(named, scope({ positionOf: () => "ses_abc@20" }));
    expect(atFourteen.id).toBe("ses_abc@14");
    expect(atTwenty.id).toBe("ses_abc@20");
    // ...and the resource bundle is the same on both, because the declaration did not change.
    expect(atFourteen.resourceKey).toBe(atTwenty.resourceKey);
  });

  it("carries `fork` off the declaration it belongs to", () => {
    expect(resolveSession({ name: "planning", in: "root", fork: true }, scope()).fork).toBe(true);
    expect(resolveSession({ name: "planning", in: "root" }, scope()).fork).toBe(false);
    expect(resolveSession({ id: "ses_1", fork: true }, scope()).fork).toBe(true);
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

  it("replaces a session WHOLE, so fork cannot arrive from a different layer than the name", () => {
    const merged = mergeOperationFields({ session: { name: "a", fork: true } }, { session: { name: "b" } });
    expect(merged.session).toEqual({ name: "b" });
  });

  /**
   * `sessionId` was a synonym for `session` so an `LlmConfiguration`-shaped block could paste in
   * unchanged. It is refused now — and refused rather than IGNORED, because anything unrecognized is
   * passed through to the LLM call config. A silently dropped `sessionId` would therefore be sent to
   * the model as a call parameter while the operation quietly started a fresh conversation.
   */
  it("refuses `sessionId`, rather than shipping it to the model as a call parameter", () => {
    expect(() => refuseSynonyms({ sessionId: null } as OperationFields)).toThrow(/the field is called 'session'/);
    expect(() => mergeOperationFields({ session: "planning" }, { sessionId: null } as OperationFields)).toThrow(
      /the field is called 'session'/,
    );
    expect(() => refuseSynonyms({ session: "planning" })).not.toThrow();
  });

  /**
   * A top-level `fork` inherited down the environment chain on its own, so a root that wrote it
   * branched every descendant's conversation whatever each of them had declared — a cross-field
   * interaction no author ever asks for. It belongs to the session it is about.
   */
  it("refuses a `fork` written beside the session, and names the new spelling", () => {
    expect(() => refuseSynonyms({ fork: true } as OperationFields)).toThrow(/fork is a property OF the session/);
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

/**
 * The reason normalization runs at the ORIGIN rather than after the merge.
 *
 * A root declaring one session for its subtree is the ordinary way to thread a conversation through
 * it (SPEC §7.1a). Scope the declaration where it LANDS and every descendant would get a private
 * session of its own instead, which is the documented behaviour inverted.
 */
describe("an inherited declaration keeps the scope of the state that WROTE it", () => {
  const files: Record<string, StateDef> = {
    root: {
      environment: { session: "planning" },
      children: { a: { state: "a" }, b: { state: "b" } },
    },
    a: {
      operation: { kind: "prompt", prompt: "hi", model: "m" },
      outputs: { answer: { schema: { type: "string" } } },
    },
    b: {
      operation: { kind: "prompt", prompt: "hi", model: "m" },
      outputs: { answer: { schema: { type: "string" } } },
    },
  };

  it("scopes at the root, so the whole subtree shares one conversation", () => {
    const bundle = loadBundle(files, "root");
    expect(bundle.states["a"]?.environment?.session).toEqual({ name: "planning", in: "root" });
    expect(bundle.states["b"]?.environment?.session).toEqual({ name: "planning", in: "root" });
  });

  it("scopes a leaf's OWN name to the leaf, which is what makes it private", () => {
    const own: Record<string, StateDef> = {
      ...files,
      a: { ...files["a"]!, operation: { ...files["a"]!.operation, session: "planning" } as never },
    };
    const bundle = loadBundle(own, "root");
    expect(bundle.states["a"]?.environment?.session).toEqual({ name: "planning", in: "a" });
    // Same word, two scopes: `a`'s own declaration does not join the root's thread.
    expect(bundle.states["b"]?.environment?.session).toEqual({ name: "planning", in: "root" });
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

  const sessionErrors = (session: unknown): ReturnType<typeof validateBundle>["errors"] =>
    bundleWith(session).errors.filter((i) => i.path.endsWith("session"));

  it("reports an empty session as an authoring error, at the line that declared it", () => {
    // `environment.session`, not `operation.session`: the complaint is found where the declaration
    // was WRITTEN now, so the path names the key the author typed rather than the merged view of it.
    const issues = sessionErrors("");
    expect(issues.some((i) => i.path === "environment.session" && /write null/.test(i.message))).toBe(true);
    expect(issues).toHaveLength(1);
  });

  it("says nothing about a well-formed one", () => {
    expect(sessionErrors("planning")).toEqual([]);
    expect(sessionErrors(null)).toEqual([]);
    expect(sessionErrors({ name: "planning" })).toEqual([]);
  });

  /**
   * Reported against the state that WROTE it, not against the descendants that inherit it.
   *
   * The old flat namespace had no choice but to report it downstream: the complaint was found after
   * the merge, by which point the declaration had already moved. Normalizing at the origin catches
   * it on the line the author actually wrote — the same reasoning `validate.ts` already applies to a
   * mount environment, which it checks at the mount rather than against the child.
   */
  it("catches a bad session on the state that wrote it, not on the ones inheriting it", () => {
    const files: Record<string, StateDef> = {
      root: { environment: { session: "" } as never, children: { leaf: { state: "leaf" } } },
      leaf: {
        operation: { kind: "prompt", prompt: "hi", model: "m" },
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    const errors = validateBundle(loadBundle(files, "root")).errors;
    expect(errors.some((i) => i.stateId === "root" && /write null/.test(i.message))).toBe(true);
    // ...and the broken declaration does not travel, so the leaf is not blamed for it twice.
    expect(errors.some((i) => i.stateId === "leaf" && i.path === "operation.session")).toBe(false);
  });

  /**
   * A scope that names no ancestor is a question about the TREE, so it is answered in the walk and
   * carried to the validator — reported beside every other authoring error rather than aborting the
   * load at the first one.
   */
  it("reports a scope that names no ancestor, against the state that wrote it", () => {
    const files: Record<string, StateDef> = {
      root: { children: { leaf: { state: "leaf" } } },
      leaf: {
        operation: { kind: "prompt", prompt: "hi", model: "m", session: { name: "x", in: "nowhere" } } as never,
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    const errors = validateBundle(loadBundle(files, "root")).errors;
    expect(errors.some((i) => i.stateId === "leaf" && /does not name an ancestor/.test(i.message))).toBe(true);
  });

  it("reports a `join` that finds no writer", () => {
    const files: Record<string, StateDef> = {
      root: { children: { leaf: { state: "leaf" } } },
      leaf: {
        operation: { kind: "prompt", prompt: "hi", model: "m", session: { join: "nearest" } } as never,
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    const errors = validateBundle(loadBundle(files, "root")).errors;
    expect(errors.some((i) => i.stateId === "leaf" && /no ancestor declares one/.test(i.message))).toBe(true);
  });

  it("resolves a `join` through the loader, leaving no `join` in the loaded document", () => {
    const files: Record<string, StateDef> = {
      root: { environment: { session: "thread" }, children: { leaf: { state: "leaf" } } },
      leaf: {
        operation: { kind: "prompt", prompt: "hi", model: "m", session: { join: "nearest", fork: true } } as never,
        outputs: { answer: { schema: { type: "string" } } },
      },
    };
    const bundle = loadBundle(files, "root");
    expect(bundle.states["leaf"]?.environment?.session).toEqual({ name: "thread", in: "root", fork: true });
  });
});

/**
 * What an evaluated `{ expr }` session is allowed to be.
 *
 * The case worth naming is absence. "Then start fresh" is the tempting reading and the wrong one:
 * the author named a conversation to continue, so quietly running in a different one produces a
 * successful-looking run that has forgotten everything.
 */
describe("an expression's resolved value", () => {
  const at = (expr: string) => ({ expr, in: "leaf" });

  it("takes a ref, and a string as a name in the WRITER's scope", () => {
    expect(sessionFromExpr(at(".inputs.t"), { id: "ses_abc@14" })).toEqual({ session: { id: "ses_abc@14" } });
    // A computed name is scoped where a written one would have been — at the state that wrote the
    // expression — so the two spellings cannot mean different things in the same place.
    expect(sessionFromExpr(at(".inputs.t"), "planning")).toEqual({ session: { name: "planning", in: "leaf" } });
  });

  it("REFUSES nothing, naming the expression so the wiring is findable", () => {
    for (const value of [undefined, null]) {
      const outcome = sessionFromExpr(at(".inputs.thread"), value);
      expect(outcome).toHaveProperty("error");
      expect((outcome as { error: string }).error).toContain(".inputs.thread");
    }
  });

  it("REFUSES an empty id and an empty name, for the same reason", () => {
    expect(sessionFromExpr(at(".inputs.t"), { id: "" })).toHaveProperty("error");
    expect(sessionFromExpr(at(".inputs.t"), "")).toHaveProperty("error");
  });

  it("REFUSES a computed `join`, which is answered against the document and not the data", () => {
    const outcome = sessionFromExpr(at(".inputs.t"), { join: "nearest" });
    expect(outcome).toHaveProperty("error");
    expect((outcome as { error: string }).error).toMatch(/resolved at load time/);
  });
});
