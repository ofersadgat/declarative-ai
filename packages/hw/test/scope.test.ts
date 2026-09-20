/**
 * Scoped names, with no position attached (`scope.ts`).
 *
 * `session.test.ts` covers the same mechanic through its first user. What is pinned here is the part
 * that makes it reusable: nothing below mentions a session, and the position's noun reaches only the
 * complaints.
 */
import { describe, expect, it } from "vitest";
import { addressPath, anchorScope, joinedWriter, keyOfScopedName, scopedKeyOf, type ScopeLink } from "../src/scope.js";

const link = (id: string, source: string = id): ScopeLink => ({ id, source });

describe("where a name is scoped", () => {
  const ancestry = [link("root"), link("batch@v2", "batch")];

  it("is the writer when nothing redirects it", () => {
    expect(anchorScope(undefined, link("leaf"), ancestry, "workspace")).toEqual({ id: "leaf" });
  });

  it("resolves a keyword or a canonical id to the id the ancestor LOADS under", () => {
    expect(anchorScope("parent", link("leaf"), ancestry, "workspace")).toEqual({ id: "batch@v2" });
    expect(anchorScope("batch", link("leaf"), ancestry, "workspace")).toEqual({ id: "batch@v2" });
    expect(anchorScope("global", link("leaf"), ancestry, "workspace")).toEqual({ id: "root" });
    // At the root the run's scope IS the writer.
    expect(anchorScope("global", link("root"), [], "workspace")).toEqual({ id: "root" });
  });

  it("complains in the position's own words", () => {
    expect(anchorScope("parent", link("root"), [], "workspace")).toEqual({
      error: `workspace '$in: "parent"' has no parent to scope to — 'root' is the root`,
    });
    expect(anchorScope("sibling", link("leaf"), ancestry, "workspace")).toMatchObject({
      error: expect.stringMatching(/^workspace '\$in: "sibling"' does not name an ancestor of 'leaf'/),
    });
  });
});

describe("taking what an ancestor wrote", () => {
  // One ancestry serves every position: each asks it for its own declaration.
  interface Link extends ScopeLink {
    wrote?: { workspace?: string };
  }
  const ancestry: Link[] = [{ ...link("root"), wrote: { workspace: "main" } }, link("wrapper"), link("batch")];
  const workspaceOf = (at: Link): string | undefined => at.wrote?.workspace;

  it("'nearest' follows any writer above; 'parent' insists on the immediate one", () => {
    expect(joinedWriter("nearest", ancestry, workspaceOf, "workspace")).toEqual({ declared: "main" });
    expect(joinedWriter("parent", ancestry, workspaceOf, "workspace")).toMatchObject({
      error: expect.stringMatching(/requires 'batch' to declare a workspace and it declares none/),
    });
  });

  it("an id pins that state, and 'global' the run root", () => {
    expect(joinedWriter("root", ancestry, workspaceOf, "workspace")).toEqual({ declared: "main" });
    expect(joinedWriter("global", ancestry, workspaceOf, "workspace")).toEqual({ declared: "main" });
    expect(joinedWriter("wrapper", ancestry, workspaceOf, "workspace")).toMatchObject({
      error: expect.stringMatching(/requires 'wrapper' to declare a workspace, and it declares none/),
    });
  });
});

describe("the key a use resolves to", () => {
  it("spells an address as a path: the root, an occurrence past the first, a fan-out element", () => {
    expect(addressPath([])).toBe("/");
    expect(addressPath([{ childKey: "loop", occurrence: 0 }, { childKey: "draft", occurrence: 2 }])).toBe("loop/draft:2");
    expect(addressPath([{ childKey: "files", occurrence: 1, element: 3 }])).toBe("files:1[3]");
  });

  it("is the name qualified by the anchoring INSTANCE, not by the scoping state", () => {
    const scoped = { $ref: "impl", $in: "loop" };
    expect(keyOfScopedName(scoped, () => "loop:1")).toBe(scopedKeyOf("impl", "loop:1"));
    expect(scopedKeyOf("impl", "loop:1")).toBe("impl#loop:1");
  });

  it("degrades to a run-global key, never a private one, when no instance of the scope encloses the use", () => {
    expect(keyOfScopedName({ $ref: "impl", $in: "loop" }, () => undefined)).toBe("impl#loop");
  });
});
