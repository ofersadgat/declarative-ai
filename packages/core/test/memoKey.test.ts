import { describe, expect, it } from "vitest";
import { hashCanonical, memoKey, sha256Hex } from "../src/hashing";

describe("memoKey (DESIGN §3.4)", () => {
  const base = { kind: "llm-call", definitionHash: "abc", inputs: { a: 1, b: "x" } };

  it("is invariant to input key order (JCS canonicalization)", () => {
    expect(memoKey({ ...base, inputs: { b: "x", a: 1 } })).toBe(memoKey(base));
  });

  it("changes when kind, definitionHash, or any input changes", () => {
    const k = memoKey(base);
    expect(memoKey({ ...base, kind: "hierarchical-workflow" })).not.toBe(k);
    expect(memoKey({ ...base, definitionHash: "abd" })).not.toBe(k);
    expect(memoKey({ ...base, inputs: { a: 2, b: "x" } })).not.toBe(k);
  });

  it("distinguishes workspace-bound executions from pure ones", () => {
    const pure = memoKey(base);
    const bound = memoKey({ ...base, workspaceTreeHash: "t1" });
    expect(bound).not.toBe(pure);
    expect(memoKey({ ...base, workspaceTreeHash: "t2" })).not.toBe(bound);
  });

  it("sha256Hex matches the known test vector", () => {
    // sha256("") — the canonical empty-string vector.
    expect(sha256Hex("")).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  });

  it("hashCanonical equates JSON-equal values", () => {
    expect(hashCanonical({ x: [1, 2], y: "z" })).toBe(hashCanonical({ y: "z", x: [1, 2] }));
  });

  it("throws on non-serializable inputs rather than hashing nothing", () => {
    expect(() => memoKey({ ...base, inputs: { f: 1n as unknown } })).toThrow();
  });

  it("drops undefined object properties per JSON semantics (JCS)", () => {
    expect(memoKey({ ...base, inputs: { a: 1, b: "x", gone: undefined as unknown } })).toBe(memoKey(base));
  });
});
