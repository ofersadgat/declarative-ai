import { describe, expect, it } from "vitest";
import { InMemoryOpMetadata } from "../src/metadata";

describe("InMemoryOpMetadata", () => {
  it("annotates by content id", () => {
    const m = new InMemoryOpMetadata();
    m.set("op:abc", "capabilities", { memoizable: false });
    expect(m.get("op:abc", "capabilities")).toEqual({ memoizable: false });
    expect(m.get("op:abc", "other")).toBeUndefined();
    expect(m.get("op:def", "capabilities")).toBeUndefined();
  });

  it("annotates by inline op object identity", () => {
    const m = new InMemoryOpMetadata();
    const op1 = { kind: "function", functionRef: "f", input: {}, output: { name: "out", kind: "json" } };
    const op2 = { ...op1 }; // structurally equal, different identity
    m.set(op1, "inferredSchema", { type: "number" });
    expect(m.get(op1, "inferredSchema")).toEqual({ type: "number" });
    expect(m.get(op2, "inferredSchema")).toBeUndefined();
  });

  it("set overwrites per key without touching other keys", () => {
    const m = new InMemoryOpMetadata();
    m.set("op:x", "a", 1);
    m.set("op:x", "b", 2);
    m.set("op:x", "a", 3);
    expect(m.get("op:x", "a")).toBe(3);
    expect(m.get("op:x", "b")).toBe(2);
  });
});
