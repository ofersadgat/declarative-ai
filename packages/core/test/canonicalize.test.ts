import { describe, expect, it } from "vitest";
import { canonicalize } from "../src/hashing";

describe("canonicalize (RFC 8785 JCS)", () => {
  it("is independent of object key insertion order", () => {
    expect(canonicalize({ a: 1, b: 2 })).toBe(canonicalize({ b: 2, a: 1 }));
  });

  it("sorts keys recursively", () => {
    const a = canonicalize({ outer: { z: 1, a: 2 }, list: [{ y: 1, x: 2 }] });
    const b = canonicalize({ list: [{ x: 2, y: 1 }], outer: { a: 2, z: 1 } });
    expect(a).toBe(b);
  });

  it("preserves array order (arrays are significant)", () => {
    expect(canonicalize([1, 2, 3])).not.toBe(canonicalize([3, 2, 1]));
  });

  it("emits no insignificant whitespace", () => {
    expect(canonicalize({ a: 1, b: [2, 3] })).toBe('{"a":1,"b":[2,3]}');
  });

  it("throws on non-serializable values rather than hashing nothing", () => {
    expect(() => canonicalize(undefined)).toThrow();
    expect(() => canonicalize(() => 1)).toThrow();
  });
});
