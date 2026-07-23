import { describe, expect, it } from "vitest";
import type { Serializable } from "@declarative-ai/json";
import { canonicalize, hashCanonical } from "../src/hashing";

/** A class with `#private` fields behind getters — an ordinary TS idiom, and the exact shape the
 *  `SerializableFields<T>` bound ACCEPTS (its declared fields are `number`/`string`) while JCS reads
 *  only own enumerable properties and sees `{}`. No cast below: `tsc` reports nothing here. */
class Money {
  readonly #amount: number;
  readonly #currency: string;
  constructor(amount: number, currency: string) {
    this.#amount = amount;
    this.#currency = currency;
  }
  get amount(): number {
    return this.#amount;
  }
  get currency(): string {
    return this.#currency;
  }
}

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
    // A function is not `Serializable` by type — smuggled past the signature to prove the
    // runtime still refuses it rather than hashing nothing.
    expect(() => canonicalize((() => 1) as unknown as Serializable)).toThrow();
  });

  it("REFUSES a class instance the type bound admits — its own properties are empty, so every instance would hash alike", () => {
    // Left unchecked this is a memo-key COLLISION: `hashCanonical` would return sha256("{}") for both
    // instances and for `{}`, so `(operation, inputs)` yields a WRONG CACHED VALUE, not a miss.
    expect(() => hashCanonical(new Money(100, "USD"))).toThrow(/not a plain JSON object/);
    expect(() => hashCanonical(new Money(999_999, "JPY"))).toThrow(/not a plain JSON object/);
    // Nested too — `tsc` catches this one, so the cast; the runtime must not depend on that.
    expect(() => canonicalize({ price: new Money(1, "USD") } as unknown as Serializable)).toThrow(/not a plain JSON object/);
  });

  it("accepts a class that declares toJSON() — the opt-in JCS actually honors", () => {
    class Stamp {
      constructor(private readonly at: number) {}
      toJSON(): { at: number } {
        return { at: this.at };
      }
    }
    // A method is not `Serializable` by type, hence the cast — the point is the RUNTIME accepts it.
    expect(canonicalize(new Stamp(5) as unknown as Serializable)).toBe('{"at":5}');
  });

  it("throws on a nested function instead of emitting invalid JSON", () => {
    // The JSDoc promises a throw; JCS instead produced the literal `{"a":1,"f":undefined}`.
    expect(() => canonicalize({ a: 1, f: () => 1 } as unknown as Serializable)).toThrow(/function/);
  });

  it("throws on a cycle instead of exhausting the stack", () => {
    const cyclic: Record<string, unknown> = { a: 1 };
    cyclic.self = cyclic;
    expect(() => canonicalize(cyclic as unknown as Serializable)).toThrow(/circular/);
  });

  it("still hashes an ALIASED (non-cyclic) node — the guard is path-scoped, not seen-once", () => {
    const shared = { a: 1 };
    expect(canonicalize({ x: shared, y: shared })).toBe('{"x":{"a":1},"y":{"a":1}}');
  });
});
