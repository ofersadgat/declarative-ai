import { describe, expect, it } from "vitest";
import { inferFromValues, joinSchemas } from "../src/index";

describe("joinSchemas — unification is STRUCTURAL over `type` (§6)", () => {
  it("unifies two equal array-valued `type`s — legal JSON Schema, and reference-unequal", () => {
    // `ka === kb` compared two freshly-built arrays by REFERENCE, so a schema failed to unify with an
    // identical copy of itself.
    expect(joinSchemas({ type: ["string", "null"] }, { type: ["string", "null"] })).toEqual({ type: ["string", "null"] });
    expect(joinSchemas({ type: ["string", "null"] }, { type: ["null", "string"] })).toEqual({ type: ["string", "null"] });
  });

  it("lets the universal `{}` ABSORB — every value inhabits it, so it cannot conflict", () => {
    expect(joinSchemas({}, { type: "string" })).toEqual({});
    expect(joinSchemas({ type: "string" }, {})).toEqual({});
  });

  it("still unifies same-kind schemas to the broad kind and still reports a real conflict", () => {
    expect(joinSchemas({ type: "string" }, { type: "string" })).toEqual({ type: "string" });
    expect(() => joinSchemas({ type: "string" }, { type: "number" })).toThrow(/cannot unify/);
    expect(() => joinSchemas({ type: ["string", "null"] }, { type: "number" })).toThrow(/cannot unify/);
    expect(joinSchemas({ type: "string" }, { type: "number" }, "widen")).toEqual({});
  });
});

describe("inferFromValues (§6)", () => {
  it("widens a value of no inferable kind to `{}` instead of throwing", () => {
    // `inferValueSchema` returns `{}` for a value outside JSON's kinds; joining it used to conflict.
    expect(inferFromValues([() => 1, "a"])).toEqual({});
    expect(inferFromValues(["a", null, "b"])).toEqual({ type: "string" });
    expect(inferFromValues([null, undefined])).toBeUndefined();
  });
});
