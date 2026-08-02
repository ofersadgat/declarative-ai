/**
 * Failures in the DATA plane (JaiRA EXPRESSIONS.md §5) — the routing predicate.
 *
 * A failure is a value with a TYPE, so routing is ordinary type checking against what the consumer
 * declared; terminating is the implicit unwrap. The value is WRAPPED (`{ error: … }`), which is what
 * lets a slot say "a value, or an error" as a union — and what stops ordinary data that happens to
 * look like a failure from being read as one.
 */
import { describe, expect, it } from "vitest";
import { ERROR_VALUE_SCHEMA, FAILURE_SCHEMA } from "@declarative-ai/json";
import { admitsError, isErrorValue, resolutionFailure } from "../src/errorValue.js";

/** "Any value, or any error" — the union the wrapper makes writable. */
const ANY_OR_ERROR = { anyOf: [{}, ERROR_VALUE_SCHEMA] };

describe("isErrorValue", () => {
  it("recognizes a wrapped classified failure", () => {
    expect(isErrorValue(resolutionFailure("child 'c' has not run"))).toBe(true);
    expect(isErrorValue({ error: { classification: "policy-denied", reason: "blocked" } })).toBe(true);
  });

  /**
   * The reason the value is wrapped at all: a classifier operation returning `{classification,
   * reason}` is an entirely plausible thing for a workflow to have, and sniffing those fields would
   * read its output as an error.
   */
  it("does not mistake ordinary data for a failure", () => {
    expect(isErrorValue({ classification: "permanent", reason: "a classifier said so" })).toBe(false);
    expect(isErrorValue({ error: "a string" })).toBe(false);
    expect(isErrorValue({ error: { classification: "made-up", reason: "x" } })).toBe(false);
    expect(isErrorValue({ error: { classification: "permanent" } })).toBe(false); // no reason
    expect(isErrorValue(null)).toBe(false);
  });
});

describe("admitsError", () => {
  it("accepts a slot that declares the error branch", () => {
    expect(admitsError(ERROR_VALUE_SCHEMA as never)).toBe(true);
    expect(admitsError({ anyOf: [{ type: "string" }, ERROR_VALUE_SCHEMA] } as never)).toBe(true);
    expect(admitsError(ANY_OR_ERROR as never)).toBe(true);
  });

  /**
   * JSON Schema objects are OPEN, so an ordinary object slot validates `{error: …}` quite happily —
   * it never said its own properties were required. Acceptance therefore has to be DECLARED, not
   * merely survivable, or failures would route into every slot nobody thought about.
   */
  it("refuses an ordinary slot that would merely validate one", () => {
    expect(admitsError({ type: "string" })).toBe(false);
    expect(admitsError({ type: "object", properties: { plan: { type: "string" } } })).toBe(false);
    expect(admitsError({ anyOf: [{ type: "string" }, { type: "number" }] } as never)).toBe(false);
  });

  /**
   * No special case any more. `{}` has no branch requiring `error`, so it declares none — the
   * ordinary reading of what the author wrote, rather than an exception carved out of the rule.
   * Saying "any value OR an error" is what `ANY_OR_ERROR` spells.
   */
  it("treats the unconstrained slot as declaring no error, with the union as the way to say otherwise", () => {
    expect(admitsError(undefined)).toBe(false);
    expect(admitsError({})).toBe(false);
    expect(admitsError(ANY_OR_ERROR as never)).toBe(true);
  });

  /** Failures differ in SHAPE by kind, so a slot handling one kind must not swallow another. */
  it("distinguishes failures by shape", () => {
    const denial = {
      type: "object",
      properties: {
        error: {
          type: "object",
          properties: { classification: { type: "string", enum: ["policy-denied"] }, reason: { type: "string" } },
          required: ["classification", "reason"],
        },
      },
      required: ["error"],
    };
    expect(admitsError(denial as never, denial as never)).toBe(true);
    // The narrow slot refuses "any classified failure" — it could be a timeout it has no branch for.
    expect(admitsError(denial as never)).toBe(false);
    // A slot declaring the general shape accepts the specific one.
    expect(admitsError(ERROR_VALUE_SCHEMA as never, denial as never)).toBe(true);
  });

  it("terminates rather than chasing a pathological schema", () => {
    const deep: Record<string, unknown> = { anyOf: [] };
    let node = deep;
    for (let i = 0; i < 40; i++) {
      const next: Record<string, unknown> = { anyOf: [] };
      (node.anyOf as unknown[]).push(next);
      node = next;
    }
    (node.anyOf as unknown[]).push(ERROR_VALUE_SCHEMA);
    expect(admitsError(deep as never)).toBe(false);
  });

  it("keeps FAILURE_SCHEMA as the failure's own shape, not the value's", () => {
    // A slot declaring the bare failure does NOT accept the wrapped value — the wrapper is the
    // discriminator, so it has to be present on both sides.
    expect(admitsError(FAILURE_SCHEMA as never)).toBe(false);
  });
});
