/**
 * Embedded bodies: statement list or single expression (SPEC §7.5.1).
 *
 * The rule is the arrow function's — `x => f(x)` returns, `x => { return f(x); }` returns, and
 * `x => { f(x); }` does not. These pin that all three spellings of one computation are one function,
 * that a body opening with `{` is a block exactly as JavaScript reads it, and that the wrapper binds
 * its parameters in the same order a CALL binds its arguments.
 */
import { describe, expect, it } from "vitest";
import { classifyBody, FunctionBodyError, synthesizeBody } from "../src/functionBody.js";
import type { ParameterDecl } from "../src/format.js";

const slots = (...names: string[]): Record<string, ParameterDecl> =>
  Object.fromEntries(names.map((n) => [n, {} as ParameterDecl]));

describe("the three spellings are one function", () => {
  const body = "Math.max(0, 1 - 0.35 * severity)";

  it("treats a bare expression as an expression", async () => {
    expect(await classifyBody(body, "f")).toBe("expression");
  });

  it("treats a trailing semicolon as deciding nothing", async () => {
    expect(await classifyBody(`${body};`, "f")).toBe("expression");
  });

  it("treats an explicit return as a statement list", async () => {
    expect(await classifyBody(`return ${body};`, "f")).toBe("statements");
  });

  it("compiles the expression form into a return, so one path exists downstream", async () => {
    const synthesized = await synthesizeBody("score", body, slots("severity"));
    expect(synthesized.form).toBe("expression");
    expect(synthesized.source).toContain("return (");
    expect(synthesized.source).toContain(body);
    expect(synthesized.source.startsWith("export default function (severity) {")).toBe(true);
  });

  it("splices a statement body through unchanged", async () => {
    const synthesized = await synthesizeBody("score", `return ${body};`, slots("severity"));
    expect(synthesized.form).toBe("statements");
    expect(synthesized.source).toContain(`return ${body};`);
    expect(synthesized.source).not.toContain("return (");
  });
});

describe("the brace ambiguity", () => {
  it("reads a body opening with '{' as a block, and says so", async () => {
    // JavaScript's own rule. The author wrote a record and got a labelled statement, so a message
    // about unreachable code or a missing return would send them looking anywhere but here.
    await expect(classifyBody("{ score: s, reasons: r }", "f")).rejects.toThrow(FunctionBodyError);
    await expect(classifyBody("{ score: s, reasons: r }", "f")).rejects.toThrow(/parenthesise it/);
  });

  it("accepts the parenthesised record as an expression", async () => {
    expect(await classifyBody("({ score: s, reasons: r })", "f")).toBe("expression");
  });

  it("accepts an explicit return of a record", async () => {
    expect(await classifyBody("return { score: s };", "f")).toBe("statements");
  });
});

describe("a statement body must return", () => {
  it("refuses one that never does", async () => {
    await expect(classifyBody("const x = 1; console.log(x);", "f")).rejects.toThrow(/must 'return'/);
  });

  it("accepts a return nested in a conditional", async () => {
    // "Is there a return at all", not "does every path return" — an `if` with no `else` is ordinary.
    expect(await classifyBody("if (a) { return 1; }", "f")).toBe("statements");
  });

  it("accepts a return inside a loop or a try", async () => {
    expect(await classifyBody("for (const x of xs) { return x; }", "f")).toBe("statements");
    expect(await classifyBody("try { return 1; } catch (e) { }", "f")).toBe("statements");
  });

  it("does NOT count a return that belongs to a nested function", async () => {
    // Each return belongs to the callback or the declaration it sits in; the body falls off the end.
    await expect(classifyBody("const f = (x) => { return x * 2; };", "f")).rejects.toThrow(/must 'return'/);
    await expect(classifyBody("function inner() { return 1; }", "f")).rejects.toThrow(/must 'return'/);
  });

  it("leaves a single call expression as the EXPRESSION form, callback and all", async () => {
    // `xs.map(x => { … })` is one expression, so it returns implicitly — the nested return belongs to
    // the callback, and neither fact is in tension with the other.
    expect(await classifyBody("xs.map(x => { return x * 2; })", "f")).toBe("expression");
  });

  it("counts a return that follows a nested function", async () => {
    expect(await classifyBody("function inner() { return 1; }\nreturn inner();", "f")).toBe("statements");
  });
});

describe("syntax errors", () => {
  it("are reported against the authored text, not against generated code", async () => {
    await expect(classifyBody("const x = ;", "confidence")).rejects.toThrow(/confidence: body does not parse/);
  });

  it("carry the parser's own message and an offset", async () => {
    await expect(classifyBody("const x = ;", "f")).rejects.toThrow(/offset \d+ — Expression expected/);
  });
});

describe("the wrapper's parameters", () => {
  it("follow declaration order when no index is declared", async () => {
    const { parameters } = await synthesizeBody("f", "a + b + c", slots("a", "b", "c"));
    expect(parameters).toEqual(["a", "b", "c"]);
  });

  it("follow the declared index when there is one — the same rule a CALL binds by", async () => {
    const input: Record<string, ParameterDecl> = {
      last: { index: 2 } as ParameterDecl,
      first: { index: 0 } as ParameterDecl,
      middle: { index: 1 } as ParameterDecl,
    };
    const { parameters, source } = await synthesizeBody("f", "first + middle + last", input);
    expect(parameters).toEqual(["first", "middle", "last"]);
    expect(source).toContain("function (first, middle, last)");
  });

  it("refuse a slot name that is not a legal identifier", async () => {
    // A slot name is a JSON key and may be anything; a parameter name may not. Better here than as
    // source that will not parse for a reason the author cannot see.
    await expect(synthesizeBody("f", "1", slots("not-an-identifier"))).rejects.toThrow(/cannot be a parameter name/);
  });

  it("produce a zero-parameter wrapper for a body that takes nothing", async () => {
    const { source } = await synthesizeBody("f", "42", {});
    expect(source).toContain("export default function () {");
  });
});

describe("TypeScript in a body", () => {
  it("is parsed without a type checker", async () => {
    expect(await classifyBody("const n: number = severity; return n * 2;", "f")).toBe("statements");
  });

  it("survives into the generated source for the compiler to handle", async () => {
    const { source } = await synthesizeBody("f", "const n: number = s; return n;", slots("s"));
    expect(source).toContain("const n: number = s;");
  });
});
