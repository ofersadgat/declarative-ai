/**
 * Reading a signature out of TypeScript, and the wire boundary it lands on (SPEC §7.5.2, §7.5.3).
 *
 * The contribution table's claim is that a parameter list already carries almost everything a
 * `ParameterDecl` holds — name, position, type, optionality, default, description — so a module
 * declares no signature in JSON and nothing can drift. These pin that, and pin the two ways the lossy
 * type conversion is allowed to fail:
 *
 *  - unrepresentable (`bigint`, `Map`, a function) is an ERROR, because a signature that does not
 *    describe the call is worse than one that will not compile;
 *  - unconstrained (`any`, `unknown`, an uninstantiated generic) is UNTYPED with a warning, because
 *    §6.2 already defines the universal schema and silence is what it argues against.
 */
import { describe, expect, it } from "vitest";
import { extractSignature, extractSignatureWith, loadSignatureContext } from "../src/signature.js";
import { requirePathFor } from "../src/moduleLoader.js";
import { marshalIn, marshalOut, needsMarshalling } from "../src/marshal.js";
import type { Vfs } from "../src/reference.js";

const FN = "/p/functions";

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (!rest.includes("/")) names.add(rest);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

/** Extract the default export of a one-file module. */
const signatureOf = (source: string, files: Record<string, string> = {}) =>
  extractSignature(`${FN}/f.ts`, ["default"], {
    vfs: vfsOf({ [`${FN}/f.ts`]: source, ...files }),
    requirePath: requirePathFor([FN]),
  });

describe("the parameter list is the signature", () => {
  it("reads name, position, type and optionality", async () => {
    const { parameters } = await signatureOf(
      "export default function (rank: number, label: string, loud?: boolean) { return 1; }",
    );
    expect(parameters).toMatchObject([
      { name: "rank", index: 0, schema: { type: "number" }, optional: false },
      { name: "label", index: 1, schema: { type: "string" }, optional: false },
      { name: "loud", index: 2, schema: { type: "boolean" }, optional: true },
    ]);
  });

  it("reads a parameter default, and marks the slot optional", async () => {
    const { parameters } = await signatureOf("export default function (limit = 3, tag = 'x') { return limit; }");
    expect(parameters).toMatchObject([
      { name: "limit", default: 3, optional: true },
      { name: "tag", default: "x", optional: true },
    ]);
  });

  it("reads array and object literal defaults", async () => {
    const { parameters } = await signatureOf(
      "export default function (xs: string[] = [], o: { a: number } = { a: 1 }) { return xs; }",
    );
    expect(parameters[0]?.default).toEqual([]);
    expect(parameters[1]?.default).toEqual({ a: 1 });
  });

  it("skips a default it cannot read, while still marking the slot optional", async () => {
    // A computed default is the function's business; reporting it as the slot's would claim a value
    // nobody can check.
    const { parameters } = await signatureOf("export default function (at = Date.now()) { return at; }");
    expect(parameters[0]?.optional).toBe(true);
    expect(parameters[0]).not.toHaveProperty("default");
  });

  it("reads a JSDoc @param as the slot's description", async () => {
    const source = "/**\n * @param rank blocker=3 … note=0.\n */\nexport default function (rank: number) { return rank; }";
    const { parameters } = await signatureOf(source);
    expect(parameters[0]?.description).toContain("blocker=3");
  });

  it("treats `T | undefined` as optional rather than as a union with null", async () => {
    const { parameters } = await signatureOf("export default function (x: string | undefined) { return x; }");
    expect(parameters[0]).toMatchObject({ optional: true, schema: { type: "string" } });
  });
});

describe("the return type", () => {
  it("becomes an object schema, closed by default", async () => {
    const source = "export default function (): { score: number; reasons: string[] } { return { score: 1, reasons: [] }; }";
    const { returns } = await signatureOf(source);
    expect(returns).toMatchObject({
      type: "object",
      properties: { score: { type: "number" }, reasons: { type: "array", items: { type: "string" } } },
      required: ["score", "reasons"],
      additionalProperties: false,
    });
  });

  it("unwraps a Promise — a function may be async, and the caller sees what it resolves with", async () => {
    const { returns } = await signatureOf("export default async function () { return 42; }");
    expect(returns).toMatchObject({ type: "number" });
  });

  it("resolves an imported interface, which a bare parse could not", async () => {
    const files = { [`${FN}/types.ts`]: "export interface Confidence { score: number }" };
    const source = "import type { Confidence } from './types.js';\nexport default function (): Confidence { return { score: 1 }; }";
    const { returns } = await signatureOf(source, files);
    expect(returns).toMatchObject({ type: "object", properties: { score: { type: "number" } } });
  });
});

describe("unions", () => {
  it("become an enum when every member is a literal of one type", async () => {
    const source = "export default function (): 'complete' | 'blocked' { return 'complete'; }";
    const { returns } = await signatureOf(source);
    expect(returns).toMatchObject({ type: "string", enum: ["complete", "blocked"] });
  });

  it("collapse `true | false` back to boolean", async () => {
    const { returns } = await signatureOf("export default function (): boolean { return true; }");
    expect(returns).toEqual({ type: "boolean" });
  });

  it("become anyOf when the members are of different kinds", async () => {
    const { returns } = await signatureOf("export default function (): string | number { return 1; }");
    expect(returns).toHaveProperty("anyOf");
  });
});

describe("unrepresentable types are errors", () => {
  const refuses = async (annotation: string, pattern: RegExp): Promise<void> => {
    await expect(signatureOf(`export default function (x: ${annotation}) { return 1; }`)).rejects.toThrow(pattern);
  };

  it("refuses bigint", async () => {
    await refuses("bigint", /bigint, which has no JSON form/);
  });

  it("refuses a Map, naming why", async () => {
    // Not "JSON Schema cannot express it" — it can, several ways. The problem is the wire has no
    // agreed encoding, and JSON.stringify(new Map()) is '{}'.
    await refuses("Map<string, number>", /Map, which has no JSON form/);
  });

  it("refuses a Set", async () => {
    await refuses("Set<string>", /Set, which has no JSON form/);
  });

  it("refuses a function type", async () => {
    await refuses("(n: number) => string", /function type, which has no JSON form/);
  });

  it("names the parameter in the message", async () => {
    await expect(signatureOf("export default function (whenDue: bigint) { return 1; }")).rejects.toThrow(
      /parameter 'whenDue'/,
    );
  });
});

describe("unconstrained types are untyped, loudly", () => {
  it("warns for `any` rather than accepting it in silence", async () => {
    const { parameters, warnings } = await signatureOf("export default function (x: any) { return x; }");
    expect(parameters[0]?.schema).toEqual({});
    expect(warnings.join("\n")).toMatch(/parameter 'x' is 'any', so it is untyped/);
  });

  it("warns for `unknown` too, and says which it was", async () => {
    const { warnings } = await signatureOf("export default function (x: unknown) { return 1; }");
    expect(warnings.join("\n")).toMatch(/'unknown'/);
  });

  it("warns for an uninstantiated generic — legal, and not usefully typed", async () => {
    const { parameters, warnings } = await signatureOf("export default function <T>(x: T): T { return x; }");
    expect(parameters[0]?.schema).toEqual({});
    expect(warnings.join("\n")).toMatch(/generic type parameter/);
  });
});

describe("Date is representable because a marshaller exists", () => {
  it("takes the date-time wire type", async () => {
    const { parameters } = await signatureOf("export default function (at: Date) { return at; }");
    expect(parameters[0]?.schema).toEqual({ type: "string", format: "date-time" });
  });

  it("needs no separate instruction nested in an array or an object", async () => {
    const source = "export default function (xs: Date[], o: { when: Date }) { return xs; }";
    const { parameters } = await signatureOf(source);
    expect(parameters[0]?.schema).toMatchObject({ type: "array", items: { format: "date-time" } });
    expect(parameters[1]?.schema).toMatchObject({ properties: { when: { format: "date-time" } } });
  });

  it("refuses a union that is indistinguishable on the wire", async () => {
    // `Date | string` is `{"type":"string","format":"date-time"}` against `{"type":"string"}` —
    // nothing at a leaf can decide which conversion applies.
    await expect(signatureOf("export default function (x: Date | string) { return x; }")).rejects.toThrow(
      /indistinguishable on the wire/,
    );
  });
});

describe("the AbortSignal carve-out", () => {
  it("is absent from the wire signature entirely", async () => {
    // SPEC §7.5.6: it carries no workflow data, so it is not a slot anybody could fill.
    const source = "export default function (n: number, signal: AbortSignal) { return n; }";
    const { parameters } = await signatureOf(source);
    expect(parameters.map((p) => p.name)).toEqual(["n"]);
  });

  it("is dropped when optional too", async () => {
    const source = "export default function (n: number, signal?: AbortSignal) { return n; }";
    const { parameters } = await signatureOf(source);
    expect(parameters.map((p) => p.name)).toEqual(["n"]);
  });
});

describe("shapes a callee cannot have", () => {
  it("refuses a rest parameter — there is no fixed slot to bind to", async () => {
    await expect(signatureOf("export default function (...xs: number[]) { return xs; }")).rejects.toThrow(
      /rest parameter/,
    );
  });

  it("refuses a destructured parameter — it names no single slot", async () => {
    await expect(signatureOf("export default function ({ a }: { a: number }) { return a; }")).rejects.toThrow(
      /destructured parameter/,
    );
  });

  it("refuses a non-callable export, which is data rather than an operation", async () => {
    await expect(signatureOf("export default { a: 1 };")).rejects.toThrow(/is not callable/);
  });
});

/**
 * The third way extraction can fail, and the only one that used to be SILENT.
 *
 * `createProgram` reads TypeScript's own `lib.*.d.ts` off the real disk, addressed only by
 * `ts.getDefaultLibFilePath()` — the directory of the `typescript` module. A host that bundles the
 * compiler moves that address to its own output directory, which holds no lib files, and the
 * program then has no global scope at all.
 *
 * Neither documented failure mode catches that. Nothing is unrepresentable, so no `WireTypeError`;
 * nothing widens to the universal schema, so no warning. `string[]` simply types as `{}` and comes
 * back as a CLOSED empty object — the most restrictive schema in the language, asserted with total
 * confidence, against which every real value fails §6.2. So the lib is checked, not assumed.
 */
describe("the standard library is a precondition, not a convenience", () => {
  /** The real context with its lib reader removed — exactly what a bundled compiler produces. */
  const withoutLib = async () => ({ ...(await loadSignatureContext()), readLib: () => undefined });

  const extract = (context: Awaited<ReturnType<typeof withoutLib>>, source: string) =>
    extractSignatureWith(context, `${FN}/f.ts`, ["default"], {
      vfs: vfsOf({ [`${FN}/f.ts`]: source }),
      requirePath: requirePathFor([FN]),
    });

  it("refuses to read a signature at all when no lib file can be reached", async () => {
    const context = await withoutLib();
    expect(() => extract(context, "export default function (): string[] { return []; }")).toThrow(
      /standard library could not be read/,
    );
  });

  it("names the directory it looked in, and the bundling that usually moved it", async () => {
    const context = await withoutLib();
    let message = "";
    try {
      extract(context, "export default function (): number { return 1; }");
    } catch (e) {
      message = (e as Error).message;
    }
    // An intrinsic return would have survived the missing lib, which is exactly why the check
    // cannot be "did this one type come out wrong" — the fault is the program, not the signature.
    expect(message).toMatch(/Cannot find global type 'Array'/);
    expect(message).toMatch(/getDefaultLibFilePath/);
    expect(message).toMatch(/keep 'typescript' external/);
  });

  it("still reads a signature whose FILE has an ordinary type error — the parameter list is intact", async () => {
    const source = `const bad: number = 'x' as unknown as number;
export default function (n: number): string[] { return [String(n), String(bad)]; }`;
    const { parameters, returns } = await signatureOf(source);
    expect(parameters).toMatchObject([{ name: "n", schema: { type: "number" } }]);
    expect(returns).toMatchObject({ type: "array", items: { type: "string" } });
  });

  it("still reads a signature whose IMPORT does not resolve, warning about the slot it could not type", async () => {
    const source = `import type { Thing } from './missing.js';
export default function (): Thing { return null as never; }`;
    const { returns, warnings } = await signatureOf(source);
    // Reported rather than refused: an unresolvable import widens ONE slot, and the warning names
    // it. That is the documented `any` path, not the silent one this block exists for.
    expect(returns).toEqual({});
    expect(warnings.join(" ")).toMatch(/untyped/);
  });
});

describe("marshalling values across the boundary", () => {
  const dateSchema = { type: "string", format: "date-time" } as const;

  it("does nothing at all when the schema names no marshalled type", () => {
    const schema = { type: "object", properties: { n: { type: "number" } } };
    expect(needsMarshalling(schema)).toBe(false);
    const value = { n: 1 };
    // The fast path returns the very same object rather than a walked copy.
    expect(marshalIn(value, schema)).toBe(value);
  });

  it("converts a leaf in both directions", () => {
    const iso = "2026-08-21T10:00:00.000Z";
    const asDate = marshalIn(iso, dateSchema) as Date;
    expect(asDate).toBeInstanceOf(Date);
    expect(asDate.toISOString()).toBe(iso);
    expect(marshalOut(asDate, dateSchema)).toBe(iso);
  });

  it("derives the traversal structurally — Date[] needs no instruction of its own", () => {
    const schema = { type: "array", items: dateSchema };
    const iso = "2026-08-21T10:00:00.000Z";
    const dates = marshalIn([iso, iso], schema) as Date[];
    expect(dates.every((d) => d instanceof Date)).toBe(true);
    expect(marshalOut(dates, schema)).toEqual([iso, iso]);
  });

  it("reaches a Date nested inside an object inside an array", () => {
    const schema = { type: "array", items: { type: "object", properties: { when: dateSchema } } };
    const iso = "2026-08-21T10:00:00.000Z";
    const value = marshalIn([{ when: iso }], schema) as { when: Date }[];
    expect(value[0]?.when).toBeInstanceOf(Date);
  });

  it("passes null through — absence is not a type question", () => {
    expect(marshalIn(null, dateSchema)).toBeNull();
    expect(marshalOut(null, dateSchema)).toBeNull();
  });

  it("leaves members the schema says nothing about alone", () => {
    const schema = { type: "object", properties: { when: dateSchema } };
    const out = marshalIn({ when: "2026-08-21T10:00:00.000Z", other: 5 }, schema) as Record<string, unknown>;
    expect(out.other).toBe(5);
  });

  it("reports a value that cannot cross", () => {
    expect(() => marshalIn("not a date", dateSchema)).toThrow(/not a valid ISO-8601/);
    expect(() => marshalOut("not a date", dateSchema)).toThrow(/expected a Date/);
  });
});
