import { describe, expect, it } from "vitest";
import type { SchemaDocument } from "@declarative-ai/json";
import { SchemaValidator } from "../src/ajv";

describe("SchemaValidator", () => {
  it("validates inline schemas synchronously and caches by content hash", () => {
    const v = new SchemaValidator();
    const schema = {
      type: "object",
      properties: { outcome: { type: "string", enum: ["clean", "needs_changes"] } },
      required: ["outcome"],
    } as SchemaDocument;
    expect(v.validateValue(schema, { outcome: "clean" }).ok).toBe(true);
    const bad = v.validateValue(schema, { outcome: "nope" });
    expect(bad.ok).toBe(false);
    expect(bad.errors).toMatch(/allowed values/);
    // Key-order variant hits the same cached validator (content-hash keyed).
    const reordered = { required: ["outcome"], properties: { outcome: { enum: ["clean", "needs_changes"], type: "string" } }, type: "object" };
    expect(v.validateValue(reordered as SchemaDocument, { outcome: "clean" }).ok).toBe(true);
  });

  it("resolves store-backed $ref graphs through the injected resolver", async () => {
    const leaf = { type: "string", minLength: 2 };
    const root = { type: "object", properties: { name: { $ref: "schema:leaf" } }, required: ["name"] };
    const v = new SchemaValidator({
      getSchema: async (id: string) => (id === "schema:leaf" ? leaf : id === "schema:root" ? root : undefined),
    });
    expect((await v.validate("schema:root", { name: "ok" })).ok).toBe(true);
    expect((await v.validate("schema:root", { name: "x" })).ok).toBe(false);
    await expect(v.validate("schema:missing", {})).rejects.toThrow(/not found/);
  });
});

describe("asBoundaryValidator — the maybe-async boundary lift", () => {
  it("resolves store-id $refs through the resolver (async path) and validates against the closure", async () => {
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv");
    const stored: Record<string, object> = {
      "json:leaf": { type: "object", properties: { n: { type: "number" } }, required: ["n"] },
    };
    const v = new SchemaValidator({ getSchema: async (id) => stored[id] as never });
    const boundary = asBoundaryValidator(v);
    const schema = { type: "object", properties: { child: { $ref: "json:leaf" } }, required: ["child"] } as never;
    expect(await boundary.validateValue(schema, { child: { n: 1 } })).toEqual({ ok: true });
    const bad = await boundary.validateValue(schema, { child: { n: "x" } });
    expect(bad.ok).toBe(false);
    expect(bad.errors).toMatch(/number/);
  });

  it("a ref-free document answers synchronously through the SYNC seam", async () => {
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv");
    const boundary = asBoundaryValidator(new SchemaValidator());
    const res = boundary.validateValue({ type: "string" } as never, "hi");
    expect(res).toEqual({ ok: true }); // NOT a promise — the inline family's truth
  });
});
