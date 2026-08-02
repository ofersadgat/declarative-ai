import { describe, expect, it } from "vitest";
import type { SchemaDocument } from "@declarative-ai/json";
import { SchemaValidator } from "../src/ajv.js";

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
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv.js");
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
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv.js");
    const boundary = asBoundaryValidator(new SchemaValidator());
    const res = boundary.validateValue({ type: "string" } as never, "hi");
    expect(res).toEqual({ ok: true }); // NOT a promise — the inline family's truth
  });

  it("getId routes EVERY document through the store-backed path under the caller's minted id", async () => {
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv.js");
    const stored: Record<string, object> = {
      "json:leaf": { type: "string", minLength: 2 },
    };
    // The caller's mint: a content-addressed id the resolver could also serve (findmyprompt: makeJson).
    const minted = new Map<object, string>();
    let seq = 0;
    const idOf = (schema: object) => {
      let id = minted.get(schema);
      if (!id) minted.set(schema, (id = `json:minted-${++seq}`));
      return id;
    };
    const v = new SchemaValidator({ getSchema: async (id) => stored[id] as never }, { getId: idOf as never });
    const boundary = asBoundaryValidator(v);

    // Ref-ful document: $refs resolve from the store; registration id is the caller's mint.
    const withRef = { type: "object", properties: { name: { $ref: "json:leaf" } }, required: ["name"] } as never;
    expect(await boundary.validateValue(withRef, { name: "ok" })).toEqual({ ok: true });
    expect((await boundary.validateValue(withRef, { name: "x" })).ok).toBe(false);

    // Ref-free document: ALSO the store-backed path (still awaited), so the boundary and
    // `validate(id, …)` entry points share one compiled cache under the same id.
    const refFree = { type: "number" } as never;
    expect(await boundary.validateValue(refFree, 5)).toEqual({ ok: true });
    const fnViaBoundary = await v.compile(idOf(refFree));
    const fnViaStore = await v.compile(idOf(refFree));
    expect(fnViaBoundary).toBe(fnViaStore); // one namespace, one cache entry
  });

  it("forJsonStore wires a content-addressed json store without a subclass", async () => {
    const { SchemaValidator, asBoundaryValidator } = await import("../src/ajv.js");
    // A store serving `{ json }` artifacts by id — the findmyprompt ArtifactStore surface.
    const artifacts: Record<string, { json: unknown }> = {
      "json:leaf": { json: { type: "string", minLength: 2 } },
    };
    const v = SchemaValidator.forJsonStore(
      { getJson: async (id) => artifacts[id] },
      (schema) => "json:mint-" + JSON.stringify(schema).length, // stand-in mint
    );
    const boundary = asBoundaryValidator(v);
    const schema = { type: "object", properties: { name: { $ref: "json:leaf" } }, required: ["name"] } as never;
    expect(await boundary.validateValue(schema, { name: "ok" })).toEqual({ ok: true });
    expect((await boundary.validateValue(schema, { name: "x" })).ok).toBe(false);
    // Store-backed validate() also works through the same adapter.
    expect((await v.validate("json:leaf", "ok")).ok).toBe(true);
  });
});
