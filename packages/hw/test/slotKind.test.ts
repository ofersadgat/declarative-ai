/**
 * Slot-kind derivation in the loader.
 *
 * `kindFor` (`@declarative-ai/ops`) documents itself as "the one place the blob/text/json split is
 * decided, so a slot authored as `{ type: "string", contentMediaType: "image/png" }` and one authored
 * as `{ type: "string" }` cannot drift apart between the loader, the checker, and the engine."
 *
 * The loader had its own copy of that rule which never derived `blob`, so an artifact slot authored
 * with JSON Schema's OWN content keywords — the thing §7 replaced the bespoke `x-artifact` marker
 * with — silently loaded as `text`. The engine gates artifact registration on `kind === "blob"`, so
 * the artifact was never registered and the content passed through as a bare string. Nothing errored.
 * The fixtures hid it by always spelling out `kind: "blob"` explicitly.
 */
import { describe, expect, it } from "vitest";
import { kindFor } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader";
import type { StateDef } from "../src/format";

const load = (outputs: Record<string, unknown>) =>
  loadBundle(
    {
      s: {
        label: "S",
        inputs: {},
        outputs: outputs as StateDef["outputs"],
        operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "m" } },
      } as StateDef,
    },
    "s",
  ).states["s"]!;

describe("the loader derives slot kinds through the SAME rule as everything else", () => {
  it("derives `blob` from contentMediaType alone — no explicit kind, no bespoke marker", () => {
    const state = load({ doc: { schema: { type: "string", contentMediaType: "markdown" } } });
    expect(state.outputs!["doc"]!.kind).toBe("blob");
  });

  it("derives `blob` from contentEncoding alone", () => {
    const state = load({ img: { schema: { type: "string", contentEncoding: "base64" } } });
    expect(state.outputs!["img"]!.kind).toBe("blob");
  });

  it("still derives `text` for a plain string slot", () => {
    expect(load({ note: { schema: { type: "string" } } }).outputs!["note"]!.kind).toBe("text");
  });

  it("still derives `json` for everything else", () => {
    expect(load({ n: { schema: { type: "number" } } }).outputs!["n"]!.kind).toBe("json");
    expect(load({ o: { schema: { type: "object" } } }).outputs!["o"]!.kind).toBe("json");
    expect(load({ bare: {} }).outputs!["bare"]!.kind).toBe("json");
  });

  it("lets an explicit authored kind win over derivation", () => {
    expect(load({ x: { kind: "json", schema: { type: "string", contentMediaType: "markdown" } } }).outputs!["x"]!.kind).toBe("json");
  });

  it("agrees with `kindFor` on every shape — that agreement IS the invariant", () => {
    const schemas: Record<string, string>[] = [
      { type: "string" },
      { type: "string", contentMediaType: "image/png" },
      { type: "string", contentEncoding: "base64" },
      { type: "string", contentEncoding: "base64", contentMediaType: "image/png" },
      { type: "number" },
      { type: "object" },
      { type: "array" },
      {},
    ];
    for (const schema of schemas) {
      expect(load({ slot: { schema } }).outputs!["slot"]!.kind).toBe(kindFor(schema));
    }
  });

  it("derives INPUT slot kinds by the same rule", () => {
    const state = loadBundle(
      {
        s: {
          label: "S",
          inputs: { issue: { schema: { type: "string", contentMediaType: "markdown" } }, note: { schema: { type: "string" } } },
          outputs: {},
          operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "m" } },
        } as StateDef,
      },
      "s",
    ).states["s"]!;
    expect(state.inputs!["issue"]!.kind).toBe("blob");
    expect(state.inputs!["note"]!.kind).toBe("text");
  });
});
