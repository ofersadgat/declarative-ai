import { describe, expect, it } from "vitest";
import { EMBEDDED_CONFIG_SCHEMA, LOCAL_CONFIG_SCHEMA, configSchemaFor } from "../src/providerConfig.js";
import { MODEL_ROUTES } from "../src/router.js";

/** The declared properties of a config schema document. */
const propsOf = (schema: unknown): Record<string, unknown> =>
  ((schema as { properties?: Record<string, unknown> }).properties ?? {});

describe("provider config schemas (§4)", () => {
  it("every route has a config space — the registry is total over ModelRoute", () => {
    // This is the check the `Record<ModelFamily, JsonSchemaDoc>` type already enforces at COMPILE time;
    // asserting it at runtime is what catches a route added to the union with an `as never` cast or a
    // registry rebuilt dynamically. Adding a route should force a config space, not silently inherit one.
    for (const route of MODEL_ROUTES) {
      const schema = configSchemaFor(route);
      expect(schema, `no config schema for route "${route}"`).toBeDefined();
      expect((schema as { required?: string[] }).required).toContain("model");
    }
  });

  it("the locally-served spaces are CLOSED, so an unmodelled knob fails authoring rather than the call", () => {
    expect((LOCAL_CONFIG_SCHEMA as { additionalProperties?: boolean }).additionalProperties).toBe(false);
    expect((EMBEDDED_CONFIG_SCHEMA as { additionalProperties?: boolean }).additionalProperties).toBe(false);
  });

  it("embedded declares the three knobs that cost MEMORY rather than money", () => {
    // These are load-time parameters, and they belong in the config space because they are what makes
    // two calls against the same weights differ in resource terms — which is what a residency planner
    // has to read. `contextSize` and `sequences` both scale the KV cache.
    const props = propsOf(EMBEDDED_CONFIG_SCHEMA);
    expect(Object.keys(props)).toEqual(expect.arrayContaining(["contextSize", "gpuLayers", "sequences"]));
    // The local SERVER space has none of them: we do not own that process's memory.
    expect(Object.keys(propsOf(LOCAL_CONFIG_SCHEMA))).not.toEqual(expect.arrayContaining(["gpuLayers"]));
  });

  it("neither local space declares `reasoning` — nothing adapts it for these routes yet", () => {
    // `adaptReasoning` knows the Anthropic and OpenRouter shapes only. Declaring the neutral spec here
    // would let an author request thinking that silently translates to nothing on the wire.
    expect(propsOf(LOCAL_CONFIG_SCHEMA)).not.toHaveProperty("reasoning");
    expect(propsOf(EMBEDDED_CONFIG_SCHEMA)).not.toHaveProperty("reasoning");
    expect(propsOf(configSchemaFor("anthropic"))).toHaveProperty("reasoning");
  });
});
