import { describe, expect, it } from "vitest";
import {
  isReasoningConfig,
  LlmConfigParseError,
  MapConfigurationRegistry,
  parseLlmConfig,
  parseReasoningSpec,
  resolveConfig,
  type ConfigLayer,
  type ReasoningConfiguration,
  type SamplingConfiguration,
} from "../src/llmConfig";

/** `ConfigLayer` now types the knobs, so a malformed layer is a COMPILE error too. These negative tests
 *  deliberately smuggle one past the signature to prove the runtime parse still refuses it. */
const badLayer = (layer: Record<string, unknown>): ConfigLayer => layer as ConfigLayer;

describe("parseLlmConfig — parse, don't validate", () => {
  it("parses a sampling config (no reasoning) into a SamplingConfiguration", () => {
    const cfg = parseLlmConfig({
      model: "claude-haiku-4-5",
      temperature: 0.7,
      topP: 0.9,
      maxOutputTokens: 512,
      stopSequences: ["\n\n"],
      seed: 42,
    });
    expect(isReasoningConfig(cfg)).toBe(false);
    const s = cfg as SamplingConfiguration;
    expect(s).toMatchObject({ model: "claude-haiku-4-5", temperature: 0.7, topP: 0.9, seed: 42 });
  });

  it("parses a reasoning config into a ReasoningConfiguration", () => {
    const cfg = parseLlmConfig({ model: "o3", reasoning: { effort: "high" }, maxOutputTokens: 1000 });
    expect(isReasoningConfig(cfg)).toBe(true);
    expect((cfg as ReasoningConfiguration).reasoning).toEqual({ effort: "high" });
  });

  it("carries providerOptions through verbatim", () => {
    const cfg = parseLlmConfig({
      model: "claude-haiku-4-5",
      providerOptions: { anthropic: { cacheControl: { type: "ephemeral" } } },
    });
    expect(cfg.providerOptions).toEqual({ anthropic: { cacheControl: { type: "ephemeral" } } });
  });

  it("THROWS on a present-but-non-number field (no best-effort coercion)", () => {
    expect(() => parseLlmConfig({ model: "m", temperature: "0.7" })).toThrow(LlmConfigParseError);
    expect(() => parseLlmConfig({ model: "m", temperature: "0.7" })).toThrow(/temperature must be a finite number/);
    expect(() => parseLlmConfig({ model: "m", seed: Number.NaN })).toThrow(/seed must be a finite number/);
  });

  it("THROWS when a required field is missing or wrong-typed", () => {
    expect(() => parseLlmConfig({ temperature: 0.5 })).toThrow(/model must be a non-empty string/);
    expect(() => parseLlmConfig({ model: "" })).toThrow(/model must be a non-empty string/);
    expect(() => parseLlmConfig(null)).toThrow(/config must be an object/);
  });

  it("THROWS when sampling knobs and reasoning are combined (illegal state)", () => {
    expect(() => parseLlmConfig({ model: "m", reasoning: { effort: "low" }, temperature: 0.5 })).toThrow(
      /cannot also set sampling knobs \(temperature\)/,
    );
  });

  it("THROWS on a malformed stopSequences / providerOptions", () => {
    expect(() => parseLlmConfig({ model: "m", stopSequences: "nope" })).toThrow(/stopSequences must be a string\[\]/);
    expect(() => parseLlmConfig({ model: "m", providerOptions: { anthropic: "nope" } })).toThrow(
      /providerOptions.anthropic must be an object/,
    );
  });

  it("parses outputModalities as a string[] (throws on a non-array)", () => {
    expect(parseLlmConfig({ model: "m", outputModalities: ["text", "image"] }).outputModalities).toEqual(["text", "image"]);
    expect(() => parseLlmConfig({ model: "m", outputModalities: "image" })).toThrow(/outputModalities must be a string\[\]/);
  });

  it("parses session ids and throws on a non-string", () => {
    expect(parseLlmConfig({ model: "m", sessionId: "thread-1" }).sessionId).toBe("thread-1");
    expect(parseLlmConfig({ model: "m", providerSessionId: "srv-abc" }).providerSessionId).toBe("srv-abc");
    expect(() => parseLlmConfig({ model: "m", sessionId: 42 })).toThrow(/sessionId must be a non-empty string/);
  });

  it("omits absent optional fields rather than defaulting them", () => {
    const cfg = parseLlmConfig({ model: "m" }) as SamplingConfiguration;
    expect(cfg.temperature).toBeUndefined();
    expect(cfg.maxOutputTokens).toBeUndefined();
  });

  it("THROWS on an unknown key — never a silent drop", () => {
    expect(() => parseLlmConfig({ model: "m", temprature: 0.7 })).toThrow(/unknown config key\(s\): temprature/);
  });

  it("names SIGNATURE fields in the unknown-key error (they belong to resolveConfig)", () => {
    expect(() => parseLlmConfig({ model: "m", system: "be terse" })).toThrow(/SIGNATURE fields/);
  });
});

describe("parseLlmConfig — tools", () => {
  it("parses function + provider tool declarations and toolChoice", () => {
    const cfg = parseLlmConfig({
      model: "claude-haiku-4-5",
      tools: [
        { name: "get_weather", description: "look up weather", inputSchema: { type: "object", properties: { city: { type: "string" } } } },
        { type: "provider", name: "search", id: "anthropic.web_search", args: { maxUses: 3 } },
      ],
      toolChoice: { type: "tool", toolName: "get_weather" },
      maxSteps: 5,
    });
    expect(cfg.tools).toHaveLength(2);
    expect(cfg.tools?.[0]).toMatchObject({ type: "function", name: "get_weather", description: "look up weather" });
    expect(cfg.tools?.[1]).toMatchObject({ type: "provider", id: "anthropic.web_search", args: { maxUses: 3 } });
    expect(cfg.toolChoice).toEqual({ type: "tool", toolName: "get_weather" });
    expect(cfg.maxSteps).toBe(5);
  });

  it("accepts the literal toolChoice values", () => {
    for (const c of ["auto", "none", "required"] as const) {
      expect(parseLlmConfig({ model: "m", toolChoice: c }).toolChoice).toBe(c);
    }
  });

  it("THROWS on a malformed tool declaration", () => {
    expect(() => parseLlmConfig({ model: "m", tools: [{ name: "t" }] })).toThrow(/tools\[0\].inputSchema must be an object/);
    expect(() => parseLlmConfig({ model: "m", tools: [{ inputSchema: {} }] })).toThrow(/tools\[0\].name must be a non-empty string/);
    expect(() => parseLlmConfig({ model: "m", tools: "nope" })).toThrow(/tools must be an array/);
    expect(() => parseLlmConfig({ model: "m", tools: [{ type: "provider", name: "s", id: "bad", args: {} }] })).toThrow(
      /tools\[0\].id must be formatted/,
    );
  });

  it("THROWS on a malformed toolChoice", () => {
    expect(() => parseLlmConfig({ model: "m", toolChoice: "sometimes" })).toThrow(/toolChoice must be/);
    expect(() => parseLlmConfig({ model: "m", toolChoice: { type: "tool" } })).toThrow(/toolChoice must be/);
  });
});

describe("resolveConfig — compose fragments (parse, don't validate)", () => {
  it("merges layers low→high (later wins per key), then parses", () => {
    const { definition: config, warnings } = resolveConfig([
      { model: "m", temperature: 0.5, maxOutputTokens: 100 },
      { temperature: 0.9, topP: 0.8 },
    ]);
    expect(config).toMatchObject({ model: "m", temperature: 0.9, topP: 0.8, maxOutputTokens: 100 });
    expect(warnings).toEqual([]);
  });

  it("an explicit `undefined` says NOTHING about a key — it never erases a lower layer's value", () => {
    // The common shape: a preset built by spreading a partially-filled record carries an `undefined` for
    // every field it did not fill. `Object.assign` copied those over the layer below and erased it —
    // silently, because `introducesSampling` also tests `!== undefined`, so no warning fired either.
    const partial: ConfigLayer = { temperature: undefined, topP: undefined };
    const { definition: config, warnings } = resolveConfig([{ model: "m", temperature: 0.5, maxOutputTokens: 100 }, partial]);
    expect((config as SamplingConfiguration).temperature).toBe(0.5);
    expect(config.maxOutputTokens).toBe(100);
    expect(warnings).toEqual([]);
  });

  it("…including `model`, which an erasing merge would turn into a parse failure", () => {
    const { definition } = resolveConfig([{ model: "m" }, { model: undefined as unknown as string }]);
    expect(definition.model).toBe("m");
  });

  it("resolves a registry preset + inline overrides", () => {
    const registry = new MapConfigurationRegistry().set("fast", { model: "m", temperature: 0.2, maxOutputTokens: 256 });
    const { definition: config } = resolveConfig([registry.get("fast"), { temperature: 0.7 }]);
    expect(config).toMatchObject({ model: "m", temperature: 0.7, maxOutputTokens: 256 });
  });

  it("family-aware: a higher reasoning layer clears inherited sampling knobs (with a warning)", () => {
    const { definition: config, warnings } = resolveConfig([
      { model: "m", temperature: 0.5, topP: 0.9 },
      { reasoning: { effort: "high" } },
    ]);
    expect(isReasoningConfig(config)).toBe(true);
    expect((config as ReasoningConfiguration).reasoning).toEqual({ effort: "high" });
    expect((config as SamplingConfiguration).temperature).toBeUndefined();
    expect(warnings[0]).toMatch(/clears inherited sampling knobs \(temperature, topP\)/);
  });

  it("family-aware: a higher sampling layer clears inherited reasoning (with a warning)", () => {
    const { definition: config, warnings } = resolveConfig([
      { model: "m", reasoning: { effort: "low" } },
      { temperature: 0.5 },
    ]);
    expect(isReasoningConfig(config)).toBe(false);
    expect((config as SamplingConfiguration).temperature).toBe(0.5);
    expect(warnings[0]).toMatch(/clears inherited reasoning/);
  });

  it("merge-then-parse THROWS when a single layer is irreconcilable (sampling + reasoning)", () => {
    expect(() => resolveConfig([{ model: "m", temperature: 0.5, reasoning: { effort: "low" } }])).toThrow(
      /cannot also set sampling knobs/,
    );
  });

  it("merge-then-parse THROWS on a malformed merged value", () => {
    expect(() => resolveConfig([{ model: "m" }, badLayer({ temperature: "hot" })])).toThrow(/temperature must be a finite number/);
  });

  it("returns ONE definition — config knobs AND signature, no split/re-merge dance (§5.3)", () => {
    const { definition } = resolveConfig([
      { model: "m", system: "be terse", timeoutMs: 5_000 },
      { messages: [{ role: "user", content: "pre" }] },
    ]);
    expect(definition).toEqual({
      model: "m",
      system: "be terse",
      timeoutMs: 5_000,
      messages: [{ role: "user", content: "pre" }],
    });
  });

  it("carries the output schema on the definition, so nothing has to smuggle it alongside (§5.2)", () => {
    const schema = { type: "object", properties: { answer: { type: "string" } } } as const;
    const { definition } = resolveConfig([{ model: "m" }, { schema }]);
    expect(definition.schema).toEqual(schema);
  });

  it("signature fields merge low→high like config keys (later system wins)", () => {
    const { definition } = resolveConfig([{ model: "m", system: "default" }, { system: "override" }]);
    expect(definition.system).toBe("override");
  });

  it("THROWS on a malformed signature field (loose shape, never silent)", () => {
    expect(() => resolveConfig([badLayer({ model: "m", timeoutMs: "5s" })])).toThrow(/timeoutMs must be a finite number/);
    expect(() => resolveConfig([badLayer({ model: "m", messages: "hi" })])).toThrow(/messages must be an array/);
    expect(() => resolveConfig([badLayer({ model: "m", attachments: [{ data: { base64: "x" } }] })])).toThrow(
      /attachments\[0\].mediaType must be a non-empty string/,
    );
  });

  it("THROWS on an unknown merged key (strict all the way through resolve)", () => {
    expect(() => resolveConfig([{ model: "m" }, badLayer({ temprature: 1 })])).toThrow(/unknown config key\(s\): temprature/);
  });
});

describe("parseReasoningSpec — strict", () => {
  it("parses effort and/or budgetTokens", () => {
    expect(parseReasoningSpec({ effort: "medium" })).toEqual({ effort: "medium" });
    expect(parseReasoningSpec({ budgetTokens: 4096 })).toEqual({ budgetTokens: 4096 });
    expect(parseReasoningSpec({ effort: "low", budgetTokens: 2048 })).toEqual({ effort: "low", budgetTokens: 2048 });
  });

  it("THROWS on an invalid effort, a non-number budget, or an empty request", () => {
    expect(() => parseReasoningSpec({ effort: "extreme" })).toThrow(/effort must be one of/);
    expect(() => parseReasoningSpec({ budgetTokens: "lots" })).toThrow(/budgetTokens must be a finite number/);
    expect(() => parseReasoningSpec({})).toThrow(/must specify effort and\/or budgetTokens/);
  });
});

describe("`schema` is a CHECKED signature field", () => {
  // `schema` joined SIGNATURE_KEYS when the output schema moved onto the definition (§5.2) but was
  // never added to the structural check, while the split-out bag is cast wholesale to `JsonSchema<T>`.
  // A malformed value then reached `adaptSchemaCached`, whose `WeakMap.get` throws a raw TypeError out
  // of `executeLlmCall` — which is documented to never throw.
  // Casts are the POINT: these values arrive from JSON config layers at runtime, where the compiler
  // never saw them — which is exactly why the parse has to check.
  it("throws on a non-document schema instead of casting it through", () => {
    expect(() => resolveConfig([{ model: "m", prompt: "x" }, { schema: "not-a-schema" as never }])).toThrow(LlmConfigParseError);
    expect(() => resolveConfig([{ model: "m", prompt: "x" }, { schema: 42 as never }])).toThrow(/schema must be an object/);
    expect(() => resolveConfig([{ model: "m", prompt: "x" }, { schema: [{ type: "string" }] as never }])).toThrow(/schema must be an object/);
  });

  it("still passes a real schema document through untouched", () => {
    const schema = { type: "object", properties: { a: { type: "string" } } };
    const { definition } = resolveConfig([{ model: "m", prompt: "x" }, { schema }]);
    expect(definition.schema).toEqual(schema);
  });
});
