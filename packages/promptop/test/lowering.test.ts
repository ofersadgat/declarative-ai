import { describe, expect, it } from "vitest";
import { MapConfigurationRegistry } from "@declarative-ai/llm";
import { createPromptExecutor } from "../src/executor";
import { lowerPromptOp } from "../src/lowering";
import { fakeRunner, OUTPUT_SCHEMA, okOutcome, promptOp, errorOf } from "./fakes";

describe("lowerPromptOp — the PromptOp → LlmCallDefinition lowering (decision #7)", () => {
  it("merges defaults ← inline config and puts the op's `user` text in the prompt", () => {
    const def = lowerPromptOp(promptOp({}, { temperature: 0.2 }), { defaults: { model: "anthropic/x", maxOutputTokens: 50 } });
    expect(def).toMatchObject({ model: "anthropic/claude-haiku-4-5", temperature: 0.2, maxOutputTokens: 100, prompt: "What is 2+2?" });
  });

  it("carries the output schema IN the definition — no spec field to smuggle it through (§5.1)", () => {
    const def = lowerPromptOp(promptOp());
    expect(def.schema).toEqual(OUTPUT_SCHEMA);
    expect(def.system).toBe("Answer tersely.");
  });

  it("resolves a configRef preset UNDER the inline config", () => {
    const configs = new MapConfigurationRegistry().set("fast", { model: "anthropic/preset", temperature: 0.9, maxOutputTokens: 32 });
    const def = lowerPromptOp(promptOp({}, { configRef: "fast", temperature: 0.1 }), { configs });
    // inline wins per key; the preset supplies the rest; `configRef` itself never reaches the config.
    expect(def).toMatchObject({ model: "anthropic/claude-haiku-4-5", temperature: 0.1 });
    expect((def as unknown as Record<string, unknown>).configRef).toBeUndefined();
  });

  it("prepends config-layer `messages` as preamble turns with `user` as the FINAL turn", () => {
    const def = lowerPromptOp(promptOp({}, { messages: [{ role: "user", content: "context" }] as never }));
    expect(def.messages).toEqual([
      { role: "user", content: "context" },
      { role: "user", content: "What is 2+2?" },
    ]);
    expect(def.prompt).toBeUndefined();
  });

  it("THROWS when the config supplies a `prompt` — the op's `user` text IS the prompt", () => {
    expect(() => lowerPromptOp(promptOp({}, { prompt: "elsewhere" }))).toThrow(/a PromptOp's prompt is its `user` text/);
  });

  it("declares injected tools as FUNCTION tool declarations, appended to any the config carries", () => {
    const def = lowerPromptOp(promptOp(), {
      tools: { add: { description: "adds", inputSchema: { type: "object" }, readOnly: true, run: () => null } },
    });
    expect(def.tools).toEqual([{ name: "add", description: "adds", inputSchema: { type: "object" } }]);
  });
});

describe("the output PARAMETER'S KIND decides whether there is a structured-output contract", () => {
  // `kindFor` (ops/model.ts) DERIVES the kind from the very schema being forwarded: a `text` parameter's
  // schema is `{type:"string"}` and a `blob` parameter's is `{type:"string",contentMediaType:…}`. Copying
  // either into `definition.schema` made `executeLlmCall` attach `Output.object`, so `generate.ts`
  // demanded parseable JSON on the text channel: a text op answering `four` came back
  // `api-retriable: unparseable structured output`, and a blob op whose bytes arrived and projected
  // correctly came back `api-retriable: structured output was empty/absent`.
  const textOut = { name: "output", kind: "text" as const, schema: { type: "string" as const } };
  const blobOut = { name: "output", kind: "blob" as const, schema: { type: "string" as const, contentMediaType: "image/png" } };

  it("a TEXT-kind output lowers to a SCHEMA-LESS call — the text path, which yields `string` (§5.2)", () => {
    // Exactly the parameter `promptOp({ output: { schema: { type: "string" } } })` builds (ops/typed.ts),
    // i.e. the DEFAULT text path through promptop.
    const def = lowerPromptOp(promptOp({ output: textOut }));
    expect(def.schema).toBeUndefined();
  });

  it("a BLOB-kind output lowers to a SCHEMA-LESS call — the bytes path (§7.1)", () => {
    const def = lowerPromptOp(promptOp({ output: blobOut }));
    expect(def.schema).toBeUndefined();
  });

  it("a JSON-kind output still carries its schema — that is the contract this gate protects", () => {
    expect(lowerPromptOp(promptOp()).schema).toEqual(OUTPUT_SCHEMA);
  });

  it("a config LAYER cannot force structured output onto a text/blob output — the op's slot is the authority", () => {
    // `schema` is a SIGNATURE key, so a `defaults`/preset layer can carry one; the output parameter's
    // kind decides, not the config.
    const def = lowerPromptOp(promptOp({ output: textOut }), { defaults: { schema: OUTPUT_SCHEMA as never } });
    expect(def.schema).toBeUndefined();
    expect("schema" in def).toBe(false);
  });
});

describe("the executor honors the never-throws contract around the lowering", () => {
  it("turns a malformed config into a permanent outcome rather than a raised error", async () => {
    const { runner, calls } = fakeRunner([okOutcome()]);
    const out = await createPromptExecutor({ runner }).start(promptOp({}, { prompt: "elsewhere" }), {}).result;
    expect(errorOf(out)?.classification).toBe("permanent");
    expect(errorOf(out)?.reason).toMatch(/invalid llm config/);
    expect(calls).toHaveLength(0);
  });

  it("surfaces a strict-parse failure (an unknown config key) the same way", async () => {
    const { runner } = fakeRunner([okOutcome()]);
    const out = await createPromptExecutor({ runner }).start(promptOp({}, { temprature: 1 }), {}).result;
    expect(errorOf(out)?.reason).toMatch(/unknown config key\(s\): temprature/);
  });
});
