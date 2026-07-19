import { MapConfigurationRegistry } from "@declarative-ai/core";
import { describe, expect, it } from "vitest";
import { llmCallBinding } from "../src/ports";

describe("llmCallBinding — declarative config resolution", () => {
  it("merges defaults ← inline config and layers the prompt", () => {
    const b = llmCallBinding({ model: "m", temperature: 0.2 });
    const def = b.definition({ prompt: "hi", config: { temperature: 0.7 } }) as Record<string, unknown>;
    expect(def).toMatchObject({ model: "m", temperature: 0.7, prompt: "hi" });
  });

  it("resolves a configRef preset from the registry (under inline overrides), consuming the ref", () => {
    const registry = new MapConfigurationRegistry().set("fast", { model: "m", maxOutputTokens: 256, temperature: 0.1 });
    const b = llmCallBinding({}, { registry });
    const def = b.definition({ prompt: "hi", config: { configRef: "fast", temperature: 0.9 } }) as Record<string, unknown>;
    expect(def).toMatchObject({ model: "m", maxOutputTokens: 256, temperature: 0.9, prompt: "hi" });
    expect(def.configRef).toBeUndefined(); // the ref is consumed, not passed through as a config field
  });

  it("family-aware: an inline reasoning config clears inherited sampling defaults", () => {
    const b = llmCallBinding({ model: "m", temperature: 0.5 });
    const def = b.definition({ prompt: "hi", config: { reasoning: { effort: "high" } } }) as Record<string, unknown>;
    expect(def).toMatchObject({ model: "m", reasoning: { effort: "high" }, prompt: "hi" });
    expect(def.temperature).toBeUndefined();
  });

  it("throws on a malformed merged config (strict parse — becomes a permanent op failure in the engine)", () => {
    const b = llmCallBinding({ model: "m" });
    expect(() => b.definition({ prompt: "hi", config: { temperature: "hot" } })).toThrow(/temperature must be a finite number/);
  });

  it("throws when no model is resolvable from defaults/preset/inline", () => {
    const b = llmCallBinding({});
    expect(() => b.definition({ prompt: "hi", config: {} })).toThrow(/model must be a non-empty string/);
  });

  it("throws on an unknown config key — never silently drops it", () => {
    const b = llmCallBinding({ model: "m" });
    expect(() => b.definition({ prompt: "hi", config: { temprature: 0.7 } })).toThrow(/unknown config key\(s\): temprature/);
  });
});

describe("llmCallBinding — definition-layer passthrough (Partial<LlmCallDefinition> config surface)", () => {
  it("passes system + timeoutMs from the state config through to the definition", () => {
    const b = llmCallBinding({ model: "m" });
    const def = b.definition({ prompt: "hi", config: { system: "be terse", timeoutMs: 5_000 } }) as Record<string, unknown>;
    expect(def).toMatchObject({ model: "m", system: "be terse", timeoutMs: 5_000, prompt: "hi" });
  });

  it("binding defaults may carry a shared system prompt; inline config overrides it", () => {
    const b = llmCallBinding({ model: "m", system: "shared instructions" });
    expect((b.definition({ prompt: "hi", config: {} }) as Record<string, unknown>).system).toBe("shared instructions");
    expect((b.definition({ prompt: "hi", config: { system: "state-specific" } }) as Record<string, unknown>).system).toBe("state-specific");
  });

  it("config messages become preamble turns with the rendered prompt appended as the final user turn", () => {
    const b = llmCallBinding({ model: "m" });
    const def = b.definition({
      prompt: "hi",
      config: { messages: [{ role: "user", content: "pre" }] },
    }) as Record<string, unknown>;
    expect(def.messages).toEqual([
      { role: "user", content: "pre" },
      { role: "user", content: "hi" },
    ]);
    expect(def.prompt).toBeUndefined();
  });

  it("throws on a config-supplied prompt (the operation's prompt is rendered from its template)", () => {
    const b = llmCallBinding({ model: "m" });
    expect(() => b.definition({ prompt: "hi", config: { prompt: "conflicting" } })).toThrow(/rendered from its template/);
  });
});
