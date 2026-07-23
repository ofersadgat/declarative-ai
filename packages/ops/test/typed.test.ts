import { describe, expect, expectTypeOf, it } from "vitest";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type { FunctionOp, InlineFamily, Parameter, PromptOp, Ref } from "../src/model";
import { isOk } from "@declarative-ai/json";
import { HOST_CAPABILITIES, isStreaming, runFunction, type FunctionRegistry } from "../src/registry";
import {
  bound,
  defineFunction,
  free,
  functionOp,
  promptOp,
  registerFunctionDef,
  runtimeOp,
  type InferSchema,
  type OperationOutput,
  type Widened,
} from "../src/typed";

// The §4 example: impl params inferred from the input schema, return checked as number.
const wordCount = defineFunction({
  name: "wordCount",
  input: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } as const,
  output: { type: "number" } as const,
  impl: ({ text }) => text.split(/\s+/).filter(Boolean).length,
});

describe("defineFunction (typed layer §4)", () => {
  it("infers impl parameter and output types from const schemas", () => {
    expectTypeOf(wordCount.impl).parameter(0).toEqualTypeOf<{ text: string }>();
    expectTypeOf(wordCount.impl).returns.toExtend<number | Promise<number>>();
    // @ts-expect-error — impl returning a string does not satisfy the number output schema
    defineFunction({ name: "bad", input: { type: "object" } as const, output: { type: "number" } as const, impl: () => "not a number" });
  });

  it("keeps the runtime schemas as the runtime truth", () => {
    expect(wordCount.input).toEqual({ type: "object", properties: { text: { type: "string" } }, required: ["text"] });
    expect(wordCount.output).toEqual({ type: "number" });
    expect(wordCount.impl({ text: "one two three" }, undefined as void)).toBe(3);
  });

  it("registers the plain string-keyed impl alongside the typed handle", async () => {
    const registry = new Map() as FunctionRegistry<{ tag: string }, { durationMs: number }>;
    registerFunctionDef(registry, wordCount);
    expect(registry.has("wordCount")).toBe(true);
    const pure = registry.get("wordCount")!;
    expect(pure.kind).toBe("pure");
    expect(pure.kind === "pure" && pure.impl({ text: "a b" })).toEqual({ value: 2 });

    const echo = defineFunction<{ readonly type: "object" }, { readonly type: "string" }, { tag: string }>({
      name: "echo",
      input: { type: "object" },
      output: { type: "string" },
      impl: (_inputs, ctx) => ctx.tag,
    });
    registerFunctionDef(registry, echo, { async: true, capabilities: HOST_CAPABILITIES, stream: true });
    const host = registry.get("echo")!;
    expect(host.kind).toBe("host");
    expect(isStreaming(host)).toBe(true);
    await expect(runFunction(host, {}, { tag: "ctx" })).resolves.toEqual({ value: "ctx" });
  });

  it("lifts a THROWING def into a classified failure rather than propagating the exception", () => {
    const registry = new Map() as FunctionRegistry<void, { durationMs: number }>;
    const boom = defineFunction({
      name: "boom",
      input: { type: "object" } as const,
      output: { type: "string" } as const,
      impl: () => {
        throw new Error("nope");
      },
    });
    registerFunctionDef(registry, boom);
    const entry = registry.get("boom")!;
    const result = entry.kind === "pure" ? entry.impl({}) : undefined;
    expect(result && !isOk(result) && result.error.classification).toBe("permanent");
    expect(result && !isOk(result) && result.error.reason).toContain("function 'boom'");
  });
});

describe("an ASYNC impl cannot be registered as `pure`", () => {
  // It compiled, registered as `pure`, and returned the PROMISE ITSELF as the value — which then flowed
  // into ResolvedValue, memo keys and acceptOpOutputs, with the impl's error channel gone entirely (a
  // rejecting impl produced a SUCCESS carrying a rejected promise).
  const asyncDef = defineFunction({
    name: "slow",
    input: { type: "object" } as const,
    output: { type: "number" } as const,
    impl: async () => 42,
  });

  it("is a COMPILE error on the sync path — the impl's return type reaches the registration site", () => {
    const registry = new Map() as FunctionRegistry<void, { durationMs: number }>;
    // @ts-expect-error — an async impl does not fit the `pure` variant; register it with { async: true }
    registerFunctionDef(registry, asyncDef);
    // @ts-expect-error — and saying so explicitly does not help either
    registerFunctionDef(registry, asyncDef, { async: false });
    // The intended registration still type-checks and behaves.
    registerFunctionDef(registry, asyncDef, { async: true, capabilities: HOST_CAPABILITIES });
    expect(registry.get("slow")!.kind).toBe("host");
  });

  it("is caught at RUNTIME for a def the compiler never saw as async", async () => {
    // Dynamic construction and widening casts bypass the builders (the module says so up top), so the
    // guard is not redundant with the types.
    const registry = new Map() as FunctionRegistry<void, { durationMs: number }>;
    registerFunctionDef(registry, asyncDef as unknown as typeof wordCount);
    const entry = registry.get("slow")!;
    const result = entry.kind === "pure" ? entry.impl({}) : undefined;
    expect(result && isOk(result)).toBe(false);
    expect(result && !isOk(result) && result.error.reason).toMatch(/returned a Promise but the function is registered as 'pure'/);
  });

  it("keeps the ERROR CHANNEL a throwing async impl would otherwise lose", async () => {
    const registry = new Map() as FunctionRegistry<void, { durationMs: number }>;
    const throwing = defineFunction({
      name: "throwsLater",
      input: { type: "object" } as const,
      output: { type: "number" } as const,
      impl: async () => {
        throw new Error("boom");
      },
    });
    registerFunctionDef(registry, throwing as unknown as typeof wordCount);
    const entry = registry.get("throwsLater")!;
    const result = entry.kind === "pure" ? entry.impl({}) : undefined;
    // Before: `{ value: Promise { <rejected> } }` — a success, and an unhandled rejection.
    expect(result && isOk(result)).toBe(false);
    expect(result && !isOk(result) && result.error.classification).toBe("permanent");
  });
});

describe("InferSchema degradation (§4 known limits)", () => {
  it("degrades to Widened for non-literal schema types instead of silent unknown", () => {
    expectTypeOf<InferSchema<JsonSchema>>().toEqualTypeOf<Widened>();
    expectTypeOf<InferSchema<{ type: "string" }>>().toEqualTypeOf<string>();
  });
});

describe("typed op builders", () => {
  it("functionOp lowers a def application to a plain FunctionOp with checked bindings", () => {
    const op = functionOp(wordCount, { text: "hello world" });
    expect(op.kind).toBe("function");
    expect(op.functionRef).toBe("wordCount");
    expect(op.input.text).toEqual({ kind: "text", schema: { type: "string" }, binding: { text: "hello world" } });
    expect(op.output.kind).toBe("json");
    expectTypeOf<OperationOutput<typeof op>>().toEqualTypeOf<number>();
  });

  it("leaves unbound schema inputs free and wires producer edges", () => {
    const freeOp = functionOp(wordCount);
    expect(freeOp.input.text).toEqual({ kind: "text", schema: { type: "string" } }); // free slot — no binding
    const textProducer = promptOp({ user: "Write a sentence.", output: { name: "sentence", schema: { type: "string" } } });
    const wired = functionOp(wordCount, { text: textProducer });
    expect(wired.input.text!.binding).toEqual({ op: textProducer });
  });

  it("rejects mis-typed literal and producer bindings at compile time", () => {
    // @ts-expect-error — a number literal cannot fill the string slot
    functionOp(wordCount, { text: 42 });
    const numberProducer = functionOp(wordCount, { text: "seed" }); // output: number
    // @ts-expect-error — a number-producing op cannot feed the string slot
    functionOp(wordCount, { text: numberProducer });
  });

  it("promptOp builds a PromptOp with inferred output typing", () => {
    const op = promptOp({
      system: "You are terse.",
      user: "Summarize: {{text}}",
      config: { model: "claude-sonnet-5" },
      input: { text: free("text", { type: "string" }), style: bound("bullets", "json") },
      output: { name: "summary", schema: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] } },
    });
    expectTypeOf<OperationOutput<typeof op>>().toEqualTypeOf<{ summary: string }>();
    const plain: PromptOp<InlineFamily> = op;
    expect(plain.kind).toBe("prompt");
    expect(plain.user).toBe("Summarize: {{text}}");
    expect(plain.input.text).toEqual({ kind: "text", schema: { type: "string" } });
    expect(plain.input.style).toEqual({ kind: "json", binding: { json: "bullets" } });
    expect(plain.output).toEqual({
      name: "summary",
      kind: "json",
      schema: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] },
    });
  });

  it("runtimeOp emits a PLAIN FunctionOp — adapter name + bound config/prompt inputs (§3.1)", () => {
    const op = runtimeOp({
      runtime: "claude-code",
      prompt: "Fix the failing test.",
      config: { permissionMode: "plan", tools: ["Read", "Grep"] },
      output: { name: "report", schema: { type: "string" } },
    });
    // The op SHAPE does not change at all: it is exactly a FunctionOp.
    const plain: FunctionOp<InlineFamily> = op;
    expect(plain.kind).toBe("function");
    expect(plain.functionRef).toBe("claude-code");
    expect(plain.input.prompt).toEqual({ kind: "text", binding: { text: "Fix the failing test." }, schema: { type: "string" } });
    expect(plain.input.config).toEqual({ kind: "json", binding: { json: { permissionMode: "plan", tools: ["Read", "Grep"] } } });
    expect(Object.keys(plain).sort()).toEqual(["functionRef", "input", "kind", "output"]); // no extra fields
    expectTypeOf<OperationOutput<typeof op>>().toEqualTypeOf<string>();
  });

  it("binding helpers desugar to base Ref cases", () => {
    const litText: Parameter<InlineFamily> = bound("hello", "text");
    const litJson: Parameter<InlineFamily> = bound({ a: 1 }, "json");
    expect(litText.binding).toEqual({ text: "hello" });
    expect(litJson.binding).toEqual({ json: { a: 1 } });
    // A blob-kind literal keeps its `{ blob }` tag (not `{ json }`), so downstream Ref-union
    // discrimination sees a blob, not arbitrary JSON.
    const litBlob: Parameter<InlineFamily> = bound("aGVsbG8=", "blob");
    expect(litBlob.binding).toEqual({ blob: "aGVsbG8=" });
    const edge: Ref<InlineFamily> | undefined = functionOp(wordCount, { text: "x" }).input.text!.binding;
    expect(edge).toEqual({ text: "x" });
  });

  it("dictionary bindings beyond the schema still wire dynamically", () => {
    const openDef = defineFunction({
      name: "combine",
      input: { type: "object" } as const,
      output: { type: "object" } as const,
      impl: (inputs) => inputs as Record<string, JsonValue>,
    });
    const op = functionOp(openDef, { extra: { nested: true } } as Record<string, JsonValue>);
    expect(op.input.extra).toEqual({ kind: "json", binding: { json: { nested: true } } });
  });
});
