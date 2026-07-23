import { describe, expectTypeOf, it } from "vitest";
import type {
  FunctionOp,
  OperationRecord,
  Id,
  IdFamily,
  InlineFamily,
  JsonRefs,
  NamedParameter,
  Operation,
  OperationRef,
  Parameter,
  PromptOp,
  Ref,
  ResolvedValue,
} from "../src/model";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type { Metrics } from "../src/metrics";

// Type-shape assertions: `Operation<IdFamily>` matches findmyprompt's
// model semantics modulo the property renames (`textId` → `text`, …, id field dropped
// from the content shape) (DESIGN §8.1); `InlineFamily` pins every leaf to a precise inline type.
describe("op model type shapes", () => {
  it("IdFamily pins every leaf to a content id", () => {
    expectTypeOf<IdFamily["text"]>().toEqualTypeOf<Id>();
    expectTypeOf<IdFamily["json"]>().toEqualTypeOf<Id>();
    expectTypeOf<IdFamily["schema"]>().toEqualTypeOf<Id>();
    expectTypeOf<IdFamily["result"]>().toEqualTypeOf<Id>();
    expectTypeOf<IdFamily["op"]>().toEqualTypeOf<OperationRef>();
    expectTypeOf<IdFamily["binding"]>().toEqualTypeOf<Ref<IdFamily>>();
  });

  it("InlineFamily pins every leaf to the value itself", () => {
    expectTypeOf<InlineFamily["text"]>().toEqualTypeOf<string>();
    expectTypeOf<InlineFamily["json"]>().toEqualTypeOf<JsonValue>();
    expectTypeOf<InlineFamily["schema"]>().toEqualTypeOf<JsonSchema>();
    expectTypeOf<InlineFamily["result"]>().toEqualTypeOf<OperationRecord<InlineFamily, ResolvedValue, Metrics>>();
    expectTypeOf<InlineFamily["op"]>().toEqualTypeOf<Operation<InlineFamily> | string>();
  });

  it("PromptOp<IdFamily> mirrors findmyprompt's PromptOp modulo renames", () => {
    // was: systemPromptTextId?/userPromptTextId/configJsonId
    expectTypeOf<PromptOp<IdFamily>["system"]>().toEqualTypeOf<Id | undefined>();
    expectTypeOf<PromptOp<IdFamily>["user"]>().toEqualTypeOf<Id>();
    expectTypeOf<PromptOp<IdFamily>["config"]>().toEqualTypeOf<Id>();
    expectTypeOf<PromptOp<IdFamily>["kind"]>().toEqualTypeOf<"prompt">();
    expectTypeOf<PromptOp<IdFamily>["input"]>().toEqualTypeOf<{ [name: string]: Parameter<IdFamily> }>();
    expectTypeOf<PromptOp<IdFamily>["output"]>().toEqualTypeOf<NamedParameter<IdFamily>>();
  });

  it("FunctionOp keeps functionRef as a plain string (registry name or op id)", () => {
    expectTypeOf<FunctionOp<IdFamily>["functionRef"]>().toEqualTypeOf<string>();
    expectTypeOf<FunctionOp<InlineFamily>["functionRef"]>().toEqualTypeOf<string>();
  });

  it("Ref<F> keeps findmyprompt's binding union case-for-case", () => {
    type IdRef = Ref<IdFamily>;
    // was { textId } / { jsonId } / { resultId } / { jsonRefs } / OperationRef & { parameters? }
    expectTypeOf<{ text: Id }>().toExtend<IdRef>();
    expectTypeOf<{ json: Id }>().toExtend<IdRef>();
    expectTypeOf<{ result: Id }>().toExtend<IdRef>();
    expectTypeOf<JsonRefs<IdFamily>>().toExtend<IdRef>();
    expectTypeOf<{ op: OperationRef; parameters?: { [name: string]: Parameter<IdFamily> } }>().toExtend<IdRef>();
  });

  it("inline producer edges may name a declared child by local key", () => {
    const childEdge: Ref<InlineFamily> = { op: "plan" };
    const embedded: Ref<InlineFamily> = {
      op: { kind: "function", functionRef: "identity", input: {}, output: { name: "out", kind: "json" } },
    };
    void childEdge;
    void embedded;
  });

  it("a mixed family is a legal instantiation (resolved-id backend handing inline values)", () => {
    interface MixedFamily {
      text: Id | string;
      json: Id | JsonValue;
      blob: Id | Uint8Array;
      schema: Id | JsonSchema;
      result: Id | OperationRecord<InlineFamily, ResolvedValue, Metrics>;
      op: OperationRef;
      binding: Ref<MixedFamily>;
    }
    const p: Parameter<MixedFamily> = { kind: "json", binding: { json: "json:abc" } };
    void p;
    expectTypeOf<Operation<MixedFamily>>().toExtend<Operation<MixedFamily>>();
  });

  it("an Operation's parameter tree is the graph — no separate Graph type, wiring in bindings", () => {
    const graph: Operation<InlineFamily> = {
      kind: "prompt",
      user: "Summarize: {{text}}",
      config: { model: "claude-sonnet-5" },
      input: {
        text: { kind: "text" }, // free/external input
        style: { kind: "json", binding: { json: "terse" } }, // literal binding
        outline: {
          kind: "json",
          binding: {
            op: { kind: "function", functionRef: "outline", input: {}, output: { name: "out", kind: "json" } },
          },
        },
      },
      output: { name: "summary", kind: "text", schema: { type: "string" } },
    };
    void graph;
  });
});
