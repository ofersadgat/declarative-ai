import { describe, expect, it } from "vitest";
import { codecs, type JsonValue } from "@declarative-ai/json";
import { generateFlat, streamingModel, usage } from "./fakes";

// The declaration-merging half of the codec contract (API.md, "Codecs and type names"): a type NAME
// bound globally. Vitest isolates module state per file, so registering the process-global `codecs`
// singleton here cannot leak into a sibling suite — and doing it once at module load (not inside a
// test) sidesteps the registry's throw on a second, different codec for the same name.
declare module "@declarative-ai/json" {
  interface TypeRegistry {
    DateTime: { value: Date; json: number };
  }
}
codecs.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });

/** A model that streams one structured JSON body as output text. */
function jsonModel(body: string) {
  return streamingModel([
    { type: "stream-start", warnings: [] },
    { type: "text-start", id: "1" },
    { type: "text-delta", id: "1", delta: body },
    { type: "text-end", id: "1" },
    { type: "finish", finishReason: "stop", usage: usage(10, 5) },
  ]);
}

describe("decode on the structured output path (API.md, 'Codecs and type names')", () => {
  it("LIFTS a validated wire value to its decoded type — an epoch number becomes a Date", async () => {
    const out = await generateFlat({
      model: jsonModel("1700000000000"),
      prompt: "when?",
      schema: { type: "number", "x-type": "DateTime" },
    });
    expect(out.value?.value).toBeInstanceOf(Date);
    expect(out.value?.value).toEqual(new Date(1_700_000_000_000));
  });

  it("decodes a nested x-type leaf, AFTER validating the WIRE (number) form", async () => {
    const seenWire: unknown[] = [];
    const out = await generateFlat({
      model: jsonModel('{"at":1700000000000,"note":"hi"}'),
      prompt: "when?",
      schema: { type: "object", properties: { at: { type: "number", "x-type": "DateTime" }, note: { type: "string" } }, required: ["at"] },
      // Validation runs on the WIRE form (a number), BEFORE decode lifts it to a Date. Capturing what the
      // validator sees proves the ordering: validate the `Jsonify<T>`, then decode.
      validate: (value) => {
        seenWire.push((value as { at: unknown }).at);
      },
    });
    expect(seenWire).toEqual([1_700_000_000_000]); // the validator saw the epoch, not a Date
    const parsed = out.value?.value as { at: unknown; note: unknown };
    expect(parsed.at).toBeInstanceOf(Date);
    expect(parsed.at).toEqual(new Date(1_700_000_000_000));
    expect(parsed.note).toBe("hi");
  });

  it("leaves an UNREGISTERED x-type name as raw JSON — a schema may name a type this process doesn't model", async () => {
    const out = await generateFlat({
      model: jsonModel("5"),
      prompt: "how much?",
      schema: { type: "number", "x-type": "Money" }, // no codec registered for "Money"
    });
    expect(out.value?.value).toBe(5);
  });

  it("is a byte-for-byte passthrough for a plain structured call with no x-type", async () => {
    const out = await generateFlat({
      model: jsonModel('{"answer":"4"}'),
      prompt: "2+2?",
      schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
    });
    expect(out.value?.value).toEqual({ answer: "4" });
  });
});
