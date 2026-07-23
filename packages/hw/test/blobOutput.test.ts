import type { ResolvedValue, FunctionResult } from "@declarative-ai/exec";
import type { WorkflowMetrics } from "../src/ports";
/**
 * A BLOB-kind operation output (DESIGN §3.7): "a produced artifact is a blob-kind output
 * slot, not a parallel output channel."
 *
 * The engine's output path assumed every operation returns a JSON RECORD of named outputs, so bytes
 * fell through it as an empty record and the state failed with "did not produce required output" —
 * about a value the operation had, in fact, produced. Nothing in the validator rejected such an op
 * either, so the shape was authorable and simply broken.
 */
import { describe, expect, it } from "vitest";
import type { ExecServices, ExecResult, HostCapabilities } from "@declarative-ai/exec";
import { hostFunction } from "@declarative-ai/exec";
import { WorkflowEngine } from "../src/engine";
import { loadBundle } from "../src/loader";
import type { StateDef } from "../src/format";
import { FakePromptExecutor, newRegistry } from "./fakes";

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);

/** A state whose prompt operation declares a BLOB output — a generated file, not a JSON record. */
function blobStates(outputs: Record<string, unknown>): Record<string, StateDef> {
  return {
    render: {
      label: "Render",
      inputs: {},
      outputs: outputs as StateDef["outputs"],
      operation: {
        kind: "prompt",
        prompt: { template: "Draw a diagram." },
        config: { model: "artist" },
        output: { kind: "blob", schema: { type: "string", contentMediaType: "image/png" } },
      },
    } as StateDef,
  };
}

function run(states: Record<string, StateDef>, produce: () => ExecResult<ResolvedValue, WorkflowMetrics>) {
  const engine = new WorkflowEngine({
    bundle: loadBundle(states, "render"),
    registry: newRegistry(),
    prompt: new FakePromptExecutor(produce),
  });
  return engine.run({ inputs: {} });
}

const ok = (value: unknown): ExecResult<ResolvedValue, WorkflowMetrics> => ({ value: value as never, metrics: { durationMs: 1, costUsd: 0, costSource: "unknown" as const } });

const BATCH: HostCapabilities = { interactive: false, readOnly: true, memoizable: true };

/** The same blob-output state, run through the FUNCTION path: a registered function that generates a
 *  file rather than a record of named outputs. */
function runFunction(produce: () => ResolvedValue, mediaType = "image/png") {
  const slot = { kind: "blob", schema: { type: "string", contentMediaType: mediaType } } as const;
  const states: Record<string, StateDef> = {
    render: {
      label: "Render",
      inputs: {},
      outputs: { image: slot } as StateDef["outputs"],
      operation: { kind: "function", function: "generate_file", output: slot },
    } as StateDef,
  };
  const registry = newRegistry();
  registry.functions.set(
    "generate_file",
    hostFunction(async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({ value: produce() }), BATCH),
  );
  const engine = new WorkflowEngine({
    bundle: loadBundle(states, "render"),
    registry,
    prompt: new FakePromptExecutor(() => {
      throw new Error("no prompt op should run");
    }),
  });
  return engine.run({ inputs: {} });
}

describe("a blob-kind operation output fills the state's single produced slot", () => {
  it("accepts raw BYTES as the output value", async () => {
    const result = await run(blobStates({ image: { kind: "blob", schema: { type: "string", contentMediaType: "image/png" } } }), () => ok(PNG));
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["image"]).toBe(PNG);
  });

  it("registers inline STRING content as an artifact, as the record path does", async () => {
    const result = await run(blobStates({ doc: { kind: "blob", schema: { type: "string", contentMediaType: "markdown" } } }), () => ok("# hi"));
    expect(result.outcome).toBe("success");
    const doc = result.outputs?.["doc"] as { artifact?: boolean; content?: string; format?: string };
    expect(doc.artifact).toBe(true);
    expect(doc.content).toBe("# hi");
    expect(doc.format).toBe("markdown");
    expect(result.artifacts).toHaveLength(1);
  });

  it("does NOT overwrite the op's blob schema with the named-outputs object contract", async () => {
    const fake = new FakePromptExecutor(() => ok(PNG));
    const engine = new WorkflowEngine({
      bundle: loadBundle(blobStates({ image: { kind: "blob", schema: { type: "string", contentMediaType: "image/png" } } }), "render"),
      registry: newRegistry(),
      prompt: fake,
    });
    await engine.run({ inputs: {} });
    // Asking a model for a JSON record and then handing back a file is incoherent; the authored blob
    // schema has to survive to the runner.
    const sent = fake.calls[0]!.op.output;
    expect(sent.kind).toBe("blob");
    expect(sent.schema).toEqual({ type: "string", contentMediaType: "image/png" });
  });

  it("fails with a NAMED reason when the state declares more than one produced slot", async () => {
    const result = await run(
      blobStates({
        image: { kind: "blob", schema: { type: "string", contentMediaType: "image/png" } },
        caption: { schema: { type: "string" } },
      }),
      () => ok(PNG),
    );
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/blob operation output, which fills exactly ONE produced output slot, but the state declares 2/);
  });

  it("does the same for a FUNCTION op — the rule is the op's output KIND, not which runner ran it", async () => {
    // The blob branch was reachable only from the prompt path, because the function path passed no
    // output kind at all. So a registered function that generates a file — the case blob-kind outputs
    // exist for — failed with "did not produce required output" about the bytes it had just returned.
    const result = await runFunction(() => PNG);
    expect(result.outcome).toBe("success");
    expect(result.outputs?.["image"]).toBe(PNG);
  });

  it("registers a FUNCTION op's inline string content as an artifact too", async () => {
    const result = await runFunction(() => "# generated", "markdown");
    expect(result.outcome).toBe("success");
    const doc = result.outputs?.["image"] as { artifact?: boolean; content?: string };
    expect(doc.artifact).toBe(true);
    expect(doc.content).toBe("# generated");
  });

  it("leaves the ordinary JSON-record path untouched", async () => {
    const states: Record<string, StateDef> = {
      render: {
        label: "Render",
        inputs: {},
        outputs: { a: { schema: { type: "string" } }, b: { schema: { type: "number" } } },
        operation: { kind: "prompt", prompt: { template: "go" }, config: { model: "m" } },
      } as StateDef,
    };
    const result = await run(states, () => ok({ a: "x", b: 2 }));
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ a: "x", b: 2 });
  });
});
