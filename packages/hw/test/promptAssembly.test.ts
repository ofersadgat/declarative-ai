/**
 * WHAT REACHES THE MODEL — the assembled call, asserted verbatim.
 *
 * Every other engine test asks what a run DECIDED. This one asks what it SAID: the exact bytes in
 * `op.user`, the exact turns left in the transcript, and the other channels a call carries beside its
 * prompt (`system`, the structured-output contract, the tool declarations).
 *
 * It exists because the assembly is the one part of the engine with no schema over it. A binding that
 * resolves wrong fails a validator; a prompt that is assembled wrong is still a valid string, reaches
 * the provider, costs money and comes back plausible. The `<conversation-history>` nesting bug lived
 * exactly there — every test passed, every run succeeded, and the prompt doubled per call until a
 * provider refused it.
 *
 * So the assertions here are DELIBERATELY literal. A test that checks `toContain("Hello")` would have
 * passed throughout that bug. Whole-string equality is what makes a change to the wire format show up
 * as a diff someone has to read.
 *
 * The stub answers on the `session` channel the way a real prompt executor does (`fakes.ts`), so the
 * transcript these tests read back is the one production writes, not an engine-private shortcut.
 */
import { MapSessionStore, withRecord, withSessionPosition } from "@declarative-ai/exec";
import { describe, expect, it } from "vitest";
import { WorkflowEngine } from "../src/engine.js";
import type { StateDef } from "../src/format.js";
import { loadBundle } from "../src/loader.js";
import { InMemoryPersistence } from "../src/ports.js";
import { FakePromptExecutor, newRegistry, ok, promptOf, toolNamesOf, type FakeCall, type Script } from "./fakes.js";

/**
 * The session stack as a host composes it: resolve the position, claim it, record what ran. Without
 * these wrappers nothing writes a transcript and every preamble here would be trivially empty.
 */
function makeEngine(files: Record<string, StateDef>, rootId: string, script: Script, tools: Record<string, unknown> = {}) {
  const fake = new FakePromptExecutor(script);
  const sessions = new MapSessionStore();
  const registry = newRegistry();
  for (const [name, tool] of Object.entries(tools)) registry.tools.set(name, tool as never);
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
    persistence: new InMemoryPersistence(),
    sessions: sessions as never,
  } as never);
  return { engine, fake, sessions };
}

/** A state whose whole job is to say one thing and hand back one string. */
function say(prompt: string, extra: Partial<StateDef> = {}): StateDef {
  return {
    outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
    operation: { kind: "prompt", model: "m", prompt, output: { r: { schema: { type: "string" } } } },
    ...extra,
  };
}

/** Answers `{"r":"A<n>"}` in call order — short enough that a whole prompt stays readable inline. */
function counting(): Script {
  let n = 0;
  return () => {
    n += 1;
    return ok({ r: `A${n}` });
  };
}

describe("a conversation, asserted verbatim", () => {
  /**
   * Three states sharing one named conversation. Each prompt is its own rendered template.
   *
   * `environment.conversation` used to prepend a rendering of the transcript, and this suite pinned
   * every byte of it. The layer below already carries the conversation — `applySession` resumes the
   * provider's session, branches one server-side, or replays the turns as messages — so the preamble
   * was a second copy of whichever had happened, travelling inside `op.user` where the session layer
   * recorded it as the turn that was asked. The next preamble then rendered it back: 8.4k → 403k →
   * 1.13M → 2.55M characters over four passes of one real run, until a provider refused it.
   */
  const files: Record<string, StateDef> = {
    root: {
      environment: { session: "chat" },
      inputs: { who: { schema: { type: "string" } } },
      children: {
        a: { state: "ask", inputs: { who: ".inputs.who" } },
        b: { state: "again", inputs: { who: ".inputs.who" } },
        c: { state: "third" },
      },
      sequence: ["a", "b", "c"],
      outputs: { r: { binding: ".children.c.output.r" } },
    },
    ask: { ...say("Hello, {{.inputs.who}}."), inputs: { who: { schema: { type: "string" } } } },
    again: { ...say("And again for {{.inputs.who}}."), inputs: { who: { schema: { type: "string" } } } },
    third: say("Third."),
  };

  it("sends each state's own prompt, whatever the conversation already holds", async () => {
    const { engine, fake } = makeEngine(files, "root", counting());
    expect((await engine.run({ inputs: { who: "world" } })).outcome).toBe("success");

    expect(promptOf(fake.calls[0]!)).toBe("Hello, world.");
    expect(promptOf(fake.calls[1]!)).toBe("And again for world.");
    // The third is the one that used to grow: by here two exchanges are on record.
    expect(promptOf(fake.calls[2]!)).toBe("Third.");
  });

  /**
   * The conversation is still there — it is just not in the prompt.
   *
   * A record's turns come from the call's own session outcome, so the transcript accumulates exactly
   * as before. What changed is that a turn is now the question that was ASKED and nothing else, which
   * is what makes the store's copy stable rather than growing with every reader of it.
   */
  it("still accumulates a transcript, holding what was asked and nothing more", async () => {
    const { engine, sessions } = makeEngine(files, "root", counting());
    await engine.run({ inputs: { who: "world" } });

    expect((sessions.messages("chat") as Array<{ role: string; content: string }>).map((t) => [t.role, t.content])).toEqual([
      ["user", "Hello, world."],
      ["assistant", '{"r":"A1"}'],
      ["user", "And again for world."],
      ["assistant", '{"r":"A2"}'],
      ["user", "Third."],
      ["assistant", '{"r":"A3"}'],
    ]);
  });
});

describe("template substitution", () => {
  /** One state, one prompt, every value shape a hole can hold. */
  const slots = {
    s: { schema: { type: "string" } },
    n: { schema: { type: "number" } },
    b: { schema: { type: "boolean" } },
    arr: { schema: { type: "array", items: { type: "string" } } },
    obj: { schema: { type: "object" } },
    maybe: { schema: { type: "string" }, optional: true },
  } as const;
  const wired = Object.fromEntries(Object.keys(slots).map((k) => [k, `.inputs.${k}`]));

  function substitutionFiles(prompt: string): Record<string, StateDef> {
    return {
      root: {
        inputs: { ...slots },
        children: { one: { state: "leaf", inputs: wired } },
        sequence: ["one"],
        outputs: { r: { binding: ".children.one.output.r" } },
      },
      leaf: {
        inputs: { ...slots },
        outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
        operation: { kind: "prompt", model: "m", prompt, output: { r: { schema: { type: "string" } } } },
      },
    };
  }

  const values = { s: "S", n: 42, b: false, arr: ["x", "y"], obj: { k: 1 } };

  it("renders each value shape as the model will read it", async () => {
    const prompt = [
      "str=[{{.inputs.s}}]",
      "num=[{{.inputs.n}}]",
      "bool=[{{.inputs.b}}]",
      "arr=[{{.inputs.arr}}]",
      "obj=[{{.inputs.obj}}]",
    ].join("\n");
    const { engine, fake } = makeEngine(substitutionFiles(prompt), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: values })).outcome).toBe("success");
    // A scalar is stringified; a container is JSON, compactly — `false` and `0` survive as themselves
    // rather than being read as absent.
    expect(promptOf(fake.calls[0]!)).toBe(["str=[S]", "num=[42]", "bool=[false]", 'arr=[["x","y"]]', 'obj=[{"k":1}]'].join("\n"));
  });

  it("renders an absent optional and an unknown path as nothing, not as a marker", async () => {
    const prompt = "absent=[{{.inputs.maybe}}] unknown=[{{.inputs.nope}}]";
    const { engine, fake } = makeEngine(substitutionFiles(prompt), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: values })).outcome).toBe("success");
    // An empty hole is the one substitution failure a prompt cannot report on its own, which is why
    // the case below — a hole that cannot even be LOWERED — is a run failure instead of an empty one.
    expect(promptOf(fake.calls[0]!)).toBe("absent=[] unknown=[]");
  });

  it("tolerates whitespace in a hole, and leaves a non-hole alone", async () => {
    const prompt = "spaced=[{{ .inputs.s }}] expr=[{{ 1 + 1 }}] empty=[{{}}] lone=[{ .inputs.s }]";
    const { engine, fake } = makeEngine(substitutionFiles(prompt), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: values })).outcome).toBe("success");
    // Only a dotted-path hole is a hole. Anything else is prose the author meant to send.
    expect(promptOf(fake.calls[0]!)).toBe("spaced=[S] expr=[{{ 1 + 1 }}] empty=[{{}}] lone=[{ .inputs.s }]");
  });

  it("FAILS the run on a hole that cannot be lowered, rather than rendering it empty", async () => {
    // `{{inputs.s}}` — the missing dot. It is a path shape, so it lowers as a document reference and
    // resolves nowhere, which is an authoring error: rendering it empty would silently drop a
    // variable from the prompt and produce a plausible answer to a question missing its subject.
    const { engine, fake } = makeEngine(substitutionFiles("nodot=[{{inputs.s}}]"), "root", () => ok({ r: "A" }));
    const result = await engine.run({ inputs: values });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/'inputs\.s' resolves to no document.*did you mean '\.inputs\.s'/);
    expect(fake.calls).toHaveLength(0); // and nothing was sent
  });

  it("renders an artifact as its CONTENT, not its name or path", async () => {
    const { engine, fake } = makeEngine(artifactFiles({}), "root", blobThenJson);
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(promptOf(fake.calls[1]!)).toBe("Review: [# Doc body]");
  });

  it("an operation's own render variable shadows the state input of the same name", async () => {
    const files: Record<string, StateDef> = {
      root: {
        inputs: { s: { schema: { type: "string" } } },
        children: { one: { state: "leaf", inputs: { s: ".inputs.s" } } },
        sequence: ["one"],
        outputs: { r: { binding: ".children.one.output.r" } },
      },
      leaf: {
        inputs: { s: { schema: { type: "string" } } },
        outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
        operation: {
          kind: "prompt",
          model: "m",
          // Authored render variables ride BOUND INPUT SLOTS (loader §3.1) — there is no separate
          // "variables" field on the op shape, so a literal is a `{ text }` binding on a slot.
          input: { s: { binding: { text: "OP" } }, extra: { binding: { text: "X" } } },
          prompt: "shadowed=[{{.inputs.s}}] opOnly=[{{.inputs.extra}}]",
          output: { r: { schema: { type: "string" } } },
        },
      },
    };
    const { engine, fake } = makeEngine(files, "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: { s: "STATE" } })).outcome).toBe("success");
    expect(promptOf(fake.calls[0]!)).toBe("shadowed=[OP] opOnly=[X]");
  });

  /**
   * The one channel substitution does NOT reach.
   *
   * `system` is carried from the declaration to the op untouched (`loader.ts`: `op.system =
   * decl.system`), while `user` goes through `renderTemplate`. Pinned as it BEHAVES rather than as it
   * probably should: an author who writes a hole in `system` today ships the braces to the model, and
   * that is worth failing a test the day it changes in either direction.
   */
  it("does NOT substitute into `system` — a hole there ships as literal braces", async () => {
    const files: Record<string, StateDef> = {
      root: {
        inputs: { s: { schema: { type: "string" } } },
        children: { one: { state: "leaf", inputs: { s: ".inputs.s" } } },
        sequence: ["one"],
        outputs: { r: { binding: ".children.one.output.r" } },
      },
      leaf: {
        inputs: { s: { schema: { type: "string" } } },
        outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
        operation: {
          kind: "prompt",
          model: "m",
          system: "You are terse. subject={{.inputs.s}}",
          prompt: "Do it for {{.inputs.s}}.",
          output: { r: { schema: { type: "string" } } },
        },
      },
    };
    const { engine, fake } = makeEngine(files, "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: { s: "S" } })).outcome).toBe("success");
    expect(fake.calls[0]!.op.system).toBe("You are terse. subject={{.inputs.s}}");
    expect(promptOf(fake.calls[0]!)).toBe("Do it for S."); // the user slot DID render
  });
});

// --- the artifact fixture, shared by the substitution and preamble suites ------

const MD = { type: "string", contentMediaType: "text/markdown" } as const;

/** A writer that produces a markdown artifact, and a reader that receives it under `consumerEnv`. */
function artifactFiles(consumerEnv: Record<string, unknown>): Record<string, StateDef> {
  return {
    root: {
      environment: { session: "chat" },
      children: { w: { state: "writer" }, r: { state: "reader", inputs: { doc: ".children.w.output.doc" } } },
      sequence: ["w", "r"],
      outputs: { r: { binding: ".children.r.output.r" } },
    },
    writer: {
      // A blob-kind STATE output is what registers the artifact; `.operation.output` (not `.doc`)
      // because a lone `kind` output means the return IS the value, not a record with one field.
      outputs: { doc: { kind: "blob", schema: MD, binding: ".operation.output" } },
      operation: { kind: "prompt", model: "m", prompt: "Write it.", output: { doc: { kind: "blob", schema: MD } } },
    },
    reader: {
      inputs: { doc: { kind: "blob", schema: MD } },
      outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
      environment: consumerEnv as never,
      operation: { kind: "prompt", model: "m", prompt: "Review: [{{.inputs.doc}}]", output: { r: { schema: { type: "string" } } } },
    },
  };
}

const blobThenJson: Script = (call: FakeCall) => (call.op.output.kind === "blob" ? ok("# Doc body") : ok({ r: "A" }));

describe("which conversation a call joins", () => {
  function pair(readerEnv: Record<string, unknown>): Record<string, StateDef> {
    return {
      root: {
        environment: { session: "chat" },
        children: { w: { state: "writer" }, r: { state: "reader" } },
        sequence: ["w", "r"],
        outputs: { r: { binding: ".children.r.output.r" } },
      },
      writer: say("First."),
      reader: { ...say("Second."), environment: readerEnv as never },
    };
  }

  async function secondPromptUnder(env: Record<string, unknown>): Promise<string> {
    const { engine, fake } = makeEngine(pair(env), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    return promptOf(fake.calls[1]!);
  }

  /** The conversation the second call resolved to — the decision the prompt used to reveal. */
  async function secondSessionUnder(env: Record<string, unknown>): Promise<string> {
    const { engine, fake } = makeEngine(pair(env), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    const at = fake.calls[1]!.ctx.session!;
    return `${at.at.id}@${at.at.seq}`;
  }

  /**
   * WHICH conversation a call joins is still decided here; what it SENDS no longer varies with it.
   *
   * Every one of these used to be readable off the prompt, because a shared session put its history
   * in front of the next call and an isolated one did not. That was the preamble, and it was a second
   * copy of what the transport already carries. So they assert the decision at its source now — the
   * session the call resolved to — and the prompt reads the same either way, which is the point.
   */
  it("an inherited session name joins the stream", async () => {
    expect(await secondPromptUnder({})).toBe("Second.");
    expect(await secondSessionUnder({})).toBe("chat@1");
  });

  it("a session name of its own isolates it", async () => {
    expect(await secondSessionUnder({ session: "other" })).toBe("other@0");
  });

  it("a null session starts a fresh stream, beating the inherited name", async () => {
    expect(await secondSessionUnder({ session: null })).not.toMatch(/^chat@/);
  });

  it("a fork branches rather than continuing, and still reads the prefix", async () => {
    const { engine, fake, sessions } = makeEngine(pair({ fork: true }), "root", () => ok({ r: "A" }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    const at = fake.calls[1]!.ctx.session!;
    expect(at.mode).toBe("fork");
    expect(at.at.id).not.toBe("chat");
    // A branch is its parent's records up to the cursor — the prefix is readable, it is simply not
    // pasted into the prompt.
    expect(await sessions.messages(sessions.refAt(at.at))).toEqual([{ role: "user", content: "First." }, { role: "assistant", content: '{"r":"A"}' }]);
  });

  /**
   * A failed call is a RECORD — it is evidence and it cost money — but it is not a TURN. It has no
   * answer, so recording it would put a question with no reply into every later preamble in the
   * conversation, and under `full_history` every later call would read it back.
   */
  it("a failed call leaves no turn behind", async () => {
    const files: Record<string, StateDef> = {
      root: {
        environment: { session: "chat" },
        children: {
          a: { state: "first" },
          b: { state: "boom", transitions: [{ when: ".children.b.outcome === 'error'", to: "c" }] },
          c: { state: "last" },
        },
        sequence: ["a", "b", "c"],
        outputs: { r: { binding: ".children.c.output.r" } },
      },
      first: say("First."),
      boom: say("Second."),
      last: say("Third."),
    };
    const { engine, fake, sessions } = makeEngine(files, "root", (call) =>
      promptOf(call).endsWith("Second.")
        ? { error: { classification: "permanent", reason: "nope" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } }
        : ok({ r: "A" }),
    );
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");

    // The failing call DID see the history before it...
    expect(fake.calls).toHaveLength(3);
    expect(promptOf(fake.calls[1]!)).toBe("Second.");
    // The TRANSCRIPT is where the absence shows: the failed call left no turn between the other two.
    expect(promptOf(fake.calls[2]!)).toBe("Third.");
    expect((sessions.messages("chat") as Array<{ content: string }>).map((t) => t.content)).toEqual([
      "First.",
      '{"r":"A"}',
      "Third.",
      '{"r":"A"}',
    ]);
  });
});

describe("the other channels a call carries", () => {
  /** The structured-output CONTRACT is part of what is sent, and its shape is decided by the op's outputs. */
  it("lowers a multi-name output to one JSON object contract, every field required", async () => {
    const files: Record<string, StateDef> = {
      root: { children: { one: { state: "leaf" } }, sequence: ["one"], outputs: { r: { binding: ".children.one.output.r" } } },
      leaf: {
        outputs: { r: { schema: { type: "string" }, binding: ".operation.output.r" } },
        operation: {
          kind: "prompt",
          model: "m",
          prompt: "Go.",
          output: { r: { schema: { type: "string" } }, n: { schema: { type: "number" } } },
        },
      },
    };
    const { engine, fake } = makeEngine(files, "root", () => ok({ r: "A", n: 1 }));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(fake.calls[0]!.op.output).toEqual({
      name: "output",
      kind: "json",
      schema: { type: "object", properties: { r: { type: "string" }, n: { type: "number" } }, required: ["r", "n"] },
    });
  });

  it("leaves a lone text or blob output as the value itself, not a record with one field", async () => {
    const files: Record<string, StateDef> = {
      root: {
        children: { t: { state: "textLeaf" }, b: { state: "blobLeaf" } },
        sequence: ["t", "b"],
        outputs: { r: { binding: ".children.t.output.r" } },
      },
      textLeaf: {
        outputs: { r: { schema: { type: "string" }, binding: ".operation.output" } },
        operation: { kind: "prompt", model: "m", prompt: "Text please.", output: { answer: { kind: "text", schema: { type: "string" } } } },
      },
      blobLeaf: {
        outputs: { d: { kind: "blob", schema: MD, binding: ".operation.output" } },
        operation: { kind: "prompt", model: "m", prompt: "Blob please.", output: { doc: { kind: "blob", schema: MD } } },
      },
    };
    const { engine, fake } = makeEngine(files, "root", () => ok("plain"));
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    // The KIND is what the lowering dispatches on: a `text` call takes the text path and a `blob`
    // call the bytes path, and neither gets a structured-output contract attached.
    expect(fake.calls[0]!.op.output).toEqual({ name: "answer", kind: "text", schema: { type: "string" } });
    expect(fake.calls[1]!.op.output).toEqual({ name: "doc", kind: "blob", schema: MD });
  });

  it("hands the call the tools its operation declared, and only those", async () => {
    const files: Record<string, StateDef> = {
      root: {
        children: { withTool: { state: "toolLeaf" }, without: { state: "bareLeaf" } },
        sequence: ["withTool", "without"],
        outputs: { r: { binding: ".children.without.output.r" } },
      },
      toolLeaf: say("Go.", { operation: { kind: "prompt", model: "m", prompt: "Go.", tools: ["read_file"], output: { r: { schema: { type: "string" } } } } as never }),
      bareLeaf: say("Also go."),
    };
    const { engine, fake } = makeEngine(files, "root", () => ok({ r: "A" }), {
      read_file: { description: "reads a file", inputSchema: { type: "object" }, run: () => ok({}) },
      write_file: { description: "writes a file", inputSchema: { type: "object" }, run: () => ok({}) },
    });
    expect((await engine.run({ inputs: {} })).outcome).toBe("success");
    expect(toolNamesOf(fake.calls[0]!)).toEqual(["read_file"]);
    expect(toolNamesOf(fake.calls[1]!)).toEqual([]);
  });
});
