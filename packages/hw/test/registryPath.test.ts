/**
 * THE REGISTRY IS A CONTRIBUTOR ON THE SEARCH PATH (SPEC §7.1, §7.5).
 *
 * A callee name used to resolve five ways — built-ins, documents on the path, the module symbol
 * index, host-shipped `documents` "searched only where the path finds nothing", and the registry as
 * a separate namespace only `operation.function` could address. The last two are the ones that broke
 * the model: a fallback can only ever be LAST, so a project could not shadow a host function, and a
 * registry nothing could search meant a host had to ship an operation document beside its own
 * implementation just to have somewhere to declare how it is called.
 *
 * So the registry is an entry in the path list. What that buys, one test each:
 *
 *  - a registered function resolves as a callee, with no document anywhere;
 *  - its declared slots are what positional arguments bind against;
 *  - a project directory earlier on the path SHADOWS it, and the overlap is reported;
 *  - `$REGISTRY` written into the path moves it, so precedence is authored rather than compiled in;
 *  - an entry that declares no signature still resolves — it simply has no positions to bind;
 *  - `operation.function` resolves the SAME way, so a state's own call is not a separate namespace.
 */
import { describe, expect, it } from "vitest";
import type { EntrySignature, InlineFamily, Operation, Signature } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader.js";
import { REGISTRY_ROOT, resolveReference, type Vfs } from "../src/reference.js";
import type { StateDef } from "../src/format.js";

const OPS = "/p/ops";

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        names.add(rest.split("/")[0]!);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

/** `shout` takes one positional `text`; `plain` declares nothing at all. */
const entries = (): Map<string, EntrySignature> =>
  new Map<string, EntrySignature>([
    [
      "shout",
      {
        signature: {
          input: { text: { kind: "text", schema: { type: "string" }, index: 0 } },
          output: { name: "value", kind: "text" },
        } as Signature<InlineFamily>,
      },
    ],
    ["plain", {}],
  ]);

/** Load one state whose single output binds `expr`, and hand back the producer edge the call lowered to. */
function calleeOf(
  expr: string,
  opts: { files?: Record<string, string>; path?: readonly string[]; onWarn?: (m: string) => void } = {},
): { op: Operation<InlineFamily>; parameters?: Record<string, unknown> } {
  const state: StateDef = {
    outputs: { v: { binding: { expr } } },
    operation: { kind: "function", function: "plain" },
    // On `environment`, because the path a state RESOLVES against has to be known before its own
    // document is expanded — the loader reads it off the raw environment chain for that reason.
    ...(opts.path !== undefined ? { environment: { path: opts.path } } : {}),
  } as unknown as StateDef;
  const bundle = loadBundle({ s: state }, "s", {
    functions: entries(),
    defaultRoot: [OPS],
    ...(opts.files !== undefined ? { vfs: vfsOf(opts.files) } : {}),
    ...(opts.onWarn !== undefined ? { onWarn: opts.onWarn } : {}),
  });
  return bundle.states["s"]!.outputs!["v"]!.binding as unknown as { op: Operation<InlineFamily>; parameters?: Record<string, unknown> };
}

/** The operation half alone, for the cases that say nothing about the arguments. */
const opOf = (expr: string, opts?: Parameters<typeof calleeOf>[1]): Operation<InlineFamily> => calleeOf(expr, opts).op;

describe("a registered function is a callee", () => {
  it("resolves from an expression with no document anywhere", () => {
    expect(opOf("shout('hi')")).toMatchObject({ kind: "function", functionRef: "shout" });
  });

  it("binds positional arguments against the slots the ENTRY declares", () => {
    // The whole point: nothing in any file says that `shout`'s first argument is called `text`. The
    // registration does, which is what removed the need for a document restating it.
    expect(calleeOf("shout('hi')").parameters).toMatchObject({ text: { binding: { text: "hi" } } });
  });

  it("resolves with no filesystem at all — an in-memory bundle still has host functions", () => {
    expect(opOf("shout('hi')")).toMatchObject({ functionRef: "shout" });
  });

  it("still resolves an entry that declares NO signature, with no slots to bind", () => {
    // Every host function written before signatures existed is this case, so refusing it would make
    // putting the registry on the path a breaking change for all of them. What it costs is exactly
    // what it should: nothing said what the positions are, so there are none.
    expect(opOf("plain()")).toMatchObject({ functionRef: "plain", input: {} });
  });
});

describe("precedence is decided by position, not by a fallback order", () => {
  const projectShout = { [`${OPS}/shout.json`]: JSON.stringify({ kind: "function", function: "local_shout", input: { s: { kind: "text", index: 0 } } }) };

  it("lets a project document on an EARLIER entry shadow the registered function", () => {
    // The default position is last, which is the precedence host-shipped documents already had: what
    // the host provides is a default, not a reservation.
    expect(opOf("shout('hi')", { files: projectShout })).toMatchObject({ functionRef: "local_shout" });
  });

  it("reports the overlap, as it does for two directories", () => {
    const warnings: string[] = [];
    calleeOf("shout('hi')", { files: projectShout, onWarn: (m) => warnings.push(m) });
    expect(warnings.join("\n")).toMatch(/a registered function 'shout' also matches further along the path/);
  });

  it("puts the registry FIRST when the workflow's path says so", () => {
    // `["$REGISTRY", "$INHERITED"]` is the whole reason the sentinel is spelled in the path rather
    // than being a rule inside a loader: the precedence is one line in the workflow file.
    expect(opOf("shout('hi')", { files: projectShout, path: [REGISTRY_ROOT, OPS] })).toMatchObject({ functionRef: "shout" });
  });
});

describe("the sentinel in isolation", () => {
  it("answers only for names the predicate accepts", () => {
    const options = { defaultRoot: [REGISTRY_ROOT], registry: (n: string) => n === "shout" };
    expect(resolveReference("shout", options)).toMatchObject({ registry: "shout", id: "shout", local: false });
    expect(() => resolveReference("nope", options)).toThrow(/matches no file on the path/);
  });

  it("is not appended when no registry is supplied — nothing changes for a caller that has none", () => {
    // The append is conditional on there BEING a registry, so every resolution that predates this
    // walks exactly the path it always did.
    expect(() => resolveReference("shout", { defaultRoot: [OPS], vfs: vfsOf({}) })).toThrow();
  });
});

/**
 * `operation.function` used to mean "a name in `registry.functions`" and did not search — so a state's
 * own call was the one call in the system that could not reach a document or a module, and the
 * registry was a namespace only it could address. Both halves of that go away together.
 */
describe("a state's own operation resolves its callee the same way", () => {
  const stateBundle = (op: Record<string, unknown>, opts: { files?: Record<string, string> } = {}) =>
    loadBundle({ s: { operation: op } } as unknown as Record<string, StateDef>, "s", {
      functions: entries(),
      defaultRoot: [OPS],
      ...(opts.files !== undefined ? { vfs: vfsOf(opts.files) } : {}),
    });

  const stateOp = (op: Record<string, unknown>, opts: { files?: Record<string, string> } = {}): Operation<InlineFamily> =>
    stateBundle(op, opts).states["s"]!.operation!;

  /**
   * An operation the loader could not build is carried as DATA and reported by the validator, rather
   * than aborting the load — the treatment every incomplete operation already gets, so one bad state
   * does not hide the rest of a workflow's findings.
   */
  const complaintOf = (op: Record<string, unknown>, opts: { files?: Record<string, string> } = {}): string =>
    stateBundle(op, opts).states["s"]!.operationError ?? "";

  it("takes the callee's declared slots, so `args` bind against a typed parameter", () => {
    expect(stateOp({ kind: "function", function: "shout", args: { text: "hi" } })).toMatchObject({
      functionRef: "shout",
      input: { text: { kind: "text", schema: { type: "string" }, index: 0, binding: { text: "hi" } } },
    });
  });

  it("refuses an argument the callee has no slot for", () => {
    // The failure the `config` blob made unaskable: an argument nothing reads. Only checkable
    // because the callee declared what it accepts.
    expect(complaintOf({ kind: "function", function: "shout", args: { txt: "hi" } })).toMatch(
      /passes 'txt', which 'shout' does not accept/,
    );
  });

  it("says nothing about arguments to a callee that declared no slots", () => {
    // An entry that declared nothing has said nothing to disagree with.
    expect(stateOp({ kind: "function", function: "plain", args: { anything: 1 } })).toMatchObject({
      functionRef: "plain",
      input: { anything: { kind: "json", binding: { json: 1 } } },
    });
  });

  it("reaches a project DOCUMENT, and dispatches to the ref that document names", () => {
    const files = { [`${OPS}/review.json`]: JSON.stringify({ kind: "function", function: "plain", input: { doc: { kind: "text", index: 0 } } }) };
    expect(stateOp({ kind: "function", function: "review", args: { doc: "d" } }, { files })).toMatchObject({
      functionRef: "plain",
      input: { doc: { kind: "text", index: 0, binding: { text: "d" } } },
    });
  });

  it("lets a callee document declare slots for the registered function of its OWN name", () => {
    // `shout.json` naming `shout` is the ordinary shape of a document that types an implementation,
    // not a cycle: the inner name is suppressed while the outer one is being resolved, so the
    // document means what it reads as — THESE slots, dispatched to the registered `shout`.
    const files = { [`${OPS}/shout.json`]: JSON.stringify({ kind: "function", function: "shout", input: { loud: { kind: "text", index: 0 } } }) };
    expect(stateOp({ kind: "function", function: "shout", args: { loud: "hi" } }, { files })).toMatchObject({
      functionRef: "shout",
      input: { loud: { binding: { text: "hi" } } },
    });
  });

  it("keeps a name that resolves NOWHERE as a bare ref, for the validator to report", () => {
    // Not a load failure: a state the run never enters never needs its function, and leaving one
    // unregistered is how a search context refuses a human gate.
    expect(stateOp({ kind: "function", function: "absent" })).toMatchObject({ functionRef: "absent", input: {} });
  });

  it("refuses a `function` naming a PROMPT document, rather than dispatching one as the other", () => {
    const files = { [`${OPS}/classify.json`]: JSON.stringify({ kind: "prompt", prompt: "Classify: {{.inputs.text}}" }) };
    expect(complaintOf({ kind: "function", function: "classify" }, { files })).toMatch(/resolves to a prompt operation/);
  });
});

/**
 * An argument past the callee's last slot used to be dropped SILENTLY by `bindPositionally`.
 *
 * Tolerable while a callee's slots came from a document somebody wrote by hand beside the call. Not
 * now: they are read off a TypeScript parameter list or a registry entry, so a signature can change
 * under a call site that still type-checks, and the argument that stops arriving does so quietly.
 */
describe("a call is checked against the callee's arity", () => {
  const load = (expr: string): void => {
    loadBundle({ s: { outputs: { v: { binding: { expr } } }, operation: { kind: "function", function: "plain" } } } as unknown as Record<string, StateDef>, "s", {
      functions: entries(),
    });
  };

  it("accepts exactly as many arguments as there are slots", () => {
    expect(() => load("shout('hi')")).not.toThrow();
  });

  it("accepts FEWER — a free slot is filled by name, which is a different question", () => {
    expect(() => load("shout()")).not.toThrow();
  });

  it("refuses more, naming the slots there were", () => {
    expect(() => load("shout('hi', 'again')")).toThrow(/'shout' takes 1 argument \(text\), but 2 were given/);
  });

  it("says 'no arguments' for a callee that declared no slots at all", () => {
    expect(() => load("plain(1)")).toThrow(/'plain' takes no arguments, but 1 were given/);
  });
});
