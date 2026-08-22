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
 *  - an entry that declares no signature still resolves — it simply has no positions to bind.
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
