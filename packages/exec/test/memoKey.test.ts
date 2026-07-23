import { describe, expect, it } from "vitest";
import type { InlineFamily, Operation } from "@declarative-ai/ops";
import { canonicalize, sha256Hex } from "@declarative-ai/ops";
import { hashOperation, memoKey } from "../src/memo";

/** A resolved prompt op: every input bound to a literal, which is what makes the OP the memo identity. */
function op(user: string, style: string): Operation<InlineFamily> {
  return {
    kind: "prompt",
    user,
    config: { model: "anthropic/claude-haiku-4-5" },
    input: { style: { kind: "text", binding: { text: style } } },
    output: { name: "output", kind: "json" },
  };
}

describe("memoKey (DESIGN §3.4)", () => {
  it("is the sha256 of the canonical {operationHash} record", () => {
    const operationHash = "op-hash";
    expect(memoKey({ operationHash })).toBe(sha256Hex(canonicalize({ operationHash })));
  });

  it("omits workspaceTreeHash entirely when absent (a pure op keys without it)", () => {
    expect(memoKey({ operationHash: "h" })).not.toBe(memoKey({ operationHash: "h", workspaceTreeHash: "t" }));
    expect(memoKey({ operationHash: "h" })).toBe(sha256Hex(canonicalize({ operationHash: "h" })));
  });

  it("changes with the workspace snapshot for a side-effecting op", () => {
    expect(memoKey({ operationHash: "h", workspaceTreeHash: "a" })).not.toBe(
      memoKey({ operationHash: "h", workspaceTreeHash: "b" }),
    );
  });

  // The op hash says what was ASKED, never who answered — so without this component two executors
  // sharing one cache (different routing, different registries, a stub and the real thing) collide on
  // every identical op and serve each other's results.
  it("changes with the executor identity, and omits it entirely when absent", () => {
    expect(memoKey({ operationHash: "h", executorId: "a" })).not.toBe(memoKey({ operationHash: "h", executorId: "b" }));
    expect(memoKey({ operationHash: "h", executorId: "a" })).not.toBe(memoKey({ operationHash: "h" }));
    expect(memoKey({ operationHash: "h" })).toBe(sha256Hex(canonicalize({ operationHash: "h" })));
  });
});

describe("hashOperation", () => {
  it("is stable across key insertion order (JCS canonicalization)", () => {
    const a: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "f",
      input: { x: { kind: "json", binding: { json: { a: 1, b: 2 } } } },
      output: { name: "output", kind: "json" },
    };
    const b: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "f",
      input: { x: { kind: "json", binding: { json: { b: 2, a: 1 } } } },
      output: { name: "output", kind: "json" },
    };
    expect(hashOperation(a)).toBe(hashOperation(b));
  });

  it("distinguishes ops whose RESOLVED inputs differ — the op embeds its inputs", () => {
    expect(hashOperation(op("summarize", "terse"))).not.toBe(hashOperation(op("summarize", "verbose")));
  });

  it("hashes blob bytes by content, not object identity", () => {
    const withBytes = (bytes: Uint8Array): Operation<InlineFamily> => ({
      kind: "function",
      functionRef: "upload",
      input: { file: { kind: "blob", binding: { blob: bytes } } },
      output: { name: "output", kind: "json" },
    });
    expect(hashOperation(withBytes(new Uint8Array([1, 2, 3])))).toBe(hashOperation(withBytes(new Uint8Array([1, 2, 3]))));
    expect(hashOperation(withBytes(new Uint8Array([1, 2, 3])))).not.toBe(hashOperation(withBytes(new Uint8Array([1, 2, 4]))));
  });

  it("refuses a live stream with the remedy instead of keying on object identity (§7.3)", () => {
    const streaming: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "upload",
      input: {
        file: {
          kind: "blob",
          binding: { blob: { getReader: () => ({ read: async () => ({ done: true }), cancel: async () => {}, releaseLock: () => {} }) } },
        },
      },
      output: { name: "output", kind: "json" },
    };
    expect(() => hashOperation(streaming)).toThrow(/materialize it to a Uint8Array/);
  });
});
