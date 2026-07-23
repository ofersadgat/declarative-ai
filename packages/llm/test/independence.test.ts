/**
 * DESIGN §2's package-independence rules, as tests rather than as prose:
 *
 *  - `npm i @declarative-ai/llm` installs NO ajv — asserted over the declared dependency closure;
 *  - a structured LLM call runs with `json + llm` and nothing else — asserted by actually making one
 *    through the direct path, importing nothing from `exec`, `ops`, `validate`, or `promptop`.
 *
 * The coupling this replaces was packaging, not code: `services/index.ts` re-exported the ajv
 * validator, so a single `import { systemClock }` in llm's executor dragged ajv into the MODULE graph.
 * Which is why the package-edge assertions are not enough on their own and there is a MODULE-graph
 * assertion below: a package.json can look clean while the imports do not.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import type { OutputValidator } from "@declarative-ai/json";
import { executeLlmCall } from "../src/call";
import type { LlmCallDefinition } from "../src/llmConfig";
import { fakeRouter, stream, usage, errorOf } from "./fakes";

const ROOT = path.resolve(__dirname, "../../..");

/**
 * The transitive closure of DECLARED dependencies, workspace links followed.
 *
 * A declared dependency that cannot be RESOLVED throws rather than being skipped. Swallowing the read
 * error truncated the walk for any package not hoisted to the root `node_modules` — nothing misses
 * today, but a future nested install would silently shrink the closure and make the "no ajv" assertion
 * pass VACUOUSLY, which is the one failure mode an exit-criterion test must not have.
 */
function dependencyClosure(start: string, seen = new Set<string>(), via: string[] = []): Set<string> {
  if (seen.has(start)) return seen;
  seen.add(start);
  const local = start.startsWith("@declarative-ai/") ? path.join(ROOT, "packages", start.slice("@declarative-ai/".length)) : path.join(ROOT, "node_modules", start);
  const manifest = path.join(local, "package.json");
  let pkg: { dependencies?: Record<string, string> };
  try {
    pkg = JSON.parse(readFileSync(manifest, "utf8")) as typeof pkg;
  } catch (e) {
    throw new Error(
      `cannot resolve declared dependency "${start}" (via ${[...via, start].join(" → ")}) at ${manifest}: ${(e as Error).message}. ` +
        "The closure walk must never silently skip a package — a skipped edge makes the 'no ajv' assertion vacuous.",
    );
  }
  for (const dep of Object.keys(pkg.dependencies ?? {})) dependencyClosure(dep, seen, [...via, start]);
  return seen;
}

/** Every `.ts` file under a directory, recursively. */
function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) return sourceFiles(full);
    return e.isFile() && e.name.endsWith(".ts") ? [full] : [];
  });
}

/** The workspace packages `llm` must never reach — the layers ABOVE it. */
const FORBIDDEN = ["@declarative-ai/exec", "@declarative-ai/ops", "@declarative-ai/validate", "@declarative-ai/promptop"] as const;

describe("§8 exit criteria — package independence", () => {
  it("installs no ajv: it is not in llm's dependency closure at all", () => {
    const closure = dependencyClosure("@declarative-ai/llm");
    expect([...closure]).not.toContain("ajv");
    // Only `validate` may carry the heavy dependency.
    expect([...dependencyClosure("@declarative-ai/validate")]).toContain("ajv");
  });

  it("depends on exactly ONE workspace package — json, the bottom of the graph", () => {
    const workspace = [...dependencyClosure("@declarative-ai/llm")].filter((d) => d.startsWith("@declarative-ai/"));
    expect(workspace.sort()).toEqual(["@declarative-ai/json", "@declarative-ai/llm"]);
    // …and the MANIFEST says so directly, not just transitively: exactly one declared workspace edge.
    // The closure walk alone would still pass if llm gained a second workspace dependency that happened
    // to depend only on json.
    const manifest = JSON.parse(readFileSync(path.join(ROOT, "packages/llm/package.json"), "utf8")) as { dependencies?: Record<string, string> };
    expect(Object.keys(manifest.dependencies ?? {}).filter((d) => d.startsWith("@declarative-ai/"))).toEqual(["@declarative-ai/json"]);
  });

  it("no file in llm's MODULE graph imports a layer above it", () => {
    // The coupling class this whole file is about was never a package edge: `services/index.ts`
    // re-exported the ajv validator, so ONE `import { systemClock }` in llm's executor put ajv in llm's
    // module graph while `package.json` still looked clean. A dependency-closure assertion cannot see
    // that; only reading the imports can. (A package.json edge can also be added by hand and forgotten
    // — the source is the ground truth for what is actually reached.)
    const offenders: string[] = [];
    for (const file of sourceFiles(path.join(ROOT, "packages/llm/src"))) {
      const src = readFileSync(file, "utf8");
      for (const pkg of FORBIDDEN) {
        // Matches `from "pkg"`, `from "pkg/sub"`, `import("pkg")`, and `require("pkg")` alike.
        if (new RegExp(String.raw`(?:from|import|require)\s*\(?\s*["']${pkg}(?:/[^"']*)?["']`).test(src)) {
          offenders.push(`${path.relative(ROOT, file)} → ${pkg}`);
        }
      }
    }
    expect(offenders).toEqual([]);
    // A live counter-witness that the matcher above actually matches something: promptop, the layer
    // that legitimately sits on both, DOES import `exec`.
    const promptopSrc = sourceFiles(path.join(ROOT, "packages/promptop/src")).map((f) => readFileSync(f, "utf8"));
    expect(promptopSrc.some((s) => /from\s+["']@declarative-ai\/exec["']/.test(s))).toBe(true);
  });

  it("runs a STRUCTURED call end to end with json + llm and nothing else", async () => {
    const model = new MockLanguageModelV3({
      doStream: async () =>
        stream([
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "1" },
          { type: "text-delta", id: "1", delta: '{"answer":"4"}' },
          { type: "text-end", id: "1" },
          { type: "finish", finishReason: "stop", usage: usage(10, 5) },
        ]),
    });
    // A hand-rolled validator: `OutputValidator` is `json`'s three-line seam, so the boundary check
    // needs no ajv and no `validate` import.
    const seen: unknown[] = [];
    const validator: OutputValidator = (() => ({
      validateValue: (_schema, value) => {
        seen.push(value);
        return { ok: true };
      },
    }))();

    const def: LlmCallDefinition<{ answer: string }> = {
      model: "anthropic/claude-haiku-4-5",
      prompt: "what is 2+2?",
      schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
      timeoutMs: 30_000,
    };
    const out = await executeLlmCall(def, { modelRouter: fakeRouter(model), validator });

    expect(errorOf(out)).toBeUndefined();
    expect(out.value?.value).toEqual({ answer: "4" });
    expect(seen).toEqual([{ answer: "4" }]); // the boundary check ran
  });

  it("a TEXT call yields `string` — a text call produces text (§5.2)", async () => {
    const model = new MockLanguageModelV3({
      doStream: async () =>
        stream([
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "1" },
          { type: "text-delta", id: "1", delta: "four" },
          { type: "text-end", id: "1" },
          { type: "finish", finishReason: "stop", usage: usage(1, 1) },
        ]),
    });
    const out = await executeLlmCall({ model: "anthropic/claude-haiku-4-5", prompt: "what is 2+2?" }, { modelRouter: fakeRouter(model) });
    // The overload discriminates on the ABSENCE of a schema, so this is `LlmCallResult<string>` — it
    // previously claimed `JsonValue`, which was simply wrong.
    const value: string | undefined = out.value?.value;
    expect(value).toBe("four");
  });
});
