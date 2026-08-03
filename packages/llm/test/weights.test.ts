import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { ModelInfo, type ModelInfoInterface } from "../src/model-catalog.js";
import { WeightsCredentialRequired, WeightsStore, WeightsUnavailable, fileNameFor, urlFor } from "../src/weights.js";

/**
 * The weights store, against a real HTTP origin — resume, gating, integrity and coalescing are all
 * transport behaviour, so a stubbed fetch would prove none of them.
 */

const BODY = Buffer.from("GGUF".repeat(4096)); // 16 KiB of stand-in weights
const SHA = createHash("sha256").update(BODY).digest("hex");

let server: Server;
let origin: string;
let dir: string;
/** Set per test to steer the origin: gating, range support, failure. */
let mode: { gated?: boolean; ignoreRange?: boolean; cutAfter?: number } = {};
let requests: { url: string; range: string | undefined; auth: string | undefined }[] = [];

beforeAll(async () => {
  server = createServer((req, res) => {
    requests.push({ url: req.url ?? "", range: req.headers.range, auth: req.headers.authorization });
    if (mode.gated && req.headers.authorization !== "Bearer good-token") {
      res.writeHead(401).end();
      return;
    }
    if (req.url?.includes("missing")) {
      res.writeHead(404).end();
      return;
    }
    const range = mode.ignoreRange ? undefined : req.headers.range;
    const from = range ? Number(/bytes=(\d+)-/.exec(range)?.[1] ?? 0) : 0;
    const slice = BODY.subarray(from);
    const payload = mode.cutAfter !== undefined ? slice.subarray(0, mode.cutAfter) : slice;
    res.writeHead(from > 0 && !mode.ignoreRange ? 206 : 200, { "content-length": String(payload.length) });
    res.end(payload);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(async () => {
  await new Promise<void>((r) => server.close(() => r()));
  rmSync(dir, { recursive: true, force: true });
});

beforeEach(() => {
  mode = {};
  requests = [];
  dir = path.join(process.env.TEMP ?? "/tmp", `dai-weights-${Math.random().toString(36).slice(2)}`);
});

/** A catalog holding just the rows a test declares. */
const catalogOf = (rows: Partial<ModelInfoInterface>[]): Pick<typeof ModelInfo.instance, "lookup"> => {
  const table = new ModelInfo(
    rows.map((r) => ({ route: "embedded", model: "m", inputPerMillion: 0, outputPerMillion: 0, ...r }) as ModelInfoInterface),
  );
  return { lookup: (id: string) => table.lookup(id) } as Pick<typeof ModelInfo.instance, "lookup">;
};

describe("reference resolution", () => {
  it("maps a HuggingFace reference onto a resolve URL", () => {
    expect(urlFor({ source: "hf", uri: "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen.gguf" }, "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen.gguf")).toBe(
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen.gguf",
    );
  });

  it("rejects a malformed hf reference rather than building a nonsense URL", () => {
    expect(() => urlFor({ source: "hf", uri: "just-a-name" }, "just-a-name")).toThrow(WeightsUnavailable);
  });

  it("keeps the basename readable while making the ORIGIN distinguishable", () => {
    // Two repos both publishing `model-q4_k_m.gguf` must not collide on one path.
    const a = fileNameFor({ source: "hf", uri: "orgA/repo/model-q4_k_m.gguf" }, "orgA/repo/model-q4_k_m.gguf");
    const b = fileNameFor({ source: "hf", uri: "orgB/repo/model-q4_k_m.gguf" }, "orgB/repo/model-q4_k_m.gguf");
    expect(a).not.toBe(b);
    expect(a).toMatch(/model-q4_k_m\.gguf$/);
  });
});

describe("WeightsStore", () => {
  const storeFor = (rows: Partial<ModelInfoInterface>[], over: Record<string, unknown> = {}) =>
    new WeightsStore({ directory: dir, catalog: catalogOf(rows), ...over });

  it("downloads, verifies the checksum, and reports progress", async () => {
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, sha256: SHA }] }]);
    const seen: number[] = [];
    const withProgress = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, sha256: SHA }] }], {
      onProgress: (p: { receivedBytes: number }) => seen.push(p.receivedBytes),
    });
    const file = await withProgress.ensure("embedded/m");
    expect(readFileSync(file).equals(BODY)).toBe(true);
    expect(seen.at(-1)).toBe(BODY.length);
    expect(store.present("embedded/m")).toBe(true);
  });

  it("RESUMES a partial file instead of starting over", async () => {
    mkdirSync(dir, { recursive: true });
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, sha256: SHA }] }]);
    // Seed a half-finished `.part`, as an interrupted 20 GB fetch would leave behind.
    const half = BODY.length / 2;
    writeFileSync(`${store.pathFor("embedded/m")}.part`, BODY.subarray(0, half));
    const file = await store.ensure("embedded/m");
    expect(readFileSync(file).equals(BODY)).toBe(true); // reassembled correctly...
    expect(requests.at(-1)!.range).toBe(`bytes=${half}-`); // ...by asking for the remainder only
  });

  it("RESTARTS when the server ignores the range request", async () => {
    // A server answering 200-with-the-whole-body to a Range request is the corruption case: appending
    // that to what we already have yields a file of the wrong length that still looks complete.
    mkdirSync(dir, { recursive: true });
    mode.ignoreRange = true;
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, sha256: SHA }] }]);
    writeFileSync(`${store.pathFor("embedded/m")}.part`, BODY.subarray(0, 1024));
    const file = await store.ensure("embedded/m");
    expect(readFileSync(file).equals(BODY)).toBe(true);
  });

  it("refuses to publish a file that fails its checksum", async () => {
    // Verification happens BEFORE the rename, because the un-suffixed name is what `present()` reads —
    // letting corruption occupy it would cache the corruption permanently.
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, sha256: "0".repeat(64) }] }]);
    await expect(store.ensure("embedded/m")).rejects.toThrow(/checksum mismatch/);
    expect(store.present("embedded/m")).toBe(false);
    expect(existsSync(`${store.pathFor("embedded/m")}.part`)).toBe(false); // and the bad bytes are gone
  });

  it("names a GATED repo as a credential problem, not a missing file", async () => {
    mode.gated = true;
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf`, gated: true }] }]);
    await expect(store.ensure("embedded/m")).rejects.toBeInstanceOf(WeightsCredentialRequired);
    // The remedy differs completely from "not found", which is why the type does.
    await expect(store.ensure("embedded/m")).rejects.toMatchObject({ kind: "credential-required", scope: "url" });
  });

  it("succeeds once the caller supplies a token — including from an async provider", async () => {
    mode.gated = true;
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf` }] }], {
      token: () => Promise.resolve("good-token"), // a host prompting its user, or reading a keychain
    });
    await expect(store.ensure("embedded/m")).resolves.toBeTruthy();
    expect(requests.at(-1)!.auth).toBe("Bearer good-token");
  });

  it("fetches EVERY part of a split model", async () => {
    // A downloader that fetched only the first part leaves a file that looks complete and loads to an
    // error — the exact failure my own 70B spike URL hit.
    const store = storeFor([
      {
        model: "m",
        downloads: [
          { source: "url", uri: `${origin}/a-00001-of-00003.gguf`, parts: [`${origin}/a-00002-of-00003.gguf`, `${origin}/a-00003-of-00003.gguf`] },
        ],
      },
    ]);
    expect(store.present("embedded/m")).toBe(false);
    const primary = await store.ensure("embedded/m");
    expect(primary).toMatch(/00001-of-00003/); // the returned path is the FIRST part
    expect(store.present("embedded/m")).toBe(true); // ...and presence means all three
    expect(requests).toHaveLength(3);
  });

  it("COALESCES concurrent requests onto one download", async () => {
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf` }] }]);
    const paths = await Promise.all([1, 2, 3, 4].map(() => store.ensure("embedded/m")));
    expect(new Set(paths).size).toBe(1);
    expect(requests).toHaveLength(1); // four callers, one fetch — a second writer would corrupt the first
  });

  it("skips work entirely when the file is already there", async () => {
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf` }] }]);
    await store.ensure("embedded/m");
    const after = requests.length;
    await store.ensure("embedded/m");
    expect(requests).toHaveLength(after);
  });

  it("reports HTTP failure against the model, not as a crash", async () => {
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/missing.gguf` }] }]);
    await expect(store.ensure("embedded/m")).rejects.toThrow(/HTTP 404/);
  });

  it("a `file` location is used in place, never copied", async () => {
    mkdirSync(dir, { recursive: true });
    const local = path.join(dir, "already-here.gguf");
    writeFileSync(local, BODY);
    const store = storeFor([{ model: "m", downloads: [{ source: "file", uri: local }] }]);
    expect(store.present("embedded/m")).toBe(true);
    expect(await store.ensure("embedded/m")).toBe(local);
    expect(requests).toHaveLength(0);
  });

  it("says NOT MINE for a model it knows nothing about", () => {
    // `undefined` rather than `false` is what lets hw ask about every model without learning what a
    // route is — the caller's predicate answers "not something I manage".
    const store = storeFor([{ model: "m", downloads: [{ source: "url", uri: `${origin}/m.gguf` }] }]);
    expect(store.present("anthropic/claude-haiku-4-5")).toBeUndefined();
    expect(store.pathFor("anthropic/claude-haiku-4-5")).toBeUndefined();
  });

  it("a row with no downloads is unavailable, and says why", async () => {
    const store = storeFor([{ model: "m", openWeights: true }]);
    await expect(store.ensure("embedded/m")).rejects.toThrow(/carries no `downloads` entries/);
  });
});
