/**
 * The WEIGHTS STORE (§5) — fetching the files an `embedded/…` model needs, from what the catalog says
 * about them.
 *
 * Two boundaries define this module.
 *
 * **It is node-only, and deliberately not in the catalog.** `model-catalog` is dependency-free and
 * published as its own `@declarative-ai/llm/model-catalog` subpath so a UI can read model identity and
 * pricing without dragging `undici`/`node:net` into a browser bundle. Weights METADATA belongs on the
 * row; the `fetch` and `node:fs` that act on it belong here, or that subpath stops being importable —
 * which is its whole reason for existing.
 *
 * **The caller owns the filesystem.** No XDG, no `~/.cache`, no "sensible default": a `directory` is
 * required. A library that picked one would scatter multi-gigabyte files somewhere the caller never
 * agreed to, and would make two consumers in one process silently share a cache.
 *
 * It does NOT need `node-llama-cpp`. Provisioning weights and running them are separate jobs — a CI
 * step or an installer should be able to pre-fetch a GGUF without installing 100 MB of native binaries.
 */
import { closeSync, createReadStream, existsSync, mkdirSync, openSync, renameSync, rmSync, statSync, writeSync } from "node:fs";
import { createHash } from "node:crypto";
import { ModelInfo, type WeightsLocation } from "./model-catalog.js";
import { createLogger } from "./logger.js";

const log = createLogger("engine.providers.weights");

/** Raised when a source needs a credential we do not have — the signal a host acts on by prompting for
 *  a token and constructing a new store with it. Distinct from "not found" because the remedies differ
 *  completely, and a 401 reported as a missing file sends the reader hunting the wrong problem. */
export class WeightsCredentialRequired extends Error {
  readonly kind = "credential-required";
  constructor(
    /** What the credential is FOR — `"hf"` today. A host maps this to its own settings key. */
    readonly scope: string,
    readonly uri: string,
  ) {
    super(`weights at "${uri}" require a credential for "${scope}" (the repo is gated, or the token is missing/invalid)`);
    this.name = "WeightsCredentialRequired";
  }
}

/** Raised when the catalog has nothing to download for a model. */
export class WeightsUnavailable extends Error {
  readonly kind = "unavailable";
  constructor(readonly modelId: string, reason: string) {
    super(`no weights available for "${modelId}": ${reason}`);
    this.name = "WeightsUnavailable";
  }
}

/** Progress for one file of a (possibly multi-part) model. */
export interface WeightsProgress {
  modelId: string;
  /** Destination filename currently being fetched. */
  file: string;
  /** 1-based index within a split model's part set. */
  part: number;
  parts: number;
  receivedBytes: number;
  /** Absent when the server reports no length. */
  totalBytes?: number;
}

/** A credential, per source. Async so a host can prompt or read a keychain. */
export type WeightsToken = string | ((scope: string) => string | undefined | Promise<string | undefined>);

export interface WeightsStoreOptions {
  /** Directory the weights live in. REQUIRED — see the module note on filesystem ownership. */
  directory: string;
  /** Credential for gated sources (a HuggingFace token). Held by the caller, never read from the
   *  environment by this library. */
  token?: WeightsToken;
  /** Transport seam (tests, proxies). */
  fetch?: typeof globalThis.fetch;
  onProgress?: (progress: WeightsProgress) => void;
  /** Catalog to read `downloads` from. Defaults to the runtime singleton. */
  catalog?: Pick<typeof ModelInfo.instance, "lookup">;
}

/** Everything a model needs on disk: the primary file plus any additional parts. */
interface Plan {
  location: WeightsLocation;
  files: { uri: string; dest: string }[];
}

/** A filesystem-safe name for a remote reference, keeping the basename readable and the origin
 *  distinguishable — two repos publishing `model-q4_k_m.gguf` must not collide on one path. */
export function fileNameFor(location: WeightsLocation, uri: string): string {
  if (location.source === "file") return uri;
  const parts = uri.split("/").filter((p) => p.length > 0);
  const base = parts.pop() ?? "weights.gguf";
  const prefix = parts.join("_").replace(/[^A-Za-z0-9._-]/g, "-");
  return prefix.length > 0 ? `${location.source}_${prefix}_${base}` : `${location.source}_${base}`;
}

/** The URL a reference resolves to. A HuggingFace `org/repo/path/to/file.gguf` becomes a `resolve/main`
 *  URL; the first two segments are the repo and everything after is the path within it. */
export function urlFor(location: WeightsLocation, uri: string): string {
  if (location.source === "url") return uri;
  if (location.source === "file") return `file://${uri}`;
  const segments = uri.split("/").filter((s) => s.length > 0);
  if (segments.length < 3) {
    throw new WeightsUnavailable(uri, `a "hf" reference must be "<org>/<repo>/<file>", got "${uri}"`);
  }
  const [org, repo, ...rest] = segments;
  return `https://huggingface.co/${org}/${repo}/resolve/main/${rest.join("/")}`;
}

export class WeightsStore {
  private readonly inFlight = new Map<string, Promise<string>>();

  constructor(private readonly options: WeightsStoreOptions) {}

  /** Where this model's PRIMARY file lives (whether or not it is there yet), or `undefined` when the
   *  catalog knows of no downloadable weights for it. */
  pathFor(modelId: string): string | undefined {
    const plan = this.planFor(modelId);
    return plan?.files[0]?.dest;
  }

  /**
   * Whether every file this model needs is already on disk — the predicate hw's validation takes.
   *
   * Synchronous on purpose. `validateBundle` is sync and backs a lint surface, so the check it can
   * perform is "is it here", never "go and get it". Returns `undefined` for a model this store knows
   * nothing about, which is how a caller says "not mine" without hw having to learn what a route is.
   */
  present(modelId: string): boolean | undefined {
    const plan = this.planFor(modelId);
    if (plan === undefined) return undefined;
    return plan.files.every((f) => existsSync(f.dest));
  }

  /**
   * Make sure the weights are on disk, downloading what is missing, and return the primary file's path.
   *
   * Concurrent calls for one model COALESCE onto a single download — two ops naming the same 20 GB GGUF
   * must not fetch it twice, and on a shared destination the second writer would corrupt the first.
   */
  ensure(modelId: string, signal?: AbortSignal): Promise<string> {
    const existing = this.inFlight.get(modelId);
    if (existing !== undefined) return existing;
    const run = this.fetchAll(modelId, signal).finally(() => this.inFlight.delete(modelId));
    this.inFlight.set(modelId, run);
    return run;
  }

  private planFor(modelId: string): Plan | undefined {
    const catalog = this.options.catalog ?? ModelInfo.instance;
    const row = catalog.lookup(modelId);
    const location = row?.downloads?.[0];
    if (location === undefined) return undefined;
    const uris = [location.uri, ...(location.parts ?? [])];
    return {
      location,
      // A `file` location is ALREADY where it belongs, so its destination is itself. Routing it through
      // the store directory would make `present()` look somewhere the bytes will never be, while
      // `ensure()` correctly returned the real path — the two disagreeing about one model.
      files: uris.map((uri) => ({
        uri,
        dest: location.source === "file" ? uri : joinPath(this.options.directory, fileNameFor(location, uri)),
      })),
    };
  }

  private async fetchAll(modelId: string, signal?: AbortSignal): Promise<string> {
    const plan = this.planFor(modelId);
    if (plan === undefined) {
      throw new WeightsUnavailable(modelId, "the catalog row carries no `downloads` entries");
    }
    // A `file` location is already local — nothing to fetch, and asserting otherwise would be a lie.
    if (plan.location.source === "file") {
      for (const f of plan.files) {
        if (!existsSync(f.uri)) throw new WeightsUnavailable(modelId, `the declared file "${f.uri}" does not exist`);
      }
      return plan.files[0]!.uri;
    }

    mkdirSync(this.options.directory, { recursive: true });
    for (const [index, file] of plan.files.entries()) {
      if (existsSync(file.dest)) continue;
      await this.fetchOne(modelId, plan, file, index, signal);
    }
    return plan.files[0]!.dest;
  }

  private async fetchOne(
    modelId: string,
    plan: Plan,
    file: { uri: string; dest: string },
    index: number,
    signal?: AbortSignal,
  ): Promise<void> {
    const fetchImpl = this.options.fetch ?? globalThis.fetch;
    const url = urlFor(plan.location, file.uri);
    const partial = `${file.dest}.part`;
    // RESUME: a 20 GB fetch dying at 90% must not start over. Whatever is already in the `.part` file is
    // what we ask the server to continue from.
    const have = existsSync(partial) ? statSync(partial).size : 0;

    const headers: Record<string, string> = {};
    const token = await this.tokenFor(plan.location.source);
    if (token !== undefined) headers["authorization"] = `Bearer ${token}`;
    if (have > 0) headers["range"] = `bytes=${have}-`;

    const res = await fetchImpl(url, { headers, ...(signal !== undefined ? { signal } : {}) });
    if (res.status === 401 || res.status === 403) {
      void res.body?.cancel();
      throw new WeightsCredentialRequired(plan.location.source, file.uri);
    }
    if (!res.ok) {
      void res.body?.cancel();
      throw new WeightsUnavailable(modelId, `fetching ${url} returned HTTP ${res.status}`);
    }
    // A server that ignores `Range` answers 200 with the WHOLE body. Appending that to what we already
    // have would produce a file that is the right size for nothing and silently corrupt.
    const resuming = have > 0 && res.status === 206;
    const startAt = resuming ? have : 0;
    if (have > 0 && !resuming) log.debug("server ignored the range request; restarting the file", { url });

    const declared = Number(res.headers.get("content-length") ?? "");
    const total = Number.isFinite(declared) ? declared + startAt : undefined;

    const handle = openSync(partial, resuming ? "a" : "w");
    let received = startAt;
    try {
      const body = res.body;
      if (body === null) throw new WeightsUnavailable(modelId, `fetching ${url} returned no body`);
      for await (const chunk of body as unknown as AsyncIterable<Uint8Array>) {
        writeSync(handle, chunk);
        received += chunk.byteLength;
        this.options.onProgress?.({
          modelId,
          file: baseName(file.dest),
          part: index + 1,
          parts: plan.files.length,
          receivedBytes: received,
          ...(total !== undefined ? { totalBytes: total } : {}),
        });
      }
    } finally {
      closeSync(handle);
    }

    // Verify BEFORE the rename: the un-suffixed name is what `present()` reads, so a file that fails its
    // checksum must never be allowed to occupy it — that would cache the corruption permanently.
    const expected = index === 0 ? plan.location.sha256 : undefined;
    if (expected !== undefined) {
      const actual = await sha256File(partial);
      if (actual !== expected.toLowerCase()) {
        rmSync(partial, { force: true });
        throw new WeightsUnavailable(modelId, `checksum mismatch for ${file.uri} (expected ${expected}, got ${actual})`);
      }
    }
    renameSync(partial, file.dest);
    log.debug("weights downloaded", { modelId, file: baseName(file.dest), bytes: received });
  }

  private async tokenFor(scope: string): Promise<string | undefined> {
    const token = this.options.token;
    if (token === undefined) return undefined;
    return typeof token === "function" ? token(scope) : token;
  }
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (c) => hash.update(c));
    stream.on("end", () => resolve());
    stream.on("error", reject);
  });
  return hash.digest("hex");
}

/** Join without `node:path`, so the sync accessor above is the only node binding this module needs. */
function joinPath(dir: string, name: string): string {
  const trimmed = dir.replace(/[\\/]+$/, "");
  return `${trimmed}/${name}`;
}

function baseName(p: string): string {
  const cut = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return cut >= 0 ? p.slice(cut + 1) : p;
}
