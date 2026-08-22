/**
 * Hashing, approval, the freeze, and the snapshot fold (SPEC §7.5.5).
 *
 * The threat is code changing without the user knowing — which, where an agent can write files,
 * includes code the user never wrote. So:
 *
 *  - an UNKNOWN file is an unapproved file, which is what covers a module dropped in to shadow one;
 *  - `node_modules` is exempt by RESOLVED PATH, never by how a specifier was spelled;
 *  - the freeze stops the run before anything executes, and names every offending file at once;
 *  - what it hands back is the TRANSPILED output, because a hash can only detect drift and refuse —
 *    it cannot execute the version that was approved;
 *  - those hashes reach the snapshot, or a pinned task runs edited code under an unchanged version.
 */
import { describe, expect, it } from "vitest";
import {
  approvalsOf,
  freezeModules,
  IntegrityError,
  isVendored,
  moduleDigest,
  moduleHash,
  pendingApprovals,
} from "../src/integrity.js";
import { requirePathFor } from "../src/moduleLoader.js";
import { loadBundle, snapshotHash } from "../src/loader.js";
import type { Vfs } from "../src/reference.js";

const FN = "/p/functions";

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (!rest.includes("/")) names.add(rest);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

const options = (files: Record<string, string>, approved: Record<string, string> = {}) => ({
  vfs: vfsOf(files),
  requirePath: requirePathFor([FN]),
  approvals: approvalsOf(approved),
});

/** Approve every file in `files` as it currently stands. */
const approveAll = (files: Record<string, string>): Record<string, string> =>
  Object.fromEntries(Object.entries(files).map(([file, source]) => [file, moduleHash(source)]));

describe("what counts as vendored", () => {
  it("is a property of the resolved path, not of the specifier", () => {
    // The require path lets a BARE specifier reach a plain file in a search entry, so "bare imports
    // are exempt" would leave open exactly the hole approval closes.
    expect(isVendored("/p/functions/node_modules/zod/index.js")).toBe(true);
    expect(isVendored("/p/node_modules/a/b/c.js")).toBe(true);
    expect(isVendored("/p/functions/helper.ts")).toBe(false);
    expect(isVendored("/p/my_node_modules_helper/x.ts")).toBe(false);
  });
});

describe("discovering what needs approval", () => {
  const files = {
    [`${FN}/main.ts`]: "import './helper.js';\nexport default () => 1;",
    [`${FN}/helper.ts`]: "export const x = 1;",
    [`${FN}/node_modules/zod/index.js`]: "module.exports = {};",
  };

  it("lists the whole closure on a first run", async () => {
    const pending = await pendingApprovals([`${FN}/main.ts`], options(files));
    expect(pending.map((p) => p.file).sort()).toEqual([`${FN}/helper.ts`, `${FN}/main.ts`]);
  });

  it("leaves node_modules out of it", async () => {
    const withDep = { ...files, [`${FN}/main.ts`]: "import 'zod';\nexport default () => 1;" };
    const pending = await pendingApprovals([`${FN}/main.ts`], options(withDep));
    expect(pending.map((p) => p.file)).not.toContain(`${FN}/node_modules/zod/index.js`);
  });

  it("says nothing when everything is already approved at its current hash", async () => {
    expect(await pendingApprovals([`${FN}/main.ts`], options(files, approveAll(files)))).toEqual([]);
  });

  it("distinguishes a FIRST approval from a re-approval", async () => {
    // The two are different questions, and only the second can be answered by looking at a diff.
    const approved = approveAll(files);
    const edited = { ...files, [`${FN}/helper.ts`]: "export const x = 2;" };
    const pending = await pendingApprovals([`${FN}/main.ts`], options(edited, approved));
    expect(pending).toHaveLength(1);
    expect(pending[0]).toMatchObject({ file: `${FN}/helper.ts`, previousHash: approved[`${FN}/helper.ts`] });
    // The current source travels with it, because a hash alone is nothing a person can act on.
    expect(pending[0]?.source).toBe("export const x = 2;");
  });

  it("reports a brand-new file with no previous hash at all", async () => {
    const approved = approveAll(files);
    const extended = {
      ...files,
      [`${FN}/main.ts`]: "import './helper.js';\nimport './dropped.js';\nexport default () => 1;",
      [`${FN}/dropped.ts`]: "export const y = 1;",
    };
    const pending = await pendingApprovals([`${FN}/main.ts`], options(extended, approved));
    const dropped = pending.find((p) => p.file.endsWith("dropped.ts"));
    expect(dropped).toBeDefined();
    expect(dropped).not.toHaveProperty("previousHash");
  });
});

describe("the freeze", () => {
  const files = {
    [`${FN}/main.ts`]: "import { x } from './helper.js';\nexport default () => x;",
    [`${FN}/helper.ts`]: "export const x = 7;",
  };

  it("hands back the transpiled output, which is what makes a frozen run frozen", async () => {
    const frozen = await freezeModules([`${FN}/main.ts`], options(files, approveAll(files)));
    expect([...frozen.emitted.keys()].sort()).toEqual([`${FN}/helper.ts`, `${FN}/main.ts`]);
    // Emitted CommonJS, not the source: replay never invokes the compiler, so a later toolchain
    // upgrade cannot change what a pinned run does.
    expect(frozen.emitted.get(`${FN}/helper.ts`)).toContain("exports");
    expect(frozen.hashes.get(`${FN}/helper.ts`)).toBe(moduleHash(files[`${FN}/helper.ts`]!));
  });

  it("refuses a file that was never approved", async () => {
    await expect(freezeModules([`${FN}/main.ts`], options(files))).rejects.toThrow(IntegrityError);
    await expect(freezeModules([`${FN}/main.ts`], options(files))).rejects.toThrow(/never approved/);
  });

  it("refuses a file that changed since it was approved", async () => {
    const approved = approveAll(files);
    const edited = { ...files, [`${FN}/helper.ts`]: "export const x = 9;" };
    await expect(freezeModules([`${FN}/main.ts`], options(edited, approved))).rejects.toThrow(
      /changed since approval.*helper\.ts/s,
    );
  });

  it("names EVERY offending file, not the first", async () => {
    // Somebody about to be asked for approvals wants the list, not a sequence of one-at-a-time
    // refusals.
    let message = "";
    try {
      await freezeModules([`${FN}/main.ts`], options(files));
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).toContain("main.ts");
    expect(message).toContain("helper.ts");
  });

  it("stops before anything executes", async () => {
    const effectful = {
      [`${FN}/main.ts`]: "globalThis.__frozeRan = true;\nexport default () => 1;",
    };
    (globalThis as Record<string, unknown>).__frozeRan = false;
    await expect(freezeModules([`${FN}/main.ts`], options(effectful))).rejects.toThrow(IntegrityError);
    expect((globalThis as Record<string, unknown>).__frozeRan).toBe(false);
  });

  it("does not demand approval for node_modules", async () => {
    const withDep = {
      [`${FN}/main.ts`]: "import 'zod';\nexport default () => 1;",
      [`${FN}/node_modules/zod/index.js`]: "module.exports = {};",
    };
    const approved = { [`${FN}/main.ts`]: moduleHash(withDep[`${FN}/main.ts`]!) };
    const frozen = await freezeModules([`${FN}/main.ts`], options(withDep, approved));
    expect(frozen.hashes.has(`${FN}/node_modules/zod/index.js`)).toBe(false);
    // It is still prepared and runnable — exempt from approval is not exempt from loading.
    expect(frozen.emitted.has(`${FN}/node_modules/zod/index.js`)).toBe(true);
  });
});

describe("the digest", () => {
  it("changes when a module's content changes", () => {
    const a = moduleDigest(new Map([[`${FN}/x.ts`, moduleHash("export const a = 1;")]]));
    const b = moduleDigest(new Map([[`${FN}/x.ts`, moduleHash("export const a = 2;")]]));
    expect(a).not.toBe(b);
  });

  it("does not change with the DIRECTORY a file sits in", () => {
    // Absolute paths are machine-specific (`/home/…` against `C:/Users/…`), and a snapshot that
    // differs between two machines running the same workflow is not an identity.
    const hash = moduleHash("export const a = 1;");
    expect(moduleDigest(new Map([["/home/ofer/fn/x.ts", hash]]))).toBe(
      moduleDigest(new Map([["C:/Users/Ofer/fn/x.ts", hash]])),
    );
  });

  it("DOES change when a file is renamed", () => {
    // Within one directory the winner of a symbol collision is decided by filename order, so a rename
    // can change which code runs while every content hash stays the same.
    const hash = moduleHash("export const a = 1;");
    expect(moduleDigest(new Map([[`${FN}/a_first.ts`, hash]]))).not.toBe(
      moduleDigest(new Map([[`${FN}/z_last.ts`, hash]])),
    );
  });

  it("is order-independent across files", () => {
    const one = moduleHash("1");
    const two = moduleHash("2");
    expect(moduleDigest(new Map([[`${FN}/a.ts`, one], [`${FN}/b.ts`, two]]))).toBe(
      moduleDigest(new Map([[`${FN}/b.ts`, two], [`${FN}/a.ts`, one]])),
    );
  });
});

describe("the snapshot fold", () => {
  const bundle = () => loadBundle({ root: { outputs: { done: { binding: { json: true } } } } }, "root");

  it("leaves a workflow that reaches no module hashing exactly as it always did", () => {
    // The key is omitted from the hashed document entirely when absent, so every snapshot taken
    // before modules existed keeps its identity.
    const before = snapshotHash(bundle());
    const withUndefined = { ...bundle(), moduleDigest: undefined };
    expect(snapshotHash(withUndefined)).toBe(before);
  });

  it("changes a workflow's identity when a module it reaches changes", () => {
    const plain = snapshotHash(bundle());
    const withModules = { ...bundle(), moduleDigest: moduleDigest(new Map([[`${FN}/x.ts`, moduleHash("1")]])) };
    const edited = { ...bundle(), moduleDigest: moduleDigest(new Map([[`${FN}/x.ts`, moduleHash("2")]])) };

    expect(snapshotHash(withModules)).not.toBe(plain);
    expect(snapshotHash(edited)).not.toBe(snapshotHash(withModules));
  });
});
