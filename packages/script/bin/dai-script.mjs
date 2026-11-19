#!/usr/bin/env node
/**
 * dai-script — run a workflow script as a plain program (SCRIPTS.md §17).
 *
 *   dai-script <file> [arg …] [--model <route/model>] [--cwd <dir>] [--no-report]
 *
 * Each positional argument after the file is parsed as JSON when it is JSON, and passed as a string
 * otherwise; they are the default export's parameters, in order. What the function returns is printed
 * as JSON on stdout; phases, logs and the input prompt go to stderr, so the result can be piped.
 *
 * When an engine that takes reports is listening (SCRIPTS.md §17.3) — JaiRA, open on this machine — the
 * run is reported to it; `--no-report` (or `DAI_SCRIPT_REPORT=0`) runs it unreported.
 */
const argv = process.argv.slice(2);
const options = {};
const positional = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "--model") options.model = argv[++i];
  else if (a === "--cwd") options.cwd = argv[++i];
  else if (a === "--no-report") options.reporting = false;
  else if (a === "-h" || a === "--help") {
    process.stdout.write("usage: dai-script <file> [arg …] [--model <route/model>] [--cwd <dir>] [--no-report]\n");
    process.exit(0);
  } else positional.push(a);
}
const [file, ...rest] = positional;
if (file === undefined) {
  process.stderr.write("dai-script: name the script to run\n");
  process.exit(2);
}
const args = rest.map((a) => {
  try {
    return JSON.parse(a);
  } catch {
    return a;
  }
});
// TypeScript runs through tsx where Node cannot strip types itself.
// Loaded as ES modules whatever the folder's package.json says, since a workflow script is one.
if (/\.(ts|mts|cts)$/.test(file) && !process.features?.typescript) {
  try {
    const { tsImport } = await import("tsx/esm/api");
    options.importer = (url) => tsImport(url, import.meta.url);
  } catch {
    process.stderr.write("dai-script: running TypeScript needs Node with --experimental-strip-types, or tsx installed\n");
    process.exit(2);
  }
}
const { runScript } = await import("../dist/index.js");
try {
  const result = await runScript(file, args, options);
  if (result !== undefined) process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
} catch (e) {
  process.stderr.write(`dai-script: ${e?.stack ?? e}\n`);
  process.exit(1);
}
