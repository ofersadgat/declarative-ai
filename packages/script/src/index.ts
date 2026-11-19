/**
 * `@declarative-ai/script` — a workflow script as a plain program (SCRIPTS.md §17).
 *
 * A script imports its hooks from `@declarative-ai/hw/script`. Under the engine they are cut into
 * states and run durably; here they run directly, through {@link createStandaloneHost}. Importing this
 * package installs that host when none is installed, so a script that imports it first can be run with
 * nothing but `node`; `dai-script <file> [args…]` does the same for a script that does not.
 */
import { installStandaloneHost, standaloneHost } from "@declarative-ai/hw/script";
import { createStandaloneHost } from "./standalone.js";

export * from "./standalone.js";
export * from "./run.js";
export * from "@declarative-ai/hw/script";

if (standaloneHost() === undefined) installStandaloneHost(createStandaloneHost());
