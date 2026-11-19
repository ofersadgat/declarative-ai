/**
 * `@declarative-ai/script` — a workflow script as a plain program (SCRIPTS.md §17).
 *
 * A script imports its hooks from `@declarative-ai/hw/script`. Under the engine they are cut into
 * states and run durably; here they run directly, through {@link createStandaloneHost}. Importing this
 * package installs that host when none is installed, so a script that imports it first can be run with
 * nothing but `node`; `dai-script <file> [args…]` does the same for a script that does not. Either way
 * the run is reported to an engine that takes reports when one is listening (`report.ts`).
 */
import { installStandaloneHost, standaloneHost } from "@declarative-ai/hw/script";
import { createDiscoveringHost } from "./run.js";

export * from "./standalone.js";
export * from "./run.js";
export * from "./report.js";
export * from "@declarative-ai/hw/script";

// Reported to an engine that takes reports, when one is listening at the run's first call (§17.4).
if (standaloneHost() === undefined) installStandaloneHost(createDiscoveringHost());
