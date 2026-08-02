import type { Tool } from "@declarative-ai/exec";
import { fsTools } from "./fsTools.js";
import { searchTools } from "./searchTools.js";
import { shellTools } from "./shellTools.js";

export * from "./workspace.js";
export * from "./fsTools.js";
export * from "./searchTools.js";
export * from "./shellTools.js";

/** Every workspace tool, keyed by logical name — register with `for (const [n, t] of Object.entries(allTools))`. */
export const allTools: Record<string, Tool> = { ...fsTools, ...searchTools, ...shellTools };
