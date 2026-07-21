import type { Tool } from "@declarative-ai/core";
import { fsTools } from "./fsTools";
import { searchTools } from "./searchTools";
import { shellTools } from "./shellTools";

export * from "./workspace";
export * from "./fsTools";
export * from "./searchTools";
export * from "./shellTools";

/** Every workspace tool, keyed by logical name — register with `for (const [n, t] of Object.entries(allTools))`. */
export const allTools: Record<string, Tool> = { ...fsTools, ...searchTools, ...shellTools };
