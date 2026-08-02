import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Published `exports` point at `dist`, so tests would otherwise run against build output.
// Alias every workspace package back to its source — one entry per package, plus one for
// each deep export (`@declarative-ai/llm/model-catalog`).
const packagesDir = fileURLToPath(new URL("./packages", import.meta.url));
const alias = readdirSync(packagesDir).flatMap((pkg) => [
  { find: new RegExp(`^@declarative-ai/${pkg}$`), replacement: `${packagesDir}/${pkg}/src/index.ts` },
  { find: new RegExp(`^@declarative-ai/${pkg}/(.*)$`), replacement: `${packagesDir}/${pkg}/src/$1.ts` },
]);

export default defineConfig({
  resolve: { alias },
  test: {
    include: ["packages/*/test/**/*.test.ts"],
    environment: "node",
  },
});
