#!/usr/bin/env node
// Loads tools/benchmark.ts through Vite's module runner, so the app's TypeScript modules and
// bundled JSON datasets work in Node without a separate build step.
import { fileURLToPath } from "node:url";
import { runnerImport } from "vite";

const entry = fileURLToPath(new URL("./benchmark.ts", import.meta.url));
const { module } = await runnerImport(entry);
try {
  await module.main(process.argv.slice(2));
} catch (e) {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
}
