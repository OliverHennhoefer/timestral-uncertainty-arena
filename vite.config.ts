import { defineConfig } from "vitest/config";

// `base: "./"` makes the build work from any sub-path, e.g. https://<user>.github.io/uncertainty-aware-timestral/.
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  build: {
    target: "es2022",
    // Two pages: the arena and the static leaderboard.
    rollupOptions: { input: { main: "index.html", leaderboard: "leaderboard.html" } },
  },
  test: { environment: "node" },
});
