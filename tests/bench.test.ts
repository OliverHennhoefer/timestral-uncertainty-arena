import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { METHODS } from "../src/conformal/methods";
import { benchDatasets, defaultBenchConfig, runBenchmark } from "../src/bench/benchmark";
import { RESULT_SCHEMA, type BenchResult } from "../src/bench/types";
import { aggregate, findings, toCsv } from "../src/bench/summary";
import { DATASETS } from "../src/datasets";
import { NaiveBackend } from "../src/model/backends";
import { PRESETS } from "../src/synthetic";

const naive = new NaiveBackend();
const forecast = (context: Float64Array, horizon: number) => naive.forecast({ context, horizon, symmetric: true, clip: true, maxContext: 256 });

describe("benchmark", () => {
  it("covers every registered dataset", () => {
    expect(benchDatasets().map((d) => d.id)).toEqual([...PRESETS, ...DATASETS].map((p) => p.id));
  });

  it("scores native plus every registered method on each dataset", async () => {
    const res = await runBenchmark({ run: "t", model: { backend: "naive", id: "naive", label: "naive" }, forecast, only: ["nile", "sine"], config: { length: 200, bootstrap: 50 } });
    expect(res.schema).toBe(RESULT_SCHEMA);
    expect(res.datasets.map((d) => d.id)).toEqual(["sine", "nile"]);
    for (const d of res.datasets) {
      expect(d.rows.map((r) => r.id).sort()).toEqual(["native", ...METHODS.map((m) => m.id)].sort());
      expect(d.rows[0].rank).toBe(1);
      for (const r of d.rows) {
        expect(r.n).toBe(d.pairs);
        expect(r.coverage).toBeGreaterThanOrEqual(0);
        expect(r.coverage).toBeLessThanOrEqual(1);
        expect(r.leadCoverage).toHaveLength(res.config.evalLeads);
      }
      expect(d.rows.find((r) => r.id === "native")!.scoreRatio).toBe(1);
    }
    // The CSV has a header plus one line per dataset x row.
    expect(toCsv(res).trim().split("\n")).toHaveLength(1 + 2 * (METHODS.length + 1));
    const agg = aggregate(res.datasets, ["native", "split"]);
    expect(agg[0].datasets).toBe(2);
    const f = findings(res);
    expect(f.findings.length).toBeGreaterThan(2);
    expect(f.caveats.join(" ")).toContain("naive baseline");
  });

  it("summarises a run with no evaluated pairs without throwing", async () => {
    const res = await runBenchmark({ run: "t", model: { backend: "naive", id: "naive", label: "naive" }, forecast, only: ["sine", "nile"], config: { length: 10, bootstrap: 10 } });
    expect(res.datasets.every((d) => d.pairs === 0)).toBe(true);
    expect(findings(res).findings[0]).toContain("No evaluated pairs");
    expect(toCsv(res).trim().split("\n").length).toBeGreaterThan(1);
  });

  it("committed results are readable and cover the current registries", () => {
    const dir = new URL("../benchmarks/results/", import.meta.url);
    const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
    expect(files.length).toBeGreaterThan(0);
    for (const f of files) {
      const r = JSON.parse(readFileSync(new URL(f, dir), "utf8")) as BenchResult;
      expect(r.schema).toBe(RESULT_SCHEMA);
      expect(r.run).toBe(f.replace(/\.json$/, ""));
      expect(r.config.alpha).toBe(defaultBenchConfig().alpha);
      const known = new Set(benchDatasets().map((d) => d.id));
      for (const d of r.datasets) expect(known.has(d.id), `${f}: unknown dataset ${d.id}`).toBe(true);
      // The naive run takes about a minute, so it must always be complete: a dataset or
      // method added to a registry comes with \`npm run benchmark\`. Model runs may be partial.
      if (r.model.id === "naive") {
        expect(r.datasets.map((d) => d.id), `${f}: re-run npm run benchmark`).toEqual(benchDatasets().map((d) => d.id));
        expect(r.methods.map((m) => m.id), `${f}: re-run npm run benchmark`).toEqual(["native", ...METHODS.map((m) => m.id)]);
      }
    }
  });
});
