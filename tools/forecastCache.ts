/**
 * Forecast cache on disk for the benchmark runner. One file per model, dataset and forecast
 * settings: a JSON header line, then the float32 quantiles. The benchmark checks a loaded
 * set against one fresh forecast before using it, so a changed model or port is noticed.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { BenchConfig, ForecastCache, StoredForecasts } from "../src/bench/benchmark";

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

export function diskForecastCache(dir: string, modelId: string): ForecastCache {
  const memory = new Map<string, StoredForecasts>();
  const path = (datasetId: string, series: number[], cfg: BenchConfig): string => {
    const h = createHash("sha256");
    h.update(JSON.stringify({ modelId, horizon: cfg.horizon, maxContext: cfg.maxContext, symmetric: cfg.symmetric, clip: cfg.clip }));
    h.update(Buffer.from(Float64Array.from(series).buffer));
    return join(dir, slug(modelId), `${slug(datasetId)}-${h.digest("hex").slice(0, 16)}.bin`);
  };
  return {
    load(datasetId, series, cfg) {
      const p = path(datasetId, series, cfg);
      const hit = memory.get(p);
      if (hit) return hit;
      if (!existsSync(p)) return undefined;
      try {
        const buf = readFileSync(p);
        const nl = buf.indexOf(10);
        const head = JSON.parse(buf.subarray(0, nl).toString("utf8")) as { horizon: number; levels: number[]; count: number };
        const body = buf.subarray(nl + 1);
        if (body.byteLength !== head.count * 4) return undefined;
        const quantiles = new Float32Array(head.count);
        Buffer.from(quantiles.buffer).set(body);
        const stored = { horizon: head.horizon, levels: head.levels, quantiles };
        memory.set(p, stored);
        return stored;
      } catch {
        return undefined;
      }
    },
    save(datasetId, series, cfg, f) {
      const p = path(datasetId, series, cfg);
      memory.set(p, f);
      mkdirSync(join(dir, slug(modelId)), { recursive: true });
      const head = JSON.stringify({ model: modelId, dataset: datasetId, points: series.length, horizon: f.horizon, levels: f.levels, count: f.quantiles.length });
      writeFileSync(p, Buffer.concat([Buffer.from(head + "\n", "utf8"), Buffer.from(f.quantiles.buffer, f.quantiles.byteOffset, f.quantiles.byteLength)]));
    },
  };
}
