/**
 * Headless benchmark: runs every registered dataset through every registered method with the
 * same engine the app uses, and returns one JSON-serialisable result per model.
 *
 * Protocol, per dataset: forecast at every origin (each forecast sees only `y[0..o]`), store
 * the forecasts, then replay the engine once. That is exactly the app's "recompute" path, so
 * a leaderboard row equals what the dashboard would show for that series and settings.
 *
 * Datasets come from the synthetic presets and the bundled real data, methods from
 * `METHODS`; adding either to its registry adds it here with no other change.
 */

import { METHODS, defaultParams } from "../conformal/methods";
import { DATASETS } from "../datasets";
import { indistinguishable } from "../engine/bootstrap";
import { Engine, NATIVE_ID, type Settings, defaultSettings, rankRows } from "../engine/engine";
import type { Preset } from "../synthetic";
import { PRESETS } from "../synthetic";
import { type BenchConfig, type BenchResult, type DatasetResult, type ModelInfo, RESULT_SCHEMA, type ResultRow } from "./types";

export * from "./types";

export function defaultBenchConfig(): BenchConfig {
  const s = defaultSettings();
  return {
    alpha: s.alpha,
    evalLeads: s.evalLeads,
    scoreType: s.scoreType,
    window: s.window,
    warmup: s.warmup,
    horizon: s.evalLeads,
    maxContext: 256,
    symmetric: true,
    clip: true,
    length: 2000,
    seed: 1,
    bootstrap: 300,
    tolerance: 0.02,
  };
}

export interface BenchDataset {
  id: string;
  label: string;
  group: string;
  source?: string;
  make(length: number, seed: number): number[];
}

/** Every dataset the app offers: synthetic presets first, then the bundled real data. */
export function benchDatasets(): BenchDataset[] {
  const toBench = (p: Preset): BenchDataset => ({ id: p.id, label: p.label, group: p.group ?? "Synthetic", source: p.source, make: p.make });
  return [...PRESETS, ...DATASETS].map(toBench);
}

/** A forecaster: `(H, Q)` quantiles for the context, on the given levels. */
export type Forecast = (context: Float64Array, horizon: number) => Promise<{ quantiles: Float32Array; levels: number[]; horizon: number }>;

const round = (v: number, d = 5) => (Number.isFinite(v) ? Math.round(v * 10 ** d) / 10 ** d : v);
const r2 = (a: [number, number]): [number, number] => [round(a[0]), round(a[1])];

export function engineSettings(cfg: BenchConfig): Settings {
  const d = defaultSettings();
  return {
    ...d,
    alpha: cfg.alpha,
    evalLeads: cfg.evalLeads,
    scoreType: cfg.scoreType,
    window: cfg.window,
    warmup: cfg.warmup,
    // Every registered method, including the ones that start switched off in the app.
    enabled: Object.fromEntries(METHODS.map((m) => [m.id, true])),
    params: Object.fromEntries(METHODS.map((m) => [m.id, defaultParams(m)])),
  };
}

/** Run one series through the engine and score every row. */
export async function runSeries(series: number[], forecast: Forecast, cfg: BenchConfig, onProgress?: (done: number, total: number) => void): Promise<{ pairs: number; rows: ResultRow[] }> {
  const engine = new Engine(engineSettings(cfg));
  engine.setSeries(series);
  const n = series.length;
  for (let o = 0; o < n; o++) {
    const res = await forecast(Float64Array.from(series.slice(0, o + 1)), cfg.horizon);
    engine.storeForecast(o, res.quantiles, res.horizon, res.levels);
    onProgress?.(o + 1, n);
  }
  engine.replay();
  const unc = engine.uncertainty(undefined, cfg.bootstrap);
  const ranked = rankRows(engine.table(), cfg.alpha, "interval_score", cfg.tolerance, unc);
  const native = ranked.find((r) => r.id === NATIVE_ID)!;
  const nativeSamples = unc.get(NATIVE_ID)?.scoreSamples;
  const rows = ranked.map((r): ResultRow => {
    const u = unc.get(r.id);
    let vsNative: -1 | 0 | 1 = 0;
    if (r.id !== NATIVE_ID && u && nativeSamples && !indistinguishable(u.scoreSamples, nativeSamples)) {
      vsNative = r.intervalScore < native.intervalScore ? -1 : 1;
    }
    return {
      id: r.id,
      n: r.n,
      coverage: round(r.coverage),
      gap: round(r.gap),
      valid: r.valid,
      meanWidth: round(r.meanWidth),
      intervalScore: round(r.intervalScore),
      rank: r.rank,
      tied: !!r.tied,
      scoreRatio: round(r.intervalScore / native.intervalScore),
      vsNative,
      ci: u ? { coverage: r2(u.coverage), intervalScore: r2(u.intervalScore) } : undefined,
      leadCoverage: engine.leadProfile(r.id).coverage.map((v) => round(v, 4)),
    };
  });
  return { pairs: native.n, rows };
}

export interface RunOptions {
  run: string;
  model: ModelInfo;
  forecast: Forecast;
  config?: Partial<BenchConfig>;
  /** Only these dataset ids (all when empty). */
  only?: string[];
  commit?: string;
  log?: (msg: string) => void;
}

export async function runBenchmark(opts: RunOptions): Promise<BenchResult> {
  const cfg = { ...defaultBenchConfig(), ...opts.config };
  if (cfg.horizon < cfg.evalLeads) cfg.horizon = cfg.evalLeads;
  const log = opts.log ?? (() => {});
  const datasets = benchDatasets().filter((d) => !opts.only?.length || opts.only.includes(d.id));
  const out: DatasetResult[] = [];
  for (const d of datasets) {
    const series = d.make(cfg.length, cfg.seed);
    const t0 = performance.now();
    let lastLog = 0;
    const { pairs, rows } = await runSeries(series, opts.forecast, cfg, (done, total) => {
      const now = performance.now();
      if (now - lastLog > 5000) {
        lastLog = now;
        log(`  ${d.id}: ${done}/${total} origins`);
      }
    });
    const seconds = round((performance.now() - t0) / 1000, 1);
    const best = rows[0];
    log(`${d.id}: ${series.length} points, ${pairs} pairs, ${seconds} s; best ${best.id} (IS ${best.intervalScore.toFixed(2)})`);
    out.push({ id: d.id, label: d.label, group: d.group, source: d.source, points: series.length, pairs, seconds, rows });
  }
  return {
    schema: RESULT_SCHEMA,
    run: opts.run,
    createdAt: new Date().toISOString(),
    commit: opts.commit,
    model: opts.model,
    config: cfg,
    methods: [
      { id: NATIVE_ID, name: "Native interval", family: "model", guarantee: false, reference: "The model's own quantiles at α/2 and 1 − α/2", color: "#8E8E93", params: {} },
      ...METHODS.map((m) => ({
        id: m.id,
        name: m.name,
        family: m.family ?? "conformal",
        guarantee: m.guarantee !== false,
        reference: m.reference,
        color: m.color,
        params: defaultParams(m),
      })),
    ],
    datasets: out,
  };
}
