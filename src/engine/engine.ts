/**
 * The benchmark engine: a deterministic function of (series, stored forecasts, settings).
 *
 * Protocol (strictly online). A forecast made at origin `o` sees `y[0..o]` and predicts
 * `y[o+1..o+H]`. When `y[t]` arrives, every stored forecast with `t - o = k <= H` yields a
 * score at lead `k`, which updates that lead's conformal methods and, if the pair was
 * issued after warm-up, the coverage metrics of every method and of the model's native
 * interval. Forecasts are added only after all arrivals up to their origin, so thresholds
 * never see the future. Replaying from scratch gives exactly the same state, which is how
 * settings changes and redrawn history are handled.
 */

import { METHODS, type MethodSpec, type OnlineMethod, defaultParams } from "../conformal/methods";
import { ScoreHistory } from "../conformal/window";

export type ScoreType = "cqr" | "abs" | "scaled";

export interface Settings {
  alpha: number;
  /** Max lead used for scoring and metrics (<= the forecast horizon). */
  evalLeads: number;
  scoreType: ScoreType;
  window: number;
  warmup: number;
  rolling: number;
  enabled: Record<string, boolean>;
  params: Record<string, Record<string, number>>;
}

export const NATIVE_ID = "native";
export const NATIVE_COLOR = "#6b7280";

export function defaultSettings(): Settings {
  return {
    alpha: 0.1,
    evalLeads: 16,
    scoreType: "cqr",
    window: 100,
    warmup: 10,
    rolling: 200,
    enabled: Object.fromEntries(METHODS.map((m) => [m.id, true])),
    params: Object.fromEntries(METHODS.map((m) => [m.id, defaultParams(m)])),
  };
}

export interface StoredForecast {
  origin: number;
  horizon: number;
  levels: number[];
  /** `(H, Q)` row-major. */
  quantiles: Float32Array;
  // Derived at the current alpha:
  median: Float64Array;
  lo: Float64Array;
  hi: Float64Array;
  /** Threshold issued per method id, per lead (index k - 1); NaN where not issued. */
  thresholds: Map<string, Float64Array>;
  /** Whether lead k - 1 was past warm-up when issued (counts toward metrics). */
  evaluated: Uint8Array;
}

export interface MethodMetrics {
  id: string;
  n: number;
  covered: number;
  widthSum: number;
  intervalScoreSum: number;
  ringCovered: Uint8Array;
  ringWidth: Float64Array;
  ringPos: number;
  ringFill: number;
  /** Rolling coverage / width after each arrival with evaluations. */
  covTrace: number[];
  widthTrace: number[];
}

export interface MetricsRow {
  id: string;
  n: number;
  coverage: number;
  rollingCoverage: number;
  meanWidth: number;
  rollingWidth: number;
  intervalScore: number;
}

/** Linear interpolation of a quantile vector at probability `p`. */
export function quantileAt(q: Float32Array, offset: number, levels: number[], p: number): number {
  const Q = levels.length;
  if (p <= levels[0]) return q[offset];
  if (p >= levels[Q - 1]) return q[offset + Q - 1];
  let i = 1;
  while (levels[i] < p) i++;
  const w = (p - levels[i - 1]) / (levels[i] - levels[i - 1]);
  return q[offset + i - 1] * (1 - w) + q[offset + i] * w;
}

export class Engine {
  series: number[] = [];
  readonly forecasts = new Map<number, StoredForecast>();
  settings: Settings;
  private histories: ScoreHistory[] = [];
  private methods: Map<string, OnlineMethod>[] = [];
  metrics = new Map<string, MethodMetrics>();
  traceT: number[] = [];
  processedUpTo = -1;
  /** Increments whenever history is rewritten, so in-flight forecasts can be dropped. */
  revision = 0;

  constructor(settings: Settings = defaultSettings()) {
    this.settings = settings;
    this.reset();
  }

  get activeSpecs(): MethodSpec[] {
    return METHODS.filter((m) => this.settings.enabled[m.id]);
  }

  get rowIds(): string[] {
    return [NATIVE_ID, ...this.activeSpecs.map((m) => m.id)];
  }

  private reset(): void {
    this.histories = [];
    this.methods = [];
    this.metrics = new Map();
    this.traceT = [];
    this.processedUpTo = -1;
    for (const id of this.rowIds) this.metrics.set(id, this.newMetrics(id));
  }

  private newMetrics(id: string): MethodMetrics {
    const R = this.settings.rolling;
    return {
      id, n: 0, covered: 0, widthSum: 0, intervalScoreSum: 0,
      ringCovered: new Uint8Array(R), ringWidth: new Float64Array(R), ringPos: 0, ringFill: 0,
      covTrace: [], widthTrace: [],
    };
  }

  private lead(k: number): { history: ScoreHistory; methods: Map<string, OnlineMethod> } {
    while (this.histories.length < k) {
      const h = new ScoreHistory(this.settings.window);
      this.histories.push(h);
      const m = new Map<string, OnlineMethod>();
      for (const spec of this.activeSpecs) m.set(spec.id, spec.create(h, this.settings.alpha, this.settings.params[spec.id] ?? defaultParams(spec)));
      this.methods.push(m);
    }
    return { history: this.histories[k - 1], methods: this.methods[k - 1] };
  }

  /** Lead-k score history (for diagnostics and tests). */
  scoresAt(k: number): readonly number[] {
    return this.histories[k - 1]?.all ?? [];
  }

  // ---- series edits -------------------------------------------------------------------

  /** Replace the series. Keeps forecasts whose inputs are unchanged and replays if needed. */
  setSeries(values: number[]): void {
    let common = 0;
    const n = Math.min(values.length, this.series.length);
    while (common < n && values[common] === this.series[common]) common++;
    const rewritten = common < this.series.length;
    this.series = values.slice();
    if (rewritten) this.truncateDerived(common);
  }

  append(value: number): void {
    this.series.push(value);
  }

  truncate(length: number): void {
    if (length >= this.series.length) return;
    this.series.length = length;
    this.truncateDerived(length);
  }

  private truncateDerived(validLength: number): void {
    this.revision++;
    for (const o of [...this.forecasts.keys()]) if (o >= validLength) this.forecasts.delete(o);
    if (this.processedUpTo >= validLength) this.replay();
  }

  clear(): void {
    this.series = [];
    this.forecasts.clear();
    this.revision++;
    this.reset();
  }

  // ---- online protocol ----------------------------------------------------------------

  /** Process arrivals up to and including `t` (bounded by the series length). */
  catchUp(t: number = this.series.length - 1): void {
    const end = Math.min(t, this.series.length - 1);
    while (this.processedUpTo < end) {
      this.processedUpTo++;
      this.processArrival(this.processedUpTo);
    }
  }

  /** Add a forecast made at `origin` (requires arrivals up to `origin` to be processed). */
  addForecast(origin: number, quantiles: Float32Array, horizon: number, levels: number[]): StoredForecast {
    this.catchUp(origin);
    const f: StoredForecast = {
      origin, horizon, levels, quantiles,
      median: new Float64Array(horizon), lo: new Float64Array(horizon), hi: new Float64Array(horizon),
      thresholds: new Map(), evaluated: new Uint8Array(horizon),
    };
    this.forecasts.set(origin, f);
    this.issue(f);
    return f;
  }

  /** Store a forecast without issuing thresholds; call `replay()` afterwards (backfill). */
  storeForecast(origin: number, quantiles: Float32Array, horizon: number, levels: number[]): void {
    this.forecasts.set(origin, {
      origin, horizon, levels, quantiles,
      median: new Float64Array(horizon), lo: new Float64Array(horizon), hi: new Float64Array(horizon),
      thresholds: new Map(), evaluated: new Uint8Array(horizon),
    });
  }

  /** Latest forecast origin, or -1. */
  latestOrigin(): number {
    let m = -1;
    for (const o of this.forecasts.keys()) if (o > m) m = o;
    return m;
  }

  /** Recompute everything from the stored forecasts (after settings changes or edits). */
  replay(): void {
    const upTo = this.processedUpTo;
    this.reset();
    const origins = [...this.forecasts.keys()].sort((a, b) => a - b);
    let oi = 0;
    const last = Math.min(Math.max(upTo, origins.length ? origins[origins.length - 1] : -1), this.series.length - 1);
    for (let t = 0; t <= last; t++) {
      this.processedUpTo = t;
      this.processArrival(t);
      while (oi < origins.length && origins[oi] === t) {
        this.issue(this.forecasts.get(t)!);
        oi++;
      }
    }
    while (oi < origins.length) this.forecasts.delete(origins[oi++]);
  }

  updateSettings(patch: Partial<Settings>): void {
    this.settings = { ...this.settings, ...patch };
    this.replay();
  }

  private derive(f: StoredForecast): void {
    const { alpha } = this.settings;
    const Q = f.levels.length;
    for (let k = 0; k < f.horizon; k++) {
      f.median[k] = quantileAt(f.quantiles, k * Q, f.levels, 0.5);
      f.lo[k] = quantileAt(f.quantiles, k * Q, f.levels, alpha / 2);
      f.hi[k] = quantileAt(f.quantiles, k * Q, f.levels, 1 - alpha / 2);
    }
  }

  private issue(f: StoredForecast): void {
    this.derive(f);
    f.thresholds = new Map();
    const H = Math.min(f.horizon, this.settings.evalLeads);
    for (const spec of this.activeSpecs) f.thresholds.set(spec.id, new Float64Array(f.horizon).fill(NaN));
    for (let k = 1; k <= H; k++) {
      const { history, methods } = this.lead(k);
      f.evaluated[k - 1] = history.count >= this.settings.warmup ? 1 : 0;
      const cap = history.maxAbsScore;
      for (const [id, m] of methods) {
        let q = m.predict(f.origin);
        // Thresholds beyond the largest score seen (including +/-inf) are capped there.
        if (history.count === 0) q = NaN;
        else if (Number.isNaN(q)) q = cap;
        else q = Math.max(-cap, Math.min(cap, q));
        f.thresholds.get(id)![k - 1] = q;
      }
    }
  }

  /** Score of truth `y` at lead index `j` of forecast `f`. */
  score(f: StoredForecast, j: number, y: number): number {
    switch (this.settings.scoreType) {
      case "cqr":
        return Math.max(f.lo[j] - y, y - f.hi[j]);
      case "abs":
        return Math.abs(y - f.median[j]);
      case "scaled":
        return Math.abs(y - f.median[j]) / Math.max((f.hi[j] - f.lo[j]) / 2, 1e-9);
    }
  }

  /** Interval of method `id` at lead index `j` (threshold `q`), or the native interval. */
  interval(f: StoredForecast, j: number, id: string): [number, number] | null {
    if (id === NATIVE_ID) return [f.lo[j], f.hi[j]];
    const q = f.thresholds.get(id)?.[j];
    if (q === undefined || Number.isNaN(q)) return null;
    switch (this.settings.scoreType) {
      case "cqr":
        return [f.lo[j] - q, f.hi[j] + q];
      case "abs":
        return [f.median[j] - q, f.median[j] + q];
      case "scaled": {
        const hw = Math.max((f.hi[j] - f.lo[j]) / 2, 1e-9);
        return [f.median[j] - q * hw, f.median[j] + q * hw];
      }
    }
  }

  private processArrival(t: number): void {
    const y = this.series[t];
    const { alpha } = this.settings;
    let evaluatedAny = false;
    const maxK = this.settings.evalLeads;
    for (let k = 1; k <= maxK; k++) {
      const f = this.forecasts.get(t - k);
      if (!f || k > f.horizon) continue;
      const j = k - 1;
      const s = this.score(f, j, y);
      if (!Number.isFinite(s)) continue;
      const { history, methods } = this.lead(k);
      if (f.evaluated[j]) {
        evaluatedAny = true;
        for (const id of this.rowIds) {
          const iv = this.interval(f, j, id);
          const m = this.metrics.get(id);
          if (!iv || !m) continue;
          const [lo, hi] = iv;
          const width = Math.max(0, hi - lo);
          const cov = y >= lo && y <= hi ? 1 : 0;
          m.n++;
          m.covered += cov;
          m.widthSum += width;
          m.intervalScoreSum += width + (2 / alpha) * Math.max(0, lo - y) + (2 / alpha) * Math.max(0, y - hi);
          m.ringCovered[m.ringPos] = cov;
          m.ringWidth[m.ringPos] = width;
          m.ringPos = (m.ringPos + 1) % m.ringCovered.length;
          m.ringFill = Math.min(m.ringFill + 1, m.ringCovered.length);
        }
      }
      for (const [, m] of methods) m.update(f.origin, s);
      history.push(s);
    }
    if (evaluatedAny) {
      this.traceT.push(t);
      for (const id of this.rowIds) {
        const m = this.metrics.get(id)!;
        const r = rolling(m);
        m.covTrace.push(r.coverage);
        m.widthTrace.push(r.width);
      }
    }
  }

  table(): MetricsRow[] {
    return this.rowIds.map((id) => {
      const m = this.metrics.get(id)!;
      const r = rolling(m);
      return {
        id,
        n: m.n,
        coverage: m.n ? m.covered / m.n : NaN,
        rollingCoverage: r.coverage,
        meanWidth: m.n ? m.widthSum / m.n : NaN,
        rollingWidth: r.width,
        intervalScore: m.n ? m.intervalScoreSum / m.n : NaN,
      };
    });
  }
}

function rolling(m: MethodMetrics): { coverage: number; width: number } {
  if (m.ringFill === 0) return { coverage: NaN, width: NaN };
  let c = 0, w = 0;
  for (let i = 0; i < m.ringFill; i++) {
    c += m.ringCovered[i];
    w += m.ringWidth[i];
  }
  return { coverage: c / m.ringFill, width: w / m.ringFill };
}

export type RankBy = "interval_score" | "validity_then_width" | "rolling_gap";

export interface RankedRow extends MetricsRow {
  rank: number;
  gap: number;
  valid: boolean;
}

/** Rank methods (including the native interval) by the chosen criterion. */
export function rankRows(rows: MetricsRow[], alpha: number, by: RankBy, tolerance = 0.02): RankedRow[] {
  const target = 1 - alpha;
  const ext = rows.map((r) => ({
    ...r,
    rank: 0,
    gap: Math.abs(r.coverage - target),
    valid: r.coverage >= target - tolerance,
  }));
  const key = (r: RankedRow): number[] => {
    if (r.n === 0) return [Infinity];
    switch (by) {
      case "interval_score":
        return [r.intervalScore];
      case "rolling_gap":
        return [Math.abs(r.rollingCoverage - target), r.rollingWidth];
      case "validity_then_width":
        return r.valid ? [0, r.meanWidth] : [1, r.gap];
    }
  };
  const sorted = [...ext].sort((a, b) => {
    const ka = key(a), kb = key(b);
    for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
      const d = (ka[i] ?? 0) - (kb[i] ?? 0);
      if (d !== 0 && !Number.isNaN(d)) return d;
    }
    return 0;
  });
  sorted.forEach((r, i) => (r.rank = i + 1));
  return sorted;
}
