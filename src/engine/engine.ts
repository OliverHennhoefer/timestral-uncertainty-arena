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
import type { QuantileMethod } from "../conformal/multiqt";
import { ScoreHistory } from "../conformal/window";
import { type ArrivalSeries, type Uncertainty, blockBootstrap, indistinguishable } from "./bootstrap";

export type ScoreType = "cqr" | "cqr_asym" | "abs" | "scaled";

/** Asymmetric scores calibrate the lower and upper bound separately, each at alpha / 2. */
export const isAsymmetric = (t: ScoreType) => t === "cqr_asym";

export interface Settings {
  alpha: number;
  /** Max lead used for scoring and metrics (<= the forecast horizon). */
  evalLeads: number;
  scoreType: ScoreType;
  window: number;
  warmup: number;
  /** Rolling window for the live curves, in time steps (arrivals), not pairs. */
  rolling: number;
  enabled: Record<string, boolean>;
  params: Record<string, Record<string, number>>;
}

export const NATIVE_ID = "native";
export const NATIVE_COLOR = "#8E8E93";
const QUANTILE_IDS = new Set(METHODS.filter((m) => m.quantile).map((m) => m.id));

export function defaultSettings(): Settings {
  return {
    alpha: 0.1,
    evalLeads: 16,
    scoreType: "cqr",
    window: 100,
    warmup: 10,
    rolling: 100,
    enabled: Object.fromEntries(METHODS.map((m) => [m.id, m.defaultOn !== false])),
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
  /**
   * Upper-side thresholds for asymmetric scores (`thresholds` then holds the lower side). For
   * quantile-level methods the two maps hold the interval's lower and upper bound themselves.
   */
  thresholdsUp: Map<string, Float64Array>;
  /** Whether lead k - 1 was past warm-up when issued (counts toward metrics). */
  evaluated: Uint8Array;
}

export interface MethodMetrics {
  id: string;
  n: number;
  covered: number;
  widthSum: number;
  intervalScoreSum: number;
  /** Sums per arrival with evaluations (aligned with `Engine.traceT`), for rolling windows and the bootstrap. */
  arr: { n: number[]; cov: number[]; width: number[]; is: number[] };
  /**
   * Per lead (index k - 1): one entry per evaluated pair, in arrival order. `below` is how far
   * the value fell under the lower bound and `above` how far over the upper bound (negative:
   * how far inside that bound). Their sum is minus the interval's width, also when the bounds
   * cross and `width` is clamped to zero.
   */
  lead: LeadPairs[];
  /** Running sums over arrivals in the last `rolling` time steps; `from` indexes `arr`. */
  roll: { n: number; cov: number; width: number; from: number };
  /** Rolling coverage / width after each arrival with evaluations. */
  covTrace: number[];
  widthTrace: number[];
}

/** The evaluated pairs of one method at one lead. */
export interface LeadPairs {
  t: number[];
  cov: number[];
  width: number[];
  is: number[];
  below: number[];
  above: number[];
}

/** What `fitLead` derives from one lead's pairs; cached until the lead gets a new pair. */
interface LeadFit {
  n: number;
  /** Margin that minimises the lead's interval score, and the summed score at that margin. */
  margin: number;
  shapeSum: number;
  /** Summed width at the margin that covers exactly the target share of the lead's pairs. */
  widthSum: number;
}

export interface MetricsRow {
  id: string;
  n: number;
  coverage: number;
  rollingCoverage: number;
  meanWidth: number;
  rollingWidth: number;
  intervalScore: number;
  /**
   * Interval score after moving both bounds out (or in) by one margin per lead, chosen in
   * hindsight to minimise that lead's score. At that margin the lead misses alpha of the
   * time, so this compares the shape of the intervals with their level set right.
   */
  shapeScore: number;
  /** `intervalScore - shapeScore`, never negative: what the wrong level costs in score units. */
  levelCost: number;
  /**
   * Mean width after moving both bounds out (or in) by one margin per lead, chosen so that
   * the target share of that lead's pairs is covered: sharpness compared at equal coverage.
   */
  widthAtTarget: number;
  /** Lowest coverage over any full rolling window of `rolling` time steps (NaN until one fills). */
  worstCoverage: number;
}

/** k-th smallest value (0-based) of `a`, which is reordered in place (quickselect). */
function kthSmallest(a: Float64Array, k: number): number {
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
}

/**
 * Smallest value whose cumulative weight (`before` plus the summed weights of all values up to
 * and including it) reaches `target`. Both arrays are reordered in place (weighted quickselect).
 */
function weightedSelect(v: Float64Array, w: Float64Array, target: number, before = 0): number {
  let lo = 0, hi = v.length - 1;
  while (lo < hi) {
    const pivot = v[(lo + hi) >> 1];
    // Three-way partition of [lo, hi] into < pivot, = pivot, > pivot.
    let lt = lo, i = lo, gt = hi;
    while (i <= gt) {
      if (v[i] < pivot) {
        const tv = v[i]; v[i] = v[lt]; v[lt] = tv;
        const tw = w[i]; w[i] = w[lt]; w[lt] = tw;
        i++; lt++;
      } else if (v[i] > pivot) {
        const tv = v[i]; v[i] = v[gt]; v[gt] = tv;
        const tw = w[i]; w[i] = w[gt]; w[gt] = tw;
        gt--;
      } else i++;
    }
    let less = 0, equal = 0;
    for (let j = lo; j < lt; j++) less += w[j];
    for (let j = lt; j <= gt; j++) equal += w[j];
    if (before + less >= target) hi = lt - 1;
    else if (before + less + equal >= target) return pivot;
    else {
      before += less + equal;
      lo = gt + 1;
    }
  }
  return v[Math.max(0, Math.min(lo, v.length - 1))];
}

/** Interval score of one pair after moving both bounds out by `margin` (see `LeadPairs`). */
export function shiftedScore(below: number, above: number, margin: number, alpha: number): number {
  return Math.max(0, 2 * margin - below - above) + (2 / alpha) * (Math.max(0, below - margin) + Math.max(0, above - margin));
}

/**
 * The margin that, added to both bounds of every pair, minimises their summed interval score.
 *
 * The summed score is convex and piecewise linear in the margin. Its slope rises by 2 / alpha
 * where a side stops being a miss (at the pair's `below` or `above`) and by 2 where an empty
 * interval starts to have width (at half their sum), and it starts at -4n / alpha. The minimum
 * is the first breakpoint where the slope is no longer negative. Unless intervals are empty
 * there, that is where the two sides together miss alpha of the time (Gneiting and Resin 2023,
 * Theorem 2.26, for each bound).
 */
export function bestMargin(below: ArrayLike<number>, above: ArrayLike<number>, alpha: number): number {
  const n = below.length;
  if (!n) return NaN;
  // In units of 2 / alpha a miss breakpoint weighs 1 and a width breakpoint alpha, and the
  // weights at or below the minimum must add up to 2n. The n width breakpoints weigh at most
  // alpha * n together, so the minimum is no smaller than this order statistic of the 2n miss
  // breakpoints. Only the few breakpoints from there on need the weighted search.
  const eps = 1e-9;
  const d = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    d[2 * i] = below[i];
    d[2 * i + 1] = above[i];
  }
  const floor = kthSmallest(d, Math.max(1, Math.ceil(2 * n - alpha * n - eps)) - 1);
  let before = 0, m = 0;
  for (let i = 0; i < n; i++) {
    if (below[i] < floor) before += 1; else m++;
    if (above[i] < floor) before += 1; else m++;
    if ((below[i] + above[i]) / 2 < floor) before += alpha; else m++;
  }
  const v = new Float64Array(m), w = new Float64Array(m);
  for (let i = 0, o = 0; i < n; i++) {
    const half = (below[i] + above[i]) / 2;
    if (below[i] >= floor) { v[o] = below[i]; w[o++] = 1; }
    if (above[i] >= floor) { v[o] = above[i]; w[o++] = 1; }
    if (half >= floor) { v[o] = half; w[o++] = alpha; }
  }
  return weightedSelect(v, w, 2 * n - eps, before);
}

/** Mean interval score with every lead moved to its own `bestMargin`. Each part is one lead. */
export function shapeScore(parts: { below: ArrayLike<number>; above: ArrayLike<number> }[], alpha: number): number {
  let n = 0, sum = 0;
  for (const p of parts) {
    const margin = bestMargin(p.below, p.above, alpha);
    for (let i = 0; i < p.below.length; i++) sum += shiftedScore(p.below[i], p.above[i], margin, alpha);
    n += p.below.length;
  }
  return n ? sum / n : NaN;
}

/** Summed width of one lead's pairs at the smallest margin that covers the target share of them. */
function widthSumAtTarget(below: ArrayLike<number>, above: ArrayLike<number>, target: number): number {
  const n = below.length;
  if (!n) return 0;
  // A pair is covered once the margin reaches the larger of its two distances.
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.max(below[i], above[i]);
  const margin = kthSmallest(d, Math.min(n, Math.max(1, Math.ceil(target * n - 1e-9))) - 1);
  let w = 0;
  for (let i = 0; i < n; i++) w += Math.max(0, 2 * margin - below[i] - above[i]);
  return w;
}

/** See `MetricsRow.widthAtTarget`. Each part is one lead and gets its own margin. */
export function widthAtTarget(parts: { below: ArrayLike<number>; above: ArrayLike<number> }[], target: number): number {
  let n = 0, w = 0;
  for (const p of parts) {
    n += p.below.length;
    w += widthSumAtTarget(p.below, p.above, target);
  }
  return n ? w / n : NaN;
}

/**
 * Lowest coverage over sliding windows of the last `R` time steps, counting only windows that
 * span a full `R` steps after the first evaluation. `t` must be increasing; `n` and `cov` hold
 * pairs and covered pairs at each `t`.
 */
export function worstWindowCoverage(t: ArrayLike<number>, n: ArrayLike<number>, cov: ArrayLike<number>, R: number): number {
  let first = 0;
  while (first < t.length && !(n[first] > 0)) first++;
  let worst = Infinity, from = first, sn = 0, sc = 0;
  for (let e = first; e < t.length; e++) {
    sn += n[e];
    sc += cov[e];
    while (t[from] <= t[e] - R) {
      sn -= n[from];
      sc -= cov[from];
      from++;
    }
    if (t[e] - t[first] >= R - 1 && sn > 0) worst = Math.min(worst, sc / sn);
  }
  return Number.isFinite(worst) ? worst : NaN;
}

/** Linear interpolation of a quantile vector at probability `p`. */
export function quantileAt(q: ArrayLike<number>, offset: number, levels: number[], p: number): number {
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
  /** Upper-side histories and methods, used only by asymmetric scores. */
  private historiesUp: ScoreHistory[] = [];
  private methodsUp: Map<string, OnlineMethod>[] = [];
  /** State shared across the leads of one side, for methods that pool leads. */
  private shared = new Map<string, unknown>();
  private sharedUp = new Map<string, unknown>();
  /** Quantile-level methods per lead, created at the first forecast (they need its levels). */
  private quantileMethods: Map<string, QuantileMethod>[] = [];
  metrics = new Map<string, MethodMetrics>();
  /** `fitLead` results per row id and lead, reused while that lead has the same pairs. */
  private fits = new Map<string, LeadFit[]>();
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

  /** Methods that calibrate a scalar threshold on the score. */
  private get scalarSpecs(): MethodSpec[] {
    return this.activeSpecs.filter((m) => !m.quantile);
  }

  get rowIds(): string[] {
    return [NATIVE_ID, ...this.activeSpecs.map((m) => m.id)];
  }

  private reset(): void {
    this.histories = [];
    this.methods = [];
    this.historiesUp = [];
    this.methodsUp = [];
    this.shared = new Map();
    this.sharedUp = new Map();
    this.quantileMethods = [];
    this.metrics = new Map();
    this.fits = new Map();
    this.traceT = [];
    this.processedUpTo = -1;
    for (const id of this.rowIds) this.metrics.set(id, this.newMetrics(id));
  }

  private newMetrics(id: string): MethodMetrics {
    return {
      id, n: 0, covered: 0, widthSum: 0, intervalScoreSum: 0,
      arr: { n: [], cov: [], width: [], is: [] },
      lead: Array.from({ length: this.settings.evalLeads }, () => ({ t: [], cov: [], width: [], is: [], below: [], above: [] })),
      roll: { n: 0, cov: 0, width: 0, from: 0 },
      covTrace: [], widthTrace: [],
    };
  }

  /** Per-lead state; `up` selects the upper side of an asymmetric score. */
  private lead(k: number, up = false): { history: ScoreHistory; methods: Map<string, OnlineMethod> } {
    const histories = up ? this.historiesUp : this.histories;
    const methods = up ? this.methodsUp : this.methods;
    // Each side of an asymmetric score targets alpha / 2, so the two-sided interval targets 1 - alpha.
    const a = isAsymmetric(this.settings.scoreType) ? this.settings.alpha / 2 : this.settings.alpha;
    while (histories.length < k) {
      const h = new ScoreHistory(this.settings.window);
      histories.push(h);
      const m = new Map<string, OnlineMethod>();
      const ctx = { lead: histories.length, shared: up ? this.sharedUp : this.shared };
      for (const spec of this.scalarSpecs) m.set(spec.id, spec.create(h, a, { ...defaultParams(spec), ...this.settings.params[spec.id] }, ctx));
      methods.push(m);
    }
    return { history: histories[k - 1], methods: methods[k - 1] };
  }

  /** Quantile-level methods of lead `k`, created for the quantile levels of the forecasts. */
  private quantileLead(k: number, levels: readonly number[]): Map<string, QuantileMethod> {
    while (this.quantileMethods.length < k) {
      const m = new Map<string, QuantileMethod>();
      const ctx = { lead: this.quantileMethods.length + 1, shared: this.shared };
      for (const spec of this.activeSpecs) {
        if (spec.quantile) m.set(spec.id, spec.quantile(levels, { ...defaultParams(spec), ...this.settings.params[spec.id] }, ctx));
      }
      this.quantileMethods.push(m);
    }
    return this.quantileMethods[k - 1];
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
      thresholds: new Map(), thresholdsUp: new Map(), evaluated: new Uint8Array(horizon),
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
      thresholds: new Map(), thresholdsUp: new Map(), evaluated: new Uint8Array(horizon),
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
    f.thresholdsUp = new Map();
    const H = Math.min(f.horizon, this.settings.evalLeads);
    const sides = isAsymmetric(this.settings.scoreType) ? [false, true] : [false];
    for (const spec of this.activeSpecs) {
      f.thresholds.set(spec.id, new Float64Array(f.horizon).fill(NaN));
      if (sides.length > 1 || spec.quantile) f.thresholdsUp.set(spec.id, new Float64Array(f.horizon).fill(NaN));
    }
    for (let k = 1; k <= H; k++) {
      for (const up of sides) {
        const { history, methods } = this.lead(k, up);
        if (!up) f.evaluated[k - 1] = history.count >= this.settings.warmup ? 1 : 0;
        const cap = history.maxAbsScore;
        const out = up ? f.thresholdsUp : f.thresholds;
        for (const [id, m] of methods) {
          let q = m.predict(f.origin);
          // Thresholds beyond the largest score seen (including +/-inf) are capped there.
          if (history.count === 0) q = NaN;
          else if (Number.isNaN(q)) q = cap;
          else q = Math.max(-cap, Math.min(cap, q));
          out.get(id)![k - 1] = q;
        }
      }
      // Quantile-level methods recalibrate the whole quantile vector; the interval is read off it.
      const Q = f.levels.length;
      for (const [id, m] of this.quantileLead(k, f.levels)) {
        const played = m.predict(f.origin, Float64Array.from(f.quantiles.subarray((k - 1) * Q, k * Q)));
        f.thresholds.get(id)![k - 1] = quantileAt(played, 0, f.levels, this.settings.alpha / 2);
        f.thresholdsUp.get(id)![k - 1] = quantileAt(played, 0, f.levels, 1 - this.settings.alpha / 2);
      }
    }
  }

  /**
   * Score of truth `y` at lead index `j` of forecast `f`. For an asymmetric score this is the
   * lower-side score (how far `y` fell below the lower bound); `scoreUp` gives the upper side.
   */
  score(f: StoredForecast, j: number, y: number): number {
    switch (this.settings.scoreType) {
      case "cqr":
        return Math.max(f.lo[j] - y, y - f.hi[j]);
      case "cqr_asym":
        return f.lo[j] - y;
      case "abs":
        return Math.abs(y - f.median[j]);
      case "scaled":
        return Math.abs(y - f.median[j]) / Math.max((f.hi[j] - f.lo[j]) / 2, 1e-9);
    }
  }

  /**
   * The band a method issued `k` steps ahead, for every target time in [from, to] that had
   * a forecast made at `t - k`: entries `[t, lo, hi, missed]`. `missed` is only set for pairs
   * that counted toward the metrics (past warm-up) and whose truth is known.
   */
  pastBand(id: string, k: number, from: number, to: number): [number, number, number, boolean][] {
    const out: [number, number, number, boolean][] = [];
    for (let t = Math.max(k, from); t <= to; t++) {
      const f = this.forecasts.get(t - k);
      if (!f || k > f.horizon) continue;
      const iv = this.interval(f, k - 1, id);
      if (!iv) continue;
      const y = this.series[t];
      const missed = y !== undefined && t <= this.processedUpTo && !!f.evaluated[k - 1] && (y < iv[0] || y > iv[1]);
      out.push([t, iv[0], iv[1], missed]);
    }
    return out;
  }

  /** Upper-side score of an asymmetric score: how far `y` rose above the upper bound. */
  scoreUp(f: StoredForecast, j: number, y: number): number {
    return y - f.hi[j];
  }

  /** Interval of method `id` at lead index `j` (threshold `q`), or the native interval. */
  interval(f: StoredForecast, j: number, id: string): [number, number] | null {
    if (id === NATIVE_ID) return [f.lo[j], f.hi[j]];
    const q = f.thresholds.get(id)?.[j];
    if (q === undefined || Number.isNaN(q)) return null;
    if (QUANTILE_IDS.has(id)) return [q, f.thresholdsUp.get(id)![j]];
    switch (this.settings.scoreType) {
      case "cqr":
        return [f.lo[j] - q, f.hi[j] + q];
      case "cqr_asym": {
        const qu = f.thresholdsUp.get(id)?.[j];
        if (qu === undefined || Number.isNaN(qu)) return null;
        return [f.lo[j] - q, f.hi[j] + qu];
      }
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
    const ids = this.rowIds;
    const acc = ids.map(() => ({ n: 0, cov: 0, width: 0, is: 0 }));
    for (let k = 1; k <= maxK; k++) {
      const f = this.forecasts.get(t - k);
      if (!f || k > f.horizon) continue;
      const j = k - 1;
      const s = this.score(f, j, y);
      if (!Number.isFinite(s)) continue;
      const { history, methods } = this.lead(k);
      if (f.evaluated[j]) {
        evaluatedAny = true;
        ids.forEach((id, i) => {
          const iv = this.interval(f, j, id);
          const m = this.metrics.get(id);
          if (!iv || !m) return;
          const [lo, hi] = iv;
          const width = Math.max(0, hi - lo);
          const cov = y >= lo && y <= hi ? 1 : 0;
          const is = width + (2 / alpha) * Math.max(0, lo - y) + (2 / alpha) * Math.max(0, y - hi);
          m.n++;
          m.covered += cov;
          m.widthSum += width;
          m.intervalScoreSum += is;
          const L = m.lead[j];
          L.t.push(t);
          L.cov.push(cov);
          L.width.push(width);
          L.is.push(is);
          L.below.push(lo - y);
          L.above.push(y - hi);
          const a = acc[i];
          a.n++;
          a.cov += cov;
          a.width += width;
          a.is += is;
        });
      }
      for (const [, m] of methods) m.update(f.origin, s);
      history.push(s);
      for (const [, m] of this.quantileMethods[j] ?? []) m.update(f.origin, y);
      if (isAsymmetric(this.settings.scoreType)) {
        const su = this.scoreUp(f, j, y);
        const upper = this.lead(k, true);
        for (const [, m] of upper.methods) m.update(f.origin, su);
        upper.history.push(su);
      }
    }
    if (!evaluatedAny) {
      // Time still advances: entries older than R steps leave the window even without new pairs.
      for (const id of ids) this.expire(this.metrics.get(id)!, t, this.settings.rolling);
    } else {
      this.traceT.push(t);
      const R = this.settings.rolling;
      ids.forEach((id, i) => {
        const m = this.metrics.get(id)!;
        const a = acc[i];
        m.arr.n.push(a.n);
        m.arr.cov.push(a.cov);
        m.arr.width.push(a.width);
        m.arr.is.push(a.is);
        m.roll.n += a.n;
        m.roll.cov += a.cov;
        m.roll.width += a.width;
        this.expire(m, t, R);
        const r = rolling(m);
        m.covTrace.push(r.coverage);
        m.widthTrace.push(r.width);
      });
    }
  }

  /** Drop arrivals at or before `t - R` from the rolling sums (window = the last R time steps). */
  private expire(m: MethodMetrics, t: number, R: number): void {
    const r = m.roll;
    while (r.from < m.arr.n.length && this.traceT[r.from] <= t - R) {
      r.n -= m.arr.n[r.from];
      r.cov -= m.arr.cov[r.from];
      r.width -= m.arr.width[r.from];
      r.from++;
    }
  }

  /** Coverage and mean width per lead (1..evalLeads), over all evaluated pairs. */
  leadProfile(id: string): { coverage: number[]; width: number[] } {
    const m = this.metrics.get(id);
    const coverage: number[] = [], width: number[] = [];
    for (const L of m?.lead ?? []) {
      const n = L.cov.length;
      let c = 0, w = 0;
      for (let i = 0; i < n; i++) { c += L.cov[i]; w += L.width[i]; }
      coverage.push(n ? c / n : NaN);
      width.push(n ? w / n : NaN);
    }
    return { coverage, width };
  }

  /** Margins, shape score and width at target of one row at one lead (index `j`), cached. */
  private fitLead(id: string, j: number): LeadFit {
    let fits = this.fits.get(id);
    if (!fits) this.fits.set(id, (fits = []));
    const L = this.metrics.get(id)?.lead[j];
    const n = L?.cov.length ?? 0;
    const have = fits[j];
    if (have && have.n === n) return have;
    const fit: LeadFit = { n, margin: NaN, shapeSum: 0, widthSum: 0 };
    if (L && n) {
      const { alpha } = this.settings;
      fit.margin = bestMargin(L.below, L.above, alpha);
      for (let i = 0; i < n; i++) fit.shapeSum += shiftedScore(L.below[i], L.above[i], fit.margin, alpha);
      fit.widthSum = widthSumAtTarget(L.below, L.above, 1 - alpha);
    }
    return (fits[j] = fit);
  }

  /**
   * Block-bootstrap uncertainty of every row (pooled, or at one lead). Blocks span at least
   * `evalLeads` arrivals, the range over which overlapping forecasts share errors.
   *
   * The shape score is resampled with the margins held at their fitted values. The margins
   * minimise the score, so refitting them per replicate would change it only to second order.
   */
  uncertainty(lead?: number, B = 300): Map<string, Uncertainty> {
    const series = new Map<string, ArrivalSeries>();
    const { alpha } = this.settings;
    const traceT = this.traceT;
    let T = 0;
    for (const id of this.rowIds) {
      const m = this.metrics.get(id)!;
      if (lead === undefined) {
        // Shape scores per arrival: every lead's pairs, added up at the arrival they belong to.
        const shape = new Float64Array(m.arr.n.length);
        for (let j = 0; j < m.lead.length; j++) {
          const { t, below, above } = m.lead[j];
          if (!t.length) continue;
          const { margin } = this.fitLead(id, j);
          for (let i = 0, a = 0; i < t.length; i++) {
            while (traceT[a] < t[i]) a++;
            shape[a] += shiftedScore(below[i], above[i], margin, alpha);
          }
        }
        series.set(id, { ...m.arr, shape });
        T = Math.max(T, m.arr.n.length);
      } else {
        const L = m.lead[lead - 1];
        if (!L) continue;
        const { margin } = this.fitLead(id, lead - 1);
        const shape = new Float64Array(L.cov.length);
        for (let i = 0; i < shape.length; i++) shape[i] = shiftedScore(L.below[i], L.above[i], margin, alpha);
        series.set(id, { n: new Array(L.cov.length).fill(1), cov: L.cov, width: L.width, is: L.is, shape });
        T = Math.max(T, L.cov.length);
      }
    }
    const block = Math.max(this.settings.evalLeads, Math.ceil(Math.cbrt(T) * 2));
    const out = blockBootstrap(series, T, block, B, 12345, this.settings.rolling);
    // What the worst window of a method at the target would look like: the target minus the
    // drop from overall to worst-window coverage that this row's own misses show when resampled.
    const target = 1 - alpha;
    for (const u of out.values()) {
      if (u.worstDrop) u.worstWindow = { typical: target - u.worstDrop[0], low: target - u.worstDrop[1] };
    }
    return out;
  }

  /**
   * Metrics table. Without `lead`, pooled over all scored leads; with `lead` (1-based),
   * only the pairs at that lead (rolling over its last `rolling` pairs, one per arrival).
   */
  table(lead?: number): MetricsRow[] {
    if (lead !== undefined) {
      const R = this.settings.rolling;
      return this.rowIds.map((id) => {
        const L = this.metrics.get(id)!.lead[lead - 1];
        const n = L?.cov.length ?? 0;
        const sum = (a: number[], from = 0) => { let v = 0; for (let i = from; i < a.length; i++) v += a[i]; return v; };
        // Rolling over the last R time steps, ending at the latest processed arrival.
        let from = n;
        while (from > 0 && L.t[from - 1] > this.processedUpTo - R) from--;
        const nr = n - from;
        const intervalScore = n ? sum(L.is) / n : NaN;
        const fit = this.fitLead(id, lead - 1);
        const shapeScore = n ? fit.shapeSum / n : NaN;
        return {
          id, n,
          coverage: n ? sum(L.cov) / n : NaN,
          rollingCoverage: nr ? sum(L.cov, from) / nr : NaN,
          meanWidth: n ? sum(L.width) / n : NaN,
          rollingWidth: nr ? sum(L.width, from) / nr : NaN,
          intervalScore,
          shapeScore,
          levelCost: Math.max(0, intervalScore - shapeScore),
          widthAtTarget: n ? fit.widthSum / n : NaN,
          worstCoverage: L ? worstWindowCoverage(L.t, new Uint8Array(n).fill(1), L.cov, R) : NaN,
        };
      });
    }
    return this.rowIds.map((id) => {
      const m = this.metrics.get(id)!;
      const r = rolling(m);
      const intervalScore = m.n ? m.intervalScoreSum / m.n : NaN;
      let shapeSum = 0, widthSum = 0;
      for (let j = 0; j < m.lead.length; j++) {
        const fit = this.fitLead(id, j);
        shapeSum += fit.shapeSum;
        widthSum += fit.widthSum;
      }
      const shapeScore = m.n ? shapeSum / m.n : NaN;
      return {
        id,
        n: m.n,
        coverage: m.n ? m.covered / m.n : NaN,
        rollingCoverage: r.coverage,
        meanWidth: m.n ? m.widthSum / m.n : NaN,
        rollingWidth: r.width,
        intervalScore,
        shapeScore,
        levelCost: Math.max(0, intervalScore - shapeScore),
        widthAtTarget: m.n ? widthSum / m.n : NaN,
        worstCoverage: worstWindowCoverage(this.traceT, m.arr.n, m.arr.cov, this.settings.rolling),
      };
    });
  }
}

function rolling(m: MethodMetrics): { coverage: number; width: number } {
  if (m.roll.n <= 0) return { coverage: NaN, width: NaN };
  return { coverage: m.roll.cov / m.roll.n, width: m.roll.width / m.roll.n };
}

export type RankBy = "valid_then_score" | "interval_score" | "rolling_gap";
export const RANK_BY: RankBy[] = ["valid_then_score", "interval_score", "rolling_gap"];

export interface RankedRow extends MetricsRow {
  rank: number;
  gap: number;
  /**
   * Coverage is significantly under the target: its whole 95% block-bootstrap interval lies
   * below it. False while there is no interval yet, so a short series flags nothing.
   */
  under: boolean;
  /** True when this row shares its rank with the row above (not distinguishable). */
  tied?: boolean;
}

/**
 * Rank methods (including the native interval) by the chosen criterion.
 *
 * `valid_then_score` puts every row that is not significantly under the target first, ordered
 * by interval (Winkler) score, and the rows that are under it last, ordered by how far they fall
 * short. The interval score charges a row for all the width it issued, so a row that covers more
 * than the target pays for its extra width. The shape score and level cost, which take that
 * width away in hindsight, are diagnostics only: ranking by shape alone put an over-wide band
 * first whenever its errors were well placed.
 */
export function rankRows(rows: MetricsRow[], alpha: number, by: RankBy, unc?: Map<string, Uncertainty>): RankedRow[] {
  const target = 1 - alpha;
  const ext: RankedRow[] = rows.map((r) => {
    const hi = unc?.get(r.id)?.coverage[1];
    return { ...r, rank: 0, gap: Math.abs(r.coverage - target), under: hi !== undefined && hi < target };
  });
  const key = (r: RankedRow): number[] => {
    if (r.n === 0) return [Infinity];
    switch (by) {
      case "interval_score":
        return [r.intervalScore];
      case "rolling_gap":
        return [Math.abs(r.rollingCoverage - target), r.rollingWidth];
      case "valid_then_score":
        return r.under ? [1, target - r.coverage] : [0, r.intervalScore];
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
  // With bootstrap replicates, rows whose score cannot be told apart from the head of their
  // group share its rank (95% paired block-bootstrap interval contains 0). Rows that are
  // significantly under the target are ordered by their shortfall and never tie.
  if (unc && by !== "rolling_gap") {
    const samples = (id: string) => unc.get(id)?.scoreSamples;
    let head = sorted[0];
    for (let i = 1; i < sorted.length; i++) {
      const r = sorted[i];
      const a = samples(r.id), b = samples(head?.id ?? "");
      const comparable = by === "interval_score" || (!r.under && !head.under);
      if (comparable && r.n > 0 && head.n > 0 && a && b && indistinguishable(a, b)) {
        r.rank = head.rank;
        r.tied = true;
      } else {
        head = r;
      }
    }
  }
  return sorted;
}
