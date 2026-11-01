/**
 * Calibration diagnostics, computed online from the same evaluated (origin, lead) pairs the
 * metrics use.
 *
 * PIT: for each pair, where the truth fell in the native predictive distribution, read off
 * the model's quantile vector by linear interpolation (below the lowest quantile counts as 0,
 * above the highest as 1). Where several quantiles tie at the truth (a flat series, or
 * forecasts clipped at zero) the PIT is a range, and a point is drawn uniformly from it (the
 * randomised PIT), seeded by the pair so replays agree. A calibrated forecaster gives a flat
 * histogram.
 *
 * Reliability: nominal versus empirical coverage over several alpha levels. The native curve
 * checks each truth against the interpolated native interval at every level, exactly as the
 * engine scores the native row (a PIT value alone would miss ties). A conformal method is
 * calibrated at one alpha only, so the probe runs a shadow engine per extra alpha with just
 * that method enabled, fed the same series and stored forecasts. Engines are deterministic
 * in (series, forecasts, settings), so each shadow equals what the main engine would report
 * at that alpha.
 */

import { Engine, NATIVE_ID, quantileAt } from "./engine";

/**
 * Range of probabilities `[F(y-), F(y)]` of `y` under the piecewise-linear CDF through the
 * quantile vector at `offset`. The two ends differ only where quantiles tie at `y`.
 */
export function pitRange(q: Float32Array, offset: number, levels: number[], y: number): [number, number] {
  const Q = levels.length;
  if (!(y >= q[offset])) return [0, 0];
  if (y > q[offset + Q - 1]) return [1, 1];
  const at = (i: number) => {
    if (i <= 0) return levels[0];
    if (i >= Q) return levels[Q - 1];
    const a = q[offset + i - 1], b = q[offset + i];
    return b > a ? levels[i - 1] + ((y - a) / (b - a)) * (levels[i] - levels[i - 1]) : levels[i];
  };
  // Lower end: the first quantile at or above y. Upper end: just past the last one at or below it.
  let i = 0;
  while (i < Q - 1 && q[offset + i] < y) i++;
  let j = Q - 1;
  while (j > 0 && q[offset + j] > y) j--;
  const lo = at(i);
  const hi = j === Q - 1 ? levels[Q - 1] : Math.max(lo, at(j + 1));
  return [lo, hi];
}

/** PIT of `y`; with ties, the point `u` (in [0, 1]) of the way through the tied range. */
export function pitValue(q: Float32Array, offset: number, levels: number[], y: number, u = 0): number {
  const [lo, hi] = pitRange(q, offset, levels, y);
  return lo + u * (hi - lo);
}

/** Deterministic uniform in [0, 1) from a pair's target time and lead. */
function pairUniform(t: number, k: number): number {
  let h = Math.imul(t ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(k + 0x27d4eb2f, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** Counts per bin over [0, 1]; a PIT of exactly 1 goes in the last bin. */
export function pitHistogram(pits: readonly number[], bins = 10): number[] {
  const out = new Array(bins).fill(0);
  for (const p of pits) out[Math.min(bins - 1, Math.max(0, Math.floor(p * bins)))]++;
  return out;
}

/** Nominal coverages for the native curve (cheap: no extra engines). */
export const NATIVE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98];
/** Nominal coverages for the conformal curve (each one costs a shadow engine). */
export const METHOD_GRID = [0.5, 0.7, 0.8, 0.9, 0.95];

export interface ReliabilityPoint {
  nominal: number;
  empirical: number;
  n: number;
}

export interface Reliability {
  native: ReliabilityPoint[];
  /** Empty when the focus is the native interval or no method is focused. */
  method: ReliabilityPoint[];
}

const same = (a: number, b: number) => Math.abs(a - b) < 1e-9;

export class CalibrationProbe {
  /** PIT values of the native forecast per lead (index k - 1), in arrival order. */
  private pits: number[][] = [];
  /** Per lead: native pairs covered at each `NATIVE_GRID` level. */
  private covered: number[][] = [];
  private cursor: number[] = [];
  private shadows: { alpha: number; engine: Engine }[] = [];
  private metricsRef: unknown = null;
  private key = "";
  private methodId: string | null = null;
  private pending = 0;
  private total = 0;

  /**
   * Bring the probe up to date with `main`. Any replay of the main engine (settings, edits,
   * new forecasts for old origins) replaces its metrics, which triggers a rebuild here;
   * otherwise only the new forecasts and arrivals are processed. Shadow work stops after
   * `budgetMs`; check `ready` and call again to continue.
   */
  sync(main: Engine, methodId: string | null, budgetMs = Infinity): void {
    const id = methodId && methodId !== NATIVE_ID && main.settings.enabled[methodId] ? methodId : null;
    const key = `${id}|${JSON.stringify(main.settings)}`;
    if (main.metrics !== this.metricsRef || key !== this.key) this.rebuild(main, id, key);
    else if (this.shadows.some(({ engine }) => engine.processedUpTo > main.processedUpTo)) this.rebuild(main, id, key);
    this.syncPits(main);
    this.syncShadows(main, budgetMs);
  }

  private rebuild(main: Engine, id: string | null, key: string): void {
    this.metricsRef = main.metrics;
    this.key = key;
    this.methodId = id;
    this.pits = [];
    this.covered = [];
    this.cursor = [];
    this.shadows = [];
    if (!id) return;
    const target = 1 - main.settings.alpha;
    for (const c of METHOD_GRID) {
      if (same(c, target)) continue; // the main engine already runs this level
      const settings = {
        ...main.settings,
        alpha: 1 - c,
        enabled: { [id]: true },
        params: { [id]: main.settings.params[id] },
      };
      this.shadows.push({ alpha: 1 - c, engine: new Engine(settings) });
    }
  }

  private syncPits(main: Engine): void {
    const m = main.metrics.get(NATIVE_ID);
    if (!m) return;
    m.lead.forEach((L, j) => {
      const k = j + 1;
      const out = (this.pits[j] ??= []);
      const cov = (this.covered[j] ??= NATIVE_GRID.map(() => 0));
      let i = this.cursor[j] ?? 0;
      for (; i < L.t.length; i++) {
        const t = L.t[i];
        const f = main.forecasts.get(t - k);
        if (!f) continue;
        const y = main.series[t], off = j * f.levels.length;
        out.push(pitValue(f.quantiles, off, f.levels, y, pairUniform(t, k)));
        NATIVE_GRID.forEach((c, g) => {
          const a = (1 - c) / 2;
          if (y >= quantileAt(f.quantiles, off, f.levels, a) && y <= quantileAt(f.quantiles, off, f.levels, 1 - a)) cov[g]++;
        });
      }
      this.cursor[j] = i;
    });
  }

  /**
   * Feed the shadows the forecasts they have not seen, round-robin so they advance together,
   * for at most `budgetMs`. A rebuild after a focus change can take seconds on a long series
   * with a heavy method, so the caller keeps calling until `ready`.
   */
  private syncShadows(main: Engine, budgetMs: number): void {
    this.pending = 0;
    if (!this.shadows.length) return;
    const t0 = performance.now();
    const origins = [...main.forecasts.keys()].sort((a, b) => a - b);
    for (const { engine } of this.shadows) engine.series = main.series;
    let i = 0, fed = 0;
    for (; i < origins.length; i++) {
      const o = origins[i];
      if (this.shadows.every(({ engine }) => engine.forecasts.has(o))) continue;
      // Always make some progress, however small the budget.
      if (fed++ > 0 && performance.now() - t0 > budgetMs) break;
      const f = main.forecasts.get(o)!;
      for (const { engine } of this.shadows) if (!engine.forecasts.has(o)) engine.addForecast(o, f.quantiles, f.horizon, f.levels);
    }
    this.total = origins.length;
    this.pending = origins.length - i;
    if (!this.pending) for (const { engine } of this.shadows) engine.catchUp(main.processedUpTo);
  }

  /** Whether the method curve is current (false while a rebuild is still catching up). */
  get ready(): boolean {
    return this.pending === 0;
  }

  /** Share of stored forecasts the shadows have processed, for a progress note. */
  get progress(): number {
    return this.total ? 1 - this.pending / this.total : 1;
  }

  /** Native PIT values, pooled over all scored leads or at one lead (1-based). */
  pit(lead?: number): number[] {
    if (lead !== undefined) return this.pits[lead - 1]?.slice() ?? [];
    return this.pits.flat();
  }

  reliability(main: Engine, lead?: number): Reliability {
    const pits = this.pit(lead);
    const rows = lead !== undefined ? [this.covered[lead - 1]].filter(Boolean) : this.covered;
    const native = NATIVE_GRID.map((c, g) => {
      let k = 0;
      for (const r of rows) k += r[g];
      return { nominal: c, empirical: pits.length ? k / pits.length : NaN, n: pits.length };
    });
    const method: ReliabilityPoint[] = [];
    const id = this.methodId;
    if (id && this.ready) {
      const row = (e: Engine) => e.table(lead).find((r) => r.id === id);
      const points: [number, Engine][] = [[main.settings.alpha, main], ...this.shadows.map((s) => [s.alpha, s.engine] as [number, Engine])];
      for (const [alpha, e] of points) {
        const r = row(e);
        if (r && r.n > 0) method.push({ nominal: 1 - alpha, empirical: r.coverage, n: r.n });
      }
      method.sort((a, b) => a.nominal - b.nominal);
    }
    return { native, method };
  }

  get focusedMethod(): string | null {
    return this.methodId;
  }
}
