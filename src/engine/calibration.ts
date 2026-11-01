/**
 * Calibration diagnostics, computed online from the same evaluated (origin, lead) pairs the
 * metrics use.
 *
 * PIT: for each pair, where the truth fell in the native predictive distribution, read off
 * the model's quantile vector by linear interpolation (below the lowest quantile counts as 0,
 * above the highest as 1). A calibrated forecaster gives a flat histogram.
 *
 * Reliability: nominal versus empirical coverage over several alpha levels. The native curve
 * follows from the PIT values, since the native central interval at coverage c contains the
 * truth exactly when its PIT lies in [(1 - c) / 2, (1 + c) / 2]. A conformal method is
 * calibrated at one alpha only, so the probe runs a shadow engine per extra alpha with just
 * that method enabled, fed the same series and stored forecasts. Engines are deterministic
 * in (series, forecasts, settings), so each shadow equals what the main engine would report
 * at that alpha.
 */

import { Engine, NATIVE_ID } from "./engine";

/** Probability of `y` under the piecewise-linear CDF through the quantile vector at `offset`. */
export function pitValue(q: Float32Array, offset: number, levels: number[], y: number): number {
  const Q = levels.length;
  if (!(y >= q[offset])) return 0;
  if (y > q[offset + Q - 1]) return 1;
  let i = 0;
  while (i < Q - 1 && q[offset + i] < y) i++;
  if (i === 0) return levels[0];
  const a = q[offset + i - 1], b = q[offset + i];
  if (!(b > a)) return levels[i];
  return levels[i - 1] + ((y - a) / (b - a)) * (levels[i] - levels[i - 1]);
}

/** Counts per bin over [0, 1]; a PIT of exactly 1 goes in the last bin. */
export function pitHistogram(pits: readonly number[], bins = 10): number[] {
  const out = new Array(bins).fill(0);
  for (const p of pits) out[Math.min(bins - 1, Math.max(0, Math.floor(p * bins)))]++;
  return out;
}

/** Empirical coverage of central native intervals at nominal coverage `c`, from PIT values. */
export function pitCoverage(pits: readonly number[], c: number): number {
  if (!pits.length) return NaN;
  const lo = (1 - c) / 2, hi = (1 + c) / 2;
  let k = 0;
  for (const p of pits) if (p >= lo - 1e-12 && p <= hi + 1e-12) k++;
  return k / pits.length;
}

/** Nominal coverages for the native curve (cheap: read off the PIT values). */
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
      let i = this.cursor[j] ?? 0;
      for (; i < L.t.length; i++) {
        const t = L.t[i];
        const f = main.forecasts.get(t - k);
        if (!f) continue;
        out.push(pitValue(f.quantiles, j * f.levels.length, f.levels, main.series[t]));
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
    const native = NATIVE_GRID.map((c) => ({ nominal: c, empirical: pitCoverage(pits, c), n: pits.length }));
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
