/**
 * What an exported animation shows, frame by frame.
 *
 * The engine is a deterministic function of (series, stored forecasts, settings), so a run
 * that was played once can be re-run offline without the model: a fresh engine is fed the
 * same observations and the forecasts stored at each origin, in the order the live run saw
 * them. At every frame time the state equals what the app showed at that step.
 */

import { Engine, NATIVE_ID, type Settings } from "../engine/engine";
import type { Align } from "../share";
import { Extent, fitRange } from "../ui/yaxis";

/** Steps shown at the start, before the window has filled. */
const MIN_WINDOW = 48;
/** How far beyond the observed range (in multiples of it) bands may stretch the y axis. */
const Y_HEADROOM = 0.5;

export interface SourceForecast {
  origin: number;
  horizon: number;
  levels: number[];
  quantiles: Float32Array;
}

/** A snapshot of the app taken when the export dialog opens. */
export interface ExportSource {
  series: number[];
  forecasts: SourceForecast[];
  settings: Settings;
  align: Align;
  /** Focused method, or the native interval. */
  focusId: string;
  title: string;
  modelName: string;
  names: Record<string, string>;
  colors: Record<string, string>;
}

export interface StatLine {
  id: string;
  name: string;
  color: string;
  coverage: number;
  width: number;
  n: number;
}

export interface CoverageTrace {
  id: string;
  color: string;
  native: boolean;
  t: number[];
  v: number[];
}

export interface FrameState {
  /** Newest observation shown. */
  t: number;
  /** Visible step range [x0, x1]. */
  x0: number;
  x1: number;
  /** Visible observations: `series[i]` for i in [s0, t]. */
  s0: number;
  series: readonly number[];
  /** Focused method's one-step interval at past times: [t, lo, hi, missed]. */
  past: [number, number, number, boolean][];
  /** Latest forecast, drawn from `origin + 1` for `median.length` steps (empty before the first). */
  origin: number;
  median: number[];
  nativeLo: number[];
  nativeHi: number[];
  /** Focused conformal band, possibly shorter than the median while leads warm up. */
  band: [number, number][];
  /** Target y range before smoothing. */
  yLo: number;
  yHi: number;
  /** Focused method first, then the native interval (one line when the focus is native). */
  stats: StatLine[];
  coverage: CoverageTrace[];
  target: number;
}

export class SceneReplay {
  readonly engine: Engine;
  private readonly byOrigin: Map<number, SourceForecast>;
  private t = -1;
  readonly focus: string;
  /** Drawn forecast length, constant over the run so the x axis does not change scale. */
  readonly drawnHorizon: number;

  constructor(readonly src: ExportSource) {
    this.engine = new Engine(structuredClone(src.settings));
    this.byOrigin = new Map(src.forecasts.map((f) => [f.origin, f]));
    this.focus = src.focusId === NATIVE_ID || src.settings.enabled[src.focusId] ? src.focusId : NATIVE_ID;
    const maxH = src.forecasts.reduce((m, f) => Math.max(m, f.horizon), 0);
    this.drawnHorizon = src.align === "full" ? maxH : Math.min(maxH, src.settings.evalLeads);
  }

  get time(): number {
    return this.t;
  }

  /** Feed observations and forecasts up to and including step `t`. */
  advanceTo(t: number): void {
    const e = this.engine;
    const end = Math.min(t, this.src.series.length - 1);
    while (this.t < end) {
      const i = ++this.t;
      e.append(this.src.series[i]);
      e.catchUp(i);
      const f = this.byOrigin.get(i);
      if (f) e.addForecast(i, f.quantiles, f.horizon, f.levels);
    }
  }

  /**
   * The frame at the current time. `history` is the number of observed steps kept on screen;
   * a run shorter than that is fitted to the width instead of scrolling. `from` is where the
   * export starts (the coverage chart spans [from, to]).
   */
  frame(history: number, from: number, to: number): FrameState {
    const e = this.engine;
    const t = this.t;
    const n = this.src.series.length;
    // The window grows with the series until it holds `history` steps, then scrolls, so
    // the first frames are not a short line in an empty chart.
    const full = Math.max(10, Math.min(history, n));
    const V = Math.min(full, Math.max(Math.min(full, MIN_WINDOW), t + 1));
    const span = V + this.drawnHorizon + 1;
    const x0 = Math.max(0, t - V + 1);
    const x1 = x0 + span;
    const s0 = x0;
    const fid = this.focus;

    const past = t >= 1 && e.settings.evalLeads >= 1 ? e.pastBand(fid, 1, Math.max(1, s0), t) : [];

    const origin = e.latestOrigin();
    const f = origin >= 0 ? e.forecasts.get(origin) : undefined;
    const median: number[] = [], nativeLo: number[] = [], nativeHi: number[] = [], band: [number, number][] = [];
    if (f) {
      const K = Math.min(f.horizon, e.settings.evalLeads);
      if (fid !== NATIVE_ID) {
        for (let j = 0; j < K; j++) {
          const iv = e.interval(f, j, fid);
          if (!iv) break;
          band.push(iv);
        }
      }
      // As in the app: aligned modes stop the native forecast where the conformal band ends.
      const H = this.src.align === "full" ? f.horizon : band.length || K;
      for (let j = 0; j < H; j++) {
        median.push(f.median[j]);
        nativeLo.push(f.lo[j]);
        nativeHi.push(f.hi[j]);
      }
    }

    // Fitted to what is on screen (no 0-100 floor as in the app): small screens need the
    // data to fill the chart. The painter eases between frames, so the axis does not jump.
    const obs = new Extent();
    for (let i = s0; i <= t; i++) obs.add(this.src.series[i]);
    const ext = new Extent();
    ext.add(obs.lo);
    ext.add(obs.hi);
    for (const [, l, u] of past) (ext.add(l), ext.add(u));
    for (let j = 0; j < median.length; j++) (ext.add(nativeLo[j]), ext.add(nativeHi[j]), ext.add(median[j]));
    for (const [l, u] of band) (ext.add(l), ext.add(u));
    // A band that blows up (e.g. thresholds capped after a level shift) is clipped rather
    // than allowed to squash the series into a thin line.
    const r = Math.max(1, obs.hi - obs.lo) * Y_HEADROOM;
    const [yLo, yHi] = fitRange(Math.max(ext.lo, obs.lo - r), Math.min(ext.hi, obs.hi + r), 0.04);

    const ids = fid === NATIVE_ID ? [NATIVE_ID] : [fid, NATIVE_ID];
    const stats = ids.map((id): StatLine => {
      const m = e.metrics.get(id);
      return {
        id,
        name: this.src.names[id] ?? id,
        color: this.src.colors[id] ?? "#888888",
        coverage: m && m.n ? m.covered / m.n : NaN,
        width: m && m.n ? m.widthSum / m.n : NaN,
        n: m?.n ?? 0,
      };
    });
    const coverage = ids.map((id): CoverageTrace => {
      const m = e.metrics.get(id);
      const ts: number[] = [], vs: number[] = [];
      if (m) {
        // Thin to about 600 points; the chart is never wider than that in logical pixels.
        const first = lowerBound(e.traceT, from);
        const step = Math.max(1, Math.floor((to - from + 1) / 600));
        for (let i = first; i < e.traceT.length; i += step) {
          ts.push(e.traceT[i]);
          vs.push(m.covTrace[i]);
        }
        const last = e.traceT.length - 1;
        if (last >= first && ts[ts.length - 1] !== e.traceT[last]) (ts.push(e.traceT[last]), vs.push(m.covTrace[last]));
      }
      return { id, color: this.src.colors[id] ?? "#888888", native: id === NATIVE_ID, t: ts, v: vs };
    });

    return { t, x0, x1, s0, series: this.src.series, past, origin, median, nativeLo, nativeHi, band, yLo, yHi, stats, coverage, target: 1 - e.settings.alpha };
  }
}

function lowerBound(a: number[], x: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}
