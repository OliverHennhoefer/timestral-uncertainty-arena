/**
 * Moving-block bootstrap for the ranking.
 *
 * Evaluated pairs are strongly dependent: forecasts made at neighbouring origins overlap,
 * and every arrival scores up to `evalLeads` of them at once. Resampling single pairs would
 * badly understate the uncertainty, so we resample contiguous blocks of arrivals (each
 * arrival carrying all of its pairs) with the same block indices for every method. That
 * keeps the comparison paired: two methods judged on the same resampled stretch of series.
 */

/** Per-arrival sums for one method: pairs, covered pairs, width, interval score and shape score. */
export interface ArrivalSeries {
  n: ArrayLike<number>;
  cov: ArrayLike<number>;
  width: ArrayLike<number>;
  is: ArrayLike<number>;
  /** Interval score with every lead at its fitted margin (see `MetricsRow.shapeScore`). */
  shape?: ArrayLike<number>;
}

export interface Uncertainty {
  /** 95% percentile intervals of the resampled statistics. */
  coverage: [number, number];
  width: [number, number];
  intervalScore: [number, number];
  /** Resampled mean interval scores, one per replicate (paired across methods). */
  scoreSamples: Float64Array;
  coverageSamples: Float64Array;
  /** Resampled mean widths, paired like `scoreSamples`. */
  widthSamples?: Float64Array;
  /** Shape score, when the series carries one; its replicates are paired like `scoreSamples`. */
  shapeScore?: [number, number];
  shapeSamples?: Float64Array;
  /**
   * How far the worst window falls below the overall coverage in the replicates: the median
   * and the 95% quantile of that drop. Present when a window length was given and fits.
   */
  worstDrop?: [number, number];
  /**
   * The worst-window coverage a method at the target would show with this row's dependence
   * between misses (`typical`), and the value it would fall below one time in twenty (`low`).
   * The engine derives it from `worstDrop`.
   */
  worstWindow?: { typical: number; low: number };
}

/** Small seeded PRNG (mulberry32) so the intervals do not flicker between renders. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile(sorted: Float64Array, p: number): number {
  if (!sorted.length) return NaN;
  const x = p * (sorted.length - 1);
  const i = Math.floor(x), f = x - i;
  return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}

/** 95% percentile interval of finite samples. */
export function interval95(samples: Float64Array): [number, number] {
  const s = samples.filter(Number.isFinite).sort();
  return [percentile(s, 0.025), percentile(s, 0.975)];
}

/**
 * Circular moving-block bootstrap over `T` arrivals, shared by all series.
 * `block` should cover the dependence range (at least the number of scored leads).
 *
 * With `window`, every replicate also records how far its worst window of that many arrivals
 * falls below its own overall coverage. A replicate keeps the dependence between misses
 * within a block but has the same coverage everywhere, so this drop is what window-to-window
 * noise alone produces, without regime changes.
 */
export function blockBootstrap(
  series: Map<string, ArrivalSeries>,
  T: number,
  block: number,
  B = 300,
  seed = 12345,
  window?: number,
): Map<string, Uncertainty> {
  const out = new Map<string, Uncertainty>();
  const L = Math.max(1, Math.round(block));
  // With fewer than two blocks every replicate is the same data in a rotated order, which
  // would report zero-width intervals. Report nothing until resampling means something.
  if (T < 2 * L) return out;
  const nBlocks = Math.ceil(T / L);
  const rand = mulberry32(seed);
  const starts = new Int32Array(B * nBlocks);
  for (let i = 0; i < starts.length; i++) starts[i] = Math.floor(rand() * T);

  const W = window !== undefined && window >= 1 && window <= T ? Math.round(window) : 0;
  // The arrivals of one replicate, in resampled order (shared by every series).
  const order = new Int32Array(T);
  const stats = [...series].map(([id, s]) => ({
    id,
    n: Float64Array.from(s.n), cov: Float64Array.from(s.cov), width: Float64Array.from(s.width), is: Float64Array.from(s.is),
    shape: s.shape ? Float64Array.from(s.shape) : undefined,
    sc: new Float64Array(B), cv: new Float64Array(B), wd: new Float64Array(B),
    sh: s.shape ? new Float64Array(B) : undefined,
    drop: W ? new Float64Array(B) : undefined,
  }));
  for (let b = 0; b < B; b++) {
    for (let j = 0, taken = 0; j < nBlocks && taken < T; j++) {
      let t = starts[b * nBlocks + j];
      for (let u = 0; u < L && taken < T; u++, taken++) {
        order[taken] = t;
        if (++t === T) t = 0;
      }
    }
    for (const s of stats) {
      let n = 0, c = 0, w = 0, is = 0, shape = 0;
      // Sums over the last W arrivals of the replicate, and the lowest coverage of a full window.
      let wn = 0, wc = 0, worst = Infinity;
      for (let i = 0; i < T; i++) {
        const t = order[i];
        const nt = s.n[t], ct = s.cov[t];
        n += nt;
        c += ct;
        w += s.width[t];
        is += s.is[t];
        if (s.shape) shape += s.shape[t];
        if (W) {
          wn += nt;
          wc += ct;
          if (i >= W) {
            const old = order[i - W];
            wn -= s.n[old];
            wc -= s.cov[old];
          }
          // Compare wc / wn with the lowest so far without dividing.
          if (i >= W - 1 && wn > 0 && wc < worst * wn) worst = wc / wn;
        }
      }
      s.sc[b] = n ? is / n : NaN;
      s.cv[b] = n ? c / n : NaN;
      s.wd[b] = n ? w / n : NaN;
      if (s.sh) s.sh[b] = n ? shape / n : NaN;
      if (s.drop) s.drop[b] = s.cv[b] - worst;
    }
  }
  for (const { id, sc, cv, wd, sh, drop } of stats) {
    const sort = (a: Float64Array) => Float64Array.from(a).sort();
    const ss = sort(sc), cs = sort(cv), ws = sort(wd);
    const u: Uncertainty = {
      coverage: [percentile(cs, 0.025), percentile(cs, 0.975)],
      width: [percentile(ws, 0.025), percentile(ws, 0.975)],
      intervalScore: [percentile(ss, 0.025), percentile(ss, 0.975)],
      scoreSamples: sc,
      coverageSamples: cv,
      widthSamples: wd,
    };
    if (sh) {
      u.shapeSamples = sh;
      u.shapeScore = interval95(sh);
    }
    if (drop) {
      const ds = drop.filter(Number.isFinite).sort();
      if (ds.length) u.worstDrop = [percentile(ds, 0.5), percentile(ds, 0.95)];
    }
    out.set(id, u);
  }
  return out;
}

/** True when the 95% interval of the paired difference `a - b` contains 0. */
export function indistinguishable(a: Float64Array, b: Float64Array): boolean {
  const d = new Float64Array(a.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] - b[i];
    if (Number.isFinite(v)) d[m++] = v;
  }
  if (m < 10) return true;
  const s = d.subarray(0, m).sort();
  return percentile(s, 0.025) <= 0 && percentile(s, 0.975) >= 0;
}
