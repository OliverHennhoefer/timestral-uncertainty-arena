/**
 * Moving-block bootstrap for the ranking.
 *
 * Evaluated pairs are strongly dependent: forecasts made at neighbouring origins overlap,
 * and every arrival scores up to `evalLeads` of them at once. Resampling single pairs would
 * badly understate the uncertainty, so we resample contiguous blocks of arrivals (each
 * arrival carrying all of its pairs) with the same block indices for every method. That
 * keeps the comparison paired: two methods judged on the same resampled stretch of series.
 */

/** Per-arrival sums for one method: pairs, covered pairs, width and interval score. */
export interface ArrivalSeries {
  n: ArrayLike<number>;
  cov: ArrayLike<number>;
  width: ArrayLike<number>;
  is: ArrayLike<number>;
}

export interface Uncertainty {
  /** 95% percentile intervals of the resampled statistics. */
  coverage: [number, number];
  width: [number, number];
  intervalScore: [number, number];
  /** Resampled mean interval scores, one per replicate (paired across methods). */
  scoreSamples: Float64Array;
  coverageSamples: Float64Array;
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

/**
 * Circular moving-block bootstrap over `T` arrivals, shared by all series.
 * `block` should cover the dependence range (at least the number of scored leads).
 */
export function blockBootstrap(
  series: Map<string, ArrivalSeries>,
  T: number,
  block: number,
  B = 300,
  seed = 12345,
): Map<string, Uncertainty> {
  const out = new Map<string, Uncertainty>();
  if (T < 2) return out;
  const L = Math.max(1, Math.min(T, Math.round(block)));
  const nBlocks = Math.ceil(T / L);
  const rand = mulberry32(seed);
  const starts = new Int32Array(B * nBlocks);
  for (let i = 0; i < starts.length; i++) starts[i] = Math.floor(rand() * T);

  for (const [id, s] of series) {
    const sc = new Float64Array(B), cv = new Float64Array(B), wd = new Float64Array(B);
    for (let b = 0; b < B; b++) {
      let n = 0, c = 0, w = 0, is = 0, taken = 0;
      for (let j = 0; j < nBlocks && taken < T; j++) {
        const st = starts[b * nBlocks + j];
        for (let u = 0; u < L && taken < T; u++, taken++) {
          const t = (st + u) % T;
          n += s.n[t];
          c += s.cov[t];
          w += s.width[t];
          is += s.is[t];
        }
      }
      sc[b] = n ? is / n : NaN;
      cv[b] = n ? c / n : NaN;
      wd[b] = n ? w / n : NaN;
    }
    const sort = (a: Float64Array) => Float64Array.from(a).sort();
    const ss = sort(sc), cs = sort(cv), ws = sort(wd);
    out.set(id, {
      coverage: [percentile(cs, 0.025), percentile(cs, 0.975)],
      width: [percentile(ws, 0.025), percentile(ws, 0.975)],
      intervalScore: [percentile(ss, 0.025), percentile(ss, 0.975)],
      scoreSamples: sc,
      coverageSamples: cv,
    });
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
