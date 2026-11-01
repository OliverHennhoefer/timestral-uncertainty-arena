/**
 * Causal per-patch normalization with an asinh squash, ported from `timestral/scaler.py`.
 * Statistics are accumulated in float64 (JS numbers) exactly like the reference.
 */

const F32_EPS = 1.1920928955078125e-7;
const F32_MAX = 3.4028234663852886e38;
const F64_TINY = 2.2250738585072014e-308;

/** Per-step causal `(loc, scale)` of the patch each step belongs to, cast to float32. */
export function causalPatchLocScale(
  data: Float32Array,
  observed: Uint8Array,
  patchSize: number,
  minimumScale: number,
): { loc: Float32Array; scale: Float32Array } {
  const T = data.length;
  if (T % patchSize !== 0) throw new Error(`length ${T} is not a multiple of patch_size ${patchSize}`);
  const loc = new Float32Array(T);
  const scale = new Float32Array(T);
  let cumW = 0, cumX = 0, m2 = 0, prevMean = 0;
  for (let t = 0; t < T; t++) {
    const w = observed[t] ? 1 : 0;
    const x = data[t];
    cumW += w;
    cumX += w * x;
    const den = Math.max(cumW, 1);
    const mean = cumX / den;
    m2 += (x - prevMean) * (x - mean) * w;
    prevMean = mean;
    if ((t + 1) % patchSize === 0) {
      const variance = m2 / Math.max(den - 1, 1);
      const s = Math.max(Math.sqrt(variance), minimumScale);
      const pl = Math.fround(mean);
      const psc = Math.fround(s);
      for (let j = t + 1 - patchSize; j <= t; j++) {
        loc[j] = pl;
        scale[j] = psc;
      }
    }
  }
  return { loc, scale };
}

/** `asinh((x - loc) / scale)` on observed steps, 0 elsewhere. */
export function applyAffineAsinh(
  data: Float32Array,
  loc: Float32Array,
  scale: Float32Array,
  observed: Uint8Array,
  useAsinh: boolean,
): Float32Array {
  const out = new Float32Array(data.length);
  for (let t = 0; t < data.length; t++) {
    if (!observed[t]) continue;
    const z = Math.fround(Math.fround(data[t] - loc[t]) / Math.max(scale[t], F32_EPS));
    out[t] = useAsinh ? Math.asinh(z) : z;
  }
  return out;
}

/** Map +/-inf along each quantile vector to its nearest finite value; all non-finite becomes 0. */
export function clampNonfinite(q: Float32Array, Q: number): void {
  for (let o = 0; o < q.length; o += Q) {
    let mx = -Infinity, mn = Infinity;
    for (let i = 0; i < Q; i++) {
      const v = q[o + i];
      if (Number.isFinite(v)) {
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
    }
    for (let i = 0; i < Q; i++) {
      const v = q[o + i];
      if (v === Infinity) q[o + i] = Number.isFinite(mx) ? mx : 0;
      else if (v === -Infinity) q[o + i] = Number.isFinite(mn) ? mn : 0;
    }
  }
}

interface Welford { w: number; mean: number; m2: number }

function welfordCombine(a: Welford, b: Welford): Welford {
  const w = a.w + b.w;
  if (w === 0) return { w, mean: 0, m2: 0 };
  const wSafe = Math.max(w, F64_TINY);
  const delta = b.mean - a.mean;
  return { w, mean: a.mean + (delta * b.w) / wSafe, m2: a.m2 + b.m2 + (delta * delta * a.w * b.w) / wSafe };
}

/**
 * Per-patch `(loc, scale)` of a decoded block in the advancing refine frame
 * (`advancing_block_frames`). `sinhMedian` is `(nPred, patchSize)` row-major.
 */
export function advancingBlockFrames(
  context: Float32Array,
  observed: Uint8Array,
  sinhMedian: Float64Array,
  nPred: number,
  patchSize: number,
  minimumScale: number,
): { locs: Float64Array; scales: Float64Array } {
  const nCtx = context.length / patchSize;
  let acc: Welford = { w: 0, mean: 0, m2: 0 };
  for (let p = 0; p < nCtx; p++) {
    let cnt = 0, sx = 0;
    for (let j = 0; j < patchSize; j++) {
      const t = p * patchSize + j;
      if (observed[t]) {
        cnt += 1;
        sx += context[t];
      }
    }
    const mean = cnt === 0 ? 0 : sx / Math.max(cnt, 1);
    let m2 = 0;
    if (cnt > 0) {
      for (let j = 0; j < patchSize; j++) {
        const t = p * patchSize + j;
        if (observed[t]) m2 += (context[t] - mean) ** 2;
      }
    }
    acc = welfordCombine(acc, { w: cnt, mean, m2 });
  }
  const offset = acc.mean;
  const clip = Math.sqrt(F32_MAX) / 1e3;
  const locs = new Float64Array(nPred);
  const scales = new Float64Array(nPred);
  const folded = new Float64Array(patchSize);
  for (let h = 0; h < nPred; h++) {
    const variance = Math.max(acc.m2 / Math.max(acc.w - 1, 1), 0);
    const s = Math.max(Math.sqrt(variance), minimumScale);
    locs[h] = acc.mean;
    scales[h] = s;
    let bm = 0;
    for (let j = 0; j < patchSize; j++) {
      const v = Math.min(Math.max(sinhMedian[h * patchSize + j] * s + acc.mean - offset, -clip), clip) + offset;
      folded[j] = v;
      bm += v;
    }
    bm /= patchSize;
    let bm2 = 0;
    for (let j = 0; j < patchSize; j++) bm2 += (folded[j] - bm) ** 2;
    acc = welfordCombine(acc, { w: patchSize, mean: bm, m2: bm2 });
  }
  return { locs, scales };
}
