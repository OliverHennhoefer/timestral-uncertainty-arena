/**
 * Quantile-level methods: instead of a scalar threshold on a score, they recalibrate the
 * model's whole quantile vector for one forecast lead. `predict(key, base)` is called when a
 * forecast is made and returns the calibrated quantiles at the same levels; `update(key, y)`
 * arrives `lead` steps later with the truth. The engine reads the interval off the returned
 * vector at alpha / 2 and 1 - alpha / 2, so these methods ignore the score type and window.
 */

export interface QuantileMethod {
  predict(key: number, base: Float64Array): Float64Array;
  update(key: number, y: number): void;
}

/**
 * Euclidean projection onto the cone x_1 <= ... <= x_m (isotonic regression) by the
 * pool-adjacent-violators algorithm, O(m). Ties are allowed.
 */
export function isotonic(x: ArrayLike<number>): Float64Array {
  const m = x.length;
  const mean = new Float64Array(m), size = new Int32Array(m);
  let blocks = 0;
  for (let i = 0; i < m; i++) {
    mean[blocks] = x[i];
    size[blocks] = 1;
    blocks++;
    while (blocks > 1 && mean[blocks - 2] > mean[blocks - 1]) {
      const n = size[blocks - 2] + size[blocks - 1];
      mean[blocks - 2] = (mean[blocks - 2] * size[blocks - 2] + mean[blocks - 1] * size[blocks - 1]) / n;
      size[blocks - 2] = n;
      blocks--;
    }
  }
  const out = new Float64Array(m);
  for (let b = 0, i = 0; b < blocks; b++) for (let c = 0; c < size[b]; c++) out[i++] = mean[b];
  return out;
}

/** Quantile of `v` at `p` with linear interpolation between order statistics. */
function linearQuantile(v: number[], p: number): number {
  const s = [...v].sort((a, b) => a - b);
  const h = (s.length - 1) * p;
  const i = Math.floor(h);
  return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[s.length - 1];
}

/**
 * Multi-level Quantile Tracker (Ding, Gibbs & Tibshirani, arXiv:2512.23671), Procedure 1 with
 * the delayed-feedback update of eq. (6), one instance per lead (delay = lead - 1).
 *
 * A hidden offset per level follows online gradient descent on the pinball loss,
 *   theta~ <- theta~ - eta (1{y <= q} - level),
 * where q is the forecast that was actually played: q = Pi_K(base + theta~), the Euclidean
 * projection of the shifted base quantiles onto ordered vectors. The played quantiles
 * therefore never cross, while the hidden offsets move freely (lazy gradient descent).
 *
 * Step size as in the paper's experiments (Section 5): eta_t = max(lr * Q_0.9, floor), with
 * Q_0.9 the 90% quantile of |y_s - base_s| pooled over all levels and the last 50 resolved
 * forecasts, lr = 0.1 and floor = 0.1. The paper does not say what to do before 50 forecasts
 * have resolved or which quantile estimator to use; here the window is whatever has resolved
 * so far (eta = floor while empty) and the quantile is linearly interpolated. The paper's
 * guarantees are stated for a fixed eta.
 */
export class MultiQT implements QuantileMethod {
  private theta: Float64Array;
  private pending = new Map<number, { base: Float64Array; played: Float64Array }>();
  /** |y - base| per level for the last `window` resolved forecasts. */
  private residuals: Float64Array[] = [];

  constructor(
    private levels: readonly number[],
    private lr = 0.1,
    private window = 50,
    private floor = 0.1,
  ) {
    this.theta = new Float64Array(levels.length);
  }

  /** Hidden offsets (for tests and diagnostics). */
  get offsets(): Float64Array {
    return this.theta;
  }

  private eta(): number {
    if (!this.residuals.length) return this.floor;
    const pooled: number[] = [];
    for (const r of this.residuals) for (const v of r) pooled.push(v);
    return Math.max(this.lr * linearQuantile(pooled, 0.9), this.floor);
  }

  predict(key: number, base: Float64Array): Float64Array {
    const shifted = new Float64Array(base.length);
    for (let i = 0; i < base.length; i++) shifted[i] = base[i] + this.theta[i];
    const played = isotonic(shifted);
    this.pending.set(key, { base, played });
    return played;
  }

  update(key: number, y: number): void {
    const p = this.pending.get(key);
    if (!p) return;
    this.pending.delete(key);
    const eta = this.eta();
    const resid = new Float64Array(p.base.length);
    for (let i = 0; i < p.base.length; i++) {
      this.theta[i] -= eta * ((y <= p.played[i] ? 1 : 0) - this.levels[i]);
      resid[i] = Math.abs(y - p.base[i]);
    }
    this.residuals.push(resid);
    if (this.residuals.length > this.window) this.residuals.shift();
  }
}
