/** Score history for one forecast lead, shared by every method at that lead. */
export class ScoreHistory {
  /** All scores in arrival order. */
  readonly all: number[] = [];
  /** The last `window` scores, sorted ascending. */
  private sorted: number[] = [];
  private maxAbs = 0;

  constructor(readonly window: number) {}

  get count(): number {
    return this.all.length;
  }

  /** Sorted view of the calibration window. Do not mutate. */
  get windowSorted(): readonly number[] {
    return this.sorted;
  }

  push(s: number): void {
    this.all.push(s);
    insertSorted(this.sorted, s);
    if (this.sorted.length > this.window) {
      const old = this.all[this.all.length - 1 - this.window];
      removeSorted(this.sorted, old);
    }
    this.maxAbs = Math.max(this.maxAbs, Math.abs(s));
  }

  /** Largest |score| in the last `n` scores (scale for learning rates). */
  recentRange(n: number): number {
    let m = 0;
    for (let i = Math.max(0, this.all.length - n); i < this.all.length; i++) m = Math.max(m, Math.abs(this.all[i]));
    return m;
  }

  get maxAbsScore(): number {
    return this.maxAbs;
  }

  /**
   * The threshold the engine actually issues: clipped to +/- the largest |score| seen
   * (which also makes +/-inf finite). Methods compare feedback against this value so
   * their adaptation agrees with the reported coverage. Unchanged before any score.
   */
  cap(q: number): number {
    if (this.all.length === 0) return q;
    if (Number.isNaN(q)) return this.maxAbs;
    return Math.max(-this.maxAbs, Math.min(this.maxAbs, q));
  }

  /**
   * Conformal quantile of the window at level `beta`: the ceil((n + 1) beta)-th smallest
   * score, `+inf` when that rank exceeds n and `-inf` when beta <= 0.
   */
  conformalQuantile(beta: number): number {
    return conformalQuantile(this.sorted, beta);
  }
}

export function conformalQuantile(sorted: readonly number[], beta: number): number {
  const n = sorted.length;
  if (beta <= 0) return -Infinity;
  if (n === 0) return Infinity;
  const rank = Math.ceil((n + 1) * beta - 1e-12);
  if (rank > n) return Infinity;
  return sorted[Math.max(rank, 1) - 1];
}

export function insertSorted(a: number[], v: number): void {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  a.splice(lo, 0, v);
}

export function removeSorted(a: number[], v: number): void {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  if (lo < a.length && a[lo] === v) a.splice(lo, 1);
}

/** Pinball (quantile) loss of predicting `q` for target `y` at level `tau`. */
export function pinball(y: number, q: number, tau: number): number {
  const d = y - q;
  return d >= 0 ? tau * d : (tau - 1) * d;
}
