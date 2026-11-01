/**
 * Further online interval methods: Bayesian models of the score stream, and recent online
 * conformal updates. All of them implement the same `OnlineMethod` contract as the methods in
 * `methods.ts`: a scalar threshold on one lead's score stream at level 1 - alpha, issued at
 * `predict(key)` and learned from `update(key, score)` once the truth arrives (delayed feedback).
 * Scores can be negative (CQR), so nothing here assumes s >= 0.
 */

import type { OnlineMethod } from "./methods";
import { ScoreHistory, pinball } from "./window";
import { type StudentT, tLogPdf, tMixtureQuantile } from "./stats";

// ---------------------------------------------------------------------------------------
// Bayesian: conjugate Normal-Inverse-Gamma models of the score

/** Weighted sufficient statistics of a Gaussian: total weight, mean, sum of squared deviations. */
interface Suff {
  n: number;
  mean: number;
  ss: number;
}

/** Add `x` with unit weight after discounting the old weight by `delta` (power prior). */
function suffAdd(st: Suff, x: number, delta = 1): Suff {
  const n0 = delta * st.n;
  const n = n0 + 1;
  const d = x - st.mean;
  return { n, mean: st.mean + d / n, ss: delta * st.ss + (n0 * d * d) / n };
}

/**
 * Posterior predictive of a Normal-Inverse-Gamma model, NIG(m0, k0, a0, b0), updated with
 * weighted statistics `st`. Student-t with nu = 2a, loc = m, scale^2 = b (k + 1) / (a k).
 */
function nigPredictive(prior: { m: number; k: number; a: number; b: number }, st: Suff): StudentT {
  const k = prior.k + st.n;
  const m = (prior.k * prior.m + st.n * st.mean) / k;
  const a = prior.a + st.n / 2;
  const b = prior.b + 0.5 * st.ss + (prior.k * st.n * (st.mean - prior.m) ** 2) / (2 * k);
  return { nu: 2 * a, mu: m, sigma: Math.sqrt((b * (k + 1)) / (a * k)) };
}

/**
 * A weak, scale-aware prior from the recent scores (empirical Bayes): centred on their mean,
 * worth `k0` pseudo-observations for the mean and nu0 = 2 a0 for the variance.
 */
function empiricalPrior(h: ScoreHistory, k0 = 0.5, a0 = 1): { m: number; k: number; a: number; b: number } {
  const xs = h.all;
  const from = Math.max(0, xs.length - h.window);
  let n = 0, mean = 0, m2 = 0;
  for (let i = from; i < xs.length; i++) {
    n++;
    const d = xs[i] - mean;
    mean += d / n;
    m2 += d * (xs[i] - mean);
  }
  const v = n > 1 ? m2 / (n - 1) : n === 1 ? Math.max(xs[from] ** 2, 1e-12) : 1;
  return { m: mean, k: k0, a: a0, b: a0 * Math.max(v, 1e-12) };
}

/**
 * Discounted conjugate Bayes (West & Harrison discount learning). The score is modelled as
 * Gaussian with unknown mean and variance; old evidence is down-weighted by `delta` per step,
 * so the posterior keeps about 1 / (1 - delta) effective observations and tracks drifting
 * levels and volatility. The threshold is the (1 - alpha) quantile of the Student-t posterior
 * predictive. Purely model-based: no coverage guarantee if the score is not Gaussian.
 */
export class DiscountedBayes implements OnlineMethod {
  private st: Suff = { n: 0, mean: 0, ss: 0 };
  constructor(private h: ScoreHistory, private alpha: number, private delta: number) {}
  predict(): number {
    if (this.h.count < 2) return Infinity;
    // Near-reference prior: negligible weight on the mean, half a pseudo-observation of variance.
    const e = empiricalPrior(this.h);
    const prior = { m: e.m, k: 1e-6, a: 0.5, b: 1e-6 * e.b };
    return tMixtureQuantile([nigPredictive(prior, this.st)], [1], 1 - this.alpha);
  }
  update(_key: number, s: number): void {
    this.st = suffAdd(this.st, s, this.delta);
  }
}

type NIGPrior = { m: number; k: number; a: number; b: number };

/** One run-length hypothesis: its posterior weight, statistics, prior and cached predictive. */
interface Run {
  len: number;
  w: number;
  st: Suff;
  prior: NIGPrior;
  pred: StudentT;
}

/**
 * Bayesian online changepoint detection (Adams & MacKay 2007) with a Normal-Inverse-Gamma
 * model per regime. Keeps the posterior over the current run length r (constant hazard
 * 1 / lambda), and each run length's sufficient statistics. A fresh regime starts from the
 * empirical prior of the recent scores. The run-length posterior is pruned to `maxRuns`.
 */
export class BOCPD {
  /** Run-length hypotheses, most recent changepoint first. */
  runs: Run[] = [];
  private hazard: number;
  constructor(private h: ScoreHistory, lambda: number, private maxRuns = 32, private minW = 1e-4) {
    this.hazard = 1 / Math.max(lambda, 1);
  }

  /** Mixture predictive for the next score: components and weights. */
  predictive(): { comps: StudentT[]; w: number[] } {
    const comps: StudentT[] = [], w: number[] = [];
    for (const r of this.runs) {
      comps.push(r.pred);
      w.push(r.w);
    }
    return { comps, w };
  }

  quantile(p: number): number {
    if (this.runs.length === 0) return Infinity;
    const { comps, w } = this.predictive();
    const q = tMixtureQuantile(comps, w, p, this.lastQ);
    if (Number.isFinite(q)) this.lastQ = q;
    return q;
  }
  private lastQ: number | undefined;

  /** P(r_t > a): probability that the score `a` steps back (0 = newest) is in the current regime. */
  survival(maxAge: number): Float64Array {
    // out[a] = sum of w over runs with len > a, as suffix sums.
    const out = new Float64Array(maxAge);
    for (const r of this.runs) if (r.len > 0) out[Math.min(r.len, maxAge) - 1] += r.w;
    for (let a = maxAge - 2; a >= 0; a--) out[a] += out[a + 1];
    return out;
  }

  /** Expected run length under the posterior. */
  get expectedRun(): number {
    let e = 0;
    for (const r of this.runs) e += r.w * r.len;
    return e;
  }

  observe(x: number): void {
    const H = this.hazard;
    if (this.runs.length === 0) {
      this.runs = [this.fresh(x, 1)];
      return;
    }
    // Joint probability of x and each hypothesis, in log space for stability. Growth: the
    // run continues (1 - H) and x comes from its predictive. Changepoint: a new regime starts
    // just before x (total mass H, as the old weights sum to 1), so x is scored under the
    // prior predictive, and the new regime is then conditioned on x.
    const runs = this.runs;
    const prior = empiricalPrior(this.h);
    const logCp = Math.log(H) + tLogPdf(x, nigPredictive(prior, { n: 0, mean: 0, ss: 0 }));
    let mx = logCp;
    for (const r of runs) {
      r.w = Math.log(r.w) + Math.log1p(-H) + tLogPdf(x, r.pred);
      if (r.w > mx) mx = r.w;
    }
    let total = 0;
    for (const r of runs) {
      r.w = Math.exp(r.w - mx);
      total += r.w;
      r.len++;
      r.st = suffAdd(r.st, x);
      r.pred = nigPredictive(r.prior, r.st);
    }
    const cp = Math.exp(logCp - mx);
    total += cp;
    runs.unshift(this.fresh(x, cp, prior));
    let kept = 0, keptW = 0;
    for (const r of runs) {
      r.w /= total;
      if (r.w >= this.minW) {
        runs[kept++] = r;
        keptW += r.w;
      }
    }
    runs.length = kept;
    while (runs.length > this.maxRuns) {
      // Drop the least probable hypothesis (rarely more than one per step).
      let k = 0;
      for (let i = 1; i < runs.length; i++) if (runs[i].w < runs[k].w) k = i;
      keptW -= runs[k].w;
      runs.splice(k, 1);
    }
    for (const r of runs) r.w /= keptW;
  }

  private fresh(x: number, w: number, prior: NIGPrior = empiricalPrior(this.h)): Run {
    const st = suffAdd({ n: 0, mean: 0, ss: 0 }, x);
    return { len: 1, w, st, prior, pred: nigPredictive(prior, st) };
  }
}

/** BOCPD predictive quantile as the threshold (pure Bayesian, no coverage guarantee). */
export class BOCPDQuantile implements OnlineMethod {
  private b: BOCPD;
  constructor(h: ScoreHistory, private alpha: number, lambda: number) {
    this.b = new BOCPD(h, lambda);
  }
  predict(): number {
    return this.b.quantile(1 - this.alpha);
  }
  update(_key: number, s: number): void {
    this.b.observe(s);
  }
}

/**
 * Conformalized BOCPD: ACI (Gibbs & Candès 2021) on the level fed to the BOCPD predictive,
 * alpha_t <- alpha_t + gamma (alpha - err_t). The Bayesian model supplies the shape and the
 * changepoint resets, and ACI's long-run coverage guarantee holds for any sequence because
 * the threshold is +inf (capped) when alpha_t <= 0 and -inf when alpha_t >= 1.
 */
export class BOCPDACI implements OnlineMethod {
  private b: BOCPD;
  private alphaT: number;
  private pending = new Map<number, number>();
  constructor(private h: ScoreHistory, private alpha: number, lambda: number, private gamma: number) {
    this.b = new BOCPD(h, lambda);
    this.alphaT = alpha;
  }
  predict(key: number): number {
    const q = this.h.cap(this.b.quantile(1 - this.alphaT));
    this.pending.set(key, q);
    return q;
  }
  update(key: number, s: number): void {
    const q = this.pending.get(key);
    if (q !== undefined) {
      this.pending.delete(key);
      this.alphaT += this.gamma * (this.alpha - (s > q ? 1 : 0));
    }
    this.b.observe(s);
  }
}

/** Scores in ascending order with their arrival index; keeps only the newest `maxLen`. */
class SortedScores {
  readonly items: { s: number; i: number }[] = [];
  n = 0;
  constructor(private maxLen: number) {}
  get newest(): number {
    return this.n - 1;
  }
  push(s: number): void {
    const items = this.items;
    let a = 0, b = items.length;
    while (a < b) {
      const mid = (a + b) >> 1;
      if (items[mid].s < s) a = mid + 1;
      else b = mid;
    }
    items.splice(a, 0, { s, i: this.n++ });
    if (items.length > this.maxLen) {
      const oldest = this.n - 1 - this.maxLen;
      const idx = items.findIndex((e) => e.i === oldest);
      if (idx >= 0) items.splice(idx, 1);
    }
  }
}

/** Table of r^0 .. r^(n-1). */
function powers(r: number, n: number): Float64Array {
  const out = new Float64Array(n);
  let v = 1;
  for (let i = 0; i < n; i++, v *= r) out[i] = v;
  return out;
}

/**
 * Changepoint-weighted split conformal: the weighted conformal quantile of Barber et al.
 * (2023), where the weight of a past score is the BOCPD posterior probability that it belongs
 * to the current regime, P(r_t > age), floored at `floor * rho^age` so a regime of length 1
 * still has a fallback. The test point carries unit weight at +inf. The weights are
 * data-dependent, so NexCP's coverage-gap bound does not apply verbatim.
 */
export class ChangepointCP implements OnlineMethod {
  private b: BOCPD;
  private scores: SortedScores;
  private decay: Float64Array;
  constructor(h: ScoreHistory, private alpha: number, lambda: number, maxLen = 500, private floor = 0.02, rho = 0.99) {
    this.b = new BOCPD(h, lambda);
    this.scores = new SortedScores(maxLen);
    this.decay = powers(rho, maxLen + 1);
  }
  private weight(surv: Float64Array, age: number): number {
    return Math.max(surv[age], this.floor * this.decay[age]);
  }
  predict(): number {
    const items = this.scores.items;
    const n = items.length;
    if (n === 0) return Infinity;
    const surv = this.b.survival(n);
    const newest = this.scores.newest;
    let total = 1; // the test point, at +inf
    for (const e of items) total += this.weight(surv, newest - e.i);
    let cum = 0;
    for (const e of items) {
      cum += this.weight(surv, newest - e.i);
      if (cum / total >= 1 - this.alpha) return e.s;
    }
    return Infinity;
  }
  update(_key: number, s: number): void {
    this.b.observe(s);
    this.scores.push(s);
  }
}

/**
 * Online conformal prediction with a regularized (Bayesian) belief, Zhang, Bombara & Yang,
 * "The Benefit of Being Bayesian in Online Conformal Prediction" (2024). The belief is a
 * mixture of a prior P0 and the discounted empirical distribution of past scores,
 * P_t = lambda P0 + (1 - lambda) Pbar_beta, and the threshold is its (1 - alpha) quantile
 * (a Dirichlet-process-style posterior mean; equivalently FTRL on the pinball loss with the
 * prior as regularizer). Uses the discounted variant with the paper's constant
 * lambda = sqrt(1 - beta) / (beta + sqrt(1 - beta)). P0 is uniform on the range of scores
 * seen (the paper uses [0, R]; scores here can be negative). It ignores past coverage
 * errors, so one belief answers every alpha at once and never returns +/-inf.
 */
export class BayesianBelief implements OnlineMethod {
  private scores: SortedScores;
  private lo = Infinity;
  private hi = -Infinity;
  private lambda: number;
  private decay: Float64Array;
  constructor(private alpha: number, private beta: number) {
    this.lambda = Math.sqrt(1 - beta) / (beta + Math.sqrt(1 - beta));
    // Beyond this age a score's weight is < 1e-6 of the newest; its mass goes to the prior.
    const maxLen = Math.min(5000, Math.ceil(Math.log(1e-6) / Math.log(Math.max(beta, 1e-9))) + 1);
    this.scores = new SortedScores(maxLen);
    this.decay = powers(beta, maxLen + 1);
  }
  predict(): number {
    const items = this.scores.items;
    if (items.length === 0) return Infinity;
    const newest = this.scores.newest;
    const c = (1 - this.lambda) * (1 - this.beta);
    let emp = 0;
    for (const e of items) emp += c * this.decay[newest - e.i];
    // Prior mass: lambda, plus the discounted initial prior and the scores dropped as too old.
    const p0 = Math.max(0, 1 - emp);
    const { lo, hi } = this;
    const range = hi - lo;
    const u = (x: number) => (range > 0 ? Math.min(1, Math.max(0, (x - lo) / range)) : x >= lo ? 1 : 0);
    const target = 1 - this.alpha;
    let cum = 0;
    for (const e of items) {
      // Below this score only the prior's uniform density adds mass.
      if (cum + p0 * u(e.s) >= target) return range > 0 && p0 > 0 ? lo + range * Math.min(1, (target - cum) / p0) : e.s;
      cum += c * this.decay[newest - e.i];
      if (cum + p0 * u(e.s) >= target) return e.s;
    }
    return range > 0 && p0 > 0 ? lo + range * Math.min(1, Math.max(0, (target - cum) / p0)) : hi;
  }
  update(_key: number, s: number): void {
    this.lo = Math.min(this.lo, s);
    this.hi = Math.max(this.hi, s);
    this.scores.push(s);
  }
}

// ---------------------------------------------------------------------------------------
// Online conformal: gradient updates on the pinball loss

/** Common start for threshold-tracking methods: the window's conformal quantile. */
function initialThreshold(h: ScoreHistory, alpha: number): number | null {
  if (h.count === 0) return null;
  const q0 = h.conformalQuantile(1 - alpha);
  return Number.isFinite(q0) ? q0 : h.maxAbsScore;
}

/** Range (max - min) of the last `n` scores: the paper's adaptive learning-rate scale. */
function spread(h: ScoreHistory, n: number): number {
  const xs = h.all;
  let lo = Infinity, hi = -Infinity;
  for (let i = Math.max(0, xs.length - n); i < xs.length; i++) {
    lo = Math.min(lo, xs[i]);
    hi = Math.max(hi, xs[i]);
  }
  return hi > lo ? hi - lo : 0;
}

/**
 * Error-quantified Conformal Inference (Wu et al., ICLR 2025). The quantile tracker's
 * update plus an "error quantification" term that also uses how far the score fell from the
 * threshold: q <- q + eta [err - alpha + (s - q) f'(s - q)], with f a sigmoid of slope c.
 * The EQ term is largest for moderate errors (|s - q| ~ 1.5 / c) and vanishes for both tiny and
 * huge ones. Its average is not zero, so long-run coverage is only guaranteed up to a bias
 * that grows with the size of the EQ term (Wu et al., Thm. 2); with the scores far more
 * often below q than above it, the bias is towards under-coverage. Both eta and the sigmoid
 * act in units of the range S of the last 100 scores: eta = lr S (the paper's adaptive rate)
 * and f(x) = sigmoid(c x / S), so the method does not depend on the units of the series.
 * With cutoff h > 0 the EQ term is applied only when |s - q| > h S (ECI-cutoff).
 */
export class ECI implements OnlineMethod {
  private q: number | null = null;
  private pending = new Map<number, number>();
  constructor(private h: ScoreHistory, private alpha: number, private lr: number, private c: number, private cutoff: number, private scaleWindow = 100) {}
  predict(key: number): number {
    if (this.q === null) this.q = initialThreshold(this.h, this.alpha);
    if (this.q === null) return Infinity;
    const q = this.h.cap(this.q);
    this.pending.set(key, q);
    return q;
  }
  update(key: number, s: number): void {
    const q = this.pending.get(key);
    if (q === undefined) return;
    this.pending.delete(key);
    if (this.q === null) return;
    const S = spread(this.h, this.scaleWindow) || Math.abs(s) || 1;
    const err = s > q ? 1 : 0;
    const u = (s - q) / S;
    let eq = 0;
    if (Math.abs(u) > this.cutoff) {
      const f = 1 / (1 + Math.exp(-this.c * u));
      eq = u * this.c * f * (1 - f);
    }
    this.q += this.lr * S * (err - this.alpha + eq);
  }
}

/**
 * Scale-free online gradient descent on the pinball loss (Orabona & Pál 2016), the base
 * learner of Bhatnagar et al. (ICML 2023): q <- q - D g / sqrt(sum g^2), with g the pinball
 * gradient. The step shrinks as gradients accumulate, so it settles on a quantile without a
 * tuned learning rate. D is the range of the recent scores (the reference uses the max |residual|
 * on a calibration set, which is the same thing for absolute residuals).
 */
export class SFOGD implements OnlineMethod {
  private q: number | null = null;
  private g2 = 0;
  private pending = new Map<number, number>();
  constructor(private h: ScoreHistory, private alpha: number, private scaleWindow = 100) {}
  predict(key: number): number {
    if (this.q === null) this.q = initialThreshold(this.h, this.alpha);
    if (this.q === null) return Infinity;
    const q = this.h.cap(this.q);
    this.pending.set(key, q);
    return q;
  }
  update(key: number, s: number): void {
    const q = this.pending.get(key);
    if (q === undefined) return;
    this.pending.delete(key);
    if (this.q === null) return;
    const g = (s > q ? 1 : 0) - this.alpha; // minus the pinball gradient
    this.g2 += g * g;
    const D = spread(this.h, this.scaleWindow) || Math.abs(s) || 1;
    this.q += (D / Math.sqrt(this.g2)) * g;
  }
}

/** One SF-OGD expert of SAOCP with a finite lifetime and coin-betting meta statistics. */
interface Expert {
  start: number;
  lifetime: number;
  q: number;
  g2: number;
  /** Sum of meta "gradients", sum of weight * gradient, and age. */
  z: number;
  wz: number;
  age: number;
}

/**
 * Strongly Adaptive Online Conformal Prediction (Bhatnagar, Wang, Xiong & Bai, ICML 2023).
 * Starts a new SF-OGD expert every step at the current prediction, each living
 * g * 2^(number of trailing zeros of its start time) steps (geometric covering intervals),
 * and combines the live experts with coin betting for changing environments (Jun et al. 2017),
 * which gives low regret on every interval rather than only overall. Follows the reference
 * implementation (salesforce/online_conformal) except that scores may be negative.
 */
export class SAOCP implements OnlineMethod {
  private experts: Expert[] = [];
  private t = 1;
  constructor(private h: ScoreHistory, private alpha: number, private g: number, private scaleWindow = 100) {}

  private weight(e: Expert): number {
    return e.age === 0 ? 0 : (e.z / e.age) * (1 + e.wz);
  }

  private meta(): number {
    if (this.experts.length === 0) return initialThreshold(this.h, this.alpha) ?? 0;
    let zp = 0, zw = 0, num = 0, numPrior = 0;
    for (const e of this.experts) {
      const prior = 1 / (e.start * e.start * (1 + Math.floor(Math.log2(e.start))));
      const p = prior * Math.max(0, this.weight(e));
      zp += prior;
      zw += p;
      num += p * e.q;
      numPrior += prior * e.q;
    }
    return zw > 0 ? num / zw : numPrior / zp;
  }

  predict(): number {
    if (this.h.count === 0) return Infinity;
    return this.meta();
  }

  update(_key: number, s: number): void {
    const D = spread(this.h, this.scaleWindow) || Math.abs(s) || 1;
    const tau = 1 - this.alpha;
    const sHat = this.meta();
    this.experts = this.experts.filter((e) => e.age <= e.lifetime);
    let u = 0, t = this.t;
    while (t % 2 === 0) {
      t /= 2;
      u++;
    }
    this.experts.push({ start: this.t, lifetime: this.g * 2 ** u, q: sHat, g2: 0, z: 0, wz: 0, age: 0 });
    const metaLoss = pinball(s, this.meta(), tau);
    const norm = Math.sqrt(3) * D * Math.max(this.alpha, tau); // the reference's scale is sqrt(3) max|r|
    for (const e of this.experts) {
      const w = this.weight(e);
      const gm = Math.min(1, Math.max(w > 0 ? -1 : 0, (metaLoss - pinball(s, e.q, tau)) / norm));
      e.z += gm;
      e.wz += gm * w;
      e.age++;
      const grad = s > e.q ? -tau : 1 - tau;
      e.g2 += grad * grad;
      e.q -= (D / Math.sqrt(e.g2)) * grad;
    }
    this.t++;
  }
}
