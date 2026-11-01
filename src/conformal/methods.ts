/**
 * Online / adaptive conformal methods. Every method calibrates a scalar threshold `q` on a
 * stream of nonconformity scores for one forecast lead, with delayed feedback: `predict(key)`
 * issues a threshold when a forecast is made, `update(key, score)` arrives `lead` steps later
 * once the truth is known. The engine turns `q` into an interval through the score function.
 */

import { ScoreHistory, pinball } from "./window";

export interface OnlineMethod {
  predict(key: number): number;
  update(key: number, score: number): void;
}

export interface ParamSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  help?: string;
}

export interface MethodSpec {
  id: string;
  name: string;
  color: string;
  reference: string;
  summary: string;
  params: ParamSpec[];
  create(history: ScoreHistory, alpha: number, params: Record<string, number>): OnlineMethod;
}

/** Rolling split conformal: empirical quantile of the last W scores. */
class RollingSplit implements OnlineMethod {
  constructor(private h: ScoreHistory, private alpha: number) {}
  predict(): number {
    return this.h.conformalQuantile(1 - this.alpha);
  }
  update(): void {}
}

/** Adaptive Conformal Inference (Gibbs & Candès, 2021). */
class ACI implements OnlineMethod {
  private alphaT: number;
  private pending = new Map<number, number>();
  constructor(private h: ScoreHistory, private alpha: number, private gamma: number) {
    this.alphaT = alpha;
  }
  predict(key: number): number {
    const q = this.h.cap(this.h.conformalQuantile(1 - this.alphaT));
    this.pending.set(key, q);
    return q;
  }
  update(key: number, s: number): void {
    const q = this.pending.get(key);
    if (q === undefined) return;
    this.pending.delete(key);
    const err = s > q ? 1 : 0;
    this.alphaT += this.gamma * (this.alpha - err);
  }
}

const DEFAULT_GAMMAS = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128];

/**
 * Dynamically-tuned ACI (Gibbs & Candès, 2024, JMLR "Conformal inference for online prediction
 * with arbitrary distribution shifts"). Runs ACI experts over a grid of step sizes and
 * reweights them with exponential weights on the pinball loss of `beta_t`. This uses the
 * weighted-average variant of the expert combination (deterministic) instead of sampling.
 */
class DtACI implements OnlineMethod {
  private alphas: number[];
  private weights: number[];
  private pending = new Map<number, number[]>();
  private eta: number;
  private sigma: number;
  constructor(private h: ScoreHistory, private alpha: number, private gammas: number[], interval: number) {
    const k = gammas.length;
    this.alphas = gammas.map(() => alpha);
    this.weights = gammas.map(() => 1 / k);
    this.sigma = 1 / (2 * interval);
    this.eta = Math.sqrt(3 / interval) * Math.sqrt((Math.log(k * interval) + 2) / ((1 - alpha) ** 2 * alpha ** 3));
  }
  predict(key: number): number {
    const qs = this.alphas.map((a) => this.h.cap(this.h.conformalQuantile(1 - a)));
    this.pending.set(key, qs);
    let a = 0;
    for (let i = 0; i < this.alphas.length; i++) a += this.weights[i] * this.alphas[i];
    return this.h.cap(this.h.conformalQuantile(1 - a));
  }
  update(key: number, s: number): void {
    const qs = this.pending.get(key);
    if (!qs) return;
    this.pending.delete(key);
    // beta_t = sup{beta : s <= Q_{1-beta}} on the current calibration window.
    const sorted = this.h.windowSorted;
    let below = 0;
    while (below < sorted.length && sorted[below] < s) below++;
    const beta = 1 - below / (sorted.length + 1);
    let total = 0;
    const k = this.alphas.length;
    for (let i = 0; i < k; i++) {
      const loss = pinball(beta, this.alphas[i], this.alpha);
      this.weights[i] *= Math.exp(-this.eta * loss);
      total += this.weights[i];
    }
    for (let i = 0; i < k; i++) {
      this.weights[i] = total > 0 && Number.isFinite(total) ? (1 - this.sigma) * (this.weights[i] / total) + this.sigma / k : 1 / k;
      const err = s > qs[i] ? 1 : 0;
      this.alphas[i] += this.gammas[i] * (this.alpha - err);
    }
  }
}

/**
 * Aggregated ACI (Zaffran et al., ICML 2022). ACI experts over a grid of step sizes,
 * combined by an online expert-aggregation rule on the pinball loss of the threshold.
 * This uses exponentially weighted aggregation (EWA) with a scale-free learning rate,
 * on the single symmetric threshold rather than separate lower/upper bounds.
 */
class AgACI implements OnlineMethod {
  private alphas: number[];
  private logW: number[];
  private pending = new Map<number, number[]>();
  private t = 0;
  constructor(private h: ScoreHistory, private alpha: number, private gammas: number[]) {
    this.alphas = gammas.map(() => alpha);
    this.logW = gammas.map(() => 0);
  }
  private expertQ(a: number): number {
    const q = this.h.conformalQuantile(1 - a);
    const cap = this.h.maxAbsScore;
    return Number.isFinite(q) ? q : q > 0 ? cap : -cap;
  }
  predict(key: number): number {
    const qs = this.alphas.map((a) => this.expertQ(a));
    this.pending.set(key, qs);
    const mx = Math.max(...this.logW);
    let num = 0, den = 0;
    for (let i = 0; i < qs.length; i++) {
      const w = Math.exp(this.logW[i] - mx);
      num += w * qs[i];
      den += w;
    }
    return this.h.count === 0 ? Infinity : num / den;
  }
  update(key: number, s: number): void {
    const qs = this.pending.get(key);
    if (!qs) return;
    this.pending.delete(key);
    this.t++;
    const scale = Math.max(this.h.recentRange(200), 1e-12);
    const eta = Math.sqrt((8 * Math.log(qs.length)) / this.t) / scale;
    for (let i = 0; i < qs.length; i++) {
      this.logW[i] -= eta * pinball(s, qs[i], 1 - this.alpha);
      const err = s > qs[i] ? 1 : 0;
      this.alphas[i] += this.gammas[i] * (this.alpha - err);
    }
  }
}

/**
 * Quantile tracking, the "P" controller of Conformal PID (Angelopoulos, Candès & Tibshirani,
 * NeurIPS 2023): online gradient descent on the pinball loss, `q += eta * (err - alpha)`, with
 * the scale-free step `eta = lr * max |score| over the recent window`.
 */
class QuantileTracker implements OnlineMethod {
  protected q: number | null = null;
  protected pending = new Map<number, number>();
  protected errSum = 0;
  protected t = 0;
  constructor(protected h: ScoreHistory, protected alpha: number, protected lr: number, protected scaleWindow: number) {}
  protected base(): number {
    if (this.q === null) {
      if (this.h.count === 0) return Infinity;
      const q0 = this.h.conformalQuantile(1 - this.alpha);
      this.q = Number.isFinite(q0) ? q0 : this.h.maxAbsScore;
    }
    return this.q;
  }
  predict(key: number): number {
    const q = this.h.cap(this.base());
    this.pending.set(key, q);
    return q;
  }
  update(key: number, s: number): void {
    const q = this.pending.get(key);
    if (q === undefined) return;
    this.pending.delete(key);
    const err = s > q ? 1 : 0;
    this.t++;
    this.errSum += err - this.alpha;
    if (this.q === null) return;
    const scale = this.h.recentRange(this.scaleWindow) || Math.abs(s);
    this.q += this.lr * scale * (err - this.alpha);
  }
}

/**
 * Conformal PID control without a scorecaster: the quantile tracker (P) plus the saturated
 * error integrator `r_t(sum (err_i - alpha)) = K_I * B * tan(x * log(t+1) / (C_sat * (t+1)))`.
 */
class ConformalPID extends QuantileTracker {
  private csat: number;
  constructor(h: ScoreHistory, alpha: number, lr: number, scaleWindow: number, private ki: number, horizonGuess: number) {
    super(h, alpha, lr, scaleWindow);
    const lt = Math.log(horizonGuess);
    this.csat = (2 / Math.PI) * (Math.ceil(lt * 0.01) - 1 / lt);
  }
  predict(key: number): number {
    const q = this.base();
    let integral = 0;
    if (Number.isFinite(q) && this.ki > 0 && this.t > 0) {
      const scale = this.h.recentRange(this.scaleWindow);
      let arg = (this.errSum * Math.log(this.t + 1)) / (this.csat * (this.t + 1));
      const lim = (Math.PI / 2) * 0.999;
      arg = Math.max(-lim, Math.min(lim, arg));
      integral = this.ki * scale * Math.tan(arg);
    }
    const out = this.h.cap(q + integral);
    this.pending.set(key, out);
    return out;
  }
}

/**
 * Non-exchangeable split conformal (Barber, Candès, Ramdas & Tibshirani, Ann. Stat. 2023):
 * weighted quantile with geometric weights `rho^age`, plus the test point's unit mass at +inf.
 */
class NexCP implements OnlineMethod {
  private sorted: { s: number; i: number }[] = [];
  private n = 0;
  constructor(private alpha: number, private rho: number, private maxLen: number) {}
  predict(): number {
    if (this.sorted.length === 0) return Infinity;
    const newest = this.n - 1;
    let total = 1; // the test point
    for (const e of this.sorted) total += Math.pow(this.rho, newest - e.i);
    let cum = 0;
    for (const e of this.sorted) {
      cum += Math.pow(this.rho, newest - e.i);
      if (cum / total >= 1 - this.alpha) return e.s;
    }
    return Infinity;
  }
  update(_key: number, s: number): void {
    const entry = { s, i: this.n++ };
    let lo = 0, hi = this.sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.sorted[mid].s < s) lo = mid + 1;
      else hi = mid;
    }
    this.sorted.splice(lo, 0, entry);
    if (this.sorted.length > this.maxLen) {
      const oldest = this.n - 1 - this.maxLen;
      const idx = this.sorted.findIndex((e) => e.i === oldest);
      if (idx >= 0) this.sorted.splice(idx, 1);
    }
  }
}

export const METHODS: MethodSpec[] = [
  {
    id: "split",
    name: "Rolling split CP",
    color: "#007AFF",
    reference: "Vovk et al. 2005; rolling calibration window",
    summary: "Empirical (1-α)(1+1/n) quantile of the last W scores. Valid under exchangeability, slow to react to shifts.",
    params: [],
    create: (h, alpha) => new RollingSplit(h, alpha),
  },
  {
    id: "aci",
    name: "ACI",
    color: "#FF9500",
    reference: "Gibbs & Candès, NeurIPS 2021",
    summary: "Adapts the working miscoverage level α_t ← α_t + γ(α − err_t). Long-run coverage holds for any sequence.",
    params: [{ key: "gamma", label: "γ step", min: 0.001, max: 0.2, step: 0.001, default: 0.01 }],
    create: (h, alpha, p) => new ACI(h, alpha, p.gamma),
  },
  {
    id: "dtaci",
    name: "DtACI",
    color: "#A2845E",
    reference: "Gibbs & Candès, JMLR 2024",
    summary: "Runs ACI over a grid of γ and reweights experts online, so the step size tunes itself.",
    params: [{ key: "interval", label: "Interval I", min: 10, max: 1000, step: 10, default: 100, help: "Local-regret interval that sets η and σ." }],
    create: (h, alpha, p) => new DtACI(h, alpha, DEFAULT_GAMMAS, p.interval),
  },
  {
    id: "agaci",
    name: "AgACI",
    color: "#30B0C7",
    reference: "Zaffran et al., ICML 2022",
    summary: "Aggregates ACI experts with online expert aggregation on the pinball loss (EWA variant).",
    params: [],
    create: (h, alpha) => new AgACI(h, alpha, DEFAULT_GAMMAS),
  },
  {
    id: "qt",
    name: "Quantile tracker",
    color: "#34C759",
    reference: "Angelopoulos, Candès & Tibshirani, NeurIPS 2023",
    summary: "Online gradient descent on the pinball loss of the threshold itself: q ← q + η(err − α).",
    params: [{ key: "lr", label: "Learning rate", min: 0.001, max: 0.5, step: 0.001, default: 0.05, help: "Step as a fraction of the recent score range." }],
    create: (h, alpha, p) => new QuantileTracker(h, alpha, p.lr, 100),
  },
  {
    id: "pid",
    name: "Conformal PID",
    color: "#5856D6",
    reference: "Angelopoulos, Candès & Tibshirani, NeurIPS 2023",
    summary: "Quantile tracker plus a saturated integrator of past coverage errors (no scorecaster).",
    params: [
      { key: "lr", label: "Learning rate", min: 0.001, max: 0.5, step: 0.001, default: 0.05 },
      { key: "ki", label: "K_I", min: 0, max: 5, step: 0.05, default: 0.5, help: "Integrator gain, in units of the recent score range." },
    ],
    create: (h, alpha, p) => new ConformalPID(h, alpha, p.lr, 100, p.ki, 1000),
  },
  {
    id: "nexcp",
    name: "NexCP",
    color: "#AF52DE",
    reference: "Barber, Candès, Ramdas & Tibshirani, Ann. Stat. 2023",
    summary: "Weighted split conformal with geometric weights ρ^age, robust to drift.",
    params: [{ key: "rho", label: "Decay ρ", min: 0.8, max: 0.9999, step: 0.0001, default: 0.99 }],
    create: (_h, alpha, p) => new NexCP(alpha, p.rho, 500),
  },
];

export function defaultParams(spec: MethodSpec): Record<string, number> {
  return Object.fromEntries(spec.params.map((p) => [p.key, p.default]));
}

