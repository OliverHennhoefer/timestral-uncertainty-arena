/**
 * Classical exponential smoothing (ETS) forecasters that run in the browser, with no weights.
 *
 * All models have additive errors, so the forecast distribution is Gaussian with the closed-form
 * variance of Hyndman et al. (2008, ch. 6, "class 1"). Each forecast refits the model to the
 * context window: the smoothing parameters and the initial level and trend are chosen by
 * Nelder-Mead on the in-sample squared one-step errors; the initial seasonal states come from a
 * classical decomposition of the context. Candidate season lengths come from the autocorrelation
 * of the differenced and of the detrended context, and AICc picks among them (or none), so a
 * hand-drawn series needs no frequency.
 *
 * Variants: simple exponential smoothing ETS(A,N,N), damped trend ETS(A,Ad,N), Holt-Winters
 * ETS(A,Ad,A) (the lowest AICc over the candidate seasons, or ETS(A,Ad,N) when no season beats
 * it) and "Auto", which picks the lowest AICc among ANN, AAN, AAdN and, for each candidate
 * season, ANA, AAA and AAdA (as `ets()` in R's forecast package does for additive errors).
 */

import { type Backend, type ForecastRequest, type ForecastResult, TIMESTRAL_LEVELS, clipNonnegative, normInv } from "./backends";

export type EtsVariant = "auto" | "ses" | "damped" | "hw";

export const ETS_VARIANTS: Record<EtsVariant, { name: string; label: string }> = {
  auto: { name: "Auto ETS", label: "Auto ETS (picked by AICc)" },
  ses: { name: "SES", label: "Simple exponential smoothing" },
  damped: { name: "Damped trend", label: "Damped trend (Holt)" },
  hw: { name: "Holt-Winters", label: "Holt-Winters seasonal" },
};

export const isEtsVariant = (v: unknown): v is EtsVariant => typeof v === "string" && v in ETS_VARIANTS;

export type Trend = "N" | "A" | "Ad";

export interface EtsSpec {
  trend: Trend;
  /** Season length; 0 for no season. */
  m: number;
}

export interface EtsFit {
  spec: EtsSpec;
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
  /** States after the last observation. */
  level: number;
  slope: number;
  /** Seasonal states, indexed by time mod m. */
  season: Float64Array;
  /** Observations the model was fitted to (`n` is the context length, used for the season index). */
  n: number;
  sigma2: number;
  aicc: number;
}

export const specName = (s: EtsSpec): string => `ETS(A,${s.trend},${s.m ? "A" : "N"})${s.m ? ` m=${s.m}` : ""}`;

const PHI_MIN = 0.8;
const PHI_MAX = 0.98;
/** Longest season searched; keeps the autocorrelation scan cheap on long contexts. */
const MAX_PERIOD = 400;

const sigmoid = (u: number) => 1 / (1 + Math.exp(-u));
const logit = (p: number) => Math.log(p / (1 - p));

/**
 * Shortest lag with a clear autocorrelation peak in `d`, searched after the autocorrelation
 * first turns negative (so the slow decay at short lags is not mistaken for a peak) and taken
 * as the shortest peak within 80% of the highest (so harmonics 2m, 3m, ... are not picked);
 * 0 if there is none.
 */
function acfPeak(d: number[], maxPeriod: number): number {
  const n = d.length;
  const maxLag = Math.min(maxPeriod, Math.floor((n + 1) / 2));
  if (maxLag < 3) return 0;
  let mean = 0;
  for (const v of d) mean += v;
  mean /= n;
  let c0 = 0;
  for (const v of d) c0 += (v - mean) ** 2;
  if (!(c0 > 0)) return 0;
  const acf = new Float64Array(maxLag + 2);
  for (let k = 1; k <= Math.min(maxLag + 1, n - 1); k++) {
    let s = 0;
    for (let t = k; t < n; t++) s += (d[t] - mean) * (d[t - k] - mean);
    acf[k] = s / c0;
  }
  const threshold = Math.max(0.3, 3 / Math.sqrt(n));
  const peaks: number[] = [];
  let best = 0;
  let start = 1;
  while (start <= maxLag && acf[start] >= 0) start++;
  for (let k = Math.max(2, start + 1); k <= maxLag; k++) {
    if (acf[k] > threshold && acf[k] > acf[k - 1] && acf[k] >= acf[k + 1]) {
      peaks.push(k);
      best = Math.max(best, acf[k]);
    }
  }
  return peaks.find((k) => acf[k] >= 0.8 * best) ?? 0;
}

/**
 * Season length from the autocorrelation of the first differences. Differencing removes trends
 * and random-walk wander, which would otherwise look like a long season, but it also amplifies
 * noise: a smooth season with noise of a similar size per step can disappear (see
 * `seasonCandidates`). 0 if there is none.
 */
export function detectPeriod(y: ArrayLike<number>, maxPeriod = MAX_PERIOD): number {
  const d: number[] = [];
  for (let t = 1; t < y.length; t++) if (Number.isFinite(y[t]) && Number.isFinite(y[t - 1])) d.push(y[t] - y[t - 1]);
  return acfPeak(d, maxPeriod);
}

/**
 * Season length from the autocorrelation of the levels after removing a least-squares line.
 * This keeps smooth seasons that differencing drowns in noise, but wander can produce
 * spurious peaks, so its answer is only a candidate for model selection. 0 if there is none.
 */
export function detectPeriodLevels(y: ArrayLike<number>, maxPeriod = MAX_PERIOD): number {
  let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
  for (let t = 0; t < y.length; t++) {
    if (!Number.isFinite(y[t])) continue;
    n++;
    st += t;
    sy += y[t];
    stt += t * t;
    sty += t * y[t];
  }
  if (n < 6) return 0;
  const den = n * stt - st * st;
  const b = den > 0 ? (n * sty - st * sy) / den : 0;
  const a = (sy - b * st) / n;
  const d: number[] = [];
  for (let t = 0; t < y.length; t++) if (Number.isFinite(y[t])) d.push(y[t] - a - b * t);
  return acfPeak(d, maxPeriod);
}

/**
 * Candidate season lengths: from the differences and from the detrended levels (distinct,
 * non-zero). Which one, if any, is used is left to AICc, as in seasonal-period selection by
 * information criterion, so a spurious candidate costs a fit but not a worse forecast.
 */
export function seasonCandidates(y: ArrayLike<number>, maxPeriod = MAX_PERIOD): number[] {
  const out: number[] = [];
  for (const m of [detectPeriod(y, maxPeriod), detectPeriodLevels(y, maxPeriod)]) if (m > 1 && !out.includes(m)) out.push(m);
  return out;
}

interface Params {
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
  l0: number;
  b0: number;
}

interface RunOut {
  sse: number;
  count: number;
  level: number;
  slope: number;
  season: Float64Array;
}

/** One pass of the error-correction recursions; missing values keep the states on their forecast. */
function run(y: ArrayLike<number>, spec: EtsSpec, p: Params, s0: Float64Array): RunOut {
  const { m } = spec;
  const trend = spec.trend !== "N";
  const phi = spec.trend === "Ad" ? p.phi : 1;
  let l = p.l0, b = trend ? p.b0 : 0;
  const s = s0.slice();
  let sse = 0, count = 0;
  for (let t = 0; t < y.length; t++) {
    const st = m ? s[t % m] : 0;
    const yt = y[t];
    const e = Number.isFinite(yt) ? yt - (l + phi * b + st) : 0;
    if (Number.isFinite(yt)) {
      sse += e * e;
      count++;
    }
    const lNew = l + phi * b + p.alpha * e;
    if (trend) b = phi * b + p.beta * e;
    if (m) s[t % m] = st + p.gamma * e;
    l = lNew;
  }
  return { sse, count, level: l, slope: b, season: s };
}

/** Starting level, slope and seasonal states from the first observations (or cycles). */
function initialStates(y: number[], spec: EtsSpec): { l0: number; b0: number; s0: Float64Array } {
  const { m } = spec;
  const trend = spec.trend !== "N";
  if (m) {
    // Classical decomposition over the whole context: a centred moving average (2 x m for an
    // even m) for the trend, then the mean detrended value at each phase for the season.
    const n = y.length;
    const half = Math.floor(m / 2);
    const sum = new Array(m).fill(0), cnt = new Array(m).fill(0);
    for (let t = half; t + half < n; t++) {
      let ma = 0;
      if (m % 2) for (let i = -half; i <= half; i++) ma += y[t + i];
      else {
        for (let i = -half + 1; i < half; i++) ma += y[t + i];
        ma += 0.5 * (y[t - half] + y[t + half]);
      }
      ma /= m;
      const v = y[t] - ma;
      if (!Number.isFinite(v)) continue;
      sum[t % m] += v;
      cnt[t % m]++;
    }
    const s0 = new Float64Array(m);
    for (let i = 0; i < m; i++) s0[i] = cnt[i] ? sum[i] / cnt[i] : 0;
    let sm = 0;
    for (const v of s0) sm += v;
    for (let i = 0; i < m; i++) s0[i] -= sm / m;
    // Level and slope from a line through the deseasonalised first two cycles.
    const k = 2 * m;
    let sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
    for (let t = 0; t < k; t++) {
      const v = y[t] - s0[t % m];
      if (!Number.isFinite(v)) continue;
      sx += t;
      sy += v;
      sxx += t * t;
      sxy += t * v;
      c++;
    }
    const slope = trend && c > 1 ? (c * sxy - sx * sy) / (c * sxx - sx * sx) : 0;
    const icept = c ? (sy - slope * sx) / c : 0;
    return { l0: icept - slope, b0: slope, s0 };
  }
  const k = Math.min(y.length, 10);
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let t = 0; t < k; t++) {
    sx += t;
    sy += y[t];
    sxx += t * t;
    sxy += t * y[t];
  }
  const slope = k > 1 ? (k * sxy - sx * sy) / (k * sxx - sx * sx) : 0;
  const icept = (sy - slope * sx) / k;
  return trend ? { l0: icept - slope, b0: slope, s0: new Float64Array(0) } : { l0: k ? y[0] : 0, b0: 0, s0: new Float64Array(0) };
}

/** Number of estimated parameters, including the initial states and the error variance. */
function numParams(spec: EtsSpec): number {
  let k = 1 + 1 + 1; // alpha, l0, sigma
  if (spec.trend !== "N") k += 2; // beta, b0
  if (spec.trend === "Ad") k += 1; // phi
  if (spec.m) k += 1 + (spec.m - 1); // gamma, seasonal states
  return k;
}

/** Nelder-Mead minimisation (standard coefficients). */
export function nelderMead(f: (x: number[]) => number, x0: number[], step: number[], maxIter = 400, tol = 1e-9): number[] {
  const n = x0.length;
  const pts: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    p[i] += step[i];
    pts.push(p);
  }
  let vals = pts.map(f);
  for (let it = 0; it < maxIter; it++) {
    const order = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
    const P = order.map((i) => pts[i]);
    const V = order.map((i) => vals[i]);
    pts.splice(0, pts.length, ...P);
    vals = V;
    if (Math.abs(vals[n] - vals[0]) <= tol * (Math.abs(vals[0]) + tol)) break;
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += pts[i][j] / n;
    const along = (t: number) => c.map((cj, j) => cj + t * (pts[n][j] - cj));
    const xr = along(-1), fr = f(xr);
    if (fr < vals[0]) {
      const xe = along(-2), fe = f(xe);
      if (fe < fr) [pts[n], vals[n]] = [xe, fe];
      else [pts[n], vals[n]] = [xr, fr];
    } else if (fr < vals[n - 1]) {
      [pts[n], vals[n]] = [xr, fr];
    } else {
      const xc = fr < vals[n] ? along(-0.5) : along(0.5);
      const fc = f(xc);
      if (fc < Math.min(fr, vals[n])) [pts[n], vals[n]] = [xc, fc];
      else {
        for (let i = 1; i <= n; i++) {
          pts[i] = pts[i].map((v, j) => pts[0][j] + 0.5 * (v - pts[0][j]));
          vals[i] = f(pts[i]);
        }
      }
    }
  }
  let bi = 0;
  for (let i = 1; i <= n; i++) if (vals[i] < vals[bi]) bi = i;
  return pts[bi];
}

/** Fit one ETS model to `y` (missing values allowed). Returns null when the context is too short. */
export function fitEts(y: number[], spec: EtsSpec): EtsFit | null {
  const finite = y.filter(Number.isFinite);
  const n = finite.length;
  const k = numParams(spec);
  if (n < k + 3 || (spec.m && y.length < 2 * spec.m)) return null;
  const trend = spec.trend !== "N";
  const damped = spec.trend === "Ad";
  let mean = 0;
  for (const v of finite) mean += v;
  mean /= n;
  let sd = 0;
  for (const v of finite) sd += (v - mean) ** 2;
  sd = Math.sqrt(sd / n) || Math.abs(mean) * 0.01 || 1;
  const init = initialStates(spec.m ? y : finite, spec);
  const sLevel = sd * 0.1, sSlope = sd * 0.01;

  // Unconstrained coordinates: alpha in (0, 1), beta in (0, alpha), gamma in (0, 1 - alpha),
  // phi in [0.8, 0.98], and offsets of the initial level and slope in units of the data spread.
  const decode = (u: number[]): Params => {
    let i = 0;
    const alpha = 1e-4 + (1 - 2e-4) * sigmoid(u[i++]);
    const beta = trend ? alpha * sigmoid(u[i++]) : 0;
    const gamma = spec.m ? (1 - alpha) * sigmoid(u[i++]) : 0;
    const phi = damped ? PHI_MIN + (PHI_MAX - PHI_MIN) * sigmoid(u[i++]) : 1;
    const l0 = init.l0 + sLevel * u[i++];
    const b0 = trend ? init.b0 + sSlope * u[i++] : 0;
    return { alpha, beta, gamma, phi, l0, b0 };
  };
  const u0: number[] = [logit(0.3)];
  if (trend) u0.push(logit(0.1));
  if (spec.m) u0.push(logit(0.1 / 0.7));
  if (damped) u0.push(logit((0.95 - PHI_MIN) / (PHI_MAX - PHI_MIN)));
  u0.push(0);
  if (trend) u0.push(0);
  const objective = (u: number[]) => run(y, spec, decode(u), init.s0).sse;
  const u = nelderMead(objective, u0, u0.map(() => 1), 120 * u0.length);
  const p = decode(u);
  const out = run(y, spec, p, init.s0);
  // A tiny floor keeps a perfectly fitted (e.g. constant) series from a zero or -inf score.
  const sse = out.sse + 1e-12 * sd * sd * n;
  const dof = Math.max(1, out.count - k);
  const logLik = -0.5 * out.count * (Math.log((2 * Math.PI * sse) / out.count) + 1);
  const aicc = -2 * logLik + 2 * k + (2 * k * (k + 1)) / Math.max(1, out.count - k - 1);
  return {
    spec,
    alpha: p.alpha,
    beta: p.beta,
    gamma: p.gamma,
    phi: p.phi,
    level: out.level,
    slope: out.slope,
    season: out.season,
    n: y.length,
    sigma2: sse / dof,
    aicc,
  };
}

/** Candidate models for a variant, given the candidate season lengths (may be empty). */
export function candidates(variant: EtsVariant, periods: number[]): EtsSpec[] {
  switch (variant) {
    case "ses":
      return [{ trend: "N", m: 0 }];
    case "damped":
      return [{ trend: "Ad", m: 0 }];
    case "hw":
      // Without a season that pays for itself, Holt-Winters reduces to the damped trend model.
      return [...periods.map((m): EtsSpec => ({ trend: "Ad", m })), { trend: "Ad", m: 0 }];
    case "auto": {
      const specs: EtsSpec[] = [{ trend: "N", m: 0 }, { trend: "A", m: 0 }, { trend: "Ad", m: 0 }];
      for (const m of periods) specs.push({ trend: "N", m }, { trend: "A", m }, { trend: "Ad", m });
      return specs;
    }
  }
}

/**
 * Fit the variant to `y`: the lowest AICc among its candidates that can be fitted (a short
 * context falls back to simpler models). For Holt-Winters and "auto" this also chooses the
 * season length, or none.
 */
export function fitVariant(y: number[], variant: EtsVariant): EtsFit | null {
  const periods = variant === "hw" || variant === "auto" ? seasonCandidates(y) : [];
  let best: EtsFit | null = null;
  for (const spec of candidates(variant, periods)) {
    const fit = fitEts(y, spec);
    if (fit && (!best || fit.aicc < best.aicc)) best = fit;
  }
  if (best || variant === "ses") return best;
  return fitEts(y, { trend: "N", m: 0 });
}

/** Point forecast and variance at leads 1..H (Hyndman et al. 2008, Table 6.1, class 1). */
export function etsMoments(fit: EtsFit, horizon: number): { mean: Float64Array; variance: Float64Array } {
  const { alpha, beta, gamma, spec } = fit;
  const phi = spec.trend === "Ad" ? fit.phi : 1;
  const mean = new Float64Array(horizon);
  const variance = new Float64Array(horizon);
  let phiH = 0; // phi + phi^2 + ... + phi^h
  let pw = 1;
  let acc = 0; // sum of c_j^2 for j < h
  for (let h = 1; h <= horizon; h++) {
    pw *= phi;
    phiH += pw;
    const s = spec.m ? fit.season[(fit.n + h - 1) % spec.m] : 0;
    mean[h - 1] = fit.level + (spec.trend === "N" ? 0 : phiH * fit.slope) + s;
    variance[h - 1] = fit.sigma2 * (1 + acc);
    // c_h = alpha + beta * (phi + ... + phi^h) + gamma * [h is a multiple of m]
    const c = alpha + (spec.trend === "N" ? 0 : beta * phiH) + (spec.m && h % spec.m === 0 ? gamma : 0);
    acc += c * c;
  }
  return { mean, variance };
}

/** `(H, Q)` Gaussian quantiles on `levels` for a fitted model. */
export function etsQuantiles(fit: EtsFit, horizon: number, levels: number[] = TIMESTRAL_LEVELS): Float32Array {
  const { mean, variance } = etsMoments(fit, horizon);
  const z = levels.map(normInv);
  const Q = levels.length;
  const q = new Float32Array(horizon * Q);
  for (let h = 0; h < horizon; h++) {
    const sd = Math.sqrt(variance[h]);
    for (let i = 0; i < Q; i++) q[h * Q + i] = mean[h] + sd * z[i];
  }
  return q;
}

export class EtsBackend implements Backend {
  readonly kind = "classical" as const;
  readonly ready = true;
  readonly modelName: string;
  /** The model the latest forecast used, e.g. "ETS(A,Ad,A) m=24". */
  lastSpec = "";

  constructor(readonly variant: EtsVariant) {
    this.modelName = ETS_VARIANTS[variant].name;
  }

  get label(): string {
    return `${ETS_VARIANTS[this.variant].label}${this.lastSpec ? ` · ${this.lastSpec}` : ""}`;
  }

  async forecast(req: ForecastRequest): Promise<ForecastResult> {
    const t0 = performance.now();
    const y = Array.from(req.context.subarray(Math.max(0, req.context.length - req.maxContext)));
    const levels = TIMESTRAL_LEVELS;
    const fit = fitVariant(y, this.variant);
    let q: Float32Array;
    if (fit) {
      this.lastSpec = specName(fit.spec);
      q = etsQuantiles(fit, req.horizon, levels);
    } else {
      // Too few points to fit anything: hold the last value with a random-walk spread.
      this.lastSpec = "";
      const obs = y.filter(Number.isFinite);
      const last = obs.length ? obs[obs.length - 1] : 0;
      let s = 0;
      for (let t = 1; t < obs.length; t++) s += (obs[t] - obs[t - 1]) ** 2;
      const sd = obs.length > 1 && s > 0 ? Math.sqrt(s / (obs.length - 1)) : Math.abs(last) * 0.05 || 1;
      q = new Float32Array(req.horizon * levels.length);
      for (let h = 0; h < req.horizon; h++) for (let i = 0; i < levels.length; i++) q[h * levels.length + i] = last + sd * Math.sqrt(h + 1) * normInv(levels[i]);
    }
    if (req.clip) clipNonnegative(q, y);
    return { quantiles: q, levels, horizon: req.horizon, ms: performance.now() - t0 };
  }

  dispose() {}
}
