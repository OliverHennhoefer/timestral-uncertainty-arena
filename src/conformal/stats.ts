/**
 * Small numerical helpers for the Bayesian methods: log-gamma, the regularized incomplete
 * beta function, and the Student-t distribution (pdf, cdf, quantile). Plain JS, no deps.
 */

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** log Γ(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Continued fraction for the incomplete beta function (modified Lentz). */
function betacf(a: number, b: number, x: number): number {
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function betai(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return (bt * betacf(a, b, x)) / a;
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Standard normal CDF (Abramowitz & Stegun 7.1.26 via erfc, |error| < 1.5e-7). */
export function normCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erfc = y * Math.exp(-x * x);
  return z >= 0 ? 1 - erfc / 2 : erfc / 2;
}

/** A Student-t distribution with `nu` degrees of freedom, location `mu` and scale `sigma`. */
export interface StudentT {
  nu: number;
  mu: number;
  sigma: number;
}

/** Normalizing constant of the standard t density, cached by nu. */
const LOG_NORM_CACHE = new Map<number, number>();
function tLogNorm(nu: number): number {
  if (Number.isInteger(nu) && nu <= 4096) {
    let v = LOG_NORM_CACHE.get(nu);
    if (v === undefined) LOG_NORM_CACHE.set(nu, (v = tLogNormRaw(nu)));
    return v;
  }
  return tLogNormRaw(nu);
}
function tLogNormRaw(nu: number): number {
  return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI);
}

export function tLogPdf(x: number, d: StudentT): number {
  const z = (x - d.mu) / d.sigma;
  return tLogNorm(d.nu) - Math.log(d.sigma) - ((d.nu + 1) / 2) * Math.log1p((z * z) / d.nu);
}

/**
 * Standard Student-t CDF. Integer nu <= 60 uses the exact finite series (Abramowitz & Stegun
 * 26.7.3-4), large nu a corrected normal (A&S 26.7.8, |error| ~1e-4 at nu = 60), and other nu
 * the incomplete beta function.
 */
export function tCdfStd(z: number, nu: number): number {
  if (nu > 60) return normCdf((z * (1 - 1 / (4 * nu))) / Math.sqrt(1 + (z * z) / (2 * nu)));
  if (Number.isInteger(nu)) {
    const th = Math.atan(Math.abs(z) / Math.sqrt(nu));
    const c2 = Math.cos(th) ** 2, s = Math.sin(th);
    let A: number;
    if (nu % 2 === 1) {
      let term = Math.cos(th), sum = nu > 1 ? term : 0;
      for (let k = 3; k <= nu - 2; k += 2) {
        term *= (c2 * (k - 1)) / k;
        sum += term;
      }
      A = (2 / Math.PI) * (th + s * sum);
    } else {
      let term = 1, sum = 1;
      for (let k = 2; k <= nu - 2; k += 2) {
        term *= (c2 * (k - 1)) / k;
        sum += term;
      }
      A = s * sum;
    }
    return z >= 0 ? 0.5 + A / 2 : 0.5 - A / 2;
  }
  const x = nu / (nu + z * z);
  const tail = 0.5 * betai(nu / 2, 0.5, x);
  return z >= 0 ? 1 - tail : tail;
}

/** Reference implementation via the incomplete beta function (for tests). */
export function tCdfStdBeta(z: number, nu: number): number {
  const tail = 0.5 * betai(nu / 2, 0.5, nu / (nu + z * z));
  return z >= 0 ? 1 - tail : tail;
}

export function tCdf(x: number, d: StudentT): number {
  return tCdfStd((x - d.mu) / d.sigma, d.nu);
}

/**
 * Quantile of a finite mixture of Student-t distributions at probability `p`: safeguarded
 * Newton on the mixture CDF, falling back to bisection inside a maintained bracket, so it
 * always converges. `w` must sum to 1. A good `guess` (e.g. the previous answer) saves steps.
 */
export function tMixtureQuantile(comps: readonly StudentT[], w: readonly number[], p: number, guess?: number, tol = 1e-7): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const K = comps.length;
  const logNorm = new Float64Array(K);
  let mean = 0, scale = 0, wsum = 0;
  for (let i = 0; i < K; i++) {
    if (!(w[i] > 0)) continue;
    logNorm[i] = tLogNorm(comps[i].nu) - Math.log(comps[i].sigma);
    mean += w[i] * comps[i].mu;
    scale = Math.max(scale, comps[i].sigma);
    wsum += w[i];
  }
  if (!(wsum > 0) || !(scale > 0)) return NaN;
  mean /= wsum;
  const evalAt = (x: number): [number, number] => {
    let F = 0, f = 0;
    for (let i = 0; i < K; i++) {
      if (!(w[i] > 0)) continue;
      const c = comps[i];
      const z = (x - c.mu) / c.sigma;
      F += w[i] * tCdfStd(z, c.nu);
      f += w[i] * Math.exp(logNorm[i] - ((c.nu + 1) / 2) * Math.log1p((z * z) / c.nu));
    }
    return [F / wsum, f / wsum];
  };
  // Newton from the guess (or the mixture mean). Every step narrows a bracket a < x* <= b;
  // steps that leave it bisect, and while one side is still open the step doubles outwards.
  let a = -Infinity, b = Infinity;
  let x = guess !== undefined && Number.isFinite(guess) ? guess : mean;
  let step = scale;
  for (let it = 0; it < 100; it++) {
    const [F, f] = evalAt(x);
    if (Math.abs(F - p) < tol) break;
    if (F < p) a = x;
    else b = x;
    if (b - a < 1e-10 * scale) break;
    let nx = f > 0 ? x - (F - p) / f : NaN;
    if (!(nx > a && nx < b)) {
      if (Number.isFinite(a) && Number.isFinite(b)) nx = 0.5 * (a + b);
      else {
        nx = F < p ? x + step : x - step;
        step *= 2;
      }
    }
    x = nx;
  }
  return x;
}

export function tQuantile(d: StudentT, p: number): number {
  return tMixtureQuantile([d], [1], p);
}
