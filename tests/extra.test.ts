import { describe, expect, it } from "vitest";
import { METHODS, defaultParams } from "../src/conformal/methods";
import { BOCPD, BayesianBelief, DiscountedBayes, SAOCP } from "../src/conformal/extra";
import { tCdfStd, tCdfStdBeta, tMixtureQuantile } from "../src/conformal/stats";
import { ScoreHistory } from "../src/conformal/window";
import { Engine, defaultSettings } from "../src/engine/engine";
import { NaiveBackend, normInv } from "../src/model/backends";
import { PRESETS } from "../src/synthetic";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s + 0.5) / 2 ** 32;
  };
}

/** Feed `scores` with feedback delayed by `delay` steps; returns coverage after `burn`. */
function runDelayed(spec: (typeof METHODS)[number], scores: number[], delay: number, alpha = 0.1, burn = 100) {
  const h = new ScoreHistory(100);
  const m = spec.create(h, alpha, defaultParams(spec));
  let covered = 0, n = 0;
  const qs: number[] = [];
  for (let t = 0; t < scores.length; t++) {
    const k = t - delay;
    if (k >= 0) {
      m.update(k, scores[k]);
      h.push(scores[k]);
    }
    qs.push(h.count ? h.cap(m.predict(t)) : Infinity);
  }
  for (let t = burn; t < scores.length; t++) {
    n++;
    if (scores[t] <= qs[t]) covered++;
  }
  return { coverage: covered / n, qs };
}

describe("Student-t helpers", () => {
  it("finite series and normal approximation match the incomplete beta", () => {
    for (const nu of [1, 2, 3, 6, 17, 60, 61, 250]) {
      for (const z of [-8, -2, -0.4, 0, 0.7, 1.9, 5]) expect(Math.abs(tCdfStd(z, nu) - tCdfStdBeta(z, nu))).toBeLessThan(5e-5);
    }
  });

  it("quantiles of single and mixed t distributions", () => {
    expect(tMixtureQuantile([{ nu: 5, mu: 0, sigma: 1 }], [1], 0.95)).toBeCloseTo(2.015, 3);
    expect(tMixtureQuantile([{ nu: 1, mu: 2, sigma: 3 }], [1], 0.9)).toBeCloseTo(2 + 3 * 3.0777, 3);
    // Two well separated equal components: the median sits between them.
    const q = tMixtureQuantile([{ nu: 30, mu: -10, sigma: 1 }, { nu: 30, mu: 10, sigma: 1 }], [0.5, 0.5], 0.5);
    expect(Math.abs(q)).toBeLessThan(5);
  });
});

describe("Bayesian methods", () => {
  it("discounted Bayes is calibrated on Gaussian scores without discounting", () => {
    const r = rng(1);
    const scores = Array.from({ length: 3000 }, () => 2 + 0.5 * normInv(r()));
    const h = new ScoreHistory(100);
    const m = new DiscountedBayes(h, 0.1, 1);
    let c = 0, n = 0;
    scores.forEach((s, t) => {
      const q = m.predict();
      if (t > 50) { n++; if (s <= q) c++; }
      m.update(t, s);
      h.push(s);
    });
    expect(Math.abs(c / n - 0.9)).toBeLessThan(0.015);
  });

  it("BOCPD drops its run length after a level shift and keeps a normalized posterior", () => {
    const r = rng(2);
    const h = new ScoreHistory(100);
    const b = new BOCPD(h, 100);
    for (let t = 0; t < 300; t++) {
      const s = (t < 200 ? 0 : 8) + normInv(r());
      b.observe(s);
      h.push(s);
      if (t === 199) expect(b.expectedRun).toBeGreaterThan(50);
    }
    expect(b.expectedRun).toBeLessThan(110);
    expect(b.expectedRun).toBeGreaterThan(80);
    const total = b.runs.reduce((a, x) => a + x.w, 0);
    expect(total).toBeCloseTo(1, 10);
    const surv = b.survival(300);
    expect(surv[0]).toBeCloseTo(1, 10);
    for (let a = 1; a < 300; a++) expect(surv[a]).toBeLessThanOrEqual(surv[a - 1] + 1e-12);
  });

  it("Bayesian belief stays inside the score range and is mildly conservative", () => {
    const r = rng(3);
    const m = new BayesianBelief(0.1, 0.99);
    const scores = Array.from({ length: 2000 }, () => normInv(r()) - 1);
    let c = 0, n = 0;
    scores.forEach((s, t) => {
      const q = m.predict();
      if (t > 0) expect(Number.isFinite(q)).toBe(true);
      if (t > 100) { n++; if (s <= q) c++; }
      m.update(t, s);
    });
    const cov = c / n;
    expect(cov).toBeGreaterThan(0.9);
    expect(cov).toBeLessThan(0.96);
    expect(m.predict()).toBeLessThanOrEqual(Math.max(...scores));
  });
});

describe("SAOCP", () => {
  it("keeps O(g log t) live experts", () => {
    const h = new ScoreHistory(100);
    const m = new SAOCP(h, 0.1, 8);
    const r = rng(4);
    for (let t = 0; t < 4096; t++) {
      const s = normInv(r());
      m.predict();
      m.update(t, s);
      h.push(s);
    }
    // Lifetimes g 2^u: at most about g (log2 t + 1) experts are alive at once.
    expect((m as unknown as { experts: unknown[] }).experts.length).toBeLessThanOrEqual(8 * 13 + 1);
  });
});

describe("delayed feedback on autocorrelated, signed scores", () => {
  // AR(1) scores, mostly negative like CQR scores, with a volatility jump: overlapping
  // multi-step forecasts produce exactly this kind of stream.
  const r = rng(7);
  let e = 0;
  const scores = Array.from({ length: 3000 }, (_, t) => {
    e = 0.9 * e + Math.sqrt(1 - 0.81) * normInv(r());
    return e * (t < 1500 ? 1 : 2.5) - 1.3;
  });

  it("every method issues usable thresholds", () => {
    for (const spec of METHODS) {
      const { qs } = runDelayed(spec, scores, 8);
      expect(qs.slice(20).every((q) => Number.isFinite(q)), spec.id).toBe(true);
    }
  });

  it("error-feedback methods keep coverage despite an 8-step delay", () => {
    for (const id of ["bocpd_aci", "sfogd"]) {
      const spec = METHODS.find((m) => m.id === id)!;
      expect(Math.abs(runDelayed(spec, scores, 8).coverage - 0.9), id).toBeLessThan(0.02);
    }
  });
});

describe("engine with every method", () => {
  it("runs all methods on CQR scores and replays deterministically", async () => {
    const naive = new NaiveBackend();
    const enabled = Object.fromEntries(METHODS.map((m) => [m.id, true]));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4, enabled });
    const y = PRESETS.find((p) => p.id === "volshift")!.make(250, 5);
    for (let t = 0; t < y.length; t++) {
      e.append(y[t]);
      const fc = await naive.forecast({ context: Float64Array.from(e.series), horizon: 8, symmetric: false, clip: false, maxContext: 256 });
      e.addForecast(t, fc.quantiles, fc.horizon, fc.levels);
    }
    e.catchUp();
    const before = e.table();
    for (const row of before) {
      expect(row.n).toBeGreaterThan(500);
      expect(Number.isFinite(row.intervalScore)).toBe(true);
    }
    e.replay();
    expect(e.table()).toEqual(before);
  });
});

describe("BOCPD changepoint mass", () => {
  it("responds to the observation that starts a new regime", () => {
    const r = rng(9);
    const h = new ScoreHistory(100);
    const b = new BOCPD(h, 100);
    for (let t = 0; t < 100; t++) {
      const s = 0.1 * normInv(r());
      b.observe(s);
      h.push(s);
    }
    // A wildly unlikely score: almost all mass should move to the fresh regime at once,
    // not the fixed hazard 1/100.
    b.observe(50);
    const fresh = b.runs.find((x) => x.len === 1)!;
    expect(fresh.w).toBeGreaterThan(0.9);
  });

  it("keeps the changepoint mass near the hazard for typical scores", () => {
    const r = rng(10);
    const h = new ScoreHistory(100);
    const b = new BOCPD(h, 100);
    for (let t = 0; t < 200; t++) {
      const s = normInv(r());
      b.observe(s);
      h.push(s);
    }
    b.observe(0);
    const fresh = b.runs.find((x) => x.len === 1);
    expect(fresh?.w ?? 0).toBeLessThan(0.02);
  });
});
