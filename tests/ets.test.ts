import { describe, expect, it } from "vitest";
import { TIMESTRAL_LEVELS } from "../src/model/backends";
import { EtsBackend, detectPeriod, etsMoments, fitEts, fitVariant, nelderMead } from "../src/model/ets";

/** Deterministic standard normal draws (mulberry32 + Box-Muller). */
function rng(seed: number) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}

const req = (y: number[], horizon = 16) => ({ context: Float64Array.from(y), horizon, symmetric: false, clip: false, maxContext: 4096 });

describe("Nelder-Mead", () => {
  it("finds the minimum of a shifted quadratic", () => {
    const x = nelderMead(([a, b]) => (a - 3) ** 2 + 10 * (b + 1) ** 2, [0, 0], [1, 1], 500);
    expect(x[0]).toBeCloseTo(3, 3);
    expect(x[1]).toBeCloseTo(-1, 3);
  });
});

describe("season detection", () => {
  it("finds the period of a noisy seasonal series with a trend", () => {
    const z = rng(1);
    const y = Array.from({ length: 300 }, (_, t) => 50 + 0.2 * t + 10 * Math.sin((2 * Math.PI * t) / 24) + z());
    expect(detectPeriod(y)).toBe(24);
  });

  it("finds no season in a random walk", () => {
    const z = rng(2);
    let v = 0;
    const y = Array.from({ length: 300 }, () => (v += z()));
    expect(detectPeriod(y)).toBe(0);
  });
});

describe("ETS fits", () => {
  it("SES recovers the smoothing parameter of a local level model", () => {
    const z = rng(3);
    let l = 10;
    const y: number[] = [];
    for (let t = 0; t < 2000; t++) {
      const e = z();
      y.push(l + e);
      l += 0.4 * e;
    }
    const fit = fitEts(y, { trend: "N", m: 0 })!;
    expect(fit.alpha).toBeGreaterThan(0.32);
    expect(fit.alpha).toBeLessThan(0.48);
    expect(fit.sigma2).toBeGreaterThan(0.85);
    expect(fit.sigma2).toBeLessThan(1.15);
  });

  it("variance grows as in the closed form for SES: sigma^2 (1 + (h-1) alpha^2)", () => {
    const fit = fitEts([1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8], { trend: "N", m: 0 })!;
    const { variance } = etsMoments(fit, 5);
    for (let h = 1; h <= 5; h++) expect(variance[h - 1]).toBeCloseTo(fit.sigma2 * (1 + (h - 1) * fit.alpha ** 2), 8);
  });

  it("damped trend continues a ramp and flattens out", () => {
    const y = Array.from({ length: 80 }, (_, t) => 2 * t + Math.sin(t));
    const fit = fitEts(y, { trend: "Ad", m: 0 })!;
    const { mean } = etsMoments(fit, 200);
    expect(mean[0]).toBeGreaterThan(y[79]);
    expect(mean[199] - mean[198]).toBeLessThan(mean[1] - mean[0]);
  });

  it("Holt-Winters tracks the season and auto picks a seasonal model", () => {
    const z = rng(4);
    const y = Array.from({ length: 240 }, (_, t) => 100 + 20 * Math.sin((2 * Math.PI * t) / 12) + z());
    const hw = fitVariant(y, "hw")!;
    expect(hw.spec.m).toBe(12);
    const { mean } = etsMoments(hw, 12);
    for (let h = 1; h <= 12; h++) expect(Math.abs(mean[h - 1] - (100 + 20 * Math.sin((2 * Math.PI * (240 + h - 1)) / 12)))).toBeLessThan(3);
    expect(fitVariant(y, "auto")!.spec.m).toBe(12);
  });

  it("auto picks a non-seasonal model for a random walk", () => {
    const z = rng(5);
    let v = 0;
    const y = Array.from({ length: 200 }, () => (v += z()));
    expect(fitVariant(y, "auto")!.spec.m).toBe(0);
  });

  it("tolerates missing values", () => {
    const y = Array.from({ length: 60 }, (_, t) => (t % 7 === 3 ? NaN : 5 + Math.sin(t / 3)));
    const fit = fitVariant(y, "damped")!;
    expect(Number.isFinite(fit.level)).toBe(true);
    expect(Number.isFinite(fit.sigma2)).toBe(true);
  });
});

describe("EtsBackend", () => {
  it("returns sorted quantiles on the shared 21-level grid", async () => {
    const z = rng(6);
    const y = Array.from({ length: 150 }, (_, t) => 20 + 5 * Math.sin(t / 4) + z());
    for (const v of ["auto", "ses", "damped", "hw"] as const) {
      const res = await new EtsBackend(v).forecast(req(y, 32));
      expect(res.levels).toEqual(TIMESTRAL_LEVELS);
      expect(res.quantiles.length).toBe(32 * 21);
      for (let h = 0; h < 32; h++) for (let i = 1; i < 21; i++) expect(res.quantiles[h * 21 + i]).toBeGreaterThanOrEqual(res.quantiles[h * 21 + i - 1]);
      expect(res.quantiles.every(Number.isFinite)).toBe(true);
    }
  });

  it("covers about 80% of the next value on a local level model", async () => {
    const z = rng(7);
    let l = 0;
    const y: number[] = [];
    for (let t = 0; t < 700; t++) {
      const e = z();
      y.push(l + e);
      l += 0.3 * e;
    }
    const b = new EtsBackend("ses");
    let hit = 0, tot = 0;
    for (let o = 200; o < 699; o += 1) {
      const res = await b.forecast({ ...req(y.slice(0, o + 1), 1), maxContext: 200 });
      const lo = res.quantiles[TIMESTRAL_LEVELS.indexOf(0.1)], hi = res.quantiles[TIMESTRAL_LEVELS.indexOf(0.9)];
      hit += y[o + 1] >= lo && y[o + 1] <= hi ? 1 : 0;
      tot++;
    }
    expect(hit / tot).toBeGreaterThan(0.74);
    expect(hit / tot).toBeLessThan(0.86);
  });

  it("forecasts from one or two points and a constant series", async () => {
    for (const y of [[], [5], [5, 6], Array(40).fill(3)]) {
      const res = await new EtsBackend("auto").forecast(req(y, 8));
      expect(res.quantiles.every(Number.isFinite)).toBe(true);
    }
  });

  it("clips at zero for a non-negative context", async () => {
    const y = Array.from({ length: 50 }, (_, t) => (t % 2 ? 0 : 1));
    const res = await new EtsBackend("ses").forecast({ ...req(y, 8), clip: true });
    expect(Math.min(...res.quantiles)).toBeGreaterThanOrEqual(0);
  });

  it("names the model it picked", async () => {
    const y = Array.from({ length: 240 }, (_, t) => 100 + 20 * Math.sin((2 * Math.PI * t) / 12));
    const b = new EtsBackend("hw");
    await b.forecast(req(y));
    expect(b.label).toContain("ETS(A,Ad,A) m=12");
  });
});
