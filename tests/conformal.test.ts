import { describe, expect, it } from "vitest";
import { METHODS, createSolo, defaultParams } from "../src/conformal/methods";
import { ScoreHistory, aciErr, conformalQuantile } from "../src/conformal/window";
import { Engine, type MetricsRow, NATIVE_ID, bestMargin, defaultSettings, quantileAt, rankRows, shapeScore, shiftedScore, widthAtTarget, worstWindowCoverage } from "../src/engine/engine";
import { NaiveBackend, TIMESTRAL_LEVELS, normInv } from "../src/model/backends";
import { PRESETS } from "../src/synthetic";
import { type Uncertainty, blockBootstrap, indistinguishable, mulberry32 } from "../src/engine/bootstrap";

/** A metrics row with only the fields the ranking reads. */
function row(id: string, coverage: number, width: number, intervalScore: number, shape = intervalScore): MetricsRow {
  return {
    id, n: 10, coverage, rollingCoverage: coverage, meanWidth: width, rollingWidth: width, intervalScore,
    shapeScore: shape, levelCost: intervalScore - shape, widthAtTarget: width, worstCoverage: coverage,
  };
}

/** Bootstrap output with only the coverage interval the ranking's coverage test reads. */
function coverageIntervals(ci: Record<string, [number, number]>): Map<string, Uncertainty> {
  const empty = new Float64Array(0);
  return new Map(Object.entries(ci).map(([id, coverage]) => [id, { coverage, width: [0, 0], intervalScore: [0, 0], scoreSamples: empty, coverageSamples: empty }]));
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s + 0.5) / 2 ** 32;
  };
}

describe("conformal quantile", () => {
  it("uses the ceil((n+1)(1-a)) order statistic", () => {
    const s = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(conformalQuantile(s, 0.9)).toBe(9);
    expect(conformalQuantile(s, 0.5)).toBe(5);
    expect(conformalQuantile(s, 0.95)).toBe(Infinity);
    expect(conformalQuantile(s, 0)).toBe(-Infinity);
  });

  it("window keeps the last W scores sorted", () => {
    const h = new ScoreHistory(3);
    [5, 1, 4, 2, 3].forEach((v) => h.push(v));
    expect([...h.windowSorted]).toEqual([2, 3, 4]);
  });
});

describe("online methods reach nominal coverage", () => {
  // Scores with a variance shift halfway: adaptive methods should still cover ~90%.
  const alpha = 0.1;
  const r = rng(7);
  const scores = Array.from({ length: 4000 }, (_, t) => Math.abs(normInv(r())) * (t < 2000 ? 1 : 3));
  // Purely model-based methods have no such guarantee (see tests/extra.test.ts).
  for (const spec of METHODS.filter((m) => m.guarantee !== false && !m.quantile)) {
    it(spec.name, () => {
      const h = new ScoreHistory(100);
      const m = createSolo(spec, h, alpha);
      let covered = 0, n = 0;
      scores.forEach((s, t) => {
        const q = m.predict(t);
        if (t >= 50) {
          n++;
          if (s <= (Number.isFinite(q) ? q : Infinity)) covered++;
        }
        m.update(t, s);
        h.push(s);
      });
      expect(Math.abs(covered / n - (1 - alpha))).toBeLessThan(0.03);
    });
  }
});

describe("engine", () => {
  const naive = new NaiveBackend();

  async function run(engine: Engine, y: number[], horizon: number) {
    for (let t = 0; t < y.length; t++) {
      engine.append(y[t]);
      const fc = await naive.forecast({ context: Float64Array.from(engine.series), horizon, symmetric: false, clip: false, maxContext: 512 });
      engine.addForecast(t, fc.quantiles, fc.horizon, fc.levels);
    }
    engine.catchUp();
  }

  it("scores a forecast only against later points and replays deterministically", async () => {
    const r = rng(3);
    const y: number[] = [];
    let v = 0;
    for (let t = 0; t < 300; t++) y.push((v += normInv(r())));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4 });
    await run(e, y, 8);
    const before = e.table();
    expect(e.scoresAt(1).length).toBe(299);
    expect(e.scoresAt(4).length).toBe(296);
    e.replay();
    expect(e.table()).toEqual(before);
    const native = before.find((row) => row.id === NATIVE_ID)!;
    expect(native.n).toBeGreaterThan(1000);
    for (const row of before) expect(row.coverage).toBeGreaterThan(0.75);
  });

  it("truncation drops later forecasts and equals a fresh run on the prefix", async () => {
    const r = rng(9);
    const y = Array.from({ length: 200 }, () => normInv(r()));
    const a = new Engine({ ...defaultSettings(), evalLeads: 3 });
    await run(a, y, 4);
    a.truncate(120);
    const b = new Engine({ ...defaultSettings(), evalLeads: 3 });
    await run(b, y.slice(0, 120), 4);
    expect(a.table()).toEqual(b.table());
    expect([...a.forecasts.keys()].every((o) => o < 120)).toBe(true);
  });

  it("ranks by interval score, after the coverage test when asked", () => {
    const rows = [row("a", 0.9, 2, 3, 2.9), row("b", 0.7, 1, 2.5, 2), row("c", 0.95, 4, 4, 2.5)];
    expect(rankRows(rows, 0.1, "interval_score").map((r) => r.id)).toEqual(["b", "a", "c"]);
    // Without bootstrap intervals nothing can be called significantly under target.
    expect(rankRows(rows, 0.1, "valid_then_score").map((r) => r.id)).toEqual(["b", "a", "c"]);
    const unc = coverageIntervals({ a: [0.87, 0.93], b: [0.65, 0.75], c: [0.92, 0.97] });
    const ranked = rankRows(rows, 0.1, "valid_then_score", unc);
    expect(ranked.map((r) => r.id)).toEqual(["a", "c", "b"]);
    expect(ranked.map((r) => r.under)).toEqual([false, false, true]);
  });

  it("an over-wide interval pays for its extra width, whatever its shape score", () => {
    // Covers 98% with a good shape once shrunk in hindsight, but issued much more width than
    // the on-target row (Holt-Winters' native band at 64 steps on "Seasonal + noise").
    const rows = [row("wide", 0.981, 130.8, 133.8, 108.6), row("ontarget", 0.954, 114.1, 120.9, 120.8)];
    const unc = coverageIntervals({ wide: [0.974, 0.987], ontarget: [0.937, 0.968] });
    expect(rankRows(rows, 0.05, "valid_then_score", unc).map((r) => r.id)).toEqual(["ontarget", "wide"]);
  });

  it("the coverage test is about significance, not a fixed tolerance", () => {
    // 86.3% coverage at width 8.04 beats 89.7% at 9.20 on the interval score alone.
    const rows = [row("narrow", 0.863, 8.04, 10.2, 10.0), row("calibrated", 0.897, 9.2, 10.6, 10.5)];
    expect(rankRows(rows, 0.1, "interval_score").map((r) => r.id)).toEqual(["narrow", "calibrated"]);
    // On a long series 86.3% is significantly under 90%, and the narrow interval ranks last.
    const long = coverageIntervals({ narrow: [0.845, 0.88], calibrated: [0.88, 0.914] });
    expect(rankRows(rows, 0.1, "valid_then_score", long).map((r) => r.id)).toEqual(["calibrated", "narrow"]);
    // On a short one the same 86.3% is within noise of the target, and the better score wins.
    const short = coverageIntervals({ narrow: [0.8, 0.92], calibrated: [0.84, 0.95] });
    expect(rankRows(rows, 0.1, "valid_then_score", short).map((r) => r.id)).toEqual(["narrow", "calibrated"]);
    // Rows that fail the test are ordered by their shortfall, whatever their score.
    const both = coverageIntervals({ narrow: [0.845, 0.88], calibrated: [0.88, 0.899] });
    expect(rankRows(rows, 0.1, "valid_then_score", both).map((r) => r.id)).toEqual(["calibrated", "narrow"]);
  });

  /** Ten unit-width intervals; values land 0.5 above two of them (80% coverage). */
  const tenPairs = () => {
    const above = [-0.5, -0.4, -0.3, -0.2, -0.1, -0.1, -0.2, -0.3, 0.5, 0.5];
    return { above, below: above.map((a) => -1 - a) };
  };

  it("shape score moves both bounds by the margin that minimises the interval score", () => {
    const p = tenPairs();
    const raw = (alpha: number) => p.above.reduce((s, a, i) => s + shiftedScore(p.below[i], a, 0, alpha), 0) / 10;
    // At an 80% target the level is already right: no margin lowers the score.
    expect(raw(0.2)).toBeCloseTo(1 + (2 / 0.2) * 0.5 * 0.2, 10);
    expect(shapeScore([p], 0.2)).toBeCloseTo(raw(0.2), 10);
    // At a 90% target the two misses cost 20 x 0.5 each. Widening by 0.5 covers them: width 2.
    expect(raw(0.1)).toBeCloseTo(3, 10);
    expect(bestMargin(p.below, p.above, 0.1)).toBeCloseTo(0.5, 10);
    expect(shapeScore([p], 0.1)).toBeCloseTo(2, 10);
    // Each lead gets its own margin, so a second lead that needs none keeps its score.
    const fine = { below: [-0.5, -0.5], above: [-0.5, -0.5] };
    expect(shapeScore([p, fine], 0.1)).toBeLessThan((10 * 2 + 2 * 1) / 12 + 1e-9);
    expect(shapeScore([{ below: [], above: [] }], 0.1)).toBeNaN();
  });

  it("the fitted margin is the exact minimum, also with empty intervals", () => {
    const r = mulberry32(3);
    for (let rep = 0; rep < 120; rep++) {
      const n = 5 + Math.floor(r() * 120), alpha = [0.02, 0.1, 0.2, 0.5][rep % 4];
      const below: number[] = [], above: number[] = [];
      for (let i = 0; i < n; i++) {
        // Every third case has bounds that cross (negative width), as some online methods issue.
        const w = rep % 3 === 0 ? (r() - 0.2) * 3 : 0.5 + r() * 3;
        const y = normInv(r()) * (rep % 5 === 0 ? 0.3 : 1.5);
        below.push(-w / 2 - y);
        above.push(y - w / 2);
      }
      const total = (m: number) => below.reduce((s, b, i) => s + shiftedScore(b, above[i], m, alpha), 0);
      // The score is piecewise linear in the margin, so its minimum is at one of the breakpoints.
      let best = Infinity;
      for (let i = 0; i < n; i++) for (const m of [below[i], above[i], (below[i] + above[i]) / 2]) best = Math.min(best, total(m));
      expect(total(bestMargin(below, above, alpha))).toBeCloseTo(best, 8);
    }
  });

  it("width at target widens or narrows every interval by one margin per lead until the target is covered", () => {
    const p = tenPairs();
    expect(widthAtTarget([p], 0.8)).toBeCloseTo(1 - 0.2, 10);
    expect(widthAtTarget([p], 0.9)).toBeCloseTo(2, 10);
    // Split across two leads, each is brought to the target with its own margin: the first
    // (all inside) narrows to 0.6, the second (two misses in six) widens to 2.
    const part = (from: number, to: number) => ({ below: p.below.slice(from, to), above: p.above.slice(from, to) });
    expect(widthAtTarget([part(0, 4), part(4, 10)], 0.9)).toBeCloseTo((4 * 0.6 + 6 * 2) / 10, 10);
    expect(widthAtTarget([{ below: [], above: [] }], 0.9)).toBeNaN();
  });

  it("worst window coverage only counts full windows", () => {
    const t = Array.from({ length: 10 }, (_, i) => i);
    const n = new Array(10).fill(1);
    const cov = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1];
    // Windows of 4 steps end at t = 3..9; the worst is t = 4..7 or 5..8 with 2 of 4 covered.
    expect(worstWindowCoverage(t, n, cov, 4)).toBe(0.5);
    expect(worstWindowCoverage(t, n, cov, 2)).toBe(0);
    expect(worstWindowCoverage(t, n, cov, 11)).toBeNaN();
  });

  it("table reports the extra scores consistently", async () => {
    const r = rng(5);
    const y = Array.from({ length: 300 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4, rolling: 50 });
    await run(e, y, 4);
    for (const row of [...e.table(), ...e.table(2)]) {
      expect(row.shapeScore).toBeLessThanOrEqual(row.intervalScore + 1e-9);
      expect(row.levelCost).toBeCloseTo(row.intervalScore - row.shapeScore, 9);
      expect(row.worstCoverage).toBeLessThanOrEqual(row.coverage);
      expect(Number.isFinite(row.widthAtTarget)).toBe(true);
    }
    // The pooled shape score is the pair-weighted mean of the per-lead ones.
    const pooled = e.table().find((x) => x.id === NATIVE_ID)!;
    const perLead = [1, 2, 3, 4].map((k) => e.table(k).find((x) => x.id === NATIVE_ID)!);
    expect(pooled.shapeScore).toBeCloseTo(perLead.reduce((s, x) => s + x.shapeScore * x.n, 0) / pooled.n, 9);
    // A second call reuses the cached fits and gives the same rows.
    expect(e.table()).toEqual(e.table());
    for (const u of [e.uncertainty(), e.uncertainty(2)]) {
      const n = u.get(NATIVE_ID)!;
      expect(n.shapeScore![0]).toBeLessThanOrEqual(n.shapeScore![1]);
      expect(n.shapeSamples).toHaveLength(n.scoreSamples.length);
      expect(n.worstWindow!.low).toBeLessThanOrEqual(n.worstWindow!.typical);
      expect(n.worstWindow!.typical).toBeLessThanOrEqual(0.9);
    }
    expect(rankRows(e.table(), 0.1, "valid_then_score", e.uncertainty())[0].rank).toBe(1);
  });

  it("the worst-window reference matches what independent misses at the target show", () => {
    // 2,000 steps with a 10% miss rate: the worst of the 100-step windows is typically 82%,
    // and under 79% in about one run in twenty (simulated separately with 4,000 runs).
    const r = mulberry32(21);
    const T = 2000;
    let typical = 0, low = 0;
    const runs = 20;
    for (let i = 0; i < runs; i++) {
      const cov = Array.from({ length: T }, () => (r() < 0.9 ? 1 : 0));
      const ones = new Array(T).fill(1);
      const u = blockBootstrap(new Map([["a", { n: ones, cov, width: ones, is: ones }]]), T, 26, 300, 100 + i, 100).get("a")!;
      const [median, q95] = u.worstDrop!;
      typical += (0.9 - median) / runs;
      low += (0.9 - q95) / runs;
    }
    expect(typical).toBeGreaterThan(0.81);
    expect(typical).toBeLessThan(0.835);
    expect(low).toBeGreaterThan(0.775);
    expect(low).toBeLessThan(0.805);
    // No window length, or one longer than the series: no reference.
    const ones = new Array(50).fill(1);
    expect(blockBootstrap(new Map([["a", { n: ones, cov: ones, width: ones, is: ones }]]), 50, 5, 50).get("a")!.worstDrop).toBeUndefined();
    expect(blockBootstrap(new Map([["a", { n: ones, cov: ones, width: ones, is: ones }]]), 50, 5, 50, 1, 80).get("a")!.worstDrop).toBeUndefined();
  });

  it("ACI's infinite interval issues the cap but never counts as a miss", () => {
    const h = new ScoreHistory(100);
    h.push(1);
    expect(h.cap(Infinity)).toBe(1);
    // ACI at a tiny alpha issues +inf, drawn as the cap 1. A score of 10 falls outside the
    // drawn band, but the paper's set is the whole line, so alpha_t must rise, not fall.
    const aci = createSolo(METHODS.find((m) => m.id === "aci")!, h, 0.01, { gamma: 0.5 });
    expect(aci.predict(0)).toBe(1);
    aci.update(0, 10);
    h.push(10);
    expect((aci as unknown as { alphaT: number }).alphaT).toBeCloseTo(0.01 + 0.5 * 0.01, 12);
    // alpha_t is still below 1 / (n + 1), so the next threshold is the cap (largest score).
    expect(aci.predict(1)).toBe(10);
  });

  it("aciErr reads +inf as the whole line and -inf as the empty set", () => {
    expect(aciErr(1e9, Infinity)).toBe(0);
    expect(aciErr(-1e9, -Infinity)).toBe(1);
    expect(aciErr(2, 1)).toBe(1);
    expect(aciErr(1, 1)).toBe(0);
  });

  // Record-breaking scores: every new score exceeds the cap, the worst case for a capped band.
  const records = Array.from({ length: 300 }, (_, t) => 1.05 ** t);
  const levels = (m: object): number[] => {
    const o = m as { alphaT?: number; alphas?: number[] };
    return o.alphas ?? [o.alphaT!];
  };

  for (const id of ["aci", "dtaci", "agaci", "bocpd_aci"]) {
    it(`${id}: one-step alpha_t stays in [-gamma, 1 + gamma] (Gibbs & Candès Lemma 4.1)`, () => {
      const spec = METHODS.find((m) => m.id === id)!;
      const gamma = 0.05;
      const h = new ScoreHistory(100);
      const m = createSolo(spec, h, 0.1, { ...defaultParams(spec), gamma });
      // DtACI and AgACI run experts over the default gamma grid.
      const gammas = id === "dtaci" || id === "agaci" ? [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128] : [gamma];
      for (let t = 0; t < records.length; t++) {
        m.predict(t);
        m.update(t, records[t]);
        h.push(records[t]);
        levels(m).forEach((a, i) => {
          expect(a).toBeGreaterThanOrEqual(-gammas[i] - 1e-12);
          expect(a).toBeLessThanOrEqual(1 + gammas[i] + 1e-12);
        });
      }
    });
  }

  it("per-lead tables add up to the pooled table", async () => {
    const r = rng(5);
    const y = Array.from({ length: 250 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4 });
    await run(e, y, 6);
    const pooled = e.table();
    for (const row of pooled) {
      const per = [1, 2, 3, 4].map((k) => e.table(k).find((x) => x.id === row.id)!);
      const n = per.reduce((a, p) => a + p.n, 0);
      expect(n).toBe(row.n);
      expect(per.reduce((a, p) => a + p.coverage * p.n, 0) / n).toBeCloseTo(row.coverage, 10);
      expect(per.reduce((a, p) => a + p.intervalScore * p.n, 0) / n).toBeCloseTo(row.intervalScore, 10);
      const prof = e.leadProfile(row.id);
      expect(prof.coverage.length).toBe(4);
      expect(prof.coverage[0]).toBeCloseTo(per[0].coverage, 10);
    }
  });

  it("rolling window counts time steps, not pairs", async () => {
    const r = rng(6);
    const y = Array.from({ length: 120 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4, rolling: 1 });
    await run(e, y, 4);
    const m = e.metrics.get(NATIVE_ID)!;
    const last = m.arr.n.length - 1;
    // R = 1 step: the rolling value is exactly the last arrival's pairs (4 leads).
    expect(m.arr.n[last]).toBe(4);
    expect(e.table().find((x) => x.id === NATIVE_ID)!.rollingCoverage).toBeCloseTo(m.arr.cov[last] / 4, 12);
  });

  it("rolling window expires by elapsed steps when forecasts are sparse", async () => {
    const r = rng(8);
    const y = Array.from({ length: 100 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4, warmup: 1, rolling: 5 });
    for (let t = 0; t < y.length; t++) {
      e.append(y[t]);
      if (t === 10 || t === 15) {
        const fc = await naive.forecast({ context: Float64Array.from(e.series), horizon: 4, symmetric: false, clip: false, maxContext: 512 });
        e.addForecast(t, fc.quantiles, fc.horizon, fc.levels);
      }
    }
    e.catchUp();
    const row = e.table().find((x) => x.id === NATIVE_ID)!;
    expect(row.n).toBe(4); // origin 15, leads 1-4, scored at t = 16..19
    expect(Number.isNaN(row.rollingCoverage)).toBe(true); // nothing in the last 5 steps
    expect(Number.isNaN(e.table(1).find((x) => x.id === NATIVE_ID)!.rollingCoverage)).toBe(true);
  });

  it("bootstrap reports nothing with fewer than two blocks of data", () => {
    const s = { n: [1, 1, 1, 1], cov: [1, 0, 1, 1], width: [1, 2, 3, 4], is: [1, 2, 3, 4] };
    expect(blockBootstrap(new Map([["a", s]]), 4, 16).size).toBe(0);
  });

  it("block bootstrap covers the truth and detects ties", () => {
    const r = rng(11);
    const T = 600;
    const mk = (shift: number) => {
      const is = Array.from({ length: T }, () => 1 + shift + Math.abs(normInv(r())));
      return { n: new Array(T).fill(1), cov: is.map((v) => (v < 2.5 ? 1 : 0)), width: is, is };
    };
    const a = mk(0), b = mk(0), c = mk(1);
    const u = blockBootstrap(new Map([["a", a], ["b", b], ["c", c]]), T, 16, 400);
    const mean = a.is.reduce((x, v) => x + v, 0) / T;
    const ci = u.get("a")!.intervalScore;
    expect(ci[0]).toBeLessThan(mean);
    expect(ci[1]).toBeGreaterThan(mean);
    expect(indistinguishable(u.get("a")!.scoreSamples, u.get("b")!.scoreSamples)).toBe(true);
    expect(indistinguishable(u.get("a")!.scoreSamples, u.get("c")!.scoreSamples)).toBe(false);
  });

  it("ranking shares ranks between indistinguishable rows", async () => {
    const r = rng(13);
    const y = Array.from({ length: 300 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4 });
    await run(e, y, 4);
    const ranked = rankRows(e.table(), 0.1, "interval_score", e.uncertainty());
    expect(ranked[0].rank).toBe(1);
    for (const row of ranked) if (row.tied) expect(row.rank).toBeLessThan(ranked.indexOf(row) + 1);
    // Several online methods are statistically indistinguishable on iid noise.
    expect(ranked.some((x) => x.tied)).toBe(true);
  });

  it("asymmetric CQR calibrates each side and beats one shared threshold on skewed errors", () => {
    // The model claims N(0, 1); the truth is exponential (always positive, right-skewed).
    const r = rng(21);
    const H = 2;
    const q = new Float32Array(H * TIMESTRAL_LEVELS.length);
    for (let k = 0; k < H; k++) TIMESTRAL_LEVELS.forEach((p, i) => (q[k * TIMESTRAL_LEVELS.length + i] = normInv(p)));
    const runWith = (scoreType: "cqr" | "cqr_asym") => {
      const e = new Engine({ ...defaultSettings(), evalLeads: H, scoreType, window: 200 });
      for (let t = 0; t < 3000; t++) {
        e.append(-Math.log(r()));
        e.addForecast(t, q, H, TIMESTRAL_LEVELS);
      }
      e.catchUp();
      return e.table().find((x) => x.id === "split")!;
    };
    const sym = runWith("cqr"), asym = runWith("cqr_asym");
    expect(Math.abs(asym.coverage - 0.9)).toBeLessThan(0.03);
    expect(asym.meanWidth).toBeLessThan(sym.meanWidth);
    expect(asym.intervalScore).toBeLessThan(sym.intervalScore);
  });

  it("naive baseline continues a ramp instead of staying flat", async () => {
    const ramp = Float64Array.from({ length: 100 }, (_, t) => 2 * t);
    const fc = await new NaiveBackend().forecast({ context: ramp, horizon: 16, symmetric: false, clip: false, maxContext: 512 });
    const Q = fc.levels.length, mi = fc.levels.indexOf(0.5);
    const med = (k: number) => fc.quantiles[(k - 1) * Q + mi];
    expect(med(1)).toBeGreaterThan(198);
    expect(med(16)).toBeGreaterThan(med(1));
    // Damped: never more than slope / (1 - phi) above the last value.
    expect(med(16) - 198).toBeLessThan(2 * 0.9 / 0.1 + 1e-6);
  });

  it("pastBand returns the interval issued k steps ahead, aligned to its target time", async () => {
    const r = rng(17);
    const y = Array.from({ length: 120 }, () => normInv(r()));
    const e = new Engine({ ...defaultSettings(), evalLeads: 4 });
    await run(e, y, 4);
    for (const id of e.rowIds) {
      const band = e.pastBand(id, 3, 50, 80);
      expect(band.length).toBe(31);
      for (const [t, lo, hi, missed] of band) {
        const iv = e.interval(e.forecasts.get(t - 3)!, 2, id)!;
        expect([lo, hi]).toEqual(iv);
        expect(missed).toBe(y[t] < lo || y[t] > hi);
      }
    }
    // The misses at lead k agree with the per-lead coverage.
    const all = e.pastBand(NATIVE_ID, 2, 0, 119).filter((b) => e.forecasts.get(b[0] - 2)!.evaluated[1]);
    const cov = e.table(2).find((x) => x.id === NATIVE_ID)!.coverage;
    expect(1 - all.filter((b) => b[3]).length / all.length).toBeCloseTo(cov, 10);
  });

  it("interpolates quantiles between knots", () => {
    const q = Float32Array.from(TIMESTRAL_LEVELS);
    expect(quantileAt(q, 0, TIMESTRAL_LEVELS, 0.025)).toBeCloseTo(0.025, 6);
    expect(quantileAt(q, 0, TIMESTRAL_LEVELS, 0.5)).toBeCloseTo(0.5, 6);
  });
});

describe("ACI variants", () => {
  const spec = (id: string) => METHODS.find((m) => m.id === id)!;
  const alphaT = (m: object) => (m as { alphaT: number }).alphaT;

  it("lead-scaled ACI steps gamma / k at lead k", () => {
    const h = new ScoreHistory(100);
    for (let i = 1; i <= 50; i++) h.push(i);
    const m = spec("aci_lead").create(h, 0.1, { gamma: 0.04 }, { lead: 4, shared: new Map() });
    m.predict(0);
    m.update(0, 1000); // a miss
    expect(alphaT(m)).toBeCloseTo(0.1 + (0.04 / 4) * (0.1 - 1), 12);
  });

  it("shared-alpha ACI moves one level for all leads, by gamma / leads", () => {
    const shared = new Map<string, unknown>();
    const h1 = new ScoreHistory(100), h2 = new ScoreHistory(100);
    for (let i = 1; i <= 50; i++) (h1.push(i), h2.push(10 * i));
    const lead1 = spec("aci_shared").create(h1, 0.1, { gamma: 0.04 }, { lead: 1, shared });
    const lead2 = spec("aci_shared").create(h2, 0.1, { gamma: 0.04 }, { lead: 2, shared });
    const before = lead2.predict(0);
    lead1.predict(0);
    lead1.update(0, 1000); // a miss at lead 1 widens lead 2 right away
    expect(lead2.predict(1)).toBeGreaterThan(before);
    const level = shared.get("aci_shared") as { alphaT: number };
    expect(level.alphaT).toBeCloseTo(0.1 + (0.04 / 2) * (0.1 - 1), 12);
  });

  it("smooth the horizon after level shifts where paper ACI spikes", async () => {
    const ids = ["aci", "aci_lead", "aci_shared", "dtaci", "dtaci_lead", "agaci", "agaci_lead"];
    const s = defaultSettings();
    for (const id of ["aci", "aci_lead", "aci_shared"]) s.params[id] = { gamma: 0.05 };
    const e = new Engine({ ...s, enabled: Object.fromEntries(ids.map((id) => [id, true])) });
    const b = new NaiveBackend();
    const y = PRESETS.find((p) => p.id === "levels")!.make(600, 3);
    const jumps = new Map<string, number[]>(ids.map((id) => [id, []]));
    for (const v of y) {
      e.append(v);
      e.catchUp();
      const t = e.series.length - 1;
      const r = await b.forecast({ context: Float64Array.from(e.series), horizon: 16, symmetric: true, clip: false, maxContext: 256 });
      const f = e.addForecast(t, r.quantiles, r.horizon, r.levels);
      if (t < 100) continue;
      for (const id of ids) {
        const w = Array.from({ length: 16 }, (_, j) => {
          const iv = e.interval(f, j, id)!;
          return iv[1] - iv[0];
        });
        const med = [...w].sort((a, c) => a - c)[8];
        let jump = 0;
        for (let j = 1; j < 16; j++) jump = Math.max(jump, Math.abs(w[j] - w[j - 1]));
        jumps.get(id)!.push(jump / med);
      }
    }
    // 95th percentile of the largest lead-to-lead width jump, relative to the median width.
    const p95 = (id: string) => {
      const a = jumps.get(id)!.sort((x, z) => x - z);
      return a[Math.floor(a.length * 0.95)];
    };
    expect(p95("aci_lead")).toBeLessThan(0.5 * p95("aci"));
    expect(p95("aci_shared")).toBeLessThan(0.5 * p95("aci"));
    expect(p95("dtaci_lead")).toBeLessThan(p95("dtaci"));
    expect(p95("agaci_lead")).toBeLessThanOrEqual(p95("agaci"));
    for (const row of e.table().filter((r) => ids.includes(r.id))) expect(row.coverage, row.id).toBeGreaterThan(0.86);
  });
});
