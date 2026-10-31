import { describe, expect, it } from "vitest";
import { METHODS, defaultParams } from "../src/conformal/methods";
import { ScoreHistory, conformalQuantile } from "../src/conformal/window";
import { Engine, NATIVE_ID, defaultSettings, quantileAt, rankRows } from "../src/engine/engine";
import { NaiveBackend, TIMESTRAL_LEVELS, normInv } from "../src/model/backends";

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
  for (const spec of METHODS) {
    it(spec.name, () => {
      const h = new ScoreHistory(100);
      const m = spec.create(h, alpha, defaultParams(spec));
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

  it("ranks by interval score and by validity", () => {
    const rows = [
      { id: "a", n: 10, coverage: 0.9, rollingCoverage: 0.9, meanWidth: 2, rollingWidth: 2, intervalScore: 3 },
      { id: "b", n: 10, coverage: 0.7, rollingCoverage: 0.7, meanWidth: 1, rollingWidth: 1, intervalScore: 5 },
      { id: "c", n: 10, coverage: 0.95, rollingCoverage: 0.95, meanWidth: 4, rollingWidth: 4, intervalScore: 4 },
    ];
    expect(rankRows(rows, 0.1, "interval_score").map((r) => r.id)).toEqual(["a", "c", "b"]);
    expect(rankRows(rows, 0.1, "validity_then_width", 0.06).map((r) => r.id)).toEqual(["a", "c", "b"]);
  });

  it("validity-first ranking treats over-coverage as valid", () => {
    const rows = [
      { id: "under", n: 10, coverage: 0.85, rollingCoverage: 0.85, meanWidth: 1, rollingWidth: 1, intervalScore: 1 },
      { id: "over", n: 10, coverage: 1, rollingCoverage: 1, meanWidth: 5, rollingWidth: 5, intervalScore: 5 },
    ];
    expect(rankRows(rows, 0.1, "validity_then_width").map((r) => r.id)).toEqual(["over", "under"]);
  });

  it("methods learn from the threshold actually issued (capped)", () => {
    const h = new ScoreHistory(100);
    h.push(1);
    expect(h.cap(Infinity)).toBe(1);
    // ACI at a tiny alpha issues +inf, capped to 1; a score of 10 must count as a miss.
    const aci = METHODS.find((m) => m.id === "aci")!.create(h, 0.01, { gamma: 0.5 });
    expect(aci.predict(0)).toBe(1);
    aci.update(0, 10);
    h.push(10);
    // After a miss alpha_t drops below 0, so the next threshold is the cap (largest score).
    expect(aci.predict(1)).toBe(10);
  });

  it("interpolates quantiles between knots", () => {
    const q = Float32Array.from(TIMESTRAL_LEVELS);
    expect(quantileAt(q, 0, TIMESTRAL_LEVELS, 0.025)).toBeCloseTo(0.025, 6);
    expect(quantileAt(q, 0, TIMESTRAL_LEVELS, 0.5)).toBeCloseTo(0.5, 6);
  });
});
