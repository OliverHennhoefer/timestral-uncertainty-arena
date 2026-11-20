import { describe, expect, it } from "vitest";
import { MultiQT, isotonic } from "../src/conformal/multiqt";
import { METHODS } from "../src/conformal/methods";
import { Engine, NATIVE_ID, defaultSettings } from "../src/engine/engine";
import { NaiveBackend, normInv } from "../src/model/backends";
import { PRESETS } from "../src/synthetic";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s + 0.5) / 2 ** 32;
  };
}

describe("isotonic projection", () => {
  it("pools adjacent violators and leaves ordered input alone", () => {
    expect([...isotonic([1, 2, 3])]).toEqual([1, 2, 3]);
    expect([...isotonic([3, 1, 2])]).toEqual([2, 2, 2]);
    expect([...isotonic([1, 3, 2, 4])]).toEqual([1, 2.5, 2.5, 4]);
    expect([...isotonic([4, 3, 2, 1])]).toEqual([2.5, 2.5, 2.5, 2.5]);
  });

  it("is the closest ordered vector", () => {
    const r = rng(3);
    for (let rep = 0; rep < 50; rep++) {
      const x = Array.from({ length: 6 }, () => normInv(r()));
      const p = isotonic(x);
      const dist = (z: ArrayLike<number>) => x.reduce((a, v, i) => a + (v - z[i]) ** 2, 0);
      for (let i = 1; i < p.length; i++) expect(p[i]).toBeGreaterThanOrEqual(p[i - 1]);
      // No ordered competitor (sorted input, or a perturbed projection re-sorted) is closer.
      expect(dist(p)).toBeLessThanOrEqual(dist([...x].sort((a, b) => a - b)) + 1e-12);
      const other = [...p].map((v) => v + 0.1 * normInv(r())).sort((a, b) => a - b);
      expect(dist(p)).toBeLessThanOrEqual(dist(other) + 1e-12);
    }
  });
});

describe("MultiQT", () => {
  const levels = [0.05, 0.25, 0.5, 0.75, 0.95];

  it("follows the paper's update on the played forecast", () => {
    const m = new MultiQT(levels);
    const base = Float64Array.from([1, 2, 3, 4, 5]);
    expect([...m.predict(0, base)]).toEqual([1, 2, 3, 4, 5]);
    m.update(0, 3.5); // covered by levels 0.75 and 0.95 only; eta is the floor 0.1 with no history
    const want = levels.map((a, i) => -0.1 * ((3.5 <= base[i] ? 1 : 0) - a));
    m.offsets.forEach((v, i) => expect(v).toBeCloseTo(want[i], 12));
  });

  it("calibrates every level under delayed feedback and never crosses", () => {
    // A base forecast that is biased and far too narrow, with a variance shift halfway.
    const r = rng(11);
    const delay = 3;
    const m = new MultiQT(levels);
    const T = 6000;
    const y = Array.from({ length: T }, (_, t) => 50 + (t < T / 2 ? 2 : 6) * normInv(r()));
    const played: Float64Array[] = [];
    const hits = levels.map(() => 0);
    let n = 0;
    for (let t = 0; t < T; t++) {
      const base = Float64Array.from(levels.map((a) => 52 + 0.5 * normInv(a)));
      const q = m.predict(t, base);
      for (let i = 1; i < q.length; i++) expect(q[i]).toBeGreaterThanOrEqual(q[i - 1]);
      played.push(q);
      if (t >= 500) {
        n++;
        levels.forEach((_, i) => { if (y[t] <= q[i]) hits[i]++; });
      }
      if (t >= delay) m.update(t - delay, y[t - delay]);
    }
    levels.forEach((a, i) => expect(Math.abs(hits[i] / n - a), `level ${a}`).toBeLessThan(0.02));
  });
});

describe("engine with MultiQT", () => {
  it("is registered off by default and scores alongside the scalar methods", async () => {
    const spec = METHODS.find((m) => m.id === "multiqt")!;
    expect(spec.defaultOn).toBe(false);
    expect(defaultSettings().enabled.multiqt).toBe(false);
    const naive = new NaiveBackend();
    const run = async (enabled: Record<string, boolean>, scoreType: "cqr" | "cqr_asym" | "abs") => {
      const e = new Engine({ ...defaultSettings(), evalLeads: 4, scoreType, enabled });
      const y = PRESETS.find((p) => p.id === "volshift")!.make(400, 5);
      for (let t = 0; t < y.length; t++) {
        e.append(y[t]);
        const fc = await naive.forecast({ context: Float64Array.from(e.series), horizon: 8, symmetric: false, clip: false, maxContext: 256 });
        e.addForecast(t, fc.quantiles, fc.horizon, fc.levels);
      }
      e.catchUp();
      return e;
    };
    const e = await run({ multiqt: true, aci: true }, "cqr");
    const rows = Object.fromEntries(e.table().map((r) => [r.id, r]));
    expect(rows.multiqt.n).toBe(rows[NATIVE_ID].n);
    expect(rows.multiqt.n).toBeGreaterThan(1000);
    expect(rows.multiqt.coverage).toBeGreaterThan(0.8);
    expect(rows.multiqt.coverage).toBeLessThan(0.98);
    // Its interval does not depend on the score type, and the other rows do not depend on it.
    const asym = Object.fromEntries((await run({ multiqt: true, aci: true }, "abs")).table().map((r) => [r.id, r]));
    expect(asym.multiqt).toEqual(rows.multiqt);
    const without = Object.fromEntries((await run({ aci: true }, "cqr")).table().map((r) => [r.id, r]));
    expect(without.aci).toEqual(rows.aci);
    const before = e.table();
    e.replay();
    expect(e.table()).toEqual(before);
  });
});
