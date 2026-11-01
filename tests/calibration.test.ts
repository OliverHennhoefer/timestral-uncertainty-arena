import { describe, expect, it } from "vitest";
import { CalibrationProbe, METHOD_GRID, pitCoverage, pitHistogram, pitValue } from "../src/engine/calibration";
import { Engine, NATIVE_ID, defaultSettings } from "../src/engine/engine";
import { NaiveBackend, normInv } from "../src/model/backends";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const naive = new NaiveBackend();

async function step(engine: Engine, y: number, horizon: number) {
  engine.append(y);
  const fc = await naive.forecast({ context: Float64Array.from(engine.series), horizon, symmetric: false, clip: false, maxContext: 512 });
  engine.addForecast(engine.series.length - 1, fc.quantiles, fc.horizon, fc.levels);
}

describe("PIT", () => {
  const levels = [0.1, 0.5, 0.9];
  const q = Float32Array.from([0, 10, 20]);

  it("interpolates the quantile vector and sends the tails to 0 and 1", () => {
    expect(pitValue(q, 0, levels, 10)).toBeCloseTo(0.5);
    expect(pitValue(q, 0, levels, 5)).toBeCloseTo(0.3);
    expect(pitValue(q, 0, levels, 0)).toBeCloseTo(0.1);
    expect(pitValue(q, 0, levels, -1)).toBe(0);
    expect(pitValue(q, 0, levels, 21)).toBe(1);
  });

  it("bins over [0, 1] and reads central coverage off the PIT values", () => {
    expect(pitHistogram([0, 0.05, 0.55, 1], 10)).toEqual([2, 0, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(pitCoverage([0.02, 0.3, 0.5, 0.7, 0.98], 0.8)).toBeCloseTo(0.6);
  });
});

describe("calibration probe", () => {
  it("matches the main engine's native coverage and a fresh engine at every alpha", async () => {
    const r = rng(5);
    const y: number[] = [];
    let v = 0;
    for (let t = 0; t < 260; t++) y.push((v += normInv(r()) * (t > 150 ? 2.5 : 1)));
    const settings = { ...defaultSettings(), evalLeads: 4 };
    const main = new Engine(settings);
    const probe = new CalibrationProbe();
    // Sync while streaming, as the UI does, so the incremental path is what gets tested.
    for (let t = 0; t < y.length; t++) {
      await step(main, y[t], 6);
      if (t % 37 === 0) probe.sync(main, "aci");
    }
    main.catchUp();
    probe.sync(main, "aci");

    const rel = probe.reliability(main);
    const native = main.table().find((row) => row.id === NATIVE_ID)!;
    const at = rel.native.find((p) => Math.abs(p.nominal - 0.9) < 1e-9)!;
    expect(at.n).toBe(native.n);
    expect(at.empirical).toBeCloseTo(native.coverage, 10);

    expect(rel.method.map((p) => p.nominal)).toEqual(METHOD_GRID);
    for (const p of rel.method) {
      const fresh = new Engine({ ...settings, alpha: 1 - p.nominal });
      fresh.series = main.series.slice();
      for (const [o, f] of [...main.forecasts].sort((a, b) => a[0] - b[0])) fresh.addForecast(o, f.quantiles, f.horizon, f.levels);
      fresh.catchUp();
      const row = fresh.table().find((x) => x.id === "aci")!;
      expect(p.empirical).toBeCloseTo(row.coverage, 10);
      expect(p.n).toBe(row.n);
    }
    // Per lead, too.
    const lead2 = probe.reliability(main, 2).method.find((p) => Math.abs(p.nominal - 0.9) < 1e-9)!;
    expect(lead2.empirical).toBeCloseTo(main.table(2).find((x) => x.id === "aci")!.coverage, 10);
  });

  it("gives the same curve when the rebuild runs in slices", async () => {
    const r = rng(7);
    const main = new Engine({ ...defaultSettings(), evalLeads: 3 });
    for (let t = 0; t < 150; t++) await step(main, normInv(r()), 4);
    const whole = new CalibrationProbe();
    whole.sync(main, "aci");
    const sliced = new CalibrationProbe();
    sliced.sync(main, "aci", 0);
    expect(sliced.ready).toBe(false);
    expect(sliced.reliability(main).method).toEqual([]);
    let calls = 1;
    while (!sliced.ready) (sliced.sync(main, "aci", 0), calls++);
    expect(calls).toBeGreaterThan(10);
    expect(sliced.reliability(main)).toEqual(whole.reliability(main));
  });

  it("rebuilds after the main engine replays", async () => {
    const r = rng(11);
    const main = new Engine({ ...defaultSettings(), evalLeads: 3 });
    const probe = new CalibrationProbe();
    for (let t = 0; t < 120; t++) await step(main, normInv(r()), 4);
    probe.sync(main, "aci");
    const before = probe.pit().length;
    main.truncate(80);
    probe.sync(main, "aci");
    expect(probe.pit().length).toBeLessThan(before);
    expect(probe.pit().length).toBe(main.table().find((x) => x.id === NATIVE_ID)!.n);
    // Focusing the native row needs no shadow engines.
    probe.sync(main, NATIVE_ID);
    expect(probe.reliability(main).method).toEqual([]);
  });
});
