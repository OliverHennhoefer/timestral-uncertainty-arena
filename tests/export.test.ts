import { describe, expect, it } from "vitest";
import { Engine, NATIVE_COLOR, NATIVE_ID, defaultSettings } from "../src/engine/engine";
import { METHODS } from "../src/conformal/methods";
import { LAYOUTS, MAX_GIF_FRAMES, schedule } from "../src/export/layouts";
import { type ExportSource, SceneReplay } from "../src/export/scene";

const LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/** A noisy seasonal series with a forecast at every origin, as a played run stores them. */
function run(n = 300, H = 24) {
  const settings = defaultSettings();
  settings.enabled = Object.fromEntries(METHODS.map((m) => [m.id, ["split", "aci"].includes(m.id)]));
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  const series = Array.from({ length: n }, (_, t) => 50 + 20 * Math.sin(t / 8) + 8 * rand());
  const live = new Engine(structuredClone(settings));
  const forecasts: ExportSource["forecasts"] = [];
  for (let t = 0; t < n; t++) {
    live.append(series[t]);
    live.catchUp(t);
    const q = new Float32Array(H * LEVELS.length);
    for (let j = 0; j < H; j++) LEVELS.forEach((p, i) => (q[j * LEVELS.length + i] = series[t] + (p - 0.5) * (6 + j)));
    live.addForecast(t, q, H, LEVELS);
    forecasts.push({ origin: t, horizon: H, levels: LEVELS, quantiles: q });
  }
  const source: ExportSource = {
    series, forecasts, settings, align: "cut", focusId: "aci", title: "Test", modelName: "Naive",
    names: { [NATIVE_ID]: "Naive native", aci: "ACI", split: "Split" },
    colors: { [NATIVE_ID]: NATIVE_COLOR, aci: "#FF9500", split: "#007AFF" },
  };
  return { live, source };
}

describe("export replay", () => {
  it("reproduces the live engine at the end of the run", () => {
    const { live, source } = run();
    const r = new SceneReplay(source);
    r.advanceTo(source.series.length - 1);
    expect(r.engine.table()).toEqual(live.table());
    const f = r.frame(150, 0, source.series.length - 1);
    const m = live.metrics.get("aci")!;
    expect(f.stats[0].id).toBe("aci");
    expect(f.stats[0].coverage).toBeCloseTo(m.covered / m.n, 12);
    expect(f.stats[1].id).toBe(NATIVE_ID);
  });

  it("matches a live engine stopped at an earlier step", () => {
    const { source } = run();
    const stop = 120;
    const partial = { ...source, series: source.series.slice(0, stop + 1), forecasts: source.forecasts.filter((f) => f.origin <= stop) };
    const a = new SceneReplay(source);
    a.advanceTo(stop);
    const b = new SceneReplay(partial);
    b.advanceTo(stop);
    expect(a.engine.table()).toEqual(b.engine.table());
  });

  it("frames a scrolling window that grows first, and keeps the band within the calibrated leads", () => {
    const { source } = run();
    const r = new SceneReplay(source);
    r.advanceTo(5);
    const early = r.frame(150, 0, 299);
    expect(early.x0).toBe(0);
    expect(early.x1 - early.x0).toBe(48 + 16 + 1);
    r.advanceTo(299);
    const late = r.frame(150, 0, 299);
    expect(late.x1 - late.x0).toBe(150 + 16 + 1);
    expect(late.t - late.x0).toBe(149);
    expect(late.median.length).toBe(16);
    expect(late.band.length).toBeLessThanOrEqual(16);
    expect(late.yLo).toBeLessThan(late.yHi);
  });

  it("falls back to the native interval when the focused method is off", () => {
    const { source } = run(80);
    const r = new SceneReplay({ ...source, focusId: "dtaci" });
    r.advanceTo(79);
    const f = r.frame(150, 0, 79);
    expect(f.stats.map((s) => s.id)).toEqual([NATIVE_ID]);
    expect(f.band).toEqual([]);
  });
});

describe("export schedule", () => {
  it("spreads long runs over the length and ends on the last step", () => {
    const s = schedule(0, 1999, 10, "mp4");
    expect(s.times.length).toBe(300);
    expect(s.times[s.times.length - 1]).toBe(1999);
    expect([...s.times].sort((a, b) => a - b)).toEqual(s.times);
  });

  it("plays short runs one step per frame and caps GIF frames", () => {
    expect(schedule(10, 59, 10, "gif").times).toEqual(Array.from({ length: 50 }, (_, i) => 10 + i));
    expect(schedule(0, 9999, 60, "gif").times.length).toBe(MAX_GIF_FRAMES);
  });

  it("uses even video sizes that match each aspect ratio", () => {
    for (const l of LAYOUTS) {
      for (const [w, h] of [l.video, l.gif]) {
        expect(w % 2 + (h % 2)).toBe(0);
        expect(w / h).toBeCloseTo(l.w / l.h, 2);
      }
    }
  });
});
