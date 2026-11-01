import { describe, expect, it } from "vitest";
import { defaultSettings } from "../src/engine/engine";
import { type Scenario, decodeScenario, encodeScenario, packValues, scenarioFromHash, seriesHash, unpackValues } from "../src/share";
import { PRESETS } from "../src/synthetic";

function scenario(over: Partial<Scenario> = {}): Scenario {
  return {
    series: { kind: "empty" },
    model: { backend: "naive" },
    horizon: 64,
    maxContext: 256,
    symmetric: true,
    clip: true,
    align: "full",
    leadsBefore: null,
    settings: defaultSettings(),
    rankBy: "interval_score",
    rankLead: 0,
    focus: "aci",
    ...over,
  };
}

describe("series packing", () => {
  it("round-trips values in hundredths, including big jumps and negatives", () => {
    const y = [0, 12.34, 99.99, 50, 50, -3.5, 1e5, 0.01];
    expect(unpackValues(packValues(y))).toEqual(y);
  });

  it("rounds to hundredths", () => {
    expect(unpackValues(packValues([1.234567]))).toEqual([1.23]);
  });

  it("rejects a truncated stream", () => {
    expect(() => unpackValues(Uint8Array.from([0x80]))).toThrow();
  });
});

describe("scenario links", () => {
  it("round-trip every setting", async () => {
    const d = defaultSettings();
    const ids = Object.keys(d.enabled);
    const enabled = Object.fromEntries(ids.map((id, i) => [id, i % 2 === 0]));
    const first = ids[0];
    const params = { ...d.params, [first]: Object.fromEntries(Object.keys(d.params[first]).map((k) => [k, 0.123])) };
    const s = scenario({
      model: { backend: "browser", repo: "mistralai/Timestral-28M" },
      horizon: 128,
      maxContext: 512,
      symmetric: false,
      clip: false,
      align: "calibrate",
      leadsBefore: 16,
      settings: { ...d, alpha: 0.2, evalLeads: 128, scoreType: "cqr_asym", window: 300, warmup: 20, rolling: 250, enabled, params },
      rankBy: "rolling_gap",
      rankLead: 3,
      focus: first,
    });
    const code = await encodeScenario(s);
    expect(code.startsWith("1z")).toBe(true);
    expect(await decodeScenario(code, defaultSettings())).toEqual(s);
  });

  it("carry a drawn series and its offset", async () => {
    const values = Array.from({ length: 800 }, (_, t) => Math.round((50 + 30 * Math.sin(t / 20)) * 100) / 100);
    const s = scenario({ series: { kind: "values", values, offset: 37 } });
    const code = await encodeScenario(s);
    expect(code.length).toBeLessThan(3000);
    expect((await decodeScenario(code, defaultSettings())).series).toEqual(s.series);
  });

  it("name a preset stream instead of its points", async () => {
    const p = PRESETS[0];
    const y = p.make(2000, 42).slice(0, 500);
    const s = scenario({ series: { kind: "preset", id: p.id, seed: 42, n: 500, hash: seriesHash(y) } });
    const code = await encodeScenario(s);
    expect(code.length).toBeLessThan(600);
    const back = await decodeScenario(code, defaultSettings());
    expect(back.series).toEqual(s.series);
    // The preset regenerates the same points from the seed.
    expect(seriesHash(p.make(2000, 42).slice(0, 500))).toBe((back.series as { hash: number }).hash);
  });

  it("clamp values and drop unknown methods", async () => {
    const d = defaultSettings();
    const s = scenario({
      horizon: 999,
      settings: { ...d, alpha: 3, window: 1, enabled: { ...d.enabled, ghost: true }, params: { ...d.params, ghost: { x: 1 } } },
    });
    const back = await decodeScenario(await encodeScenario(s), d);
    expect(back.horizon).toBe(64);
    expect(back.settings.alpha).toBe(0.5);
    expect(back.settings.window).toBe(10);
    expect("ghost" in back.settings.enabled).toBe(false);
    expect("ghost" in back.settings.params).toBe(false);
  });

  it("reject garbage", async () => {
    await expect(decodeScenario("1zAAAA", defaultSettings())).rejects.toThrow();
    await expect(decodeScenario("9zAAAA", defaultSettings())).rejects.toThrow(/newer version/);
  });

  it("are found in the hash", () => {
    expect(scenarioFromHash("#s=1zAbc-_9")).toBe("1zAbc-_9");
    expect(scenarioFromHash("#foo=1&s=1pXYZ")).toBe("1pXYZ");
    expect(scenarioFromHash("#nothing")).toBeNull();
    expect(scenarioFromHash("")).toBeNull();
  });
});
