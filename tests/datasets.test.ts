import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { DATASETS, SCALE_HI, SCALE_LO, rescale } from "../src/datasets";
import { PRESETS } from "../src/synthetic";

describe("bundled datasets", () => {
  const readme = readFileSync(new URL("../src/data/README.md", import.meta.url), "utf8");

  it("have unique ids across all presets", () => {
    const ids = [...PRESETS, ...DATASETS].map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  for (const d of DATASETS) {
    it(`${d.id}: finite, within the drawing range, at most the stream length`, () => {
      const y = d.make(2000, 1);
      expect(y.length).toBeGreaterThanOrEqual(100);
      expect(y.length).toBeLessThanOrEqual(2000);
      expect(y.every(Number.isFinite)).toBe(true);
      expect(Math.min(...y)).toBe(SCALE_LO);
      expect(Math.max(...y)).toBe(SCALE_HI);
      expect(d.make(2000, 2)).toEqual(y); // real data ignores the seed
      expect(d.make(50, 1)).toEqual(y.slice(0, 50));
    });

    it(`${d.id}: source and licence are documented`, () => {
      const raw = JSON.parse(readFileSync(new URL(`../src/data/${d.id}.json`, import.meta.url), "utf8"));
      expect(raw.values.length).toBe(raw.n);
      expect(readme).toContain(`\`${d.id}.json\``);
    });
  }

  it("rescale is affine and handles a constant series", () => {
    expect(rescale([2, 4, 6])).toEqual([SCALE_LO, (SCALE_LO + SCALE_HI) / 2, SCALE_HI]);
    expect(rescale([3, 3])).toEqual([50, 50]);
  });
});
