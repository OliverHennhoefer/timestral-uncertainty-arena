import { describe, expect, it } from "vitest";
import { fitRange, niceStep, niceTicks, zoomRange } from "../src/ui/yaxis";

describe("y axis", () => {
  it("picks 1-2-5 tick steps", () => {
    expect(niceStep(100, 5)).toBe(20);
    expect(niceStep(7, 5)).toBe(1);
    expect(niceStep(0.3, 5)).toBeCloseTo(0.05);
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
  });

  it("fits with a margin, snapped outwards", () => {
    const [lo, hi] = fitRange(-37, 412);
    expect(lo).toBeLessThan(-37);
    expect(hi).toBeGreaterThan(412);
    expect(hi - lo).toBeLessThan(600);
  });

  it("falls back to 0-100 for an empty extent and widens a flat one", () => {
    expect(fitRange(Infinity, -Infinity)).toEqual([0, 100]);
    const [lo, hi] = fitRange(5, 5);
    expect(lo).toBeLessThan(5);
    expect(hi).toBeGreaterThan(5);
  });

  it("zooms around an anchor", () => {
    const [lo, hi] = zoomRange(0, 100, 2, 25);
    expect(hi - lo).toBeCloseTo(200);
    // 25 stays at a quarter of the way up.
    expect((25 - lo) / (hi - lo)).toBeCloseTo(0.25);
  });
});
