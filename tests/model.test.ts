import { describe, expect, it } from "vitest";
import { LAYER_KIND_AXES, type LayerKind, parseConfig } from "../src/model/config";
import { parseSafetensors, writeSafetensors, type Tensor } from "../src/model/safetensors";
import { causalPatchLocScale } from "../src/model/scaler";
import { TimestralForecaster, TimestralModel, monotoneQuantiles } from "../src/model/timestral";

const KNOTS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function randomModel(seed = 1): TimestralModel {
  const r = rng(seed);
  const randn = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const schedule: LayerKind[] = ["time_variate", "time", "variate_time"];
  const raw = {
    patch_size: 32, d_model: 16, num_heads: 2, mlp_hidden_dim: 24, block_schedule: schedule,
    quantile_knots: KNOTS, block_size: 4, context_length: 512,
  };
  const c = parseConfig(raw);
  const D = c.d_model, ps = c.patch_size, Q = KNOTS.length;
  const w: Record<string, Tensor> = {};
  const lin = (name: string, out: number, inp: number, bias: boolean, s = 1 / Math.sqrt(inp)) => {
    w[`${name}.weight`] = { shape: [out, inp], data: Float32Array.from({ length: out * inp }, () => randn() * s) };
    if (bias) w[`${name}.bias`] = { shape: [out], data: Float32Array.from({ length: out }, () => randn() * 0.1) };
  };
  const res = (p: string, i: number, h: number, o: number, b: boolean) => {
    lin(`${p}.hidden_layer`, h, i, b);
    lin(`${p}.output_layer`, o, h, b);
    lin(`${p}.residual_layer`, o, i, b);
  };
  res("representation.patch_proj", 2 * ps, 4 * D, D, true);
  w["representation.forecast_query"] = { shape: [D], data: Float32Array.from({ length: D }, () => randn() * 0.5) };
  schedule.forEach((kind, li) => {
    LAYER_KIND_AXES[kind].forEach((_, j) => {
      w[`transformer.layers.${li}.attn_norms.${j}.scale`] = { shape: [D], data: Float32Array.from({ length: D }, () => 1 + 0.2 * randn()) };
      lin(`transformer.layers.${li}.attns.${j}.wQKV`, 3 * D, D, false);
      lin(`transformer.layers.${li}.attns.${j}.wO`, D, D, false);
      lin(`transformer.layers.${li}.attns.${j}._gate`, D, D, false);
    });
    w[`transformer.layers.${li}.mlp_norm.scale`] = { shape: [D], data: Float32Array.from({ length: D }, () => 1 + 0.2 * randn()) };
    lin(`transformer.layers.${li}.mlp.w12`, 2 * c.mlp_hidden_dim, D, false);
    lin(`transformer.layers.${li}.mlp.w3`, D, c.mlp_hidden_dim, false);
  });
  w["transformer.out_norm.scale"] = { shape: [D], data: new Float32Array(D).fill(1) };
  res("head.projection", D, 4 * D, ps * Q, false);
  // Round-trip through the file format, as in the browser.
  return new TimestralModel(c, parseSafetensors(writeSafetensors(w)));
}

const series = (n: number, seed = 3) => {
  const r = rng(seed);
  return Array.from({ length: n }, (_, t) => 20 + 5 * Math.sin(t / 6) + (r() - 0.5) * 2);
};

describe("Timestral TypeScript port", () => {
  const fc = new TimestralForecaster(randomModel());

  it("returns sorted, finite quantiles for every horizon step", () => {
    for (const [T, H] of [[5, 1], [70, 32], [100, 64], [200, 150]]) {
      const out = fc.forecast(series(T), H);
      expect(out.quantiles.length).toBe(H * KNOTS.length);
      for (let t = 0; t < H; t++) {
        for (let i = 0; i < KNOTS.length; i++) {
          const v = out.quantiles[t * KNOTS.length + i];
          expect(Number.isFinite(v)).toBe(true);
          if (i > 0) expect(v).toBeGreaterThanOrEqual(out.quantiles[t * KNOTS.length + i - 1] - 1e-5);
        }
      }
    }
  });

  it("is affine-equivariant (causal scaling makes the network see the same input)", () => {
    const y = series(150);
    const a = 3.5, b = -40;
    const base = fc.forecast(y, 70, { clipNonnegative: false });
    const moved = fc.forecast(y.map((v) => a * v + b), 70, { clipNonnegative: false });
    let maxRel = 0;
    for (let i = 0; i < base.quantiles.length; i++) {
      const expected = a * base.quantiles[i] + b;
      maxRel = Math.max(maxRel, Math.abs(moved.quantiles[i] - expected) / (Math.abs(expected) + 1));
    }
    expect(maxRel).toBeLessThan(1e-3);
  });

  it("symmetric averaging makes the forecast odd: f(-y) = -reverse(f(y))", () => {
    const y = series(90).map((v) => v - 20);
    const p = fc.forecast(y, 40, { clipNonnegative: false });
    const n = fc.forecast(y.map((v) => -v), 40, { clipNonnegative: false });
    const Q = KNOTS.length;
    for (let t = 0; t < 40; t++) for (let i = 0; i < Q; i++) expect(n.quantiles[t * Q + i]).toBeCloseTo(-p.quantiles[t * Q + (Q - 1 - i)], 3);
  });

  it("clips at zero for non-negative contexts", () => {
    const y = Array.from({ length: 80 }, (_, t) => (t % 10 === 0 ? 5 : 0.01));
    const out = fc.forecast(y, 32);
    for (const v of out.quantiles) expect(v).toBeGreaterThanOrEqual(0);
  });

  it("handles missing values", () => {
    const y = series(120);
    y[3] = NaN;
    y[60] = NaN;
    y[61] = NaN;
    const out = fc.forecast(y, 10);
    for (const v of out.quantiles) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("scaler", () => {
  it("matches a direct causal mean/std at patch ends", () => {
    const y = Float32Array.from(series(96));
    const obs = new Uint8Array(96).fill(1);
    const { loc, scale } = causalPatchLocScale(y, obs, 32, 1e-6);
    for (const end of [31, 63, 95]) {
      const xs = Array.from(y.slice(0, end + 1));
      const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
      expect(loc[end]).toBeCloseTo(m, 4);
      expect(scale[end]).toBeCloseTo(sd, 4);
      expect(loc[end - 31]).toBe(loc[end]);
    }
  });
});

describe("monotone quantile head", () => {
  it("never crosses and keeps the median channel", () => {
    const raw = Float32Array.from({ length: 21 * 3 }, (_, i) => Math.sin(i) * 3);
    const q = monotoneQuantiles(raw, 3, KNOTS, 2, 21);
    for (let r = 0; r < 3; r++) {
      expect(q[r * 21 + 10]).toBeCloseTo(raw[r * 21 + 10], 6);
      for (let i = 1; i < 21; i++) expect(q[r * 21 + i]).toBeGreaterThanOrEqual(q[r * 21 + i - 1]);
    }
  });
});
