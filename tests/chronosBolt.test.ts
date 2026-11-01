import { describe, expect, it } from "vitest";
import { TIMESTRAL_LEVELS, clipNonnegative, normInv, toQuantileGrid } from "../src/model/backends";
import { ChronosBoltForecaster, ChronosBoltModel, isChronosBoltConfig, parseChronosBoltConfig, relativePositionBucket } from "../src/model/chronosBolt";
import { parseSafetensors, writeSafetensors, type Tensor } from "../src/model/safetensors";

const BOLT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const RAW_CONFIG = {
  architectures: ["ChronosBoltModelForForecasting"],
  d_model: 16, d_kv: 4, d_ff: 32, num_layers: 2, num_decoder_layers: 1, num_heads: 4,
  relative_attention_num_buckets: 32, relative_attention_max_distance: 128, feed_forward_proj: "relu",
  layer_norm_epsilon: 1e-6, decoder_start_token_id: 0,
  chronos_config: { context_length: 128, prediction_length: 16, input_patch_size: 16, input_patch_stride: 16, quantiles: BOLT_LEVELS, use_reg_token: true },
};

function randomModel(seed = 1): ChronosBoltModel {
  const r = rng(seed);
  const randn = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const c = parseChronosBoltConfig(RAW_CONFIG);
  const D = c.d_model, inner = c.num_heads * c.d_kv, Q = c.quantiles.length;
  const w: Record<string, Tensor> = {};
  const mat = (name: string, out: number, inp: number, s = 1 / Math.sqrt(inp)) => {
    w[name] = { shape: [out, inp], data: Float32Array.from({ length: out * inp }, () => randn() * s) };
  };
  const vec = (name: string, n: number, f: () => number) => (w[name] = { shape: [n], data: Float32Array.from({ length: n }, f) });
  const res = (p: string, i: number, h: number, o: number) => {
    for (const [n, a, b] of [["hidden_layer", h, i], ["output_layer", o, h], ["residual_layer", o, i]] as const) {
      mat(`${p}.${n}.weight`, a, b);
      vec(`${p}.${n}.bias`, a, () => 0.1 * randn());
    }
  };
  const attn = (p: string) => ["q", "k", "v", "o"].forEach((k) => (k === "o" ? mat(`${p}.o.weight`, D, inner) : mat(`${p}.${k}.weight`, inner, D)));
  const norm = (name: string) => vec(name, D, () => 1 + 0.2 * randn());
  mat("shared.weight", 2, D, 1);
  res("input_patch_embedding", 2 * c.input_patch_size, c.d_ff, D);
  res("output_patch_embedding", D, c.d_ff, Q * c.prediction_length);
  for (let i = 0; i < c.num_layers; i++) {
    attn(`encoder.block.${i}.layer.0.SelfAttention`);
    norm(`encoder.block.${i}.layer.0.layer_norm.weight`);
    mat(`encoder.block.${i}.layer.1.DenseReluDense.wi.weight`, c.d_ff, D);
    mat(`encoder.block.${i}.layer.1.DenseReluDense.wo.weight`, D, c.d_ff);
    norm(`encoder.block.${i}.layer.1.layer_norm.weight`);
  }
  mat("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", c.relative_attention_num_buckets, c.num_heads, 0.5);
  norm("encoder.final_layer_norm.weight");
  for (let i = 0; i < c.num_decoder_layers; i++) {
    attn(`decoder.block.${i}.layer.0.SelfAttention`);
    norm(`decoder.block.${i}.layer.0.layer_norm.weight`);
    attn(`decoder.block.${i}.layer.1.EncDecAttention`);
    norm(`decoder.block.${i}.layer.1.layer_norm.weight`);
    mat(`decoder.block.${i}.layer.2.DenseReluDense.wi.weight`, c.d_ff, D);
    mat(`decoder.block.${i}.layer.2.DenseReluDense.wo.weight`, D, c.d_ff);
    norm(`decoder.block.${i}.layer.2.layer_norm.weight`);
  }
  norm("decoder.final_layer_norm.weight");
  return new ChronosBoltModel(c, parseSafetensors(writeSafetensors(w)));
}

const series = (n: number, seed = 3) => {
  const r = rng(seed);
  return Array.from({ length: n }, (_, t) => 20 + 5 * Math.sin(t / 6) + (r() - 0.5) * 2);
};

describe("Chronos-Bolt TypeScript port", () => {
  const fc = new ChronosBoltForecaster(randomModel());

  it("recognises Chronos-Bolt configs and rejects Timestral ones", () => {
    expect(isChronosBoltConfig(RAW_CONFIG)).toBe(true);
    expect(isChronosBoltConfig({ patch_size: 32, d_model: 16, quantile_knots: TIMESTRAL_LEVELS })).toBe(false);
  });

  it("matches T5's relative position buckets", () => {
    // Reference values from transformers' T5Attention._relative_position_bucket.
    const rel = [-200, -129, -128, -64, -20, -9, -8, -7, -1, 0, 1, 7, 8, 9, 20, 64, 127, 128, 500];
    const bi = [15, 15, 15, 14, 10, 8, 8, 7, 1, 0, 17, 23, 24, 24, 26, 30, 31, 31, 31];
    const uni = [31, 31, 31, 26, 17, 9, 8, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    expect(rel.map((r) => relativePositionBucket(r, true, 32, 128))).toEqual(bi);
    expect(rel.map((r) => relativePositionBucket(r, false, 32, 128))).toEqual(uni);
  });

  it("returns finite quantiles for short, long and multi-block horizons", () => {
    for (const [T, H] of [[3, 1], [40, 16], [100, 50], [300, 20]]) {
      const out = fc.forecast(series(T), H);
      expect(out.levels).toEqual(BOLT_LEVELS);
      expect(out.quantiles.length).toBe(H * 9);
      for (const v of out.quantiles) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("is affine-equivariant (instance normalisation)", () => {
    const y = series(90);
    const a = 3.5, b = -40;
    const base = fc.forecast(y, 20).quantiles;
    const moved = fc.forecast(y.map((v) => a * v + b), 20).quantiles;
    for (let i = 0; i < base.length; i++) expect(moved[i]).toBeCloseTo(a * base[i] + b, 2);
  });

  it("caps the context at maxContext", () => {
    const y = series(200);
    const capped = fc.forecast(y, 8, { maxContext: 64 }).quantiles;
    const tail = fc.forecast(y.slice(-64), 8).quantiles;
    expect(Array.from(capped)).toEqual(Array.from(tail));
  });

  it("handles missing values and a constant series", () => {
    const y = series(80);
    for (let t = 10; t < 30; t++) y[t] = NaN;
    for (const v of fc.forecast(y, 16).quantiles) expect(Number.isFinite(v)).toBe(true);
    for (const v of fc.forecast(new Array(40).fill(5), 4).quantiles) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("mapping onto the 21-level grid", () => {
  it("is exact inside the range and Gaussian in the tails", () => {
    // A Gaussian forecast is reproduced exactly at every grid level, tails included.
    const mu = 3, sd = 2;
    const q = Float32Array.from(BOLT_LEVELS, (p) => mu + sd * normInv(p));
    const g = toQuantileGrid(q, 1, BOLT_LEVELS);
    expect(g.length).toBe(TIMESTRAL_LEVELS.length);
    TIMESTRAL_LEVELS.forEach((p, i) => {
      // Between knots the interpolation is linear, so allow its error away from the knots.
      const exact = mu + sd * normInv(p);
      expect(Math.abs(g[i] - exact)).toBeLessThan(BOLT_LEVELS.some((l) => Math.abs(l - p) < 1e-9) || p < 0.1 || p > 0.9 ? 1e-5 : 0.1);
    });
  });

  it("fits each tail separately for skewed forecasts and stays sorted", () => {
    const q = Float32Array.from([-1, -0.6, -0.3, -0.1, 0, 0.5, 1.2, 2, 4]);
    const g = toQuantileGrid(q, 1, BOLT_LEVELS);
    for (let i = 1; i < g.length; i++) expect(g[i]).toBeGreaterThanOrEqual(g[i - 1]);
    const i05 = TIMESTRAL_LEVELS.indexOf(0.05), i95 = TIMESTRAL_LEVELS.indexOf(0.95);
    expect(q[0] - g[i05]).toBeCloseTo((0 - -1) * (normInv(0.1) - normInv(0.05)) / -normInv(0.1), 5);
    expect(g[i95] - q[8]).toBeCloseTo((4 - 0) * (normInv(0.95) - normInv(0.9)) / normInv(0.9), 5);
  });

  it("sorts crossing quantiles and leaves on-grid results alone", () => {
    const q = Float32Array.from([0, 2, 1, 3, 4, 5, 6, 7, 8]);
    const g = toQuantileGrid(q, 1, BOLT_LEVELS);
    for (let i = 1; i < g.length; i++) expect(g[i]).toBeGreaterThanOrEqual(g[i - 1]);
    const on = new Float32Array(21).map((_, i) => i);
    expect(toQuantileGrid(on, 1, TIMESTRAL_LEVELS)).toBe(on);
  });

  it("clips at zero only for non-negative contexts", () => {
    const q = Float32Array.from([-1, 0.5, 2]);
    clipNonnegative(q, [1, 2, NaN, 0]);
    expect(Array.from(q)).toEqual([0, 0.5, 2]);
    const r = Float32Array.from([-1, 0.5]);
    clipNonnegative(r, [1, -2]);
    expect(Array.from(r)).toEqual([-1, 0.5]);
  });
});
