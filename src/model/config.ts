/**
 * Timestral checkpoint configuration (`config.json` on the Hugging Face Hub).
 *
 * Mirrors `timestral.config.TimestralConfig` from the official inference package
 * (https://github.com/mistralai/timestral), including its defaults.
 */

export type LayerKind = "time" | "variate" | "time_variate" | "variate_time";

/** Ordered attention axes each layer kind runs before its MLP; true = time axis. */
export const LAYER_KIND_AXES: Record<LayerKind, boolean[]> = {
  time: [true],
  variate: [false],
  time_variate: [true, false],
  variate_time: [false, true],
};

export interface TimestralConfig {
  patch_size: number;
  d_model: number;
  num_heads: number;
  mlp_hidden_dim: number;
  block_schedule: LayerKind[];
  quantile_knots: number[];
  block_size: number;
  context_length: number;
  rope_theta: number;
  rope_partial_factor: number;
  norm_eps: number;
  attn_bias: boolean;
  mlp_bias: boolean;
  time_attn_gate: "none" | "sdpa_sigmoid";
  variate_attn_gate: "none" | "sdpa_sigmoid";
  minimum_scale: number;
  use_asinh: boolean;
  stabilize_with_global: boolean;
  covariate_enabled: boolean;
  split_context_horizon_attn: boolean;
  attend_missing_patches: boolean;
  quantile_width_factor: number;
}

const DEFAULTS = {
  rope_theta: 10000.0,
  rope_partial_factor: 0.5,
  norm_eps: 1e-6,
  attn_bias: false,
  mlp_bias: false,
  time_attn_gate: "sdpa_sigmoid",
  variate_attn_gate: "sdpa_sigmoid",
  minimum_scale: 1e-6,
  use_asinh: true,
  stabilize_with_global: false,
  covariate_enabled: true,
  split_context_horizon_attn: true,
  attend_missing_patches: true,
  quantile_width_factor: 2.0,
} as const;

export function parseConfig(raw: Record<string, unknown>): TimestralConfig {
  const c = { ...DEFAULTS, ...raw } as unknown as TimestralConfig;
  for (const key of ["patch_size", "d_model", "num_heads", "mlp_hidden_dim", "block_schedule", "quantile_knots", "block_size", "context_length"]) {
    if (!(key in raw)) throw new Error(`config.json is missing "${key}"`);
  }
  if (c.d_model % c.num_heads !== 0) throw new Error("d_model must be divisible by num_heads");
  for (const k of c.block_schedule) {
    if (!(k in LAYER_KIND_AXES)) throw new Error(`unknown layer kind ${k}`);
  }
  if (!c.quantile_knots.includes(0.5)) throw new Error("quantile_knots must contain 0.5");
  if (c.stabilize_with_global) throw new Error("stabilize_with_global=true is not supported (the refine frame needs it off)");
  return c;
}

export const headDim = (c: TimestralConfig) => c.d_model / c.num_heads;
export const ropeDim = (c: TimestralConfig) => Math.trunc(headDim(c) * c.rope_partial_factor);
export const medianIndex = (c: TimestralConfig) => c.quantile_knots.indexOf(0.5);
