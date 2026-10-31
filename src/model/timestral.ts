/**
 * Pure TypeScript port of Timestral inference for a single univariate series
 * without covariates. It follows `timestral/model.py`, `layers.py` and `forecaster.py`
 * from the official package (https://github.com/mistralai/timestral) and loads the
 * released `config.json` + `model.safetensors` unchanged.
 *
 * Supported: causal per-patch scaling, DHF two-stream decoding, the advancing refine
 * frame, block-autoregressive decoding past one block, the quantile cap, symmetric
 * averaging and non-negative clipping. Not supported: multivariate targets and covariates.
 */

import { type TimestralConfig, LAYER_KIND_AXES, headDim, medianIndex, parseConfig, ropeDim } from "./config";
import { addInPlace, linear, rmsNorm, sigmoid, silu, softplus } from "./ops";
import { type Tensor, parseSafetensors } from "./safetensors";
import { advancingBlockFrames, applyAffineAsinh, causalPatchLocScale, clampNonfinite } from "./scaler";

const F32_MAX = 3.4028234663852886e38;

export interface ForecastOptions {
  symmetricAveraging?: boolean;
  clipNonnegative?: boolean;
  /** Context cap in steps (most recent kept); defaults to the trained context length. */
  maxContext?: number;
  quantileRealCapK?: number;
  extremeValueThreshold?: number;
}

export interface QuantileForecast {
  /** `(H, Q)` row-major quantiles, sorted along Q. */
  quantiles: Float32Array;
  horizon: number;
  levels: number[];
}

interface ResidualBlockW {
  hiddenW: Float32Array; hiddenB?: Float32Array;
  outputW: Float32Array; outputB?: Float32Array;
  residualW: Float32Array; residualB?: Float32Array;
  inDim: number; hiddenDim: number; outDim: number;
}

interface AttentionW {
  isTime: boolean;
  wQKV: Float32Array; bQKV?: Float32Array;
  wO: Float32Array; bO?: Float32Array;
  gateW?: Float32Array; gateB?: Float32Array;
}

interface LayerW {
  attnNorms: Float32Array[];
  attns: AttentionW[];
  mlpNorm: Float32Array;
  w12: Float32Array; b12?: Float32Array;
  w3: Float32Array; b3?: Float32Array;
}

export class TimestralModel {
  readonly config: TimestralConfig;
  readonly numParams: number;
  private readonly patchProj: ResidualBlockW;
  private readonly forecastQuery: Float32Array;
  private readonly layers: LayerW[];
  private readonly outNorm: Float32Array;
  private readonly head: ResidualBlockW;
  private readonly invFreq: Float32Array;

  constructor(config: TimestralConfig, weights: Map<string, Tensor>) {
    this.config = config;
    const c = config;
    let numParams = 0;
    const used = new Set<string>();
    const get = (name: string): Float32Array => {
      const t = weights.get(name);
      if (!t) throw new Error(`missing weight ${name}`);
      used.add(name);
      return t.data;
    };
    const opt = (name: string, present: boolean): Float32Array | undefined => (present ? get(name) : undefined);
    const residual = (prefix: string, inDim: number, hiddenDim: number, outDim: number, bias: boolean): ResidualBlockW => ({
      hiddenW: get(`${prefix}.hidden_layer.weight`), hiddenB: opt(`${prefix}.hidden_layer.bias`, bias),
      outputW: get(`${prefix}.output_layer.weight`), outputB: opt(`${prefix}.output_layer.bias`, bias),
      residualW: get(`${prefix}.residual_layer.weight`), residualB: opt(`${prefix}.residual_layer.bias`, bias),
      inDim, hiddenDim, outDim,
    });
    const D = c.d_model;
    this.patchProj = residual("representation.patch_proj", 2 * c.patch_size, 4 * D, D, true);
    this.forecastQuery = get("representation.forecast_query");
    this.layers = c.block_schedule.map((kind, li) => {
      const p = `transformer.layers.${li}`;
      const axes = LAYER_KIND_AXES[kind];
      return {
        attnNorms: axes.map((_, j) => get(`${p}.attn_norms.${j}.scale`)),
        attns: axes.map((isTime, j) => {
          const gate = (isTime ? c.time_attn_gate : c.variate_attn_gate) === "sdpa_sigmoid";
          return {
            isTime,
            wQKV: get(`${p}.attns.${j}.wQKV.weight`), bQKV: opt(`${p}.attns.${j}.wQKV.bias`, c.attn_bias),
            wO: get(`${p}.attns.${j}.wO.weight`), bO: opt(`${p}.attns.${j}.wO.bias`, c.attn_bias),
            gateW: gate ? get(`${p}.attns.${j}._gate.weight`) : undefined,
            gateB: gate ? opt(`${p}.attns.${j}._gate.bias`, c.attn_bias) : undefined,
          };
        }),
        mlpNorm: get(`${p}.mlp_norm.scale`),
        w12: get(`${p}.mlp.w12.weight`), b12: opt(`${p}.mlp.w12.bias`, c.mlp_bias),
        w3: get(`${p}.mlp.w3.weight`), b3: opt(`${p}.mlp.w3.bias`, c.mlp_bias),
      };
    });
    this.outNorm = get("transformer.out_norm.scale");
    this.head = residual("head.projection", D, 4 * D, c.patch_size * c.quantile_knots.length, false);
    for (const [name, t] of weights) if (used.has(name)) numParams += t.data.length;
    this.numParams = numParams;

    // RoPE inverse frequencies in float32, zero-padded past rope_dim / 2 (identity rotation).
    const hd = headDim(c);
    const rd = ropeDim(c);
    this.invFreq = new Float32Array(hd / 2);
    for (let i = 0; i < rd / 2; i++) {
      const expo = Math.fround((2 * i) / rd);
      this.invFreq[i] = Math.fround(1 / Math.fround(Math.pow(c.rope_theta, expo)));
    }
  }

  static fromBuffers(configJson: Record<string, unknown>, safetensors: ArrayBuffer): TimestralModel {
    return new TimestralModel(parseConfig(configJson), parseSafetensors(safetensors));
  }

  private residualBlock(b: ResidualBlockW, x: Float32Array, nTok: number): Float32Array {
    const h = linear(x, nTok, b.inDim, b.hiddenW, b.hiddenDim, b.hiddenB);
    for (let i = 0; i < h.length; i++) h[i] = silu(h[i]);
    const out = linear(h, nTok, b.hiddenDim, b.outputW, b.outDim, b.outputB);
    addInPlace(out, linear(x, nTok, b.inDim, b.residualW, b.outDim, b.residualB));
    return out;
  }

  /** Time attention over one row: `allowed(i, j)` is the boolean mask, RoPE on q/k. */
  private timeAttention(
    a: AttentionW, xn: Float32Array, N: number, cos: Float32Array, sin: Float32Array, allowed: (i: number, j: number) => boolean,
  ): Float32Array {
    const c = this.config;
    const D = c.d_model, H = c.num_heads, hd = D / H, half = hd / 2;
    const qkv = linear(xn, N, D, a.wQKV, 3 * D, a.bQKV);
    // Fused layout is (qkv, head_dim, n_heads); regroup to (token, head, head_dim).
    const q = new Float32Array(N * D), k = new Float32Array(N * D), v = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      for (let h = 0; h < H; h++) {
        for (let d = 0; d < hd; d++) {
          const src = n * 3 * D + d * H + h;
          const dst = n * D + h * hd + d;
          q[dst] = qkv[src];
          k[dst] = qkv[src + D];
          v[dst] = qkv[src + 2 * D];
        }
      }
    }
    // Rotary embedding (rotate_half convention).
    const rot = (x: Float32Array) => {
      const tmp = new Float32Array(hd);
      for (let n = 0; n < N; n++) {
        for (let h = 0; h < H; h++) {
          const o = n * D + h * hd;
          for (let d = 0; d < hd; d++) {
            const r = d < half ? -x[o + d + half] : x[o + d - half];
            tmp[d] = Math.fround(x[o + d] * cos[n * hd + d]) + Math.fround(r * sin[n * hd + d]);
          }
          x.set(tmp, o);
        }
      }
    };
    rot(q);
    rot(k);
    const out = new Float32Array(N * D);
    const scaleF = 1 / Math.sqrt(hd);
    const scores = new Float64Array(N);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < N; i++) {
        let mx = -Infinity;
        const qo = i * D + h * hd;
        for (let j = 0; j < N; j++) {
          if (!allowed(i, j)) { scores[j] = -Infinity; continue; }
          const ko = j * D + h * hd;
          let s = 0;
          for (let d = 0; d < hd; d++) s += q[qo + d] * k[ko + d];
          s *= scaleF;
          scores[j] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let j = 0; j < N; j++) {
          const e = scores[j] === -Infinity ? 0 : Math.exp(scores[j] - mx);
          scores[j] = e;
          z += e;
        }
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let j = 0; j < N; j++) if (scores[j] !== 0) acc += scores[j] * v[j * D + h * hd + d];
          out[qo + d] = acc / z;
        }
      }
    }
    return this.gateAndProject(a, xn, out, N);
  }

  /** Variate attention with a single row: softmax over one key is 1, so the output is V. */
  private variateAttentionSingleRow(a: AttentionW, xn: Float32Array, N: number): Float32Array {
    const D = this.config.d_model, H = this.config.num_heads, hd = D / H;
    const vRaw = linear(xn, N, D, a.wQKV, D, a.bQKV, 2 * D);
    const out = new Float32Array(N * D);
    for (let n = 0; n < N; n++) for (let h = 0; h < H; h++) for (let d = 0; d < hd; d++) out[n * D + h * hd + d] = vRaw[n * D + d * H + h];
    return this.gateAndProject(a, xn, out, N);
  }

  private gateAndProject(a: AttentionW, xn: Float32Array, out: Float32Array, N: number): Float32Array {
    const D = this.config.d_model;
    if (a.gateW) {
      const g = linear(xn, N, D, a.gateW, D, a.gateB);
      for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] * sigmoid(g[i]));
    }
    return linear(out, N, D, a.wO, D, a.bO);
  }

  /**
   * One forward pass of the no-covariate path: normalized quantiles `(h * patch, Q)` for the
   * horizon patches after a context of `L` patches.
   */
  forwardNormalized(scaled: Float32Array, observed: Uint8Array, horizonPatches: number): Float32Array {
    const c = this.config;
    const ps = c.patch_size, D = c.d_model, Q = c.quantile_knots.length;
    const L = scaled.length / ps, h = horizonPatches, N = L + h;

    // Patch embedding from [values, missing] channels.
    const inp = new Float32Array(L * 2 * ps);
    for (let p = 0; p < L; p++) {
      for (let j = 0; j < ps; j++) {
        inp[p * 2 * ps + j] = scaled[p * ps + j];
        inp[p * 2 * ps + ps + j] = observed[p * ps + j] ? 0 : 1;
      }
    }
    let x: Float32Array = new Float32Array(N * D);
    x.set(this.residualBlock(this.patchProj, inp, L), 0);
    for (let n = L; n < N; n++) x.set(this.forecastQuery, n * D);

    // Masks and positions.
    const obsPerPatch = new Uint8Array(L);
    for (let p = 0; p < L; p++) for (let j = 0; j < ps; j++) if (observed[p * ps + j]) { obsPerPatch[p] = 1; break; }
    const keyValid = new Uint8Array(L);
    let started = 0;
    for (let p = 0; p < L; p++) {
      started = started || obsPerPatch[p];
      keyValid[p] = c.attend_missing_patches ? (obsPerPatch[p] | started) : obsPerPatch[p];
    }
    let firstObs = obsPerPatch.indexOf(1);
    if (firstObs < 0) firstObs = 0;
    const allowed = (i: number, j: number): boolean => {
      if (i === j) return true;
      if (i < L) return j < i && keyValid[j] === 1;
      return j < L ? keyValid[j] === 1 : j <= i;
    };

    const hd = headDim(c), halfHd = hd / 2;
    const cos = new Float32Array(N * hd), sin = new Float32Array(N * hd);
    for (let n = 0; n < N; n++) {
      const pos = n - firstObs;
      for (let d = 0; d < hd; d++) {
        const f = Math.fround(this.invFreq[d % halfHd] * pos);
        cos[n * hd + d] = Math.cos(f);
        sin[n * hd + d] = Math.sin(f);
      }
    }

    for (const layer of this.layers) {
      layer.attns.forEach((a, j) => {
        const xn = rmsNorm(x, N, D, layer.attnNorms[j], c.norm_eps);
        const y = a.isTime ? this.timeAttention(a, xn, N, cos, sin, allowed) : this.variateAttentionSingleRow(a, xn, N);
        addInPlace(x, y);
      });
      const xn = rmsNorm(x, N, D, layer.mlpNorm, c.norm_eps);
      const Hd = c.mlp_hidden_dim;
      const gu = linear(xn, N, D, layer.w12, 2 * Hd, layer.b12);
      const act = new Float32Array(N * Hd);
      for (let n = 0; n < N; n++) for (let i = 0; i < Hd; i++) act[n * Hd + i] = Math.fround(gu[n * 2 * Hd + i] * Math.fround(silu(gu[n * 2 * Hd + Hd + i])));
      addInPlace(x, linear(act, N, Hd, layer.w3, D, layer.b3));
    }
    x = rmsNorm(x, N, D, this.outNorm, c.norm_eps);
    const dec = x.subarray(L * D, N * D);
    const raw = this.residualBlock(this.head, dec, h); // (h, ps * Q)
    return monotoneQuantiles(raw, h * ps, c.quantile_knots, c.quantile_width_factor, Q);
  }

  /** `predict_block` for one row: raw-space quantiles `(h * patch, Q)`. */
  predictBlock(context: Float32Array, observed: Uint8Array, horizonPatches: number): Float32Array {
    const c = this.config;
    const { loc, scale } = causalPatchLocScale(context, observed, c.patch_size, c.minimum_scale);
    const scaled = applyAffineAsinh(context, loc, scale, observed, c.use_asinh);
    const preds = this.forwardNormalized(scaled, observed, horizonPatches);
    const loc0 = loc[loc.length - 1], scale0 = scale[scale.length - 1];
    const out = new Float32Array(preds.length);
    for (let i = 0; i < preds.length; i++) {
      const v = c.use_asinh ? Math.fround(Math.sinh(preds[i])) : preds[i];
      out[i] = Math.fround(v * scale0) + loc0;
    }
    clampNonfinite(out, c.quantile_knots.length);
    return out;
  }
}

/** Non-crossing quantiles from raw head outputs (`monotone_cumsum`). `raw` is `(rows, Q)`. */
export function monotoneQuantiles(raw: Float32Array, rows: number, levels: number[], widthFactor: number, Q: number): Float32Array {
  const m = levels.indexOf(0.5);
  const widths = new Array<number>(Q).fill(0);
  for (let i = m + 1; i < Q; i++) widths[i] = Math.fround(levels[i] - levels[i - 1]);
  for (let i = 0; i < m; i++) widths[i] = Math.fround(levels[i + 1] - levels[i]);
  const out = new Float32Array(rows * Q);
  const inc = new Float32Array(Q);
  for (let r = 0; r < rows; r++) {
    const o = r * Q;
    for (let i = 0; i < Q; i++) inc[i] = Math.fround(Math.fround(widthFactor * widths[i]) * Math.fround(softplus(raw[o + i])));
    const median = raw[o + m];
    out[o + m] = median;
    let up = 0;
    for (let i = m + 1; i < Q; i++) {
      up = Math.fround(up + inc[i]);
      out[o + i] = Math.min(Math.max(median + up, -F32_MAX), F32_MAX);
    }
    let down = 0;
    for (let i = m - 1; i >= 0; i--) {
      down = Math.fround(down + inc[i]);
      out[o + i] = Math.min(Math.max(median - down, -F32_MAX), F32_MAX);
    }
  }
  return out;
}

/** `TimestralForecaster.forecast` for one univariate series (NaN = missing). */
export class TimestralForecaster {
  constructor(readonly model: TimestralModel) {}

  get levels(): number[] {
    return this.model.config.quantile_knots;
  }

  forecast(context: ArrayLike<number>, predictionLength: number, opts: ForecastOptions = {}): QuantileForecast {
    if (predictionLength < 1) throw new Error("predictionLength must be >= 1");
    const symmetric = opts.symmetricAveraging ?? true;
    const clip = opts.clipNonnegative ?? true;
    const T = context.length;
    const inputs = new Float32Array(T);
    const mask = new Uint8Array(T);
    for (let t = 0; t < T; t++) {
      const v = Math.fround(context[t]);
      if (Number.isFinite(v)) { inputs[t] = v; mask[t] = 1; }
    }
    const Q = this.levels.length;
    let q = this.forecastRaw(inputs, mask, predictionLength, opts);
    if (symmetric) {
      const neg = inputs.map((v) => -v);
      const mirrored = this.forecastRaw(neg, mask, predictionLength, opts);
      for (let t = 0; t < predictionLength; t++) {
        for (let i = 0; i < Q; i++) q[t * Q + i] = Math.fround(0.5 * Math.fround(q[t * Q + i] - mirrored[t * Q + (Q - 1 - i)]));
      }
    }
    if (clip) {
      let any = false, nonneg = true;
      for (let t = 0; t < T; t++) if (mask[t]) { any = true; if (inputs[t] < 0) nonneg = false; }
      if (any && nonneg) q = q.map((v) => (v < 0 ? 0 : v));
    }
    return { quantiles: q, horizon: predictionLength, levels: this.levels };
  }

  private forecastRaw(inputs: Float32Array, observed: Uint8Array, predictionLength: number, opts: ForecastOptions): Float32Array {
    const c = this.model.config;
    const ps = c.patch_size, Q = c.quantile_knots.length, mi = medianIndex(c);
    const maxContext = opts.maxContext ?? c.context_length;
    // Forward-fill missing values (leading ones become 0), cap and left-pad to a patch multiple.
    let rows = new Float32Array(inputs.length);
    let last = 0, seen = false;
    for (let t = 0; t < inputs.length; t++) {
      if (observed[t]) { last = inputs[t]; seen = true; }
      rows[t] = observed[t] ? inputs[t] : seen ? last : 0;
    }
    let mask = observed;
    if (rows.length > maxContext) {
      rows = rows.slice(rows.length - maxContext);
      mask = mask.slice(mask.length - maxContext);
    }
    if (rows.length % ps !== 0 || rows.length === 0) {
      const pad = ps - (rows.length % ps);
      const r2 = new Float32Array(rows.length + pad);
      r2.set(rows, pad);
      const m2 = new Uint8Array(mask.length + pad);
      m2.set(mask, pad);
      rows = r2;
      mask = m2;
    }
    const capRows = rows, capMask = mask;

    // Block-autoregressive decode.
    const perShot = Math.min(32, c.block_size);
    let remaining = Math.ceil(predictionLength / ps);
    const blocks: Float32Array[] = [];
    let ctx = rows, ctxMask = mask;
    const extreme = opts.extremeValueThreshold ?? 1e30;
    while (remaining > 0) {
      const block = Math.min(perShot, remaining);
      let quantiles = this.model.predictBlock(ctx, ctxMask, block);
      quantiles = this.refineBlock(quantiles, ctx, ctxMask, block);
      blocks.push(quantiles);
      remaining -= block;
      if (remaining <= 0) break;
      const blen = block * ps;
      const nextCtx = new Float32Array(ctx.length + blen);
      nextCtx.set(ctx);
      const nextMask = new Uint8Array(ctxMask.length + blen);
      nextMask.set(ctxMask);
      for (let t = 0; t < blen; t++) {
        const v = quantiles[t * Q + mi];
        nextCtx[ctx.length + t] = Number.isFinite(v) && Math.abs(v) < extreme ? v : 0;
        nextMask[ctx.length + t] = 1;
      }
      ctx = nextCtx;
      ctxMask = nextMask;
      if (ctx.length > maxContext) {
        const keep = Math.max(ps, Math.floor(maxContext / ps) * ps);
        ctx = ctx.slice(ctx.length - keep);
        ctxMask = ctxMask.slice(ctxMask.length - keep);
      }
    }
    const out = new Float32Array(predictionLength * Q);
    let off = 0;
    for (const b of blocks) {
      const n = Math.min(b.length, out.length - off);
      out.set(b.subarray(0, n), off);
      off += n;
      if (off >= out.length) break;
    }

    // Quantile cap around the observed context range.
    const k = opts.quantileRealCapK ?? 1e4;
    if (k > 0) {
      let mx = -Infinity, mn = Infinity;
      for (let t = 0; t < capRows.length; t++) if (capMask[t]) { mx = Math.max(mx, capRows[t]); mn = Math.min(mn, capRows[t]); }
      if (!Number.isFinite(mx)) mx = 0;
      if (!Number.isFinite(mn)) mn = 0;
      const { scale } = causalPatchLocScale(capRows, capMask, ps, c.minimum_scale);
      const a = Math.fround(scale[scale.length - 1] * k);
      const lo = Math.fround(mn - a), hi = Math.fround(mx + a);
      for (let i = 0; i < out.length; i++) out[i] = Math.min(Math.max(out[i], lo), hi);
    }
    for (let i = 0; i < out.length; i++) if (Number.isNaN(out[i])) throw new Error("forecast produced NaN values; the input is degenerate");
    return out;
  }

  /** Move one decoded block from the anchor frame to the advancing refine frame. */
  private refineBlock(quantiles: Float32Array, context: Float32Array, mask: Uint8Array, nPred: number): Float32Array {
    if (nPred <= 1) return quantiles;
    const c = this.model.config;
    const ps = c.patch_size, Q = c.quantile_knots.length, mi = medianIndex(c);
    const { loc, scale } = causalPatchLocScale(context, mask, ps, c.minimum_scale);
    const loc0 = loc[loc.length - 1], scale0 = scale[scale.length - 1];
    const sinhMedian = new Float64Array(nPred * ps);
    for (let t = 0; t < nPred * ps; t++) sinhMedian[t] = (quantiles[t * Q + mi] - loc0) / scale0;
    const { locs, scales } = advancingBlockFrames(context, mask, sinhMedian, nPred, ps, c.minimum_scale);
    const out = new Float32Array(quantiles.length);
    for (let t = 0; t < nPred * ps; t++) {
      const p = Math.floor(t / ps);
      const lj = p === 0 ? loc0 : Math.fround(locs[p]);
      const sj = p === 0 ? scale0 : Math.fround(scales[p]);
      for (let i = 0; i < Q; i++) {
        const norm = Math.fround(Math.fround(quantiles[t * Q + i] - loc0) / scale0);
        out[t * Q + i] = Math.fround(norm * sj) + lj;
      }
    }
    clampNonfinite(out, Q);
    return out;
  }
}
