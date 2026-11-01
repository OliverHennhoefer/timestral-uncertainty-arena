/**
 * Pure TypeScript port of Chronos-Bolt inference (Amazon, Apache-2.0) for a single
 * univariate series. It follows `chronos/chronos_bolt.py` from the official package
 * (https://github.com/amazon-science/chronos-forecasting) and the T5 encoder-decoder it
 * builds on (`transformers.models.t5.modeling_t5`), and loads the released `config.json` +
 * `model.safetensors` from the Hugging Face Hub unchanged.
 *
 * Chronos-Bolt scales the context by its mean and standard deviation, cuts it into patches,
 * runs a T5 encoder over the patch embeddings (plus a [REG] token), decodes one token, and
 * reads `prediction_length` steps of quantiles (0.1, ..., 0.9) off that token. Longer
 * horizons use the official pipeline's heuristic: every quantile path is fed back as
 * context and the pooled quantiles of the next block are taken.
 */

import { addInPlace, linear, rmsNorm } from "./ops";
import { type Tensor, parseSafetensors } from "./safetensors";

export interface ChronosBoltConfig {
  d_model: number;
  d_kv: number;
  d_ff: number;
  num_layers: number;
  num_decoder_layers: number;
  num_heads: number;
  relative_attention_num_buckets: number;
  relative_attention_max_distance: number;
  layer_norm_epsilon: number;
  dense_act_fn: string;
  is_gated_act: boolean;
  decoder_start_token_id: number;
  context_length: number;
  prediction_length: number;
  input_patch_size: number;
  input_patch_stride: number;
  quantiles: number[];
  use_reg_token: boolean;
}

/** True when a Hugging Face `config.json` describes a Chronos-Bolt checkpoint. */
export function isChronosBoltConfig(raw: Record<string, unknown>): boolean {
  const archs = raw.architectures as string[] | undefined;
  return typeof raw.chronos_config === "object" && raw.chronos_config !== null && (!archs || archs.some((a) => a.startsWith("ChronosBolt")));
}

export function parseChronosBoltConfig(raw: Record<string, unknown>): ChronosBoltConfig {
  const cc = raw.chronos_config as Record<string, unknown> | undefined;
  if (!cc) throw new Error("config.json has no chronos_config: not a Chronos-Bolt checkpoint");
  for (const key of ["context_length", "prediction_length", "input_patch_size", "input_patch_stride", "quantiles"]) {
    if (!(key in cc)) throw new Error(`chronos_config is missing "${key}"`);
  }
  // T5Config derives the activation from feed_forward_proj ("relu", "gated-gelu", ...).
  const ffp = String(raw.feed_forward_proj ?? "relu");
  const parts = ffp.split("-");
  const gated = parts[0] === "gated";
  let act = parts[parts.length - 1];
  if (ffp === "gated-gelu") act = "gelu_new";
  const num = (k: string, d?: number): number => {
    const v = raw[k] ?? d;
    if (typeof v !== "number") throw new Error(`config.json is missing "${k}"`);
    return v;
  };
  const c: ChronosBoltConfig = {
    d_model: num("d_model"),
    d_kv: num("d_kv"),
    d_ff: num("d_ff"),
    num_layers: num("num_layers"),
    num_decoder_layers: num("num_decoder_layers", num("num_layers")),
    num_heads: num("num_heads"),
    relative_attention_num_buckets: num("relative_attention_num_buckets", 32),
    relative_attention_max_distance: num("relative_attention_max_distance", 128),
    layer_norm_epsilon: num("layer_norm_epsilon", 1e-6),
    dense_act_fn: typeof raw.dense_act_fn === "string" ? raw.dense_act_fn : act,
    is_gated_act: typeof raw.is_gated_act === "boolean" ? raw.is_gated_act : gated,
    decoder_start_token_id: num("decoder_start_token_id", 0),
    context_length: cc.context_length as number,
    prediction_length: cc.prediction_length as number,
    input_patch_size: cc.input_patch_size as number,
    input_patch_stride: cc.input_patch_stride as number,
    quantiles: (cc.quantiles as number[]).slice(),
    use_reg_token: Boolean(cc.use_reg_token ?? false),
  };
  if (!ACTS[c.dense_act_fn]) throw new Error(`unsupported activation ${c.dense_act_fn}`);
  for (let i = 1; i < c.quantiles.length; i++) if (!(c.quantiles[i] > c.quantiles[i - 1])) throw new Error("quantiles must be increasing");
  return c;
}

/** Abramowitz and Stegun 7.1.26 (|error| < 1.5e-7), enough for float32 parity. */
function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

const ACTS: Record<string, (v: number) => number> = {
  relu: (v) => (v > 0 ? v : 0),
  gelu: (v) => 0.5 * v * (1 + erf(v / Math.SQRT2)),
  gelu_new: (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v))),
  gelu_pytorch_tanh: (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v))),
  silu: (v) => v / (1 + Math.exp(-v)),
  swish: (v) => v / (1 + Math.exp(-v)),
};

interface ResidualBlockW {
  hiddenW: Float32Array; hiddenB?: Float32Array;
  outputW: Float32Array; outputB?: Float32Array;
  residualW: Float32Array; residualB?: Float32Array;
  inDim: number; hiddenDim: number; outDim: number;
}

interface AttnW { q: Float32Array; k: Float32Array; v: Float32Array; o: Float32Array }

interface FfW { wi?: Float32Array; wi0?: Float32Array; wi1?: Float32Array; wo: Float32Array }

interface EncLayerW { saNorm: Float32Array; sa: AttnW; ffNorm: Float32Array; ff: FfW }

interface DecLayerW { saNorm: Float32Array; saV: Float32Array; saO: Float32Array; caNorm: Float32Array; ca: AttnW; ffNorm: Float32Array; ff: FfW }

/**
 * T5 relative position bucket (`T5Attention._relative_position_bucket`), for the encoder's
 * bidirectional self-attention. `rel` is key position minus query position.
 */
export function relativePositionBucket(rel: number, bidirectional: boolean, numBuckets: number, maxDistance: number): number {
  let bucket = 0;
  let nb = numBuckets;
  let r: number;
  if (bidirectional) {
    nb = Math.trunc(nb / 2);
    if (rel > 0) bucket += nb;
    r = Math.abs(rel);
  } else {
    r = -Math.min(rel, 0);
  }
  const maxExact = Math.trunc(nb / 2);
  if (r < maxExact) return bucket + r;
  // Float32 log, as torch does it on `relative_position.float()`.
  const large = maxExact + Math.trunc(Math.fround(Math.fround(Math.fround(Math.log(Math.fround(r / maxExact))) / Math.fround(Math.log(maxDistance / maxExact))) * (nb - maxExact)));
  return bucket + Math.min(large, nb - 1);
}

export class ChronosBoltModel {
  readonly config: ChronosBoltConfig;
  readonly numParams: number;
  private readonly shared: Float32Array;
  private readonly inEmb: ResidualBlockW;
  private readonly outEmb: ResidualBlockW;
  private readonly enc: EncLayerW[];
  private readonly encFinal: Float32Array;
  private readonly relBias: Float32Array;
  private readonly dec: DecLayerW[];
  private readonly decFinal: Float32Array;
  private readonly act: (v: number) => number;

  constructor(config: ChronosBoltConfig, weights: Map<string, Tensor>) {
    this.config = config;
    const c = config;
    this.act = ACTS[c.dense_act_fn];
    const get = (name: string): Float32Array => {
      const t = weights.get(name);
      if (!t) throw new Error(`missing weight ${name}`);
      return t.data;
    };
    const maybe = (name: string): Float32Array | undefined => weights.get(name)?.data;
    const residual = (prefix: string, inDim: number, hiddenDim: number, outDim: number): ResidualBlockW => ({
      hiddenW: get(`${prefix}.hidden_layer.weight`), hiddenB: maybe(`${prefix}.hidden_layer.bias`),
      outputW: get(`${prefix}.output_layer.weight`), outputB: maybe(`${prefix}.output_layer.bias`),
      residualW: get(`${prefix}.residual_layer.weight`), residualB: maybe(`${prefix}.residual_layer.bias`),
      inDim, hiddenDim, outDim,
    });
    const ff = (p: string): FfW =>
      c.is_gated_act
        ? { wi0: get(`${p}.DenseReluDense.wi_0.weight`), wi1: get(`${p}.DenseReluDense.wi_1.weight`), wo: get(`${p}.DenseReluDense.wo.weight`) }
        : { wi: get(`${p}.DenseReluDense.wi.weight`), wo: get(`${p}.DenseReluDense.wo.weight`) };
    const attn = (p: string): AttnW => ({ q: get(`${p}.q.weight`), k: get(`${p}.k.weight`), v: get(`${p}.v.weight`), o: get(`${p}.o.weight`) });

    // The token embedding is tied: newer checkpoints store only `shared.weight`.
    this.shared = maybe("shared.weight") ?? maybe("encoder.embed_tokens.weight") ?? get("decoder.embed_tokens.weight");
    this.inEmb = residual("input_patch_embedding", 2 * c.input_patch_size, c.d_ff, c.d_model);
    this.outEmb = residual("output_patch_embedding", c.d_model, c.d_ff, c.quantiles.length * c.prediction_length);
    this.enc = Array.from({ length: c.num_layers }, (_, i) => {
      const p = `encoder.block.${i}.layer`;
      return { saNorm: get(`${p}.0.layer_norm.weight`), sa: attn(`${p}.0.SelfAttention`), ffNorm: get(`${p}.1.layer_norm.weight`), ff: ff(`${p}.1`) };
    });
    this.relBias = get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
    this.encFinal = get("encoder.final_layer_norm.weight");
    this.dec = Array.from({ length: c.num_decoder_layers }, (_, i) => {
      const p = `decoder.block.${i}.layer`;
      return {
        saNorm: get(`${p}.0.layer_norm.weight`),
        // Self-attention over the single decoder token: softmax over one key is 1, so only
        // the value and output projections matter (and the position bias cancels).
        saV: get(`${p}.0.SelfAttention.v.weight`), saO: get(`${p}.0.SelfAttention.o.weight`),
        caNorm: get(`${p}.1.layer_norm.weight`), ca: attn(`${p}.1.EncDecAttention`),
        ffNorm: get(`${p}.2.layer_norm.weight`), ff: ff(`${p}.2`),
      };
    });
    this.decFinal = get("decoder.final_layer_norm.weight");

    // Count each tensor once: the embed_tokens copies and lm_head are tied or unused.
    const skip = new Set(["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]);
    let n = this.shared.length;
    for (const [name, t] of weights) if (name !== "shared.weight" && !skip.has(name)) n += t.data.length;
    this.numParams = n;
  }

  static fromBuffers(configJson: Record<string, unknown>, safetensors: ArrayBuffer): ChronosBoltModel {
    return new ChronosBoltModel(parseChronosBoltConfig(configJson), parseSafetensors(safetensors));
  }

  private residualBlock(b: ResidualBlockW, x: Float32Array, nTok: number): Float32Array {
    const h = linear(x, nTok, b.inDim, b.hiddenW, b.hiddenDim, b.hiddenB);
    for (let i = 0; i < h.length; i++) h[i] = this.act(h[i]);
    const out = linear(h, nTok, b.hiddenDim, b.outputW, b.outDim, b.outputB);
    addInPlace(out, linear(x, nTok, b.inDim, b.residualW, b.outDim, b.residualB));
    return out;
  }

  private feedForward(f: FfW, xn: Float32Array, N: number): Float32Array {
    const c = this.config;
    let h: Float32Array;
    if (f.wi) {
      h = linear(xn, N, c.d_model, f.wi, c.d_ff);
      for (let i = 0; i < h.length; i++) h[i] = this.act(h[i]);
    } else {
      h = linear(xn, N, c.d_model, f.wi0!, c.d_ff);
      const lin = linear(xn, N, c.d_model, f.wi1!, c.d_ff);
      for (let i = 0; i < h.length; i++) h[i] = Math.fround(this.act(h[i]) * lin[i]);
    }
    return linear(h, N, c.d_ff, f.wo, c.d_model);
  }

  /**
   * T5 attention without scaling: `softmax(q k^T + bias) v`. `bias(h, i, j)` returns the
   * additive position bias, `keyOk[j]` masks keys.
   */
  private attention(a: AttnW, xq: Float32Array, nq: number, xkv: Float32Array, nk: number, keyOk: Uint8Array, bias: ((h: number, i: number, j: number) => number) | null): Float32Array {
    const c = this.config;
    const D = c.d_model, H = c.num_heads, dk = c.d_kv, inner = H * dk;
    const q = linear(xq, nq, D, a.q, inner);
    const k = linear(xkv, nk, D, a.k, inner);
    const v = linear(xkv, nk, D, a.v, inner);
    const out = new Float32Array(nq * inner);
    const scores = new Float64Array(nk);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < nq; i++) {
        const qo = i * inner + h * dk;
        let mx = -Infinity;
        for (let j = 0; j < nk; j++) {
          if (!keyOk[j]) { scores[j] = -Infinity; continue; }
          const ko = j * inner + h * dk;
          let s = 0;
          for (let d = 0; d < dk; d++) s += q[qo + d] * k[ko + d];
          s = Math.fround(s);
          if (bias) s += bias(h, i, j);
          scores[j] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let j = 0; j < nk; j++) {
          const e = scores[j] === -Infinity ? 0 : Math.exp(scores[j] - mx);
          scores[j] = e;
          z += e;
        }
        for (let d = 0; d < dk; d++) {
          let acc = 0;
          for (let j = 0; j < nk; j++) if (scores[j] !== 0) acc += scores[j] * v[j * inner + h * dk + d];
          out[qo + d] = acc / z;
        }
      }
    }
    return linear(out, nq, inner, a.o, D);
  }

  /**
   * One forward pass (`ChronosBoltModelForForecasting.forward`) for one series of raw values
   * (NaN = missing). Returns `(Q, prediction_length)` row-major quantiles in data units.
   */
  predict(context: Float32Array): Float32Array {
    const c = this.config;
    const D = c.d_model, ps = c.input_patch_size, stride = c.input_patch_stride;
    let x0 = context.length > c.context_length ? context.subarray(context.length - c.context_length) : context;

    // Instance norm in float32: nan-mean and nan-std (population) over the context.
    let sum = 0, cnt = 0;
    for (let t = 0; t < x0.length; t++) if (!Number.isNaN(x0[t])) { sum += x0[t]; cnt++; }
    const loc = cnt ? Math.fround(sum / cnt) : 0;
    let ss = 0;
    for (let t = 0; t < x0.length; t++) if (!Number.isNaN(x0[t])) ss += Math.fround(x0[t] - loc) ** 2;
    let scale = cnt ? Math.fround(Math.sqrt(Math.fround(ss / cnt))) : 1;
    if (scale === 0) scale = 1e-5;

    // Left-pad with NaN to a patch multiple, then unfold.
    const pad = x0.length % ps === 0 ? 0 : ps - (x0.length % ps);
    const L = x0.length + pad;
    const nPatch = Math.floor((L - ps) / stride) + 1;
    const vals = new Float32Array(L).fill(NaN);
    for (let t = 0; t < x0.length; t++) vals[pad + t] = Math.fround(Math.fround(x0[t] - loc) / scale);
    const N = nPatch + (c.use_reg_token ? 1 : 0);
    const inp = new Float32Array(nPatch * 2 * ps);
    const keyOk = new Uint8Array(N);
    for (let p = 0; p < nPatch; p++) {
      for (let j = 0; j < ps; j++) {
        const v = vals[p * stride + j];
        const obs = !Number.isNaN(v);
        inp[p * 2 * ps + j] = obs ? v : 0;
        inp[p * 2 * ps + ps + j] = obs ? 1 : 0;
        if (obs) keyOk[p] = 1;
      }
    }
    let x: Float32Array = new Float32Array(N * D);
    x.set(this.residualBlock(this.inEmb, inp, nPatch), 0);
    if (c.use_reg_token) {
      x.set(this.shared.subarray(D, 2 * D), nPatch * D); // reg_token_id = 1
      keyOk[nPatch] = 1;
    }

    // Encoder: the relative position bias of layer 0 is shared by every layer.
    const nb = c.relative_attention_num_buckets, md = c.relative_attention_max_distance, H = c.num_heads;
    const bucketOf = new Int32Array(2 * N - 1);
    for (let r = -(N - 1); r <= N - 1; r++) bucketOf[r + N - 1] = relativePositionBucket(r, true, nb, md);
    const bias = (h: number, i: number, j: number) => this.relBias[bucketOf[j - i + N - 1] * H + h];
    const eps = c.layer_norm_epsilon;
    for (const l of this.enc) {
      const xn = rmsNorm(x, N, D, l.saNorm, eps);
      addInPlace(x, this.attention(l.sa, xn, N, xn, N, keyOk, bias));
      addInPlace(x, this.feedForward(l.ff, rmsNorm(x, N, D, l.ffNorm, eps), N));
    }
    const encOut = rmsNorm(x, N, D, this.encFinal, eps);

    // Decoder: a single token starting from `decoder_start_token_id`.
    const sid = c.decoder_start_token_id;
    let y: Float32Array = this.shared.slice(sid * D, (sid + 1) * D);
    const inner = H * c.d_kv;
    for (const l of this.dec) {
      const v = linear(rmsNorm(y, 1, D, l.saNorm, eps), 1, D, l.saV, inner);
      addInPlace(y, linear(v, 1, inner, l.saO, D));
      addInPlace(y, this.attention(l.ca, rmsNorm(y, 1, D, l.caNorm, eps), 1, encOut, N, keyOk, null));
      addInPlace(y, this.feedForward(l.ff, rmsNorm(y, 1, D, l.ffNorm, eps), 1));
    }
    y = rmsNorm(y, 1, D, this.decFinal, eps);
    const out = this.residualBlock(this.outEmb, y, 1); // (Q * P)
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] * scale) + loc;
    return out;
  }
}

/** `torch.quantile(values, p)` with linear interpolation; sorts `values` in place. */
function empiricalQuantiles(values: Float32Array, levels: number[], out: Float32Array, outOffset: number, outStride: number): void {
  values.sort();
  const n = values.length;
  for (let i = 0; i < levels.length; i++) {
    const pos = levels[i] * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, n - 1);
    const w = pos - lo;
    out[outOffset + i * outStride] = values[lo] + (values[hi] - values[lo]) * w;
  }
}

export interface BoltForecastOptions {
  /** Context cap in steps (most recent kept) before the model sees the series. */
  maxContext?: number;
}

/** `ChronosBoltPipeline.predict` for one univariate series (NaN = missing). */
export class ChronosBoltForecaster {
  constructor(readonly model: ChronosBoltModel) {}

  get levels(): number[] {
    return this.model.config.quantiles;
  }

  /** `(H, Q)` row-major quantiles on the model's own levels (0.1, ..., 0.9 for released models). */
  forecast(context: ArrayLike<number>, predictionLength: number, opts: BoltForecastOptions = {}) {
    if (predictionLength < 1) throw new Error("predictionLength must be >= 1");
    const c = this.model.config;
    const Q = c.quantiles.length, P = c.prediction_length;
    let ctx = Float32Array.from(context as ArrayLike<number>, (v) => (Number.isFinite(v) ? Math.fround(v) : NaN));
    const cap = Math.min(opts.maxContext ?? c.context_length, c.context_length);
    if (ctx.length > cap) ctx = ctx.slice(ctx.length - cap);

    const blocks: Float32Array[] = []; // each (Q, P)
    let pred = this.model.predict(ctx);
    blocks.push(pred);
    let remaining = predictionLength - P;
    if (remaining > 0) {
      // Long-horizon heuristic: one context per quantile path, pooled quantiles per block.
      let ctxs = Array.from({ length: Q }, () => ctx);
      while (remaining > 0) {
        ctxs = ctxs.map((cq, q) => {
          const next = new Float32Array(cq.length + P);
          next.set(cq);
          next.set(pred.subarray(q * P, (q + 1) * P), cq.length);
          return next.length > c.context_length ? next.slice(next.length - c.context_length) : next;
        });
        const preds = ctxs.map((cq) => this.model.predict(cq));
        const pooled = new Float32Array(Q * P);
        const col = new Float32Array(Q * Q);
        for (let t = 0; t < P; t++) {
          for (let a = 0; a < Q; a++) for (let b = 0; b < Q; b++) col[a * Q + b] = preds[a][b * P + t];
          empiricalQuantiles(col, c.quantiles, pooled, t, P);
        }
        pred = pooled;
        blocks.push(pred);
        remaining -= P;
      }
    }
    const quantiles = new Float32Array(predictionLength * Q);
    for (let t = 0; t < predictionLength; t++) {
      const b = blocks[Math.floor(t / P)], k = t % P;
      for (let q = 0; q < Q; q++) quantiles[t * Q + q] = b[q * P + k];
    }
    for (let i = 0; i < quantiles.length; i++) if (!Number.isFinite(quantiles[i])) throw new Error("forecast produced non-finite values; the input is degenerate");
    return { quantiles, horizon: predictionLength, levels: this.levels };
  }
}
