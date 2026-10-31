/** Dense float32 kernels for the browser port. Row-major, tokens-by-features layout. */

/** `y[t, o] = b[o] + sum_i W[o, i] * x[t, i]` with `W` shaped `(out, in)` like `torch.nn.Linear`. */
export function linear(
  x: Float32Array,
  nTok: number,
  inDim: number,
  W: Float32Array,
  outDim: number,
  bias?: Float32Array,
  rowOffset = 0,
): Float32Array {
  const y = new Float32Array(nTok * outDim);
  // Register-blocked over 4 tokens: each weight is loaded once per 4 tokens.
  const t4 = nTok - (nTok % 4);
  for (let o = 0; o < outDim; o++) {
    const wo = (o + rowOffset) * inDim;
    const b = bias ? bias[o + rowOffset] : 0;
    for (let t = 0; t < t4; t += 4) {
      const x0 = t * inDim, x1 = x0 + inDim, x2 = x1 + inDim, x3 = x2 + inDim;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < inDim; i++) {
        const w = W[wo + i];
        s0 += w * x[x0 + i];
        s1 += w * x[x1 + i];
        s2 += w * x[x2 + i];
        s3 += w * x[x3 + i];
      }
      y[t * outDim + o] = s0 + b;
      y[(t + 1) * outDim + o] = s1 + b;
      y[(t + 2) * outDim + o] = s2 + b;
      y[(t + 3) * outDim + o] = s3 + b;
    }
    for (let t = t4; t < nTok; t++) {
      const xo = t * inDim;
      let s0 = 0, s1 = 0;
      let i = 0;
      for (; i + 1 < inDim; i += 2) {
        s0 += W[wo + i] * x[xo + i];
        s1 += W[wo + i + 1] * x[xo + i + 1];
      }
      if (i < inDim) s0 += W[wo + i] * x[xo + i];
      y[t * outDim + o] = s0 + s1 + b;
    }
  }
  return y;
}

export function rmsNorm(x: Float32Array, nTok: number, dim: number, scale: Float32Array, eps: number): Float32Array {
  const y = new Float32Array(x.length);
  for (let t = 0; t < nTok; t++) {
    const o = t * dim;
    let ss = 0;
    for (let i = 0; i < dim; i++) ss += x[o + i] * x[o + i];
    const r = Math.fround(1 / Math.sqrt(Math.fround(ss / dim) + eps));
    for (let i = 0; i < dim; i++) y[o + i] = Math.fround(x[o + i] * r) * scale[i];
  }
  return y;
}

export const silu = (v: number) => v / (1 + Math.exp(-v));
export const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
/** `torch.nn.functional.softplus` with its default threshold of 20. */
export const softplus = (v: number) => (v > 20 ? v : Math.log1p(Math.exp(v)));

export function addInPlace(a: Float32Array, b: Float32Array): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}
