/**
 * Y-axis range for the main chart.
 *
 * "bands" (default) fits every interval on screen (past band, native fan, conformal bands)
 * and never shrinks below the 0-100 canvas. "series" fits only the visible series and the
 * forecast median. "fixed" is the 0-100 drawing canvas every stream is scaled to. "manual"
 * is whatever the user zoomed or panned to.
 */
export type YMode = "fixed" | "series" | "bands" | "manual";

export const FIXED_RANGE: [number, number] = [0, 100];

/** A "nice" tick step (1, 2 or 5 times a power of ten) giving about `target` ticks. */
export function niceStep(span: number, target = 5): number {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** Tick values inside [lo, hi]. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  const step = niceStep(hi - lo, target);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/**
 * Fit [lo, hi] with a small margin, then snap outwards to the tick step so the axis does not
 * twitch on every new point. A degenerate or empty extent falls back to the fixed range.
 */
export function fitRange(lo: number, hi: number, margin = 0.06): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [...FIXED_RANGE];
  if (hi - lo < 1) {
    const mid = (hi + lo) / 2;
    lo = mid - 0.5;
    hi = mid + 0.5;
  }
  const m = (hi - lo) * margin;
  const step = niceStep(hi - lo + 2 * m, 8);
  return [Math.floor((lo - m) / step) * step, Math.ceil((hi + m) / step) * step];
}

/** Zoom [lo, hi] by `factor` (>1 zooms out) keeping `anchor` at the same screen position. */
export function zoomRange(lo: number, hi: number, factor: number, anchor = (lo + hi) / 2): [number, number] {
  const span = Math.min(Math.max((hi - lo) * factor, 1e-3), 1e9);
  const f = (anchor - lo) / (hi - lo);
  return [anchor - f * span, anchor + (1 - f) * span];
}

/** Running min/max over values, ignoring non-finite ones. */
export class Extent {
  lo = Infinity;
  hi = -Infinity;
  add(v: number): void {
    if (!Number.isFinite(v)) return;
    if (v < this.lo) this.lo = v;
    if (v > this.hi) this.hi = v;
  }
}
