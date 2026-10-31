/** Small canvas helpers (HiDPI sizing, theme colors, line charts). */

export function fitCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function withAlpha(hex: string, a: number): string {
  const v = hex.replace("#", "");
  const r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16), b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export interface Trace {
  color: string;
  values: number[];
  width?: number;
  dash?: number[];
}

/** Line chart of several traces sharing an x axis. */
export function lineChart(
  canvas: HTMLCanvasElement,
  xs: number[],
  traces: Trace[],
  opts: { yMin?: number; yMax?: number; floor?: number; target?: number; format?: (v: number) => string } = {},
): void {
  const { ctx, w, h } = fitCanvas(canvas);
  const pad = { l: 40, r: 8, t: 6, b: 18 };
  ctx.clearRect(0, 0, w, h);
  const ink = cssVar("--muted"), grid = cssVar("--grid");
  ctx.font = "11px system-ui, sans-serif";
  if (xs.length < 2) {
    ctx.fillStyle = ink;
    ctx.fillText("waiting for evaluated forecasts…", pad.l, h / 2);
    return;
  }
  let lo = opts.yMin ?? Infinity, hi = opts.yMax ?? -Infinity;
  if (opts.yMin === undefined || opts.yMax === undefined) {
    for (const t of traces) for (const v of t.values) if (Number.isFinite(v)) { if (opts.yMin === undefined) lo = Math.min(lo, v); if (opts.yMax === undefined) hi = Math.max(hi, v); }
    if (opts.target !== undefined) { lo = Math.min(lo, opts.target); hi = Math.max(hi, opts.target); }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const m = (hi - lo) * 0.05;
    if (opts.yMin === undefined) lo -= m;
    if (opts.yMax === undefined) hi += m;
    if (opts.floor !== undefined) lo = Math.max(lo, opts.floor);
  }
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const X = (x: number) => pad.l + ((x - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
  const Y = (y: number) => pad.t + (1 - (y - lo) / (hi - lo)) * (h - pad.t - pad.b);
  ctx.strokeStyle = grid;
  ctx.fillStyle = ink;
  ctx.lineWidth = 1;
  const fmt = opts.format ?? ((v: number) => v.toFixed(2));
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(v));
    ctx.lineTo(w - pad.r, Y(v));
    ctx.stroke();
    ctx.fillText(fmt(v), 2, Y(v) + 4);
  }
  ctx.fillText(String(x0), pad.l, h - 4);
  const lab = String(x1);
  ctx.fillText(lab, w - pad.r - ctx.measureText(lab).width, h - 4);
  if (opts.target !== undefined) {
    ctx.strokeStyle = cssVar("--ink");
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(opts.target));
    ctx.lineTo(w - pad.r, Y(opts.target));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const t of traces) {
    ctx.strokeStyle = t.color;
    ctx.lineWidth = t.width ?? 1.5;
    ctx.setLineDash(t.dash ?? []);
    ctx.beginPath();
    let started = false;
    const step = Math.max(1, Math.floor(xs.length / (w * 2)));
    for (let i = 0; i < xs.length; i += step) {
      const v = t.values[i];
      if (!Number.isFinite(v)) { started = false; continue; }
      const px = X(xs[i]), py = Y(Math.min(hi, Math.max(lo, v)));
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
