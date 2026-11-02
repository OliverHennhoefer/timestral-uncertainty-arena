/** Calibration charts: the native PIT histogram and the reliability diagram. */

import type { ReliabilityPoint } from "../engine/calibration";
import { FONT, cssVar, fitCanvas, withAlpha } from "./plot";

const PAD = { l: 44, r: 8, t: 10, b: 24 };

function empty(ctx: CanvasRenderingContext2D, h: number) {
  ctx.fillStyle = cssVar("--muted");
  ctx.fillText("Awaiting forecasts", PAD.l, h / 2);
}

/** Hover text per region, kept on the canvas so the listener can read the latest state. */
type Hit = { x0: number; x1: number; y0: number; y1: number; tip: string };
const hits = new WeakMap<HTMLCanvasElement, Hit[]>();

function bindHover(canvas: HTMLCanvasElement) {
  if (hits.has(canvas)) return;
  hits.set(canvas, []);
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const hit = hits.get(canvas)!.find((h) => x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1);
    canvas.title = hit?.tip ?? "";
  });
}

/**
 * Density histogram of PIT values. The flat line is a calibrated forecaster; the shaded band
 * is a rough 95% range for a bin under that ideal (it treats pairs as independent, which
 * overlapping forecasts are not, so read it as a guide).
 */
export function pitChart(canvas: HTMLCanvasElement, counts: number[], color: string): void {
  bindHover(canvas);
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;
  const n = counts.reduce((a, b) => a + b, 0);
  const regions: Hit[] = [];
  hits.set(canvas, regions);
  if (n < 2) return empty(ctx, h);
  const B = counts.length;
  const dens = counts.map((c) => (c / n) * B);
  const p = 1 / B;
  const half = 1.96 * Math.sqrt((p * (1 - p)) / n) * B;
  const yMax = Math.max(2, 1 + half * 1.15, ...dens.map((d) => d * 1.08));
  const X = (u: number) => PAD.l + u * (w - PAD.l - PAD.r);
  const Y = (d: number) => PAD.t + (1 - d / yMax) * (h - PAD.t - PAD.b);
  const ink = cssVar("--muted"), grid = cssVar("--grid");

  ctx.strokeStyle = grid;
  ctx.fillStyle = ink;
  ctx.lineWidth = 1;
  const step = yMax > 4 ? 1 : 0.5;
  for (let v = 0; v <= yMax + 1e-9; v += step) {
    ctx.beginPath();
    ctx.moveTo(PAD.l, Y(v));
    ctx.lineTo(w - PAD.r, Y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(1), 6, Y(v) + 4);
  }
  // Ideal band and line.
  ctx.fillStyle = withAlpha("#8E8E93", 0.14);
  ctx.fillRect(X(0), Y(1 + half), X(1) - X(0), Y(Math.max(0, 1 - half)) - Y(1 + half));

  const gap = 2;
  for (let i = 0; i < B; i++) {
    const x0 = X(i / B) + gap / 2, x1 = X((i + 1) / B) - gap / 2;
    const top = Y(dens[i]), base = Y(0);
    ctx.fillStyle = withAlpha(color, 0.75);
    const r = Math.min(4, (x1 - x0) / 2, base - top);
    ctx.beginPath();
    ctx.moveTo(x0, base);
    ctx.lineTo(x0, top + r);
    ctx.arcTo(x0, top, x0 + r, top, r);
    ctx.lineTo(x1 - r, top);
    ctx.arcTo(x1, top, x1, top + r, r);
    ctx.lineTo(x1, base);
    ctx.closePath();
    ctx.fill();
    const pct = ((counts[i] / n) * 100).toFixed(1);
    regions.push({ x0, x1, y0: PAD.t, y1: base, tip: `PIT ${(i / B).toFixed(1)}–${((i + 1) / B).toFixed(1)}: ${pct}% of ${n} pairs (calibrated: ${(100 / B).toFixed(0)}%)` });
  }
  ctx.strokeStyle = ink;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(X(0), Y(1));
  ctx.lineTo(X(1), Y(1));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = ink;
  for (const u of [0, 0.5, 1]) {
    const lab = u.toFixed(1);
    const tw = ctx.measureText(lab).width;
    ctx.fillText(lab, Math.min(w - PAD.r - tw, Math.max(PAD.l, X(u) - tw / 2)), h - 5);
  }
}

export interface ReliabilityCurve {
  name: string;
  color: string;
  points: ReliabilityPoint[];
  dash?: number[];
  /** Thicker, drawn last. */
  focus?: boolean;
}

/** Empirical against nominal coverage; the diagonal is perfect calibration. */
export function reliabilityChart(canvas: HTMLCanvasElement, curves: ReliabilityCurve[], current: number): void {
  bindHover(canvas);
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;
  const regions: Hit[] = [];
  hits.set(canvas, regions);
  const live = curves.filter((c) => c.points.some((p) => Number.isFinite(p.empirical)));
  if (!live.length) return empty(ctx, h);
  const X = (u: number) => PAD.l + u * (w - PAD.l - PAD.r);
  const Y = (u: number) => PAD.t + (1 - u) * (h - PAD.t - PAD.b);
  const ink = cssVar("--muted"), grid = cssVar("--grid"), surface = cssVar("--surface") || "#fff";

  ctx.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.strokeStyle = grid;
    ctx.beginPath();
    ctx.moveTo(PAD.l, Y(v));
    ctx.lineTo(w - PAD.r, Y(v));
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.fillText(`${v * 100}%`, 2, Y(v) + 4);
    const lab = `${v * 100}%`;
    const tw = ctx.measureText(lab).width;
    ctx.fillText(lab, Math.min(w - PAD.r - tw, Math.max(PAD.l, X(v) - tw / 2)), h - 5);
  }
  // The current alpha, and the diagonal.
  ctx.strokeStyle = withAlpha("#8E8E93", 0.35);
  ctx.setLineDash([2, 4]);
  ctx.beginPath();
  ctx.moveTo(X(current), Y(0));
  ctx.lineTo(X(current), Y(1));
  ctx.stroke();
  ctx.strokeStyle = ink;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(X(0), Y(0));
  ctx.lineTo(X(1), Y(1));
  ctx.stroke();
  ctx.setLineDash([]);

  const order = [...live.filter((c) => !c.focus), ...live.filter((c) => c.focus)];
  for (const c of order) {
    const pts = c.points.filter((p) => Number.isFinite(p.empirical));
    ctx.strokeStyle = c.color;
    ctx.lineWidth = c.focus ? 2 : 1.5;
    ctx.lineJoin = "round";
    ctx.setLineDash(c.dash ?? []);
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.nominal), Y(p.empirical)) : ctx.moveTo(X(p.nominal), Y(p.empirical))));
    ctx.stroke();
    ctx.setLineDash([]);
    for (const p of pts) {
      const x = X(p.nominal), y = Y(p.empirical);
      const at = Math.abs(p.nominal - current) < 1e-9;
      const r = at ? 4.5 : 3;
      // A surface ring keeps overlapping markers apart.
      ctx.fillStyle = surface;
      ctx.beginPath();
      ctx.arc(x, y, r + 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      regions.push({
        x0: x - 7, x1: x + 7, y0: y - 7, y1: y + 7,
        tip: `${c.name}: ${(p.empirical * 100).toFixed(1)}% covered at ${(p.nominal * 100).toFixed(0)}% nominal (${p.n} pairs)`,
      });
    }
  }
  // Focused curve's markers win the hover when they overlap.
  regions.reverse();
}

/** Largest |empirical - nominal| over a curve, in percentage points. */
export function maxGap(points: ReliabilityPoint[]): number {
  let g = NaN;
  for (const p of points) if (Number.isFinite(p.empirical)) g = Math.max(Number.isFinite(g) ? g : 0, Math.abs(p.empirical - p.nominal));
  return g * 100;
}
