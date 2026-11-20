/**
 * Draws one exported frame. The layout is purpose-built for small screens and projectors:
 * large type, thick lines, one method against the model's own interval, a short legend,
 * and the live coverage numbers. Everything is in the layout's logical pixels; the caller
 * scales the context to the output size.
 */

import { niceTicks } from "../ui/yaxis";
import type { Layout, Typography } from "./layouts";
import type { CoverageTrace, FrameState, StatLine } from "./scene";

export type Theme = "light" | "dark";

interface Palette {
  bg: string;
  ink: string;
  muted: string;
  grid: string;
  zone: string;
  native: string;
  miss: string;
}

const PALETTES: Record<Theme, Palette> = {
  light: { bg: "#ffffff", ink: "#1d1d1f", muted: "#6e6e73", grid: "#e8e8ed", zone: "#f5f5f7", native: "#8e8e93", miss: "#ff3b30" },
  dark: { bg: "#111214", ink: "#f5f5f7", muted: "#a1a1a6", grid: "#2a2b30", zone: "#18191c", native: "#a1a1a6", miss: "#ff453a" },
};

const FAMILY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Helvetica Neue", Roboto, system-ui, sans-serif';
const font = (px: number, weight = 400) => `${weight} ${px}px ${FAMILY}`;

export interface Texts {
  title: string;
  subtitle: string;
  /** Short label of the coverage target, e.g. "90%". */
  level: string;
  brand: string;
}

function rgba(hex: string, a: number): string {
  const v = hex.replace("#", "");
  const r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16), b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

/** The native row is named after the model in the subtitle already; "Native" is enough here. */
const shortName = (s: StatLine) => (s.id === "native" ? "Native" : s.name);

const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "–");

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Keeps the y range between frames: it eases towards each frame's fitted range, so a band
 * that widens for a moment does not make the whole chart jump.
 */
export class Painter {
  private y: [number, number] | null = null;
  private cov: [number, number] | null = null;

  constructor(
    readonly layout: Layout,
    readonly theme: Theme,
    readonly texts: Texts,
  ) {}

  /** Forget the eased ranges (e.g. for a still preview). */
  reset(): void {
    this.y = null;
    this.cov = null;
  }

  draw(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, f: FrameState, progress: number, ease = 0.18): void {
    const L = this.layout;
    const T = L.type;
    const S = L.safe;
    const P = PALETTES[this.theme];
    const wide = L.w / L.h > 1.4;
    this.y = this.y ? [this.y[0] + (f.yLo - this.y[0]) * ease, this.y[1] + (f.yHi - this.y[1]) * ease] : [f.yLo, f.yHi];
    const stats = f.stats.map((st) => ({ ...st, name: shortName(st) }));
    const x0 = S.left, cw = L.w - S.left - S.right;

    ctx.save();
    ctx.fillStyle = P.bg;
    ctx.fillRect(0, 0, L.w, L.h);
    ctx.textBaseline = "alphabetic";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    let top = S.top;

    // Title and subtitle; in wide layouts the coverage numbers sit to their right.
    let statW = 0;
    if (wide) {
      ctx.font = font(T.label, 500);
      for (const st of stats) statW = Math.max(statW, ctx.measureText(st.name).width + T.label * 1.4);
      ctx.font = font(T.value, 650);
      statW = Math.max(statW, ctx.measureText("100.0%").width + T.value * 0.4);
    }
    const textW = cw - (wide ? statW * stats.length + 24 : 0);
    ctx.fillStyle = P.ink;
    // A title slightly too long for one line is set a little smaller rather than wrapped.
    let titlePx = T.title;
    ctx.font = font(titlePx, 650);
    const tw = ctx.measureText(this.texts.title).width;
    if (tw > textW && tw * 0.8 <= textW) titlePx = Math.floor((T.title * textW) / tw);
    ctx.font = font(titlePx, 650);
    for (const line of wrap(ctx, this.texts.title, textW, 2)) {
      top += T.title * 0.92;
      ctx.fillText(line, x0, top);
      top += T.title * 0.2;
    }
    ctx.font = font(T.sub);
    ctx.fillStyle = P.muted;
    top += T.sub * 0.2;
    for (const line of wrap(ctx, this.texts.subtitle, textW, L.subLines)) {
      top += T.sub;
      ctx.fillText(line, x0, top);
      top += T.sub * 0.3;
    }

    if (wide) {
      const h = this.statsRow(ctx, P, stats, { x: x0 + cw - statW * stats.length, y: S.top, w: statW * stats.length, h: 0 }, T, true);
      top = Math.max(top, S.top + h);
    } else {
      top += T.sub;
      top += this.statsRow(ctx, P, stats, { x: x0, y: top, w: cw, h: 0 }, T, false);
    }

    // Bottom of the content box: the legend (and brand). Below it, outside the safe box,
    // only the progress bar on the very edge.
    const items = this.legendItems(P, stats);
    ctx.font = font(T.legend);
    const rows = layoutLegend(ctx, items, cw, T.legend);
    const rowH = T.legend * 1.7;
    const brandBelow = !wide;
    const bottom = L.h - S.bottom - (brandBelow ? T.brand * 1.9 : 0);
    const legendTop = bottom - rows.length * rowH;
    const gap = T.tick * 1.2;
    const chartTop = top + gap;

    let covH = 0;
    if (!wide && L.h / L.w > 1.15) covH = Math.round((legendTop - chartTop) * (L.h / L.w > 1.6 ? 0.32 : 0.28));
    const chart: Rect = { x: x0, y: chartTop, w: cw, h: legendTop - gap - chartTop - (covH ? covH + gap : 0) };
    this.mainChart(ctx, P, f, chart, stats);
    if (covH) this.coverageChart(ctx, P, f, { x: x0, y: chart.y + chart.h + gap, w: cw, h: covH }, f.coverage);

    ctx.font = font(T.legend);
    rows.forEach((row, r) => {
      let x = x0;
      const y = legendTop + r * rowH + T.legend;
      for (const it of row) {
        drawKey(ctx, it, x, y - T.legend * 0.35, T.legend);
        ctx.fillStyle = P.muted;
        ctx.fillText(it.label, x + T.legend * 1.6, y);
        x += it.width + T.legend * 1.3;
      }
    });
    ctx.font = font(T.brand, 500);
    ctx.fillStyle = P.muted;
    const brand = this.texts.brand;
    if (!brandBelow) {
      const lastRow = rows[rows.length - 1] ?? [];
      const used = lastRow.reduce((sum, it) => sum + it.width + T.legend * 1.3, 0);
      const bw = ctx.measureText(brand).width;
      if (used + bw + 40 < cw) ctx.fillText(brand, x0 + cw - bw, legendTop + (rows.length - 1) * rowH + T.legend);
    } else {
      ctx.fillText(brand, x0, L.h - S.bottom);
    }

    const barH = Math.round(5 * T.line);
    ctx.fillStyle = rgba(P.muted, 0.18);
    ctx.fillRect(0, L.h - barH, L.w, barH);
    ctx.fillStyle = stats[0]?.color ?? P.ink;
    ctx.fillRect(0, L.h - barH, L.w * Math.max(0, Math.min(1, progress)), barH);
    ctx.restore();
  }

  /** Coverage numbers. Returns the height used. */
  private statsRow(
    ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
    P: Palette,
    stats: StatLine[],
    r: Rect,
    T: Typography,
    alignRight: boolean,
  ): number {
    const labelPx = T.label, valuePx = T.value;
    const colW = r.w / Math.max(1, stats.length);
    const gap = alignRight ? 0 : 24;
    const top = alignRight ? 0 : 26;
    const subPx = Math.round(labelPx * 0.9);
    const height = top + labelPx + valuePx * 1.08 + subPx * 1.35 + (alignRight ? 0 : 24);
    stats.forEach((s, i) => {
      const x0 = r.x + i * colW;
      const cw = colW - (alignRight ? 0 : i < stats.length - 1 ? gap : 0);
      if (!alignRight) {
        ctx.fillStyle = P.zone;
        roundRect(ctx, x0, r.y, cw, height, 22);
        ctx.fill();
      }
      const ix = alignRight ? x0 + colW : x0 + 28;
      const ly = r.y + top + labelPx * 0.9;
      ctx.font = font(labelPx, 500);
      const dotR = labelPx * 0.28;
      const label = ellipsize(ctx, s.name, cw - dotR * 2 - (alignRight ? 20 : 70));
      const lw = ctx.measureText(label).width;
      const lx = alignRight ? ix - lw : ix + dotR * 2 + 10;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(lx - dotR - 10, ly - labelPx * 0.33, dotR, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = P.muted;
      ctx.fillText(label, lx, ly);
      ctx.font = font(valuePx, 650);
      ctx.fillStyle = P.ink;
      const val = pct(s.coverage);
      const vy = ly + valuePx * 1.04;
      const vw = ctx.measureText(val).width;
      ctx.fillText(val, alignRight ? ix - vw : ix, vy);
      ctx.font = font(subPx);
      ctx.fillStyle = P.muted;
      const sub = alignRight || !Number.isFinite(s.width) ? "coverage" : `coverage · width ${s.width.toFixed(1)}`;
      const sw = ctx.measureText(sub).width;
      ctx.fillText(sub, alignRight ? ix - sw : ix, vy + subPx * 1.3);
    });
    return height;
  }

  private legendItems(P: Palette, stats: StatLine[]): LegendItem[] {
    const lvl = this.texts.level;
    const items: LegendItem[] = this.layout.shortLegend
      ? []
      : [
          { kind: "line", color: P.ink, label: "Observed", width: 0 },
          { kind: "dash", color: P.ink, label: "Forecast", width: 0 },
        ];
    const focus = stats[0];
    const native = stats.find((s) => s.id === "native");
    if (focus && focus !== native) items.push({ kind: "fill", color: focus.color, label: `${focus.name} ${lvl}`, width: 0 });
    if (native) items.push({ kind: focus === native ? "fill" : "dashband", color: P.native, label: `${native.name} ${lvl}`, width: 0 });
    items.push({ kind: "dot", color: P.miss, label: "Miss", width: 0 });
    return items;
  }

  private mainChart(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, P: Palette, f: FrameState, r: Rect, stats: StatLine[]): void {
    const [yLo, yHi] = this.y!;
    const tickPx = this.layout.type.tick;
    ctx.font = font(tickPx);
    const gutter = Math.ceil(ctx.measureText("100").width + 18);
    const plot: Rect = { x: r.x + gutter, y: r.y, w: r.w - gutter, h: r.h };
    const span = Math.max(1, f.x1 - f.x0);
    const px = plot.w / span;
    const X = (i: number) => plot.x + (i - f.x0 + 0.5) * px;
    const Y = (v: number) => plot.y + (1 - (v - yLo) / (yHi - yLo)) * plot.h;
    const lw = this.layout.type.line;

    // Grid and y labels.
    const ticks = niceTicks(yLo, yHi, Math.max(3, Math.min(5, Math.round(plot.h / (tickPx * 4.5)))));
    const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    ctx.font = font(tickPx);
    ctx.lineWidth = 1.5;
    for (const v of ticks) {
      const y = Y(v);
      if (y < plot.y - 1 || y > plot.y + plot.h + 1) continue;
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.fillStyle = P.muted;
      const s = v.toFixed(digits);
      ctx.fillText(s, plot.x - 14 - ctx.measureText(s).width, y + tickPx * 0.35);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();

    // Forecast zone to the right of "now".
    const nowX = X(f.t) + px / 2;
    if (f.median.length) {
      ctx.fillStyle = P.zone;
      ctx.fillRect(nowX, plot.y, Math.min(plot.x + plot.w, X(f.origin + f.median.length) + px / 2) - nowX, plot.h);
    }

    const focus = stats[0];
    const focusNative = focus?.id === "native";
    const fcol = focus?.color ?? P.native;

    // Past one-step intervals as columns.
    // One stepped polygon per contiguous stretch, so neighbouring columns leave no seams.
    if (f.past.length) {
      ctx.fillStyle = rgba(fcol, this.theme === "dark" ? 0.2 : 0.18);
      const pts = f.past.filter(([t]) => t >= f.s0);
      let a = 0;
      while (a < pts.length) {
        let b = a;
        while (b + 1 < pts.length && pts[b + 1][0] === pts[b][0] + 1) b++;
        ctx.beginPath();
        for (let i = a; i <= b; i++) {
          const x = X(pts[i][0]);
          ctx.lineTo(x - px / 2, Y(pts[i][2]));
          ctx.lineTo(x + px / 2, Y(pts[i][2]));
        }
        for (let i = b; i >= a; i--) {
          const x = X(pts[i][0]);
          ctx.lineTo(x + px / 2, Y(pts[i][1]));
          ctx.lineTo(x - px / 2, Y(pts[i][1]));
        }
        ctx.closePath();
        ctx.fill();
        a = b + 1;
      }
    }

    // Latest forecast: native interval, focused band, median.
    if (f.median.length) {
      const xs = (j: number) => X(f.origin + 1 + j);
      const fill = (lo: (j: number) => number, hi: (j: number) => number, n: number, col: string) => {
        ctx.fillStyle = col;
        ctx.beginPath();
        // Start at "now" so the fan opens from the last observation.
        ctx.moveTo(X(f.origin), Y(f.series[f.origin]));
        for (let j = 0; j < n; j++) ctx.lineTo(xs(j), Y(hi(j)));
        for (let j = n - 1; j >= 0; j--) ctx.lineTo(xs(j), Y(lo(j)));
        ctx.closePath();
        ctx.fill();
      };
      const H = f.median.length;
      fill((j) => f.nativeLo[j], (j) => f.nativeHi[j], H, rgba(P.native, focusNative ? 0.26 : 0.13));
      ctx.strokeStyle = rgba(P.native, 0.95);
      ctx.lineWidth = (focusNative ? 3.5 : 2.5) * lw;
      ctx.setLineDash(focusNative ? [] : [7 * lw, 7 * lw]);
      for (const arr of [f.nativeLo, f.nativeHi]) {
        ctx.beginPath();
        ctx.moveTo(X(f.origin), Y(f.series[f.origin]));
        for (let j = 0; j < H; j++) ctx.lineTo(xs(j), Y(arr[j]));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (f.band.length && !focusNative) {
        const B = f.band.length;
        fill((j) => f.band[j][0], (j) => f.band[j][1], B, rgba(fcol, this.theme === "dark" ? 0.24 : 0.2));
        ctx.strokeStyle = fcol;
        ctx.lineWidth = 4 * lw;
        for (const side of [0, 1] as const) {
          ctx.beginPath();
          ctx.moveTo(X(f.origin), Y(f.series[f.origin]));
          for (let j = 0; j < B; j++) ctx.lineTo(xs(j), Y(f.band[j][side]));
          ctx.stroke();
        }
      }
      ctx.strokeStyle = rgba(P.ink.startsWith("#") ? P.ink : "#000000", 0.85);
      ctx.lineWidth = 3 * lw;
      ctx.setLineDash([9 * lw, 8 * lw]);
      ctx.beginPath();
      ctx.moveTo(X(f.origin), Y(f.series[f.origin]));
      for (let j = 0; j < H; j++) ctx.lineTo(xs(j), Y(f.median[j]));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Observed series on top.
    ctx.strokeStyle = P.ink;
    ctx.lineWidth = 4 * lw;
    ctx.beginPath();
    for (let i = f.s0; i <= f.t; i++) {
      const x = X(i), y = Y(f.series[i]);
      if (i === f.s0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = P.miss;
    const missR = 4.5 * lw * Math.min(1, Math.max(0.6, px / 3));
    for (const [t, , , missed] of f.past) {
      if (!missed || t < f.s0) continue;
      ctx.beginPath();
      ctx.arc(X(t), Y(f.series[t]), missR, 0, Math.PI * 2);
      ctx.fill();
    }
    // Newest point.
    if (f.t >= 0) {
      ctx.fillStyle = P.ink;
      ctx.beginPath();
      ctx.arc(X(f.t), Y(f.series[f.t]), 5.5 * lw, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    if (f.median.length) {
      ctx.strokeStyle = rgba(P.native, 0.6);
      ctx.lineWidth = 1.5 * lw;
      ctx.setLineDash([3 * lw, 7 * lw]);
      ctx.beginPath();
      ctx.moveTo(nowX, plot.y);
      ctx.lineTo(nowX, plot.y + plot.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = font(tickPx, 500);
      ctx.fillStyle = P.muted;
      const label = "forecast";
      if (plot.x + plot.w - nowX > ctx.measureText(label).width + 20) ctx.fillText(label, nowX + 10, plot.y + tickPx + 8);
    }
  }

  private coverageChart(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, P: Palette, f: FrameState, r: Rect, traces: CoverageTrace[]): void {
    const T = this.layout.type;
    const labelPx = T.label;
    ctx.font = font(labelPx, 500);
    ctx.fillStyle = P.muted;
    ctx.fillText(`Rolling coverage · target ${this.texts.level}`, r.x, r.y + labelPx);
    ctx.font = font(T.tick);
    const gutter = Math.ceil(ctx.measureText("100%").width + 18);
    const plot: Rect = { x: r.x + gutter, y: r.y + labelPx + T.tick, w: r.w - gutter, h: r.h - labelPx - T.tick };
    let lo = f.target;
    for (const tr of traces) for (const v of tr.v) if (Number.isFinite(v)) lo = Math.min(lo, v);
    // Early rolling values rest on a handful of pairs; do not let them flatten the chart.
    const want = Math.max(0, f.target - 0.4, Math.floor((lo - 0.05) * 10) / 10);
    this.cov = this.cov ? [this.cov[0] + (want - this.cov[0]) * 0.15, 1] : [want, 1];
    const [cLo, cHi] = this.cov;
    const from = this.covFrom ?? (traces[0]?.t[0] ?? 0);
    const to = this.covTo ?? f.t;
    const X = (t: number) => plot.x + ((t - from) / Math.max(1, to - from)) * plot.w;
    const Y = (v: number) => plot.y + (1 - (Math.min(cHi, Math.max(cLo, v)) - cLo) / (cHi - cLo)) * plot.h;

    ctx.lineWidth = 1.5;
    ctx.font = font(T.tick);
    for (const v of [cLo, (cLo + cHi) / 2, cHi]) {
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      ctx.moveTo(plot.x, Y(v));
      ctx.lineTo(plot.x + plot.w, Y(v));
      ctx.stroke();
      ctx.fillStyle = P.muted;
      const s = `${Math.round(v * 100)}%`;
      ctx.fillText(s, plot.x - 14 - ctx.measureText(s).width, Y(v) + T.tick * 0.35);
    }
    ctx.strokeStyle = P.ink;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(plot.x, Y(f.target));
    ctx.lineTo(plot.x + plot.w, Y(f.target));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    // Native first, so the focused method is drawn on top.
    for (const tr of [...traces].sort((a, b) => Number(b.native) - Number(a.native))) {
      ctx.strokeStyle = tr.native ? P.native : tr.color;
      ctx.lineWidth = (tr.native ? 2.2 : 3) * T.line;
      ctx.setLineDash(tr.native && traces.length > 1 ? [5 * T.line, 5 * T.line] : []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < tr.t.length; i++) {
        const v = tr.v[i];
        if (!Number.isFinite(v)) {
          started = false;
          continue;
        }
        if (!started) ctx.moveTo(X(tr.t[i]), Y(v));
        else ctx.lineTo(X(tr.t[i]), Y(v));
        started = true;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  /** The time span of the coverage chart (the exported range). */
  covFrom?: number;
  covTo?: number;
}

interface LegendItem {
  kind: "line" | "dash" | "fill" | "dashband" | "dot";
  color: string;
  label: string;
  width: number;
}

function layoutLegend(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, items: LegendItem[], maxW: number, px: number): LegendItem[][] {
  const rows: LegendItem[][] = [[]];
  let x = 0;
  for (const it of items) {
    it.width = px * 1.6 + ctx.measureText(it.label).width;
    if (x > 0 && x + it.width > maxW) {
      rows.push([]);
      x = 0;
    }
    rows[rows.length - 1].push(it);
    x += it.width + px * 1.3;
  }
  return rows;
}

function drawKey(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, it: LegendItem, x: number, y: number, px: number): void {
  const w = px * 1.15;
  ctx.save();
  ctx.lineCap = "butt";
  switch (it.kind) {
    case "line":
    case "dash":
      ctx.strokeStyle = it.color;
      ctx.lineWidth = px * 0.16;
      ctx.setLineDash(it.kind === "dash" ? [px * 0.28, px * 0.22] : []);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + w, y);
      ctx.stroke();
      break;
    case "fill":
    case "dashband":
      ctx.fillStyle = rgba(it.color, 0.25);
      ctx.fillRect(x, y - px * 0.36, w, px * 0.72);
      ctx.strokeStyle = it.color;
      ctx.lineWidth = px * 0.11;
      ctx.setLineDash(it.kind === "dashband" ? [px * 0.22, px * 0.18] : []);
      ctx.beginPath();
      ctx.moveTo(x, y - px * 0.36);
      ctx.lineTo(x + w, y - px * 0.36);
      ctx.moveTo(x, y + px * 0.36);
      ctx.lineTo(x + w, y + px * 0.36);
      ctx.stroke();
      break;
    case "dot":
      ctx.fillStyle = it.color;
      ctx.beginPath();
      ctx.arc(x + w / 2, y, px * 0.25, 0, Math.PI * 2);
      ctx.fill();
      break;
  }
  ctx.restore();
}

function roundRect(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function ellipsize(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, s: string, maxW: number): string {
  if (ctx.measureText(s).width <= maxW) return s;
  let t = s;
  while (t.length > 1 && ctx.measureText(`${t}…`).width > maxW) t = t.slice(0, -1);
  return `${t.trimEnd()}…`;
}

/** Greedy word wrap into at most `max` lines; the last line takes the rest and is ellipsized. */
function wrap(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, s: string, maxW: number, max: number): string[] {
  const words = s.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let cur = "";
  for (let i = 0; i < words.length; i++) {
    const next = cur ? `${cur} ${words[i]}` : words[i];
    if (ctx.measureText(next).width <= maxW || !cur) {
      cur = next;
      continue;
    }
    if (lines.length === max - 1) {
      cur = [cur, ...words.slice(i)].join(" ");
      break;
    }
    lines.push(cur);
    cur = words[i];
  }
  if (cur) lines.push(cur);
  return lines.map((l, i, a) => (i === a.length - 1 ? ellipsize(ctx, l, maxW) : l));
}
