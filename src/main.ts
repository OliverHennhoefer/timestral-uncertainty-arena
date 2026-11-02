import "./styles.css";
import { METHODS, defaultParams } from "./conformal/methods";
import type { Uncertainty } from "./engine/bootstrap";
import { Engine, NATIVE_COLOR, NATIVE_ID, type RankBy, type ScoreType, type Settings, defaultSettings, rankRows } from "./engine/engine";
import { CalibrationProbe, pitHistogram } from "./engine/calibration";
import { ForecastScheduler } from "./engine/scheduler";
import { type ReliabilityCurve, maxGap, pitChart, reliabilityChart } from "./ui/calibration";
import { type Backend, BrowserBackend, NaiveBackend, ServerBackend } from "./model/backends";
import { FAMILY_NAMES, MODELS, type ModelFamily } from "./model/registry";
import { DATASETS } from "./datasets";
import { PRESETS as SYNTHETIC } from "./synthetic";
import { type Align, HASH_KEY, type ModelRef, type Scenario, type SeriesRef, decodeScenario, encodeScenario, scenarioFromHash, seriesHash } from "./share";
import { FONT, cssVar, fitCanvas, lineChart, withAlpha } from "./ui/plot";
import { Extent, FIXED_RANGE, type YMode, fitRange, niceTicks, zoomRange } from "./ui/yaxis";

const PRESETS = [...SYNTHETIC, ...DATASETS];

// ---------------------------------------------------------------------------------------
// State

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const engine = new Engine(loadSettings());
let backend: Backend = new NaiveBackend();
let browserBackend: BrowserBackend | null = null;
let activeRepo: string | null = null;
let cachedRepos = new Set<string>();

const view = { start: 0, pxPerStep: 2, yMin: 0, yMax: 100 };
let seriesOffset = 0; // canvas step of series[0]
let horizon = 64;
/**
 * How the native forecast and the conformal bands line up in time:
 * "full" draws the whole model horizon while conformal stops at the calibrated leads,
 * "calibrate" sets the calibrated leads to the horizon so conformal covers every lead,
 * "cut" (default) trims the drawn native forecast to where the conformal bands end.
 */
let align: Align = (() => {
  try {
    // v2: the default changed to an aligned view, so an old stored "full" default is not reused.
    const v = localStorage.getItem("ctimestral-align-v2");
    return v === "full" || v === "calibrate" ? v : "cut";
  } catch {
    return "cut";
  }
})();
/** Y-axis range of the main chart; see ui/yaxis.ts. "manual" is set by zooming or panning. */
let yMode: YMode = (() => {
  try {
    const v = localStorage.getItem("ctimestral-ymode");
    return v === "series" || v === "bands" || v === "fixed" ? v : "bands";
  } catch {
    return "bands";
  }
})();
/** The auto mode to go back to on reset (double-click the axis). */
let yAuto = yMode as Exclude<YMode, "manual">;
let maxContext = 256;
let symmetric = true;
let clip = true;
let rankBy: RankBy = "interval_score";
/** 0 = pooled over all scored leads, else a single lead. */
let rankLead = 0;
let uncCache: { key: string; at: number; len: number; value: Map<string, Uncertainty> } | null = null;
let uncTimer = 0;
let focusId = "aci";
/** Show the focused method's one-step interval behind the observed series. */
const histLead = 1;
const MISS_COLOR = "#FF3B30";
/** Calibration diagnostics; the probe only runs while its card is open. */
const calib = new CalibrationProbe();
let calibTimer = 0;
let calibOpen = (() => {
  try {
    return localStorage.getItem("ctimestral-calib-open") === "1";
  } catch {
    return false;
  }
})();
let playing = false;
let playTimer: number | null = null;
let playData: number[] = [];
let playIdx = 0;
/** The preset stream in playData, so a link can name it instead of carrying every point. */
let streamRef: { id: string; seed: number } | null = null;
/** Pause the stream at this index (a shared link replays up to where it was shared). */
let playStopAt: number | null = null;
/** A shared scenario is loaded and starts playing once its model is ready. */
let pendingReplay = false;
/** Browser checkpoint a shared link asked for, loaded once the cache listing arrives. */
let wantRepo: string | null = null;
/**
 * The link names a browser checkpoint that is not loaded yet: hold the replay until it is,
 * or until the user picks a backend or checkpoint themselves, so it never runs on the wrong model.
 */
let replayNeedsModel = false;
let backfilling = false;

const NAMES: Record<string, string> = { [NATIVE_ID]: "Timestral native", ...Object.fromEntries(METHODS.map((m) => [m.id, m.name])) };
const COLORS: Record<string, string> = { [NATIVE_ID]: NATIVE_COLOR, ...Object.fromEntries(METHODS.map((m) => [m.id, m.color])) };

function loadSettings(): Settings {
  const d = defaultSettings();
  // Start with a useful, compact comparison. The full method library stays available.
  d.enabled = Object.fromEntries(METHODS.map((m) => [m.id, ["split", "aci", "dtaci"].includes(m.id)]));
  try {
    const raw = localStorage.getItem("ctimestral-settings");
    if (raw) {
      const s = JSON.parse(raw) as Partial<Settings>;
      return { ...d, ...s, enabled: { ...d.enabled, ...s.enabled }, params: { ...d.params, ...s.params } };
    }
  } catch {
    // ignore
  }
  return d;
}

function saveSettings() {
  try {
    localStorage.setItem("ctimestral-settings", JSON.stringify(engine.settings));
  } catch {
    // ignore
  }
}

function toast(msg: string, ms = 4000) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  window.clearTimeout((toast as unknown as { t?: number }).t);
  (toast as unknown as { t?: number }).t = window.setTimeout(() => el.classList.add("hidden"), ms);
}

// ---------------------------------------------------------------------------------------
// Forecast scheduling: the newest origin first, then the origins skipped while the model
// could not keep up with the pen (see engine/scheduler.ts).

const scheduler = new ForecastScheduler({
  engine,
  backend: () => backend,
  options: () => ({ horizon, symmetric, clip, maxContext }),
  paused: () => backfilling,
  changed: () => {
    setPill();
    requestRender();
  },
  failed: (e) => {
    toast(`Forecast failed: ${e instanceof Error ? e.message : e}. Draw a new point or switch backend to retry.`);
    playing && togglePlay();
  },
});
const pump = () => void scheduler.pump();

async function backfill() {
  if (!backend.ready || backfilling || engine.series.length === 0) return;
  if (playing) togglePlay();
  backfilling = true;
  requestRender();
  const b = backend;
  const n = engine.series.length;
  const rev = engine.revision;
  const t0 = performance.now();
  try {
    for (let o = 0; o < n; o++) {
      if (engine.revision !== rev || b !== backend) break;
      const res = await b.forecast({ context: Float64Array.from(engine.series.slice(0, o + 1)), horizon, symmetric, clip, maxContext });
      // The series, backend or forecast settings may have changed while we awaited.
      if (engine.revision !== rev || b !== backend) break;
      engine.storeForecast(o, res.quantiles, res.horizon, res.levels);
      if (o % 10 === 0 || o === n - 1) {
        $("backfill-progress").textContent = `${o + 1}/${n} origins`;
        await new Promise((r) => setTimeout(r));
      }
    }
    if (engine.revision !== rev || b !== backend) {
      $("backfill-progress").textContent = "recompute cancelled (series or model changed)";
      return;
    }
    engine.replay();
    $("backfill-progress").textContent = `recomputed ${n} origins in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (e) {
    toast(`Recompute failed: ${e instanceof Error ? e.message : e}`);
  } finally {
    backfilling = false;
    requestRender();
    pump();
  }
}

/** Drop all forecasts (new model or forecast settings) and start over from the newest origin. */
function resetForecasts() {
  scheduler.forget();
  engine.forecasts.clear();
  engine.revision++;
  engine.replay();
  requestRender();
  pump();
}

// ---------------------------------------------------------------------------------------
// Drawing

const canvas = $<HTMLCanvasElement>("main-canvas");
let drawing = false;

function stepAt(clientX: number): number {
  const rect = canvas.getBoundingClientRect();
  return Math.round(view.start + (clientX - rect.left) / view.pxPerStep);
}

function valueAt(clientY: number): number {
  const rect = canvas.getBoundingClientRect();
  const f = 1 - (clientY - rect.top) / rect.height;
  return Math.max(view.yMin, Math.min(view.yMax, view.yMin + f * (view.yMax - view.yMin)));
}

function drawTo(step: number, value: number) {
  if (engine.series.length === 0) seriesOffset = step;
  const i = Math.max(0, step - seriesOffset);
  const n = engine.series.length;
  if (i < n) return; // strokes only extend the series
  const last = n ? engine.series[n - 1] : value;
  for (let j = n; j <= i; j++) {
    const w = n ? (j - (n - 1)) / (i - (n - 1)) : 1;
    engine.append(round(last + (value - last) * w));
    scheduler.arrived(engine.series.length - 1);
  }
  $("draw-hint").classList.add("hidden");
  requestRender();
  pump();
}

const round = (v: number) => Math.round(v * 100) / 100;

// Y axis: scroll on the axis strip to zoom, drag the strip
// to pan, double-click it to go back to the fitted range.
const AXIS_GUTTER = 44;
let yDrag: { y: number; lo: number; hi: number } | null = null;
const inGutter = (e: MouseEvent) => e.clientX - canvas.getBoundingClientRect().left < AXIS_GUTTER;

function setYMode(m: YMode) {
  yMode = m;
  if (m !== "manual") yAuto = m;
  $<HTMLSelectElement>("y-mode").value = m;
  try {
    localStorage.setItem("ctimestral-ymode", m === "manual" ? yAuto : m);
  } catch {
    /* storage unavailable */
  }
  requestRender();
}

function zoomY(factor: number, clientY?: number) {
  const rect = canvas.getBoundingClientRect();
  const anchor = clientY === undefined ? undefined : view.yMax - ((clientY - rect.top) / rect.height) * (view.yMax - view.yMin);
  [view.yMin, view.yMax] = zoomRange(view.yMin, view.yMax, factor, anchor);
  setYMode("manual");
}

canvas.addEventListener(
  "wheel",
  (e) => {
    if (!inGutter(e)) return;
    e.preventDefault();
    zoomY(Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 200) * 0.002), e.clientY);
  },
  { passive: false },
);
canvas.addEventListener("dblclick", (e) => {
  if (inGutter(e)) setYMode(yAuto);
});
canvas.addEventListener("mousemove", (e) => {
  if (!drawing && !yDrag) canvas.style.cursor = inGutter(e) ? "ns-resize" : "";
});

canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  if (inGutter(e)) {
    if (!engine.series.length) return;
    yDrag = { y: e.clientY, lo: view.yMin, hi: view.yMax };
    canvas.setPointerCapture(e.pointerId);
    return;
  }
  if (playing) togglePlay();
  pendingReplay = false;
  setFollow(true);
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const step = stepAt(e.clientX);
  drawTo(step, valueAt(e.clientY));
});
canvas.addEventListener("pointermove", (e) => {
  if (yDrag) {
    const d = ((e.clientY - yDrag.y) / canvas.getBoundingClientRect().height) * (yDrag.hi - yDrag.lo);
    view.yMin = yDrag.lo + d;
    view.yMax = yDrag.hi + d;
    if (yMode !== "manual") setYMode("manual");
    else requestRender();
    return;
  }
  if (!drawing) return;
  for (const ev of e.getCoalescedEvents?.() ?? [e]) drawTo(stepAt(ev.clientX), valueAt(ev.clientY));
});
const endStroke = () => {
  yDrag = null;
  if (!drawing) return;
  drawing = false;
  followEnd(0.8, 0.45);
};
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

function viewSteps(): number {
  return canvas.getBoundingClientRect().width / view.pxPerStep;
}

/** Scroll so the series end sits at `to` of the width once it passes `trigger`. */
function followEnd(trigger: number, to: number) {
  if (!follow && engine.series.length) {
    requestRender();
    return;
  }
  const endStep = seriesOffset + engine.series.length;
  const len = viewSteps();
  if (endStep - view.start > trigger * len) view.start = Math.max(0, Math.round(endStep - to * len));
  if (engine.series.length === 0) view.start = 0;
  requestRender();
}


// ---------------------------------------------------------------------------------------
// Navigation: x zoom and pan, the overview strip, and follow-latest.

/** Whether the view keeps the newest point in sight. Any manual navigation turns it off. */
let follow = true;
const overview = $<HTMLCanvasElement>("overview-canvas");
const MIN_PX = 0.2, MAX_PX = 40;

function setFollow(on: boolean) {
  follow = on;
  $("x-latest").classList.toggle("active", on);
  $("x-latest").setAttribute("aria-pressed", String(on));
}

/** Forecast steps actually shown by the current alignment. */
function forecastSteps(): number {
  const f = engine.forecasts.get(engine.latestOrigin());
  return f ? align === "full" ? f.horizon : Math.min(f.horizon, engine.settings.evalLeads) : 0;
}

function overviewSpan(): number {
  const end = seriesOffset + engine.series.length + forecastSteps() + 4;
  return Math.max(end, view.start + viewSteps(), 1);
}

function setXView(start: number, px = view.pxPerStep) {
  view.pxPerStep = Math.min(MAX_PX, Math.max(MIN_PX, px));
  const end = Math.max(seriesOffset + engine.series.length + forecastSteps() + 4, viewSteps());
  view.start = Math.min(Math.max(0, start), end - viewSteps());
  setFollow(false);
  requestRender();
}

/** Zoom the time axis by `factor` (>1 zooms out) keeping the step under `clientX` in place. */
function zoomX(factor: number, clientX?: number) {
  const rect = canvas.getBoundingClientRect();
  const fx = clientX === undefined ? 0.5 : (clientX - rect.left) / rect.width;
  const anchor = view.start + fx * viewSteps();
  const px = Math.min(MAX_PX, Math.max(MIN_PX, view.pxPerStep / factor));
  setXView(anchor - (fx * rect.width) / px, px);
}

function fitAll() {
  const w = canvas.getBoundingClientRect().width;
  const steps = Math.max(50, engine.series.length + forecastSteps() + 4);
  setXView(seriesOffset - 2, w / steps);
  setYMode(yAuto);
}

function goLatest() {
  setFollow(true);
  // Unconditional, so it also comes back from a view panned past the end of the series.
  view.start = Math.max(0, Math.round(seriesOffset + engine.series.length - 0.75 * viewSteps()));
  requestRender();
}

// Main chart: Ctrl/Alt + scroll (or a trackpad pinch) zooms time, Shift + scroll or a
// sideways scroll pans it. Plain scroll is left to the page, except on the y-axis strip.
canvas.addEventListener(
  "wheel",
  (e) => {
    if (inGutter(e)) return; // the y-axis handler
    if (e.ctrlKey || e.altKey) {
      e.preventDefault();
      zoomX(Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 200) * 0.003), e.clientX);
    } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault();
      const d = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
      setXView(view.start + d / view.pxPerStep);
    }
  },
  { passive: false },
);

/** Overview strip: the whole series, with the visible window as a draggable box. */
function renderOverview() {
  const { ctx, w, h } = fitCanvas(overview);
  ctx.fillStyle = cssVar("--surface-2");
  ctx.fillRect(0, 0, w, h);
  const span = overviewSpan();
  const OX = (step: number) => (step / span) * w;
  const n = engine.series.length;
  if (!n) return;
  if (n) {
    let lo = Infinity, hi = -Infinity;
    for (const v of engine.series) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
    if (hi - lo < 1e-9) (lo -= 1), (hi += 1);
    const OY = (v: number) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8);
    ctx.strokeStyle = withAlpha(cssVar("--series").startsWith("#") ? cssVar("--series") : "#1d1d1f", 0.55);
    ctx.lineWidth = 1;
    ctx.beginPath();
    // One min-max pair per pixel column keeps long series cheap and spikes visible.
    const per = Math.max(1, Math.floor(n / w));
    for (let i = 0; i < n; i += per) {
      let a = Infinity, b = -Infinity;
      for (let j = i; j < Math.min(n, i + per); j++) (a = Math.min(a, engine.series[j])), (b = Math.max(b, engine.series[j]));
      const x = OX(seriesOffset + i);
      if (i === 0) ctx.moveTo(x, OY(engine.series[0]));
      ctx.lineTo(x, OY(b));
      if (per > 1) ctx.lineTo(x, OY(a));
    }
    ctx.stroke();
  }
  const x0 = OX(view.start), x1 = OX(view.start + viewSteps());
  ctx.fillStyle = withAlpha(cssVar("--accent"), 0.06);
  ctx.fillRect(x0, 0, Math.max(3, x1 - x0), h);
  ctx.strokeStyle = withAlpha(cssVar("--accent"), 0.38);
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, 0.5, Math.max(3, x1 - x0) - 1, h - 1);
}

let ovDrag: { mode: "move" | "l" | "r"; x: number; start: number; end: number } | null = null;
const ovStep = (clientX: number) => {
  const r = overview.getBoundingClientRect();
  return ((clientX - r.left) / r.width) * overviewSpan();
};
overview.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !engine.series.length) return;
  const r = overview.getBoundingClientRect();
  const span = overviewSpan();
  const x0 = r.left + (view.start / span) * r.width, x1 = r.left + ((view.start + viewSteps()) / span) * r.width;
  let mode: "move" | "l" | "r" = "move";
  if (Math.abs(e.clientX - x0) < 7) mode = "l";
  else if (Math.abs(e.clientX - x1) < 7) mode = "r";
  else if (e.clientX < x0 || e.clientX > x1) setXView(ovStep(e.clientX) - viewSteps() / 2); // jump: centre on the click
  ovDrag = { mode, x: e.clientX, start: view.start, end: view.start + viewSteps() };
  overview.setPointerCapture(e.pointerId);
});
overview.addEventListener("pointermove", (e) => {
  if (!ovDrag) {
    const r = overview.getBoundingClientRect();
    const span = overviewSpan();
    const x0 = r.left + (view.start / span) * r.width, x1 = r.left + ((view.start + viewSteps()) / span) * r.width;
    overview.style.cursor = Math.abs(e.clientX - x0) < 7 || Math.abs(e.clientX - x1) < 7 ? "ew-resize" : "grab";
    return;
  }
  const w = canvas.getBoundingClientRect().width;
  const d = ((e.clientX - ovDrag.x) / overview.getBoundingClientRect().width) * overviewSpan();
  if (ovDrag.mode === "move") setXView(ovDrag.start + d);
  else if (ovDrag.mode === "l") {
    const s0 = Math.min(ovDrag.start + d, ovDrag.end - 10);
    setXView(s0, w / (ovDrag.end - s0));
  } else {
    const e1 = Math.max(ovDrag.end + d, ovDrag.start + 10);
    setXView(ovDrag.start, w / (e1 - ovDrag.start));
  }
});
const endOv = () => (ovDrag = null);
overview.addEventListener("pointerup", endOv);
overview.addEventListener("pointercancel", endOv);
overview.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomX(Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 200) * 0.003), r.left + r.width / 2);
  },
  { passive: false },
);

// ---------------------------------------------------------------------------------------
// Example streams

function togglePlay() {
  playing = !playing;
  $("btn-play").textContent = playing ? "Pause" : "Play";
  $("btn-play").setAttribute("aria-label", playing ? "Pause" : "Play example");
  $("btn-play").setAttribute("aria-pressed", String(playing));
  pendingReplay = false;
  if (playing) {
    if (playIdx >= playData.length || (engine.series.length === 0 && playIdx > 0)) startStream();
    tick();
  } else if (playTimer !== null) {
    clearTimeout(playTimer);
    playTimer = null;
  }
}

function startStream(seed = Math.floor(Math.random() * 1e9)) {
  const preset = PRESETS.find((p) => p.id === $<HTMLSelectElement>("preset").value) ?? PRESETS[0];
  engine.clear();
  scheduler.forget();
  seriesOffset = 0;
  view.start = 0;
  setFollow(true);
  playData = preset.make(2000, seed);
  playIdx = 0;
  playStopAt = null;
  streamRef = { id: preset.id, seed };
  $("draw-hint").classList.add("hidden");
}

function tick() {
  if (!playing) return;
  const delay = 1000 / 30;
  // Lockstep: wait for the forecast of the newest origin so every origin is scored.
  const waiting = backend.ready && (scheduler.inFlight || engine.latestOrigin() < engine.series.length - 1);
  if (!waiting) {
    if (playIdx >= playData.length || (playStopAt !== null && playIdx >= playStopAt)) {
      playStopAt = null;
      togglePlay();
      return;
    }
    engine.append(playData[playIdx++]);
    followEnd(0.75, 0.75);
    pump();
  }
  playTimer = window.setTimeout(tick, waiting ? 5 : delay);
}

// ---------------------------------------------------------------------------------------
// Rendering

let renderQueued = false;
let lastTableRender = 0;

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderMain();
    renderOverview();
    const now = performance.now();
    if (now - lastTableRender > 150) {
      lastTableRender = now;
      renderCharts();
      renderTable();
    } else {
      setTimeout(requestRender, 160);
    }
  });
}

function renderMain() {
  const { ctx, w, h } = fitCanvas(canvas);
  const plotBg = cssVar("--plot-bg"), grid = cssVar("--grid"), ink = cssVar("--series"), muted = cssVar("--muted");
  ctx.fillStyle = plotBg;
  ctx.fillRect(0, 0, w, h);
  const X = (step: number) => (step - view.start) * view.pxPerStep;
  const S = (i: number) => X(i + seriesOffset); // series index to x

  const n = engine.series.length;
  $("series-count").textContent = n ? `${n.toLocaleString()} points · ${Math.min(horizon, engine.settings.evalLeads)} calibrated steps` : "No observations";
  $("axis-start").textContent = `Step ${Math.floor(view.start)}`;
  $("axis-mid").textContent = String(Math.round(view.start + viewSteps() / 2));
  $("axis-end").textContent = String(Math.round(view.start + viewSteps()));
  $<HTMLButtonElement>("btn-clear").disabled = n === 0;
  $<HTMLButtonElement>("btn-backfill").disabled = n === 0 || !backend.ready || backfilling;
  $<HTMLButtonElement>("btn-play").disabled = !backend.ready || backfilling;
  const first = Math.max(0, Math.floor(view.start - seriesOffset) - 1);
  const last = Math.min(n - 1, Math.ceil(view.start + w / view.pxPerStep - seriesOffset) + 1);

  // Past band: the interval the focused method issued `histLead` steps ahead, drawn at the
  // time it was about, with misses marked. One method at a time; pick it with the legend chips.
  const fid = engine.settings.enabled[focusId] || focusId === NATIVE_ID ? focusId : NATIVE_ID;
  const k = Math.min(histLead, engine.settings.evalLeads);
  const past = k > 0 ? engine.pastBand(fid, k, first, last) : [];

  // Latest forecast: conformal intervals over the calibrated leads; a band stops early at a
  // lead that has no threshold yet (lead k is first scored k steps after its first forecast).
  const lo = engine.latestOrigin();
  const f = engine.forecasts.get(lo);
  const bands = new Map<string, [number, number][]>();
  let H = 0;
  if (f) {
    const K = Math.min(f.horizon, engine.settings.evalLeads);
    let reach = 0;
    for (const id of engine.activeSpecs.map((m) => m.id).filter((id) => id === fid)) {
      const ivs: [number, number][] = [];
      for (let j = 0; j < K; j++) {
        const iv = engine.interval(f, j, id);
        if (!iv) break;
        ivs.push(iv);
      }
      if (ivs.length) bands.set(id, ivs);
      reach = Math.max(reach, ivs.length);
    }
    // Aligned modes draw the native forecast exactly as far as the conformal bands reach,
    // so both always end at the same lead (falling back to the calibrated leads while warming up).
    H = align === "full" ? f.horizon : reach || K;
  }
  const Q = f ? f.levels.length : 0;

  // Y range: fitted to what is on screen unless the user zoomed or panned (or is drawing,
  // when the axis must not move under the pen).
  if (yMode !== "manual" && !drawing) {
    if (yMode === "fixed" || n === 0) [view.yMin, view.yMax] = FIXED_RANGE;
    else {
      const e = new Extent();
      for (let i = first; i <= last; i++) e.add(engine.series[i]);
      // Only forecast leads that are on screen count, so a wide latest forecast does not
      // squash the history while you look at an earlier stretch.
      const onScreen = (j: number) => {
        const x = S(lo + 1 + j);
        return x >= 0 && x <= w;
      };
      if (f) for (let j = 0; j < H; j++) if (onScreen(j)) e.add(f.median[j]);
      if (yMode === "bands") {
        // Never narrower than the 0-100 canvas, so drawing keeps a stable frame.
        e.add(FIXED_RANGE[0]);
        e.add(FIXED_RANGE[1]);
        for (const [, l, u] of past) (e.add(l), e.add(u));
        if (f)
          for (let j = 0; j < H; j++) {
            if (!onScreen(j)) continue;
            e.add(f.quantiles[j * Q + 2]);
            e.add(f.quantiles[j * Q + Q - 3]);
            e.add(f.lo[j]);
            e.add(f.hi[j]);
          }
        for (const ivs of bands.values()) ivs.forEach(([l, u], j) => onScreen(j) && (e.add(l), e.add(u)));
      }
      [view.yMin, view.yMax] = fitRange(e.lo, e.hi);
    }
  }
  const Y = (v: number) => h - ((v - view.yMin) / (view.yMax - view.yMin)) * h;

  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.font = FONT;
  const labelCol = muted;
  const ticks = niceTicks(view.yMin, view.yMax, Math.max(3, Math.round(h / 90)));
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  for (const v of ticks) {
    ctx.beginPath();
    ctx.moveTo(0, Y(v));
    ctx.lineTo(w, Y(v));
    ctx.stroke();
  }
  ctx.fillStyle = labelCol;
  for (const v of ticks) {
    const y = Y(v);
    if (y > 14 && y < h - 2) ctx.fillText(v.toFixed(digits), 6, y - 4);
  }

  // Edge markers for anything clipped by the y range: a thin strip on the edge, in the
  // band's colour, so a band running off the chart is never silently cut.
  const EDGE = 3;
  const clipped: [number, number, string, boolean][] = []; // x, width, colour, top?
  const markClip = (x: number, wd: number, l: number, u: number, col: string) => {
    if (u > view.yMax) clipped.push([x, wd, col, true]);
    if (l < view.yMin) clipped.push([x, wd, col, false]);
  };

  const misses: [number, number][] = [];
  if (past.length) {
    ctx.fillStyle = withAlpha(COLORS[fid], 0.16);
    for (const [t, l, u, missed] of past) {
      const x = S(t);
      ctx.fillRect(x - view.pxPerStep / 2, Y(u), view.pxPerStep, Math.max(1, Y(l) - Y(u)));
      markClip(x - view.pxPerStep / 2, view.pxPerStep, l, u, COLORS[fid]);
      if (missed) misses.push([x, Y(engine.series[t])]);
    }
  }

  // Series.
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.75;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = first; i <= last; i++) (i === first ? ctx.moveTo(S(i), Y(engine.series[i])) : ctx.lineTo(S(i), Y(engine.series[i])));
  ctx.stroke();
  ctx.fillStyle = MISS_COLOR;
  for (const [x, y] of misses) {
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  if (f) {
    const xAt = (j: number) => S(lo + 1 + j);
    const colW = Math.max(2, view.pxPerStep);
    // Native fan: 10-90% shaded, the alpha band dashed.
    const band = (loIdx: number, hiIdx: number, a: number) => {
      ctx.fillStyle = withAlpha(NATIVE_COLOR, a);
      ctx.beginPath();
      for (let j = 0; j < H; j++) ctx.lineTo(xAt(j), Y(f.quantiles[j * Q + hiIdx]));
      for (let j = H - 1; j >= 0; j--) ctx.lineTo(xAt(j), Y(f.quantiles[j * Q + loIdx]));
      ctx.closePath();
      ctx.fill();
    };
    band(2, Q - 3, 0.1);
    ctx.strokeStyle = withAlpha(NATIVE_COLOR, 0.9);
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (const arr of [f.lo, f.hi]) {
      ctx.beginPath();
      for (let j = 0; j < H; j++) ctx.lineTo(xAt(j), Y(arr[j]));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (let j = 0; j < H; j++) markClip(xAt(j) - colW / 2, colW, f.lo[j], f.hi[j], NATIVE_COLOR);
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(S(lo), Y(engine.series[lo]));
    for (let j = 0; j < H; j++) ctx.lineTo(xAt(j), Y(f.median[j]));
    ctx.stroke();
    ctx.setLineDash([]);

    // Conformal bands.
    for (const [id, ivs] of bands) {
      const col = COLORS[id];
      if (id === fid) {
        ctx.fillStyle = withAlpha(col, 0.14);
        ctx.beginPath();
        ivs.forEach((iv, j) => ctx.lineTo(xAt(j), Y(iv[1])));
        for (let j = ivs.length - 1; j >= 0; j--) ctx.lineTo(xAt(j), Y(ivs[j][0]));
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = id === fid ? col : withAlpha(col, 0.55);
      ctx.lineWidth = id === fid ? 1.75 : 1;
      for (const side of [0, 1] as const) {
        ctx.beginPath();
        ivs.forEach((iv, j) => ctx.lineTo(xAt(j), Y(iv[side])));
        ctx.stroke();
      }
      ivs.forEach(([l, u], j) => markClip(xAt(j) - colW / 2, colW, l, u, col));
    }
    ctx.strokeStyle = withAlpha("#8e8e93", 0.35);
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(S(lo) + 0.5, 0);
    ctx.lineTo(S(lo) + 0.5, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Clipped edges last, so they sit on top. The focused method is drawn after the rest.
  clipped.sort((a, b) => Number(a[2] === COLORS[fid]) - Number(b[2] === COLORS[fid]));
  for (const [x, wd, col, top] of clipped) {
    ctx.fillStyle = col;
    ctx.fillRect(x, top ? 0 : h - EDGE, wd, EDGE);
  }
  const nTop = clipped.some((c) => c[3]), nBot = clipped.some((c) => !c[3]);
  if (nTop || nBot) {
    ctx.fillStyle = labelCol;
    const note = "Intervals extend beyond the view. Use Fit or scroll on the vertical axis.";
    if (yMode !== "bands" && w > 520) ctx.fillText(note, w - ctx.measureText(note).width - 10, h - 10);
  }
}

function renderCharts() {
  const ids = engine.rowIds;
  const xs = engine.traceT;
  const target = 1 - engine.settings.alpha;
  $("cov-target").textContent = `target ${(target * 100).toFixed(0)}%`;
  $("cov-target").title = `Over the last ${engine.settings.rolling} time steps (all scored leads)`;
  // The focused method is drawn last and solid; the others recede.
  const order = [...ids.filter((id) => id === NATIVE_ID && id !== focusId), ...ids.filter((id) => id === focusId)];
  const trace = (id: string, values: number[]) => ({
    color: id === focusId ? COLORS[id] : withAlpha(COLORS[id], 0.7),
    values,
    width: id === focusId ? 2 : 1.25,
    dash: id === NATIVE_ID ? [4, 3] : undefined,
  });
  lineChart(
    $<HTMLCanvasElement>("cov-canvas"),
    xs,
    order.map((id) => trace(id, engine.metrics.get(id)!.covTrace)),
    { target, yMax: 1, floor: 0, format: (v) => `${(v * 100).toFixed(0)}%` },
  );
  lineChart(
    $<HTMLCanvasElement>("width-canvas"),
    xs,
    order.map((id) => trace(id, engine.metrics.get(id)!.widthTrace)),
    { yMin: 0, format: (v) => v.toFixed(1) },
  );
  // Keep the workspace quiet and avoid drawing canvases in a closed disclosure.
  if (!calibOpen) {
    $("calib-sub").textContent = "";
    return;
  }
  // Per-lead profile: is the method calibrated at every lead, not just on average?
  const K = engine.settings.evalLeads;
  const leads = Array.from({ length: K }, (_, i) => i + 1);
  const prof = new Map(order.map((id) => [id, engine.leadProfile(id)]));
  const anyData = [...prof.values()].some((p) => p.coverage.some(Number.isFinite));
  lineChart(
    $<HTMLCanvasElement>("lead-cov-canvas"),
    anyData ? leads : [],
    order.map((id) => trace(id, prof.get(id)!.coverage)),
    { target, yMax: 1, floor: 0, format: (v) => `${(v * 100).toFixed(0)}%` },
  );
  lineChart(
    $<HTMLCanvasElement>("lead-width-canvas"),
    anyData ? leads : [],
    order.map((id) => trace(id, prof.get(id)!.width)),
    { yMin: 0, format: (v) => v.toFixed(1) },
  );
  renderCalibration();
}

/** PIT histogram and reliability diagram for the native interval and the focused method. */
function renderCalibration() {
  // Mid-recompute the stored forecasts are ahead of the engine; wait for its replay.
  if (!calibOpen || backfilling) return;
  // Shadow engines rebuild in slices so a focus change on a long series never blocks the page.
  calib.sync(engine, focusId, 25);
  if (!calib.ready && !calibTimer) calibTimer = window.setTimeout(() => ((calibTimer = 0), renderCalibration()), 0);
  const lead = rankLead || undefined;
  const pits = calib.pit(lead);
  pitChart($<HTMLCanvasElement>("pit-canvas"), pitHistogram(pits, 10), NATIVE_COLOR);
  const rel = calib.reliability(engine, lead);
  const mid = calib.focusedMethod;
  const curves: ReliabilityCurve[] = [{ name: NAMES[NATIVE_ID], color: NATIVE_COLOR, points: rel.native, dash: [4, 3], focus: !mid }];
  if (mid) curves.push({ name: NAMES[mid], color: COLORS[mid], points: rel.method, focus: true });
  reliabilityChart($<HTMLCanvasElement>("rel-canvas"), curves, 1 - engine.settings.alpha);
  const gap = (pts: ReliabilityCurve["points"]) => {
    const g = maxGap(pts);
    return Number.isFinite(g) ? ` · max gap ${g.toFixed(1)} pts` : "";
  };
  $("rel-legend").innerHTML = curves
    .map((c) => `<span class="rel-key" title="${c.name}${gap(c.points)}"><span class="line${c.dash ? " dash" : ""}" style="border-color:${c.color}"></span>${c.name}</span>`)
    .join("");
  const busy = mid && !calib.ready ? ` · Updating ${Math.round(calib.progress * 100)}%` : "";
  $("calib-sub").textContent = `${pits.length} pairs · ${lead ? `lead ${lead}` : `leads 1–${engine.settings.evalLeads}`}${busy}`;
}

/** Bootstrap intervals, recomputed at most once a second (they are not cheap). */
function uncertainty(): Map<string, Uncertainty> {
  const lead = rankLead || undefined;
  const key = `${rankLead}|${engine.revision}|${engine.rowIds.join(",")}|${JSON.stringify(engine.settings)}`;
  const now = performance.now();
  const len = engine.traceT.length;
  if (!uncCache || uncCache.key !== key || (uncCache.len !== len && now - uncCache.at > 1000)) {
    uncCache = { key, at: now, len, value: engine.uncertainty(lead) };
  } else if (uncCache.len !== len && !uncTimer) {
    // Too soon to refresh: make sure the final state still gets its intervals.
    uncTimer = window.setTimeout(() => { uncTimer = 0; renderTable(); }, 1050);
  }
  return uncCache.value;
}

// Persisted so a reload while in "calibrate" can still restore the user's own choice.
const LEADS_KEY = "ctimestral-leads-before-calibrate";
let leadsBeforeCalibrate: number | null = (() => {
  try {
    const v = Number(localStorage.getItem(LEADS_KEY));
    return Number.isInteger(v) && v > 0 ? v : null;
  } catch {
    return null;
  }
})();
function storeLeadsBefore(v: number | null) {
  leadsBeforeCalibrate = v;
  try {
    if (v === null) localStorage.removeItem(LEADS_KEY);
    else localStorage.setItem(LEADS_KEY, String(v));
  } catch {
    /* storage unavailable */
  }
}

/** Apply the horizon-alignment choice: tie the calibrated leads to the horizon, or not. */
function applyAlign() {
  const el = $<HTMLInputElement>("eval-leads");
  const hint = $("align-hint");
  if (align === "calibrate") {
    if (engine.settings.evalLeads !== horizon) {
      if (leadsBeforeCalibrate === null) storeLeadsBefore(engine.settings.evalLeads);
      applySettings({ evalLeads: horizon });
      renderLeadOptions();
    }
    el.value = String(horizon);
    el.disabled = true;
    hint.textContent = `Calibrates all ${horizon} steps.`;
  } else {
    // Leaving "calibrate" restores the calibrated leads the user had before.
    if (leadsBeforeCalibrate !== null) {
      const prev = leadsBeforeCalibrate;
      storeLeadsBefore(null);
      if (prev !== engine.settings.evalLeads) {
        applySettings({ evalLeads: prev });
        renderLeadOptions();
      }
    }
    el.disabled = false;
    el.value = String(engine.settings.evalLeads);
    hint.textContent =
      align === "cut"
        ? `${Math.min(horizon, engine.settings.evalLeads)} calibrated steps.`
        : engine.settings.evalLeads < horizon
          ? `Shows all ${horizon} model steps; ${engine.settings.evalLeads} are calibrated.`
          : "";
  }
  requestRender();
}

function renderLeadOptions() {
  const sel = $<HTMLSelectElement>("rank-lead");
  const K = engine.settings.evalLeads;
  if (rankLead > K) rankLead = 0;
  sel.innerHTML = `<option value="0">All leads 1–${K}</option>` + Array.from({ length: K }, (_, i) => `<option value="${i + 1}">Lead ${i + 1}</option>`).join("");
  sel.value = String(rankLead);
}

function renderTable() {
  const alpha = engine.settings.alpha;
  const target = 1 - alpha;
  const unc = uncertainty();
  const rows = rankRows(engine.table(rankLead || undefined), alpha, rankBy, 0.02, unc);
  const hasData = rows.some((r) => r.n > 0);
  $("rank-empty").classList.toggle("hidden", hasData);
  $("rank-table").classList.toggle("hidden", !hasData);
  const criteria: Record<RankBy, string> = {
    interval_score: "Interval score",
    validity_then_width: "Coverage, then width",
    rolling_gap: "Rolling coverage",
  };
  $("ranking-scope").textContent = `${criteria[rankBy]} · ${rankLead ? `step ${rankLead}` : "all steps"}`;
  if (!hasData) return;
  // Percentile intervals need not be symmetric, so show the lower and upper distances separately.
  const pm = (est: number, ci: [number, number] | undefined, scale = 1, digits = 2) => {
    if (!ci || !Number.isFinite(est) || !Number.isFinite(ci[0]) || !Number.isFinite(ci[1])) return `<span class="pm"></span>`;
    const lo = Math.max(0, est - ci[0]) * scale, hi = Math.max(0, ci[1] - est) * scale;
    const tip = `95% interval ${(ci[0] * scale).toFixed(digits)} – ${(ci[1] * scale).toFixed(digits)}`;
    return `<span class="pm" title="${tip}"><span>+${hi.toFixed(digits)}</span><span>−${lo.toFixed(digits)}</span></span>`;
  };
  const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "–");
  const num = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
  const bar = (v: number, col: string) =>
    `<span class="covbar"><span class="fill" style="width:${Number.isFinite(v) ? v * 100 : 0}%;background:${col}"></span><span class="target" style="left:${target * 100}%"></span></span>`;
  const html = [
    `<thead><tr><th scope="col">#</th><th scope="col">Method</th><th scope="col">Coverage</th><th scope="col">Width</th><th scope="col">Score</th></tr></thead><tbody>`,
    ...rows.map((r, i) => {
      const off = Number.isFinite(r.coverage) && Math.abs(r.coverage - target) > 0.05 ? ' class="off"' : "";
      const u = unc.get(r.id);
      const tip = `Rolling coverage ${pct(r.rollingCoverage)} · ${r.n} pairs${r.tied ? " · tied with the row above" : ""}`;
      return `<tr class="${i === 0 && r.n > 0 ? "first" : ""}${r.id === focusId ? " focus" : ""}${r.tied ? " tied" : ""}" data-id="${r.id}" title="${tip}">
        <td>${r.n ? r.rank : "–"}</td>
        <td><button class="rank-method" aria-pressed="${r.id === focusId}" aria-label="Focus ${NAMES[r.id]}"><span class="dot" style="background:${COLORS[r.id]}"></span>${NAMES[r.id]}</button></td>
        <td${off}>${bar(r.coverage, COLORS[r.id])}${pct(r.coverage)}${pm(r.coverage, u?.coverage, 100, 1)}</td>
        <td>${num(r.meanWidth)}</td>
        <td>${num(r.intervalScore)}${pm(r.intervalScore, u?.intervalScore)}</td></tr>`;
    }),
  ];
  const activeRow = document.activeElement?.closest<HTMLElement>("#rank-table tr[data-id]")?.dataset.id;
  $("rank-table").innerHTML = html.join("") + "</tbody>";
  if (activeRow) $("rank-table").querySelector<HTMLButtonElement>(`tr[data-id="${activeRow}"] button`)?.focus({ preventScroll: true });
}

function setFocus(id: string) {
  focusId = id;
  renderLegend();
  requestRender();
}

function renderLegend() {
  if (focusId !== NATIVE_ID && !engine.settings.enabled[focusId]) focusId = NATIVE_ID;
  const focus = $<HTMLSelectElement>("focus-method");
  focus.innerHTML = `<option value="${NATIVE_ID}">${backend.modelName} native</option>` + engine.activeSpecs.map((m) => `<option value="${m.id}">${m.name}</option>`).join("");
  focus.value = focusId;
  $("method-count").textContent = String(engine.activeSpecs.length);
  $("legend").innerHTML =
    `<span class="legend-key"><i class="legend-line" style="color:var(--series)"></i>Observed</span>` +
    `<span class="legend-key"><i class="legend-line dashed" style="color:${NATIVE_COLOR}"></i>Native</span>` +
    (focusId === NATIVE_ID ? "" : `<span class="legend-key"><i class="legend-line" style="color:${COLORS[focusId]}"></i>${NAMES[focusId]}</span>`) +
    `<span class="legend-key"><i class="dot" style="background:${MISS_COLOR}"></i>Misses</span>`;
}

/** Model family the browser panel lists, from the backend dropdown. */
function browserFamily(): ModelFamily {
  return $<HTMLSelectElement>("backend").value === "chronos" ? "chronos-bolt" : "timestral";
}

function setPill() {
  // The native row is named after the model that produced it.
  const native = backend.modelName;
  if (NAMES[NATIVE_ID] !== `${native} native`) {
    NAMES[NATIVE_ID] = `${native} native`;
    renderLegend();
  }
  const pill = $("backend-pill");
  pill.textContent = (backend.kind === "naive" ? "Naive baseline" : backend.label) + (backend.ready ? "" : " · not ready");
  pill.className = `pill ${!backend.ready ? "err" : scheduler.inFlight ? "busy" : "ok"}`;
  $("latency").textContent = scheduler.lastLatency ? `${scheduler.lastLatency.toFixed(0)} ms` : "";
}

// ---------------------------------------------------------------------------------------
// Controls

function applySettings(patch: Partial<Settings>) {
  engine.updateSettings(patch);
  saveSettings();
  renderLegend();
  requestRender();
}

function renderMethods() {
  const list = $("method-list");
  const s = engine.settings;
  const groups: [string, string][] = [["conformal", "Conformal"], ["bayesian", "Bayesian"]];
  const item = (m: (typeof METHODS)[number]) => `<div class="method" data-id="${m.id}">
      <label class="check-row" title="${m.summary} (${m.reference})${m.guarantee === false ? " · no distribution-free coverage guarantee" : ""}">
        <input type="checkbox" class="toggle" ${s.enabled[m.id] ? "checked" : ""} />
        <i class="dot" style="background:${m.color}"></i><span class="name">${m.name}</span>
      </label>
      ${m.params.length ? `<details class="params-box"><summary aria-label="${m.name} parameters">Parameters</summary>
        <div class="ref">${m.reference}</div>
        <div class="params">${m.params
          .map((p) => `<label title="${p.help ?? ""}">${p.label}<input type="number" data-param="${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${s.params[m.id]?.[p.key] ?? p.default}" /></label>`)
          .join("")}</div></details>` : ""}
    </div>`;
  list.innerHTML = groups
    .map(([fam, label]) => {
      const ms = METHODS.filter((m) => (m.family ?? "conformal") === fam);
      return ms.length ? `<div class="method-group">${label}</div>${ms.map(item).join("")}` : "";
    })
    .join("");
  list.querySelectorAll<HTMLElement>(".method[data-id]").forEach((row) => {
    const id = row.dataset.id!;
    row.querySelector<HTMLInputElement>(".toggle")!.addEventListener("change", (e) => {
      applySettings({ enabled: { ...engine.settings.enabled, [id]: (e.target as HTMLInputElement).checked } });
    });
    row.querySelectorAll<HTMLInputElement>("input[data-param]").forEach((inp) =>
      inp.addEventListener("change", () => {
        const spec = METHODS.find((m) => m.id === id)!;
        const v = Number(inp.value);
        if (!Number.isFinite(v)) return;
        const value = Math.max(Number(inp.min), Math.min(Number(inp.max), v));
        inp.value = String(value);
        const params = { ...defaultParams(spec), ...engine.settings.params[id], [inp.dataset.param!]: value };
        applySettings({ params: { ...engine.settings.params, [id]: params } });
      }),
    );
  });
}

function renderModels() {
  const list = $("model-list");
  const family = browserFamily();
  $("bolt-note").classList.toggle("hidden", family !== "chronos-bolt");
  list.innerHTML = MODELS.filter((m) => m.family === family).map(
    (m) => `<div class="model-row ${activeRepo === m.repo && backend.kind === "browser" && backend.ready ? "active" : ""}">
      <div class="info"><div class="title">${m.label} <span class="muted small">${m.params} · ~${m.approxMB} MB${cachedRepos.has(m.repo) ? " · cached" : ""}</span></div>
      </div>
      <button data-load="${m.repo}">${cachedRepos.has(m.repo) ? "Use" : "Download"}</button>
      ${cachedRepos.has(m.repo) ? `<button data-evict="${m.repo}" title="Remove from browser cache" aria-label="Remove ${m.label} from cache">×</button>` : ""}
    </div>`,
  ).join("");
  list.querySelectorAll<HTMLButtonElement>("button[data-load]").forEach((b) => b.addEventListener("click", () => loadBrowserModel(b.dataset.load!)));
  list.querySelectorAll<HTMLButtonElement>("button[data-evict]").forEach((b) => b.addEventListener("click", () => browserBackend?.evict(b.dataset.evict!)));
}

function ensureBrowserBackend(): BrowserBackend {
  if (!browserBackend) {
    browserBackend = new BrowserBackend();
    browserBackend.onCached = (repos) => {
      cachedRepos = new Set(repos);
      renderModels();
      if (wantRepo) useSharedRepo();
    };
    browserBackend.onProgress = (p) => {
      const box = $("dl-progress");
      box.classList.remove("hidden");
      const frac = p.total ? p.loaded / p.total : 0;
      $("dl-bar").style.width = `${(frac * 100).toFixed(1)}%`;
      $("dl-text").textContent = `${p.file}: ${(p.loaded / 1e6).toFixed(1)}${p.total ? ` / ${(p.total / 1e6).toFixed(1)}` : ""} MB`;
    };
  }
  return browserBackend;
}

async function loadBrowserModel(repo: string) {
  const b = ensureBrowserBackend();
  backend = b;
  activeRepo = repo;
  setPill();
  $("backend-pill").textContent = `loading ${repo}…`;
  try {
    await b.load(repo);
    $("dl-progress").classList.add("hidden");
    toast(`${repo} ready (${(b.numParams / 1e6).toFixed(1)}M parameters)`);
  } catch (e) {
    toast(`Could not load ${repo}: ${e instanceof Error ? e.message : e}. Try the local server or load the files manually.`, 8000);
  }
  setPill();
  renderModels();
  resetForecasts();
  maybeStartReplay(true);
}

async function loadFromFiles(files: FileList) {
  const arr = [...files];
  const cfg = arr.find((f) => f.name.endsWith(".json"));
  const st = arr.find((f) => f.name.endsWith(".safetensors"));
  if (!cfg || !st) {
    toast("Select both config.json and model.safetensors.");
    return;
  }
  const b = ensureBrowserBackend();
  backend = b;
  activeRepo = null;
  try {
    await b.loadFiles(`${st.name} (local file)`, JSON.parse(await cfg.text()), await st.arrayBuffer());
    toast(`Loaded ${(b.numParams / 1e6).toFixed(1)}M parameters from files`);
  } catch (e) {
    toast(`Could not load files: ${e instanceof Error ? e.message : e}`);
  }
  setPill();
  renderModels();
  resetForecasts();
  maybeStartReplay(true);
}

async function connectServer() {
  const url = $<HTMLInputElement>("server-url").value;
  const repo = $<HTMLSelectElement>("server-model").value;
  const sb = new ServerBackend(url, repo);
  try {
    await sb.connect();
    backend = sb;
    toast(`Connected to ${url}; the first forecast downloads ${repo} on the server.`);
  } catch (e) {
    toast(`Cannot reach ${url}: ${e instanceof Error ? e.message : e}. Is server/serve.py running?`, 8000);
  }
  setPill();
  resetForecasts();
  maybeStartReplay(true);
}

function bindControls() {
  const preset = $<HTMLSelectElement>("preset");
  // A small set covers seasonality, a distribution shift, and two real series.
  // Older shared scenarios can still restore any bundled stream.
  for (const id of ["sine", "levels", "airline", "electricity"]) {
    const p = PRESETS.find((p) => p.id === id)!;
    preset.add(new Option(p.label, p.id));
  }
  $("focus-method").addEventListener("change", (e) => setFocus((e.target as HTMLSelectElement).value));
  $("btn-play").addEventListener("click", togglePlay);
  preset.addEventListener("change", () => {
    if (playing) togglePlay();
    pendingReplay = false;
    playIdx = playData.length;
  });
  $("btn-clear").addEventListener("click", () => {
    if (playing) togglePlay();
    pendingReplay = false;
    streamRef = null;
    engine.clear();
    scheduler.forget();
    seriesOffset = 0;
    view.start = 0;
    view.pxPerStep = 2;
    setYMode(yAuto);
    setFollow(true);
    playIdx = playData.length;
    $("draw-hint").classList.remove("hidden");
    requestRender();
  });
  $("btn-backfill").addEventListener("click", backfill);
  $("btn-share").addEventListener("click", copyLink);
  window.addEventListener("hashchange", restoreFromHash);
  $("rank-table").addEventListener("click", (e) => {
    const row = (e.target as HTMLElement).closest<HTMLElement>("tr[data-id]");
    if (row) setFocus(row.dataset.id!);
  });

  const backendSel = $<HTMLSelectElement>("backend");
  backendSel.addEventListener("change", () => {
    const inBrowser = backendSel.value === "browser" || backendSel.value === "chronos";
    $("browser-panel").classList.toggle("hidden", !inBrowser);
    $("server-panel").classList.toggle("hidden", backendSel.value !== "server");
    if (backendSel.value === "naive") {
      backend = new NaiveBackend();
      setPill();
      resetForecasts();
    } else if (inBrowser) {
      ensureBrowserBackend();
      renderModels();
      // Reuse the loaded model only if it is the family picked; otherwise wait for a download.
      if (browserBackend?.ready && browserBackend.family === browserFamily()) {
        backend = browserBackend;
        setPill();
        resetForecasts();
      }
    }
    maybeStartReplay(true);
  });
  $<HTMLSelectElement>("server-model").innerHTML = (Object.keys(FAMILY_NAMES) as ModelFamily[])
    .map((f) => `<optgroup label="${FAMILY_NAMES[f]}">${MODELS.filter((m) => m.family === f).map((m) => `<option value="${m.repo}">${m.label}</option>`).join("")}</optgroup>`)
    .join("");
  $("btn-connect").addEventListener("click", connectServer);
  $<HTMLInputElement>("file-input").addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files) loadFromFiles(files);
  });

  $<HTMLSelectElement>("horizon").addEventListener("change", (e) => {
    horizon = Number((e.target as HTMLSelectElement).value);
    resetForecasts();
    applyAlign();
  });
  const yModeSel = $<HTMLSelectElement>("y-mode");
  yModeSel.value = yMode;
  yModeSel.addEventListener("change", () => setYMode(yModeSel.value as YMode));
  $("x-in").addEventListener("click", () => zoomX(1 / 1.5));
  $("x-out").addEventListener("click", () => zoomX(1.5));
  $("x-fit").addEventListener("click", fitAll);
  $("x-latest").addEventListener("click", goLatest);
  const navigate = (e: KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case "ArrowLeft": setXView(view.start - viewSteps() * 0.15); break;
      case "ArrowRight": setXView(view.start + viewSteps() * 0.15); break;
      case "+": case "=": zoomX(1 / 1.5); break;
      case "-": zoomX(1.5); break;
      case "0": fitAll(); break;
      case "End": goLatest(); break;
      default: return;
    }
    e.preventDefault();
  };
  canvas.addEventListener("keydown", navigate);
  overview.addEventListener("keydown", navigate);

  const alignSel = $<HTMLSelectElement>("align");
  alignSel.value = align;
  alignSel.addEventListener("change", () => {
    align = alignSel.value as Align;
    try {
      localStorage.setItem("ctimestral-align-v2", align);
    } catch {
      // ignore
    }
    applyAlign();
  });
  $<HTMLInputElement>("max-context").addEventListener("change", (e) => {
    const input = e.target as HTMLInputElement;
    maxContext = Math.min(32768, Math.max(32, Math.round(Number(input.value)) || 256));
    input.value = String(maxContext);
    resetForecasts();
  });
  $<HTMLInputElement>("symmetric").addEventListener("change", (e) => {
    symmetric = (e.target as HTMLInputElement).checked;
    resetForecasts();
  });
  $<HTMLInputElement>("clip").addEventListener("change", (e) => {
    clip = (e.target as HTMLInputElement).checked;
    resetForecasts();
  });

  const alpha = $<HTMLInputElement>("alpha");
  alpha.addEventListener("input", () => {
    const a = (100 - Number(alpha.value)) / 100;
    $("alpha-val").textContent = alphaLabel(a);
    applySettings({ alpha: a });
  });
  const scoreSel = $<HTMLSelectElement>("score-type");
  scoreSel.addEventListener("change", () => applySettings({ scoreType: scoreSel.value as ScoreType }));
  const num = (id: string, key: "evalLeads" | "window" | "warmup" | "rolling") => {
    const el = $<HTMLInputElement>(id);
    el.addEventListener("change", () => {
      const v = Math.min(Number(el.max), Math.max(Number(el.min) || 1, Math.round(Number(el.value))));
      if (Number.isFinite(v)) {
        el.value = String(v);
        applySettings({ [key]: v } as Partial<Settings>);
      }
    });
  };
  syncSettingControls();
  num("eval-leads", "evalLeads");
  $("eval-leads").addEventListener("change", () => {
    renderLeadOptions();
    applyAlign();
  });
  applyAlign();
  num("window", "window");
  num("warmup", "warmup");
  num("rolling", "rolling");
  renderLeadOptions();
  $<HTMLSelectElement>("rank-lead").addEventListener("change", (e) => {
    rankLead = Number((e.target as HTMLSelectElement).value) || 0;
    renderTable();
    requestRender();
  });
  const calibEl = $<HTMLDetailsElement>("calib");
  calibEl.open = calibOpen;
  calibEl.addEventListener("toggle", () => {
    calibOpen = calibEl.open;
    try {
      localStorage.setItem("ctimestral-calib-open", calibOpen ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
    requestRender();
  });
  $<HTMLSelectElement>("rank-by").addEventListener("change", (e) => {
    rankBy = (e.target as HTMLSelectElement).value as RankBy;
    renderTable();
  });
  window.addEventListener("resize", () => followEnd(0.8, 0.7));
  const resize = new ResizeObserver(requestRender);
  [canvas, overview, $("cov-canvas"), $("width-canvas")].forEach((el) => resize.observe(el));
}

const alphaLabel = (a: number) => `${((1 - a) * 100).toFixed(0)}%`;

/** Write the conformal settings into their controls (on start and after opening a link). */
function syncSettingControls() {
  const s = engine.settings;
  $<HTMLInputElement>("alpha").value = String(Math.round((1 - s.alpha) * 100));
  $("alpha-val").textContent = alphaLabel(s.alpha);
  $<HTMLSelectElement>("score-type").value = s.scoreType;
  $<HTMLInputElement>("eval-leads").value = String(s.evalLeads);
  $<HTMLInputElement>("window").value = String(s.window);
  $<HTMLInputElement>("warmup").value = String(s.warmup);
  $<HTMLInputElement>("rolling").value = String(s.rolling);
}

// ---------------------------------------------------------------------------------------
// Shareable links: the scenario lives in the URL hash (see share.ts).

function currentScenario(): Scenario {
  const y = engine.series;
  let series: SeriesRef;
  if (!y.length) series = { kind: "empty" };
  else if (streamRef && seriesOffset === 0 && y.length <= playData.length && y.every((v, i) => v === playData[i]))
    series = { kind: "preset", ...streamRef, n: y.length, hash: seriesHash(y) };
  else series = { kind: "values", values: y, offset: seriesOffset };
  const model: ModelRef =
    backend instanceof ServerBackend
      ? { backend: "server", repo: backend.repo }
      : backend.kind === "browser"
        ? { backend: "browser", repo: activeRepo }
        : { backend: "naive" };
  return {
    series,
    model,
    horizon,
    maxContext,
    symmetric,
    clip,
    align,
    leadsBefore: align === "calibrate" ? leadsBeforeCalibrate : null,
    settings: engine.settings,
    rankBy,
    rankLead,
    focus: focusId,
  };
}

async function copyLink() {
  let url: string;
  try {
    url = `${location.origin}${location.pathname}${location.search}#${HASH_KEY}=${await encodeScenario(currentScenario())}`;
  } catch (e) {
    toast(`Could not make a link: ${e instanceof Error ? e.message : e}`);
    return;
  }
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    toast("Link copied.");
  } catch {
    toast("Copy the link from the address bar.");
  }
}

function modelLabel(repo: string): string {
  const m = MODELS.find((x) => x.repo === repo);
  return m ? `${m.label} (~${m.approxMB} MB)` : repo;
}

/** Load the browser checkpoint a link asked for if it is cached; otherwise ask for the download. */
function useSharedRepo() {
  const repo = wantRepo!;
  wantRepo = null;
  if (activeRepo === repo && browserBackend?.ready) {
    backend = browserBackend;
    setPill();
    replayNeedsModel = false;
    maybeStartReplay();
  } else if (cachedRepos.has(repo)) {
    loadBrowserModel(repo);
  } else {
    setPanel(true);
    toast(`This scenario uses ${modelLabel(repo)}. Press Download under Model to replay it, or pick another backend.`, 10000);
  }
}

function selectModel(m: ModelRef) {
  const sel = $<HTMLSelectElement>("backend");
  sel.value = m.backend;
  $("browser-panel").classList.toggle("hidden", m.backend !== "browser");
  $("server-panel").classList.toggle("hidden", m.backend !== "server");
  if (m.backend === "naive") {
    if (backend.kind !== "naive") backend = new NaiveBackend();
  } else if (m.backend === "browser") {
    const existed = browserBackend !== null;
    backend = ensureBrowserBackend();
    replayNeedsModel = true;
    renderModels();
    if (m.repo) {
      wantRepo = m.repo;
      if (existed) useSharedRepo(); // otherwise the cache listing triggers it
    } else {
      setPanel(true);
      toast("This scenario used a checkpoint loaded from files. Load it under Model to replay it, or pick another backend.", 10000);
    }
  } else {
    const repo = m.repo && MODELS.some((x) => x.repo === m.repo) ? m.repo : MODELS[0].repo;
    $<HTMLSelectElement>("server-model").value = repo;
    backend = new ServerBackend($<HTMLInputElement>("server-url").value, repo);
    setPanel(true);
    toast(`This scenario used ${repo} on the local server. Run server/serve.py and press Connect to replay it.`, 10000);
  }
  setPill();
}

/** `chosen`: the user (or the link's own checkpoint load) just picked the model to replay with. */
function maybeStartReplay(chosen = false) {
  if (chosen) replayNeedsModel = false;
  if (!pendingReplay || replayNeedsModel || !backend.ready || playing) return;
  pendingReplay = false;
  togglePlay();
}

/** Open the scenario in the URL hash: apply its settings, select its model, replay its series. */
async function restoreFromHash() {
  const code = scenarioFromHash(location.hash);
  if (!code) return;
  let s: Scenario;
  try {
    s = await decodeScenario(code, defaultSettings());
  } catch (e) {
    toast(`Could not open this link: ${e instanceof Error ? e.message : e}`, 8000);
    return;
  }
  if (playing) togglePlay();
  pendingReplay = false;
  replayNeedsModel = false;
  wantRepo = null;

  horizon = s.horizon;
  $<HTMLSelectElement>("horizon").value = String(horizon);
  maxContext = s.maxContext;
  $<HTMLInputElement>("max-context").value = String(maxContext);
  symmetric = s.symmetric;
  $<HTMLInputElement>("symmetric").checked = symmetric;
  clip = s.clip;
  $<HTMLInputElement>("clip").checked = clip;
  align = s.align;
  $<HTMLSelectElement>("align").value = align;
  try {
    localStorage.setItem("ctimestral-align-v2", align);
  } catch {
    // ignore
  }
  storeLeadsBefore(s.leadsBefore);
  engine.updateSettings(s.settings);
  saveSettings();
  syncSettingControls();
  rankBy = s.rankBy;
  $<HTMLSelectElement>("rank-by").value = rankBy;
  rankLead = s.rankLead;
  focusId = s.focus;
  renderMethods();
  renderLeadOptions();
  applyAlign();
  renderLegend();

  // Series: replayed through the stream so every origin is forecast and scored as it arrives.
  engine.clear();
  view.start = 0;
  setFollow(true);
  if (yMode === "manual") setYMode(yAuto);
  seriesOffset = 0;
  playIdx = 0;
  playStopAt = null;
  streamRef = null;
  playData = [];
  if (s.series.kind === "preset") {
    const ref = s.series;
    const p = PRESETS.find((x) => x.id === ref.id);
    if (!p) {
      toast(`This link uses a stream ("${ref.id}") this version does not have.`, 8000);
    } else {
      const select = $<HTMLSelectElement>("preset");
      if (![...select.options].some((o) => o.value === p.id)) select.add(new Option(p.label, p.id));
      select.value = p.id;
      playData = p.make(2000, ref.seed);
      playStopAt = Math.min(ref.n, playData.length);
      streamRef = { id: p.id, seed: ref.seed };
      if (seriesHash(playData.slice(0, playStopAt)) !== ref.hash) toast(`The "${p.label}" stream has changed since this link was made, so the replay differs.`, 8000);
    }
  } else if (s.series.kind === "values") {
    playData = s.series.values;
    seriesOffset = s.series.offset;
    view.start = Math.max(0, seriesOffset - 20);
  }
  $("draw-hint").classList.toggle("hidden", playData.length > 0);
  pendingReplay = playData.length > 0;

  resetForecasts();
  selectModel(s.model);
  maybeStartReplay();
}

// ---------------------------------------------------------------------------------------
// Configuration side panel

function setPanel(open: boolean, persist = true) {
  const panel = $("panel");
  const narrow = window.matchMedia("(max-width: 900px)").matches;
  const hadPanelFocus = panel.contains(document.activeElement);
  const openedFromButton = document.activeElement === $("btn-panel-open");
  $("shell").classList.toggle("collapsed", !open);
  panel.inert = !open;
  $("btn-panel-open").setAttribute("aria-expanded", String(open));
  $("main-col").inert = narrow && open;
  if (narrow && open) {
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    $("btn-panel-close").focus({ preventScroll: true });
  } else {
    panel.removeAttribute("role");
    panel.removeAttribute("aria-modal");
    if (hadPanelFocus && !open) $("btn-panel-open").focus({ preventScroll: true });
    if (open && openedFromButton) $("btn-panel-close").focus({ preventScroll: true });
  }
  if (persist) {
    try {
      localStorage.setItem("ctimestral-panel", open ? "open" : "closed");
    } catch {
      // ignore
    }
  }
  requestRender();
  // The plot width changes with the panel; keep the series end in view.
  requestAnimationFrame(() => followEnd(0.8, 0.7));
}

function initPanel() {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem("ctimestral-panel");
  } catch {
    // ignore
  }
  const narrow = window.matchMedia("(max-width: 900px)").matches;
  setPanel(narrow ? false : stored !== "closed", false);
  $("btn-panel-open").addEventListener("click", () => setPanel(true));
  $("btn-panel-close").addEventListener("click", () => setPanel(false));
  $("scrim").addEventListener("click", () => setPanel(false));
  window.matchMedia("(max-width: 900px)").addEventListener("change", (e) => setPanel(!e.matches, false));
  window.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    const panelOpen = !$("shell").classList.contains("collapsed");
    if (e.key === "Escape" && panelOpen) {
      setPanel(false);
      e.preventDefault();
      return;
    }
    if (e.key === "Tab" && panelOpen && window.matchMedia("(max-width: 900px)").matches) {
      const closedSections = [...$("panel").querySelectorAll<HTMLDetailsElement>("details:not([open])")];
      const controls = [...$("panel").querySelectorAll<HTMLElement>("button, select, input, summary, a[href]")]
        .filter((el) => !el.matches(":disabled") && el.getClientRects().length > 0
          && !closedSections.some((section) => section.contains(el) && section.firstElementChild !== el));
      const first = controls[0], last = controls[controls.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        last?.focus();
        e.preventDefault();
      } else if (!e.shiftKey && document.activeElement === last) {
        first?.focus();
        e.preventDefault();
      }
    }
    if (e.key !== "[" || e.metaKey || e.ctrlKey || e.altKey || ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
    setPanel($("shell").classList.contains("collapsed"));
  });
}

initPanel();
bindControls();
renderMethods();
renderLegend();
setPill();
requestRender();
restoreFromHash();

// Dev-only handle for automated UI checks.
if (import.meta.env.DEV) (window as unknown as { __ctimestral: unknown }).__ctimestral = { engine };
