import "./styles.css";
import { METHODS, defaultParams } from "./conformal/methods";
import { Engine, NATIVE_COLOR, NATIVE_ID, type RankBy, type ScoreType, type Settings, defaultSettings, rankRows } from "./engine/engine";
import { type Backend, BrowserBackend, NaiveBackend, ServerBackend } from "./model/backends";
import { MODELS } from "./model/registry";
import { PRESETS } from "./synthetic";
import { cssVar, fitCanvas, lineChart, withAlpha } from "./ui/plot";

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
let maxContext = 256;
let symmetric = true;
let clip = true;
let rankBy: RankBy = "interval_score";
let focusId = "aci";
let showAll = true;

let inFlight = false;
let lastLatency = 0;
let playing = false;
let playTimer: number | null = null;
let playData: number[] = [];
let playIdx = 0;
let backfilling = false;

const NAMES: Record<string, string> = { [NATIVE_ID]: "Timestral native", ...Object.fromEntries(METHODS.map((m) => [m.id, m.name])) };
const COLORS: Record<string, string> = { [NATIVE_ID]: NATIVE_COLOR, ...Object.fromEntries(METHODS.map((m) => [m.id, m.color])) };

function loadSettings(): Settings {
  const d = defaultSettings();
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
// Forecast scheduling: always forecast the newest origin; skip origins the model could not
// keep up with. Arrivals after the in-flight origin wait until its forecast is issued.

async function pump() {
  if (inFlight || backfilling || !backend.ready) return;
  const t = engine.series.length - 1;
  if (t < 0 || t <= engine.latestOrigin()) return;
  inFlight = true;
  setPill();
  const rev = engine.revision;
  const b = backend;
  try {
    engine.catchUp(t);
    const res = await b.forecast({ context: Float64Array.from(engine.series.slice(0, t + 1)), horizon, symmetric, clip, maxContext });
    lastLatency = res.ms;
    if (b === backend && engine.revision === rev && engine.series.length > t) {
      engine.addForecast(t, res.quantiles, res.horizon, res.levels);
    }
  } catch (e) {
    toast(`Forecast failed: ${e instanceof Error ? e.message : e}`);
    playing && togglePlay();
  } finally {
    inFlight = false;
    setPill();
    requestRender();
    queueMicrotask(pump);
  }
}

async function backfill() {
  if (!backend.ready || backfilling || engine.series.length === 0) return;
  if (playing) togglePlay();
  backfilling = true;
  const b = backend;
  const n = engine.series.length;
  const rev = engine.revision;
  const t0 = performance.now();
  try {
    for (let o = 0; o < n; o++) {
      if (engine.revision !== rev || b !== backend) break;
      const res = await b.forecast({ context: Float64Array.from(engine.series.slice(0, o + 1)), horizon, symmetric, clip, maxContext });
      engine.storeForecast(o, res.quantiles, res.horizon, res.levels);
      if (o % 10 === 0 || o === n - 1) {
        $("backfill-progress").textContent = `${o + 1}/${n} origins`;
        await new Promise((r) => setTimeout(r));
      }
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
  let i = step - seriesOffset;
  if (i < 0) i = 0;
  const n = engine.series.length;
  if (i < n) {
    if (i === n - 1) return; // same column: keep the first value
    engine.truncate(i);
    engine.append(round(value));
  } else {
    const last = n ? engine.series[n - 1] : value;
    for (let j = n; j <= i; j++) {
      const w = n ? (j - (n - 1)) / (i - (n - 1)) : 1;
      engine.append(round(last + (value - last) * w));
    }
  }
  $("draw-hint").classList.add("hidden");
  requestRender();
  pump();
}

const round = (v: number) => Math.round(v * 100) / 100;

canvas.addEventListener("pointerdown", (e) => {
  if (playing) togglePlay();
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const step = stepAt(e.clientX);
  if (engine.series.length && step - seriesOffset < engine.series.length - 1) engine.truncate(Math.max(0, step - seriesOffset));
  drawTo(step, valueAt(e.clientY));
});
canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  for (const ev of e.getCoalescedEvents?.() ?? [e]) drawTo(stepAt(ev.clientX), valueAt(ev.clientY));
});
const endStroke = () => {
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
  const endStep = seriesOffset + engine.series.length;
  const len = viewSteps();
  if (endStep - view.start > trigger * len) view.start = Math.max(0, Math.round(endStep - to * len));
  if (engine.series.length === 0) view.start = 0;
  requestRender();
}

// ---------------------------------------------------------------------------------------
// Synthetic streams

function togglePlay() {
  playing = !playing;
  $("btn-play").textContent = playing ? "❚❚ Pause" : "▶ Play";
  if (playing) {
    if (playIdx >= playData.length || engine.series.length === 0) startStream();
    tick();
  } else if (playTimer !== null) {
    clearTimeout(playTimer);
    playTimer = null;
  }
}

function startStream() {
  const preset = PRESETS.find((p) => p.id === $<HTMLSelectElement>("preset").value) ?? PRESETS[0];
  engine.clear();
  seriesOffset = 0;
  view.start = 0;
  playData = preset.make(2000, Math.floor(Math.random() * 1e9));
  playIdx = 0;
  $("draw-hint").classList.add("hidden");
}

function tick() {
  if (!playing) return;
  const delay = 1000 / Number($<HTMLInputElement>("speed").value);
  // Lockstep: wait for the forecast of the newest origin so every origin is scored.
  const waiting = backend.ready && (inFlight || engine.latestOrigin() < engine.series.length - 1);
  if (!waiting) {
    if (playIdx >= playData.length) {
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
  const Y = (v: number) => h - ((v - view.yMin) / (view.yMax - view.yMin)) * h;
  const S = (i: number) => X(i + seriesOffset); // series index to x

  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  for (let v = 0; v <= 100; v += 20) {
    ctx.beginPath();
    ctx.moveTo(0, Y(v));
    ctx.lineTo(w, Y(v));
    ctx.stroke();
  }
  ctx.fillStyle = muted;
  ctx.font = "11px system-ui, sans-serif";
  for (let v = 20; v < 100; v += 20) ctx.fillText(String(v), 4, Y(v) - 3);

  const n = engine.series.length;
  const first = Math.max(0, Math.floor(view.start - seriesOffset) - 1);
  const last = Math.min(n - 1, Math.ceil(view.start + w / view.pxPerStep - seriesOffset) + 1);

  // Historical lead-1 intervals of the focus method, with misses marked.
  const fid = engine.settings.enabled[focusId] || focusId === NATIVE_ID ? focusId : NATIVE_ID;
  const fcol = COLORS[fid];
  ctx.fillStyle = withAlpha(fcol, 0.18);
  const misses: [number, number][] = [];
  for (let o = first - 1; o <= last; o++) {
    const f = engine.forecasts.get(o);
    if (!f) continue;
    const iv = engine.interval(f, 0, fid);
    if (!iv) continue;
    const x = S(o + 1);
    ctx.fillRect(x - view.pxPerStep / 2, Y(iv[1]), view.pxPerStep, Math.max(1, Y(iv[0]) - Y(iv[1])));
    const y = engine.series[o + 1];
    if (y !== undefined && f.evaluated[0] && (y < iv[0] || y > iv[1])) misses.push([x, Y(y)]);
  }

  // Series.
  ctx.strokeStyle = ink;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = first; i <= last; i++) (i === first ? ctx.moveTo(S(i), Y(engine.series[i])) : ctx.lineTo(S(i), Y(engine.series[i])));
  ctx.stroke();
  ctx.fillStyle = "#e4572e";
  for (const [x, y] of misses) {
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Latest forecast fan.
  const lo = engine.latestOrigin();
  const f = engine.forecasts.get(lo);
  if (f) {
    const H = f.horizon;
    const Q = f.levels.length;
    const xAt = (k: number) => S(lo + 1 + k);
    // Native fan: 1-99%, 10-90% and the alpha band.
    const band = (loIdx: number, hiIdx: number, a: number) => {
      ctx.fillStyle = withAlpha(NATIVE_COLOR, a);
      ctx.beginPath();
      for (let k = 0; k < H; k++) ctx.lineTo(xAt(k), Y(f.quantiles[k * Q + hiIdx]));
      for (let k = H - 1; k >= 0; k--) ctx.lineTo(xAt(k), Y(f.quantiles[k * Q + loIdx]));
      ctx.closePath();
      ctx.fill();
    };
    band(0, Q - 1, 0.08);
    band(2, Q - 3, 0.12);
    ctx.strokeStyle = NATIVE_COLOR;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    for (const arr of [f.lo, f.hi]) {
      ctx.beginPath();
      for (let k = 0; k < H; k++) ctx.lineTo(xAt(k), Y(arr[k]));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(S(lo), Y(engine.series[lo]));
    for (let k = 0; k < H; k++) ctx.lineTo(xAt(k), Y(f.median[k]));
    ctx.stroke();
    ctx.setLineDash([]);

    // Conformal bands over the calibrated leads.
    const K = Math.min(H, engine.settings.evalLeads);
    const ids = engine.activeSpecs.map((m) => m.id).filter((id) => showAll || id === fid);
    for (const id of ids) {
      const ivs: [number, number][] = [];
      for (let k = 0; k < K; k++) {
        const iv = engine.interval(f, k, id);
        if (!iv) break;
        ivs.push(iv);
      }
      if (!ivs.length) continue;
      const col = COLORS[id];
      if (id === fid) {
        ctx.fillStyle = withAlpha(col, 0.18);
        ctx.beginPath();
        ivs.forEach((iv, k) => ctx.lineTo(xAt(k), Y(iv[1])));
        for (let k = ivs.length - 1; k >= 0; k--) ctx.lineTo(xAt(k), Y(ivs[k][0]));
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = col;
      ctx.lineWidth = id === fid ? 2 : 1.25;
      for (const side of [0, 1] as const) {
        ctx.beginPath();
        ivs.forEach((iv, k) => ctx.lineTo(xAt(k), Y(iv[side])));
        ctx.stroke();
      }
    }
    ctx.strokeStyle = withAlpha("#888888", 0.6);
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(S(lo) + 0.5, 0);
    ctx.lineTo(S(lo) + 0.5, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = muted;
  const info = `${n} points · ${engine.forecasts.size} forecasts${inFlight ? " · forecasting…" : ""}`;
  ctx.fillText(info, w - ctx.measureText(info).width - 6, 14);
}

function renderCharts() {
  const ids = engine.rowIds;
  const xs = engine.traceT;
  const target = 1 - engine.settings.alpha;
  $("cov-target").textContent = `target ${(target * 100).toFixed(0)}%, last ${engine.settings.rolling} evaluated pairs`;
  lineChart(
    $<HTMLCanvasElement>("cov-canvas"),
    xs,
    ids.map((id) => ({ color: COLORS[id], values: engine.metrics.get(id)!.covTrace, width: id === focusId ? 2.5 : 1.25, dash: id === NATIVE_ID ? [4, 3] : undefined })),
    { target, yMax: 1, floor: 0, format: (v) => `${(v * 100).toFixed(0)}%` },
  );
  lineChart(
    $<HTMLCanvasElement>("width-canvas"),
    xs,
    ids.map((id) => ({ color: COLORS[id], values: engine.metrics.get(id)!.widthTrace, width: id === focusId ? 2.5 : 1.25, dash: id === NATIVE_ID ? [4, 3] : undefined })),
    { yMin: 0, format: (v) => v.toFixed(1) },
  );
}

function renderTable() {
  const alpha = engine.settings.alpha;
  const target = 1 - alpha;
  const rows = rankRows(engine.table(), alpha, rankBy);
  const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "–");
  const num = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
  const bar = (v: number, col: string) =>
    `<span class="covbar"><span class="fill" style="width:${Number.isFinite(v) ? v * 100 : 0}%;background:${col}"></span><span class="target" style="left:${target * 100}%"></span></span>`;
  const html = [
    `<tr><th>#</th><th>Method</th><th>Coverage</th><th>Rolling</th><th>Mean width</th><th>Interval score</th><th>Pairs</th></tr>`,
    ...rows.map((r, i) => {
      const gapCls = Number.isFinite(r.coverage) && Math.abs(r.coverage - target) > 0.05 ? ' style="color:#e4572e"' : "";
      return `<tr class="${i === 0 && r.n > 0 ? "first" : ""}">
        <td>${r.n ? r.rank : "–"}</td>
        <td><span class="swatch" style="background:${COLORS[r.id]}"></span> ${NAMES[r.id]}</td>
        <td${gapCls}>${pct(r.coverage)}${bar(r.coverage, COLORS[r.id])}</td>
        <td>${pct(r.rollingCoverage)}</td>
        <td>${num(r.meanWidth)}</td>
        <td>${num(r.intervalScore)}</td>
        <td>${r.n}</td></tr>`;
    }),
  ];
  $("rank-table").innerHTML = html.join("");
}

function renderLegend() {
  const items = [
    `<span class="item"><span class="swatch" style="background:${cssVar("--series")}"></span>series / median</span>`,
    `<span class="item"><span class="swatch" style="background:${withAlpha(NATIVE_COLOR, 0.35)}"></span>Timestral native (α band dashed, 10–90% and 1–99% shaded)</span>`,
    ...engine.activeSpecs.map((m) => `<span class="item"><span class="swatch" style="background:${m.color}"></span>${m.name}${m.id === focusId ? " (focus)" : ""}</span>`),
    `<span class="item"><span class="swatch" style="background:#e4572e;border-radius:50%"></span>miss of the focus method at lead 1</span>`,
  ];
  $("legend").innerHTML = items.join("");
}

function setPill() {
  const pill = $("backend-pill");
  pill.textContent = backend.label + (backend.ready ? "" : " (not ready)");
  pill.className = `pill ${!backend.ready ? "err" : inFlight ? "busy" : "ok"}`;
  $("latency").textContent = lastLatency ? `${lastLatency.toFixed(0)} ms / forecast` : "";
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
  const nativeRow = `<div class="method"><div class="head">
      <input type="radio" name="focus" value="${NATIVE_ID}" ${focusId === NATIVE_ID ? "checked" : ""} title="Focus" />
      <span class="swatch" style="background:${NATIVE_COLOR}"></span>
      <span class="name">Timestral native</span></div>
      <div class="ref">The model's own ${"("}α/2, 1−α/2) quantiles, interpolated between its 21 levels. The baseline to beat.</div></div>`;
  list.innerHTML =
    nativeRow +
    METHODS.map(
      (m) => `<div class="method" data-id="${m.id}">
      <div class="head">
        <input type="radio" name="focus" value="${m.id}" ${focusId === m.id ? "checked" : ""} title="Focus: drawn filled, with its lead-1 history" />
        <span class="swatch" style="background:${m.color}"></span>
        <span class="name" title="${m.summary}">${m.name}</span>
        <input type="checkbox" class="toggle" ${s.enabled[m.id] ? "checked" : ""} title="Enable" />
      </div>
      <div class="ref" title="${m.summary}">${m.reference}</div>
      ${m.params.length ? `<div class="params">${m.params
        .map((p) => `<label title="${p.help ?? ""}">${p.label}<input type="number" data-param="${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${s.params[m.id]?.[p.key] ?? p.default}" /></label>`)
        .join("")}</div>` : ""}
    </div>`,
    ).join("");
  list.querySelectorAll<HTMLInputElement>('input[name="focus"]').forEach((el) =>
    el.addEventListener("change", () => {
      focusId = el.value;
      renderLegend();
      requestRender();
    }),
  );
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
        const params = { ...defaultParams(spec), ...engine.settings.params[id], [inp.dataset.param!]: v };
        applySettings({ params: { ...engine.settings.params, [id]: params } });
      }),
    );
  });
}

function renderModels() {
  const list = $("model-list");
  list.innerHTML = MODELS.map(
    (m) => `<div class="model-row ${activeRepo === m.repo && backend.kind === "browser" && backend.ready ? "active" : ""}">
      <div class="info"><div class="title">${m.label} <span class="muted small">${m.params} · ~${m.approxMB} MB${cachedRepos.has(m.repo) ? " · cached" : ""}</span></div>
      <div class="note">${m.note}</div></div>
      <button data-load="${m.repo}">${cachedRepos.has(m.repo) ? "Use" : "Download"}</button>
      ${cachedRepos.has(m.repo) ? `<button data-evict="${m.repo}" title="Remove from browser cache">✕</button>` : ""}
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
}

function bindControls() {
  const preset = $<HTMLSelectElement>("preset");
  preset.innerHTML = PRESETS.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  $("btn-play").addEventListener("click", togglePlay);
  preset.addEventListener("change", () => {
    if (playing) togglePlay();
    playIdx = playData.length;
  });
  $("btn-clear").addEventListener("click", () => {
    if (playing) togglePlay();
    engine.clear();
    seriesOffset = 0;
    view.start = 0;
    playIdx = playData.length;
    $("draw-hint").classList.remove("hidden");
    requestRender();
  });
  $("btn-backfill").addEventListener("click", backfill);

  const backendSel = $<HTMLSelectElement>("backend");
  backendSel.addEventListener("change", () => {
    $("browser-panel").classList.toggle("hidden", backendSel.value !== "browser");
    $("server-panel").classList.toggle("hidden", backendSel.value !== "server");
    if (backendSel.value === "naive") {
      backend = new NaiveBackend();
      setPill();
      resetForecasts();
    } else if (backendSel.value === "browser") {
      ensureBrowserBackend();
      renderModels();
      if (browserBackend?.ready) {
        backend = browserBackend;
        setPill();
        resetForecasts();
      }
    }
  });
  $<HTMLSelectElement>("server-model").innerHTML = MODELS.map((m) => `<option value="${m.repo}">${m.label}</option>`).join("");
  $("btn-connect").addEventListener("click", connectServer);
  $<HTMLInputElement>("file-input").addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files) loadFromFiles(files);
  });

  $<HTMLSelectElement>("horizon").addEventListener("change", (e) => {
    horizon = Number((e.target as HTMLSelectElement).value);
    resetForecasts();
  });
  $<HTMLInputElement>("max-context").addEventListener("change", (e) => {
    maxContext = Math.max(32, Number((e.target as HTMLInputElement).value) || 256);
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

  const s = engine.settings;
  const alpha = $<HTMLInputElement>("alpha");
  alpha.value = String(s.alpha);
  $("alpha-val").textContent = `${s.alpha.toFixed(2)} (target ${((1 - s.alpha) * 100).toFixed(0)}%)`;
  alpha.addEventListener("input", () => {
    const a = Number(alpha.value);
    $("alpha-val").textContent = `${a.toFixed(2)} (target ${((1 - a) * 100).toFixed(0)}%)`;
    applySettings({ alpha: a });
  });
  const scoreSel = $<HTMLSelectElement>("score-type");
  scoreSel.value = s.scoreType;
  scoreSel.addEventListener("change", () => applySettings({ scoreType: scoreSel.value as ScoreType }));
  const num = (id: string, key: "evalLeads" | "window" | "warmup" | "rolling") => {
    const el = $<HTMLInputElement>(id);
    el.value = String(s[key]);
    el.addEventListener("change", () => {
      const v = Math.max(Number(el.min) || 1, Math.round(Number(el.value)));
      if (Number.isFinite(v)) applySettings({ [key]: v } as Partial<Settings>);
    });
  };
  num("eval-leads", "evalLeads");
  num("window", "window");
  num("warmup", "warmup");
  num("rolling", "rolling");
  $<HTMLInputElement>("show-all").addEventListener("change", (e) => {
    showAll = (e.target as HTMLInputElement).checked;
    requestRender();
  });
  $<HTMLSelectElement>("rank-by").addEventListener("change", (e) => {
    rankBy = (e.target as HTMLSelectElement).value as RankBy;
    renderTable();
  });
  window.addEventListener("resize", requestRender);
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    renderLegend();
    requestRender();
  });
}

bindControls();
renderMethods();
renderLegend();
setPill();
requestRender();
