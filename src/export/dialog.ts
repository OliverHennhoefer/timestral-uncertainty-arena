/**
 * The export dialog: pick a layout, file type, length and theme, preview the last frame,
 * then render and download the run as a GIF or a video.
 */

import "./export.css";
import { NATIVE_ID } from "../engine/engine";
import { type EncodeResult, drawStill, encode, outputSize, videoSupport } from "./encode";
import { type ExportFormat, GIF_SOFT_LIMIT, LAYOUTS, type Layout, layoutById, schedule } from "./layouts";
import type { Texts, Theme } from "./render";
import { type ExportSource, SceneReplay } from "./scene";

export interface ExportRequest extends ExportSource {
  /** Shown in the subtitle, e.g. "Timestral 5M". */
  modelLabel: string;
  /** Ids the method picker offers (enabled methods). */
  methodIds: string[];
}

const STORE = "ctimestral-export";

interface Prefs {
  layout: string;
  format: "gif" | "video";
  seconds: number;
  theme: Theme;
}

function loadPrefs(): Prefs {
  const d: Prefs = { layout: "social", format: "gif", seconds: 10, theme: "light" };
  try {
    const raw = localStorage.getItem(STORE);
    if (raw) return { ...d, ...(JSON.parse(raw) as Partial<Prefs>) };
  } catch {
    // storage unavailable
  }
  return d;
}

function savePrefs(p: Prefs) {
  try {
    localStorage.setItem(STORE, JSON.stringify(p));
  } catch {
    // storage unavailable
  }
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const mb = (bytes: number) => (bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`);
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "run";

let dialog: HTMLDialogElement | null = null;

export function openExportDialog(req: ExportRequest): void {
  dialog?.remove();
  const prefs = loadPrefs();
  const n = req.series.length;
  const focus0 = req.focusId === NATIVE_ID || req.methodIds.includes(req.focusId) ? req.focusId : NATIVE_ID;
  const ranges = [n, 2000, 1000, 500, 250].filter((k, i) => i === 0 || k < n * 0.9);
  const level = `${Math.round((1 - req.settings.alpha) * 100)}%`;

  const el = document.createElement("dialog");
  el.className = "export-dialog";
  el.setAttribute("aria-labelledby", "export-title");
  el.innerHTML = `<div class="export-inner">
    <div class="export-head">
      <div>
        <h2 id="export-title">Export animation</h2>
        <p class="muted small">Replays this run in a layout made for slides and feeds.</p>
      </div>
      <button class="icon" data-close aria-label="Close" title="Close (Esc)">
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="export-body">
      <div class="export-stage">
        <div class="export-frame" id="export-frame">
          <canvas id="export-canvas" aria-label="Preview of the last frame"></canvas>
          <img id="export-img" class="hidden" alt="Exported animation" />
          <video id="export-video" class="hidden" muted loop playsinline autoplay aria-label="Exported video"></video>
        </div>
        <p class="export-meta muted small" id="export-meta" role="status"></p>
      </div>
      <div class="export-options">
        <fieldset class="export-layouts">
          <legend>Layout</legend>
          ${LAYOUTS.map(
            (l) => `<label class="export-layout" title="${esc(l.use)}">
              <input type="radio" name="export-layout" value="${l.id}" ${l.id === prefs.layout ? "checked" : ""} />
              <span class="glyph" style="aspect-ratio:${l.w}/${l.h}"></span>
              <span class="txt"><strong>${l.label}</strong><span>${esc(l.use)}</span></span>
            </label>`,
          ).join("")}
        </fieldset>
        <div class="pair">
          <label class="field">File
            <select id="export-format">
              <option value="gif">GIF</option>
              <option value="video">Video</option>
            </select>
          </label>
          <label class="field">Length
            <select id="export-seconds">
              <option value="6">Up to 6 s</option>
              <option value="10">Up to 10 s</option>
              <option value="15">Up to 15 s</option>
              <option value="30">Up to 30 s</option>
            </select>
          </label>
          <label class="field">Steps
            <select id="export-range">
              ${ranges.map((k, i) => `<option value="${k}">${i === 0 ? `Whole run · ${n.toLocaleString()}` : `Last ${k.toLocaleString()}`}</option>`).join("")}
            </select>
          </label>
          <label class="field">Theme
            <select id="export-theme">
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
        </div>
        <label class="field">Method
          <select id="export-method">
            ${[...req.methodIds, NATIVE_ID]
              .map((id) => `<option value="${id}" ${id === focus0 ? "selected" : ""}>${esc(id === NATIVE_ID ? `${req.names[id]} only` : req.names[id] ?? id)}</option>`)
              .join("")}
          </select>
        </label>
        <label class="field">Title <input id="export-title-input" type="text" maxlength="80" value="${esc(req.title)}" /></label>
        <label class="field">Subtitle <input id="export-sub-input" type="text" maxlength="120" /></label>
      </div>
    </div>
    <div class="export-foot">
      <div class="export-progress hidden" id="export-progress" aria-hidden="true"><div class="bar" id="export-bar"></div></div>
      <span class="muted small" id="export-status" role="status"></span>
      <div class="export-actions">
        <button class="ghost" data-close id="export-cancel">Close</button>
        <a class="button-link hidden" id="export-download" download>Download</a>
        <button class="primary" id="export-go">Create GIF</button>
      </div>
    </div></div>`;
  document.body.append(el);
  dialog = el;

  const q = <T extends HTMLElement>(sel: string) => el.querySelector(sel) as T;
  const canvas = q<HTMLCanvasElement>("#export-canvas");
  const img = q<HTMLImageElement>("#export-img");
  const video = q<HTMLVideoElement>("#export-video");
  const fmtSel = q<HTMLSelectElement>("#export-format");
  const secSel = q<HTMLSelectElement>("#export-seconds");
  const rangeSel = q<HTMLSelectElement>("#export-range");
  const themeSel = q<HTMLSelectElement>("#export-theme");
  const methodSel = q<HTMLSelectElement>("#export-method");
  const titleIn = q<HTMLInputElement>("#export-title-input");
  const subIn = q<HTMLInputElement>("#export-sub-input");
  const go = q<HTMLButtonElement>("#export-go");
  const download = q<HTMLAnchorElement>("#export-download");
  const status = q<HTMLElement>("#export-status");
  const meta = q<HTMLElement>("#export-meta");
  const progress = q<HTMLElement>("#export-progress");
  const bar = q<HTMLElement>("#export-bar");

  fmtSel.value = prefs.format;
  secSel.value = String(prefs.seconds);
  if (!secSel.value) secSel.value = "10";
  themeSel.value = prefs.theme;

  let video0: "mp4" | "webm" | null | undefined;
  let replay: SceneReplay | null = null;
  let replayFocus = "";
  let running: AbortController | null = null;
  let resultUrl: string | null = null;
  let subEdited = false;

  const autoSub = (focus: string) => {
    const vs = focus === NATIVE_ID ? `${level} intervals` : `${req.names[focus] ?? focus} vs. native · ${level} intervals`;
    return `${req.modelLabel} · ${vs}`;
  };
  subIn.value = autoSub(focus0);
  subIn.addEventListener("input", () => (subEdited = true));

  const layout = (): Layout => layoutById(el.querySelector<HTMLInputElement>('input[name="export-layout"]:checked')?.value ?? "social");
  const format = (): ExportFormat => (fmtSel.value === "gif" ? "gif" : video0 === "webm" ? "webm" : "mp4");
  const source = (): ExportSource => ({ ...req, focusId: methodSel.value });
  const range = (): [number, number] => [Math.max(0, n - Number(rangeSel.value)), n - 1];
  const texts = (): Texts => ({ title: titleIn.value.trim() || req.title, subtitle: subIn.value.trim(), level, brand: "Timestral Uncertainty Arena" });

  const clearResult = () => {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = null;
    img.classList.add("hidden");
    video.classList.add("hidden");
    video.removeAttribute("src");
    canvas.classList.remove("hidden");
    download.classList.add("hidden");
    status.textContent = "";
  };

  const describe = () => {
    const L = layout();
    const f = format();
    const [w, h] = outputSize(L, f);
    const [from, to] = range();
    const s = schedule(from, to, Number(secSel.value), f);
    const secs = s.times.length * s.frameSec;
    const fps = Math.round(1 / s.frameSec);
    const perFrame = s.times.length > 1 ? Math.max(1, Math.round((to - from + 1) / s.times.length)) : 1;
    meta.textContent = `${w} × ${h} · ${f === "gif" ? "GIF" : f.toUpperCase()} · ${secs.toFixed(1)} s at ${fps} fps${perFrame > 1 ? ` · ${perFrame} steps per frame` : ""}`;
    go.textContent = f === "gif" ? "Create GIF" : `Create ${f.toUpperCase()}`;
    secSel.querySelector<HTMLOptionElement>('option[value="30"]')!.disabled = f === "gif";
    if (f === "gif" && secSel.value === "30") secSel.value = "15";
  };

  const preview = () => {
    if (running) return;
    clearResult();
    const L = layout();
    const frame = q<HTMLElement>("#export-frame");
    frame.style.aspectRatio = `${L.w} / ${L.h}`;
    frame.style.setProperty("--ar", String(L.w / L.h));
    // A crisp still at roughly the displayed size.
    const scale = Math.min(2, (window.devicePixelRatio || 1)) * 0.5;
    canvas.width = Math.round(L.w * scale);
    canvas.height = Math.round(L.h * scale);
    if (!subEdited) subIn.value = autoSub(methodSel.value);
    const [from, to] = range();
    if (replayFocus !== methodSel.value) (replay = null), (replayFocus = methodSel.value);
    replay = drawStill(canvas, { source: source(), layout: L, theme: themeSel.value as Theme, texts: texts(), from, to }, replay ?? undefined);
    describe();
    savePrefs({ layout: L.id, format: fmtSel.value === "video" ? "video" : "gif", seconds: Number(secSel.value), theme: themeSel.value as Theme });
  };

  el.querySelectorAll("input, select").forEach((c) => c.addEventListener(c.matches("input[type=text]") ? "input" : "change", preview));

  const close = () => {
    running?.abort();
    clearResult();
    el.close();
    el.remove();
    if (dialog === el) dialog = null;
  };
  el.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", (e) => {
    if (running && b.id === "export-cancel") {
      e.preventDefault();
      running.abort();
      return;
    }
    close();
  }));
  el.addEventListener("cancel", (e) => {
    e.preventDefault();
    close();
  });
  // Keep Escape and shortcuts away from the page underneath.
  el.addEventListener("keydown", (e) => e.stopPropagation());
  el.addEventListener("click", (e) => {
    if (e.target === el) close();
  });

  go.addEventListener("click", async () => {
    if (running) return;
    const L = layout();
    const f = format();
    const [from, to] = range();
    const job = { source: source(), layout: L, format: f, theme: themeSel.value as Theme, texts: texts(), from, to, seconds: Number(secSel.value) };
    running = new AbortController();
    el.classList.add("busy");
    go.disabled = true;
    q<HTMLButtonElement>("#export-cancel").textContent = "Cancel";
    progress.classList.remove("hidden");
    bar.style.width = "0%";
    download.classList.add("hidden");
    status.textContent = "Rendering…";
    const t0 = performance.now();
    let res: EncodeResult;
    try {
      res = await encode({
        ...job,
        signal: running.signal,
        onProgress: (p) => {
          bar.style.width = `${(p * 100).toFixed(1)}%`;
          status.textContent = `Rendering… ${Math.round(p * 100)}%`;
        },
      });
    } catch (e) {
      status.textContent = e instanceof DOMException && e.name === "AbortError" ? "Cancelled." : `Export failed: ${e instanceof Error ? e.message : e}`;
      return;
    } finally {
      running = null;
      el.classList.remove("busy");
      go.disabled = false;
      q<HTMLButtonElement>("#export-cancel").textContent = "Close";
      progress.classList.add("hidden");
    }
    resultUrl = URL.createObjectURL(res.blob);
    canvas.classList.add("hidden");
    if (res.format === "gif") {
      img.src = resultUrl;
      img.classList.remove("hidden");
    } else {
      video.src = resultUrl;
      video.classList.remove("hidden");
      void video.play().catch(() => undefined);
    }
    const ext = res.format;
    download.href = resultUrl;
    download.download = `timestral-${slug(job.texts.title)}-${L.id}.${ext}`;
    download.textContent = `Download ${ext.toUpperCase()} · ${mb(res.blob.size)}`;
    download.classList.remove("hidden");
    const took = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent =
      res.format === "gif" && res.blob.size > GIF_SOFT_LIMIT
        ? `Over 15 MB, the GIF limit on X. A shorter length, the Post layout, or video keeps it smaller.`
        : `${res.frames} frames in ${took} s.`;
    download.focus();
  });

  el.showModal();
  preview();
  void (async () => {
    video0 = await videoSupport(layout());
    const opt = fmtSel.querySelector<HTMLOptionElement>('option[value="video"]')!;
    if (!video0) {
      opt.disabled = true;
      opt.textContent = "Video (not supported in this browser)";
      if (fmtSel.value === "video") fmtSel.value = "gif";
    } else opt.textContent = video0 === "mp4" ? "MP4 video" : "WebM video";
    describe();
  })();
}
