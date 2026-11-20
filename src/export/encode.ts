/**
 * Renders a run frame by frame into an offscreen canvas and encodes it, entirely in the
 * browser: GIF with gifenc (one global palette, so colours do not flicker between frames),
 * MP4 (H.264) or, where the browser cannot encode H.264, WebM (VP9) with WebCodecs through
 * mediabunny. Encoders are loaded on first use, so the arena itself does not grow.
 */

import { type ExportFormat, GIF_FRAME_MS, HOLD_SECONDS, type Layout, VIDEO_FPS, schedule } from "./layouts";
import { Painter, type Texts, type Theme } from "./render";
import { type ExportSource, SceneReplay } from "./scene";

export interface EncodeJob {
  source: ExportSource;
  layout: Layout;
  format: ExportFormat;
  theme: Theme;
  texts: Texts;
  /** Series range to animate, inclusive. */
  from: number;
  to: number;
  /** Longest animation, before the final hold. */
  seconds: number;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

export interface EncodeResult {
  blob: Blob;
  width: number;
  height: number;
  frames: number;
  seconds: number;
  format: ExportFormat;
}

export function outputSize(layout: Layout, format: ExportFormat): [number, number] {
  return format === "gif" ? layout.gif : layout.video;
}

/** The video container this browser can write, or null without WebCodecs. */
export async function videoSupport(layout: Layout): Promise<"mp4" | "webm" | null> {
  if (typeof VideoEncoder === "undefined") return null;
  const [width, height] = layout.video;
  const ok = async (codec: string) => {
    try {
      return !!(await VideoEncoder.isConfigSupported({ codec, width, height, bitrate: 8e6, framerate: VIDEO_FPS })).supported;
    } catch {
      return false;
    }
  };
  // H.264 High at level 4.0 covers 1920 × 1080 at 30 fps; VP9 profile 0 is the fallback.
  if (await ok("avc1.640028")) return "mp4";
  if (await ok("vp09.00.40.08")) return "webm";
  return null;
}

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  return { canvas, ctx };
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** Draw a single still (the dialog preview) at the end of the range. */
export function drawStill(canvas: HTMLCanvasElement, job: Omit<EncodeJob, "format" | "seconds">, replay?: SceneReplay): SceneReplay {
  const r = replay && replay.time <= job.to ? replay : new SceneReplay(job.source);
  r.advanceTo(job.to);
  const ctx = canvas.getContext("2d")!;
  const painter = new Painter(job.layout, job.theme, job.texts);
  painter.covFrom = job.from;
  painter.covTo = job.to;
  ctx.setTransform(canvas.width / job.layout.w, 0, 0, canvas.height / job.layout.h, 0, 0);
  painter.draw(ctx, r.frame(job.layout.history, job.from, job.to), 1);
  return r;
}

export async function encode(job: EncodeJob): Promise<EncodeResult> {
  const [w, h] = outputSize(job.layout, job.format);
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.setTransform(w / job.layout.w, 0, 0, h / job.layout.h, 0, 0);
  const sched = schedule(job.from, job.to, job.seconds, job.format);
  const frames = sched.times.length;

  const render = (painter: Painter, replay: SceneReplay, i: number) => {
    replay.advanceTo(sched.times[i]);
    painter.draw(ctx, replay.frame(job.layout.history, job.from, job.to), (i + 1) / frames, i === 0 ? 1 : 0.18);
  };
  const newPainter = () => {
    const p = new Painter(job.layout, job.theme, job.texts);
    p.covFrom = job.from;
    p.covTo = job.to;
    return p;
  };
  const aborted = () => {
    if (job.signal?.aborted) throw new DOMException("Export cancelled", "AbortError");
  };

  if (job.format === "gif") {
    const { GIFEncoder, quantize, applyPalette } = await import("gifenc");
    // Pass 1: a palette from a handful of frames across the run (subsampled pixels).
    const samples = Math.min(frames, 8);
    const chunks: Uint8ClampedArray[] = [];
    {
      const replay = new SceneReplay(job.source);
      const painter = newPainter();
      for (let s = 0; s < samples; s++) {
        const i = samples === 1 ? frames - 1 : Math.round((s * (frames - 1)) / (samples - 1));
        render(painter, replay, i);
        const px = ctx.getImageData(0, 0, w, h).data;
        const sub = new Uint8ClampedArray(Math.ceil(px.length / 16) * 4);
        for (let p = 0, q = 0; p < px.length; p += 16, q += 4) {
          sub[q] = px[p];
          sub[q + 1] = px[p + 1];
          sub[q + 2] = px[p + 2];
          sub[q + 3] = 255;
        }
        chunks.push(sub);
        aborted();
      }
    }
    const all = new Uint8ClampedArray(chunks.reduce((n, c) => n + c.length, 0));
    let off = 0;
    for (const c of chunks) (all.set(c, off), (off += c.length));
    const palette = quantize(all, 256);
    job.onProgress?.(0.05);
    await tick();

    // Pass 2: every frame against that palette.
    const gif = GIFEncoder();
    const replay = new SceneReplay(job.source);
    const painter = newPainter();
    for (let i = 0; i < frames; i++) {
      aborted();
      render(painter, replay, i);
      const index = applyPalette(ctx.getImageData(0, 0, w, h).data, palette);
      const last = i === frames - 1;
      gif.writeFrame(index, w, h, { palette: i === 0 ? palette : undefined, delay: last ? Math.round(HOLD_SECONDS * 1000) : GIF_FRAME_MS, repeat: 0 });
      if (i % 2 === 0 || last) {
        job.onProgress?.(0.05 + (0.95 * (i + 1)) / frames);
        await tick();
      }
    }
    gif.finish();
    return { blob: new Blob([gif.bytes()], { type: "image/gif" }), width: w, height: h, frames, seconds: frames * sched.frameSec + HOLD_SECONDS, format: "gif" };
  }

  const { createVideoWriter } = await import("./video");
  const webm = job.format === "webm";
  const writer = await createVideoWriter(canvas, webm);
  const replay = new SceneReplay(job.source);
  const painter = newPainter();
  const dt = 1 / VIDEO_FPS;
  const hold = Math.round(HOLD_SECONDS * VIDEO_FPS);
  const total = frames + hold;
  try {
    for (let i = 0; i < total; i++) {
      aborted();
      if (i < frames) render(painter, replay, i);
      // The hold repeats the last frame instead of one long frame, which some players stretch badly.
      await writer.add(i * dt, dt);
      if (i % 4 === 0) {
        job.onProgress?.((i + 1) / total);
        await tick();
      }
    }
  } catch (e) {
    await writer.cancel();
    throw e;
  }
  const buffer = await writer.finish();
  job.onProgress?.(1);
  return {
    blob: new Blob([buffer], { type: webm ? "video/webm" : "video/mp4" }),
    width: w,
    height: h,
    frames: total,
    seconds: total * dt,
    format: job.format,
  };
}
