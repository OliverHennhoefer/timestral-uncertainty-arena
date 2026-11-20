/**
 * MP4 (H.264) or WebM (VP9) writing with WebCodecs, via mediabunny's named exports so it
 * tree-shakes.
 *
 * Frames are converted to I420 here, in limited ("TV") range with BT.709 coefficients and
 * tags, instead of handing the encoder the canvas. A canvas frame is full-range sRGB, which
 * browsers encode as yuvj420p; some players and upload pipelines (PowerPoint on Windows,
 * Instagram, X) expect the yuv420p that cameras and editors produce.
 */

import { BufferTarget, Mp4OutputFormat, Output, QUALITY_HIGH, VideoSample, VideoSampleSource, WebMOutputFormat } from "mediabunny";
import { VIDEO_FPS } from "./layouts";

export interface VideoWriter {
  /** Encode the canvas as the frame at `timestamp` (seconds); `repeat` reuses the last capture. */
  add(timestamp: number, duration: number, repeat?: boolean): Promise<void>;
  finish(): Promise<ArrayBuffer>;
  cancel(): Promise<void>;
}

const BT709_LIMITED: VideoColorSpaceInit = { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false };

/** RGBA to planar I420 (Y, then U and V at half resolution), BT.709 limited range. */
export function rgbaToI420(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number, out?: Uint8Array): Uint8Array {
  const cw = w >> 1, ch = h >> 1;
  const buf = out ?? new Uint8Array(w * h + 2 * cw * ch);
  const U = w * h, V = U + cw * ch;
  const ky = 219 / 255, kc = 224 / 255 / 4;
  for (let y = 0; y < ch; y++) {
    const r0 = 2 * y * w, r1 = r0 + w;
    for (let x = 0; x < cw; x++) {
      const i0 = r0 + 2 * x, i1 = r1 + 2 * x;
      const a = i0 * 4, b = a + 4, c = i1 * 4, d = c + 4;
      const R0 = rgba[a], G0 = rgba[a + 1], B0 = rgba[a + 2];
      const R1 = rgba[b], G1 = rgba[b + 1], B1 = rgba[b + 2];
      const R2 = rgba[c], G2 = rgba[c + 1], B2 = rgba[c + 2];
      const R3 = rgba[d], G3 = rgba[d + 1], B3 = rgba[d + 2];
      buf[i0] = 16.5 + (0.2126 * R0 + 0.7152 * G0 + 0.0722 * B0) * ky;
      buf[i0 + 1] = 16.5 + (0.2126 * R1 + 0.7152 * G1 + 0.0722 * B1) * ky;
      buf[i1] = 16.5 + (0.2126 * R2 + 0.7152 * G2 + 0.0722 * B2) * ky;
      buf[i1 + 1] = 16.5 + (0.2126 * R3 + 0.7152 * G3 + 0.0722 * B3) * ky;
      const r = R0 + R1 + R2 + R3, g = G0 + G1 + G2 + G3, bl = B0 + B1 + B2 + B3;
      const j = y * cw + x;
      buf[U + j] = 128.5 + (-0.1146 * r - 0.3854 * g + 0.5 * bl) * kc;
      buf[V + j] = 128.5 + (0.5 * r - 0.4542 * g - 0.0458 * bl) * kc;
    }
  }
  return buf;
}

export async function createVideoWriter(canvas: HTMLCanvasElement, webm: boolean): Promise<VideoWriter> {
  const { width: w, height: h } = canvas;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const target = new BufferTarget();
  const output = new Output({ format: webm ? new WebMOutputFormat() : new Mp4OutputFormat({ fastStart: "in-memory" }), target });
  const source = new VideoSampleSource({ codec: webm ? "vp9" : "avc", bitrate: QUALITY_HIGH, keyFrameInterval: 2 });
  output.addVideoTrack(source, { frameRate: VIDEO_FPS });
  await output.start();
  let yuv: Uint8Array | null = null;
  return {
    async add(timestamp, duration, repeat = false) {
      if (!repeat || !yuv) yuv = rgbaToI420(ctx.getImageData(0, 0, w, h).data, w, h, yuv ?? undefined);
      const sample = new VideoSample(yuv, { format: "I420", codedWidth: w, codedHeight: h, timestamp, duration, colorSpace: BT709_LIMITED });
      try {
        await source.add(sample);
      } finally {
        sample.close();
      }
    },
    async finish() {
      await output.finalize();
      return target.buffer!;
    },
    async cancel() {
      await output.cancel().catch(() => undefined);
    },
  };
}
