/** MP4 (H.264) or WebM (VP9) writing with WebCodecs, via mediabunny's named exports so it tree-shakes. */

import { BufferTarget, CanvasSource, Mp4OutputFormat, Output, QUALITY_HIGH, WebMOutputFormat } from "mediabunny";
import { VIDEO_FPS } from "./layouts";

export interface VideoWriter {
  /** Capture the canvas as the frame at `timestamp` (seconds). */
  add(timestamp: number, duration: number): Promise<void>;
  finish(): Promise<ArrayBuffer>;
  cancel(): Promise<void>;
}

export async function createVideoWriter(canvas: HTMLCanvasElement, webm: boolean): Promise<VideoWriter> {
  const target = new BufferTarget();
  const output = new Output({ format: webm ? new WebMOutputFormat() : new Mp4OutputFormat({ fastStart: "in-memory" }), target });
  const source = new CanvasSource(canvas, { codec: webm ? "vp9" : "avc", bitrate: QUALITY_HIGH, keyFrameInterval: 2 });
  output.addVideoTrack(source, { frameRate: VIDEO_FPS });
  await output.start();
  return {
    add: (timestamp, duration) => source.add(timestamp, duration),
    async finish() {
      await output.finalize();
      return target.buffer!;
    },
    async cancel() {
      await output.cancel().catch(() => undefined);
    },
  };
}
