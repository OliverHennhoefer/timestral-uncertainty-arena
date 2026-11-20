/**
 * Export presets: an aspect ratio, a logical canvas the layout is drawn in (so type sizes
 * are fixed in logical pixels and stay legible after a platform shrinks the file), and the
 * pixel sizes written for video and for GIF. GIFs are smaller because every frame is a full
 * palette image; video codecs compress the scrolling chart far better.
 */

export type LayoutId = "slides" | "social" | "square" | "portrait" | "story";

export interface Layout {
  id: LayoutId;
  label: string;
  /** Where it is meant to go, shown under the label. */
  use: string;
  /** Logical drawing size. */
  w: number;
  h: number;
  video: [number, number];
  gif: [number, number];
  /** Observed steps kept on screen before the forecast. */
  history: number;
}

export const LAYOUTS: Layout[] = [
  { id: "slides", label: "Slides", use: "16:9 · PowerPoint, Keynote", w: 1600, h: 900, video: [1920, 1080], gif: [1280, 720], history: 240 },
  { id: "social", label: "Post", use: "16:9 · X, LinkedIn, Bluesky", w: 1200, h: 675, video: [1280, 720], gif: [800, 450], history: 180 },
  { id: "square", label: "Square", use: "1:1 · feeds", w: 1080, h: 1080, video: [1080, 1080], gif: [720, 720], history: 150 },
  { id: "portrait", label: "Portrait", use: "4:5 · mobile feeds", w: 1080, h: 1350, video: [1080, 1350], gif: [640, 800], history: 140 },
  { id: "story", label: "Story", use: "9:16 · Stories, Reels, Shorts", w: 1080, h: 1920, video: [1080, 1920], gif: [540, 960], history: 120 },
];

export const layoutById = (id: string): Layout => LAYOUTS.find((l) => l.id === id) ?? LAYOUTS[0];

export type ExportFormat = "gif" | "mp4" | "webm";

/** Frame rates: GIF delays are in 1/100 s, so 60 ms frames (16.7 fps) play at the same speed everywhere. */
export const GIF_FRAME_MS = 60;
export const VIDEO_FPS = 30;
/** The last frame is held so a looping file does not jump straight back to the start. */
export const HOLD_SECONDS = 2.5;
/** Upload limits worth warning about (X: 15 MB for GIFs). */
export const GIF_SOFT_LIMIT = 15e6;
export const MAX_GIF_FRAMES = 360;

export interface Schedule {
  /** Series index shown by each frame. */
  times: number[];
  /** Seconds per frame. */
  frameSec: number;
}

/**
 * Spread the steps [t0, t1] over at most `seconds` of animation. A run shorter than that
 * plays one step per frame instead of being stretched.
 */
export function schedule(t0: number, t1: number, seconds: number, format: ExportFormat): Schedule {
  const frameSec = format === "gif" ? GIF_FRAME_MS / 1000 : 1 / VIDEO_FPS;
  const steps = Math.max(1, t1 - t0 + 1);
  let frames = Math.max(1, Math.min(steps, Math.round(seconds / frameSec)));
  if (format === "gif") frames = Math.min(frames, MAX_GIF_FRAMES);
  const times: number[] = [];
  for (let i = 0; i < frames; i++) times.push(t0 + Math.round(((i + 1) * steps) / frames) - 1);
  return { times, frameSec };
}
