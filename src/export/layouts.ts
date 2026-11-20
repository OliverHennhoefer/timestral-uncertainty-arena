/**
 * Export presets: an aspect ratio, a logical canvas the layout is drawn in (so type sizes
 * are fixed in logical pixels and stay legible after a platform shrinks the file), and the
 * pixel sizes written for video and for GIF. GIFs are smaller because every frame is a full
 * palette image; video codecs compress the scrolling chart far better.
 */

export type LayoutId = "slides" | "social" | "square" | "portrait" | "story";

/** Type sizes and line weight in logical pixels. */
export interface Typography {
  title: number;
  sub: number;
  /** The big coverage numbers and their labels. */
  value: number;
  label: number;
  tick: number;
  legend: number;
  brand: number;
  /** Multiplier on stroke widths. */
  line: number;
}

/** Content box insets; outside it only the background, brand line and progress bar. */
export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

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
  type: Typography;
  safe: Insets;
  /** Lines the subtitle may wrap to. */
  subLines: number;
  /** Small 16:9 feed cards drop the self-explanatory legend keys (observed, forecast). */
  shortLegend: boolean;
}

/*
 * Sizing. Feed media is seen on phones about 390 CSS px wide (X indents it to roughly
 * 310 px), so 1080-wide layouts use labels of at least 30 logical px (about 11 px on a
 * phone) and titles near 60 (about 22 px). Slides target 18 pt on a projected 13.3 in slide,
 * which is 30 px of 1600. The Story layout keeps its content out of the Stories / Reels
 * overlays: about 250 px at the top (progress bar, account) and 380 px at the bottom
 * (reply bar, caption), plus the Reels buttons on the right.
 */
const PHONE: Typography = { title: 60, sub: 32, value: 66, label: 30, tick: 30, legend: 32, brand: 26, line: 1.5 };

export const LAYOUTS: Layout[] = [
  {
    id: "slides", label: "Slides", use: "16:9 · PowerPoint, Keynote, Google Slides",
    w: 1600, h: 900, video: [1920, 1080], gif: [1280, 720], history: 240,
    type: { title: 60, sub: 34, value: 68, label: 30, tick: 28, legend: 30, brand: 24, line: 1.2 },
    safe: { top: 56, right: 64, bottom: 44, left: 64 }, subLines: 1, shortLegend: false,
  },
  {
    id: "social", label: "Post", use: "16:9 · X, LinkedIn, Bluesky",
    w: 1200, h: 675, video: [1280, 720], gif: [800, 450], history: 180,
    type: { title: 60, sub: 34, value: 68, label: 34, tick: 30, legend: 34, brand: 26, line: 1.6 },
    safe: { top: 44, right: 48, bottom: 36, left: 48 }, subLines: 1, shortLegend: true,
  },
  {
    id: "square", label: "Square", use: "1:1 · X, LinkedIn, Instagram feed",
    w: 1080, h: 1080, video: [1080, 1080], gif: [720, 720], history: 150,
    type: PHONE, safe: { top: 64, right: 64, bottom: 52, left: 64 }, subLines: 2, shortLegend: false,
  },
  {
    id: "portrait", label: "Portrait", use: "4:5 · Instagram, LinkedIn, X on phones",
    w: 1080, h: 1350, video: [1080, 1350], gif: [640, 800], history: 140,
    type: PHONE, safe: { top: 72, right: 64, bottom: 56, left: 64 }, subLines: 2, shortLegend: false,
  },
  {
    id: "story", label: "Story", use: "9:16 · Stories, Reels, Shorts, TikTok",
    w: 1080, h: 1920, video: [1080, 1920], gif: [540, 960], history: 120,
    type: { ...PHONE, value: 72 }, safe: { top: 250, right: 120, bottom: 380, left: 72 }, subLines: 2, shortLegend: false,
  },
];

export const layoutById = (id: string): Layout => LAYOUTS.find((l) => l.id === id) ?? LAYOUTS[0];

export type ExportFormat = "gif" | "mp4" | "webm";

/** Frame rates: GIF delays are in 1/100 s, so 60 ms frames (16.7 fps) play at the same speed everywhere. */
export const GIF_FRAME_MS = 60;
export const VIDEO_FPS = 30;
/** The last frame is held so a looping file does not jump straight back to the start. */
export const HOLD_SECONDS = 2.5;
/** X's limits for animated GIFs: 15 MB, 1280 × 1080 and 350 frames (the held last frame included). */
export const GIF_SOFT_LIMIT = 15e6;
export const MAX_GIF_FRAMES = 340;

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
