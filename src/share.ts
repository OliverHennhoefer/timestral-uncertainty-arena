/**
 * Shareable scenario links: the series (or the preset and seed that made it), the model and
 * every setting that changes the ranking, packed into the URL hash.
 *
 * Format: `#s=1z<base64url>` where the payload is deflate-raw compressed (`1p` when the browser
 * has no CompressionStream). Inside: the header as UTF-8 JSON, then, for a drawn series, a NUL
 * byte and the values in hundredths as zigzag varint deltas. The hash never reaches the server,
 * so links work on GitHub Pages and stay private to whoever holds them.
 */

import type { RankBy, ScoreType, Settings } from "./engine/engine";

export type Align = "full" | "calibrate" | "cut";
export type ModelRef = { backend: "naive" } | { backend: "browser" | "server"; repo: string | null };
export type SeriesRef =
  | { kind: "empty" }
  /** The first `n` points of a preset stream; `hash` detects a preset whose data changed since. */
  | { kind: "preset"; id: string; seed: number; n: number; hash: number }
  | { kind: "values"; values: number[]; offset: number };

export interface Scenario {
  series: SeriesRef;
  model: ModelRef;
  horizon: number;
  maxContext: number;
  symmetric: boolean;
  clip: boolean;
  align: Align;
  /** The calibrated leads to restore when leaving "calibrate". */
  leadsBefore: number | null;
  settings: Settings;
  rankBy: RankBy;
  rankLead: number;
  focus: string;
}

export const HASH_KEY = "s";
const VERSION = "1";
const SCALE = 100; // drawn points are rounded to hundredths

/** FNV-1a over the values in hundredths. */
export function seriesHash(values: readonly number[]): number {
  let h = 0x811c9dc5;
  for (const v of values) {
    let q = Math.round(v * SCALE) | 0;
    for (let b = 0; b < 4; b++, q >>>= 8) h = Math.imul(h ^ (q & 0xff), 0x01000193);
  }
  return h >>> 0;
}

// ---- series bytes -------------------------------------------------------------------------

export function packValues(values: readonly number[]): Uint8Array {
  const out: number[] = [];
  let prev = 0;
  for (const v of values) {
    const q = Math.round(v * SCALE);
    let z = q - prev;
    prev = q;
    z = z >= 0 ? 2 * z : -2 * z - 1; // zigzag, kept in doubles so large jumps cannot overflow
    while (z >= 0x80) {
      out.push((z % 0x80) | 0x80);
      z = Math.floor(z / 0x80);
    }
    out.push(z);
  }
  return Uint8Array.from(out);
}

export function unpackValues(bytes: Uint8Array): number[] {
  const values: number[] = [];
  let prev = 0;
  for (let i = 0; i < bytes.length; ) {
    let z = 0, mul = 1, b: number;
    do {
      if (i >= bytes.length) throw new Error("truncated series");
      b = bytes[i++];
      z += (b & 0x7f) * mul;
      mul *= 0x80;
    } while (b & 0x80);
    prev += z % 2 ? -(z + 1) / 2 : z / 2;
    values.push(prev / SCALE);
  }
  return values;
}

// ---- base64url and compression -----------------------------------------------------------

function toBase64Url(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(s: string): Uint8Array {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const buf = await new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(stream)).arrayBuffer();
  return new Uint8Array(buf);
}

const canCompress = () => typeof CompressionStream !== "undefined" && typeof DecompressionStream !== "undefined";

// ---- scenario <-> hash ---------------------------------------------------------------------

/** Compact wire form; short keys keep the links short. */
interface Wire {
  src: ["e"] | ["p", string, number, number, number] | ["v", number];
  m: ["n"] | ["b" | "s", string | null];
  h: number;
  mc: number;
  sym: 0 | 1;
  clip: 0 | 1;
  al: Align;
  lb?: number;
  a: number;
  k: number;
  sc: ScoreType;
  w: number;
  wu: number;
  r: number;
  /** Enabled method ids. */
  on: string[];
  /** Parameters of the enabled methods. */
  p: Record<string, Record<string, number>>;
  rb: RankBy;
  rl: number;
  f: string;
}

export async function encodeScenario(s: Scenario): Promise<string> {
  const on = Object.keys(s.settings.enabled).filter((id) => s.settings.enabled[id]);
  const wire: Wire = {
    src: s.series.kind === "empty" ? ["e"] : s.series.kind === "preset" ? ["p", s.series.id, s.series.seed, s.series.n, s.series.hash] : ["v", s.series.offset],
    m: s.model.backend === "naive" ? ["n"] : [s.model.backend === "browser" ? "b" : "s", s.model.repo],
    h: s.horizon,
    mc: s.maxContext,
    sym: s.symmetric ? 1 : 0,
    clip: s.clip ? 1 : 0,
    al: s.align,
    ...(s.leadsBefore !== null ? { lb: s.leadsBefore } : {}),
    a: s.settings.alpha,
    k: s.settings.evalLeads,
    sc: s.settings.scoreType,
    w: s.settings.window,
    wu: s.settings.warmup,
    r: s.settings.rolling,
    on,
    p: Object.fromEntries(on.filter((id) => s.settings.params[id] && Object.keys(s.settings.params[id]).length).map((id) => [id, s.settings.params[id]])),
    rb: s.rankBy,
    rl: s.rankLead,
    f: s.focus,
  };
  const head = new TextEncoder().encode(JSON.stringify(wire));
  let raw = head;
  if (s.series.kind === "values") {
    const body = packValues(s.series.values);
    raw = new Uint8Array(head.length + 1 + body.length);
    raw.set(head);
    raw.set(body, head.length + 1);
  }
  if (canCompress()) return `${VERSION}z${toBase64Url(await pipe(raw, new CompressionStream("deflate-raw")))}`;
  return `${VERSION}p${toBase64Url(raw)}`;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const intIn = (v: unknown, lo: number, hi: number, fallback: number) => (isNum(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : fallback);

/**
 * Decode a link payload onto `defaults` (the current settings): known methods only, numbers
 * clamped to what the controls accept. Throws on a payload that is not a scenario.
 */
export async function decodeScenario(code: string, defaults: Settings): Promise<Scenario> {
  if (code[0] !== VERSION) throw new Error("this link was made by a newer version of the arena");
  if (code[1] !== "z" && code[1] !== "p") throw new Error("unknown link encoding");
  if (code[1] === "z" && !canCompress()) throw new Error("this browser cannot decompress the link");
  let raw: Uint8Array;
  let w: Partial<Wire>;
  try {
    raw = fromBase64Url(code.slice(2));
    if (code[1] === "z") raw = await pipe(raw, new DecompressionStream("deflate-raw"));
    const end = raw.indexOf(0);
    w = JSON.parse(new TextDecoder().decode(end < 0 ? raw : raw.subarray(0, end))) as Partial<Wire>;
  } catch {
    throw new Error("the link is damaged or incomplete");
  }
  const nul = raw.indexOf(0);
  if (!w || typeof w !== "object" || !Array.isArray(w.src)) throw new Error("not a scenario");

  let series: SeriesRef = { kind: "empty" };
  if (w.src[0] === "p" && typeof w.src[1] === "string") {
    series = { kind: "preset", id: w.src[1], seed: intIn(w.src[2], 0, 2 ** 32, 1), n: intIn(w.src[3], 0, 1e6, 0), hash: Number(w.src[4]) >>> 0 };
  } else if (w.src[0] === "v") {
    series = { kind: "values", values: nul < 0 ? [] : unpackValues(raw.subarray(nul + 1)), offset: intIn(w.src[1], 0, 1e7, 0) };
  }

  const m = Array.isArray(w.m) ? w.m : ["n"];
  const repo = typeof m[1] === "string" ? m[1] : null;
  const model: ModelRef = m[0] === "b" ? { backend: "browser", repo } : m[0] === "s" ? { backend: "server", repo } : { backend: "naive" };

  const known = Object.keys(defaults.enabled);
  const on = new Set(Array.isArray(w.on) ? w.on : []);
  const params = { ...defaults.params };
  for (const [id, p] of Object.entries(w.p ?? {})) {
    if (!known.includes(id) || !p || typeof p !== "object") continue;
    params[id] = { ...params[id], ...Object.fromEntries(Object.entries(p).filter(([, v]) => isNum(v))) };
  }
  const scoreTypes: ScoreType[] = ["cqr", "cqr_asym", "abs", "scaled"];
  const horizon = [32, 64, 128].includes(w.h as number) ? (w.h as number) : 64;
  const settings: Settings = {
    alpha: isNum(w.a) ? Math.min(0.5, Math.max(0.02, Math.round(w.a * 100) / 100)) : defaults.alpha,
    evalLeads: intIn(w.k, 1, 128, defaults.evalLeads),
    scoreType: scoreTypes.includes(w.sc as ScoreType) ? (w.sc as ScoreType) : defaults.scoreType,
    window: intIn(w.w, 10, 2000, defaults.window),
    warmup: intIn(w.wu, 1, 500, defaults.warmup),
    rolling: intIn(w.r, 5, 5000, defaults.rolling),
    enabled: Object.fromEntries(known.map((id) => [id, on.has(id)])),
    params,
  };
  const align: Align = w.al === "full" || w.al === "calibrate" ? w.al : "cut";
  return {
    series,
    model,
    horizon,
    maxContext: intIn(w.mc, 32, 32768, 256),
    symmetric: w.sym !== 0,
    clip: w.clip !== 0,
    align,
    leadsBefore: w.lb === undefined ? null : intIn(w.lb, 1, 128, defaults.evalLeads),
    settings,
    rankBy: (["interval_score", "validity_then_width", "rolling_gap"] as RankBy[]).includes(w.rb as RankBy) ? (w.rb as RankBy) : "interval_score",
    rankLead: intIn(w.rl, 0, settings.evalLeads, 0),
    focus: typeof w.f === "string" ? w.f : "aci",
  };
}

/** The scenario payload in a location hash such as `#s=1z...`, or null. */
export function scenarioFromHash(hash: string): string | null {
  const m = /^#?(?:.*&)?s=([0-9][zp][A-Za-z0-9_-]*)/.exec(hash);
  return m ? m[1] : null;
}
