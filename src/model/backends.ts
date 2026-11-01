/**
 * Pluggable forecast backends. Every backend returns the same thing: `(H, Q)` quantiles on
 * Timestral's 21 levels, so the conformal engine does not care where they came from.
 * Models trained on fewer levels (Chronos-Bolt: 0.1, ..., 0.9) are mapped onto that grid by
 * `toQuantileGrid`.
 *
 * - `BrowserBackend`: the TypeScript ports (Timestral, Chronos-Bolt) in a web worker (works
 *   on GitHub Pages).
 * - `ServerBackend`: `server/serve.py`, the official PyTorch packages on localhost.
 * - `NaiveBackend`: a model-free baseline (last value plus empirical k-step changes), so the
 *   dashboard is usable before any weights are downloaded.
 */

import { FAMILY_NAMES, type ModelFamily, familyOf } from "./registry";
import type { WorkerIn, WorkerOut } from "./worker";

export const TIMESTRAL_LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];

/**
 * Map `(H, Q)` quantiles on `levels` onto `grid`. Inside the model's range this interpolates
 * linearly, as `ChronosBoltPipeline.predict_quantiles` does. Beyond it (Chronos-Bolt has no
 * 1%, 5%, 95% or 99% quantile) each tail is extended with a Gaussian shape fitted to the
 * outermost quantile and the median on that side, instead of clamping: clamping would turn
 * the native 90% band into an 80% one. Crossing quantiles (Chronos-Bolt does not prevent them)
 * are sorted first.
 */
export function toQuantileGrid(q: Float32Array, horizon: number, levels: number[], grid: number[] = TIMESTRAL_LEVELS): Float32Array {
  const Q = levels.length, G = grid.length;
  if (Q === G && levels.every((l, i) => Math.abs(l - grid[i]) < 1e-9)) return q;
  const lo = levels[0], hi = levels[Q - 1];
  const zLo = normInv(lo), zHi = normInv(hi);
  const out = new Float32Array(horizon * G);
  const row = new Float32Array(Q);
  const at = (p: number): number => {
    let i = 1;
    while (i < Q - 1 && levels[i] < p) i++;
    const w = (p - levels[i - 1]) / (levels[i] - levels[i - 1]);
    return row[i - 1] * (1 - w) + row[i] * w;
  };
  for (let t = 0; t < horizon; t++) {
    row.set(q.subarray(t * Q, (t + 1) * Q));
    row.sort();
    const med = lo < 0.5 && hi > 0.5 ? at(0.5) : NaN;
    // Spread per unit of z on each side; zero (clamping) when the tail cannot be fitted.
    const sLo = Number.isFinite(med) ? (med - row[0]) / -zLo : 0;
    const sHi = Number.isFinite(med) ? (row[Q - 1] - med) / zHi : 0;
    for (let g = 0; g < G; g++) {
      const p = grid[g];
      let v: number;
      if (p < lo) v = row[0] + sLo * (normInv(p) - zLo);
      else if (p > hi) v = row[Q - 1] + sHi * (normInv(p) - zHi);
      else v = at(p);
      out[t * G + g] = v;
    }
  }
  return out;
}

/** Clip quantiles at 0 when every observed context value is non-negative (Timestral's rule). */
export function clipNonnegative(q: Float32Array, context: ArrayLike<number>): void {
  let any = false;
  for (let t = 0; t < context.length; t++) {
    const v = context[t];
    if (!Number.isFinite(v)) continue;
    if (v < 0) return;
    any = true;
  }
  if (any) for (let i = 0; i < q.length; i++) if (q[i] < 0) q[i] = 0;
}

/** Bring a result onto the shared grid; only results off the grid are touched. */
function adapt(res: ForecastResult, req: ForecastRequest): ForecastResult {
  if (res.levels.length === TIMESTRAL_LEVELS.length && res.levels.every((l, i) => Math.abs(l - TIMESTRAL_LEVELS[i]) < 1e-9)) return res;
  const quantiles = toQuantileGrid(res.quantiles, res.horizon, res.levels);
  if (req.clip) clipNonnegative(quantiles, req.context.subarray(Math.max(0, req.context.length - req.maxContext)));
  return { ...res, quantiles, levels: TIMESTRAL_LEVELS };
}

export interface ForecastRequest {
  context: Float64Array;
  horizon: number;
  symmetric: boolean;
  clip: boolean;
  maxContext: number;
}

export interface ForecastResult {
  quantiles: Float32Array;
  levels: number[];
  horizon: number;
  ms: number;
}

export interface Backend {
  readonly kind: "browser" | "server" | "naive";
  readonly label: string;
  /** Name of the model whose native quantiles this backend serves, e.g. "Chronos-Bolt". */
  readonly modelName: string;
  readonly ready: boolean;
  forecast(req: ForecastRequest): Promise<ForecastResult>;
  dispose(): void;
}

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
}

export class BrowserBackend implements Backend {
  readonly kind = "browser" as const;
  label = "Timestral (browser)";
  family: ModelFamily = "timestral";
  ready = false;
  numParams = 0;
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { req: ForecastRequest; resolve: (r: ForecastResult) => void; reject: (e: Error) => void }>();
  private loadWaiter: { resolve: () => void; reject: (e: Error) => void } | null = null;
  onProgress: (p: LoadProgress) => void = () => {};
  onCached: (repos: string[]) => void = () => {};

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<WorkerOut>) => this.handle(ev.data);
    this.send({ type: "listCached" });
  }

  get modelName(): string {
    return FAMILY_NAMES[this.family];
  }

  private send(m: WorkerIn, transfer: Transferable[] = []) {
    this.worker.postMessage(m, transfer);
  }

  private handle(m: WorkerOut) {
    switch (m.type) {
      case "progress":
        this.onProgress(m);
        break;
      case "ready":
        this.ready = true;
        this.label = m.label;
        this.family = m.family;
        this.numParams = m.numParams;
        this.loadWaiter?.resolve();
        this.loadWaiter = null;
        this.send({ type: "listCached" });
        break;
      case "result": {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        p?.resolve(adapt({ quantiles: m.quantiles, levels: m.levels, horizon: m.horizon, ms: m.ms }, p.req));
        break;
      }
      case "error":
        if (m.id !== undefined) {
          const p = this.pending.get(m.id);
          this.pending.delete(m.id);
          p?.reject(new Error(m.message));
        } else {
          this.loadWaiter?.reject(new Error(m.message));
          this.loadWaiter = null;
        }
        break;
      case "cached":
        this.onCached(m.repos);
        break;
    }
  }

  private waitLoad(): Promise<void> {
    this.ready = false;
    return new Promise((resolve, reject) => (this.loadWaiter = { resolve, reject }));
  }

  load(repo: string): Promise<void> {
    const p = this.waitLoad();
    this.send({ type: "load", repo });
    return p;
  }

  loadFiles(label: string, config: Record<string, unknown>, weights: ArrayBuffer): Promise<void> {
    const p = this.waitLoad();
    this.send({ type: "loadBuffers", label, config, weights }, [weights]);
    return p;
  }

  evict(repo: string) {
    this.send({ type: "evict", repo });
  }

  forecast(req: ForecastRequest): Promise<ForecastResult> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { req, resolve, reject });
      const context = req.context.slice();
      this.send({ type: "forecast", id, context, horizon: req.horizon, symmetric: req.symmetric, clip: req.clip, maxContext: req.maxContext }, [context.buffer]);
    });
  }

  dispose() {
    this.worker.terminate();
  }
}

export class ServerBackend implements Backend {
  readonly kind = "server" as const;
  ready = false;
  label: string;
  models: string[] = [];
  readonly modelName: string;
  constructor(public url: string, public repo: string) {
    this.label = `${repo} (local server)`;
    this.modelName = FAMILY_NAMES[familyOf(repo) ?? "timestral"];
  }

  async connect(): Promise<string[]> {
    const res = await fetch(`${this.url.replace(/\/$/, "")}/health`);
    if (!res.ok) throw new Error(`server answered HTTP ${res.status}`);
    const body = (await res.json()) as { models: string[]; device: string };
    this.models = body.models;
    this.ready = true;
    this.label = `${this.repo} (server, ${body.device})`;
    return body.models;
  }

  async forecast(req: ForecastRequest): Promise<ForecastResult> {
    const t0 = performance.now();
    const res = await fetch(`${this.url.replace(/\/$/, "")}/forecast`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.repo,
        context: Array.from(req.context.slice(-req.maxContext)),
        horizon: req.horizon,
        symmetric_averaging: req.symmetric,
        clip_nonnegative: req.clip,
      }),
    });
    if (!res.ok) throw new Error(`server: HTTP ${res.status} ${await res.text()}`);
    const body = (await res.json()) as { quantiles: number[][]; levels: number[] };
    const Q = body.levels.length;
    const q = new Float32Array(req.horizon * Q);
    body.quantiles.forEach((row, t) => q.set(row, t * Q));
    return adapt({ quantiles: q, levels: body.levels, horizon: req.horizon, ms: performance.now() - t0 }, req);
  }

  dispose() {}
}

/**
 * Model-free baseline: median = last value, quantiles = last value + empirical quantiles of
 * the observed k-step changes `y[t] - y[t-k]` (falling back to a Gaussian random walk).
 */
/** Points used for the naive baseline's local drift, and its per-step damping factor. */
export const NAIVE_DRIFT_WINDOW = 8;
export const NAIVE_DAMPING = 0.9;

/**
 * Naive baseline (no model): damped local drift for the median, empirical k-step changes
 * for the spread. Lets you try the app without downloading weights.
 */
export class NaiveBackend implements Backend {
  readonly kind = "naive" as const;
  readonly label = "Naive baseline (no model)";
  readonly modelName = "Naive";
  readonly ready = true;

  async forecast(req: ForecastRequest): Promise<ForecastResult> {
    const t0 = performance.now();
    const y = req.context.slice(-req.maxContext);
    const n = y.length;
    const levels = TIMESTRAL_LEVELS;
    const Q = levels.length;
    const q = new Float32Array(req.horizon * Q);
    const last = n ? y[n - 1] : 0;
    let sd1 = 0;
    for (let t = 1; t < n; t++) sd1 += (y[t] - y[t - 1]) ** 2;
    sd1 = n > 1 ? Math.sqrt(sd1 / (n - 1)) : 1;
    // Local drift: mean step over the last few points, extrapolated with damping so a
    // ramp is continued but a turning curve is not shot off to infinity.
    const W = Math.min(n - 1, NAIVE_DRIFT_WINDOW);
    const slope = W > 0 ? (y[n - 1] - y[n - 1 - W]) / W : 0;
    let damp = 0;
    for (let k = 1; k <= req.horizon; k++) {
      damp += NAIVE_DAMPING ** k;
      const center = last + slope * damp;
      const diffs: number[] = [];
      for (let t = k; t < n; t++) diffs.push(y[t] - y[t - k]);
      diffs.sort((a, b) => a - b);
      // The spread comes from the k-step changes seen so far, recentred on the drift path.
      const med = diffs.length ? diffs[Math.floor((diffs.length - 1) / 2)] : 0;
      for (let i = 0; i < Q; i++) {
        let v: number;
        if (diffs.length >= 20) {
          const pos = levels[i] * (diffs.length - 1);
          const lo = Math.floor(pos);
          const w = pos - lo;
          v = diffs[lo] * (1 - w) + diffs[Math.min(lo + 1, diffs.length - 1)] * w - med;
        } else {
          v = sd1 * Math.sqrt(k) * normInv(levels[i]);
        }
        q[(k - 1) * Q + i] = center + v;
      }
      // Pin the median exactly on the drift path.
      const mi = levels.indexOf(0.5);
      const shift = center - q[(k - 1) * Q + mi];
      for (let i = 0; i < Q; i++) q[(k - 1) * Q + i] += shift;
    }
    return { quantiles: q, levels, horizon: req.horizon, ms: performance.now() - t0 };
  }

  dispose() {}
}

/** Inverse standard normal CDF (Acklam's rational approximation). */
export function normInv(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -normInv(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
