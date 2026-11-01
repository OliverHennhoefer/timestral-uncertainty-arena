/**
 * Pluggable forecast backends. Every backend returns the same thing: `(H, Q)` quantiles on
 * Timestral's 21 levels, so the conformal engine does not care where they came from.
 *
 * - `BrowserBackend`: the TypeScript port in a web worker (works on GitHub Pages).
 * - `ServerBackend`: `server/serve.py`, the official PyTorch package on localhost.
 * - `NaiveBackend`: a model-free baseline (last value plus empirical k-step changes), so the
 *   dashboard is usable before any weights are downloaded.
 */

import type { WorkerIn, WorkerOut } from "./worker";

export const TIMESTRAL_LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];

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
  ready = false;
  numParams = 0;
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: ForecastResult) => void; reject: (e: Error) => void }>();
  private loadWaiter: { resolve: () => void; reject: (e: Error) => void } | null = null;
  onProgress: (p: LoadProgress) => void = () => {};
  onCached: (repos: string[]) => void = () => {};

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<WorkerOut>) => this.handle(ev.data);
    this.send({ type: "listCached" });
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
        this.numParams = m.numParams;
        this.loadWaiter?.resolve();
        this.loadWaiter = null;
        this.send({ type: "listCached" });
        break;
      case "result": {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        p?.resolve({ quantiles: m.quantiles, levels: m.levels, horizon: m.horizon, ms: m.ms });
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
      this.pending.set(id, { resolve, reject });
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
  constructor(public url: string, public repo: string) {
    this.label = `${repo} (local server)`;
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
    return { quantiles: q, levels: body.levels, horizon: req.horizon, ms: performance.now() - t0 };
  }

  dispose() {}
}

/**
 * Model-free baseline: median = last value, quantiles = last value + empirical quantiles of
 * the observed k-step changes `y[t] - y[t-k]` (falling back to a Gaussian random walk).
 */
export class NaiveBackend implements Backend {
  readonly kind = "naive" as const;
  readonly label = "Naive baseline (no model)";
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
    for (let k = 1; k <= req.horizon; k++) {
      const diffs: number[] = [];
      for (let t = k; t < n; t++) diffs.push(y[t] - y[t - k]);
      diffs.sort((a, b) => a - b);
      for (let i = 0; i < Q; i++) {
        let v: number;
        if (diffs.length >= 20) {
          const pos = levels[i] * (diffs.length - 1);
          const lo = Math.floor(pos);
          const w = pos - lo;
          v = diffs[lo] * (1 - w) + diffs[Math.min(lo + 1, diffs.length - 1)] * w;
        } else {
          v = sd1 * Math.sqrt(k) * normInv(levels[i]);
        }
        q[(k - 1) * Q + i] = last + v;
      }
      // The median is the last value.
      const mi = levels.indexOf(0.5);
      const shift = last - q[(k - 1) * Q + mi];
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
