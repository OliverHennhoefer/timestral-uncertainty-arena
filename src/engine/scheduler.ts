/**
 * Forecast scheduling for live input (drawing, streaming).
 *
 * The newest origin always comes first, so the latest forecast keeps up with the pen. When
 * the model is slower than the input, the origins drawn in between are skipped at first.
 * They would otherwise stay without a forecast for good, which leaves holes in the past band
 * (lead k at time t needs the forecast made at t - k) and starves the conformal methods of
 * scores, so bands stay patchy or never appear. Once the newest origin is covered, the
 * scheduler forecasts those skipped origins, newest first, in short batches, and commits
 * each batch with a replay, which leaves the engine exactly as if every origin had been
 * forecast as it arrived (the engine is deterministic in series, forecasts and settings).
 */

import type { Backend, ForecastRequest, ForecastResult } from "../model/backends";
import type { Engine } from "./engine";

export interface SchedulerHost {
  engine: Engine;
  /** The current backend; a change discards work for the old one. */
  backend(): Backend;
  /** Forecast settings for the next request. */
  options(): Omit<ForecastRequest, "context">;
  /** True while something else owns the stored forecasts (a full recompute). */
  paused(): boolean;
  /** Something visible changed: the busy state or the stored forecasts. */
  changed(): void;
  /** Forecasting the newest origin failed. */
  failed(error: unknown): void;
}

/** Least wall time spent filling skipped origins before a batch is committed and shown. */
const BATCH_MS = 250;
/** A batch also runs for this many times the last commit's replay, so long series stay cheap. */
const REPLAY_SHARE = 4;

export class ForecastScheduler {
  inFlight = false;
  lastLatency = 0;
  /** Origin whose forecast last failed; not retried until the series or backend changes. */
  private failedOrigin: { origin: number; revision: number; backend: Backend } | null = null;
  /** Filling skipped origins failed; not retried until the series or backend changes. */
  private fillFailed: { revision: number; backend: Backend } | null = null;
  /** Duration of the last batch commit (a full replay), in ms. */
  private replayMs = 0;
  /** Origins that arrived as live input, so a forecast for each is owed. */
  private readonly live = new Set<number>();

  constructor(private readonly host: SchedulerHost) {}

  /** Record an origin that arrived as live input (call after appending it). */
  arrived(origin: number): void {
    this.live.add(origin);
  }

  /**
   * Forget owed origins: after clearing the series, or after dropping all forecasts for a
   * new model, where re-forecasting the history is left to an explicit recompute.
   */
  forget(): void {
    this.live.clear();
  }

  /** Live origins the model skipped to keep up, newest first. */
  skipped(): number[] {
    const { engine } = this.host;
    const latest = engine.latestOrigin();
    const out: number[] = [];
    for (const o of this.live) {
      if (o >= engine.series.length || engine.forecasts.has(o)) this.live.delete(o);
      else if (o < latest) out.push(o);
    }
    return out.sort((a, b) => b - a);
  }

  /** Run until there is nothing left to forecast. Safe to call at any time. */
  async pump(): Promise<void> {
    const { engine } = this.host;
    const b = this.host.backend();
    if (this.inFlight || this.host.paused() || !b.ready) return;
    const t = engine.series.length - 1;
    if (t < 0) return;
    const rev = engine.revision;
    let job: () => Promise<void>;
    if (t > engine.latestOrigin()) {
      const f = this.failedOrigin;
      if (f && f.origin === t && f.revision === rev && f.backend === b) return;
      job = () => this.forecastNewest(t, b);
    } else {
      const f = this.fillFailed;
      if (f && f.revision === rev && f.backend === b) return;
      if (!this.skipped().length) return;
      job = () => this.fillSkipped(b);
    }
    this.inFlight = true;
    this.host.changed();
    try {
      await job();
    } finally {
      this.inFlight = false;
      this.host.changed();
      queueMicrotask(() => void this.pump());
    }
  }

  private request(o: number): ForecastRequest {
    return { context: Float64Array.from(this.host.engine.series.slice(0, o + 1)), ...this.host.options() };
  }

  private async forecastNewest(t: number, b: Backend): Promise<void> {
    const { engine } = this.host;
    const rev = engine.revision;
    try {
      engine.catchUp(t);
      const res = await b.forecast(this.request(t));
      this.lastLatency = res.ms;
      if (b === this.host.backend() && engine.revision === rev && engine.series.length > t) {
        engine.addForecast(t, res.quantiles, res.horizon, res.levels);
      }
      this.failedOrigin = null;
    } catch (e) {
      this.failedOrigin = { origin: t, revision: rev, backend: b };
      this.host.failed(e);
    }
  }

  private async fillSkipped(b: Backend): Promise<void> {
    const { engine } = this.host;
    const rev = engine.revision;
    const stale = () => engine.revision !== rev || this.host.backend() !== b || this.host.paused();
    const done: [number, ForecastResult][] = [];
    const t0 = performance.now();
    const budget = Math.max(BATCH_MS, REPLAY_SHARE * this.replayMs);
    try {
      for (const o of this.skipped()) {
        // New input goes first; what is done so far is still committed below.
        if (engine.series.length - 1 > engine.latestOrigin()) break;
        if (done.length && performance.now() - t0 > budget) break;
        const res = await b.forecast(this.request(o));
        if (stale()) return;
        this.lastLatency = res.ms;
        done.push([o, res]);
      }
    } catch (e) {
      // Keep the batch so far; stop filling until the series or backend changes.
      this.fillFailed = { revision: rev, backend: b };
      console.warn("Forecasting skipped origins failed:", e);
    }
    if (!done.length || stale()) return;
    for (const [o, res] of done) engine.storeForecast(o, res.quantiles, res.horizon, res.levels);
    const r0 = performance.now();
    engine.replay();
    this.replayMs = performance.now() - r0;
  }
}
