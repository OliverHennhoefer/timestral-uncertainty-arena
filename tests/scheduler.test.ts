import { describe, expect, it } from "vitest";
import { Engine, NATIVE_ID, defaultSettings } from "../src/engine/engine";
import { ForecastScheduler } from "../src/engine/scheduler";
import { type Backend, type ForecastRequest, NaiveBackend } from "../src/model/backends";

const OPTIONS = { horizon: 16, symmetric: true, clip: false, maxContext: 256 };

/** The naive model behind a manually released queue, so a test decides when forecasts land. */
class SlowBackend implements Backend {
  readonly kind = "naive" as const;
  readonly label = "slow";
  readonly modelName = "Slow";
  readonly ready = true;
  readonly waiting: (() => void)[] = [];
  fail = false;
  private readonly naive = new NaiveBackend();
  forecast(req: ForecastRequest) {
    return new Promise<void>((resolve) => this.waiting.push(resolve)).then(() => {
      if (this.fail) throw new Error("boom");
      return this.naive.forecast(req);
    });
  }
  dispose() {}
}

const tick = () => new Promise((r) => setTimeout(r, 0));

/** Let the backend answer every request until the scheduler goes idle. */
async function drain(b: SlowBackend, s: ForecastScheduler) {
  for (let i = 0; i < 10_000; i++) {
    await tick();
    if (b.waiting.length) b.waiting.shift()!();
    else if (!s.inFlight) return;
  }
  throw new Error("scheduler did not settle");
}

function setup() {
  const settings = { ...defaultSettings(), warmup: 3, evalLeads: 8 };
  const engine = new Engine(settings);
  const backend = new SlowBackend();
  let paused = false;
  const errors: unknown[] = [];
  const scheduler = new ForecastScheduler({
    engine,
    backend: () => backend,
    options: () => OPTIONS,
    paused: () => paused,
    changed: () => {},
    failed: (e) => errors.push(e),
  });
  const draw = (v: number) => {
    engine.append(v);
    scheduler.arrived(engine.series.length - 1);
    void scheduler.pump();
  };
  return { engine, backend, scheduler, draw, errors, pause: (p: boolean) => (paused = p) };
}

const wave = (t: number) => Math.round((50 + 30 * Math.sin(t / 7) + ((t * 37) % 11)) * 100) / 100;

/** Reference: every origin forecast in lockstep, as it arrived. */
async function lockstep(values: number[]) {
  const e = new Engine({ ...defaultSettings(), warmup: 3, evalLeads: 8 });
  const b = new NaiveBackend();
  for (const v of values) {
    e.append(v);
    e.catchUp();
    const t = e.series.length - 1;
    const r = await b.forecast({ context: Float64Array.from(e.series), ...OPTIONS });
    e.addForecast(t, r.quantiles, r.horizon, r.levels);
  }
  return e;
}

describe("forecast scheduler", () => {
  it("fills the origins skipped while drawing faster than the model", async () => {
    const { engine, backend, scheduler, draw } = setup();
    const values = Array.from({ length: 60 }, (_, t) => wave(t));
    // Draw a whole stroke while the model answers only every tenth point.
    values.forEach((v, t) => {
      draw(v);
      if (t % 10 === 9 && backend.waiting.length) backend.waiting.shift()!();
    });
    await tick();
    const before = engine.forecasts.size;
    expect(before).toBeLessThan(values.length);

    await drain(backend, scheduler);
    expect(engine.forecasts.size).toBe(values.length);
    expect(scheduler.skipped()).toEqual([]);

    // The past band has no holes and the result matches forecasting every point as it came.
    const ref = await lockstep(values);
    const k = 1;
    expect(engine.pastBand("aci", k, 0, values.length - 1)).toEqual(ref.pastBand("aci", k, 0, values.length - 1));
    // Every step has a band except the first, whose forecast came before any score.
    expect(engine.pastBand("aci", k, 0, values.length - 1).length).toBe(values.length - k - 1);
    for (const id of [NATIVE_ID, "aci", "split"]) expect(engine.metrics.get(id)!.n).toBe(ref.metrics.get(id)!.n);
  });

  it("puts the newest point ahead of the backlog", async () => {
    const { engine, backend, draw } = setup();
    for (let t = 0; t < 30; t++) draw(wave(t)); // origin 0 in flight, 1..29 skipped
    backend.waiting.shift()!();
    await tick();
    await tick();
    // Newest first: origin 29 is requested before any skipped origin.
    backend.waiting.shift()!();
    await tick();
    await tick();
    expect(engine.latestOrigin()).toBe(29);
    // Now filling, newest skipped origin first; new input preempts the backlog again.
    draw(wave(30));
    backend.waiting.shift()!(); // the fill request for origin 28 completes
    await tick();
    await tick();
    expect(engine.forecasts.has(28)).toBe(true);
    backend.waiting.shift()!();
    await tick();
    await tick();
    expect(engine.latestOrigin()).toBe(30);
  });

  it("does not re-forecast history after forget (e.g. a new model)", async () => {
    const { engine, backend, scheduler, draw } = setup();
    for (let t = 0; t < 20; t++) draw(wave(t));
    scheduler.forget();
    engine.forecasts.clear();
    engine.revision++;
    engine.replay();
    await drain(backend, scheduler);
    expect([...engine.forecasts.keys()]).toEqual([19]);
  });

  it("drops a batch when the series is rewritten mid-fill and stops after a failure", async () => {
    const { engine, backend, scheduler, draw } = setup();
    for (let t = 0; t < 20; t++) draw(wave(t));
    backend.waiting.shift()!();
    await tick();
    backend.waiting.shift()!(); // newest (19)
    await tick();
    await tick();
    expect(engine.latestOrigin()).toBe(19);
    // A fill request is out; rewrite history before it lands.
    engine.truncate(10);
    backend.waiting.shift()!();
    await tick();
    await tick();
    expect([...engine.forecasts.keys()].every((o) => o < 10)).toBe(true);

    backend.fail = true;
    await drain(backend, scheduler);
    expect(scheduler.inFlight).toBe(false);
    expect(backend.waiting.length).toBe(0);
  });
});
