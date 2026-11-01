/// <reference lib="webworker" />
/**
 * Web worker hosting the TypeScript Timestral port. Downloads `config.json` and
 * `model.safetensors` straight from the Hugging Face Hub (cached in Cache Storage so later
 * visits work offline), or takes user-provided files, and answers forecast requests.
 */

import { CACHE_NAME, fileUrl } from "./registry";
import { TimestralForecaster, TimestralModel } from "./timestral";

export type WorkerIn =
  | { type: "load"; repo: string }
  | { type: "loadBuffers"; label: string; config: Record<string, unknown>; weights: ArrayBuffer }
  | { type: "forecast"; id: number; context: Float64Array; horizon: number; symmetric: boolean; clip: boolean; maxContext: number }
  | { type: "listCached" }
  | { type: "evict"; repo: string };

export type WorkerOut =
  | { type: "progress"; file: string; loaded: number; total: number }
  | { type: "ready"; label: string; numParams: number; config: Record<string, unknown> }
  | { type: "result"; id: number; quantiles: Float32Array; levels: number[]; horizon: number; ms: number }
  | { type: "error"; id?: number; message: string }
  | { type: "cached"; repos: string[] };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let forecaster: TimestralForecaster | null = null;

const post = (m: WorkerOut, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

async function fetchWithProgress(url: string, file: string): Promise<ArrayBuffer> {
  const cache = await openCache();
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const buf = await hit.arrayBuffer();
    post({ type: "progress", file, loaded: buf.byteLength, total: buf.byteLength });
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status} ${res.status === 401 || res.status === 403 ? "(the repository may be gated or private)" : ""}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  let buf: Uint8Array<ArrayBuffer>;
  if (!res.body) {
    buf = new Uint8Array(await res.arrayBuffer());
  } else {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastPost = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      const now = performance.now();
      if (now - lastPost > 100) {
        post({ type: "progress", file, loaded, total });
        lastPost = now;
      }
    }
    buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    post({ type: "progress", file, loaded, total: total || loaded });
  }
  if (cache) {
    try {
      await cache.put(url, new Response(buf as Uint8Array<ArrayBuffer>, { headers: { "content-type": "application/octet-stream" } }));
    } catch {
      // Quota exceeded: keep working without the cache.
    }
  }
  return buf.buffer as ArrayBuffer;
}

async function listCached() {
  const cache = await openCache();
  const keys = cache ? await cache.keys() : [];
  const repos = keys
    .map((r) => r.url)
    .filter((u) => u.endsWith("/model.safetensors"))
    .map((u) => u.replace(/^https:\/\/huggingface\.co\//, "").replace(/\/resolve\/.*$/, ""));
  post({ type: "cached", repos });
}

function install(label: string, config: Record<string, unknown>, weights: ArrayBuffer) {
  const model = TimestralModel.fromBuffers(config, weights);
  forecaster = new TimestralForecaster(model);
  post({ type: "ready", label, numParams: model.numParams, config });
}

ctx.onmessage = async (ev: MessageEvent<WorkerIn>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case "load": {
        forecaster = null;
        const cfgBuf = await fetchWithProgress(fileUrl(msg.repo, "config.json"), "config.json");
        const config = JSON.parse(new TextDecoder().decode(cfgBuf));
        const weights = await fetchWithProgress(fileUrl(msg.repo, "model.safetensors"), "model.safetensors");
        install(msg.repo, config, weights);
        break;
      }
      case "loadBuffers":
        forecaster = null;
        install(msg.label, msg.config, msg.weights);
        break;
      case "forecast": {
        if (!forecaster) throw new Error("no model loaded");
        const t0 = performance.now();
        const fc = forecaster.forecast(msg.context, msg.horizon, {
          symmetricAveraging: msg.symmetric,
          clipNonnegative: msg.clip,
          maxContext: msg.maxContext,
        });
        post({ type: "result", id: msg.id, quantiles: fc.quantiles, levels: fc.levels, horizon: fc.horizon, ms: performance.now() - t0 }, [fc.quantiles.buffer]);
        break;
      }
      case "listCached":
        await listCached();
        break;
      case "evict": {
        const cache = await openCache();
        if (cache) {
          await cache.delete(fileUrl(msg.repo, "model.safetensors"));
          await cache.delete(fileUrl(msg.repo, "config.json"));
        }
        await listCached();
        break;
      }
    }
  } catch (e) {
    post({ type: "error", id: msg.type === "forecast" ? msg.id : undefined, message: e instanceof Error ? e.message : String(e) });
  }
};
