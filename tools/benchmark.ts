/**
 * Command-line runner for the leaderboard (see benchmarks/README.md). Started through
 * `tools/benchmark.mjs`, which loads this file with Vite so the app's modules and bundled
 * JSON datasets are imported exactly as the browser build imports them.
 */

import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { type BenchConfig, type Forecast, type ModelInfo, benchDatasets, defaultBenchConfig, runBenchmark } from "../src/bench/benchmark";
import { NaiveBackend, ServerBackend } from "../src/model/backends";
import { fileUrl } from "../src/model/registry";
import { TimestralForecaster, TimestralModel } from "../src/model/timestral";

const HELP = `Usage: npm run benchmark -- [options]

Model (default: the naive baseline)
  --model <repo>        Timestral checkpoint, e.g. mistralai/Timestral-5M. Runs the
                        TypeScript port in Node; weights are downloaded from the
                        Hugging Face Hub once and cached in ~/.cache/timestral-arena.
  --weights <dir>       Use config.json + model.safetensors from this folder instead
                        of downloading (label it with --model).
  --server <url>        Forecast through server/serve.py (the official PyTorch
                        package), e.g. http://127.0.0.1:8765. Needs --model.

Run
  --datasets a,b        Only these dataset ids (default: all). Merged into an
                        existing result file for the same model.
  --out <dir>           Output folder (default: benchmarks/results).
  --run <name>          File stem (default: derived from the model).
  --list                Print the dataset ids and exit.

Settings (defaults match the app)
  --alpha 0.1 --leads 16 --score cqr|cqr_asym|abs|scaled --window 100 --warmup 10
  --horizon <=leads> --context 256 --length 2000 --seed 1 --bootstrap 300
  --no-symmetric --no-clip
`;

function parseArgs(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) throw new Error(`unexpected argument ${a}`);
    const key = a.slice(2);
    if (key.startsWith("no-")) out[key.slice(3)] = false;
    else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) out[key] = argv[++i];
    else out[key] = true;
  }
  return out;
}

const slug = (s: string) => s.replace(/^.*\//, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

async function download(repo: string): Promise<string> {
  const dir = join(homedir(), ".cache", "timestral-arena", repo);
  mkdirSync(dir, { recursive: true });
  for (const file of ["config.json", "model.safetensors"]) {
    const path = join(dir, file);
    if (existsSync(path)) continue;
    const url = fileUrl(repo, file);
    console.error(`downloading ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  }
  return dir;
}

function loadLocal(dir: string): TimestralForecaster {
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8"));
  const buf = readFileSync(join(dir, "model.safetensors"));
  const weights = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  return new TimestralForecaster(TimestralModel.fromBuffers(config, weights));
}

function gitCommit(): string | undefined {
  try {
    const sha = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
    const dirty = execSync("git status --porcelain -- src", { encoding: "utf8" }).trim();
    return dirty ? `${sha}+dirty` : sha;
  } catch {
    return undefined;
  }
}

export async function main(argv: string[]): Promise<void> {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(HELP);
    return;
  }
  if (args.list) {
    for (const d of benchDatasets()) console.log(`${d.id.padEnd(12)} ${d.group.padEnd(10)} ${d.label}`);
    return;
  }
  const d = defaultBenchConfig();
  const num = (k: string, v: number) => (typeof args[k] === "string" ? Number(args[k]) : v);
  const config: BenchConfig = {
    ...d,
    alpha: num("alpha", d.alpha),
    evalLeads: num("leads", d.evalLeads),
    scoreType: (typeof args.score === "string" ? args.score : d.scoreType) as BenchConfig["scoreType"],
    window: num("window", d.window),
    warmup: num("warmup", d.warmup),
    horizon: num("horizon", num("leads", d.evalLeads)),
    maxContext: num("context", d.maxContext),
    symmetric: args.symmetric !== false,
    clip: args.clip !== false,
    length: num("length", d.length),
    seed: num("seed", d.seed),
    bootstrap: num("bootstrap", d.bootstrap),
  };

  const repo = typeof args.model === "string" ? args.model : undefined;
  let forecast: Forecast;
  let model: ModelInfo;
  const req = (context: Float64Array, horizon: number) => ({ context, horizon, symmetric: config.symmetric, clip: config.clip, maxContext: config.maxContext });
  if (typeof args.server === "string") {
    if (!repo) throw new Error("--server needs --model <repo>");
    const b = new ServerBackend(args.server, repo);
    await b.connect();
    forecast = (c, h) => b.forecast(req(c, h));
    model = { backend: "server", id: repo, label: b.label };
  } else if (repo || typeof args.weights === "string") {
    const dir = typeof args.weights === "string" ? resolve(args.weights) : await download(repo!);
    const fc = loadLocal(dir);
    forecast = async (c, h) => fc.forecast(c, h, { symmetricAveraging: config.symmetric, clipNonnegative: config.clip, maxContext: config.maxContext });
    const id = repo ?? dir;
    model = { backend: "browser", id, label: `${id.replace(/^.*\//, "")} (TypeScript port, ${fc.model.numParams.toLocaleString("en")} params)` };
  } else {
    const b = new NaiveBackend();
    forecast = (c, h) => b.forecast(req(c, h));
    model = { backend: "naive", id: "naive", label: b.label };
  }

  const run = typeof args.run === "string" ? args.run : slug(model.id);
  const outDir = resolve(typeof args.out === "string" ? args.out : "benchmarks/results");
  const only = typeof args.datasets === "string" ? args.datasets.split(",").map((s) => s.trim()).filter(Boolean) : undefined;
  const known = new Set(benchDatasets().map((x) => x.id));
  for (const id of only ?? []) if (!known.has(id)) throw new Error(`unknown dataset "${id}" (see --list)`);

  console.error(`model: ${model.label}; run: ${run}`);
  const result = await runBenchmark({ run, model, forecast, config, only, commit: gitCommit(), log: (m) => console.error(m) });

  // A partial run (--datasets) updates those datasets in an existing file for the same
  // model and settings, keeping the rest; anything else replaces the file.
  const path = join(outDir, `${run}.json`);
  if (only && existsSync(path)) {
    const prev = JSON.parse(readFileSync(path, "utf8"));
    // Kept datasets must have been scored with the same settings, model and methods
    // (ids and default parameters); otherwise every dataset is from this run only.
    const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
    if (prev.schema === result.schema && same(prev.config, result.config) && prev.model?.id === result.model.id && same(prev.methods, result.methods)) {
      const fresh = new Map(result.datasets.map((x) => [x.id, x]));
      const order = benchDatasets().map((x) => x.id);
      const merged = [...prev.datasets.filter((x: { id: string }) => !fresh.has(x.id) && known.has(x.id)), ...result.datasets];
      merged.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
      result.datasets = merged;
    } else {
      console.error("existing result has other settings, another model or other method definitions; replacing it with only the datasets run now");
    }
  }
  mkdirSync(dirname(path), { recursive: true });
  // One line per row object field, with number arrays kept on one line: diff-friendly, compact.
  const json = JSON.stringify(result, null, 1).replace(/\[\s*([^[\]{}]*?)\s*\]/g, (_, inner: string) => `[${inner.replace(/\s*\n\s*/g, " ")}]`);
  writeFileSync(path, json + "\n");
  console.error(`wrote ${path}`);
}
