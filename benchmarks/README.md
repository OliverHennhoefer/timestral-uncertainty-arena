# Leaderboard benchmark

The [leaderboard page](../leaderboard.html) (`leaderboard.html` in the built site) shows these results. They are produced offline and committed here, so the page stays static and works on GitHub Pages.

## What is run

Every dataset in the app's stream menu (the synthetic presets in `src/synthetic.ts` and the real series in `src/datasets.ts`) goes through every method in `src/conformal/methods.ts`, plus the model's own ("native") interval. Each dataset is forecast at every origin, with each forecast seeing only the past, then scored by the same engine the app uses (`src/engine/engine.ts`), with the app's default settings: α = 0.1, CQR score, leads 1–16, window 100, warm-up 10, context 256, default method parameters. Synthetic presets use their first 2000 points with seed 1; real series use all their points (up to 2000).

Per dataset and method the result stores coverage, coverage gap, mean width, mean interval (Winkler) score, block-bootstrap 95% intervals, the rank (ties share a rank when a paired block bootstrap cannot separate them), the score ratio to native with a significance flag, and coverage per lead. The page aggregates these across datasets and writes the Findings from them.

## Run it

```bash
npm install
npm run benchmark                                   # naive baseline, about a minute -> results/naive.json
npm run benchmark -- --model mistralai/Timestral-5M # TypeScript port in Node -> results/timestral-5m.json
npm run benchmark -- --help                         # all options
```

`--model` downloads `config.json` and `model.safetensors` from the Hugging Face Hub once (cached in `~/.cache/timestral-arena`); `--weights <dir>` uses a local copy instead. For the larger checkpoints, the official PyTorch package is faster:

```bash
pip install -r server/requirements.txt
python server/serve.py --port 8765
npm run benchmark -- --server http://127.0.0.1:8765 --model mistralai/Timestral-209M
```

`--datasets co2,nile` re-runs only those datasets and merges them into the existing file for the same model and settings. Each model writes its own file, named after the checkpoint (`--run` overrides it); the page offers one entry per file in its model menu.

## Sweeps and reruns

These options are for experiments from the command line. They do not change the app, the leaderboard page or the files in `results/`.

```bash
# every combination of the lists -> benchmarks/sweeps/tails/
npm run benchmark -- --alpha 0.2,0.1,0.05,0.02 --window 100,250,500,1000 --sweep tails
# other method parameters (ids and keys as in src/conformal/methods.ts)
npm run benchmark -- --param aci.gamma=0.005,bocpd.lambda=100 --run naive-tuned --out /tmp/out
# aggregated table, findings and caveats as text
npm run benchmark -- --summary
npm run benchmark -- --report benchmarks/results/naive.json
```

A sweep writes one result file per combination (same format as `results/`), `summary.csv` with every dataset and method of every combination, and `summary.txt`: one line per combination (where the native interval ranks, which method does best) followed by the table of each run. `--score` takes a list too.

Forecasts are cached in `benchmarks/cache/`, one file per model, dataset and forecast settings (horizon, context, symmetric averaging, clipping). They do not depend on α, the score, the window, the warm-up, the leads or any method, so changing those reuses the stored forecasts and only the scoring runs again; with a Timestral checkpoint that turns a rerun from minutes or hours into seconds per dataset. Before a cached set is used, one forecast is computed again and compared with it, so a changed checkpoint or port is noticed and the dataset is forecast afresh. `--no-cache` switches the cache off, `--cache <dir>` moves it. `benchmarks/cache/` and `benchmarks/sweeps/` are not committed.

## Adding a dataset or method

Add it to its registry (`src/datasets.ts`, `src/synthetic.ts` or `METHODS`), then run `npm run benchmark` and rebuild. Nothing in the page or the script needs editing. `tests/bench.test.ts` fails while `results/naive.json` is missing a registered dataset or method, so the naive run cannot silently fall behind; model runs may cover a subset.

## File format

`results/<run>.json`, described by `src/bench/types.ts`. `schema` is bumped when the format changes; the page ignores files with another schema. `commit` records the code version that produced the file (`+dirty` when `src/` had uncommitted changes) and `model` the backend and checkpoint.
