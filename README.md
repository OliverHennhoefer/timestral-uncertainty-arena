# cTimestral

Draw a time series with the mouse, let [Timestral](https://huggingface.co/collections/mistralai/timestral) (Mistral AI's time-series foundation model) re-forecast it on every new step, and watch online conformal prediction methods calibrate its intervals live. Coverage, width and interval score are tracked for every method and for Timestral's own quantile interval, and a ranking updates as you draw.

Everything runs locally. The model runs **inside the browser** through a TypeScript port of the official inference code, so the app also works as a static GitHub Pages site. The weights are downloaded once from the Hugging Face Hub and cached in the browser.

## What Timestral is (checked October 2026)

| | |
|---|---|
| Checkpoints | `mistralai/Timestral-5M`, `-28M`, `-123M`, `-209M` (same architecture, same training recipe) |
| Weights | `config.json` + `model.safetensors` (float32) on the Hugging Face Hub, Modified MIT licence |
| Output | 21 quantiles: 0.01, 0.05, 0.10, …, 0.95, 0.99 per horizon step (no point forecast; the median is the 0.5 quantile) |
| Context / horizon | up to 32,768 steps of context, 1,024 steps per decode block (longer horizons are decoded block by block) |
| Official runtime | PyTorch only ([mistralai/timestral](https://github.com/mistralai/timestral)) |
| Browser | no official ONNX/transformers.js build. A community ONNX export exists for 209M only (`mrfakename/Timestral-209M-ONNX`, 841 MB fp32). |

So "native intervals" exist: the central (α/2, 1 − α/2) interval comes straight from the quantile head (linearly interpolated between knots when α/2 is not one of the 21 levels).

### Running it in the browser

Rather than depending on a single third-party ONNX export, `src/model/` is a dependency-free TypeScript port of the official `timestral` package (model, causal per-patch asinh scaling, DHF two-stream decoding, advancing refine frame, block-autoregressive decoding, quantile cap, symmetric averaging and non-negative clipping). It reads the released `config.json` and `model.safetensors` unchanged, so all four sizes work. It covers the univariate, no-covariate path.

Rough CPU cost of one forecast in a web worker (single thread, 256-step context, symmetric averaging on): about 0.1 to 0.3 s for 5M, which is fine for per-stroke forecasts. 28M is usable. 123M and 209M are better served by the local server below. Turning off symmetric averaging halves the cost.

**Parity.** `tools/make_parity_fixture.py` builds a random checkpoint with the official package and records its forecasts; `tests/parity.test.ts` checks the port against them (relative error < 2e-4). The `parity` CI job runs this on every PR.

## Run it

```bash
npm install
npm run dev            # http://localhost:5173
```

In the app, pick a backend:

- **Naive baseline**: no download; last value plus empirical k-step changes. Handy to try the conformal side right away.
- **Timestral in this browser**: download 5M / 28M / 123M / 209M from the Hub (progress bar, cached for offline use), or load `config.json` + `model.safetensors` from disk.
- **Timestral on a local server**: the official PyTorch package, any size, CPU or GPU:

  ```bash
  pip install -r server/requirements.txt
  python server/serve.py --port 8765 --preload mistralai/Timestral-209M
  ```

  The server allows CORS and Chrome's private-network access, so the GitHub Pages build can also use it.

### GitHub Pages

`.github/workflows/pages.yml` builds and deploys `dist/` on every push to `main`. Enable it once under *Settings → Pages → Source: GitHub Actions*. The site ships no weights; visitors' browsers fetch them from Hugging Face.

## How the benchmark works

- **Drawing.** The canvas is a grid of steps (2 px each). Each new step is appended to the series and triggers a forecast from that origin. Drawing back to the left rewinds the series; forecasts made from the untouched prefix are kept and every method is replayed, so the state always equals a clean run on the current series.
- **Streams.** Synthetic presets (seasonal, trend, volatility shift, level shifts, random walk, heavy tails, regime change) play point by point in lockstep with the model, so every origin gets a forecast.
- **Online protocol.** A forecast at origin `o` predicts leads `k = 1..H`. When `y[o+k]` arrives it yields a nonconformity score at lead `k`, which updates that lead's methods (feedback is delayed by `k`). Thresholds are issued only from scores whose truth was known at the origin. Each lead has its own calibration stream.
- **Scores.** CQR on the native quantiles, `max(lo − y, y − hi)` (default; the interval is `[lo − q, hi + q]`), absolute residual `|y − median|`, or the residual scaled by the native half-width.
- **Metrics.** For the calibrated leads (default 1–16), after a warm-up, every method and the native interval are scored on the same pairs: cumulative and rolling coverage, mean width, and the interval (Winkler) score `width + (2/α)·miss distance`.
- **Ranking.** By interval score (default), by "valid coverage first, then narrowest", or by rolling coverage closest to target.

### Methods

| Method | Reference | Notes |
|---|---|---|
| Rolling split CP | Vovk et al. 2005 | (1−α)(1+1/n) quantile of the last W scores |
| ACI | Gibbs & Candès, NeurIPS 2021 | α_t ← α_t + γ(α − err_t) |
| DtACI | Gibbs & Candès, JMLR 2024 | ACI experts over a γ grid, exponential reweighting on the pinball loss; weighted-average variant |
| AgACI | Zaffran et al., ICML 2022 | ACI experts aggregated online; uses EWA on the threshold's pinball loss |
| Quantile tracker | Angelopoulos, Candès & Tibshirani, NeurIPS 2023 | q ← q + η(err − α), η scaled by the recent score range |
| Conformal PID | Angelopoulos, Candès & Tibshirani, NeurIPS 2023 | quantile tracker plus a saturated error integrator (no scorecaster) |
| NexCP | Barber, Candès, Ramdas & Tibshirani, Ann. Stat. 2023 | weighted quantile with weights ρ^age |

Each method can be switched on or off and its parameters changed live; everything is replayed from the stored forecasts so the comparison stays fair. Thresholds are clipped to ± the largest |score| seen so far at that lead, which also turns infinite thresholds (e.g. ACI with α_t ≤ 0, or a saturated PID integrator) into finite, comparable widths.

New methods implement `OnlineMethod` (`predict(key)` / `update(key, score)`) in `src/conformal/methods.ts` and are added to `METHODS`.

## Layout

```
src/model/       Timestral TypeScript port, safetensors reader, worker, backends, model registry
src/conformal/   score windows and online conformal methods
src/engine/      online protocol, replay, metrics, ranking
src/ui/          canvas helpers
server/          local PyTorch server using the official package
tools/           parity fixture generator
tests/           vitest: port invariants, parity, methods, engine
```

```bash
npm test          # unit tests (parity runs when tests/fixtures/parity exists)
npm run build     # static site in dist/
```

## Licences

Code: MIT (this repository). Timestral weights: Mistral AI, Modified MIT licence. The official inference code that the port follows: Apache-2.0.
