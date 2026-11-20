"""Local forecast server for Timestral Uncertainty Arena, backed by the official model packages.

Serves Timestral (``timestral``), Chronos-Bolt and Chronos-2 (``chronos-forecasting``), TimesFM 2.5
and 3.0 (``timesfm``) and Moirai 2.0 (``uni2ts``). Use it for the larger checkpoints, for the
models that have no in-browser port (Chronos-2, TimesFM, Moirai), or for a GPU.

    pip install -r server/requirements.txt
    python server/serve.py --port 8765            # then pick "Local server" in the app

Moirai's package pins old versions of torch, numpy and scipy, so install it without them:
``pip install --no-deps uni2ts && pip install einops jaxtyping gluonts lightning hydra-core scipy``.
Each family's package is imported only when one of its models is first used.

Endpoints (JSON, CORS-enabled, bound to 127.0.0.1 by default):
    GET  /health    -> {"models": [...loaded repo ids...], "device": "cpu"}
    POST /forecast  {"model": "mistralai/Timestral-28M", "context": [...], "horizon": 32,
                     "symmetric_averaging": true, "clip_nonnegative": true}
                 -> {"quantiles": [[Q floats] * horizon], "levels": [...], "ms": 12.3}
Every model answers on its own quantile levels, as its official package returns them:
Timestral and Chronos-2 on 21 levels (0.01, 0.05, ..., 0.95, 0.99), Chronos-Bolt, TimesFM and
Moirai on 9 (0.1, ..., 0.9). The browser maps 9-level answers onto the 21-level grid and applies
``clip_nonnegative``, exactly as it does for the in-browser Chronos-Bolt port; the server applies it
for the 21-level models. Symmetric averaging is Timestral-only. Models are downloaded from the
Hugging Face Hub on first use and kept in memory.
"""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

TIMESTRAL = {
    "mistralai/Timestral-5M",
    "mistralai/Timestral-28M",
    "mistralai/Timestral-123M",
    "mistralai/Timestral-209M",
}
CHRONOS_BOLT = {
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
}
CHRONOS_2 = {"amazon/chronos-2"}
TIMESFM = {"google/timesfm-2.5-200m-pytorch", "google/timesfm-3.0-pytorch"}
MOIRAI = {"Salesforce/moirai-2.0-R-small"}
ALLOWED = TIMESTRAL | CHRONOS_BOLT | CHRONOS_2 | TIMESFM | MOIRAI

_models: dict[str, object] = {}
_lock = threading.Lock()
# TimesFM 2.5 (recompiled on demand) and Moirai (horizon set per call) change shared state while
# forecasting, so their forecasts run one at a time.
_stateful = threading.Lock()
DEVICE = "cpu"

# TimesFM 2.5 is compiled for a maximum context and horizon; it is recompiled when a request needs
# more. Context plus horizon must fit in 16k; the horizon is rounded up to 128-step patches.
TIMESFM_MAX_CONTEXT = 16384


def is_chronos(repo: str) -> bool:
    if repo in CHRONOS_BOLT or repo in CHRONOS_2:
        return True
    # Local checkpoints: Chronos configs carry a "chronos_config" block.
    cfg = Path(repo) / "config.json"
    return repo.startswith("./") and cfg.is_file() and "chronos_config" in json.loads(cfg.read_text())


def _need(package: str, family: str):
    return ValueError(f"{family} needs `pip install {package}` on the server (see server/serve.py)")


def _load(repo: str):
    if is_chronos(repo):
        try:
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise _need("chronos-forecasting", "Chronos") from e
        return BaseChronosPipeline.from_pretrained(repo, device_map=DEVICE, torch_dtype=torch.float32)
    if repo in TIMESFM:
        try:
            import timesfm
        except ImportError as e:
            raise _need("timesfm", "TimesFM") from e
        if repo == "google/timesfm-3.0-pytorch":
            return timesfm.TimesFM3Forecaster.from_pretrained(repo, device=DEVICE)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
        model._compiled_for = (0, 0)  # (max_context, max_horizon), set on first use
        return model
    if repo in MOIRAI:
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as e:
            raise _need("uni2ts", "Moirai") from e
        module = Moirai2Module.from_pretrained(repo)
        model = Moirai2Forecast(
            module=module,
            prediction_length=1,
            context_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.to(DEVICE).eval()
    return TimestralForecaster.from_pretrained(repo, device=DEVICE)


def get_model(repo: str):
    if repo not in ALLOWED and not repo.startswith("./"):
        raise ValueError(f"unknown model {repo!r}")
    with _lock:
        if repo not in _models:
            _models[repo] = _load(repo)
        return _models[repo]


def _interpolate(context: np.ndarray) -> np.ndarray:
    """Fill missing values linearly (ends held flat), for models that take no NaNs."""
    ok = np.isfinite(context)
    if ok.all() or not ok.any():
        return np.nan_to_num(context)
    idx = np.arange(len(context))
    return np.interp(idx, idx[ok], context[ok]).astype(np.float32)


def _clip_nonnegative(q: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Timestral's rule: clip at 0 when every observed context value is non-negative."""
    obs = context[np.isfinite(context)]
    return np.maximum(q, 0) if obs.size and (obs >= 0).all() else q


def _timesfm25(model, context: np.ndarray, horizon: int) -> tuple[list, list]:
    import timesfm

    max_horizon = max(256, -(-horizon // 128) * 128)
    # Context and horizon share the 16k limit; a longer context keeps its latest points.
    max_context = min(TIMESFM_MAX_CONTEXT - max_horizon, max(1024, -(-len(context) // 1024) * 1024))
    context = context[-max_context:]
    need = (max_context, max_horizon)
    if model._compiled_for[0] < need[0] or model._compiled_for[1] < need[1]:
        # The configuration recommended in the TimesFM 2.5 model card, except that clipping at
        # zero is left to clip_nonnegative so that the app's checkbox decides it.
        model.compile(
            timesfm.ForecastConfig(
                max_context=need[0],
                max_horizon=need[1],
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        model._compiled_for = need
    _, q = model.forecast(horizon=horizon, inputs=[context])
    # (1, H, 10): the mean, then the 10%, ..., 90% quantiles.
    return q[0, :, 1:].tolist(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def forecast(repo: str, context: np.ndarray, horizon: int, symmetric: bool, clip: bool) -> tuple[list, list]:
    model = get_model(repo)
    if repo in CHRONOS_2:
        # (variates, Q, H) per series on the model's 21 training quantiles.
        q = model.predict([torch.tensor(context)], prediction_length=horizon)[0][0].T.cpu().numpy()
        q = np.sort(q, axis=1)
        return (_clip_nonnegative(q, context) if clip else q).tolist(), list(model.quantiles)
    if is_chronos(repo):
        # (1, Q, H) on the model's training quantiles, as the official pipeline returns them.
        q = model.predict(torch.tensor(context), prediction_length=horizon)
        return q[0].T.cpu().numpy().tolist(), list(model.quantiles)
    if repo == "google/timesfm-3.0-pytorch":
        out = model.predict(_interpolate(context), horizon=horizon, return_quantiles=True)
        return np.asarray(out.quantiles).tolist(), list(model.config.quantiles)
    if repo in TIMESFM:
        with _stateful:
            return _timesfm25(model, _interpolate(context), horizon)
    if repo in MOIRAI:
        # Missing values are imputed by the package, as in pretraining.
        with _stateful, model.hparams_context(prediction_length=horizon, context_length=len(context)):
            q = model.predict([context[:, None]])  # (1, Q, H)
        return q[0].T.tolist(), list(model.module.quantile_levels)
    fc = model.forecast(context, prediction_length=horizon, symmetric_averaging=symmetric, clip_nonnegative=clip)
    return fc.quantiles[0, 0].cpu().numpy().tolist(), list(fc.quantile_levels)


class Handler(BaseHTTPRequestHandler):
    def _headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        # Lets an https page (e.g. GitHub Pages) call this localhost server in Chromium.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def _json(self, body: dict, status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(body).encode())

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(204)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._json({"models": sorted(_models), "device": DEVICE, "available": sorted(ALLOWED)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/forecast":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            repo = req.get("model", "mistralai/Timestral-5M")
            context = np.array([math.nan if v is None else float(v) for v in req["context"]], dtype=np.float32)
            horizon = int(req["horizon"])
            if not 1 <= horizon <= 4096:
                raise ValueError("horizon must be in [1, 4096]")
            get_model(repo)  # download outside the timed section
            t0 = time.perf_counter()
            quantiles, levels = forecast(
                repo,
                context,
                horizon,
                symmetric=bool(req.get("symmetric_averaging", True)),
                clip=bool(req.get("clip_nonnegative", True)),
            )
            ms = (time.perf_counter() - t0) * 1e3
            self._json({"quantiles": quantiles, "levels": levels, "ms": ms})
        except Exception as e:  # report errors to the browser
            self._json({"error": str(e)}, 400)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    global DEVICE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preload", nargs="*", default=[], help="repo ids to load at start")
    args = p.parse_args()
    DEVICE = args.device
    for repo in args.preload:
        get_model(repo)
    print(f"Timestral Uncertainty Arena server on http://{args.host}:{args.port} (device={DEVICE})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
