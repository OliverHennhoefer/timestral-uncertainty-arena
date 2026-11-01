"""Local forecast server for Timestral Uncertainty Arena, backed by the official model packages.

Serves Timestral (``timestral``) and Chronos-Bolt (``chronos-forecasting``). Use it for the larger
checkpoints (Timestral 123M/209M, Chronos-Bolt Base) or a GPU, where the in-browser ports are slow.

    pip install -r server/requirements.txt
    python server/serve.py --port 8765            # then pick "Local server" in the app

Endpoints (JSON, CORS-enabled, bound to 127.0.0.1 by default):
    GET  /health    -> {"models": [...loaded repo ids...], "device": "cpu"}
    POST /forecast  {"model": "mistralai/Timestral-28M", "context": [...], "horizon": 32,
                     "symmetric_averaging": true, "clip_nonnegative": true}
                 -> {"quantiles": [[Q floats] * horizon], "levels": [...], "ms": 12.3}
Timestral answers on its 21 levels. Chronos-Bolt answers on its own 9 levels (0.1, ..., 0.9) as
``ChronosBoltPipeline.predict`` returns them; the browser maps them onto the 21-level grid and
applies ``clip_nonnegative``, exactly as it does for the in-browser port. Symmetric averaging is
Timestral-only. Models are downloaded from the Hugging Face Hub on first use and kept in memory.
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
ALLOWED = TIMESTRAL | CHRONOS_BOLT

_models: dict[str, object] = {}
_lock = threading.Lock()
DEVICE = "cpu"


def is_chronos(repo: str) -> bool:
    if repo in CHRONOS_BOLT:
        return True
    # Local checkpoints: Chronos-Bolt configs carry a "chronos_config" block.
    cfg = Path(repo) / "config.json"
    return repo.startswith("./") and cfg.is_file() and "chronos_config" in json.loads(cfg.read_text())


def get_model(repo: str):
    if repo not in ALLOWED and not repo.startswith("./"):
        raise ValueError(f"unknown model {repo!r}")
    with _lock:
        if repo not in _models:
            if is_chronos(repo):
                try:
                    from chronos import BaseChronosPipeline
                except ImportError as e:
                    raise ValueError("Chronos-Bolt needs `pip install chronos-forecasting`") from e
                _models[repo] = BaseChronosPipeline.from_pretrained(repo, device_map=DEVICE, torch_dtype=torch.float32)
            else:
                _models[repo] = TimestralForecaster.from_pretrained(repo, device=DEVICE)
        return _models[repo]


def forecast(repo: str, context: np.ndarray, horizon: int, symmetric: bool, clip: bool) -> tuple[list, list]:
    model = get_model(repo)
    if is_chronos(repo):
        # (1, Q, H) on the model's training quantiles, as the official pipeline returns them.
        q = model.predict(torch.tensor(context), prediction_length=horizon)
        return q[0].T.cpu().numpy().tolist(), list(model.quantiles)
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
