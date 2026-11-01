"""Local forecast server for cTimestral, backed by the official Timestral package.

Use it for the larger checkpoints (123M, 209M) or a GPU, where the in-browser port is slow.

    pip install -r server/requirements.txt
    python server/serve.py --port 8765            # then pick "Local server" in the app

Endpoints (JSON, CORS-enabled, bound to 127.0.0.1 by default):
    GET  /health    -> {"models": [...loaded repo ids...], "device": "cpu"}
    POST /forecast  {"model": "mistralai/Timestral-28M", "context": [...], "horizon": 32,
                     "symmetric_averaging": true, "clip_nonnegative": true}
                 -> {"quantiles": [[21 floats] * horizon], "levels": [...], "ms": 12.3}
Models are downloaded from the Hugging Face Hub on first use and kept in memory.
"""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from timestral import TimestralForecaster

ALLOWED = {
    "mistralai/Timestral-5M",
    "mistralai/Timestral-28M",
    "mistralai/Timestral-123M",
    "mistralai/Timestral-209M",
}

_models: dict[str, TimestralForecaster] = {}
_lock = threading.Lock()
DEVICE = "cpu"


def get_model(repo: str) -> TimestralForecaster:
    if repo not in ALLOWED and not repo.startswith("./"):
        raise ValueError(f"unknown model {repo!r}")
    with _lock:
        if repo not in _models:
            _models[repo] = TimestralForecaster.from_pretrained(repo, device=DEVICE)
        return _models[repo]


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
            model = get_model(req.get("model", "mistralai/Timestral-5M"))
            context = np.array([math.nan if v is None else float(v) for v in req["context"]], dtype=np.float32)
            horizon = int(req["horizon"])
            if not 1 <= horizon <= 4096:
                raise ValueError("horizon must be in [1, 4096]")
            t0 = time.perf_counter()
            fc = model.forecast(
                context,
                prediction_length=horizon,
                symmetric_averaging=bool(req.get("symmetric_averaging", True)),
                clip_nonnegative=bool(req.get("clip_nonnegative", True)),
            )
            ms = (time.perf_counter() - t0) * 1e3
            self._json({"quantiles": fc.quantiles[0, 0].cpu().numpy().tolist(), "levels": list(fc.quantile_levels), "ms": ms})
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
    print(f"cTimestral server on http://{args.host}:{args.port} (device={DEVICE})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
