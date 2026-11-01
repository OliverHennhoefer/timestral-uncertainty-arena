"""Generate a parity fixture for the TypeScript port of Timestral.

Builds a small, randomly initialised Timestral network with the official inference package,
runs `TimestralForecaster.forecast` on a few series, and writes

    <out>/config.json
    <out>/model.safetensors
    <out>/cases.json

`tests/parity.test.ts` loads the same checkpoint with the browser port and checks that the
quantiles match. No Hugging Face download is needed, so this runs in CI.

Usage:
    pip install "git+https://github.com/mistralai/timestral.git"
    python tools/make_parity_fixture.py tests/fixtures/parity
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from timestral import TimestralConfig, TimestralForecaster, TimestralModel

KNOTS = [0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]


def main(out: Path) -> None:
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    config = TimestralConfig(
        patch_size=32,
        d_model=48,
        num_heads=4,
        mlp_hidden_dim=96,
        block_schedule=("time_variate", "time", "variate_time", "variate"),
        quantile_knots=tuple(KNOTS),
        block_size=4,
        context_length=1024,
    )
    model = TimestralModel(config)
    with torch.no_grad():
        # Default init leaves norms at 1 and the forecast query at 0; perturb everything so
        # every weight is exercised.
        for name, p in model.named_parameters():
            if name.endswith(".scale"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.mul_(1.5)
    model.eval()
    config.save(out / "config.json")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))

    forecaster = TimestralForecaster(model)
    rng = np.random.default_rng(1)
    t = np.arange(400)
    series = {
        "sine": 50 + 20 * np.sin(t / 9.0) + rng.normal(0, 2, t.size),
        "walk": np.cumsum(rng.normal(0, 1, t.size)) - 10,
        "short": np.array([3.0, 4.0, 2.5, 5.0, 6.5, 6.0, 7.5]),
    }
    walk_nan = series["walk"].copy()
    walk_nan[[5, 50, 51, 52, 300]] = np.nan
    series["walk_nan"] = walk_nan

    cases = []
    for name, values in series.items():
        for length in sorted({min(len(values), n) for n in (7, 45, 160, 400)}):
            ctx = values[:length]
            for horizon in (1, 32, 70, 200):
                for sym in (True, False):
                    fc = forecaster.forecast(ctx, prediction_length=horizon, symmetric_averaging=sym, clip_nonnegative=True)
                    cases.append(
                        {
                            "name": f"{name}-T{length}-H{horizon}-sym{int(sym)}",
                            "context": [None if np.isnan(v) else float(v) for v in ctx],
                            "horizon": horizon,
                            "symmetric_averaging": sym,
                            "clip_nonnegative": True,
                            "max_context": None,
                            "quantiles": fc.quantiles[0, 0].numpy().astype(float).tolist(),
                        }
                    )
    # A context cap shorter than the series.
    capped = TimestralForecaster(model, max_context=96)
    fc = capped.forecast(series["sine"], prediction_length=40)
    cases.append(
        {
            "name": "sine-capped96-H40",
            "context": series["sine"].tolist(),
            "horizon": 40,
            "symmetric_averaging": True,
            "clip_nonnegative": True,
            "max_context": 96,
            "quantiles": fc.quantiles[0, 0].numpy().astype(float).tolist(),
        }
    )
    (out / "cases.json").write_text(json.dumps({"levels": KNOTS, "cases": cases}))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/parity"))
