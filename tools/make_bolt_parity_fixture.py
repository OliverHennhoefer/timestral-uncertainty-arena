"""Generate a parity fixture for the TypeScript port of Chronos-Bolt.

Builds a small, randomly initialised Chronos-Bolt network with the official
``chronos-forecasting`` package, runs ``ChronosBoltPipeline.predict`` on a few series, and writes

    <out>/config.json
    <out>/model.safetensors
    <out>/cases.json

``tests/bolt-parity.test.ts`` loads the same checkpoint with the browser port and checks that the
quantiles match. No Hugging Face download is needed, so this runs in CI.

Usage:
    pip install chronos-forecasting
    python tools/make_bolt_parity_fixture.py tests/fixtures/bolt-parity
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from chronos import ChronosBoltPipeline
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from transformers import T5Config

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 256


def main(out: Path) -> None:
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    # Same layout as the released checkpoints (T5 with ReLU, [REG] token, 16-step patches),
    # only smaller, with a short prediction length so long horizons exercise the heuristic.
    config = T5Config(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        dropout_rate=0.0,
        architectures=["ChronosBoltModelForForecasting"],
        chronos_config={
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "quantiles": QUANTILES,
            "use_reg_token": True,
        },
        chronos_pipeline_class="ChronosBoltPipeline",
    )
    model = ChronosBoltModelForForecasting(config)
    with torch.no_grad():
        # Perturb norms and biases so every weight is exercised.
        for name, p in model.named_parameters():
            if name.endswith("layer_norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn_like(p))
            elif "relative_attention_bias" in name:
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.mul_(1.5)
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    pipeline = ChronosBoltPipeline(model=model)

    rng = np.random.default_rng(1)
    t = np.arange(400)
    series = {
        "sine": 50 + 20 * np.sin(t / 9.0) + rng.normal(0, 2, t.size),
        "walk": np.cumsum(rng.normal(0, 1, t.size)) - 10,
        "short": np.array([3.0, 4.0, 2.5, 5.0, 6.5, 6.0, 7.5]),
        "constant": np.full(40, 7.0),
    }
    walk_nan = series["walk"].copy()
    walk_nan[[5, 50, 51, 52, 300]] = np.nan
    walk_nan[100:140] = np.nan  # a fully missing patch
    series["walk_nan"] = walk_nan

    def case(name: str, ctx: np.ndarray, horizon: int, max_context: int | None = None) -> dict:
        fed = ctx if max_context is None else ctx[-max_context:]
        q = pipeline.predict(torch.tensor(fed, dtype=torch.float32), prediction_length=horizon)
        return {
            "name": name,
            "context": [None if np.isnan(v) else float(v) for v in ctx],
            "horizon": horizon,
            "max_context": max_context,
            "quantiles": q[0].T.numpy().astype(float).tolist(),  # (H, Q)
        }

    cases = []
    for name, values in series.items():
        for length in sorted({min(len(values), n) for n in (7, 45, 160, 400)}):
            for horizon in (1, PREDICTION_LENGTH, 70):
                cases.append(case(f"{name}-T{length}-H{horizon}", values[:length], horizon))
    cases.append(case("sine-capped96-H40", series["sine"], 40, max_context=96))
    (out / "cases.json").write_text(json.dumps({"levels": QUANTILES, "cases": cases}))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/bolt-parity"))
