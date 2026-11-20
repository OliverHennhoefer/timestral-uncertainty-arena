"""Extract one univariate series per GIFT-Eval config into compact JSON files.

Usage:
    python extract_gift_eval.py <download_dir> <out_dir>

<download_dir> holds the GIFT-Eval Arrow files at their repo paths, e.g.
<download_dir>/LOOP_SEATTLE/D/data-00000-of-00001.arrow, downloaded from
https://huggingface.co/datasets/Salesforce/GiftEval/resolve/<REVISION>/<path>

Rules: first item (row 0) of each config, first variate if multivariate,
last 2000 points (all if shorter), missing values filled by linear
interpolation (leading/trailing gaps take the nearest observed value),
values rounded to at most 4 decimal places. A config is skipped if the
series has fewer than 100 points or is constant.
Requires: numpy, pandas, pyarrow.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
MAX_POINTS = 2000
MIN_POINTS = 100

# id, repo path, GIFT-Eval config name, title, unit, human frequency
CONFIGS = [
    ("loop_seattle_d", "LOOP_SEATTLE/D", "LOOP_SEATTLE/D", "Seattle loop detector traffic speed, daily", "mph", "daily"),
    ("loop_seattle_h", "LOOP_SEATTLE/H", "LOOP_SEATTLE/H", "Seattle loop detector traffic speed, hourly", "mph", "hourly"),
    ("sz_taxi_15t", "SZ_TAXI/15T", "SZ_TAXI/15T", "Shenzhen taxi road speed, 15 min", "km/h", "15 min"),
    ("ett1_h", "ett1/H", "ett1/H", "ETTh1 transformer high useful load (HUFL), hourly", "load (unit not stated)", "hourly"),
    ("electricity_h", "electricity/H", "electricity/H", "Portuguese client electricity consumption, hourly", "kW (hourly aggregate)", "hourly"),
    ("solar_h", "solar/H", "solar/H", "Alabama solar plant power output, hourly", "power (hourly aggregate, unit not stated)", "hourly"),
    ("jena_weather_h", "jena_weather/H", "jena_weather/H", "Jena weather station air pressure, hourly", "mbar", "hourly"),
    ("bizitobs_l2c_h", "bizitobs_l2c/H", "bizitobs_l2c/H", "BizITObs L2C application metric (variate 0), hourly", "not stated", "hourly"),
    ("bitbrains_fast_storage_5t", "bitbrains_fast_storage/5T", "bitbrains_fast_storage/5T", "Bitbrains fastStorage VM CPU usage, 5 min", "MHz (inferred)", "5 min"),
    ("saugeen_d", "saugeenday/D", "saugeen/D", "Saugeen River flow, daily", "m³/s", "daily"),
    ("us_births_d", "us_births/D", "us_births/D", "US births, daily", "births", "daily"),
    ("m4_hourly", "m4_hourly", "m4_hourly", "M4 competition hourly series", "unitless", "hourly"),
    ("covid_deaths_d", "covid_deaths", "covid_deaths/D", "COVID-19 cumulative deaths, daily", "deaths (cumulative)", "daily"),
    ("hospital_m", "hospital", "hospital/M", "Hospital monthly patient counts", "patients", "monthly"),
]

PANDAS_FREQ = {"D": "D", "H": "h", "15T": "15min", "5T": "5min", "10T": "10min", "W": "W", "M": "MS"}


def read_table(path):
    try:
        return pa.ipc.open_stream(path).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(path).read_all()


def interpolate(a):
    a = a.astype(float)
    mask = np.isnan(a)
    if mask.all():
        return a
    idx = np.arange(len(a))
    a[mask] = np.interp(idx[mask], idx[~mask], a[~mask])
    return a


def fmt(v):
    r = round(float(v), 4)
    return int(r) if r == int(r) else r


def main(dl_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = []
    for cid, path, cfg, title, unit, human_freq in CONFIGS:
        table = read_table(os.path.join(dl_dir, path, "data-00000-of-00001.arrow"))
        row = table.slice(0, 1).to_pylist()[0]
        target = np.array(row["target"], dtype=float)
        variate = None
        if target.ndim == 2:
            variate = 0
            target = target[0]
        full_len = len(target)
        offset = max(0, full_len - MAX_POINTS)
        series = target[offset:]
        n_missing = int(np.isnan(series).sum())
        series = interpolate(series)
        if len(series) < MIN_POINTS:
            report.append(f"SKIP {cid}: only {len(series)} points")
            continue
        if np.nanstd(series) == 0:
            report.append(f"SKIP {cid}: constant series")
            continue
        freq = row["freq"]
        idx = pd.date_range(pd.Timestamp(row["start"]), periods=full_len, freq=PANDAS_FREQ[freq])
        start, end = idx[offset], idx[-1]
        tfmt = "%Y-%m-%d" if freq in ("D", "W", "M") else "%Y-%m-%dT%H:%M:%S"
        out = {
            "title": title,
            "unit": unit,
            "frequency": human_freq,
            "start": start.strftime(tfmt),
            "end": end.strftime(tfmt),
            "n": len(series),
            "values": [fmt(v) for v in series],
            "giftEval": {"dataset": cfg, "freq": freq, "item_id": str(row["item_id"]), "variate": variate},
        }
        with open(os.path.join(out_dir, cid + ".json"), "w") as f:
            f.write(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
        report.append(
            f"OK   {cid}: n={len(series)} of {full_len}, item={row['item_id']}, variate={variate}, "
            f"filled={n_missing}, {out['start']} .. {out['end']}"
        )
    print("\n".join(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
