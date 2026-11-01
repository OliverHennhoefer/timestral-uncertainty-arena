"""Rebuild the bundled real-world datasets in src/data/ from their published sources.

Every source is a pinned release of an openly licensed package, so the output is
reproducible. Files are read as data (CSV text, R data, number literals); no
downloaded code is executed.

    pip install rdata pandas
    python tools/make_datasets.py            # writes src/data/*.json

Sources and licences are listed in src/data/README.md.
"""

from __future__ import annotations

import datetime
import io
import json
import math
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "data"

STATSMODELS = "statsmodels==0.14.4"
PMDARIMA = "pmdarima==2.0.4"
TIDYQUANT_FANG = "https://raw.githubusercontent.com/cran/tidyquant/1.0.11/data/FANG.rda"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def wheel(spec: str) -> zipfile.ZipFile:
    """Any wheel of a pinned PyPI release; the bundled data files are platform independent."""
    name, version = spec.split("==")
    meta = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    url = next(f["url"] for f in meta["urls"] if f["packagetype"] == "bdist_wheel")
    return zipfile.ZipFile(io.BytesIO(fetch(url)))


def numbers_after(src: str, marker: str) -> list[float]:
    """Number literals of the first `np.array([...])` after `marker` in a Python source file."""
    body = src[src.index(marker) :]
    body = body[body.index("np.array([") + len("np.array([") : body.index("])")]
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", body)]


def interpolate(values: list[float | None]) -> list[float]:
    """Linear interpolation over interior gaps."""
    out = list(values)
    known = [i for i, v in enumerate(out) if v is not None]
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            out[i] = out[a] + (out[b] - out[a]) * (i - a) / (b - a)
    return out  # type: ignore[return-value]


def write(id_: str, meta: dict, values: list[float], digits: int) -> None:
    values = [round(v, digits) if digits else int(round(v)) for v in values]
    assert values and all(math.isfinite(v) for v in values), id_
    doc = {**meta, "n": len(values), "values": values}
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    (OUT / f"{id_}.json").write_text(text + "\n", encoding="utf-8")
    print(f"{id_:12s} n={len(values):5d}  min={min(values)}  max={max(values)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    sm = wheel(STATSMODELS)
    pm = wheel(PMDARIMA)

    def sm_csv(name: str) -> list[list[str]]:
        text = sm.read(f"statsmodels/datasets/{name}/{name}.csv").decode()
        return [line.replace('"', "").split(",") for line in text.strip().splitlines()[1:]]

    # --- Stocks -------------------------------------------------------------------------
    # MSFT: Kaggle "Huge Stock Market Dataset" (CC0), as shipped in pmdarima (MIT).
    tar = tarfile.open(fileobj=io.BytesIO(pm.read("pmdarima/datasets/data/msft.tar.gz")))
    rows = tar.extractfile("msft.csv").read().decode().strip().splitlines()[1:]  # type: ignore[union-attr]
    msft = [(r.split(",")[0], float(r.split(",")[4])) for r in rows]
    msft = [(d, c) for d, c in msft if "2008-01-01" <= d <= "2009-12-31"]
    write("msft", {
        "title": "Microsoft (MSFT) daily close",
        "unit": "USD, split and dividend adjusted", "frequency": "trading day",
        "start": msft[0][0], "end": msft[-1][0],
    }, [c for _, c in msft], 3)

    # AMZN: tidyquant's FANG data set (MIT).
    import rdata  # noqa: PLC0415

    fang = rdata.conversion.convert(rdata.parser.parse_data(fetch(TIDYQUANT_FANG)))["FANG"]
    amzn = fang[fang["symbol"] == "AMZN"].sort_values("date")
    # R stores Date as days since 1970-01-01.
    dates = [str(datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d))) for d in amzn["date"]]
    write("amzn", {
        "title": "Amazon (AMZN) daily close",
        "unit": "USD, split adjusted", "frequency": "trading day",
        "start": dates[0], "end": dates[-1],
    }, [float(v) for v in amzn["adjusted"]], 2)

    # --- Classic series -----------------------------------------------------------------
    co2 = sm_csv("co2")
    co2 = co2[-2000:]  # the last 2000 weeks, to match the synthetic stream length
    write("co2", {
        "title": "Mauna Loa atmospheric CO2, weekly",
        "unit": "ppm", "frequency": "week",
        "start": f"{co2[0][0][:4]}-{co2[0][0][4:6]}-{co2[0][0][6:]}",
        "end": f"{co2[-1][0][:4]}-{co2[-1][0][4:6]}-{co2[-1][0][6:]}",
        "note": "gaps filled by linear interpolation",
    }, interpolate([float(v) if v else None for _, v in co2]), 1)

    sun = sm_csv("sunspots")
    write("sunspots", {
        "title": "Yearly sunspot number",
        "unit": "Wolf number", "frequency": "year", "start": sun[0][0], "end": sun[-1][0],
    }, [float(v) for _, v in sun], 1)

    nile = sm_csv("nile")
    write("nile", {
        "title": "Nile river annual flow at Aswan",
        "unit": "10^8 m^3", "frequency": "year", "start": nile[0][0], "end": nile[-1][0],
    }, [float(v) for _, v in nile], 0)

    nino = sm_csv("elnino")
    write("elnino", {
        "title": "Sea surface temperature, Nino 1+2 region",
        "unit": "°C", "frequency": "month", "start": f"{nino[0][0]}-01", "end": f"{nino[-1][0]}-12",
    }, [float(v) for row in nino for v in row[1:]], 2)

    air = numbers_after(pm.read("pmdarima/datasets/airpassengers.py").decode(), "rslt = np.array")
    write("airline", {
        "title": "International airline passengers (Box & Jenkins Series G)",
        "unit": "thousands", "frequency": "month", "start": "1949-01", "end": "1960-12",
    }, air, 0)

    taylor = numbers_after(pm.read("pmdarima/datasets/taylor.py").decode(), "rslt = np.array")
    taylor = taylor[: 6 * 7 * 48]  # six weeks, half-hourly
    write("electricity", {
        "title": "Electricity demand, England and Wales, half-hourly",
        "unit": "MW", "frequency": "30 min", "start": "2000-06-05 00:00", "end": "2000-07-16 23:30",
    }, taylor, 0)


if __name__ == "__main__":
    sys.exit(main())
