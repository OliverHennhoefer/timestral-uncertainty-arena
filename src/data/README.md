# Bundled real-world series

These JSON files feed the "Real data" presets in `src/datasets.ts`. They are imported at build time, so they work offline and on GitHub Pages with no network request. Each file keeps the original values and units; the app rescales the whole series linearly onto 10 to 90 so it sits in the same range as the synthetic streams and the drawing canvas. All series end, so a stream stops after the last point.

Rebuild them from the pinned sources with `python tools/make_datasets.py`.

| File | Series | Points | Taken from | Licence |
| --- | --- | ---: | --- | --- |
| `msft.json` | Microsoft (MSFT) daily close, adjusted, 2008-01-02 to 2009-12-31 | 505 | Boris Marjanovic, [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) on Kaggle, via the copy in [pmdarima](https://github.com/alkaline-ml/pmdarima) 2.0.4 (`load_msft`) | CC0 1.0 (Kaggle dataset); pmdarima is MIT |
| `amzn.json` | Amazon (AMZN) daily close, split adjusted, 2013-01-02 to 2016-12-30 | 1008 | `FANG` data set of the R package [tidyquant](https://github.com/business-science/tidyquant) 1.0.11 | MIT |
| `co2.json` | Mauna Loa atmospheric CO2, weekly, 1963-09-07 to 2001-12-29 (last 2000 weeks; 59 missing weeks filled by linear interpolation) | 2000 | Scripps Institution of Oceanography via CDIAC, as `co2` in [statsmodels](https://www.statsmodels.org/stable/datasets/generated/co2.html) 0.14.4 | Public domain (as documented by statsmodels) |
| `electricity.json` | Electricity demand in England and Wales, half-hourly, 5 June to 16 July 2000 (first six weeks) | 2016 | James W. Taylor (2003), *J. Oper. Res. Soc.* 54, 799-805, as `load_taylor` in pmdarima 2.0.4 (also `taylor` in the R package forecast) | Redistributed under MIT by pmdarima and GPL-3 by forecast; no separate licence from the author |
| `elnino.json` | Sea surface temperature, Niño 1+2 region, monthly, 1950 to 2010 | 732 | NOAA Climate Prediction Center, ERSST v3b, as `elnino` in statsmodels 0.14.4 | Public domain (US government work) |
| `sunspots.json` | Yearly mean sunspot number, 1700 to 2008 | 309 | NOAA National Geophysical Data Center, as `sunspots` in statsmodels 0.14.4 | Public domain (US government work) |
| `airline.json` | International airline passengers, monthly totals in thousands, 1949 to 1960 | 144 | Box, Jenkins and Reinsel (1976), *Time Series Analysis*, Series G, as `load_airpassengers` in pmdarima 2.0.4 (also `AirPassengers` in R) | Classic textbook data, redistributed under MIT by pmdarima, BSD-3 by sktime and GPL by R |
| `nile.json` | Annual flow of the Nile at Aswan, 10^8 m^3, 1871 to 1970 | 100 | Cobb (1978), *Biometrika* 65, 243-251, as `nile` in statsmodels 0.14.4 | Public domain (as documented by statsmodels) |

Stock prices are factual market records. The two stock series come from copies that are published under an explicit open licence (CC0 and MIT), not from a market-data vendor's terms of service.
