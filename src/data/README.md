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

## GIFT-Eval series

The `gift_*.json` files feed the "GIFT-Eval" presets. Each is one series from the [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) benchmark (Aksu et al., "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation", arXiv:2410.10393), taken from the Hugging Face dataset `Salesforce/GiftEval` at revision `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`: the first item of the configuration, its first variate if it has several, and its last 2000 points (all of them if it is shorter), rounded to four decimals. Each file records the configuration, item and variate under `giftEval`. `hospital/M` is left out because its first item has only 84 points.

Rebuild them with `python tools/make_gift_eval.py <download_dir> <out_dir>` (the script lists the files to download and needs numpy, pandas and pyarrow), then prefix the file names with `gift_`.

Licence: the dataset card declares Apache-2.0 for the GIFT-Eval repository as a whole and adds "This release is for research purposes only in support of an academic paper". It names no licence per dataset, and the original publishers' terms have not been checked here. The units in the files are not part of GIFT-Eval: "MHz" for Bitbrains is inferred from the value range, and the ETT and solar units are not stated.

| File | Series (GIFT-Eval configuration, item) | Points |
| --- | --- | ---: |
| `gift_loop_seattle_d.json` | Seattle loop-detector traffic speed, daily, 2015 (`LOOP_SEATTLE/D`, item `0`) | 365 |
| `gift_loop_seattle_h.json` | Seattle loop-detector traffic speed, hourly, last 2000 hours of 2015 (`LOOP_SEATTLE/H`, item `0`) | 2000 |
| `gift_sz_taxi_15t.json` | Shenzhen taxi road speed, 15 min, January 2015 (`SZ_TAXI/15T`, item `90217`) | 2000 |
| `gift_ett1_h.json` | ETTh1 transformer load, hourly, last 2000 hours (`ett1/H`, item `item_0`, variate 0) | 2000 |
| `gift_electricity_h.json` | Electricity consumption of one Portuguese client, hourly, last 2000 hours of 2014 (`electricity/H`, item `MT_001`) | 2000 |
| `gift_solar_h.json` | Solar plant power output, hourly, last 2000 hours of 2006 (`solar/H`, item `item_0`) | 2000 |
| `gift_jena_weather_h.json` | Jena weather station, hourly, last 2000 hours of 2020 (`jena_weather/H`, item `item_0`, variate 0) | 2000 |
| `gift_bizitobs_l2c_h.json` | BizITObs application metric, hourly, last 2000 hours (`bizitobs_l2c/H`, item `L2C`, variate 0) | 2000 |
| `gift_bitbrains_fast_storage_5t.json` | Bitbrains fastStorage VM trace, 5 min, last 2000 points; 13 gaps filled by linear interpolation (`bitbrains_fast_storage/5T`, item `fastStorage_860`, variate 0) | 2000 |
| `gift_saugeen_d.json` | Saugeen River flow, daily, last 2000 days to 1979 (`saugeen/D`, item `T1`) | 2000 |
| `gift_us_births_d.json` | US births, daily, last 2000 days to 1988 (`us_births/D`, item `T1`) | 2000 |
| `gift_m4_hourly.json` | M4 competition hourly series (`m4_hourly`, item `0`); its timestamps are placeholders | 748 |
| `gift_covid_deaths_d.json` | COVID-19 cumulative deaths, daily, 22 Jan to 20 Aug 2020 (`covid_deaths/D`, item `T1`) | 212 |
