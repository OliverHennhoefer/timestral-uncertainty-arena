/**
 * Real-world series bundled as static JSON (see src/data/README.md for sources and licences).
 * Each one is finite, so a stream ends with the data; values are rescaled to the 0-100
 * drawing range the synthetic presets use.
 */

import type { Preset } from "./synthetic";
import airline from "./data/airline.json";
import amzn from "./data/amzn.json";
import co2 from "./data/co2.json";
import electricity from "./data/electricity.json";
import elnino from "./data/elnino.json";
import giftLoopSeattleD from "./data/gift_loop_seattle_d.json";
import giftLoopSeattleH from "./data/gift_loop_seattle_h.json";
import giftSzTaxi15t from "./data/gift_sz_taxi_15t.json";
import giftEtt1H from "./data/gift_ett1_h.json";
import giftElectricityH from "./data/gift_electricity_h.json";
import giftSolarH from "./data/gift_solar_h.json";
import giftJenaWeatherH from "./data/gift_jena_weather_h.json";
import giftBizitobsL2cH from "./data/gift_bizitobs_l2c_h.json";
import giftBitbrainsFastStorage5t from "./data/gift_bitbrains_fast_storage_5t.json";
import giftSaugeenD from "./data/gift_saugeen_d.json";
import giftUsBirthsD from "./data/gift_us_births_d.json";
import giftM4Hourly from "./data/gift_m4_hourly.json";
import giftCovidDeathsD from "./data/gift_covid_deaths_d.json";
import msft from "./data/msft.json";
import nile from "./data/nile.json";
import sunspots from "./data/sunspots.json";

export interface Dataset {
  title: string;
  unit: string;
  frequency: string;
  start: string;
  end: string;
  n: number;
  values: number[];
}

/** Lowest and highest value land on these drawing coordinates, leaving headroom for intervals. */
export const SCALE_LO = 10;
export const SCALE_HI = 90;

/** Affine map of the whole series onto [SCALE_LO, SCALE_HI], rounded like drawn points. */
export function rescale(values: readonly number[]): number[] {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const k = hi > lo ? (SCALE_HI - SCALE_LO) / (hi - lo) : 0;
  const mid = (SCALE_LO + SCALE_HI) / 2;
  return values.map((v) => Math.round((k ? SCALE_LO + (v - lo) * k : mid) * 100) / 100);
}

/** A series taken from the GIFT-Eval benchmark, with the item it came from. */
interface GiftEvalDataset extends Dataset {
  giftEval: { dataset: string; freq: string; item_id: string; variate: number | null };
}

export const GIFT_EVAL_GROUP = "GIFT-Eval";

function preset(id: string, label: string, data: Dataset): Preset {
  const scaled = rescale(data.values);
  return {
    id,
    label,
    group: "Real data",
    source: `${data.title}, ${data.start} to ${data.end} (${data.n} points, ${data.frequency}; ${data.unit})`,
    make: (n) => scaled.slice(0, n),
  };
}

function giftEval(id: string, label: string, data: GiftEvalDataset): Preset {
  const scaled = rescale(data.values);
  const g = data.giftEval;
  // M4 series carry placeholder timestamps, so no dates are shown for them.
  const span = g.dataset.startsWith("m4") ? "" : `, ${data.start.slice(0, 10)} to ${data.end.slice(0, 10)}`;
  const item = `item ${g.item_id}${g.variate === null ? "" : `, variate ${g.variate}`}`;
  return {
    id,
    label,
    group: GIFT_EVAL_GROUP,
    source: `GIFT-Eval ${g.dataset}, ${item}: ${data.title}${span} (${data.n} points, ${data.frequency}; ${data.unit})`,
    make: (n) => scaled.slice(0, n),
  };
}

export const DATASETS: Preset[] = [
  preset("msft", "MSFT stock, 2008–09", msft),
  preset("amzn", "AMZN stock, 2013–16", amzn),
  preset("electricity", "Electricity demand, 30 min", electricity),
  preset("co2", "CO₂ Mauna Loa, weekly", co2),
  preset("elnino", "Sea temperature (El Niño)", elnino),
  preset("sunspots", "Sunspots, yearly", sunspots),
  preset("airline", "Airline passengers", airline),
  preset("nile", "Nile flow (changepoint)", nile),
  // One series per GIFT-Eval configuration (Aksu et al. 2024), see src/data/README.md.
  giftEval("gift_loop_seattle_d", "Seattle traffic speed, daily", giftLoopSeattleD),
  giftEval("gift_loop_seattle_h", "Seattle traffic speed, hourly", giftLoopSeattleH),
  giftEval("gift_sz_taxi_15t", "Shenzhen taxi speed, 15 min", giftSzTaxi15t),
  giftEval("gift_ett1_h", "ETTh1 transformer load, hourly", giftEtt1H),
  giftEval("gift_electricity_h", "Electricity client MT_001, hourly", giftElectricityH),
  giftEval("gift_solar_h", "Solar power, hourly", giftSolarH),
  giftEval("gift_jena_weather_h", "Jena air pressure, hourly", giftJenaWeatherH),
  giftEval("gift_bizitobs_l2c_h", "BizITObs L2C, hourly", giftBizitobsL2cH),
  giftEval("gift_bitbrains_fast_storage_5t", "Bitbrains VM CPU, 5 min", giftBitbrainsFastStorage5t),
  giftEval("gift_saugeen_d", "Saugeen River flow, daily", giftSaugeenD),
  giftEval("gift_us_births_d", "US births, daily", giftUsBirthsD),
  giftEval("gift_m4_hourly", "M4 hourly", giftM4Hourly),
  giftEval("gift_covid_deaths_d", "COVID-19 deaths, cumulative", giftCovidDeathsD),
];
