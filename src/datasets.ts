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

export const DATASETS: Preset[] = [
  preset("msft", "MSFT stock, 2008–09", msft),
  preset("amzn", "AMZN stock, 2013–16", amzn),
  preset("electricity", "Electricity demand, 30 min", electricity),
  preset("co2", "CO₂ Mauna Loa, weekly", co2),
  preset("elnino", "Sea temperature (El Niño)", elnino),
  preset("sunspots", "Sunspots, yearly", sunspots),
  preset("airline", "Airline passengers", airline),
  preset("nile", "Nile flow (changepoint)", nile),
];
