/**
 * Released checkpoints on the Hugging Face Hub. Timestral (collection `mistralai/timestral`) and
 * Chronos-Bolt (`amazon/chronos-bolt-*`) run in the browser port and on the local server;
 * Chronos-2, TimesFM and Moirai have no browser port and run on the local server only.
 */

export type ModelFamily = "timestral" | "chronos-bolt" | "chronos-2" | "timesfm" | "moirai";

export const FAMILY_NAMES: Record<ModelFamily, string> = {
  timestral: "Timestral",
  "chronos-bolt": "Chronos-Bolt",
  "chronos-2": "Chronos-2",
  timesfm: "TimesFM",
  moirai: "Moirai",
};

/** Families that predict only the 10%, ..., 90% quantiles, so bands beyond 80% use estimated tails. */
export const NINE_LEVEL_FAMILIES: ReadonlySet<ModelFamily> = new Set(["chronos-bolt", "timesfm", "moirai"]);

export interface ModelEntry {
  family: ModelFamily;
  repo: string;
  label: string;
  params: string;
  /** Approximate float32 download size in MB. */
  approxMB: number;
  note: string;
  /** No browser port: listed for the local server only. */
  serverOnly?: true;
}

export const MODELS: ModelEntry[] = [
  { family: "timestral", repo: "mistralai/Timestral-5M", label: "Timestral 5M", params: "5.2M", approxMB: 21, note: "Fastest; smooth per-stroke forecasts on a laptop CPU." },
  { family: "timestral", repo: "mistralai/Timestral-28M", label: "Timestral 28M", params: "27.8M", approxMB: 111, note: "Good balance in the browser." },
  { family: "timestral", repo: "mistralai/Timestral-123M", label: "Timestral 123M", params: "123M", approxMB: 490, note: "Heavy in the browser; prefer the local server." },
  { family: "timestral", repo: "mistralai/Timestral-209M", label: "Timestral 209M", params: "209M", approxMB: 840, note: "Flagship; the local server is recommended." },
  { family: "chronos-bolt", repo: "amazon/chronos-bolt-tiny", label: "Chronos-Bolt Tiny", params: "8.6M", approxMB: 35, note: "Fastest; smooth per-stroke forecasts on a laptop CPU." },
  { family: "chronos-bolt", repo: "amazon/chronos-bolt-mini", label: "Chronos-Bolt Mini", params: "21M", approxMB: 85, note: "Good balance in the browser." },
  { family: "chronos-bolt", repo: "amazon/chronos-bolt-small", label: "Chronos-Bolt Small", params: "48M", approxMB: 191, note: "Slower per forecast; fine with a short max context." },
  { family: "chronos-bolt", repo: "amazon/chronos-bolt-base", label: "Chronos-Bolt Base", params: "205M", approxMB: 821, note: "Largest; the local server is recommended." },
  { family: "chronos-2", repo: "amazon/chronos-2", label: "Chronos-2", params: "120M", approxMB: 478, note: "Predicts 21 quantile levels, like Timestral.", serverOnly: true },
  { family: "timesfm", repo: "google/timesfm-2.5-200m-pytorch", label: "TimesFM 2.5", params: "231M", approxMB: 925, note: "About 0.3 s per forecast on a laptop CPU.", serverOnly: true },
  { family: "timesfm", repo: "google/timesfm-3.0-pytorch", label: "TimesFM 3.0", params: "331M", approxMB: 1323, note: "About 0.3 s per forecast on a laptop CPU.", serverOnly: true },
  { family: "moirai", repo: "Salesforce/moirai-2.0-R-small", label: "Moirai 2.0 Small", params: "11M", approxMB: 46, note: "Fast; non-commercial license (CC BY-NC 4.0).", serverOnly: true },
];

export const familyOf = (repo: string): ModelFamily | undefined => MODELS.find((m) => m.repo === repo)?.family;

export const HF_BASE = "https://huggingface.co";

export function fileUrl(repo: string, file: string, revision = "main"): string {
  return `${HF_BASE}/${repo}/resolve/${revision}/${file}`;
}

export const CACHE_NAME = "ctimestral-weights-v1";
