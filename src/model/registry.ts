/**
 * Released checkpoints on the Hugging Face Hub that the browser port and the local server can run:
 * Timestral (collection `mistralai/timestral`) and Chronos-Bolt (`amazon/chronos-bolt-*`).
 */

export type ModelFamily = "timestral" | "chronos-bolt";

export const FAMILY_NAMES: Record<ModelFamily, string> = { timestral: "Timestral", "chronos-bolt": "Chronos-Bolt" };

export interface ModelEntry {
  family: ModelFamily;
  repo: string;
  label: string;
  params: string;
  /** Approximate float32 download size in MB. */
  approxMB: number;
  note: string;
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
];

export const familyOf = (repo: string): ModelFamily | undefined => MODELS.find((m) => m.repo === repo)?.family;

export const HF_BASE = "https://huggingface.co";

export function fileUrl(repo: string, file: string, revision = "main"): string {
  return `${HF_BASE}/${repo}/resolve/${revision}/${file}`;
}

export const CACHE_NAME = "ctimestral-weights-v1";
