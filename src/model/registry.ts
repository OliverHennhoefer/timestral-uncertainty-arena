/** The released Timestral checkpoints on the Hugging Face Hub (collection `mistralai/timestral`). */

export interface ModelEntry {
  repo: string;
  label: string;
  params: string;
  /** Approximate float32 download size in MB. */
  approxMB: number;
  note: string;
}

export const MODELS: ModelEntry[] = [
  { repo: "mistralai/Timestral-5M", label: "Timestral 5M", params: "5.2M", approxMB: 21, note: "Fastest; smooth per-stroke forecasts on a laptop CPU." },
  { repo: "mistralai/Timestral-28M", label: "Timestral 28M", params: "27.8M", approxMB: 111, note: "Good balance in the browser." },
  { repo: "mistralai/Timestral-123M", label: "Timestral 123M", params: "123M", approxMB: 490, note: "Heavy in the browser; prefer the local server." },
  { repo: "mistralai/Timestral-209M", label: "Timestral 209M", params: "209M", approxMB: 840, note: "Flagship; the local server is recommended." },
];

export const HF_BASE = "https://huggingface.co";

export function fileUrl(repo: string, file: string, revision = "main"): string {
  return `${HF_BASE}/${repo}/resolve/${revision}/${file}`;
}

export const CACHE_NAME = "ctimestral-weights-v1";
