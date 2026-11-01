/**
 * The result format the benchmark writes and the leaderboard page reads. Kept free of
 * runtime imports so the page does not bundle the engine or the datasets.
 */

import type { ScoreType } from "../engine/engine";

/** Bumped whenever the result format changes, so the page can refuse files it cannot read. */
export const RESULT_SCHEMA = 1;

export interface BenchConfig {
  alpha: number;
  evalLeads: number;
  scoreType: ScoreType;
  window: number;
  warmup: number;
  /** Forecast horizon requested from the model (at least `evalLeads`). */
  horizon: number;
  maxContext: number;
  symmetric: boolean;
  clip: boolean;
  /** Points drawn from each synthetic preset (real datasets use all their points, up to this). */
  length: number;
  /** Seed of the synthetic presets. */
  seed: number;
  /** Block-bootstrap replicates for the error bars and ties. */
  bootstrap: number;
  /** Coverage tolerance for "valid" (coverage >= 1 - alpha - tolerance). */
  tolerance: number;
}

export interface ResultRow {
  /** Method id, or "native" for the model's own interval. */
  id: string;
  n: number;
  coverage: number;
  /** |coverage - (1 - alpha)| */
  gap: number;
  valid: boolean;
  meanWidth: number;
  /** Mean interval (Winkler) score; lower is better. */
  intervalScore: number;
  /** Rank by interval score within the dataset (1 = best); ties share a rank. */
  rank: number;
  tied: boolean;
  /** Interval score relative to the native interval (1 = same). */
  scoreRatio: number;
  /** Paired bootstrap against native: -1 significantly better, 1 significantly worse, 0 not distinguishable. */
  vsNative: -1 | 0 | 1;
  /** 95% block-bootstrap intervals; absent when the series is too short to resample. */
  ci?: { coverage: [number, number]; intervalScore: [number, number] };
  /** Coverage per lead (1..evalLeads). */
  leadCoverage: number[];
}

export interface DatasetResult {
  id: string;
  label: string;
  group: string;
  source?: string;
  points: number;
  /** Evaluated (forecast, truth) pairs per row. */
  pairs: number;
  seconds: number;
  rows: ResultRow[];
}

export interface ModelInfo {
  /** "naive", "browser" (the TypeScript port) or "server" (the official PyTorch package). */
  backend: string;
  /** Checkpoint, e.g. "mistralai/Timestral-5M", or "naive". */
  id: string;
  label: string;
}

export interface BenchResult {
  schema: number;
  /** File stem and selector key, e.g. "naive" or "timestral-5m". */
  run: string;
  createdAt: string;
  /** Git commit of the code that produced the result, when known. */
  commit?: string;
  model: ModelInfo;
  config: BenchConfig;
  methods: { id: string; name: string; family: string; guarantee: boolean; reference: string; color: string; params: Record<string, number> }[];
  datasets: DatasetResult[];
}
