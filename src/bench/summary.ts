/**
 * Cross-dataset summaries of a benchmark result: the aggregated leaderboard, the CSV export
 * and the "Findings" text. Findings are computed from the numbers every time, so they stay
 * true when the benchmark is re-run or datasets are added.
 */

import type { BenchResult, DatasetResult, ResultRow } from "./types";

export const NATIVE = "native";

export interface Aggregate {
  id: string;
  /** Datasets with at least one evaluated pair for this row. */
  datasets: number;
  meanRank: number;
  /** Datasets where this row ranked first (ties count). */
  wins: number;
  meanCoverage: number;
  /** Mean |coverage - target| over datasets. */
  meanGap: number;
  /** Lowest coverage on any dataset. */
  minCoverage: number;
  /** Datasets where coverage reached the target within the tolerance. */
  valid: number;
  /** Geometric mean of the interval-score ratio to native (1 = same, lower is better). */
  scoreRatio: number;
  /** Datasets where the paired bootstrap says significantly better / worse than native. */
  better: number;
  worse: number;
}

const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
const ok = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

export function aggregate(datasets: DatasetResult[], ids: string[]): Aggregate[] {
  return ids.map((id) => {
    const rows = datasets.map((d) => d.rows.find((r) => r.id === id)).filter((r): r is ResultRow => !!r && r.n > 0);
    const logs = rows.map((r) => r.scoreRatio).filter((v) => ok(v) && v > 0).map(Math.log);
    return {
      id,
      datasets: rows.length,
      meanRank: mean(rows.map((r) => r.rank)),
      wins: rows.filter((r) => r.rank === 1).length,
      meanCoverage: mean(rows.map((r) => r.coverage).filter(ok)),
      meanGap: mean(rows.map((r) => r.gap).filter(ok)),
      minCoverage: rows.length ? Math.min(...rows.map((r) => (ok(r.coverage) ? r.coverage : Infinity))) : NaN,
      valid: rows.filter((r) => r.valid).length,
      scoreRatio: logs.length ? Math.exp(mean(logs)) : NaN,
      better: rows.filter((r) => r.vsNative === -1).length,
      worse: rows.filter((r) => r.vsNative === 1).length,
    };
  });
}

/** Sort aggregates by mean rank, then wins, then score ratio. */
export function byMeanRank(a: Aggregate, b: Aggregate): number {
  return (a.meanRank - b.meanRank) || (b.wins - a.wins) || (a.scoreRatio - b.scoreRatio);
}

// ---- CSV ---------------------------------------------------------------------------------

const CSV_COLUMNS = [
  "run", "model", "backend", "created_at", "commit", "alpha", "score", "eval_leads", "window", "warmup",
  "dataset", "dataset_label", "group", "points", "method", "method_name", "family", "rank", "tied",
  "n_pairs", "coverage", "coverage_lo", "coverage_hi", "coverage_gap", "valid", "mean_width",
  "interval_score", "interval_score_lo", "interval_score_hi", "score_ratio_vs_native", "vs_native",
] as const;

function cell(v: unknown): string {
  if (v === undefined || v === null || (typeof v === "number" && !Number.isFinite(v))) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One line per (dataset, method), long format. */
export function toCsv(result: BenchResult): string {
  const names = new Map(result.methods.map((m) => [m.id, m]));
  const c = result.config;
  const lines = [CSV_COLUMNS.join(",")];
  for (const d of result.datasets) {
    for (const r of d.rows) {
      const m = names.get(r.id);
      const vs = r.id === NATIVE ? "" : r.vsNative === -1 ? "better" : r.vsNative === 1 ? "worse" : "tie";
      lines.push([
        result.run, result.model.id, result.model.backend, result.createdAt, result.commit, c.alpha, c.scoreType, c.evalLeads, c.window, c.warmup,
        d.id, d.label, d.group, d.points, r.id, m?.name ?? r.id, m?.family, r.rank, r.tied,
        r.n, r.coverage, r.ci?.coverage[0], r.ci?.coverage[1], r.gap, r.valid, r.meanWidth,
        r.intervalScore, r.ci?.intervalScore[0], r.ci?.intervalScore[1], r.scoreRatio, vs,
      ].map(cell).join(","));
    }
  }
  return lines.join("\n") + "\n";
}

// ---- Findings ----------------------------------------------------------------------------

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const list = (a: string[]) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);

export interface Findings {
  findings: string[];
  caveats: string[];
}

/** Plain statements that follow directly from the result, plus the caveats that bound them. */
export function findings(result: BenchResult): Findings {
  const out: string[] = [];
  const caveats: string[] = [];
  const name = (id: string) => result.methods.find((m) => m.id === id)?.name ?? id;
  const ds = result.datasets;
  const N = ds.length;
  if (!N) return { findings: ["No datasets in this result."], caveats };
  const target = 1 - result.config.alpha;
  const tol = result.config.tolerance;
  const ids = result.methods.map((m) => m.id);
  const agg = aggregate(ds, ids).filter((a) => a.datasets > 0).sort(byMeanRank);
  const native = agg.find((a) => a.id === NATIVE);
  const methods = agg.filter((a) => a.id !== NATIVE);
  const modelName = result.model.id === "naive" ? "the naive baseline" : result.model.id.replace(/^.*\//, "");

  // 1. Who ranks best overall.
  const top = agg[0];
  const second = agg[1];
  out.push(
    `${name(top.id)} has the best mean rank across the ${N} datasets (${top.meanRank.toFixed(1)} of ${agg.length}) and ranks first on ${top.wins} of them` +
      (second ? `; next is ${name(second.id)} (${second.meanRank.toFixed(1)}, first on ${second.wins}).` : "."),
  );
  const mostWins = [...agg].sort((a, b) => b.wins - a.wins)[0];
  if (mostWins.id !== top.id && mostWins.wins > top.wins) {
    out.push(`${name(mostWins.id)} ranks first most often (${mostWins.wins} of ${N} datasets) but is less consistent elsewhere (mean rank ${mostWins.meanRank.toFixed(1)}).`);
  }

  // 2. Does the model's own interval hold up?
  if (native) {
    const under = ds.filter((d) => {
      const r = d.rows.find((x) => x.id === NATIVE);
      return r && r.n > 0 && r.coverage < target - tol;
    });
    const worst = [...ds].map((d) => ({ d, r: d.rows.find((x) => x.id === NATIVE)! })).filter((x) => x.r && x.r.n > 0).sort((a, b) => a.r.coverage - b.r.coverage)[0];
    let s = `The native interval of ${modelName} covers ${pct(native.meanCoverage)} on average against the ${pct(target)} target, and falls short by more than ${(100 * tol).toFixed(0)} points on ${under.length} of ${N} datasets`;
    if (worst && under.length) s += ` (lowest: ${worst.d.label}, ${pct(worst.r.coverage)})`;
    s += `. It ranks ${native.meanRank.toFixed(1)} on average and first on ${native.wins}.`;
    out.push(s);
  }

  // 3. Which methods reliably beat native (paired block bootstrap, 95%).
  const reliable = methods.filter((a) => a.better >= Math.ceil(N / 2)).sort((a, b) => b.better - a.better);
  if (reliable.length) {
    out.push(
      `Significantly lower interval score than native on at least half of the datasets: ${list(reliable.slice(0, 5).map((a) => `${name(a.id)} (${a.better}/${N})`))}` +
        (reliable.length > 5 ? `, and ${reliable.length - 5} more.` : "."),
    );
  } else {
    const best = [...methods].sort((a, b) => b.better - a.better)[0];
    if (best) out.push(`No method beats the native interval significantly on half of the datasets; the most often is ${name(best.id)} (${best.better} of ${N}).`);
  }
  const hurt = methods.filter((a) => a.worse > a.better).sort((a, b) => b.worse - a.worse);
  if (hurt.length) {
    out.push(`Worse than native more often than better: ${list(hurt.slice(0, 4).map((a) => `${name(a.id)} (worse on ${a.worse}, better on ${a.better})`))}.`);
  }

  // 4. Coverage validity.
  const alwaysValid = methods.filter((a) => a.valid === a.datasets && a.datasets === N);
  if (alwaysValid.length) {
    out.push(`Coverage within ${(100 * tol).toFixed(0)} points of the target (or above) on every dataset: ${list(alwaysValid.map((a) => name(a.id)))}.`);
  } else {
    const best = [...methods].sort((a, b) => b.valid - a.valid)[0];
    if (best) out.push(`No method reaches the coverage target (within ${(100 * tol).toFixed(0)} points) on every dataset; the most often is ${name(best.id)} (${best.valid} of ${N}).`);
  }
  const lowest = [...methods].sort((a, b) => a.minCoverage - b.minCoverage)[0];
  if (lowest && lowest.minCoverage < target - 0.05) {
    const where = ds.find((d) => d.rows.find((r) => r.id === lowest.id)?.coverage === lowest.minCoverage);
    out.push(`The largest single shortfall is ${name(lowest.id)} on ${where?.label ?? "one dataset"}, at ${pct(lowest.minCoverage)} coverage.`);
  }

  // 5. Synthetic versus real data, when both are present.
  const groups = [...new Set(ds.map((d) => d.group))];
  if (groups.length > 1) {
    const parts = groups.map((g) => {
      const sub = ds.filter((d) => d.group === g);
      const a = aggregate(sub, ids).filter((x) => x.datasets > 0).sort(byMeanRank)[0];
      return `${g.toLowerCase()} (${sub.length}): ${name(a.id)}, mean rank ${a.meanRank.toFixed(1)}`;
    });
    out.push(`Best mean rank by group: ${parts.join("; ")}.`);
  }

  // Caveats that bound every statement above.
  if (result.model.id === "naive") {
    caveats.push("These results use the naive baseline forecaster, not Timestral. They show how the calibration methods repair a simple model's intervals and say nothing about Timestral's own uncertainty. Run the benchmark with a Timestral checkpoint for that (see benchmarks/README.md).");
  }
  const s = ds.filter((d) => d.group === "Synthetic").length;
  if (s) caveats.push(`Each of the ${s} synthetic presets is one seeded draw (seed ${result.config.seed}), so its ranks carry sampling noise of their own.`);
  const short = ds.filter((d) => d.points < 400);
  if (short.length) caveats.push(`${list(short.map((d) => `${d.label} (${d.points} points)`))} ${short.length === 1 ? "is" : "are"} short; ${short.length === 1 ? "its" : "their"} ranks rest on few independent stretches and many rows tie.`);
  caveats.push(
    `One configuration: α = ${result.config.alpha}, ${result.config.scoreType.toUpperCase()} score, leads 1–${result.config.evalLeads}, window ${result.config.window}, warm-up ${result.config.warmup}, default method parameters. "Significantly" means the 95% paired block-bootstrap interval of the score difference excludes zero, with no correction for testing many methods on many datasets.`,
  );
  caveats.push("Ranks are by mean interval (Winkler) score, which rewards narrow intervals and penalises misses by 2/α times the miss distance; a method can rank well with coverage somewhat below target.");
  return { findings: out, caveats };
}
