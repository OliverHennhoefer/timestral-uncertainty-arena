/**
 * Plain-text reports of benchmark results, for reading in a terminal or a log: one table per
 * run, and a comparison across the runs of a sweep. The leaderboard page does not use this.
 */

import { NATIVE, aggregate, byMeanRank, findings, paramNote } from "./summary";
import type { BenchResult } from "./types";

const pct = (v: number, d = 1) => (Number.isFinite(v) ? `${(100 * v).toFixed(d)}%` : "–");
const num = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");

function table(head: string[], rows: string[][], left = 1): string {
  const w = head.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)));
  const line = (r: string[]) => r.map((c, i) => (i < left ? c.padEnd(w[i]) : c.padStart(w[i]))).join("  ").trimEnd();
  return [line(head), ...rows.map(line)].join("\n");
}

/** `timestral-5m · 98% (α = 0.02) · CQR · leads 1–16 · window 500 · warm-up 10 · 18 datasets` */
export function runTitle(r: BenchResult): string {
  const c = r.config;
  return `${r.run} · ${pct(1 - c.alpha, 0)} (α = ${c.alpha}) · ${c.scoreType.toUpperCase()} · leads 1–${c.evalLeads} · window ${c.window} · warm-up ${c.warmup} · ${r.datasets.length} datasets`;
}

/** The aggregated leaderboard of one run, its findings and caveats, as text. */
export function textSummary(r: BenchResult, opts: { findings?: boolean } = {}): string {
  const N = r.datasets.length;
  const name = (id: string) => r.methods.find((m) => m.id === id)?.name ?? id;
  const agg = aggregate(r.datasets, r.methods.map((m) => m.id)).filter((a) => a.datasets > 0).sort(byMeanRank);
  const rows = agg.map((a) => [
    name(a.id), num(a.meanRank, 1), `${a.wins}`, pct(a.meanCoverage), pct(a.meanGap), pct(a.minCoverage), `${a.valid}/${a.datasets}`,
    num(a.scoreRatio, 3), a.id === NATIVE ? "–" : `${a.better}`, a.id === NATIVE ? "–" : `${a.worse}`,
  ]);
  const parts = [
    runTitle(r),
    `model: ${r.model.label}; ${paramNote(r)}${r.commit ? `; commit ${r.commit}` : ""}`,
    "",
    table(["method", "rank", "wins", "cover", "gap", "min", "valid", "ratio", "better", "worse"], rows),
    "",
    `rank: mean rank by interval score over ${N} datasets (1 = best). ratio: geometric mean interval score relative to native. better/worse: datasets where the paired bootstrap separates the method from native.`,
  ];
  if (opts.findings !== false) {
    const f = findings(r);
    parts.push("", "Findings", ...f.findings.map((s) => `- ${s}`), "", "Caveats", ...f.caveats.map((s) => `- ${s}`));
  }
  return parts.join("\n") + "\n";
}

/** One line per run of a sweep: where native stands and which method does best. */
export function sweepSummary(results: BenchResult[]): string {
  if (!results.length) return "No runs.\n";
  const name = (r: BenchResult, id: string) => r.methods.find((m) => m.id === id)?.name ?? id;
  const rows = results.map((r) => {
    const c = r.config;
    const agg = aggregate(r.datasets, r.methods.map((m) => m.id)).filter((a) => a.datasets > 0).sort(byMeanRank);
    const native = agg.find((a) => a.id === NATIVE);
    const methods = agg.filter((a) => a.id !== NATIVE);
    const best = methods[0];
    const ratio = [...methods].sort((a, b) => a.scoreRatio - b.scoreRatio)[0];
    return [
      pct(1 - c.alpha, 0), `${c.window}`, c.scoreType,
      native ? `${num(native.meanRank, 1)}/${agg.length}` : "–", native ? `${native.wins}/${r.datasets.length}` : "–", native ? pct(native.meanCoverage) : "–",
      best ? `${name(r, best.id)} (${num(best.meanRank, 1)})` : "–",
      ratio ? `${name(r, ratio.id)} (${num(ratio.scoreRatio, 3)})` : "–",
      `${methods.filter((a) => a.scoreRatio < 1).length}/${methods.length}`,
    ];
  });
  const head = ["target", "window", "score", "native rank", "native wins", "native cover", "best method (rank)", "lowest ratio to native", "methods below 1"];
  const first = results[0];
  return [
    `Sweep of ${results.length} runs · ${first.model.label} · leads 1–${first.config.evalLeads} · ${first.datasets.length} datasets`,
    "",
    table(head, rows, 0),
    "",
    "native rank: mean rank of the model's own interval among all rows. lowest ratio: the method with the lowest geometric mean interval score relative to native (below 1 beats native on average). methods below 1: how many methods have a ratio below 1.",
    "",
    ...results.map((r) => textSummary(r, { findings: false })),
  ].join("\n");
}
