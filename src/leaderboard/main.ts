import "../styles.css";
import "./leaderboard.css";
import { type BenchResult, type DatasetResult, RESULT_SCHEMA, type ResultRow } from "../bench/types";
import { type Aggregate, NATIVE, aggregate, byMeanRank, findings, toCsv } from "../bench/summary";

// Every committed result file is bundled at build time; adding a model is adding a file.
const files = import.meta.glob<BenchResult>("../../benchmarks/results/*.json", { eager: true, import: "default" });
const RUNS = Object.values(files)
  .filter((r) => r && r.schema === RESULT_SCHEMA && Array.isArray(r.datasets))
  // Real models first, the naive baseline last.
  .sort((a, b) => Number(a.model.id === "naive") - Number(b.model.id === "naive") || a.model.id.localeCompare(b.model.id));

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const state = {
  run: RUNS[0],
  group: "all",
  dataset: "",
  sort: { overall: { key: "pos", dir: 1 }, matrix: { key: "meanRank", dir: 1 }, detail: { key: "rank", dir: 1 } } as Record<string, { key: string; dir: number }>,
};

// ---- formatting ---------------------------------------------------------------------------

const fin = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);
const f = (v: number | null | undefined, d = 2) => (fin(v) ? v.toFixed(d) : "–");
const pct = (v: number | null | undefined, d = 1) => (fin(v) ? `${(100 * v).toFixed(d)}%` : "–");
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

function method(id: string) {
  return state.run.methods.find((m) => m.id === id) ?? { id, name: id, color: "#8E8E93", family: "", reference: "", guarantee: false, params: {} };
}

function nameCell(id: string): string {
  const m = method(id);
  const note = m.guarantee || id === NATIVE ? "" : " · No distribution-free coverage guarantee";
  return `<td class="name" title="${esc(m.reference + note)}">${esc(m.name)}</td>`;
}

const target = () => 1 - state.run.config.alpha;
const covClass = (c: number) => (fin(c) && c < target() - state.run.config.tolerance ? "off" : "");

function covCell(c: number, ci?: [number, number]): string {
  const t = target();
  const w = 120;
  // Coverage from 50% to 100% mapped onto the bar.
  const x = (v: number) => Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * w;
  const bar = fin(c)
    ? `<span class="lbbar" style="width:${w}px">${ci ? `<span class="ci" style="left:${x(ci[0])}px;width:${Math.max(1.5, x(ci[1]) - x(ci[0]))}px"></span>` : ""}<span class="mark ${covClass(c)}" style="left:${x(c) - 3}px"></span><span class="target" style="left:${x(t)}px"></span></span>`
    : "";
  return `<td class="cov ${covClass(c)}">${bar}${pct(c)}</td>`;
}

// ---- sorting ------------------------------------------------------------------------------

interface Col<T> {
  key: string;
  label: string;
  title?: string;
  value: (r: T) => number | string;
  cell: (r: T) => string;
  /** Natural direction: 1 = ascending is best first. */
  dir?: number;
}

function sortRows<T>(rows: T[], cols: Col<T>[], which: string): T[] {
  const s = state.sort[which];
  const col = cols.find((c) => c.key === s.key) ?? cols[0];
  return [...rows].sort((a, b) => {
    const va = col.value(a), vb = col.value(b);
    if (typeof va === "string" || typeof vb === "string") return String(va).localeCompare(String(vb)) * s.dir;
    const na = fin(va) ? va : Infinity * s.dir, nb = fin(vb) ? vb : Infinity * s.dir;
    // Equal values keep the table's natural order (its first column: position or rank).
    return (na - nb) * s.dir || (Number(cols[0].value(a)) - Number(cols[0].value(b))) || 0;
  });
}

function head<T>(cols: Col<T>[], which: string): string {
  const s = state.sort[which];
  return `<thead><tr>${cols
    .map((c) => {
      const on = c.key === s.key;
      const arrow = on ? (s.dir === 1 ? " ↑" : " ↓") : "";
      return `<th scope="col" tabindex="0" data-sort="${c.key}" data-dir="${c.dir ?? 1}" class="${on ? "sorted" : ""}" title="${esc(c.title ?? "Sort")}" aria-sort="${on ? (s.dir === 1 ? "ascending" : "descending") : "none"}">${c.label}${arrow}</th>`;
    })
    .join("")}</tr></thead>`;
}

function bindSort(table: HTMLTableElement, which: string, rerender: () => void) {
  table.querySelectorAll<HTMLTableCellElement>("th[data-sort]").forEach((th) => {
    const sort = () => {
      const key = th.dataset.sort!;
      const s = state.sort[which];
      if (s.key === key) s.dir = -s.dir;
      else state.sort[which] = { key, dir: Number(th.dataset.dir) || 1 };
      const focused = document.activeElement === th;
      rerender();
      if (focused) table.querySelector<HTMLElement>(`th[data-sort="${key}"]`)?.focus({ preventScroll: true });
    };
    th.addEventListener("click", sort);
    th.addEventListener("keydown", (e) => {
      if (e.target !== th || (e.key !== "Enter" && e.key !== " ")) return;
      e.preventDefault();
      sort();
    });
  });
}

// ---- views --------------------------------------------------------------------------------

function selectedDatasets(): DatasetResult[] {
  return state.run.datasets.filter((d) => state.group === "all" || d.group === state.group);
}

function renderOverall() {
  const ds = selectedDatasets();
  const N = ds.length;
  const aggs = aggregate(ds, state.run.methods.map((m) => m.id)).filter((a) => a.datasets > 0);
  // The overall position is always by mean rank, whatever column the table is sorted by.
  const pos = new Map([...aggs].sort(byMeanRank).map((a, i) => [a.id, i + 1]));
  const cols: Col<Aggregate>[] = [
    { key: "pos", label: "#", value: (a) => pos.get(a.id)!, cell: (a) => `<td class="pos">${pos.get(a.id)}</td>` },
    { key: "name", label: "Method", value: (a) => method(a.id).name, cell: (a) => nameCell(a.id) },
    { key: "meanRank", label: "Mean rank", title: "Mean of the per-dataset ranks (lower is better)", value: (a) => a.meanRank, cell: (a) => `<td>${f(a.meanRank, 1)}</td>` },
    { key: "meanCoverage", label: "Coverage", title: `Mean coverage; target ${pct(target(), 0)}`, dir: -1, value: (a) => a.meanCoverage, cell: (a) => covCell(a.meanCoverage) },
    { key: "scoreRatio", label: "Score vs native", title: "Geometric mean of interval score ÷ native's (lower is better)", value: (a) => a.scoreRatio, cell: (a) => `<td class="${a.id === NATIVE ? "muted" : a.scoreRatio < 1 ? "good" : ""}">${a.id === NATIVE ? "1" : `${f(a.scoreRatio, 3)}`}</td>` },
  ];
  const table = $<HTMLTableElement>("overall");
  table.innerHTML = head(cols, "overall") + `<tbody>${sortRows(aggs, cols, "overall").map((a) => `<tr class="${a.id === NATIVE ? "native" : ""}">${cols.map((c) => c.cell(a)).join("")}</tr>`).join("")}</tbody>`;
  bindSort(table, "overall", renderOverall);
  $("overall-sub").textContent = `${N} dataset${N === 1 ? "" : "s"}`;
}

/** Sequential shading for ranks: one hue, darker = better. */
function rankShade(rank: number, of: number): string {
  const t = of > 1 ? 1 - (rank - 1) / (of - 1) : 1;
  return `color-mix(in srgb, var(--accent) ${Math.round(6 + 64 * t ** 1.6)}%, var(--surface))`;
}

function renderMatrix() {
  const ds = selectedDatasets();
  const ids = state.run.methods.map((m) => m.id);
  const aggs = new Map(aggregate(ds, ids).map((a) => [a.id, a]));
  const rows = ids.filter((id) => (aggs.get(id)?.datasets ?? 0) > 0);
  const R = rows.length;
  const rowOf = (d: DatasetResult, id: string) => d.rows.find((r) => r.id === id);
  const cols: Col<string>[] = [
    { key: "name", label: "Method", value: (id) => method(id).name, cell: (id) => nameCell(id) },
    { key: "meanRank", label: "Mean", title: "Mean rank over these datasets", value: (id) => aggs.get(id)!.meanRank, cell: (id) => `<td class="mean">${f(aggs.get(id)!.meanRank, 1)}</td>` },
    ...ds.map(
      (d): Col<string> => ({
        key: `d:${d.id}`,
        label: `<button class="dhead" data-dataset="${d.id}" aria-label="Open ${esc(d.label)} results">${esc(d.label)}</button>`,
        title: `${d.label}: sort by rank; click the name to open its table`,
        value: (id) => rowOf(d, id)?.rank ?? Infinity,
        cell: (id) => {
          const r = rowOf(d, id);
          if (!r || !r.n) return `<td class="rk empty">–</td>`;
          const dark = r.rank <= Math.max(2, Math.round(R * 0.25));
          return `<td class="rk${dark ? " dark" : ""}${covClass(r.coverage) ? " under" : ""}" style="background:${rankShade(r.rank, R)}" data-tip="${esc(`${method(id).name} on ${d.label}`)}|${r.rank}${r.tied ? " (tied)" : ""}|${pct(r.coverage)}|${f(r.intervalScore)}|${f(r.meanWidth)}">${r.rank}${r.tied ? "=" : ""}</td>`;
        },
      }),
    ),
  ];
  const table = $<HTMLTableElement>("matrix");
  table.innerHTML = head(cols, "matrix") + `<tbody>${sortRows(rows, cols, "matrix").map((id) => `<tr class="${id === NATIVE ? "native" : ""}">${cols.map((c) => c.cell(id)).join("")}</tr>`).join("")}</tbody>`;
  bindSort(table, "matrix", renderMatrix);
  $("under-note").textContent = `coverage below ${pct(target() - state.run.config.tolerance, 0)} (target ${pct(target(), 0)})`;
  table.querySelectorAll<HTMLElement>(".dhead").forEach((el) =>
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      setDataset(el.dataset.dataset!);
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      $("detail-card").scrollIntoView({ behavior: reducedMotion ? "instant" : "smooth", block: "start" });
    }),
  );
}

function renderDetail() {
  const d = state.run.datasets.find((x) => x.id === state.dataset) ?? state.run.datasets[0];
  if (!d) return;
  $("detail-title").textContent = d.label;
  $("detail-meta").textContent = `${d.group} · ${d.points} points · ${d.pairs.toLocaleString("en")} evaluated pairs per row${d.source ? ` · ${d.source}` : ""}`;
  const cols: Col<ResultRow>[] = [
    { key: "rank", label: "#", value: (r) => r.rank, cell: (r) => `<td class="pos">${r.rank}${r.tied ? "=" : ""}</td>` },
    { key: "name", label: "Method", value: (r) => method(r.id).name, cell: (r) => nameCell(r.id) },
    { key: "coverage", label: "Coverage", title: `95% block-bootstrap interval shaded; target ${pct(target(), 0)}`, dir: -1, value: (r) => r.coverage, cell: (r) => covCell(r.coverage, r.ci?.coverage) },
    { key: "meanWidth", label: "Width", title: "Mean interval width (series rescaled to 10–90)", value: (r) => r.meanWidth, cell: (r) => `<td>${f(r.meanWidth)}</td>` },
    {
      key: "intervalScore",
      label: "Interval score",
      title: "Mean Winkler score with its 95% block-bootstrap interval (lower is better)",
      value: (r) => r.intervalScore,
      cell: (r) => `<td><b>${f(r.intervalScore)}</b>${r.ci ? `<span class="ci-text">${f(r.ci.intervalScore[0], 1)}–${f(r.ci.intervalScore[1], 1)}</span>` : ""}</td>`,
    },
  ];
  const table = $<HTMLTableElement>("detail");
  table.innerHTML = head(cols, "detail") + `<tbody>${sortRows(d.rows.filter((r) => r.n > 0), cols, "detail").map((r) => `<tr class="${r.id === NATIVE ? "native" : ""}${r.rank === 1 ? " first" : ""}">${cols.map((c) => c.cell(r)).join("")}</tr>`).join("")}</tbody>`;
  bindSort(table, "detail", renderDetail);
}

function renderFindings() {
  const { findings: fs, caveats } = findings(state.run);
  $("findings").innerHTML = fs.map((s) => `<li>${esc(s)}</li>`).join("");
  $("caveats").innerHTML = caveats.map((s) => `<li>${esc(s)}</li>`).join("");
}

function renderMeta() {
  const r = state.run;
  const c = r.config;
  const when = new Date(r.createdAt);
  $("meta").innerHTML = `<b>${esc(r.model.id === "naive" ? "Naive baseline" : r.model.label)}</b> · ${r.datasets.length} datasets · Target ${pct(1 - c.alpha, 0)}`;
  $("run-config").innerHTML =
    `${when.toISOString().slice(0, 10)}${r.commit ? ` · <code>${esc(r.commit)}</code>` : ""} · ${esc(c.scoreType.toUpperCase())} · ` +
    `Steps 1–${c.evalLeads}, window ${c.window}, warm-up ${c.warmup}, context ${c.maxContext}, up to ${c.length} points`;
}

function renderGroups() {
  const groups = ["all", ...new Set(state.run.datasets.map((d) => d.group))];
  if (!groups.includes(state.group)) state.group = "all";
  const seg = $("group");
  seg.innerHTML = groups
    .map((g) => {
      const n = g === "all" ? state.run.datasets.length : state.run.datasets.filter((d) => d.group === g).length;
      return `<button class="${g === state.group ? "on" : ""}" data-group="${esc(g)}" aria-pressed="${g === state.group}">${g === "all" ? "All" : esc(g)} <span class="muted">${n}</span></button>`;
    })
    .join("");
  seg.querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
    b.addEventListener("click", () => {
      state.group = b.dataset.group!;
      writeHash();
      renderGroups();
      renderOverall();
      renderMatrix();
    }),
  );
}

function renderDatasetSelect() {
  const sel = $<HTMLSelectElement>("dataset");
  const groups = [...new Set(state.run.datasets.map((d) => d.group))];
  sel.innerHTML = groups
    .map((g) => `<optgroup label="${esc(g)}">${state.run.datasets.filter((d) => d.group === g).map((d) => `<option value="${d.id}">${esc(d.label)}</option>`).join("")}</optgroup>`)
    .join("");
  if (!state.run.datasets.some((d) => d.id === state.dataset)) state.dataset = state.run.datasets[0]?.id ?? "";
  sel.value = state.dataset;
}

function setDataset(id: string) {
  state.dataset = id;
  $<HTMLSelectElement>("dataset").value = id;
  writeHash();
  renderDetail();
}

function renderAll() {
  renderMeta();
  renderFindings();
  renderGroups();
  renderDatasetSelect();
  renderOverall();
  renderMatrix();
  renderDetail();
}

// ---- URL state, CSV, tooltip ----------------------------------------------------------------

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const run = RUNS.find((r) => r.run === p.get("model"));
  if (run) state.run = run;
  if (p.get("group")) state.group = p.get("group")!;
  if (p.get("dataset")) state.dataset = p.get("dataset")!;
}

function writeHash() {
  const p = new URLSearchParams({ model: state.run.run, group: state.group, dataset: state.dataset });
  history.replaceState(null, "", `#${p}`);
}

function downloadCsv() {
  const blob = new Blob([toCsv(state.run)], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `timestral-arena-leaderboard-${state.run.run}.csv`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function bindTooltip() {
  const tip = $("tip");
  const table = $("matrix");
  table.addEventListener("mousemove", (e) => {
    const td = (e.target as HTMLElement).closest<HTMLElement>("td[data-tip]");
    if (!td) return tip.classList.add("hidden");
    const [title, rank, cov, is, w] = td.dataset.tip!.split("|");
    tip.innerHTML = `<b>${title}</b><span>Rank</span><span>${rank}</span><span>Coverage</span><span>${cov}</span><span>Interval score</span><span>${is}</span><span>Width</span><span>${w}</span>`;
    tip.classList.remove("hidden");
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - r.width - 8)}px`;
    tip.style.top = `${Math.min(e.clientY + 14, innerHeight - r.height - 8)}px`;
  });
  table.addEventListener("mouseleave", () => tip.classList.add("hidden"));
}

function init() {
  if (!RUNS.length) {
    document.querySelector("main")!.innerHTML = `<section class="card"><h3>No results yet</h3><p>Run <code>npm run benchmark</code> to create <code>benchmarks/results/naive.json</code>, then rebuild.</p></section>`;
    return;
  }
  readHash();
  const runSel = $<HTMLSelectElement>("run");
  runSel.innerHTML = RUNS.map((r) => `<option value="${esc(r.run)}">${esc(r.model.id === "naive" ? "Naive baseline" : r.model.id.replace(/^.*\//, ""))}</option>`).join("");
  runSel.value = state.run.run;
  runSel.addEventListener("change", () => {
    state.run = RUNS.find((r) => r.run === runSel.value) ?? RUNS[0];
    writeHash();
    renderAll();
  });
  $<HTMLSelectElement>("dataset").addEventListener("change", (e) => setDataset((e.target as HTMLSelectElement).value));
  $("btn-csv").addEventListener("click", downloadCsv);
  bindTooltip();
  renderAll();
}

init();
