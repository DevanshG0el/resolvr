"use client";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import PageHead from "@/components/PageHead";
import { api, pct, type Overview } from "@/lib/api";
import { AXIS, BAR_LABEL, CAT_TICK, CURSOR, GRID, TIP } from "@/lib/chart";

const FAMILY_COLOR: Record<string, string> = {
  "Rule-based": "#8a86a3",
  "Transformer (bi-encoder)": "#a99ff8",
  "Transformer (cross-encoder)": "#7b6cf0",
  Linear: "#b3aecb",
  "Bagged trees": "#5b8def",
  "Neural network": "#ec6fb0",
  "Gradient boosting": "#2f6bf0",
  "Hybrid (ours)": "#e5486b",
};
const short = (m: string) => m.replace(" (RapidFuzz)", "").replace(" (fine-tuned e5)", " e5").replace(" (fine-tuned)", "").replace(" (3-layer, PyTorch)", "")
  .replace("Ours: 2-stage XGBoost + cross-encoder", "Ours (2-stage)").replace("Ours + expected-F0.5 decision", "Ours + F0.5 rule");

function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="glass rounded-2xl p-4 sm:p-6">
      <h2 className="text-xl font-medium">{title}</h2>
      {sub && <p className="mt-1 text-sm text-fg/65">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

export default function Models() {
  const [d, setD] = useState<Overview | null>(null);
  const [err, setErr] = useState("");
  const [narrow, setNarrow] = useState(false);
  useEffect(() => { api.overview().then(setD).catch((e) => setErr(e instanceof Error ? e.message : String(e))); }, []);
  // phones: narrower label column so the bars keep most of the width
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 639px)");
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  if (!d) {
    return (
      <main>
        <PageHead eyebrow="Models" title="State-of-the-art comparison" sub="Loading results…" />
        <div className="flex justify-center p-10">{err ? <p className="text-accent">{err}</p> : <Loader2 className="animate-spin text-fg/60" />}</div>
      </main>
    );
  }

  const models = [...d.models].sort((a, b) => b.entity_f05 - a.entity_f05);
  const bars = [...models].reverse().map((m) => ({ name: short(m.model), f05: +m.entity_f05.toFixed(4), family: m.family }));
  const curveModels = ["Fuzzy string rule (RapidFuzz)", "Bi-encoder cosine (fine-tuned e5)", "BGE-reranker-v2-m3 (fine-tuned)", "Logistic Regression", "XGBoost", "Ours: 2-stage XGBoost + cross-encoder"];
  const famOf = Object.fromEntries(d.models.map((m) => [m.model, m.family]));
  const countries = ["India", "US"];
  const perCountry = models.filter((m) => m.model !== "Ours + expected-F0.5 decision").map((m) => {
    const row: Record<string, string | number> = { name: short(m.model) };
    countries.forEach((c) => { row[c] = d.per_country.find((p) => p.model === m.model && p.country === c)?.f05 ?? 0; });
    return row;
  });
  const hard = [...d.hard_pairs].filter((h) => h.model !== "Ours + expected-F0.5 decision").sort((a, b) => a.pr_auc - b.pr_auc)
    .map((h) => ({ name: short(h.model), pr: +h.pr_auc.toFixed(3), fam: famOf[h.model] ?? "Transformer (cross-encoder)" }));
  const best = models[0];

  return (
    <main>
      <PageHead eyebrow="Models" title="State-of-the-art comparison"
        sub={`11 approaches score the same ${(models[0].tp + models[0].fp + models[0].fn + models[0].tn).toLocaleString()} candidate pairs of 221k held-out validation businesses. Each model's threshold is tuned on a separate half of the validation set.`} />
      <div className="relative mx-auto -mt-8 max-w-6xl space-y-6 px-4 pb-20 sm:px-8">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            ["Best macro F0.5", best.entity_f05.toFixed(4)],
            ["Precision", pct(best.entity_precision, 2)],
            ["Recall", pct(best.entity_recall, 2)],
            ["Blocking ceiling", d.ceiling.f05.toFixed(4)],
          ].map(([k, v]) => (
            <div key={k} className="glass rounded-2xl p-5">
              <div className="font-display text-3xl font-semibold tabular-nums tracking-tight">{v}</div>
              <div className="mt-1 text-sm text-fg/65">{k}</div>
            </div>
          ))}
        </div>

        <Panel title="Macro F0.5 by model" sub="The competition metric: F0.5 per Source 1 business (singletons included), averaged.">
          <div className="h-[440px]">
            <ResponsiveContainer>
              <BarChart data={bars} layout="vertical" margin={{ left: 4, right: 48 }}>
                <CartesianGrid horizontal={false} stroke={GRID} />
                <XAxis type="number" domain={[0.7, 1]} tick={AXIS} stroke={AXIS.stroke} />
                <YAxis type="category" dataKey="name" width={narrow ? 112 : 150} interval={0} tick={{ ...CAT_TICK, fontSize: narrow ? 10 : 12 }} stroke={AXIS.stroke} />
                <Tooltip {...TIP} cursor={CURSOR} />
                <Bar dataKey="f05" radius={[0, 6, 6, 0]} label={{ position: "right", fill: BAR_LABEL, fontSize: 11, formatter: (v: number) => v.toFixed(4) }}>
                  {bars.map((b) => <Cell key={b.name} fill={FAMILY_COLOR[b.family]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg/70">
            {Object.entries(FAMILY_COLOR).map(([f, c]) => (
              <span key={f} className="inline-flex items-center gap-1.5"><i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: c }} />{f}</span>
            ))}
          </div>
        </Panel>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Precision–recall curves" sub="Pair level, on V-test.">
            <div className="h-96 sm:h-80">
              <ResponsiveContainer>
                <LineChart margin={{ right: 10, bottom: 12 }}>
                  <CartesianGrid stroke={GRID} />
                  <XAxis type="number" dataKey="x" domain={[0, 1]} tick={AXIS} stroke={AXIS.stroke} label={{ value: "Recall", position: "insideBottom", offset: -8, fill: "rgba(21,18,58,.6)", fontSize: 12 }} />
                  <YAxis type="number" domain={[0.4, 1]} tick={AXIS} stroke={AXIS.stroke} width={44} label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 8, fill: "rgba(21,18,58,.6)", fontSize: 12 }} />
                  <Tooltip {...TIP} />
                  <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, paddingTop: 14 }} />
                  {curveModels.filter((m) => d.curves[m]).map((m) => (
                    <Line key={m} name={short(m)} data={d.curves[m].rc.map((x, i) => ({ x, y: d.curves[m].pr[i] }))} dataKey="y"
                      dot={false} activeDot={false} strokeWidth={2} stroke={FAMILY_COLOR[famOf[m]]} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Hard pairs" sub={`PR-AUC on the ${d.hard_pairs[0]?.n_pairs.toLocaleString()} uncertain pairs where the Qwen3-Reranker-4B was run.`}>
            <div className="h-96 sm:h-80">
              <ResponsiveContainer>
                <BarChart data={hard} layout="vertical" margin={{ left: 4, right: 40 }}>
                  <CartesianGrid horizontal={false} stroke={GRID} />
                  <XAxis type="number" domain={[0, 1]} tick={AXIS} stroke={AXIS.stroke} />
                  <YAxis type="category" dataKey="name" width={narrow ? 112 : 150} interval={0} tick={{ ...CAT_TICK, fontSize: narrow ? 10 : 11 }} stroke={AXIS.stroke} />
                  <Tooltip {...TIP} cursor={CURSOR} />
                  <Bar dataKey="pr" radius={[0, 6, 6, 0]} label={{ position: "right", fill: BAR_LABEL, fontSize: 10, formatter: (v: number) => v.toFixed(3) }}>
                    {hard.map((h) => <Cell key={h.name} fill={h.name.startsWith("Qwen") ? "#ec6fb0" : FAMILY_COLOR[h.fam]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="F0.5 per country" sub="France has no labels, so it is evaluated only on the leaderboard.">
            <div className="h-96 sm:h-80">
              <ResponsiveContainer>
                <BarChart data={perCountry.slice(0, 7)} layout="vertical" margin={{ left: 4, right: 20 }} barGap={2}>
                  <CartesianGrid horizontal={false} stroke={GRID} />
                  <XAxis type="number" domain={[0.9, 1]} allowDataOverflow tick={AXIS} stroke={AXIS.stroke} />
                  <YAxis type="category" dataKey="name" width={narrow ? 112 : 150} interval={0} tick={{ ...CAT_TICK, fontSize: narrow ? 10 : 11 }} stroke={AXIS.stroke} />
                  <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => v.toFixed(4)} />
                  <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  <Bar dataKey="India" fill="#e5486b" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="US" fill="#7b6cf0" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="What the matcher looks at" sub="XGBoost gain of the top hand-crafted features (single-stage matcher, as deployed in Live match).">
            <div className="h-96 sm:h-80">
              <ResponsiveContainer>
                <BarChart data={[...d.importance_live].slice(0, narrow ? 8 : 12)} layout="vertical" margin={{ left: 4, right: 20 }}>
                  <CartesianGrid horizontal={false} stroke={GRID} />
                  <XAxis type="number" tick={AXIS} stroke={AXIS.stroke} />
                  <YAxis type="category" dataKey="label" width={narrow ? 112 : 170} interval={0} tick={{ ...CAT_TICK, fontSize: narrow ? 10 : 11 }} stroke={AXIS.stroke} />
                  <Tooltip {...TIP} cursor={CURSOR} />
                  <Bar dataKey="gain" fill="#5b8def" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <Panel title="Full results" sub="Pair metrics at each model's tuned threshold; entity metrics are the competition's.">
          <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-fg/60">
                <tr>{["Model", "Family", "ROC-AUC", "PR-AUC", "Pair F1", "Macro F0.5", "Precision", "Recall", "Train (s)"].map((h) => <th key={h} className="pb-3 pr-3 font-normal">{h}</th>)}</tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.model} className={`border-t border-fg/10 ${m.family === "Hybrid (ours)" ? "text-accent" : ""}`}>
                    <td className="py-2.5 pr-3 font-medium">{m.model}</td>
                    <td className="pr-3 text-fg/70">{m.family}</td>
                    <td className="pr-3 font-mono">{m.roc_auc.toFixed(4)}</td>
                    <td className="pr-3 font-mono">{m.pr_auc.toFixed(4)}</td>
                    <td className="pr-3 font-mono">{m.pair_f1.toFixed(4)}</td>
                    <td className="pr-3 font-mono font-semibold">{m.entity_f05.toFixed(4)}</td>
                    <td className="pr-3 font-mono">{m.entity_precision.toFixed(4)}</td>
                    <td className="pr-3 font-mono">{m.entity_recall.toFixed(4)}</td>
                    <td className="font-mono">{m.train_s ? m.train_s.toFixed(0) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </main>
  );
}
