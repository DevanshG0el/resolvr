"use client";
import { useEffect, useState } from "react";
import { Loader2, TrendingDown, TrendingUp, Trophy } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import PageHead from "@/components/PageHead";
import { api, type LogRow } from "@/lib/api";
import { AXIS, GRID, TIP } from "@/lib/chart";

const PHASE_COLOR: Record<string, string> = {
  "US/India matcher": "#5b4bd8", "France route": "#d63d61", "All countries": "#2f6bd8", Release: "#6b6790",
};
const STATUS: Record<LogRow["status"], string> = {
  kept: "bg-emerald-500/10 text-emerald-700", reverted: "bg-fg/[0.06] text-fg/70", probe: "bg-amber-500/10 text-amber-700",
};
const f5 = (x: number | null) => (x == null ? "–" : x.toFixed(5));

function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="glass rounded-2xl p-4 sm:p-6">
      <h2 className="text-xl font-medium">{title}</h2>
      {sub && <p className="mt-1 text-sm text-fg/65">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

function Metric({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="inline-flex items-baseline gap-2 rounded-lg bg-fg/[0.04] px-2.5 py-1.5">
      <dt className="text-fg/60">{k}</dt>
      <dd className="font-mono text-[13px]">{v}</dd>
    </div>
  );
}

export default function Progress() {
  const [rows, setRows] = useState<LogRow[] | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.log().then(setRows).catch((e) => setErr(e instanceof Error ? e.message : String(e))); }, []);

  if (!rows) {
    return (
      <main>
        <PageHead eyebrow="Progress" title="Experiment log" sub="Loading the log…" />
        <div className="flex justify-center p-10">{err ? <p className="text-accent">{err}</p> : <Loader2 className="animate-spin text-fg/60" />}</div>
      </main>
    );
  }

  // change in validation F0.5 against the best validated run kept so far
  let ref: number | null = null;
  const withDelta = rows.map((r) => {
    const delta = r.val_f05 != null && ref != null ? r.val_f05 - ref : null;
    if (r.val_f05 != null && r.status === "kept") ref = Math.max(ref ?? 0, r.val_f05);
    return { ...r, delta };
  });
  const validated = rows.filter((r) => r.val_f05 != null && r.status === "kept");
  const base = validated[0]?.val_f05 ?? 0;
  const best = Math.max(...validated.map((r) => r.val_f05 ?? 0));
  const errCut = base < 1 ? 1 - (1 - best) / (1 - base) : 0;
  const lb = rows.filter((r) => r.lb_f05 != null || r.lb_rank != null);
  const bestRank = lb.length ? Math.min(...lb.map((r) => r.lb_rank ?? Infinity)) : null;
  const bestLb = lb.length ? Math.max(...lb.map((r) => r.lb_f05 ?? 0)) : null;
  const chart = rows.filter((r) => r.status !== "reverted" && (r.val_f05 != null || r.lb_f05 != null))
    .map((r) => ({ id: r.id, val: r.val_f05, lb: r.lb_f05, rank: r.lb_rank }));

  return (
    <main>
      <PageHead eyebrow="Progress" title="Experiment log"
        sub="Every submission and validation experiment between the first pipeline and the final upload: what changed, what it did to held-out F0.5, and where it put us on the leaderboard." />
      <div className="relative mx-auto -mt-8 max-w-6xl space-y-6 px-4 pb-20 sm:px-8">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            ["Baseline validation F0.5", base.toFixed(5)],
            ["Best validation F0.5", best.toFixed(5)],
            ["Errors removed vs baseline", `${(errCut * 100).toFixed(0)}%`],
            [bestRank != null && Number.isFinite(bestRank) ? "Best leaderboard rank" : "Best leaderboard F0.5",
              bestRank != null && Number.isFinite(bestRank) ? `#${bestRank}` : bestLb ? bestLb.toFixed(5) : "–"],
          ].map(([k, v]) => (
            <div key={k} className="glass rounded-2xl p-5">
              <div className="font-display text-3xl font-semibold tabular-nums tracking-tight">{v}</div>
              <div className="mt-1 text-sm text-fg/65">{k}</div>
            </div>
          ))}
        </div>

        <Panel title="Score over time"
          sub={`Validation macro F0.5 on 441,849 held-out US/India businesses (kept runs)${lb.length ? ", and the public leaderboard score" : ""}. France has no labels, so its changes show only on the leaderboard.`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={chart} margin={{ right: 16, left: 4, bottom: 4 }}>
                <CartesianGrid stroke={GRID} />
                <XAxis dataKey="id" tick={{ ...AXIS, fontSize: 11 }} stroke={AXIS.stroke} interval="preserveStartEnd" />
                <YAxis domain={["dataMin - 0.0005", "dataMax + 0.0003"]} tickFormatter={(v: number) => v.toFixed(4)} tick={AXIS} stroke={AXIS.stroke} width={58} />
                <Tooltip {...TIP} formatter={(v: number, n: string) => [v.toFixed(5), n]} />
                <Line name="Validation F0.5" dataKey="val" stroke="#5b4bd8" strokeWidth={2.5} dot={{ r: 4, fill: "#5b4bd8" }} connectNulls isAnimationActive={false} />
                {lb.length > 0 && <Line name="Leaderboard F0.5" dataKey="lb" stroke="#e5486b" strokeWidth={2.5} dot={{ r: 4, fill: "#e5486b" }} connectNulls isAnimationActive={false} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-fg/70">
            <span className="inline-flex items-center gap-1.5"><i className="inline-block h-0.5 w-4 bg-primary" />Validation F0.5</span>
            {lb.length > 0 && <span className="inline-flex items-center gap-1.5"><i className="inline-block h-0.5 w-4 bg-accent" />Leaderboard F0.5</span>}
          </div>
        </Panel>

        <Panel title="Change log" sub="Δ is the change in validation F0.5 against the best kept run before it.">
          <ol className="relative space-y-3 border-l border-fg/10 pl-5 sm:pl-6">
            {[...withDelta].reverse().map((r) => (
              <li key={r.id} className="relative">
                <span className="absolute -left-[27px] top-5 h-3 w-3 rounded-full ring-4 ring-page sm:-left-[31px]" style={{ background: PHASE_COLOR[r.phase] ?? "#6b6790" }} />
                <div className="rounded-2xl border border-fg/10 bg-fg/[0.03] p-4">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="rounded-full bg-fg/[0.06] px-2 py-0.5 font-mono font-semibold text-fg">{r.id}</span>
                    <span className="text-fg/60">{r.date}</span>
                    <span className="rounded-full px-2 py-0.5" style={{ color: PHASE_COLOR[r.phase], background: `${PHASE_COLOR[r.phase] ?? "#6b6790"}14` }}>{r.phase}</span>
                    <span className={`rounded-full px-2 py-0.5 ${STATUS[r.status]}`}>{r.status}</span>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed text-fg/90">{r.change}</p>
                  {r.note && <p className="mt-1 text-xs text-fg/60">{r.note}</p>}
                  <dl className="mt-3 flex flex-wrap gap-2 text-xs">
                    <Metric k="Validation F0.5" v={f5(r.val_f05)} />
                    {r.delta != null && (
                      <Metric k="Δ" v={<span className={`inline-flex items-center gap-1 ${r.delta >= 0 ? "text-emerald-600" : "text-accent"}`}>
                        {r.delta >= 0 ? <TrendingUp size={13} /> : <TrendingDown size={13} />}{r.delta >= 0 ? "+" : ""}{r.delta.toFixed(5)}</span>} />
                    )}
                    <Metric k="Leaderboard" v={f5(r.lb_f05)} />
                    <Metric k="Rank" v={<span className="inline-flex items-center gap-1">
                      {r.lb_rank != null && r.lb_rank === bestRank && <Trophy size={13} className="text-amber-500" />}
                      {r.lb_rank != null ? `#${r.lb_rank}` : "–"}</span>} />
                  </dl>
                </div>
              </li>
            ))}
          </ol>
        </Panel>
      </div>
    </main>
  );
}
