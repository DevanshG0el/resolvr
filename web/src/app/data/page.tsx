"use client";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import PageHead from "@/components/PageHead";
import { api, pct, type DataStats } from "@/lib/api";
import { AXIS, BAR_LABEL, CAT_TICK, CURSOR, GRID, TIP } from "@/lib/chart";

const COUNTRY_COLOR: Record<string, string> = { India: "#e5486b", US: "#7b6cf0", France: "#2f6bd8" };
const SOURCE_NAME: Record<string, string> = { S1: "Source 1", S2: "Source 2", S3: "Source 3" };
const AXIS_LABEL = { fill: "rgba(21,18,58,.6)", fontSize: 12 };
const millions = (n: number) => `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`;

// a real labelled business from the training set and every record matched to it
const EXAMPLE = [
  { id: "S1-845929181", name: "Vijay Trading Private Limited", address: "J-3/299, Ground Floor Dda Flats Kalka Ji, New Delhi, South Delhi, Delhi", note: "reference record" },
  { id: "S2-524546173", name: "विजय ट्रेडिंग प्राइवेट लिमिटेड", address: "J-D/3/299, GROUND FLOOR DDA FLATS KALKA JI, NEW DELHI, SOUTH DELHI, Delhi", note: "name in Devanagari, address in capitals" },
  { id: "S2-127361516", name: "Vijay Toiarding Private Limited", address: "J-D/3/299, GROUND FLOOR DDA FLATS KALKA JI, NEW DELHI, Delhi", note: "typo in the name" },
  { id: "S3-100270922", name: "विजय ट्रेडिंग प्राइवेट लिमिटेड", address: "J-3/299, New Delhi, DL", note: "short address, state as a code" },
  { id: "S3-902383139", name: "-- Vijay Trading  Private Limited", address: "J-3/299, New Delhi, दिल्ली", note: "stray punctuation, state in Hindi" },
];

function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="glass rounded-2xl p-4 sm:p-6">
      <h2 className="text-xl font-medium">{title}</h2>
      {sub && <p className="mt-1 text-sm text-fg/65">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

function Toggle<T extends string>({ value, options, onChange }: { value: T; options: T[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-xl bg-fg/[0.05] p-1 text-sm" role="group">
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} aria-pressed={value === o}
          className={`rounded-lg px-3 py-1 capitalize transition ${value === o ? "bg-surface text-fg shadow-sm" : "text-fg/65 hover:text-fg"}`}>
          {o}
        </button>
      ))}
    </div>
  );
}

export default function Dataset() {
  const [d, setD] = useState<DataStats | null>(null);
  const [err, setErr] = useState("");
  const [split, setSplit] = useState<"train" | "test">("train");
  useEffect(() => { api.data().then(setD).catch((e) => setErr(e instanceof Error ? e.message : String(e))); }, []);

  if (!d) {
    return (
      <main>
        <PageHead eyebrow="Dataset" title="The data behind the matcher" sub="Loading dataset statistics…" />
        <div className="flex justify-center p-10">{err ? <p className="text-accent">{err}</p> : <Loader2 className="animate-spin text-fg/60" />}</div>
      </main>
    );
  }

  const total = d.sources.reduce((s, r) => s + r.records, 0);
  const labelled = d.truth.reduce((s, r) => s + r.s1_entities, 0);
  const pairs = d.splits.reduce((s, r) => s + r.true_pairs, 0);
  const singles = d.truth.reduce((s, r) => s + r.singleton_share * r.s1_entities, 0) / labelled;

  // records per source, one bar per country present in the chosen split
  const countries = split === "train" ? ["India", "US"] : ["India", "US", "France"];
  const bySource = ["S1", "S2", "S3"].map((s) => ({
    name: SOURCE_NAME[s],
    ...Object.fromEntries(countries.map((c) => [c, d.sources.find((r) => r.split === split && r.source === s && r.country === c)?.records ?? 0])),
  }));

  // share of S1 businesses with k matched records
  const card = ["0", "1", "2", "3", "4", "5", "6+"].map((k) => {
    const row: Record<string, string | number> = { k };
    for (const c of ["India", "US"]) {
      const all = d.cardinality.filter((r) => r.country === c).reduce((s, r) => s + r.len, 0);
      row[c] = (d.cardinality.find((r) => r.country === c && r.k === k)?.len ?? 0) / all;
    }
    return row;
  });

  // noise in Sources 2 and 3 of the test set, record-weighted per country
  const NOISE: [keyof DataStats["sources"][number], string][] = [
    ["address_missing", "Address missing"], ["name_non_latin", "Name in Indic script"],
    ["address_non_latin", "Address in Indic script"], ["landmark", "Landmark reference"],
  ];
  const noise = NOISE.map(([m, label]) => ({
    label,
    ...Object.fromEntries(["India", "US", "France"].map((c) => {
      const rows = d.sources.filter((r) => r.split === "test" && r.source !== "S1" && r.country === c);
      const w = rows.reduce((s, r) => s + r.records, 0);
      return [c, rows.reduce((s, r) => s + (r[m] as number) * r.records, 0) / w];
    })),
  }));

  const cleaning = d.cleaning.flatMap((r) => [
    { label: `Name equal · ${r.a_country}`, Raw: r.name_exact_raw, Cleaned: r.name_exact_clean },
    { label: `House no. equal · ${r.a_country}`, Raw: r.number_eq_raw, Cleaned: r.number_eq_clean },
  ]);

  const SPLIT_NAME: Record<string, string> = { A: "A", B: "B", V: "V", ghost: "Ghost" };
  const splits = d.splits.map((s) => ({ name: SPLIT_NAME[s.split] ?? s.split, businesses: s.len, pairs: s.true_pairs }));

  const blocker = d.blocking.filter((r) => r.rule.startsWith("blocker") && !r.rule.includes("top-2")).map((r) => ({ x: r.cands_per_s1, y: r.recall }));
  const topk = d.blocking.filter((r) => r.rule.startsWith("fwd") && r.rule.endsWith("rev<=1")).map((r) => ({ x: r.cands_per_s1, y: r.recall }));
  const lc = d.learning_curve.map((r) => ({ rows: r.rows / 1e6, f05: r.f05 }));

  const profile = [...d.sources].sort((a, b) => a.split.localeCompare(b.split) * -1 || a.source.localeCompare(b.source) || a.country.localeCompare(b.country));

  return (
    <main>
      <PageHead eyebrow="Dataset" title="The data behind the matcher"
        sub="Three sources of business records, two labelled countries and one unseen one. How big each part is, how noisy it is, and how cleaning and splitting prepare it for the models." />
      <div className="relative mx-auto -mt-8 max-w-6xl space-y-6 px-4 pb-20 sm:px-8">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            [millions(total), "Records across train and test"],
            [millions(labelled), "Labelled Source 1 businesses"],
            [millions(pairs), "True matching pairs"],
            [pct(singles), "Businesses with no match"],
          ].map(([v, k]) => (
            <div key={k} className="glass rounded-2xl p-5">
              <div className="font-display text-3xl font-semibold tabular-nums tracking-tight">{v}</div>
              <div className="mt-1 text-sm text-fg/65">{k}</div>
            </div>
          ))}
        </div>

        <Panel title="One business, five records"
          sub="A real labelled business from the training set. Source 1 is the clean reference; the model has to find every Source 2 and Source 3 record that describes it.">
          <div className="space-y-2">
            {EXAMPLE.map((r) => (
              <div key={r.id} className={`grid gap-1 rounded-xl border p-3 sm:grid-cols-[120px_1fr_200px] sm:items-center sm:gap-4 ${r.id.startsWith("S1") ? "border-primary/30 bg-primary/[0.05]" : "border-fg/10 bg-fg/[0.025]"}`}>
                <span className="font-mono text-xs text-fg/60">{r.id}</span>
                <div className="min-w-0">
                  <div className="font-medium">{r.name}</div>
                  <div className="text-sm text-fg/70">{r.address}</div>
                </div>
                <span className="text-xs text-fg/60">{r.note}</span>
              </div>
            ))}
          </div>
          <div className="mt-5 grid gap-3 text-sm text-fg/75 sm:grid-cols-3">
            <div><div className="font-medium text-fg">Four columns</div>entity_id (prefix gives the source), business_name, business_address, country.</div>
            <div><div className="font-medium text-fg">Ground truth</div>Each Source 1 id lists its matched Source 2/3 ids; an empty list means no match.</div>
            <div><div className="font-medium text-fg">Unseen country</div>Training covers India and the US. The test set adds France, with no labels.</div>
          </div>
        </Panel>

        <Panel title="Records by source and country" sub="Sources 2 and 3 are the noisy copies, so they are 2–3 times larger than the Source 1 reference.">
          <div className="mb-4"><Toggle value={split} options={["train", "test"]} onChange={setSplit} /></div>
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={bySource} margin={{ left: 4, right: 8 }}>
                <CartesianGrid vertical={false} stroke={GRID} />
                <XAxis dataKey="name" tick={CAT_TICK} stroke={AXIS.stroke} />
                <YAxis tickFormatter={(v: number) => millions(v)} tick={AXIS} stroke={AXIS.stroke} width={52} />
                <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => v.toLocaleString()} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                {countries.map((c) => <Bar key={c} dataKey={c} fill={COUNTRY_COLOR[c]} radius={[6, 6, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Matches per business" sub="How many Source 2/3 records belong to each labelled business. Most have 2 to 5; about 1 in 18 has none.">
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={card} margin={{ left: 4, right: 8, bottom: 12 }}>
                  <CartesianGrid vertical={false} stroke={GRID} />
                  <XAxis dataKey="k" tick={CAT_TICK} stroke={AXIS.stroke} label={{ value: "matched records", position: "insideBottom", offset: -8, ...AXIS_LABEL }} />
                  <YAxis ticks={[0, 0.05, 0.1, 0.15, 0.2, 0.25]} tickFormatter={(v: number) => pct(v, 0)} tick={AXIS} stroke={AXIS.stroke} width={44} />
                  <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => pct(v)} labelFormatter={(k) => `${k} matched records`} />
                  <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  <Bar dataKey="India" fill={COUNTRY_COLOR.India} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="US" fill={COUNTRY_COLOR.US} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Noise in Sources 2 and 3" sub="Share of test records with each problem. Indian records mix scripts; France looks like the US.">
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={noise} layout="vertical" margin={{ left: 4, right: 16 }}>
                  <CartesianGrid horizontal={false} stroke={GRID} />
                  <XAxis type="number" ticks={[0, 0.05, 0.1, 0.15, 0.2, 0.25]} domain={[0, 0.25]} tickFormatter={(v: number) => pct(v, 0)} tick={AXIS} stroke={AXIS.stroke} />
                  <YAxis type="category" dataKey="label" width={118} interval={0} tick={{ ...CAT_TICK, fontSize: 11 }} stroke={AXIS.stroke} />
                  <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => pct(v)} />
                  <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  {["India", "US", "France"].map((c) => <Bar key={c} dataKey={c} fill={COUNTRY_COLOR[c]} radius={[0, 4, 4, 0]} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="What cleaning buys" sub="Share of true matching pairs whose names (or house numbers) agree exactly, before and after normalisation.">
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={cleaning} layout="vertical" margin={{ left: 4, right: 40 }}>
                  <CartesianGrid horizontal={false} stroke={GRID} />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => pct(v, 0)} tick={AXIS} stroke={AXIS.stroke} />
                  <YAxis type="category" dataKey="label" width={118} interval={0} tick={{ ...CAT_TICK, fontSize: 11 }} stroke={AXIS.stroke} />
                  <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => pct(v)} />
                  <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  <Bar dataKey="Raw" fill="#b3aecb" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="Cleaned" fill="#5b4bd8" radius={[0, 4, 4, 0]} label={{ position: "right", fill: BAR_LABEL, fontSize: 10, formatter: (v: number) => pct(v, 0) }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Leak-free splits" sub="Training businesses split by a hash of their id, so all records of one business stay in one split. A fine-tunes the retriever, B trains the matchers, V validates, Ghost keeps orphan records.">
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={splits} margin={{ left: 4, right: 8 }}>
                  <CartesianGrid vertical={false} stroke={GRID} />
                  <XAxis dataKey="name" tick={{ ...CAT_TICK, fontSize: 11 }} stroke={AXIS.stroke} interval={0} />
                  <YAxis tickFormatter={(v: number) => millions(v)} tick={AXIS} stroke={AXIS.stroke} width={52} />
                  <Tooltip {...TIP} cursor={CURSOR} formatter={(v: number) => v.toLocaleString()} />
                  <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  <Bar name="Businesses" dataKey="businesses" fill="#5b4bd8" radius={[4, 4, 0, 0]} />
                  <Bar name="True pairs" dataKey="pairs" fill="#e5486b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Blocking: candidates vs recall" sub="Scoring every pair is impossible, so each business keeps a few candidates. The learned blocker reaches higher recall with fewer.">
            <div className="h-80">
              <ResponsiveContainer>
                <LineChart margin={{ left: 4, right: 12, bottom: 12 }}>
                  <CartesianGrid stroke={GRID} />
                  <XAxis type="number" dataKey="x" domain={[3, 22]} allowDataOverflow tick={AXIS} stroke={AXIS.stroke}
                    label={{ value: "candidates kept per business", position: "insideBottom", offset: -8, ...AXIS_LABEL }} />
                  <YAxis type="number" domain={[0.89, 1]} ticks={[0.9, 0.92, 0.94, 0.96, 0.98, 1]} tickFormatter={(v: number) => pct(v, 0)} tick={AXIS} stroke={AXIS.stroke} width={44} />
                  <Tooltip {...TIP} formatter={(v: number) => pct(v, 2)} labelFormatter={(x) => `${Number(x).toFixed(1)} candidates`} />
                  <Legend verticalAlign="top" iconType="plainline" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                  <Line name="Learned blocker" data={blocker} dataKey="y" stroke="#5b4bd8" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
                  <Line name="Top-k by retrieval rank" data={topk} dataKey="y" stroke="#b3aecb" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="More data, better matcher" sub="Macro F0.5 on the validation split when the matcher trains on a quarter, half or all of split B.">
            <div className="h-80">
              <ResponsiveContainer>
                <LineChart data={lc} margin={{ left: 4, right: 24, top: 16, bottom: 12 }}>
                  <CartesianGrid stroke={GRID} />
                  <XAxis type="number" dataKey="rows" domain={[0.5, 3.2]} tickFormatter={(v: number) => `${v.toFixed(1)}M`} tick={AXIS} stroke={AXIS.stroke}
                    label={{ value: "training pairs", position: "insideBottom", offset: -8, ...AXIS_LABEL }} />
                  <YAxis domain={[0.9874, 0.9888]} ticks={[0.9875, 0.9878, 0.9881, 0.9884, 0.9887]} tickFormatter={(v: number) => v.toFixed(4)} tick={AXIS} stroke={AXIS.stroke} width={58} />
                  <Tooltip {...TIP} formatter={(v: number) => v.toFixed(5)} labelFormatter={(x) => `${Number(x).toFixed(2)}M training pairs`} />
                  <Line name="Validation F0.5" dataKey="f05" stroke="#e5486b" strokeWidth={2.5} dot={{ r: 4, fill: "#e5486b" }} isAnimationActive={false}
                    label={{ position: "top", fill: BAR_LABEL, fontSize: 11, formatter: (v: number) => v.toFixed(4) }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <Panel title="Source profile" sub="Every source, split and country: size, average field length and how often fields are missing or in Indic script.">
          <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-fg/10 text-left text-xs text-fg/60">
                  {["Split", "Source", "Country", "Records", "Name length", "Address length", "Address missing", "Name non-Latin"].map((h) => (
                    <th key={h} className="py-2 pr-4 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profile.map((r) => (
                  <tr key={`${r.split}-${r.source}-${r.country}`} className="border-b border-fg/5">
                    <td className="py-2 pr-4 capitalize">{r.split}</td>
                    <td className="py-2 pr-4">{SOURCE_NAME[r.source]}</td>
                    <td className="py-2 pr-4"><span className="inline-flex items-center gap-1.5"><i className="inline-block h-2 w-2 rounded-full" style={{ background: COUNTRY_COLOR[r.country] }} />{r.country}</span></td>
                    <td className="py-2 pr-4">{r.records.toLocaleString()}</td>
                    <td className="py-2 pr-4">{r.name_len.toFixed(1)}</td>
                    <td className="py-2 pr-4">{r.address_len.toFixed(1)}</td>
                    <td className="py-2 pr-4">{pct(r.address_missing)}</td>
                    <td className="py-2 pr-4">{pct(r.name_non_latin)}</td>
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
