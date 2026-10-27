"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Check, Loader2, Search, Wand2 } from "lucide-react";
import PageHead from "@/components/PageHead";
import ScoreRing from "@/components/ScoreRing";
import { api, COUNTRIES, type Entity, type EntityDetail } from "@/lib/api";

export default function Explore() {
  const [q, setQ] = useState("");
  const [country, setCountry] = useState("");
  const [list, setList] = useState<Entity[]>([]);
  const [sel, setSel] = useState<EntityDetail | null>(null);
  const [busy, setBusy] = useState(false);

  async function search(query = q, c = country) {
    setBusy(true);
    try { setList(await api.search(query, c)); } finally { setBusy(false); }
  }
  useEffect(() => { search("", ""); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const matched = sel?.candidates.filter((c) => c.matched) ?? [];
  const others = sel?.candidates.filter((c) => !c.matched) ?? [];

  return (
    <main>
      <PageHead eyebrow="Explore" title="Browse the submitted predictions"
        sub="60,000 test businesses (20k per country) with the candidates blocking kept for them and the matches the full pipeline submitted." />
      <div className="mx-auto -mt-8 grid max-w-6xl gap-6 px-4 pb-20 sm:px-8 lg:grid-cols-[1fr_1.2fr]">
        <section className="glass min-w-0 rounded-2xl p-4 sm:p-5">
          <form className="flex flex-col gap-2 sm:flex-row" onSubmit={(e) => { e.preventDefault(); search(); }}>
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3.5 top-3.5 text-white/40" />
              <input className="field pl-10" placeholder="Search name or address" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <select className="field sm:w-36" value={country} onChange={(e) => { setCountry(e.target.value); search(q, e.target.value); }}>
              <option value="" className="bg-ink">All countries</option>
              {COUNTRIES.map((c) => <option key={c} className="bg-ink">{c}</option>)}
            </select>
          </form>
          <ul className="mt-4 max-h-[62vh] space-y-1.5 overflow-y-auto pr-1">
            {busy && <li className="flex justify-center p-6"><Loader2 className="animate-spin text-white/50" /></li>}
            {!busy && list.length === 0 && <li className="p-6 text-center text-sm text-white/50">No businesses found.</li>}
            {!busy && list.map((e) => (
              <li key={e.id}>
                <button onClick={async () => setSel(await api.entity(e.id))}
                  className={`w-full rounded-xl px-3 py-2.5 text-left transition hover:bg-white/10 ${sel?.entity.id === e.id ? "bg-white/10" : ""}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{e.name}</span>
                    <span className="shrink-0 rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-mist">{e.country}</span>
                  </div>
                  <div className="truncate text-xs text-white/50">{e.address}</div>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="min-w-0">
          {!sel && <div className="glass grid min-h-64 place-items-center rounded-2xl p-8 text-center text-white/50">Pick a business to see its matches.</div>}
          {sel && (
            <div className="space-y-4">
              <div className="glass rounded-2xl p-5">
                <div className="flex items-center gap-2 text-xs text-white/45"><span className="font-mono">{sel.entity.id}</span> · {sel.entity.country}</div>
                <h2 className="mt-1 text-2xl font-medium">{sel.entity.name}</h2>
                <p className="mt-1 text-sm text-white/60">{sel.entity.address}</p>
                <Link className="btn-ghost mt-4 !py-2"
                  href={`/match?name=${encodeURIComponent(sel.entity.name)}&address=${encodeURIComponent(sel.entity.address ?? "")}&country=${sel.entity.country}`}>
                  <Wand2 size={15} /> Re-match live
                </Link>
              </div>
              <h3 className="px-1 text-sm uppercase tracking-wider text-white/45">Submitted matches ({matched.length})</h3>
              {matched.length === 0 && <div className="glass rounded-2xl p-4 text-sm text-mist">Predicted singleton: no match in Sources 2/3.</div>}
              {matched.map((c) => <Row key={c.id} c={c} />)}
              {others.length > 0 && <h3 className="px-1 pt-2 text-sm uppercase tracking-wider text-white/45">Rejected candidates ({others.length})</h3>}
              {others.map((c) => <Row key={c.id} c={c} />)}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function Row({ c }: { c: EntityDetail["candidates"][number] }) {
  return (
    <div className={`glass flex items-center gap-4 rounded-2xl p-4 ${c.matched ? "ring-1 ring-coral/60" : "opacity-75"}`}>
      <ScoreRing p={c.p ?? 0} size={48} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-xs">
          <span className="rounded-full bg-white/10 px-2 py-0.5 text-mist">S{c.src}</span>
          <span className="font-mono text-white/40">{c.id}</span>
          {!!c.matched && <Check size={14} className="text-coral" />}
        </div>
        <div className="truncate font-medium">{c.name || <em className="text-white/40">no name</em>}</div>
        <div className="truncate text-sm text-white/55">{c.address || "no address"}</div>
      </div>
    </div>
  );
}
