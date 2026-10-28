"use client";
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Loader2, Wand2 } from "lucide-react";
import PageHead from "@/components/PageHead";
import CandidateCard from "@/components/CandidateCard";
import { api, COUNTRIES, type MatchResult } from "@/lib/api";

// real test-set businesses, retyped with the kinds of noise the sources contain
const EXAMPLES = [
  { name: "Dream Mktg Pvt Ltd", address: "H.No. 243, Vill Naharpur, NW Delhi", country: "India" },
  { name: "ड्रीम मार्केटिंग प्राइवेट लिमिटेड", address: "हाउस नं 243, नाहरपुर, दिल्ली", country: "India" },
  { name: "Martin Life Sciences Company", address: "261 Edgewood Ave, Giddings TX", country: "US" },
  { name: "Lieux Sportive", address: "66 bis r. de Bouvines, Lille", country: "France" },
  { name: "Zxqw Holdings", address: "1 Nowhere Lane, Austin TX", country: "US" },
];

function MatchInner() {
  const params = useSearchParams();
  const [name, setName] = useState(params.get("name") ?? "");
  const [address, setAddress] = useState(params.get("address") ?? "");
  const [country, setCountry] = useState(params.get("country") ?? "India");
  const [res, setRes] = useState<MatchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [runId, setRunId] = useState(0);
  const seq = useRef(0);
  const out = useRef<HTMLElement>(null);

  async function run(n = name, a = address, c = country) {
    if (!n.trim()) return;
    const id = ++seq.current;
    setBusy(true);
    setErr("");
    try {
      const r = await api.match(n, a, c);
      if (id !== seq.current) return; // a newer query was sent meanwhile
      setRes(r);
      setRunId(id);
    } catch (e) {
      if (id === seq.current) setErr(e instanceof TypeError ? "Could not reach the matching service." : e instanceof Error ? e.message : "Request failed");
    } finally {
      if (id === seq.current) setBusy(false);
    }
    // on one-column layouts the results sit below the form
    if (id === seq.current && window.matchMedia("(max-width: 1023px)").matches) {
      requestAnimationFrame(() => out.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }

  useEffect(() => {
    if (params.get("name")) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main>
      <PageHead eyebrow="Live match" title="Find a business in every source"
        sub="Type a business as it appears in Source 1. The live matcher cleans it, retrieves candidates from 390k Source 2/3 records and scores each one." />
      <div className="relative mx-auto -mt-8 grid max-w-6xl gap-6 px-4 pb-20 sm:px-8 lg:grid-cols-[380px_1fr]">
        <form className="glass h-fit min-w-0 rounded-2xl p-5 lg:sticky lg:top-6" onSubmit={(e) => { e.preventDefault(); run(); }}>
          <label className="mb-1.5 block text-sm text-fg/75" htmlFor="name">Business name</label>
          <input id="name" className="field" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Sharma Traders Pvt Ltd" required />
          <label className="mb-1.5 mt-4 block text-sm text-fg/75" htmlFor="addr">Address</label>
          <textarea id="addr" className="field min-h-20" value={address} onChange={(e) => setAddress(e.target.value)} placeholder="Street, landmark, city, PIN" />
          <label className="mb-1.5 mt-4 block text-sm text-fg/75" htmlFor="country">Country</label>
          <select id="country" className="field" value={country} onChange={(e) => setCountry(e.target.value)}>
            {COUNTRIES.map((c) => <option key={c} className="bg-surface">{c}</option>)}
          </select>
          <button className="btn-solid mt-5 w-full" disabled={busy}>
            {busy ? <Loader2 className="animate-spin" size={16} /> : <Wand2 size={16} />} Resolve
          </button>
          <p className="mb-2 mt-6 text-xs uppercase tracking-wider text-fg/55">Try an example</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex.name} disabled={busy} className="rounded-full border border-fg/15 px-3 py-1.5 text-xs text-fg/80 transition hover:bg-fg/[0.06] disabled:opacity-50"
                onClick={() => { setName(ex.name); setAddress(ex.address); setCountry(ex.country); run(ex.name, ex.address, ex.country); }}>
                {ex.name}
              </button>
            ))}
          </div>
        </form>

        <section ref={out} aria-live="polite" aria-busy={busy} className="min-w-0 scroll-mt-4">
          {err && <div className="glass rounded-2xl p-5 text-accent">{err}</div>}
          {busy && !res && (
            <div className="glass grid min-h-64 place-items-center rounded-2xl p-8 text-fg/70">
              <span className="inline-flex items-center gap-2"><Loader2 className="animate-spin" size={18} /> Cleaning, retrieving and scoring…</span>
            </div>
          )}
          {!res && !err && !busy && (
            <div className="glass grid min-h-64 place-items-center rounded-2xl p-8 text-center text-fg/60">
              Results appear here: each candidate with its match probability and the features that drove it.
            </div>
          )}
          {res && (
            <div className={`space-y-4 transition-opacity ${busy ? "opacity-50" : ""}`}>
              <div className="glass rounded-2xl p-5">
                <p className="text-xs uppercase tracking-wider text-fg/55">After cleaning</p>
                <div className="mt-2 grid gap-3 break-words text-sm sm:grid-cols-2">
                  <div><span className="text-fg/60">Name </span>{res.query.name_clean || "–"}</div>
                  <div><span className="text-fg/60">Address </span>{res.query.address_clean || "–"}</div>
                  <div><span className="text-fg/60">Legal form </span>{res.query.legal || "–"}</div>
                  <div><span className="text-fg/60">State/region </span>{res.query.state || "–"}</div>
                </div>
                <p className="mt-3 text-xs text-fg/60">
                  {res.n_retrieved} retrieved / {res.n_scored} scored / {res.n_matched} accepted · {res.ms} ms
                </p>
                {res.replaced && (
                  <p className="mt-2 text-xs text-fg/60">
                    Closest reference business: <span className="font-mono">{res.replaced.id}</span> {res.replaced.name} (similarity {res.replaced.cos}).
                    The query stands in for it, so it is left out of the reverse comparison.
                  </p>
                )}
              </div>
              {res.n_matched === 0 && (
                <div className="glass rounded-2xl p-4 text-sm text-primary">
                  No candidate is confident enough, so this business is predicted to have no match in Sources 2 and 3.
                </div>
              )}
              {res.candidates.map((c) => <CandidateCard key={`${runId}-${c.id}`} c={c} />)}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default function MatchPage() {
  return <Suspense><MatchInner /></Suspense>;
}
