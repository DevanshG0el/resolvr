import Link from "next/link";
import { ArrowRight, Boxes, Brain, Filter, ScanSearch, Sparkles } from "lucide-react";
import Hero3D from "@/components/Hero3D";

const STATS = [
  { k: "0.989", v: "macro F0.5 on held-out validation" },
  { k: "99.8%", v: "precision of accepted matches" },
  { k: "11", v: "models compared on the same split" },
  { k: "24M", v: "records across 3 sources" },
];

const STEPS = [
  { icon: Filter, t: "Clean", d: "Transliterate Devanagari to Latin, expand abbreviations (Pvt to Private, Rd to Road), pull out legal form, state, PIN and house numbers." },
  { icon: ScanSearch, t: "Block", d: "A fine-tuned multilingual-e5 retriever proposes ~6 candidates per business out of millions, keeping 99.7% of true matches." },
  { icon: Boxes, t: "Score", d: "81 similarity features feed a two-stage XGBoost matcher, with e5, BGE and Qwen3 cross-encoders as extra evidence." },
  { icon: Brain, t: "Decide", d: "Per business, keep the candidate set that maximises expected F0.5, so uncertain merges are left out." },
];

export default function Home() {
  return (
    <main>
      <section className="sky relative isolate overflow-hidden">
        <div className="relative z-10 mx-auto max-w-4xl px-4 pt-36 text-center sm:pt-44">
          <p className="glass mx-auto mb-6 inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs text-mist">
            <Sparkles size={14} /> Business entity resolution across US, India and France
          </p>
          <h1 className="glow-text text-5xl font-medium leading-[1.05] tracking-tight sm:text-7xl">
            One business,<br />every source.
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-base text-white/75 sm:text-lg">
            Resolvr finds which noisy records from three data sources describe the same real-world business,
            even with typos, abbreviations and names written in another script.
          </p>
          <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link href="/match" className="btn-solid w-full sm:w-auto">Try live matching</Link>
            <Link href="/models" className="btn-ghost w-full sm:w-auto">See the models <ArrowRight size={16} /></Link>
          </div>
        </div>
        <Hero3D className="relative -mt-2 block h-[300px] w-full sm:-mt-16 sm:aspect-[12/5] sm:h-auto" />
      </section>

      <section className="mx-auto grid max-w-6xl grid-cols-2 gap-3 px-4 py-14 sm:px-8 lg:grid-cols-4">
        {STATS.map((s) => (
          <div key={s.v} className="glass rounded-2xl p-5">
            <div className="font-display text-3xl font-semibold tabular-nums tracking-tight sm:text-4xl">{s.k}</div>
            <div className="mt-1 text-sm text-white/60">{s.v}</div>
          </div>
        ))}
      </section>

      <section className="mx-auto max-w-6xl px-4 pb-20 sm:px-8">
        <h2 className="text-3xl font-medium tracking-tight sm:text-4xl">How it works</h2>
        <p className="mt-2 max-w-2xl text-white/60">Four stages, each one cheap enough to run on millions of records.</p>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <div key={s.t} className="glass relative rounded-2xl p-6">
              <span className="absolute right-5 top-5 text-sm text-white/30">0{i + 1}</span>
              <s.icon className="text-coral" size={26} />
              <h3 className="mt-4 text-xl font-medium">{s.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-white/65">{s.d}</p>
            </div>
          ))}
        </div>
        <div className="mt-10 flex justify-center">
          <Link href="/pipeline" className="btn-ghost">Read the full pipeline <ArrowRight size={16} /></Link>
        </div>
      </section>
    </main>
  );
}
