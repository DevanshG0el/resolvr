import { ArrowRight } from "lucide-react";
import PageHead from "@/components/PageHead";

const CLEAN = [
  { raw: "இண்டோ ஃபவுண்டேஷன் ஸ்டோர்ஸ்", clean: "indo foundation stors", note: "Tamil to Latin (mined dictionary + anyascii)" },
  { raw: "विजय इंडस्ट्रीज प्राइवेट लिमिटेड", clean: "vijay industries private limited · legal: private limited", note: "Devanagari to Latin, legal form extracted" },
  { raw: "ଷ୍ଟାର୍ ପ୍ରୋଜେକ୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", clean: "star projects private limited", note: "Odia to Latin" },
  { raw: "HARDMAN + WILLIAMS CORPORATION", clean: "hardman williams corp · legal: corp", note: "case, punctuation, suffix normalised" },
  { raw: "34 R Sainte-monique, Bordeaux, Nouvelle-Aquitaine", clean: "34 rue sainte monique bordeaux · region NAQ", note: "French street abbreviations, region code" },
  { raw: "NO-103, SANKARAPURAM NEW COLONY, THOOTHUKUDI, தமிழ்நாடு", clean: "no 103 sankarapuram new colony thoothukudi · state TN", note: "native-script state to code" },
];

const SPLITS = [
  { k: "Ghost 20%", d: "S1 rows removed, their S2/S3 records kept as orphans, so the model sees realistic non-matching records" },
  { k: "A 40%", d: "fine-tunes the e5 retriever and mines transliteration dictionaries" },
  { k: "B 20%", d: "trains the matchers (all 11 models in the comparison)" },
  { k: "V 20%", d: "validation: half tunes thresholds, half reports every number" },
];

const STAGES: [string, string, string][] = [
  ["Normalise", "Polars + RapidFuzz", "Script transliteration, abbreviation expansion, legal form, state/PIN/house-number parsing, phonetic keys."],
  ["Retrieve", "multilingual-e5-base, fine-tuned", "Bi-encoder kNN on GPU: top-40 per S1 and top-10 reverse; a learned blocker keeps ~6 pairs per S1 with 99.7% recall ceiling."],
  ["Featurise", "81 features", "Name/address fuzzy similarities, token rarity, number agreement, retrieval ranks and gaps, per-S1 group context."],
  ["Re-rank", "e5 / BGE-reranker-v2-m3 / Qwen3-Reranker-4B", "Fine-tuned cross-encoders (LoRA for Qwen, 4B params, Apache-2.0) score uncertain pairs, used as features."],
  ["Match", "2-stage XGBoost", "Stage 1 scores pairs; stage 2 adds each S1's context (best score, rank, gap, open source slots)."],
  ["Decide", "Expected-F0.5 rule", "Records go to their best S1; per S1 the subset maximising expected F0.5 is kept, or none."],
];

export default function Pipeline() {
  return (
    <main>
      <PageHead eyebrow="Pipeline" title="From noisy rows to resolved businesses"
        sub="Data cleaning, splitting, candidate generation, matching and deployment: what runs where, and why." />
      <div className="relative mx-auto -mt-8 max-w-6xl space-y-6 px-4 pb-20 sm:px-8">
        <section className="glass rounded-2xl p-5 sm:p-6">
          <h2 className="text-xl font-medium">Pipeline stages</h2>
          <ol className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {STAGES.map(([t, tech, d], i) => (
              <li key={t} className="relative rounded-2xl border border-fg/10 bg-fg/[0.03] p-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">{i + 1}. {t}</h3>
                  {i < STAGES.length - 1 && <ArrowRight size={16} className="hidden shrink-0 text-accent lg:block" aria-hidden />}
                </div>
                <p className="mt-1 text-xs text-primary">{tech}</p>
                <p className="mt-2 text-sm leading-relaxed text-fg/72">{d}</p>
              </li>
            ))}
          </ol>
        </section>

        <section className="glass rounded-2xl p-5 sm:p-6">
          <h2 className="text-xl font-medium">Data cleaning: before and after</h2>
          <p className="mt-1 text-sm text-fg/65">Real records from the test set.</p>
          <div className="mt-5 divide-y divide-fg/10">
            {CLEAN.map((c) => (
              <div key={c.raw} className="grid gap-1 py-3 sm:grid-cols-[1fr_1fr_220px] sm:gap-4">
                <div className="text-fg/85">{c.raw}</div>
                <div className="font-mono text-sm text-accent">{c.clean}</div>
                <div className="text-xs text-fg/60">{c.note}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="glass rounded-2xl p-5 sm:p-6">
          <h2 className="text-xl font-medium">Splitting without leakage</h2>
          <p className="mt-1 text-sm text-fg/65">Training S1 businesses are split by a hash of their id, so all records of a business land in one split.</p>
          <div className="mt-5 flex h-4 overflow-hidden rounded-full">
            <div className="w-[20%] bg-[#8a86a3]" /><div className="w-[40%] bg-primary" /><div className="w-[20%] bg-cobalt" /><div className="w-[20%] bg-accent" />
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            {SPLITS.map((s) => (
              <div key={s.k}><div className="font-medium">{s.k}</div><div className="text-sm text-fg/65">{s.d}</div></div>
            ))}
          </div>
        </section>

        <section className="glass rounded-2xl p-5 sm:p-6">
          <h2 className="text-xl font-medium">Deployment</h2>
          <div className="mt-4 grid gap-4 text-sm text-fg/75 sm:grid-cols-3">
            <div><div className="font-medium text-fg">Next.js frontend</div>Responsive, installable as a PWA on Android, iOS and desktop; proxies /api to the backend.</div>
            <div><div className="font-medium text-fg">FastAPI backend</div>Live matcher (normalise, e5 retrieval, 81 features, XGBoost + SHAP), SQLite FTS for browsing.</div>
            <div><div className="font-medium text-fg">Public access</div>Served from a GPU box through a Cloudflare tunnel over HTTPS; Docker files included.</div>
          </div>
        </section>
      </div>
    </main>
  );
}
