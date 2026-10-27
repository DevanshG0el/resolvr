export type Reason = { feature: string; label: string; value: number; impact: number };
export type Candidate = {
  id: string; source: string; name: string; address: string; p: number; matched: boolean;
  cos: number; name_sim: number; addr_sim: number; reasons: Reason[];
};
export type MatchResult = {
  query: { name: string; address: string; country: string; name_clean: string; address_clean: string; legal: string; state: string | null };
  replaced: { id: string; name: string; cos: number } | null;
  candidates: Candidate[]; n_retrieved: number; n_scored: number; n_matched: number; ms: number;
};
export type Entity = { id: string; name: string; address: string; country: string };
export type EntityDetail = {
  entity: Entity & { src: number };
  candidates: { id: string; src: number; name: string; address: string; p: number | null; matched: number }[];
};
export type ModelRow = {
  model: string; family: string; roc_auc: number; pr_auc: number; threshold: number;
  pair_precision: number; pair_recall: number; pair_f1: number; tp: number; fp: number; fn: number; tn: number;
  entity_f05: number; entity_precision: number; entity_recall: number; singleton_acc: number; train_s: number;
};
export type Overview = {
  models: ModelRow[];
  per_country: { model: string; country: string; f05: number; precision: number; recall: number }[];
  hard_pairs: { model: string; roc_auc: number; pr_auc: number; n_pairs: number; n_pos: number }[];
  importance: { feature: string; gain: number }[];
  importance_live: { feature: string; label: string; gain: number }[];
  ceiling: { f05: number };
  curves: Record<string, { fpr: number[]; tpr: number[]; pr: number[]; rc: number[] }>;
  eda: null | Record<string, unknown>;
};

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    let msg = body || res.statusText;
    try { const d = JSON.parse(body).detail; if (d) msg = typeof d === "string" ? d : JSON.stringify(d); } catch {}
    throw new Error(res.status >= 500 ? `The matching service is unavailable (${res.status}). Try again in a moment.` : msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  match: (name: string, address: string, country: string) =>
    fetch("/api/match", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, address, country }) })
      .then(j<MatchResult>),
  search: (q: string, country: string) =>
    fetch(`/api/search?q=${encodeURIComponent(q)}&country=${encodeURIComponent(country)}&limit=24`).then(j<Entity[]>),
  entity: (id: string) => fetch(`/api/entity/${encodeURIComponent(id)}`).then(j<EntityDetail>),
  overview: () => fetch("/api/overview").then(j<Overview>),
};

export const COUNTRIES = ["India", "US", "France"];
// the raw sources spell missing fields as "<NULL>" or "nan"; hide those placeholders when displaying a record
export const tidy = (s: string | null | undefined) =>
  (s ?? "").replace(/(^|,\s*)(<null>|null|nan)(?=\s*(,|$))/gi, "").replace(/^\s*,\s*|\s*,\s*$/g, "").trim();
export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
