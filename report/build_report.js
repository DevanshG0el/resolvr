// Builds report/Report_Business_Entity_Resolution.docx from results/*.csv and report/figures/*.png.
// Usage: node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak,
  PageNumber, Paragraph, ShadingType, Table, TableCell, TableOfContents, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const FIG = path.join(ROOT, "report", "figures");
const W = 9638; // content width in DXA (A4, 2 cm margins)
const INDIGO = "2B1F7A";
const CORAL = "E03A57";

// ------------------------------------------------------------------ helpers
function csv(file) {
  const lines = fs.readFileSync(path.join(RES, file), "utf8").trim().split("\n");
  const split = (l) => {
    const out = []; let cur = ""; let q = false;
    for (const ch of l) {
      if (ch === '"') q = !q;
      else if (ch === "," && !q) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  };
  const head = split(lines[0]);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}

// **bold** and *italic* inline markup
function runs(text, base = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, ...base });
    if (p.startsWith("*")) return new TextRun({ text: p.slice(1, -1), italics: true, ...base });
    return new TextRun({ text: p, ...base });
  });
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } }));

let figNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(name, widthCm, caption) {
  const file = path.join(FIG, name);
  const { w, h } = pngSize(file);
  const px = (widthCm / 2.54) * 96;
  figNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(px), height: Math.round((px * h) / w) } })],
    }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`**Fig. ${figNo}.** ${caption}`, { size: 18, color: "555366" }) }),
  ];
}

function figurePair(a, b, widthCm, caption) {
  const img = (name) => {
    const file = path.join(FIG, name);
    const { w, h } = pngSize(file);
    const px = (widthCm / 2.54) * 96;
    return new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(px), height: Math.round((px * h) / w) } });
  };
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true, children: [img(a), new TextRun("  "), img(b)] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`**Fig. ${figNo}.** ${caption}`, { size: 18, color: "555366" }) }),
  ];
}

let tabNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "C9C6D6" };
function table(header, rows, widths, caption, { highlight = () => false, fontSize = 17 } = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, hl) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { fill: INDIGO, type: ShadingType.CLEAR, color: "auto" } : hl ? { fill: "FDE7EB", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ children: runs(String(t), { size: fontSize, bold: head || undefined, color: head ? "FFFFFF" : undefined }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true, children: runs(`**Table ${tabNo}.** ${caption}`, { size: 18, color: "555366" }) }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false)) }),
        ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, i, false, highlight(r))) })),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const f4 = (x) => Number(x).toFixed(4);
const f3 = (x) => Number(x).toFixed(3);
const pct = (x, d = 1) => `${(Number(x) * 100).toFixed(d)}%`;

// ------------------------------------------------------------------ data
const models = csv("model_comparison.csv").sort((a, b) => b.entity_f05 - a.entity_f05);
const byName = Object.fromEntries(models.map((m) => [m.model, m]));
const ours = byName["Ours + expected-F0.5 decision"];
const xgbRow = byName["XGBoost"];
const hard = csv("hard_pairs.csv");
const hardOf = Object.fromEntries(hard.map((h) => [h.model, h]));
const perCountry = csv("per_country.csv");
const clean = Object.fromEntries(csv("eda_cleaning.csv").map((r) => [r.a_country, r]));
const truth = Object.fromEntries(csv("eda_truth.csv").map((r) => [r.country, r]));
const splits = Object.fromEntries(csv("eda_splits.csv").map((r) => [r.split, r]));
const ceiling = JSON.parse(fs.readFileSync(path.join(RES, "ceiling.json"), "utf8"));
const n = (x) => Number(x).toLocaleString("en-US");

// ------------------------------------------------------------------ content
const cover = [
  new Paragraph({ spacing: { before: 1800 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "MINI PROJECT REPORT", size: 24, color: "6B6880", characterSpacing: 60 })] }),
  new Paragraph({ spacing: { before: 400, after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Resolvr: Business Entity Resolution", size: 52, bold: true, color: INDIGO })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 }, children: [new TextRun({ text: "Matching noisy business records across sources, scripts and countries with a hybrid retrieval, gradient-boosting and transformer pipeline", size: 26, color: "444155" })] }),
  ...[
    ["Team name", "[Team Name]"],
    ["Team members", "[Name 1 (Roll No.)], [Name 2 (Roll No.)], [Name 3 (Roll No.)]"],
    ["Guide", "[Guide Name]"],
    ["Department", "[Department of Computer Science and Engineering]"],
    ["Institute", "[Institute Name]"],
    ["Code repository", "[GitHub URL]"],
    ["Live demo", "[Demo URL]"],
    ["Date", "[Submission Date]"],
  ].map(([k, v]) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: `${k}: `, bold: true, size: 22 }), new TextRun({ text: v, size: 22 })] })),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstract = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Abstract", bold: true, size: 26, color: INDIGO })] }),
  P(`Business identity data arrives from independent sources that share no identifiers: the same shop appears as "Vijay Industries Pvt Ltd" in one source and as "विजय इंडस्ट्रीज प्राइवेट लिमिटेड" in another, with a partial, landmark-based address. We address the ML Challenge 2026 business entity resolution task: for every record of a deduplicated reference source (Source 1), find all records of Sources 2 and 3 that describe the same business, across the US and India (training) and an unseen country, France (test only), scored by a precision-weighted, per-entity macro F0.5. Our pipeline normalises and transliterates names and addresses, generates candidates with a fine-tuned multilingual-e5 bi-encoder and a learned blocker (6.8 candidates per entity, 99.7% F0.5 ceiling), scores pairs with 81 hand-crafted similarity features in a two-stage XGBoost matcher that also consumes fine-tuned e5, BGE-reranker-v2-m3 and Qwen3-Reranker-4B cross-encoder scores, and selects matches with an expected-F0.5 decision rule. On 221,041 held-out validation entities it reaches **macro F0.5 = ${f4(ours.entity_f05)}** (precision ${pct(ours.entity_precision, 2)}, recall ${pct(ours.entity_recall, 2)}), ahead of ten other approaches including LightGBM, CatBoost, a random forest, an MLP and fine-tuned transformer cross-encoders used alone. The system is deployed as a responsive, installable Next.js web application backed by a FastAPI service that matches a typed record live in about 0.35 s and explains each decision with SHAP contributions.`),
  P(`**Keywords:** entity resolution, record linkage, blocking, gradient boosting, cross-encoder, transliteration, F0.5.`),
  new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "Contents", bold: true, size: 24, color: INDIGO })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const intro = [
  H1("1. Introduction"),
  P("Entity resolution (ER), also called record linkage, decides which records refer to the same real-world entity when the records share no common key. It underpins know-your-customer checks, supplier master data, credit bureaus and marketplaces, where a missed link costs revenue but a false merge, attaching one company's history to another, is far more damaging. That asymmetry is why the challenge scores with F0.5, which weights precision four times as heavily as recall, computed per Source 1 entity and averaged over all of them, including singletons that must be predicted as having no match."),
  P("The task has four properties that make it harder than the standard ER benchmarks. First, scale: 24.2 million records in total, so the quadratic set of possible pairs must be cut to a few candidates per entity without losing true matches. Second, multilingual noise: about 24% of Indian Source 2 names are written in an Indic script (Devanagari, Tamil, Odia and others) while the reference is in Latin script, and addresses mix abbreviations, landmarks (\"Near SBI ATM\") and missing components. Third, one-to-many structure: an entity has on average 3.5 matching records and 5.6% have none. Fourth, distribution shift: the test set adds France, a country absent from training, and the challenge forbids hard-coding the country list. Finally, any model used must be MIT/Apache-2.0 licensed and at most 8 billion parameters."),
  P("**Objectives.** (i) Clean and normalise the three sources and split the labelled data without leakage; (ii) build a scalable blocking stage with a measured recall ceiling; (iii) compare at least four state-of-the-art matching models on the same candidate set and metric; (iv) design a matcher and decision rule that maximise macro F0.5; and (v) deploy the result as a cross-platform web application. **Contributions.** A hybrid pipeline that treats transformer cross-encoders as features of a gradient-boosted matcher rather than as the final classifier, an expected-F0.5 subset-selection rule, a controlled comparison of 11 approaches, and a live, explainable demo."),
];

const lit = [
  H1("2. Literature Review"),
  P("Classical ER follows the Fellegi-Sunter model: compare attribute pairs with similarity functions and combine the evidence probabilistically [13]. Modern work keeps the two-step structure, blocking to reduce the pair space and matching to classify the surviving pairs, but replaces hand-tuned rules with learned models. We review recent work in three groups and compare it in Table 1."),
  P("**Pre-trained language-model matchers.** Ditto [1] serialises a record pair into one sequence and fine-tunes BERT/RoBERTa as a cross-encoder, adding domain-knowledge highlighting, summarisation of long values and hard-example augmentation; it improved the previous state of the art by up to 29 F1 points (e.g. 89.33 F1 on Abt-Buy) and reached 96.5% F1 matching two company tables of 789K and 412K records. Sudowoodo [2] learns record representations by contrastive self-supervision and uses one model for blocking and matching; with only 500 labels it averages 78.3 F1 versus 69.9 for Ditto. Unicorn [3] trains one DeBERTa encoder with a mixture-of-experts layer (147M parameters) on 20 datasets across seven matching tasks and beats task-specific models on 15 of them (94.56 versus 91.84 average). These models are accurate in-distribution, but Peeters et al. [5] show that fine-tuned PLM matchers lose 22-61 F1 points on entities unseen during training."),
  P("**Large language models.** Narayan et al. [4] first cast entity matching as prompting GPT-3 (175B): zero-shot it trails Ditto (54.3 versus 75.58 F1 on Amazon-Google) but ten demonstrations close most of the gap. Peeters et al. [5] study GPT-4-class and open LLMs systematically; zero-shot GPT-4 reaches 89.61 F1 on WDC Products against 84.90 for fine-tuned Ditto, and LLMs do not suffer the unseen-entity collapse. The cost is size: the strongest models are proprietary or 70B+. Jellyfish [7] instruction-tunes 7-13B local models for data preprocessing, scoring 88.84 F1 on unseen Abt-Buy with an 8B model, and AnyMatch [6] shows that a 124M-parameter GPT-2, trained with careful data selection, comes within 4.4% of GPT-4 in a zero-shot transfer setting at about 3,900 times lower inference cost."),
  P("**Blocking and unsupervised matching.** DeepBlocker [8] explores a design space of self-supervised deep blockers and finds that they beat earlier blockers on dirty and textual data. Sparkly [9] shows that a well-engineered top-k TF/IDF blocker beats eight state-of-the-art blockers on recall, candidate-set size and run time, and argues it should be the default baseline. ZeroER [10] fits a Gaussian mixture to hand-crafted similarity features and matches without labels."),
  ...table(
    ["Paper", "Year", "Approach", "Labels", "Size / cost", "Reported result", "Limitation for our task"],
    [
      ["Ditto [1]", "2020", "Fine-tuned PLM cross-encoder", "Yes", "~110-125M", "Abt-Buy 89.33 F1", "Drops on unseen entities; English"],
      ["Sudowoodo [2]", "2023", "Contrastive self-supervised", "Few / none", "RoBERTa-base", "78.3 avg F1 @500 labels", "English; no address/number logic"],
      ["Unicorn [3]", "2023", "Multi-task DeBERTa + MoE", "Yes (20 sets)", "147M", "94.56 avg (vs 91.84)", "Needs many labelled datasets"],
      ["Narayan et al. [4]", "2022", "Prompting GPT-3", "0-10 shots", "175B, paid API", "W-A 87.0 (10-shot)", "Closed, costly per pair"],
      ["Peeters et al. [5]", "2025", "LLM prompting / fine-tuning", "None / few", "GPT-4, 70B", "WDC 89.61 F1 zero-shot", "Best models > 8B or closed"],
      ["AnyMatch [6]", "2024", "Small LM transfer", "None on target", "124M", "81.96 mean F1, 9 sets", "Preprint; no blocking"],
      ["Jellyfish [7]", "2024", "Instruction-tuned local LLM", "Yes (tuning)", "7-13B", "Abt-Buy 88.84 (8B)", "Too slow for 12M pairs"],
      ["Sparkly [9]", "2023", "TF/IDF top-k blocking", "No", "CPU, cheap", "Beats 8 blockers", "Lexical: breaks on transliteration"],
      ["**Ours**", "2026", "Dense retrieval + features + GBDT + cross-encoders", "Yes", "<= 4B per model", `F0.5 ${f4(ours.entity_f05)}`, "France evaluated only on leaderboard"],
    ],
    [1250, 560, 1900, 1000, 1150, 1720, 2058],
    "Comparison of recent entity-matching work with our approach.",
    { highlight: (r) => r[0] === "**Ours**", fontSize: 15 },
  ),
  P("**Research gap.** Transformer matchers [1]-[3] reach 85-99 F1 on English product and bibliographic benchmarks but degrade out of distribution [5]; LLM matchers [4], [5], [7] generalise better but at a cost that is prohibitive for 12 million candidate pairs and often outside an open-licence, 8B budget. Blocking studies [8], [9] are monolingual. None of these works handles business names and addresses across scripts with house-number agreement, an unseen country at test time, or a precision-weighted per-entity metric. This motivates our hybrid: multilingual dense retrieval for recall across scripts [11], interpretable similarity features with gradient boosting [12] that transfer across countries, and small fine-tuned cross-encoders used only as additional evidence on a short candidate list."),
];

const method = [
  H1("3. Methodology"),
  P("Fig. 1 shows the pipeline. Each stage writes Parquet files so that stages can be re-run independently; all models are open-licence (MIT/Apache-2.0) and the largest, Qwen3-Reranker-4B [14], has 4 billion parameters."),
  ...figure("fig_architecture.png", 16.5, "End-to-end pipeline, from raw TSV files to the two submission files and the web demo."),
  H2("3.1 Candidate generation (blocking)"),
  P("A bi-encoder retriever, multilingual-e5-base [11] (278M parameters), is fine-tuned with an in-batch contrastive loss on true pairs from split A, with each record represented as its cleaned name and address. Every record is embedded once on the GPU; for each Source 1 entity we take its 40 nearest Source 2/3 records in the same country label, and for each Source 2/3 record its 10 nearest Source 1 entities (reverse search). A pair is kept if it is within the top 20 forward or top 3 reverse neighbours. A small XGBoost *learned blocker* then scores each pair from retrieval signals only (cosine, ranks, gaps and margins) and keeps pairs with probability q >= 0.01. Fig. 2 shows the trade-off that fixed this threshold: the learned blocker reaches 98.9% pair recall with 6.7 candidates per entity, where a plain top-k cut needs about 15. On the test set this yields 11.8 million candidate pairs for 1.73 million entities (6.8 each)."),
  ...figure("fig_blocking.png", 11.5, "Blocking recall against candidate-set size on validation data."),
  H2("3.2 Pairwise features"),
  P("For each candidate pair we compute 81 features in five groups: (a) **name similarity**: RapidFuzz ratio, token-set, token-sort, partial and Jaro-Winkler scores on the core, full, no-space and phonetic name, token Jaccard, whether the rarest name token is shared, legal-form agreement or conflict and numbers inside names; (b) **address similarity**: fuzzy scores on the normalised address, house-number set overlap, conflict and numeric distance, first-number equality, state agreement and locality-token overlap; (c) **retrieval signals**: cosine, forward and reverse rank, gaps to the best candidate and margins; (d) **ambiguity**: how many reference entities share the same name (overall and per state); (e) **group context**: each feature's gap and rank among the entity's other candidates. Missing values are encoded as -1 so that \"unknown\" is distinguishable from \"different\"."),
  H2("3.3 Cross-encoders and the two-stage matcher"),
  P("Three cross-encoders read both records jointly: the fine-tuned e5 model, BAAI/bge-reranker-v2-m3 [15] fine-tuned on split B, and Qwen3-Reranker-4B [14] adapted with LoRA [16] (rank 16) and run only on uncertain pairs (stage-1 probability in a band). All are cross-fitted (each half of the data is scored by a model trained on the other half) so that their scores are out-of-sample features. XGBoost [12] stage 1 scores each pair; stage 2 adds per-entity context of the stage-1 probability (maximum, rank, gap to the best, number of confident matches from the same and the other source) because whether a record is a match depends on its competitors."),
  H2("3.4 Decision rule"),
  P("Two constraints turn probabilities into match lists. **Exclusivity:** a Source 2/3 record keeps only its highest-scoring Source 1 entity. **Expected-F0.5 selection:** for each entity, candidates are sorted by calibrated probability p and we keep the top-k set that maximises the expected F0.5, computed exactly from the Poisson-binomial distribution of true matches inside and outside the set; k = 0 (predict a singleton) is scored by the probability that no candidate matches. Calibration temperature, bias and a singleton weight are tuned on V-tune."),
];

const cleaning = [
  H1("4. Data Cleaning and Preprocessing"),
  P(`**Dataset.** The training set has ${n(2206821)} Source 1 entities (883,188 India, 1,323,633 US) with 10.3 million Source 2/3 records; the test set has 1,732,544 entities including 259,452 French ones and 10.0 million Source 2/3 records (Fig. 3). True clusters have 3.46 records on average (1.67 from S2, 1.79 from S3), and ${pct(truth.India.singleton_share)} of entities are singletons in both countries (Fig. 4a). Source 1 is clean; Sources 2 and 3 carry the noise: about 3% of addresses are missing, 24% of Indian S2 names and addresses are in an Indic script, and 9% of Indian addresses use landmarks (Fig. 4b).`),
  ...figure("fig_data_overview.png", 15, "Records per source and country in the training and test sets."),
  ...figurePair("fig_cardinality.png", "fig_noise.png", 8.1, "(a) Matching Source 2/3 records per Source 1 entity (training ground truth); (b) noise in Sources 2 and 3 of the test set."),
  P("**Cleaning steps** (src/normalize.py, implemented as vectorised Polars expressions so that 24 million records are processed in minutes):"),
  ...bullets([
    "**Transliteration.** Indic-script tokens are mapped to Latin with dictionaries mined from aligned true pairs in split A (token alignment by lift, minimum count and probability), falling back to the anyascii library; state names in native script map to state codes.",
    "**Name normalisation.** Unicode folding, lower-casing, punctuation and bracket removal; legal forms (Pvt, Private, Ltd, LLC, Corp, SARL, SAS, SCI, EI ...) are extracted into a separate field so that a name and its legal form are compared separately; derived keys: core name, sorted tokens, no-space form and a phonetic code.",
    "**Address normalisation.** Abbreviation expansion (Rd to Road, St to Street, R to Rue, Av to Avenue, Frs to Freres), landmark and filler removal, extraction of house numbers, PIN/ZIP codes, state/region codes (including French regions via an alias table) and locality tokens.",
    "**Robust loading.** Files are read as tab-separated text with quoting disabled and a row-count assertion, because addresses contain commas and quote characters; missing countries are kept as an explicit UNKNOWN label rather than dropped.",
  ]),
  P(`Fig. 5 measures the effect on true validation pairs: after cleaning, the exact core name agrees for ${pct(clean.India.name_exact_clean, 0)} of Indian pairs (raw: ${pct(clean.India.name_exact_raw, 0)}) and ${pct(clean.US.name_exact_clean, 0)} of US pairs (raw: ${pct(clean.US.name_exact_raw, 0)}); house numbers agree more often, and where both records have a state it matches in ${pct(clean.India.state_eq_clean)} (India) and ${pct(clean.US.state_eq_clean)} (US) of pairs.`),
  ...figure("fig_cleaning.png", 14, "Share of true pairs whose name or house-number key agrees, before and after cleaning."),
  H2("4.1 Splitting without leakage"),
  P(`Labelled Source 1 entities are assigned to four splits by a multiplicative hash of their id, so that all records of a cluster land in the same split and the assignment is reproducible: **ghost** 20% (${n(splits.ghost.len)} entities; the S1 rows are removed but their S2/S3 records stay, which simulates records whose entity is absent and teaches the model to reject them), **A** 40% (${n(splits.A.len)}; retriever fine-tuning and dictionary mining), **B** 20% (${n(splits.B.len)}; matcher training) and **V** 20% (${n(splits.V.len)}; validation). V is halved again by a second hash: V-tune selects every threshold and decision parameter, and V-test (${n(221041)} entities) produces every number in Section 5. Because France has no labels, we additionally ran leave-one-country-out experiments (train on one country, test on the other) to check that the features transfer across countries.`),
];

const top = models.map((m) => [
  m.model, m.family, f4(m.roc_auc), f4(m.pr_auc), f4(m.pair_f1), `**${f4(m.entity_f05)}**`, f4(m.entity_precision), f4(m.entity_recall), Number(m.train_s) ? Number(m.train_s).toFixed(0) : "-",
]);
const pc = (model, c) => perCountry.find((r) => r.model === model && r.country === c);

const results = [
  H1("5. Results and Discussion"),
  H2("5.1 State-of-the-art models comparison"),
  P(`All approaches score the same ${n(Number(models[0].tp) + Number(models[0].fp) + Number(models[0].fn) + Number(models[0].tn))} candidate pairs of the 221,041 V-test entities, so differences come from the models alone, and all use the same decision step (exclusivity, then a global threshold tuned on V-tune). We compare: a **fuzzy string rule** (mean of RapidFuzz name and address token-set scores); the **fine-tuned e5 bi-encoder** cosine; two **transformer cross-encoders** used alone (fine-tuned e5 and BGE-reranker-v2-m3); and six learned classifiers trained on split B with the 81 features: **logistic regression**, **random forest**, a **3-layer MLP** (PyTorch), **LightGBM** [17], **CatBoost** [18] and **XGBoost** [12]; finally **our matcher** (two-stage XGBoost with the cross-encoder scores), with and without the expected-F0.5 rule. The blocking stage bounds every model: a perfect matcher on these candidates would score F0.5 = ${f4(ceiling.f05)}.`),
  ...table(["Model", "Family", "ROC-AUC", "PR-AUC", "Pair F1", "Macro F0.5", "Precision", "Recall", "Train s"], top,
    [2300, 1500, 780, 780, 780, 900, 860, 860, 878],
    "Model comparison on V-test (221,041 entities). Pair metrics at each model's tuned threshold; entity metrics are the competition's.",
    { highlight: (r) => r[1] === "Hybrid (ours)", fontSize: 15 }),
  ...figure("fig_models_f05.png", 14, "Macro F0.5 of all eleven approaches, coloured by model family."),
  P(`**Discussion.** Unsupervised similarity is not enough: the fuzzy rule (${f4(byName["Fuzzy string rule (RapidFuzz)"].entity_f05)}) and the retriever cosine (${f4(byName["Bi-encoder cosine (fine-tuned e5)"].entity_f05)}) cannot separate a business from a decoy with the same name on the same street. Cross-encoders, which read both records jointly, are strong on their own (BGE-reranker ${f4(byName["BGE-reranker-v2-m3 (fine-tuned)"].entity_f05)}), but gradient-boosted trees on hand-crafted features are better still (XGBoost ${f4(xgbRow.entity_f05)}, LightGBM and CatBoost within 0.001) because they combine explicit evidence, such as conflicting house numbers or a rare token shared, with the ranking context of the other candidates; the linear model lacks the interactions (${f4(byName["Logistic Regression"].entity_f05)}). Adding the cross-encoder scores and the second stage lifts F0.5 to ${f4(byName["Ours: 2-stage XGBoost + cross-encoder"].entity_f05)}, and the expected-F0.5 rule to ${f4(ours.entity_f05)}, closing ${pct((ours.entity_f05 - xgbRow.entity_f05) / (ceiling.f05 - xgbRow.entity_f05), 0)} of the remaining gap between XGBoost and the blocking ceiling. Precision is ${pct(ours.entity_precision, 2)}: the model accepts few false merges, as the metric demands. Pair-level precision-recall and ROC curves (Fig. 7) show the same ordering, and the confusion matrices (Fig. 8) show that the hybrid more than halves XGBoost's false positives (3,436 to 1,527) and cuts its false negatives by a quarter (19,546 to 14,524).`),
  ...figure("fig_pr_roc.png", 16, "Pair-level precision-recall and ROC curves on V-test."),
  ...figure("fig_confusion.png", 12, "Pair-level confusion matrices of XGBoost and of our matcher at their tuned thresholds."),
  P(`**Hard pairs.** The differences are small on average because most candidates are easy. Restricting the evaluation to the ${n(hard[0].n_pairs)} uncertain pairs on which the Qwen3-Reranker-4B was run (${pct(Number(hard[0].n_pos) / Number(hard[0].n_pairs), 0)} true matches) separates the models clearly (Fig. 9): PR-AUC is ${f3(hardOf["Ours: 2-stage XGBoost + cross-encoder"].pr_auc)} for our matcher, ${f3(hardOf.XGBoost.pr_auc)} for XGBoost, ${f3(hardOf["Qwen3-Reranker-4B (LoRA)"].pr_auc)} for the Qwen reranker alone and ${f3(hardOf["BGE-reranker-v2-m3 (fine-tuned)"].pr_auc)} for BGE. The 4B LLM alone is not better than trees on features, but its errors differ, which is why it helps as a feature.`),
  ...figure("fig_hard_pairs.png", 12, "PR-AUC on the hard, uncertain pairs."),
  P(`**Per country and features.** India and US are almost equally hard for the best model (F0.5 ${f4(pc("Ours: 2-stage XGBoost + cross-encoder", "India").f05)} and ${f4(pc("Ours: 2-stage XGBoost + cross-encoder", "US").f05)}, Fig. 10), even though Indian records carry transliteration and landmark noise, which suggests that the cleaning stage removes most of the script gap. The most important features of the deployed matcher (Fig. 11a) are address-number overlap and conflict, whether the record has an address at all, address token overlap, the record's margin to its best reference entity and legal-form conflict: once retrieval has found candidates with similar names, the address decides between a business and its namesakes. The learning curve (Fig. 11b) is almost flat (F0.5 0.9878 with 25% of the training pairs, 0.9884 with all of them), so more labels would not help much; the remaining errors are ambiguous clusters, not a lack of data.`),
  ...figure("fig_country.png", 13, "Macro F0.5 per country for the eight best models."),
  ...figurePair("fig_importance.png", "fig_learning_curve.png", 8.1, "(a) Feature importance (XGBoost gain) of the deployed matcher; (b) validation F0.5 against the number of training pairs."),
  P("**France.** The unseen country has no labels, so it was evaluated only through leaderboard submissions. Leave-one-country-out experiments indicate how well the matcher transfers: trained without any Indian labels it still reaches F0.5 0.988 on India (0.986 on the US when the US is held out) with the cross-encoder features, against 0.949 and 0.975 without them, so the multilingual cross-encoders matter most for an unseen country. Two further measures kept the pipeline country-agnostic: country is used only to restrict retrieval to records with the same label (never as a feature or one-hot), and the French route uses cross-encoders adapted with pseudo-labels from high-confidence French pairs plus French legal-form and street abbreviation rules. The final test submission (1,732,544 rows, 5.82 million matches, 5.8% predicted singletons) passes the official validator."),
];

const ui = [
  H1("6. User Interface Development and Deployment"),
  P("The demo, **Resolvr**, makes the pipeline usable by a non-expert. It has two parts. A **FastAPI backend** (app/server.py) serves the live matcher and the analysis data: GET /api/health, /api/overview (model comparison, curves, per-country and feature data), /api/search and /api/entity/{id} (SQLite with FTS5 full-text search over a stratified sample of 60,000 test entities, their candidates and the submitted matches), POST /api/match and POST /api/batch (TSV upload). A **Next.js 15 frontend** (web/, TypeScript, Tailwind CSS, Recharts) provides four screens: a landing page, *Live match*, *Explore* and *Models* (Fig. 13), plus a *Pipeline* page that documents cleaning and splitting."),
  P("**Live matching.** A typed record goes through the same code path as the batch pipeline (Fig. 12): normalisation (src/normalize.py), embedding with the fine-tuned e5 retriever on CPU, forward kNN over 390,000 indexed Source 2/3 records of the same country and reverse ranking against the Source 1 pool, the 81 features (src/features.py), XGBoost scoring, and the expected-F0.5 rule. If the query is a variant of an existing reference entity, that entity is left out of the reverse comparison so that the query does not compete with its own copy. Each candidate shows its probability, name, address and semantic similarity and the six features that pushed its score up or down (SHAP values from XGBoost's pred_contribs [19]). A query takes about 0.35 s. On 300 sampled test entities the live matcher's accepted matches agree with the full pipeline's submitted matches with 98.4% precision and 93.6% recall."),
  ...figure("ui_match.png", 15.5, "Live match: a Devanagari query, the cleaned record, and ranked candidates with SHAP explanations."),
  P("**Cross-platform accessibility.** The interface is responsive from 360 px phones to desktop (Fig. 14), is installable as a Progressive Web App (web manifest, icons, standalone display) on Android, iOS, Windows, macOS and Linux, and uses semantic HTML, labelled form fields, keyboard-reachable controls, an aria-live results region and a reduced-motion mode. Indic scripts render with Noto Sans Devanagari and Tamil fallbacks. The Next.js server proxies /api to the backend so that a single HTTPS origin is exposed; for the demo it runs on the GPU server behind a Cloudflare tunnel, and app/serve.sh starts the whole stack. Dockerfiles for the backend and frontend and a docker-compose file are included for redeployment elsewhere."),
  ...figure("ui_models.png", 13, "The Models page: headline metrics and the interactive comparison charts (Recharts)."),
  ...figure("ui_mobile.png", 14, "The app on a 390 px phone: landing page, live match with SHAP explanation, and Explore with submitted matches."),
];

const conclusion = [
  H1("7. Conclusion and Future Work"),
  P(`We built and deployed an entity resolution system for noisy, multilingual business records. Careful cleaning (transliteration, legal-form and address parsing) raises exact name agreement on true pairs from 7-14% to 56-63%; a fine-tuned dense retriever with a learned blocker keeps a 99.7% F0.5 ceiling with under seven candidates per entity; and a hybrid matcher that feeds cross-encoder scores into two-stage gradient boosting, followed by an expected-F0.5 decision rule, achieves macro F0.5 = ${f4(ours.entity_f05)} on held-out data, the best of eleven compared approaches. Future work: labelled French data or active learning for the unseen country, distilling the cross-encoders into the live matcher, and graph-based clustering across all three sources.`),
];

const refs = [
  "Y. Li, J. Li, Y. Suhara, A. Doan, and W.-C. Tan, \"Deep entity matching with pre-trained language models,\" Proc. VLDB Endow., vol. 14, no. 1, pp. 50-60, 2020, doi: 10.14778/3421424.3421431.",
  "R. Wang, Y. Li, and J. Wang, \"Sudowoodo: Contrastive self-supervised learning for multi-purpose data integration and preparation,\" in Proc. IEEE 39th Int. Conf. Data Engineering (ICDE), 2023; arXiv:2207.04122.",
  "J. Tu, J. Fan, N. Tang, P. Wang, G. Li, X. Du, X. Jia, and S. Gao, \"Unicorn: A unified multi-tasking model for supporting matching tasks in data integration,\" Proc. ACM Manag. Data, vol. 1, no. 1, art. 84, 2023, doi: 10.1145/3588938.",
  "A. Narayan, I. Chami, L. Orr, and C. Re, \"Can foundation models wrangle your data?\" Proc. VLDB Endow., vol. 16, no. 4, pp. 738-746, 2022, doi: 10.14778/3574245.3574258.",
  "R. Peeters, A. Steiner, and C. Bizer, \"Entity matching using large language models,\" in Proc. 28th Int. Conf. Extending Database Technology (EDBT), 2025, pp. 529-541, doi: 10.48786/edbt.2025.42.",
  "Z. Zhang, P. Groth, I. Calixto, and S. Schelter, \"AnyMatch - Efficient zero-shot entity matching with a small language model,\" arXiv:2409.04073, 2024.",
  "H. Zhang, Y. Dong, C. Xiao, and M. Oyamada, \"Jellyfish: Instruction-tuning local large language models for data preprocessing,\" in Proc. EMNLP, 2024, pp. 8754-8782, doi: 10.18653/v1/2024.emnlp-main.497.",
  "S. Thirumuruganathan, H. Li, N. Tang, M. Ouzzani, Y. Govind, D. Paulsen, G. Fung, and A. Doan, \"Deep learning for blocking in entity matching: A design space exploration,\" Proc. VLDB Endow., vol. 14, no. 11, pp. 2459-2472, 2021, doi: 10.14778/3476249.3476294.",
  "D. Paulsen, Y. Govind, and A. Doan, \"Sparkly: A simple yet surprisingly strong TF/IDF blocker for entity matching,\" Proc. VLDB Endow., vol. 16, no. 6, pp. 1507-1519, 2023.",
  "R. Wu, S. Chaba, S. Sawlani, X. Chu, and S. Thirumuruganathan, \"ZeroER: Entity resolution using zero labeled examples,\" in Proc. ACM SIGMOD, 2020, pp. 1149-1164, doi: 10.1145/3318464.3389743.",
  "L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei, \"Multilingual E5 text embeddings: A technical report,\" arXiv:2402.05672, 2024.",
  "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.",
  "I. P. Fellegi and A. B. Sunter, \"A theory for record linkage,\" J. Amer. Statist. Assoc., vol. 64, no. 328, pp. 1183-1210, 1969.",
  "Y. Zhang et al., \"Qwen3 Embedding: Advancing text embedding and reranking through foundation models,\" arXiv:2506.05176, 2025.",
  "J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, \"BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation,\" arXiv:2402.03216, 2024.",
  "E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" in Proc. ICLR, 2022; arXiv:2106.09685.",
  "G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" in Proc. NeurIPS, 2017, pp. 3146-3154.",
  "L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, \"CatBoost: Unbiased boosting with categorical features,\" in Proc. NeurIPS, 2018, pp. 6638-6648.",
  "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Proc. NeurIPS, 2017, pp. 4765-4774.",
];
const references = [
  H1("References"),
  ...refs.map((r, i) => new Paragraph({ spacing: { after: 80 }, indent: { left: 500, hanging: 500 }, children: [new TextRun({ text: `[${i + 1}]\t`, size: 18 }), new TextRun({ text: r, size: 18 })] })),
];

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "[Team Name]",
  title: "Resolvr: Business Entity Resolution",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: INDIGO, font: "Calibri" },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0, keepNext: true,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: CORAL, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "3B2E99", font: "Calibri" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "6B6880" })] })] }),
    },
    children: [...cover, ...abstract, ...intro, ...lit, ...method, ...cleaning, ...results, ...ui, ...conclusion, ...references],
  }],
});

const out = path.join(ROOT, "report", "Report_Business_Entity_Resolution.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
