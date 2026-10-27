"""Report figures (report/figures/*.png) from results/*.csv written by eda.py and compare_models.py.

Usage: python experiments/figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
WORK = Path.home() / "ber" / "work"

INK, MUTED, GRID = "#1d1b2e", "#6b6880", "#e6e4ef"
FAMILY = {
    "Rule-based": "#a9a6b8", "Transformer (bi-encoder)": "#c3b9f7", "Transformer (cross-encoder)": "#7c6cf0",
    "Linear": "#8f8ba6", "Bagged trees": "#5b8def", "Neural network": "#e86aa8", "Gradient boosting": "#2f55d4",
    "Hybrid (ours)": "#f0435f",
}
COUNTRY = {"India": "#f0435f", "US": "#5b6cf0", "France": "#2bb3a3"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def short(m):
    return (m.replace(" (RapidFuzz)", "").replace(" (fine-tuned e5)", " (e5)").replace(" (fine-tuned)", "")
            .replace(" (3-layer, PyTorch)", "").replace("Ours: 2-stage XGBoost + cross-encoder", "Ours: 2-stage + x-enc")
            .replace("Ours + expected-F0.5 decision", "Ours + F0.5 rule"))


def data_overview():
    src = pl.read_csv(RES / "eda_sources.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, split in zip(axes, ("train", "test")):
        d = src.filter(pl.col("split") == split)
        countries = [c for c in ("India", "US", "France") if c in d["country"].to_list()]
        x = np.arange(3)
        w = 0.8 / len(countries)
        for i, c in enumerate(countries):
            v = [d.filter((pl.col("source") == s) & (pl.col("country") == c))["records"].sum() / 1e6 for s in ("S1", "S2", "S3")]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, color=COUNTRY[c], label=c)
        ax.set_xticks(x, ["Source 1", "Source 2", "Source 3"])
        ax.set_ylabel("records (millions)")
        ax.set_title(f"{split.capitalize()} set")
        ax.legend()
    save(fig, "fig_data_overview")


def noise_profile():
    src = pl.read_csv(RES / "eda_sources.csv").filter((pl.col("split") == "test") & (pl.col("source") != "S1"))
    metrics = [("address_missing", "address missing"), ("name_non_latin", "name in Indic script"),
               ("address_non_latin", "address in Indic script"), ("landmark", "landmark reference")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(metrics))
    countries = ["India", "US", "France"]
    for i, c in enumerate(countries):
        d = src.filter(pl.col("country") == c)
        w = d["records"].to_numpy()
        v = [float((d[m].to_numpy() * w).sum() / w.sum()) * 100 for m, _ in metrics]
        ax.bar(x + (i - 1) * 0.26, v, 0.26, color=COUNTRY[c], label=c)
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylabel("% of S2/S3 test records")
    ax.set_title("Noise in Sources 2 and 3 (test set)")
    ax.legend()
    save(fig, "fig_noise")


def cardinality():
    c = pl.read_csv(RES / "eda_cardinality.csv")
    ks = ["0", "1", "2", "3", "4", "5", "6+"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, ctry in enumerate(("India", "US")):
        d = c.filter(pl.col("country") == ctry)
        tot = d["len"].sum()
        v = [d.filter(pl.col("k").cast(pl.Utf8) == k)["len"].sum() / tot * 100 for k in ks]
        ax.bar(np.arange(7) + (i - 0.5) * 0.4, v, 0.4, color=COUNTRY[ctry], label=ctry)
    ax.set_xticks(np.arange(7), ks)
    ax.set_xlabel("matching S2/S3 records per Source 1 entity")
    ax.set_ylabel("% of S1 entities")
    ax.set_title("Ground-truth cluster sizes (train)")
    ax.legend()
    save(fig, "fig_cardinality")


def cleaning():
    c = pl.read_csv(RES / "eda_cleaning.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, ctry in zip(axes, ("India", "US")):
        r = c.filter(pl.col("a_country") == ctry).row(0, named=True)
        labels = ["exact name", "house number"]
        before = [r["name_exact_raw"], r["number_eq_raw"]]
        after = [r["name_exact_clean"], r["number_eq_clean"]]
        x = np.arange(2)
        ax.bar(x - 0.2, np.array(before) * 100, 0.4, color="#c9c6d6", label="raw")
        ax.bar(x + 0.2, np.array(after) * 100, 0.4, color=COUNTRY[ctry], label="after cleaning")
        for xi, (b, a) in enumerate(zip(before, after)):
            ax.text(xi - 0.2, b * 100 + 1.5, f"{b * 100:.0f}%", ha="center", fontsize=8, color=MUTED)
            ax.text(xi + 0.2, a * 100 + 1.5, f"{a * 100:.0f}%", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 105)
        ax.set_title(f"{ctry}: true pairs whose key agrees")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("% of true pairs (V split)")
    save(fig, "fig_cleaning")


def blocking():
    b = pl.read_csv(WORK / "blocking_study.csv")
    shutil.copy(WORK / "blocking_study.csv", RES / "blocking_study.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    series = (("blocker", "", "#f0435f", "learned blocker (threshold q)"),
              ("fwd", "rev<=1", "#5b6cf0", "top-k retrieval (reverse rank <= 1)"),
              ("fwd", "rev<=3", "#a9a6b8", "top-k retrieval (reverse rank <= 3)"))
    for kind, rev, color, label in series:
        d = b.filter(pl.col("rule").str.starts_with(kind) & pl.col("rule").str.contains(rev, literal=True)
                     & ~pl.col("rule").str.contains("top-2")).sort("cands_per_s1")
        ax.plot(d["cands_per_s1"], d["recall"] * 100, "o-", color=color, ms=3.5, label=label)
    ax.set_xlabel("candidates per Source 1 entity")
    ax.set_ylabel("pair recall (%)")
    ax.set_title("Blocking: recall vs candidate-set size")
    ax.set_xlim(3, 30)
    ax.legend(loc="lower right")
    save(fig, "fig_blocking")


def models():
    m = pl.read_csv(RES / "model_comparison.csv").sort("entity_f05")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(m.height)
    ax.barh(y, m["entity_f05"], color=[FAMILY[f] for f in m["family"]])
    for yi, v in zip(y, m["entity_f05"]):
        ax.text(v + 0.002, yi, f"{v:.4f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [short(x) for x in m["model"]])
    ax.set_xlim(0.7, 1.01)
    ax.set_xlabel("macro F0.5 (V-test, 221k S1 entities)")
    ax.set_title("Model comparison on the competition metric")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
    ax.legend(handles, FAMILY.keys(), fontsize=7, loc="lower right", ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_models_f05")

    curves = np.load(RES / "curves.npz")
    fam = dict(zip(m["model"], m["family"]))
    show = ["Fuzzy string rule (RapidFuzz)", "Bi-encoder cosine (fine-tuned e5)", "Logistic Regression",
            "BGE-reranker-v2-m3 (fine-tuned)", "Random Forest", "XGBoost", "Ours: 2-stage XGBoost + cross-encoder"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in show:
        c = FAMILY[fam[name]]
        ls = "--" if fam[name] in ("Linear", "Bagged trees") else "-"
        axes[0].plot(curves[f"{name}|rc"], curves[f"{name}|pr"], ls, color=c, lw=1.6, label=short(name))
        axes[1].plot(curves[f"{name}|fpr"], curves[f"{name}|tpr"], ls, color=c, lw=1.6, label=short(name))
    axes[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (pair level)", ylim=(0.4, 1.01))
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (pair level, log x)", xscale="log", xlim=(1e-4, 1))
    axes[1].legend(fontsize=7, loc="lower right")
    save(fig, "fig_pr_roc")

    pc = pl.read_csv(RES / "per_country.csv").filter(pl.col("model") != "Ours + expected-F0.5 decision")
    order = m.filter(pl.col("model") != "Ours + expected-F0.5 decision").sort("entity_f05", descending=True)["model"].to_list()[:8]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(order))
    for i, ctry in enumerate(("India", "US")):
        v = [pc.filter((pl.col("model") == o) & (pl.col("country") == ctry))["f05"][0] for o in order]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=COUNTRY[ctry], label=ctry)
    ax.set_xticks(x, [short(o) for o in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.95, 0.995)
    ax.set_ylabel("macro F0.5")
    ax.set_title("Per-country F0.5 of the top 8 models")
    ax.legend(ncol=2)
    save(fig, "fig_country")

    h = pl.read_csv(RES / "hard_pairs.csv").filter(pl.col("model") != "Ours + expected-F0.5 decision").sort("pr_auc")
    fam["Qwen3-Reranker-4B (LoRA)"] = "Transformer (cross-encoder)"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(h.height)
    ax.barh(y, h["pr_auc"], color=["#e86aa8" if n.startswith("Qwen") else FAMILY[fam[n]] for n in h["model"]])
    for yi, v in zip(y, h["pr_auc"]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, [short(x) for x in h["model"]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC")
    ax.set_title(f"Hard pairs: {h['n_pairs'][0]:,} uncertain candidates ({h['n_pos'][0] / h['n_pairs'][0]:.0%} true)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_hard_pairs")

    ours = m.filter(pl.col("model") == "Ours: 2-stage XGBoost + cross-encoder").row(0, named=True)
    xgbr = m.filter(pl.col("model") == "XGBoost").row(0, named=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, r, t in ((axes[0], xgbr, "XGBoost (81 features)"), (axes[1], ours, "Ours (2-stage + cross-encoders)")):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Purples")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=9)
        ax.set_xticks([0, 1], ["pred non-match", "pred match"])
        ax.set_yticks([0, 1], ["non-match", "match"])
        ax.set_title(t, fontsize=10)
        ax.grid(False)
    save(fig, "fig_confusion")


def importance():
    import xgboost as xgb
    bst = xgb.Booster(model_file=str(ROOT / "app" / "data" / "model.json"))
    g = sorted(bst.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([k for k, _ in g], [v for _, v in g], color="#5b6cf0")
    ax.set_xlabel("total gain")
    ax.set_title("Top features of the XGBoost matcher")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_importance")


def learning_curve():
    lc = pl.read_csv(WORK / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(lc["rows"] / 1e6, lc["f05"], "o-", color="#f0435f")
    for r, f in zip(lc["rows"], lc["f05"]):
        ax.annotate(f"{f:.4f}", (r / 1e6, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("training pairs (millions)")
    ax.set_ylabel("macro F0.5 on V")
    ax.set_title("Learning curve of the matcher")
    save(fig, "fig_learning_curve")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [("Raw TSVs", "3 sources\nUS / India / France", "#ece9f7"),
             ("Normalise", "transliterate, expand\nabbrev., parse address", "#ddd7fb"),
             ("Retrieve", "fine-tuned e5 kNN\n+ learned blocker", "#c3b9f7"),
             ("Featurise", "81 similarity +\ncontext features", "#a99cf3"),
             ("Re-rank", "e5 / BGE / Qwen3-4B\ncross-encoders", "#8f80ee"),
             ("Match", "2-stage XGBoost", "#f7a1b0"),
             ("Decide", "exclusive records +\nexpected-F0.5 subset", "#f0435f")]
    w, gap = 1.36, 0.22
    for i, (t, d, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 1.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="none"))
        ax.text(x + w / 2, 1.95, t, ha="center", va="center", fontsize=10.5, weight="bold",
                color="white" if i == 6 else INK)
        ax.text(x + w / 2, 1.35, d, ha="center", va="center", fontsize=7.6, color="white" if i == 6 else INK)
        if i < len(steps) - 1:
            ax.annotate("", (x + w + gap - 0.02, 1.6), (x + w + 0.02, 1.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(5.5, 0.45, "matching_results.tsv  ·  candidate_pairs.tsv  ·  FastAPI + Next.js demo", ha="center", fontsize=9, color=MUTED)
    save(fig, "fig_architecture")


def screenshots():
    shots = Path.home() / ".shot"
    for n in ("home", "match", "models", "explore-m", "home-m", "pipeline"):
        if (shots / f"{n}.png").exists():
            shutil.copy(shots / f"{n}.png", FIG / f"ui_{n}.png")


if __name__ == "__main__":
    data_overview()
    noise_profile()
    cardinality()
    cleaning()
    blocking()
    models()
    importance()
    learning_curve()
    architecture()
    screenshots()
    print(json.dumps(sorted(p.name for p in FIG.iterdir())))
