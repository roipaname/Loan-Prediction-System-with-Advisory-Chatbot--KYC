"""
Regenerates the seven figures referenced in the LAPAS research paper into
figs/, using the warm/neutral palette from RESEARCH_PAPER_BRIEF.md Section 7.
Each figure is written as both a vector .pdf and a 300dpi .png.

Fig 1-2 are architecture/flow diagrams (no CSV source). Fig 3-4 come from
models/model_comparison.csv, Fig 5-7 from reports/tfidf_chroma_metrics.csv.

    .venv/bin/python -m scripts.generate_paper_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FIGS_DIR = BASE_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# palette (research_paper_brief.md section 7 - fixed, no blue)
INK = "#2B2825"          # warm charcoal - primary text / strongest series
INK_SOFT = "#6B6259"     # warm grey - secondary ink / muted series
IVORY = "#FAF7F2"        # warm ivory - background
SEPIA = "#A9784C"        # sepia tan - primary accent
BRONZE = "#8B7355"       # muted bronze - secondary accent
TAUPE = "#B5A48C"        # warm taupe - tertiary accent
GRIDLINE = "#D8D2C4"     # pale warm grey - structural lines
RUST = "#9C5B3E"         # muted rust - caution emphasis, used sparingly

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.facecolor": IVORY,
    "figure.facecolor": IVORY,
    "savefig.facecolor": IVORY,
    "axes.grid": False,
})


def _save(fig, name, **savefig_kwargs):
    """Save fig as both vector PDF (for LaTeX/print) and 300 dpi PNG (for
    copy-paste into Word/Google Docs)."""
    pdf_path = FIGS_DIR / f"{name}.pdf"
    png_path = FIGS_DIR / f"{name}.png"
    fig.savefig(pdf_path, **savefig_kwargs)
    fig.savefig(png_path, dpi=300, **savefig_kwargs)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def _clean_axes(ax, top=False, right=False, left=True, bottom=True):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if left:
        ax.spines["left"].set_color(GRIDLINE)
    if bottom:
        ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SOFT, length=3)


# fig 1: system architecture diagram

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fc=IVORY, ec=INK, lw=1.1, fs=8.3, tc=INK, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fs, color=tc, weight=weight, linespacing=1.35)
        return patch

    def arrow(x1, y1, x2, y2, ec=INK_SOFT, lw=1.1):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                               mutation_scale=11, linewidth=lw, color=ec,
                               shrinkA=1, shrinkB=1)
        ax.add_patch(arr)

    # Row 1: three pipelines (sepia accent fill)
    pipe_w, pipe_h, gap = 2.85, 1.55, 0.225
    x0 = 0.35
    labels = [
        "ML Classification\nPipeline\n(4 classifiers)",
        "RAG Advisory\nPipeline\n(LIME + Mistral-7B)",
        "Retrieval Benchmark\nPipeline\n(TF-IDF vs Chroma)",
    ]
    xs = [x0, x0 + pipe_w + gap, x0 + 2 * (pipe_w + gap)]
    for x, label in zip(xs, labels):
        box(x, 9.9, pipe_w, pipe_h, label, fc=SEPIA, ec=INK, tc=IVORY, fs=8.0, weight="bold")

    # Row 2: data layer
    box(1.1, 7.55, 7.8, 1.15,
        "PostgreSQL (8 tables): loan_applicants · engineered_features · ml_models\n"
        "model_predictions · retrieval_documents · rag_explanations\n"
        "rag_explanation_chunks · evaluator_assessments",
        fc=TAUPE, ec=INK, tc=INK, fs=7.3)

    # Row 3: serving layer
    box(1.1, 5.35, 7.8, 1.55,
        "FastAPI Backend\nsingletons: classifier (warmed at startup) · TF-IDF store · dense store\n"
        "routers: applicants · advisory · comparisons",
        fc=IVORY, ec=INK, tc=INK, fs=7.6)

    # Row 4: presentation layer
    box(1.1, 3.15, 7.8, 1.55,
        "Streamlit Frontend (5 pages)\nApplication · Customers · Dashboard · AI Advisory · Findings",
        fc=IVORY, ec=INK, tc=INK, fs=7.6)

    # Row 5: platform constraint note
    box(1.1, 1.05, 7.8, 1.55,
        "macOS Intel (x86–64) runtime constraints\n"
        "XGBoost loaded before PyTorch · sequential embedding · OMP_NUM_THREADS=1",
        fc=IVORY, ec=BRONZE, lw=1.0, tc=INK_SOFT, fs=7.2)

    # arrows
    for x in xs:
        arrow(x + pipe_w / 2, 9.9, 5.0 + (x + pipe_w / 2 - 5.0) * 0.15, 8.70)
    arrow(5.0, 7.55, 5.0, 6.90)
    arrow(5.0, 5.35, 5.0, 4.70)
    arrow(5.0, 3.15, 5.0, 2.60)

    ax.text(5.0, 11.75, "LAPAS System Architecture", ha="center", va="center",
            fontsize=12, color=INK, weight="bold")

    fig.tight_layout(pad=0.6)
    _save(fig, "Fig1_Architecture")


# fig 2: data flow diagram

def fig2_dataflow():
    fig, ax = plt.subplots(figsize=(6.6, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.5)
    ax.axis("off")

    def box(x, y, w, h, text, fc=IVORY, ec=INK, lw=1.1, fs=7.8, tc=INK, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fs, color=tc, weight=weight, linespacing=1.35)
        return patch

    def arrow(x1, y1, x2, y2, ec=INK_SOFT, lw=1.1, style="-|>"):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                               mutation_scale=11, linewidth=lw, color=ec,
                               shrinkA=1, shrinkB=1)
        ax.add_patch(arr)

    main_w, main_h = 6.2, 1.05
    mx = 1.9

    steps = [
        (14.1, "data/raw/loan_data.csv\n45,000 rows × 14 columns", IVORY, INK),
        (12.55, "Feature Engineering\ndatabase/feature_eng.py → 36 columns", SEPIA, IVORY),
        (11.0, "data/processed/loan_features.csv", IVORY, INK),
        (9.45, "Model Training\nscripts/train_model.py (4 classifiers)", SEPIA, IVORY),
        (7.9, "models/best_model.joblib\nchampion: XGBoost (ROC-AUC 0.9776)", IVORY, INK),
        (6.05, "LoanContextBuilder\nfeature alignment + inference + top-15 LIME features", SEPIA, IVORY),
        (4.5, "LoanAdvisor.advise()\nMistral-7B-Instruct via HuggingFace Inference API", SEPIA, IVORY),
        (2.95, "Markdown Advisory Report", IVORY, INK),
        (1.4, "FastAPI Backend → Streamlit Frontend", IVORY, INK),
    ]
    for y, label, fc, tc in steps:
        box(mx, y, main_w, main_h, label, fc=fc, tc=tc)

    for i in range(len(steps) - 1):
        y_top = steps[i][0]
        y_bot = steps[i + 1][0] + main_h
        arrow(mx + main_w / 2, y_top, mx + main_w / 2, y_bot)

    # side branch: strategy docs feeding into LoanContextBuilder / retrieval
    side_w = 1.65
    box(0.10, 6.05, side_w, 1.55,
        "Loan Strategy Docs\n9 sources\n(5 PDF + 4 TXT)",
        fc=TAUPE, ec=INK, tc=INK, fs=6.4)
    box(0.10, 4.35, side_w, 1.35,
        "TF-IDF Store /\nChromaDB\nDense Store",
        fc=TAUPE, ec=INK, tc=INK, fs=6.4)
    arrow(0.10 + side_w / 2, 6.05, 0.10 + side_w / 2, 5.70)
    arrow(0.10 + side_w, 4.35 + 1.35 / 2, mx, 6.05 + main_h / 2)

    ax.text(5.0, 15.05, "LAPAS Data Flow", ha="center", va="center",
            fontsize=12, color=INK, weight="bold")

    fig.tight_layout(pad=0.6)
    _save(fig, "Fig2_DataFlow")


# fig 3: classifier comparison grouped bar chart

def fig3_classifier_bars():
    df = pd.read_csv(BASE_DIR / "models" / "model_comparison.csv")
    name_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "logistic_regression": "Logistic\nRegression",
        "naive_bayes": "Naive Bayes",
    }
    df["display"] = df["model"].map(name_map)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    metrics = ["roc_auc", "f1_weighted", "mcc"]
    metric_labels = ["ROC-AUC", "F1 (weighted)", "MCC"]
    colors = [SEPIA, BRONZE, TAUPE]

    n_models = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, (m, lab, c) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, df[m], width=width, label=lab,
                       color=c, edgecolor=INK, linewidth=0.6)
        for b, v in zip(bars, df[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.0, color=INK_SOFT, rotation=90)

    champ_idx = df.index[df["is_champion"] == True].tolist()
    for idx in champ_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color=SEPIA, alpha=0.08, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["display"], fontsize=8.3)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison (n = 45,000 rows, 80/20 split)", fontsize=9.5, pad=10)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    fig.tight_layout()
    _save(fig, "Fig3_ClassifierBars")


# fig 4: composite rank vs roc-auc rank slope chart

def fig4_rank_comparison():
    df = pd.read_csv(BASE_DIR / "models" / "model_comparison.csv")
    name_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Naive Bayes",
    }
    df["display"] = df["model"].map(name_map)
    df["composite_order"] = df["composite_rank"].rank(method="first").astype(int)
    df["auc_order"] = df["roc_auc"].rank(ascending=False, method="first").astype(int)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = {"XGBoost": SEPIA, "Random Forest": BRONZE, "Logistic Regression": TAUPE, "Naive Bayes": INK_SOFT}

    for _, row in df.iterrows():
        c = colors[row["display"]]
        lw = 2.0 if row["display"] in ("XGBoost", "Random Forest") else 1.2
        ax.plot([0, 1], [-row["composite_order"], -row["auc_order"]],
                color=c, linewidth=lw, marker="o", markersize=6,
                markerfacecolor=c, markeredgecolor=INK, markeredgewidth=0.5, zorder=3)
        ax.text(-0.04, -row["composite_order"], row["display"], ha="right", va="center",
                fontsize=8.2, color=INK)
        ax.text(1.04, -row["auc_order"], row["display"], ha="left", va="center",
                fontsize=8.2, color=INK)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-4.6, -0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Composite\nmulti-metric rank", "Single-metric\nROC-AUC rank"], fontsize=8.5)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Champion Selection: Composite Rank vs ROC-AUC Rank", fontsize=9.3, pad=12)

    fig.tight_layout()
    _save(fig, "Fig4_RankComparison")


# fig 5: retrieval latency comparison (log scale)

def fig5_latency():
    df = pd.read_csv(BASE_DIR / "reports" / "tfidf_chroma_metrics.csv")
    n = len(df)
    idx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (t, c) in zip(idx, zip(df["tfidf_latency_ms"], df["chroma_latency_ms"])):
        ax.plot([i, i], [t, c], color=GRIDLINE, linewidth=1.0, zorder=1)
    ax.scatter(idx, df["tfidf_latency_ms"], color=INK, s=26, zorder=3,
               label="TF-IDF (sparse)", edgecolor=IVORY, linewidth=0.4)
    ax.scatter(idx, df["chroma_latency_ms"], color=SEPIA, s=26, zorder=3,
               label="ChromaDB (dense)", edgecolor=IVORY, linewidth=0.4)

    ax.axhline(df["tfidf_latency_ms"].mean(), color=INK, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(df["chroma_latency_ms"].mean(), color=SEPIA, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(n + 0.4, df["tfidf_latency_ms"].mean(), f"mean {df['tfidf_latency_ms'].mean():.2f} ms",
            fontsize=6.6, color=INK_SOFT, va="center")
    ax.text(n + 0.4, df["chroma_latency_ms"].mean(), f"mean {df['chroma_latency_ms'].mean():.2f} ms",
            fontsize=6.6, color=SEPIA, va="center")

    ax.set_yscale("log")
    ax.set_xlabel("Query index (n = 20)")
    ax.set_ylabel("Retrieval latency (ms, log scale)")
    ax.set_title("Retrieval Latency: TF-IDF vs ChromaDB", fontsize=9.5, pad=10)
    ax.set_xlim(0, n + 3.2)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.yaxis.grid(True, which="major", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    fig.tight_layout()
    _save(fig, "Fig5_Latency")


# fig 6: retrieval similarity score comparison (top-1 and mean)

def fig6_similarity():
    df = pd.read_csv(BASE_DIR / "reports" / "tfidf_chroma_metrics.csv")
    n = len(df)
    idx = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

    panels = [
        ("tfidf_top1_score", "chroma_top1_score", "Top-1 similarity score"),
        ("tfidf_mean_score", "chroma_mean_score", "Mean similarity score (all results)"),
    ]
    for ax, (tcol, ccol, title) in zip(axes, panels):
        for i, (t, c) in zip(idx, zip(df[tcol], df[ccol])):
            ax.plot([i, i], [t, c], color=GRIDLINE, linewidth=1.0, zorder=1)
        ax.scatter(idx, df[tcol], color=INK, s=24, zorder=3, label="TF-IDF",
                   edgecolor=IVORY, linewidth=0.4)
        ax.scatter(idx, df[ccol], color=SEPIA, s=24, zorder=3, label="ChromaDB",
                   edgecolor=IVORY, linewidth=0.4)
        ax.set_title(title, fontsize=8.6)
        ax.set_xlabel("Query index")
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        _clean_axes(ax)

    axes[0].set_ylabel("Cosine similarity")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.6)
    fig.suptitle("Retrieval Similarity Scores: TF-IDF vs ChromaDB (n = 20 queries)",
                 fontsize=9.5, y=1.02)

    fig.tight_layout()
    _save(fig, "Fig6_Similarity", bbox_inches="tight")


# fig 7: jaccard overlap + spearman rank correlation

def fig7_overlap():
    df = pd.read_csv(BASE_DIR / "reports" / "tfidf_chroma_metrics.csv")
    n = len(df)
    idx = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    ax = axes[0]
    ax.scatter(idx, df["jaccard_overlap"], color=INK, s=30, zorder=3,
               edgecolor=IVORY, linewidth=0.4)
    mean_j = df["jaccard_overlap"].mean()
    ax.axhline(mean_j, color=SEPIA, linewidth=1.3, linestyle="--")
    ax.text(n + 0.3, mean_j, f"mean {mean_j:.3f}", fontsize=7.2, color=SEPIA, va="center")
    ax.set_title("Jaccard overlap (n = 20)", fontsize=8.8)
    ax.set_xlabel("Query index")
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(0, n + 2.6)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    ax2 = axes[1]
    defined = df["spearman_rho"].notna()
    mean_s = df.loc[defined, "spearman_rho"].mean()
    ax2.scatter(idx[defined], df.loc[defined, "spearman_rho"], color=INK, s=30, zorder=3,
                label=f"defined (n = {defined.sum()})", edgecolor=IVORY, linewidth=0.4)
    ax2.scatter(idx[~defined], np.zeros((~defined).sum()), marker="x", color=RUST, s=34,
                zorder=3, label=f"undefined (n = {(~defined).sum()})")
    ax2.axhline(mean_s, color=SEPIA, linewidth=1.3, linestyle="--")
    ax2.text(n + 0.3, mean_s, f"mean {mean_s:.3f}", fontsize=7.2, color=SEPIA, va="center")
    ax2.axhline(0, color=GRIDLINE, linewidth=0.8)
    ax2.set_title("Spearman rank correlation (n = 17 of 20)", fontsize=8.8)
    ax2.set_xlabel("Query index")
    ax2.set_ylabel("Spearman rho")
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_xlim(0, n + 2.6)
    ax2.legend(frameon=False, loc="lower left", fontsize=7.0)
    ax2.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    _clean_axes(ax2)

    fig.suptitle("Result-Set Agreement Between TF-IDF and ChromaDB", fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, "Fig7_Overlap", bbox_inches="tight")


if __name__ == "__main__":
    fig1_architecture()
    fig2_dataflow()
    fig3_classifier_bars()
    fig4_rank_comparison()
    fig5_latency()
    fig6_similarity()
    fig7_overlap()
    print(f"\nAll figures written to {FIGS_DIR}")
