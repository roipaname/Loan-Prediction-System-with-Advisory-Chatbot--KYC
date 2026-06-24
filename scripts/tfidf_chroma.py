"""
scripts/tfidf_chroma.py
========================
Benchmarks the custom TF-IDF vector store against ChromaDB on the
loan strategy document corpus and produces comparison metrics and charts.

Experiment design
-----------------
Because no human-labelled relevance judgements exist for this corpus, the
comparison uses self-supervised and system-level metrics:

  1. Retrieval Latency         — average query time per method (ms)
  2. Score Distribution        — distribution of similarity scores returned
  3. Result-Set Overlap        — Jaccard similarity between top-k results
                                 for each query, across the two methods
  4. Rank Correlation          — Spearman rho between the ordered result lists
  5. Throughput                — queries per second for each method
  6. Vocabulary/Coverage       — TF-IDF vocab size vs ChromaDB embedding dims

Outputs
-------
  reports/tfidf_chroma_latency.png
  reports/tfidf_chroma_score_distribution.png
  reports/tfidf_chroma_overlap_heatmap.png
  reports/tfidf_chroma_rank_correlation.png
  Console table with all numeric metrics

Usage
-----
    uv run python -m scripts.tfidf_chroma
    uv run python -m scripts.tfidf_chroma --docs data/loan_strategy_docs --n-results 5
    uv run python -m scripts.tfidf_chroma --no-chroma   # TF-IDF only (faster)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger as log
from scipy.stats import spearmanr

from config.settings import BASE_DIR, DATA_DIR
from src.ai_advisor.document_loader import load_documents
from src.tf_idf import TFIDFStore

# ---------------------------------------------------------------------------
# Test query suite  — representative loan advisory retrieval scenarios
# ---------------------------------------------------------------------------

TEST_QUERIES: List[str] = [
    "applicant with poor credit score and high debt to income ratio",
    "loan rejected due to thin credit file and short credit history",
    "young inexperienced applicant no employment history",
    "high loan burden flag loan to income ratio above thirty percent",
    "steps to improve credit score before reapplying for a loan",
    "previous loan default on record impact on application",
    "unstable employment income stability requirements",
    "debt consolidation loan high risk intent score",
    "education loan home improvement loan low risk intent",
    "composite risk score high risk classification flags",
    "approved loan factors credit score good employment stable",
    "how long to wait before reapplying after rejection",
    "income bucket classification low medium high thresholds",
    "home ownership score mortgage rent own financial stability",
    "venture loan speculative purpose highest risk",
    "credit history length minimum requirements two years",
    "affordability ratio monthly income loan burden repayment capacity",
    "credit risk interaction previous default high interest rate flag",
    "score per history year credit quality relative to age",
    "reapplication checklist minimum credit score employment income",
]

_REPORTS_DIR   = BASE_DIR / "reports"
_DOCS_DIR_DEF  = DATA_DIR / "loan_strategy_docs"
_TFIDF_PERSIST = BASE_DIR / "data" / "tfidf_index"


# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------

PALETTE = {
    "tfidf":  "#4C72B0",   # muted blue
    "chroma": "#DD8452",   # muted orange
    "grid":   "#CCCCCC",
    "bg":     "#FAFAFA",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    "#888888",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Store builders
# ---------------------------------------------------------------------------

def _build_tfidf(docs_dir: Path) -> TFIDFStore:
    log.info("Building TF-IDF store from %s …", docs_dir)
    store = TFIDFStore.from_directory(docs_dir)
    log.info("TF-IDF store ready: %d docs, vocab=%d", store.count(), store.vocab_size())
    return store


def _build_chroma(docs_dir: Path) -> Any:
    from src.ai_advisor.vector_store import VectorStore
    log.info("Building ChromaDB store from %s …", docs_dir)
    store = VectorStore.from_directory(docs_dir, force_reindex=False)
    log.info("ChromaDB store ready: %d docs", store.count())
    return store


# ---------------------------------------------------------------------------
# Benchmarking primitives
# ---------------------------------------------------------------------------

def _timed_query(
    store: Any,
    queries: List[str],
    n_results: int,
    n_warmup: int = 3,
) -> Tuple[List[Dict], List[float]]:
    """
    Run queries against a store and return results + per-query latencies (ms).

    Performs n_warmup warm-up queries (excluded from timing) to allow any
    JIT compilation or caching to settle before measurement.
    """
    # Warm-up
    for q in queries[:n_warmup]:
        store.query([q], n_results=n_results)

    results: List[Dict]  = []
    latencies: List[float] = []

    for q in queries:
        t0 = time.perf_counter()
        r  = store.query([q], n_results=n_results)
        t1 = time.perf_counter()
        results.append(r)
        latencies.append((t1 - t0) * 1000.0)

    return results, latencies


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _spearman(ids_a: List[str], ids_b: List[str]) -> float:
    """Spearman rank correlation between two ordered lists of document IDs."""
    common = list(set(ids_a) & set(ids_b))
    if len(common) < 2:
        return float("nan")
    ranks_a = [ids_a.index(d) if d in ids_a else len(ids_a) for d in common]
    ranks_b = [ids_b.index(d) if d in ids_b else len(ids_b) for d in common]
    rho, _ = spearmanr(ranks_a, ranks_b)
    return float(rho)


def _extract_scores(results: List[Dict]) -> List[List[float]]:
    """Extract per-query distance lists from query result dicts."""
    scores = []
    for r in results:
        dists = r.get("distances", [[]])[0]
        # Convert distances to similarities (1 - distance) for interpretability
        scores.append([1.0 - d for d in dists])
    return scores


def _extract_ids(results: List[Dict]) -> List[List[str]]:
    return [r.get("ids", [[]])[0] for r in results]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    tfidf_results: List[Dict],
    tfidf_latencies: List[float],
    chroma_results: Optional[List[Dict]],
    chroma_latencies: Optional[List[float]],
) -> pd.DataFrame:
    """
    Compute all comparison metrics and return a summary DataFrame.
    """
    tfidf_scores = _extract_scores(tfidf_results)
    tfidf_ids    = _extract_ids(tfidf_results)

    rows = []
    for i, query in enumerate(TEST_QUERIES):
        row: Dict[str, Any] = {
            "query":                query[:60],
            "tfidf_latency_ms":     tfidf_latencies[i],
            "tfidf_top1_score":     tfidf_scores[i][0] if tfidf_scores[i] else float("nan"),
            "tfidf_mean_score":     float(np.mean(tfidf_scores[i])) if tfidf_scores[i] else float("nan"),
        }

        if chroma_results is not None:
            chroma_scores = _extract_scores(chroma_results)
            chroma_ids    = _extract_ids(chroma_results)
            row.update({
                "chroma_latency_ms":  chroma_latencies[i],
                "chroma_top1_score":  chroma_scores[i][0] if chroma_scores[i] else float("nan"),
                "chroma_mean_score":  float(np.mean(chroma_scores[i])) if chroma_scores[i] else float("nan"),
                "jaccard_overlap":    _jaccard(set(tfidf_ids[i]), set(chroma_ids[i])),
                "spearman_rho":       _spearman(tfidf_ids[i], chroma_ids[i]),
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _save_latency_chart(metrics: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Retrieval Latency Comparison: TF-IDF vs ChromaDB", fontsize=13, fontweight="bold")

    has_chroma = "chroma_latency_ms" in metrics.columns

    # Left: per-query latency line chart
    ax = axes[0]
    xs = range(len(metrics))
    ax.plot(xs, metrics["tfidf_latency_ms"], color=PALETTE["tfidf"],
            marker="o", ms=4, lw=1.5, label="TF-IDF")
    if has_chroma:
        ax.plot(xs, metrics["chroma_latency_ms"], color=PALETTE["chroma"],
                marker="s", ms=4, lw=1.5, label="ChromaDB")
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Query Latency")
    ax.legend()
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # Right: mean latency bar chart
    ax2 = axes[1]
    labels = ["TF-IDF"]
    means  = [metrics["tfidf_latency_ms"].mean()]
    stds   = [metrics["tfidf_latency_ms"].std()]
    colors = [PALETTE["tfidf"]]
    if has_chroma:
        labels.append("ChromaDB")
        means.append(metrics["chroma_latency_ms"].mean())
        stds.append(metrics["chroma_latency_ms"].std())
        colors.append(PALETTE["chroma"])
    bars = ax2.bar(labels, means, yerr=stds, color=colors, width=0.4,
                   capsize=6, error_kw={"lw": 1.5})
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{mean:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Mean Latency (ms)")
    ax2.set_title("Mean Latency (lower is better)")
    ax2.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    plt.tight_layout()
    out = out_dir / "tfidf_chroma_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved latency chart → %s", out)
    return out


def _save_score_distribution(metrics: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Similarity Score Distribution: TF-IDF vs ChromaDB", fontsize=13, fontweight="bold")

    has_chroma = "chroma_mean_score" in metrics.columns

    # Left: top-1 scores
    ax = axes[0]
    data_t = metrics["tfidf_top1_score"].dropna()
    ax.hist(data_t, bins=12, color=PALETTE["tfidf"], alpha=0.7, label="TF-IDF")
    if has_chroma:
        data_c = metrics["chroma_top1_score"].dropna()
        ax.hist(data_c, bins=12, color=PALETTE["chroma"], alpha=0.7, label="ChromaDB")
    ax.set_xlabel("Top-1 Similarity Score (higher = better match)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Top-1 Similarity Score Distribution")
    ax.legend()
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # Right: mean-score box plot
    ax2 = axes[1]
    plot_data = [metrics["tfidf_mean_score"].dropna().values]
    plot_labels = ["TF-IDF"]
    plot_colors = [PALETTE["tfidf"]]
    if has_chroma:
        plot_data.append(metrics["chroma_mean_score"].dropna().values)
        plot_labels.append("ChromaDB")
        plot_colors.append(PALETTE["chroma"])

    bp = ax2.boxplot(plot_data, labels=plot_labels, patch_artist=True, notch=False,
                     medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Mean Similarity Score (top-k results)")
    ax2.set_title("Mean Score Distribution (higher = better)")
    ax2.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    plt.tight_layout()
    out = out_dir / "tfidf_chroma_score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved score distribution chart → %s", out)
    return out


def _save_overlap_heatmap(metrics: pd.DataFrame, out_dir: Path) -> Path:
    if "jaccard_overlap" not in metrics.columns:
        log.info("Skipping overlap heatmap (no ChromaDB results).")
        return out_dir / "tfidf_chroma_overlap_heatmap.png"

    # Reshape into a 4x5 grid for visual clarity
    values = metrics["jaccard_overlap"].values
    n      = len(values)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    grid   = np.full((n_rows, n_cols), np.nan)
    for idx, v in enumerate(values):
        grid[idx // n_cols, idx % n_cols] = v

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Jaccard Overlap (0 = no overlap, 1 = identical)")
    ax.set_title(
        "Result-Set Overlap per Query: TF-IDF vs ChromaDB (Jaccard Similarity)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Query Column")
    ax.set_ylabel("Query Row")

    for r in range(n_rows):
        for c in range(n_cols):
            v = grid[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if v > 0.5 else "white")

    plt.tight_layout()
    out = out_dir / "tfidf_chroma_overlap_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved overlap heatmap → %s", out)
    return out


def _save_rank_correlation(metrics: pd.DataFrame, out_dir: Path) -> Path:
    if "spearman_rho" not in metrics.columns:
        log.info("Skipping rank correlation chart (no ChromaDB results).")
        return out_dir / "tfidf_chroma_rank_correlation.png"

    rho_vals = metrics["spearman_rho"].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rank Correlation: TF-IDF vs ChromaDB (Spearman Rho)", fontsize=13, fontweight="bold")

    # Left: histogram of rho values
    ax = axes[0]
    ax.hist(rho_vals, bins=10, color="#55A868", alpha=0.85, edgecolor="white")
    ax.axvline(rho_vals.mean(), color="crimson", lw=2, linestyle="--",
               label=f"Mean = {rho_vals.mean():.3f}")
    ax.set_xlabel("Spearman Rho (1 = identical ranking)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Distribution of Rank Correlation Scores")
    ax.legend()
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # Right: per-query rho scatter
    ax2 = axes[1]
    xs = range(len(rho_vals))
    ax2.scatter(xs, rho_vals, color="#55A868", s=40, alpha=0.8, zorder=3)
    ax2.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax2.axhline(rho_vals.mean(), color="crimson", lw=1.5, linestyle="--", label=f"Mean {rho_vals.mean():.3f}")
    ax2.set_xlabel("Query Index")
    ax2.set_ylabel("Spearman Rho")
    ax2.set_title("Per-Query Rank Agreement")
    ax2.legend()
    ax2.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    out = out_dir / "tfidf_chroma_rank_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved rank correlation chart → %s", out)
    return out


def _save_summary_dashboard(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """
    A single-figure summary dashboard combining the key metrics side by side.
    """
    has_chroma = "chroma_latency_ms" in metrics.columns

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "LAPAS Retrieval System Benchmark: TF-IDF vs ChromaDB",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1. Latency bar
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["TF-IDF"]
    means  = [metrics["tfidf_latency_ms"].mean()]
    colors = [PALETTE["tfidf"]]
    if has_chroma:
        labels.append("ChromaDB")
        means.append(metrics["chroma_latency_ms"].mean())
        colors.append(PALETTE["chroma"])
    bars = ax1.bar(labels, means, color=colors, width=0.5)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{m:.1f} ms", ha="center", va="bottom", fontsize=8)
    ax1.set_title("Mean Latency (ms)", fontsize=10)
    ax1.set_ylabel("ms")
    ax1.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # 2. Top-1 score distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(metrics["tfidf_top1_score"].dropna(), bins=8, color=PALETTE["tfidf"],
             alpha=0.7, label="TF-IDF")
    if has_chroma:
        ax2.hist(metrics["chroma_top1_score"].dropna(), bins=8, color=PALETTE["chroma"],
                 alpha=0.7, label="ChromaDB")
    ax2.set_title("Top-1 Score Distribution", fontsize=10)
    ax2.set_xlabel("Similarity")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # 3. Throughput
    ax3 = fig.add_subplot(gs[0, 2])
    n_q = len(TEST_QUERIES)
    tps_tfidf  = n_q / (metrics["tfidf_latency_ms"].sum() / 1000)
    throughput_labels = ["TF-IDF"]
    throughput_vals   = [tps_tfidf]
    tp_colors         = [PALETTE["tfidf"]]
    if has_chroma:
        tps_chroma = n_q / (metrics["chroma_latency_ms"].sum() / 1000)
        throughput_labels.append("ChromaDB")
        throughput_vals.append(tps_chroma)
        tp_colors.append(PALETTE["chroma"])
    bars3 = ax3.bar(throughput_labels, throughput_vals, color=tp_colors, width=0.5)
    for bar, v in zip(bars3, throughput_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f} q/s", ha="center", va="bottom", fontsize=8)
    ax3.set_title("Throughput (queries/sec)", fontsize=10)
    ax3.set_ylabel("q/s")
    ax3.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # 4. Jaccard overlap distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if has_chroma and "jaccard_overlap" in metrics.columns:
        jacc = metrics["jaccard_overlap"].dropna()
        ax4.hist(jacc, bins=8, color="#55A868", alpha=0.85, edgecolor="white")
        ax4.axvline(jacc.mean(), color="crimson", lw=1.5, linestyle="--",
                    label=f"Mean {jacc.mean():.2f}")
        ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, "ChromaDB\nnot available", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=9, color="grey")
    ax4.set_title("Result-Set Jaccard Overlap", fontsize=10)
    ax4.set_xlabel("Jaccard Similarity")
    ax4.set_ylabel("Queries")
    ax4.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # 5. Rank correlation distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if has_chroma and "spearman_rho" in metrics.columns:
        rho = metrics["spearman_rho"].dropna()
        ax5.hist(rho, bins=8, color="#C44E52", alpha=0.85, edgecolor="white")
        ax5.axvline(rho.mean(), color="navy", lw=1.5, linestyle="--",
                    label=f"Mean {rho.mean():.2f}")
        ax5.legend(fontsize=7)
        ax5.set_xlim(-1.1, 1.1)
    else:
        ax5.text(0.5, 0.5, "ChromaDB\nnot available", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=9, color="grey")
    ax5.set_title("Rank Correlation (Spearman)", fontsize=10)
    ax5.set_xlabel("Spearman Rho")
    ax5.set_ylabel("Queries")
    ax5.grid(axis="y", color=PALETTE["grid"], lw=0.7)

    # 6. Numeric summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    tfidf_row = [
        f"{metrics['tfidf_latency_ms'].mean():.2f} ms",
        f"{tps_tfidf:.1f}",
        f"{metrics['tfidf_top1_score'].mean():.4f}",
        f"{metrics['tfidf_mean_score'].mean():.4f}",
    ]
    table_data  = [tfidf_row]
    col_labels  = ["Latency", "Throughput\n(q/s)", "Top-1\nScore", "Mean\nScore"]
    row_labels  = ["TF-IDF"]

    if has_chroma:
        chroma_row = [
            f"{metrics['chroma_latency_ms'].mean():.2f} ms",
            f"{tps_chroma:.1f}",
            f"{metrics['chroma_top1_score'].mean():.4f}",
            f"{metrics['chroma_mean_score'].mean():.4f}",
        ]
        table_data.append(chroma_row)
        row_labels.append("ChromaDB")

    tbl = ax6.table(
        cellText=table_data,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1 and r > 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
    ax6.set_title("Summary Statistics", fontsize=10, pad=10)

    out = out_dir / "tfidf_chroma_summary_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved summary dashboard → %s", out)
    return out


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _print_summary(metrics: pd.DataFrame, tfidf_store: Any, chroma_store: Any) -> None:
    has_chroma = "chroma_latency_ms" in metrics.columns
    sep = "=" * 68

    print(f"\n{sep}")
    print("  LAPAS RETRIEVAL BENCHMARK: TF-IDF vs ChromaDB")
    print(sep)
    print(f"  Queries run   : {len(metrics)}")
    print(f"  Corpus size   : {tfidf_store.count()} chunks")
    print(f"  TF-IDF vocab  : {tfidf_store.vocab_size()} terms")
    if chroma_store is not None:
        print(f"  ChromaDB docs : {chroma_store.count()}")
    print()

    print("  LATENCY (ms)")
    print(f"  {'Metric':<30} {'TF-IDF':>10}  {'ChromaDB':>10}")
    print("  " + "-" * 55)
    for label, tkey, ckey in [
        ("Mean",   "tfidf_latency_ms", "chroma_latency_ms"),
        ("Median", "tfidf_latency_ms", "chroma_latency_ms"),
        ("Std",    "tfidf_latency_ms", "chroma_latency_ms"),
        ("Min",    "tfidf_latency_ms", "chroma_latency_ms"),
        ("Max",    "tfidf_latency_ms", "chroma_latency_ms"),
    ]:
        fn = {"Mean": np.mean, "Median": np.median, "Std": np.std,
              "Min": np.min, "Max": np.max}[label]
        t_val = fn(metrics[tkey].values)
        c_val = fn(metrics[ckey].values) if has_chroma and ckey in metrics else float("nan")
        print(f"  {label:<30} {t_val:>10.3f}  {c_val:>10.3f}")

    print()
    n_q      = len(metrics)
    tps_t    = n_q / (metrics["tfidf_latency_ms"].sum() / 1000)
    tps_c    = n_q / (metrics["chroma_latency_ms"].sum() / 1000) if has_chroma else float("nan")
    print(f"  {'Throughput (q/s)':<30} {tps_t:>10.1f}  {tps_c:>10.1f}")

    print()
    print("  SIMILARITY SCORES (higher = better match)")
    print(f"  {'Metric':<30} {'TF-IDF':>10}  {'ChromaDB':>10}")
    print("  " + "-" * 55)
    for label, tkey, ckey in [
        ("Mean Top-1 Score",   "tfidf_top1_score",  "chroma_top1_score"),
        ("Median Top-1 Score", "tfidf_top1_score",  "chroma_top1_score"),
        ("Mean Avg Score",     "tfidf_mean_score",  "chroma_mean_score"),
    ]:
        fn = {"Mean Top-1 Score": np.mean, "Median Top-1 Score": np.median,
              "Mean Avg Score": np.mean}[label]
        t_val = fn(metrics[tkey].dropna().values)
        c_val = fn(metrics[ckey].dropna().values) if has_chroma and ckey in metrics else float("nan")
        print(f"  {label:<30} {t_val:>10.4f}  {c_val:>10.4f}")

    if has_chroma:
        print()
        print("  AGREEMENT BETWEEN METHODS")
        print(f"  {'Metric':<30} {'Value':>10}")
        print("  " + "-" * 44)
        jacc = metrics["jaccard_overlap"].dropna()
        rho  = metrics["spearman_rho"].dropna()
        print(f"  {'Mean Jaccard Overlap':<30} {jacc.mean():>10.4f}")
        print(f"  {'Median Jaccard Overlap':<30} {jacc.median():>10.4f}")
        print(f"  {'Mean Spearman Rho':<30} {rho.mean():>10.4f}")
        print(f"  {'Queries with Rho > 0.5':<30} {(rho > 0.5).sum():>10d}")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--docs", default=str(_DOCS_DIR_DEF), metavar="DIR",
        help="Directory containing loan strategy documents (default: %(default)s).",
    )
    p.add_argument(
        "--n-results", type=int, default=5, metavar="K",
        help="Top-k documents to retrieve per query (default: %(default)s).",
    )
    p.add_argument(
        "--no-chroma", action="store_true",
        help="Skip ChromaDB and benchmark TF-IDF only (faster, no embeddings).",
    )
    p.add_argument(
        "--output-dir", default=str(_REPORTS_DIR), metavar="DIR",
        help="Directory for output charts (default: %(default)s).",
    )
    return p


def main() -> None:
    args       = _build_parser().parse_args()
    docs_dir   = Path(args.docs)
    out_dir    = Path(args.output_dir)
    n_results  = args.n_results
    run_chroma = not args.no_chroma

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build stores
    # ------------------------------------------------------------------
    tfidf_store  = _build_tfidf(docs_dir)
    chroma_store = _build_chroma(docs_dir) if run_chroma else None

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    log.info("Running %d queries against TF-IDF store …", len(TEST_QUERIES))
    tfidf_results, tfidf_latencies = _timed_query(tfidf_store, TEST_QUERIES, n_results)

    chroma_results  = None
    chroma_latencies = None
    if chroma_store is not None:
        log.info("Running %d queries against ChromaDB store …", len(TEST_QUERIES))
        chroma_results, chroma_latencies = _timed_query(chroma_store, TEST_QUERIES, n_results)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(tfidf_results, tfidf_latencies, chroma_results, chroma_latencies)

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    _print_summary(metrics, tfidf_store, chroma_store)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    log.info("Generating charts in %s …", out_dir)
    _save_latency_chart(metrics, out_dir)
    _save_score_distribution(metrics, out_dir)
    _save_overlap_heatmap(metrics, out_dir)
    _save_rank_correlation(metrics, out_dir)
    _save_summary_dashboard(metrics, out_dir)

    # ------------------------------------------------------------------
    # Persist metrics CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "tfidf_chroma_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    log.info("Metrics saved to %s", csv_path)

    log.success("Benchmark complete.  All outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
