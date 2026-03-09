"""Visualization for XGBoost combiner: SHAP, per-dataset accuracy, ROC, error analysis.

Usage:
    python -m proteinqc.cli.visualize_combiner \
        --model models/combiner/xgb_v2.json \
        --data data/features/all_datasets.parquet \
        --baseline-json data/results/benchmark_zeroshot.json \
        --output-dir data/results/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_curve, auc

from proteinqc.cli.train_combiner import FEATURE_COLS, engineer_features


# ---------------------------------------------------------------------------
# Species grouping
# ---------------------------------------------------------------------------

_SPECIES_KEYWORDS: dict[str, list[str]] = {
    "Human": ["Human", "human"],
    "Mouse": ["Mouse", "mouse"],
    "Plant": [
        "Rice", "Arabidopsis", "Tomato", "Maize", "Wheat", "Grape", "Soybean",
        "Amborella", "Chlamydomonas", "Brachypodium", "Cassava", "Citrus",
        "Ricinus", "Solanum", "Sorghum", "Selaginella", "Physcomitrella",
        "Potato", "Monocot", "Dicot",
    ],
    "Fungal": ["S. cerevisiae", "cerevisiae"],
    "Vertebrate": [
        "Cow", "Rat", "Chicken", "Zebrafish", "Fruitfly", "Fruit fly",
        "C. elegans", "Mammals", "Vertebrates", "Invertebrates",
    ],
}


def _classify_species(dataset: str) -> str:
    for group, keywords in _SPECIES_KEYWORDS.items():
        if any(kw.lower() in dataset.lower() for kw in keywords):
            return group
    return "Other"


def _load_baseline(path: Path) -> dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    return {
        f"{e['tool']}/{e['species']}": e.get("mlp_cls", {}).get("ACC")
        for e in data
    }


# ---------------------------------------------------------------------------
# Plot 1: Per-dataset accuracy comparison (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_per_dataset(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    baseline_map: dict[str, float] | None,
    out_dir: Path,
) -> None:
    datasets = sorted(df["dataset"].unique())
    xgb_acc, calm_acc, is_short = [], [], []

    for ds in datasets:
        group = df[df["dataset"] == ds]
        preds = model.predict(group[FEATURE_COLS].values)
        xgb_acc.append(accuracy_score(group["label"].values, preds) * 100)
        calm_acc.append(baseline_map.get(ds) if baseline_map else None)
        is_short.append("short" in ds.lower() or "sORF" in ds)

    # Sort by CaLM accuracy ascending (worst first)
    order = sorted(range(len(datasets)),
                   key=lambda i: calm_acc[i] if calm_acc[i] is not None else xgb_acc[i])
    datasets = [datasets[i] for i in order]
    xgb_acc = [xgb_acc[i] for i in order]
    calm_acc = [calm_acc[i] for i in order]
    is_short = [is_short[i] for i in order]

    n = len(datasets)
    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.28)))
    y = np.arange(n)

    xgb_bars = ax.barh(y + 0.18, xgb_acc, height=0.34, color="#E87C28",
                        label="XGBoost", zorder=2)
    calm_filled = [c if c is not None else 0 for c in calm_acc]
    calm_bars = ax.barh(y - 0.18, calm_filled, height=0.34, color="#3A7FC1",
                         label="CaLM baseline", zorder=2)

    for i, short in enumerate(is_short):
        if short:
            xgb_bars[i].set_edgecolor("red")
            xgb_bars[i].set_linewidth(1.5)
            calm_bars[i].set_edgecolor("red")
            calm_bars[i].set_linewidth(1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-dataset accuracy: XGBoost vs CaLM baseline"
                 "\n(red outline = short-seq datasets)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    ax.axvline(90.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    out = out_dir / "per_dataset_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: SHAP summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    out_dir: Path,
) -> np.ndarray:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    out = out_dir / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return shap_values


# ---------------------------------------------------------------------------
# Plot 3: SHAP dependence plots (2 key interactions)
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    out_dir: Path,
) -> None:
    import shap
    pairs = [
        ("calm_score", "longest_orf_codons", "shap_dependence_calm_orf.png"),
        ("num_pfam_domains", "orf_fraction", "shap_dependence_pfam_fraction.png"),
    ]
    for feat, interact, fname in pairs:
        if feat not in X_sample.columns or interact not in X_sample.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, X_sample,
                             interaction_index=interact, ax=ax, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / fname}")


# ---------------------------------------------------------------------------
# Plot 4: ROC by species group
# ---------------------------------------------------------------------------

def plot_roc_by_species(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    df = df.copy()
    df["species_group"] = df["dataset"].apply(_classify_species)
    groups = sorted(df["species_group"].unique())
    n = len(groups)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, group in enumerate(groups):
        ax = axes_flat[idx]
        sub = df[df["species_group"] == group]
        y = sub["label"].values
        if len(np.unique(y)) < 2:
            ax.set_title(f"{group} (single class)")
            continue
        probs = model.predict_proba(sub[FEATURE_COLS].values)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_title(f"{group} (n={len(sub):,})")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("ROC Curves by Species Group", fontsize=13)
    plt.tight_layout()
    out = out_dir / "roc_by_species.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5: Error analysis violin plots
# ---------------------------------------------------------------------------

def plot_error_analysis(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    import seaborn as sns

    X = df[FEATURE_COLS].values
    y = df["label"].values
    preds = model.predict(X)

    df = df.copy()
    conditions = [
        (y == 1) & (preds == 1),
        (y == 0) & (preds == 1),
        (y == 0) & (preds == 0),
        (y == 1) & (preds == 0),
    ]
    labels = ["TP", "FP", "TN", "FN"]
    df["error_type"] = np.select(conditions, labels, default="TN")

    top_feats = ["calm_score", "longest_orf_codons", "orf_fraction",
                 "num_pfam_domains", "seq_length_bp", "codon_entropy"]
    top_feats = [f for f in top_feats if f in df.columns]

    # Subsample for speed
    sample = df.sample(min(20_000, len(df)), random_state=42)
    palette = {"TP": "#2ecc71", "FP": "#e74c3c", "TN": "#3498db", "FN": "#e67e22"}

    fig, axes = plt.subplots(1, len(top_feats),
                              figsize=(3.5 * len(top_feats), 5))
    for ax, feat in zip(axes, top_feats):
        sns.violinplot(data=sample, x="error_type", y=feat, hue="error_type",
                       order=labels, hue_order=labels,
                       palette=palette, ax=ax, inner="box", cut=0,
                       legend=False)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Feature distributions by prediction outcome (TP/FP/TN/FN)",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "error_analysis_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize XGBoost combiner results",
    )
    ap.add_argument("--model", required=True, help="XGBoost model path (.json)")
    ap.add_argument("--data", required=True, help="Parquet with extracted features")
    ap.add_argument("--baseline-json",
                    help="Benchmark JSON for CaLM baseline comparison")
    ap.add_argument("--output-dir", default="data/results/figures",
                    help="Directory for output plots")
    ap.add_argument("--shap-sample", type=int, default=5000,
                    help="Random samples for SHAP plots (default: 5000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}", flush=True)
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    print(f"Loading data: {args.data}", flush=True)
    df = pd.read_parquet(args.data)
    df = engineer_features(df)
    print(f"  {len(df):,} rows, {df['dataset'].nunique()} datasets", flush=True)

    baseline_map = None
    if args.baseline_json:
        baseline_map = _load_baseline(Path(args.baseline_json))
        print(f"Loaded baseline for {len(baseline_map)} datasets", flush=True)

    sample_df = df.sample(min(args.shap_sample, len(df)), random_state=args.seed)
    X_sample = sample_df[FEATURE_COLS]

    print("\n--- Plot 1: Per-dataset accuracy ---")
    plot_per_dataset(model, df, baseline_map, out_dir)

    print("\n--- Plot 2: SHAP summary ---")
    shap_values = plot_shap_summary(model, X_sample, out_dir)

    print("\n--- Plot 3: SHAP dependence ---")
    plot_shap_dependence(shap_values, X_sample, out_dir)

    print("\n--- Plot 4: ROC by species group ---")
    plot_roc_by_species(model, df, out_dir)

    print("\n--- Plot 5: Error analysis ---")
    plot_error_analysis(model, df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
