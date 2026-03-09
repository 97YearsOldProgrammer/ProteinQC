"""XGBoost combiner for Phase 2: train on extracted features, evaluate vs CaLM baseline.

Usage:
    # Fast first pass
    python -m proteinqc.cli.train_combiner \
        --data data/features/all_datasets.parquet \
        --output models/combiner/xgb_v1.json \
        --baseline-json data/results/benchmark_multispecies_longest_orf.json

    # With Optuna tuning (~30 min, 50 trials)
    python -m proteinqc.cli.train_combiner \
        --data data/features/all_datasets.parquet \
        --output models/combiner/xgb_v1.json \
        --tune --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Raw columns from parquet (protein_length dropped: r≈0.999 with longest_orf_codons)
_RAW_COLS = [
    "seq_length_bp", "gc_content", "longest_orf_codons",
    "num_orfs", "orf_fraction", "kozak_score",
    "codon_entropy", "codon_gini", "calm_score",
    "num_pfam_domains",
]

FEATURE_COLS = _RAW_COLS + ["log_pfam_evalue", "has_pfam"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_pfam_evalue and has_pfam; return copy."""
    df = df.copy()
    # -log10(evalue) when not null and > 0, else 0
    df["log_pfam_evalue"] = 0.0
    mask = df["best_pfam_evalue"].notna() & (df["best_pfam_evalue"] > 0)
    df.loc[mask, "log_pfam_evalue"] = -np.log10(df.loc[mask, "best_pfam_evalue"])
    df["has_pfam"] = (df["num_pfam_domains"].fillna(0) > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Dataset-level train/test split (prevents leakage across datasets)
# ---------------------------------------------------------------------------

def split_by_dataset(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    datasets = np.array(df["dataset"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(datasets)
    n_test = max(12, int(test_frac * len(datasets)))
    test_datasets = set(datasets[:n_test])
    return (
        df[~df["dataset"].isin(test_datasets)].copy(),
        df[df["dataset"].isin(test_datasets)].copy(),
        test_datasets,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    m: dict[str, float] = {}
    m["ACC"] = float(accuracy_score(y_true, y_pred) * 100)
    m["MCC"] = float(matthews_corrcoef(y_true, y_pred) * 100)
    m["F1"] = float(f1_score(y_true, y_pred, average="macro") * 100)
    try:
        m["AUC"] = float(roc_auc_score(y_true, y_prob) * 100)
    except ValueError:
        m["AUC"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------

def run_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, float]:
    print("\n[Baseline] LogisticRegressionCV (5-fold)...", flush=True)
    scaler = StandardScaler()
    # LR can't handle NaN — fill with 0 (XGBoost handles NaN natively)
    Xtr = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    Xte = scaler.transform(np.nan_to_num(X_test, nan=0.0))
    lr = LogisticRegressionCV(cv=5, max_iter=1000, n_jobs=-1)
    lr.fit(Xtr, y_train)
    preds = lr.predict(Xte)
    probs = lr.predict_proba(Xte)[:, 1]
    m = compute_metrics(y_test, preds, probs)
    print(
        f"  LR  ACC={m['ACC']:.2f}%  MCC={m['MCC']:.2f}"
        f"  F1={m['F1']:.2f}  AUC={m['AUC']:.2f}"
    )
    return m


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scale_pos_weight: float,
    params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    defaults: dict[str, Any] = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        device="cpu",
        random_state=42,
    )
    if params:
        defaults.update(params)
    model = xgb.XGBClassifier(**defaults)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    return model


# ---------------------------------------------------------------------------
# Optuna hyperparameter sweep
# ---------------------------------------------------------------------------

def tune_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scale_pos_weight: float,
    n_trials: int = 50,
) -> dict[str, Any]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
            "tree_method": "hist",
            "device": "cpu",
            "random_state": 42,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return float(accuracy_score(y_val, m.predict(X_val)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest trial: ACC={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_per_dataset(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    baseline_json: Path | None,
) -> pd.DataFrame:
    rows = []
    for dataset, group in test_df.groupby("dataset"):
        X = group[FEATURE_COLS].values
        y = group["label"].values
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        m = compute_metrics(y, preds, probs)
        m["dataset"] = dataset
        m["n"] = len(group)
        rows.append(m)

    res_df = pd.DataFrame(rows).sort_values("ACC")

    if baseline_json and baseline_json.exists():
        with open(baseline_json) as f:
            raw = json.load(f)
        entries = raw.get("per_dataset", raw) if isinstance(raw, dict) else raw
        baseline_map = {
            e["dataset"]: e.get("ACC") or e.get("mlp_cls", {}).get("ACC")
            for e in entries
        }
        res_df["calm_acc"] = res_df["dataset"].map(baseline_map)
        res_df["delta"] = res_df["ACC"] - res_df["calm_acc"].fillna(res_df["ACC"])

    return res_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train XGBoost combiner for Phase 2 coding/noncoding classification",
    )
    ap.add_argument("--data", required=True,
                    help="Parquet with extracted features (all_datasets.parquet)")
    ap.add_argument("--output", required=True,
                    help="Output XGBoost model path (e.g. models/combiner/xgb_v1.json)")
    ap.add_argument("--baseline-json",
                    help="Benchmark JSON for per-dataset CaLM comparison")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Fraction of datasets held out for test (default: 0.2)")
    ap.add_argument("--tune", action="store_true",
                    help="Run Optuna hyperparameter sweep before training")
    ap.add_argument("--n-trials", type=int, default=50,
                    help="Number of Optuna trials (default: 50)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load & engineer features ---
    print(f"Loading {args.data}...", flush=True)
    df = pd.read_parquet(args.data)
    df = engineer_features(df)
    print(f"  {len(df):,} rows, {df['dataset'].nunique()} datasets", flush=True)

    # --- Dataset-level split ---
    train_df, test_df, test_datasets = split_by_dataset(df, args.test_frac, args.seed)
    print(
        f"\nDataset split: {train_df['dataset'].nunique()} train / "
        f"{test_df['dataset'].nunique()} test"
    )
    print("Test datasets:")
    for ds in sorted(test_datasets):
        print(f"  - {ds}")

    X_all = train_df[FEATURE_COLS].values
    y_all = train_df["label"].values

    # 10% of training data as XGB validation (early stopping)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.1)
    X_val, y_val = X_all[idx[:n_val]], y_all[idx[:n_val]]
    X_tr, y_tr = X_all[idx[n_val:]], y_all[idx[n_val:]]

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(
        f"\nClass balance: {n_pos:,} coding / {n_neg:,} noncoding"
        f"  (scale_pos_weight={scale_pos_weight:.3f})",
        flush=True,
    )

    # --- Logistic regression baseline ---
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    run_baseline(X_tr, y_tr, X_test, y_test)

    # --- Optional Optuna sweep ---
    best_params: dict[str, Any] | None = None
    if args.tune:
        print(f"\n[Tuning] Running {args.n_trials} Optuna trials...", flush=True)
        best_params = tune_xgb(X_tr, y_tr, X_val, y_val, scale_pos_weight, args.n_trials)
        params_path = output_path.parent / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved best params: {params_path}", flush=True)

    # --- Train XGBoost ---
    print("\n[XGBoost] Training...", flush=True)
    model = train_xgb(X_tr, y_tr, X_val, y_val, scale_pos_weight, best_params)

    # --- Overall test metrics ---
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    overall = compute_metrics(y_test, preds, probs)
    print(
        f"\n[Overall test]  ACC={overall['ACC']:.2f}%  MCC={overall['MCC']:.2f}"
        f"  F1={overall['F1']:.2f}  AUC={overall['AUC']:.2f}",
        flush=True,
    )

    # --- Per-dataset evaluation ---
    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    res_df = evaluate_per_dataset(model, test_df, baseline_path)
    print("\n[Per-dataset results]")
    print(res_df.to_string(index=False))

    if "calm_acc" in res_df.columns:
        mean_xgb = res_df["ACC"].mean()
        mean_calm = res_df["calm_acc"].dropna().mean()
        mean_delta = res_df["delta"].mean()
        print(f"\nMean XGBoost ACC : {mean_xgb:.2f}%")
        print(f"Mean CaLM ACC    : {mean_calm:.2f}%")
        print(f"Mean delta       : {mean_delta:+.2f}%")

    # --- Save ---
    model.save_model(str(output_path))

    feat_path = output_path.parent / "feature_cols.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    results_path = output_path.parent / "test_results.json"
    res_df.to_json(results_path, orient="records", indent=2)

    print(f"\nSaved model    : {output_path}")
    print(f"Saved features : {feat_path}")
    print(f"Saved results  : {results_path}")


if __name__ == "__main__":
    main()
