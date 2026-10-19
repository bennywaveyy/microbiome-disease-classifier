"""
Main training script: stratified K-fold cross-validation across all models,
plus feature-importance extraction for the best classic-ML model.

Usage
-----
Synthetic demo (default):
    python train.py

With your own data:
    python train.py --abundance abundance.tsv --metadata metadata.tsv \
        --label-col disease --sample-id-col sample_id

Outputs:
    results/cv_metrics.csv          per-fold metrics for every model
    results/cv_summary.csv          mean ± std summary
    results/confusion_<model>.csv   confusion matrix on held-out folds
    results/top_features.csv        top discriminative taxa (Random Forest)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from data_utils import generate_synthetic_microbiome, load_abundance_table
from models import build_models
from preprocessing import preprocess

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
def evaluate_fold(model, X_train, y_train, X_test, y_test, classes) -> dict:
    """Train on (X_train, y_train), evaluate on (X_test, y_test), return metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    # Multi-class AUROC (one-vs-rest, macro-averaged) requires probabilities.
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            lb = LabelBinarizer().fit(classes)
            y_test_bin = lb.transform(y_test)
            if y_test_bin.shape[1] == 1:  # binary edge case
                y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
            metrics["auroc_ovr_macro"] = roc_auc_score(
                y_test_bin, proba, average="macro", multi_class="ovr"
            )
        except Exception:
            metrics["auroc_ovr_macro"] = np.nan
    else:
        metrics["auroc_ovr_macro"] = np.nan

    return metrics, y_pred


# --------------------------------------------------------------------------- #
def run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42):
    classes = np.sort(y.unique())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = build_models(random_state=seed)

    all_metrics = []
    all_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in models}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
        print(f"\n{'='*60}\nFold {fold}/{n_splits}  |  train={len(train_idx)}, test={len(test_idx)}\n{'='*60}")

        for name, model in models.items():
            print(f"  [{name}]", end=" ", flush=True)
            # Fresh estimator per fold (clone via re-instantiation through the factory)
            fresh_model = build_models(random_state=seed)[name]
            metrics, y_pred = evaluate_fold(fresh_model, X_train, y_train, X_test, y_test, classes)
            metrics.update({"model": name, "fold": fold})
            all_metrics.append(metrics)
            all_predictions[name].append((y_test, y_pred))
            print(
                f"acc={metrics['accuracy']:.3f}  "
                f"bal_acc={metrics['balanced_accuracy']:.3f}  "
                f"f1m={metrics['f1_macro']:.3f}  "
                f"auroc={metrics['auroc_ovr_macro']:.3f}"
            )

    return pd.DataFrame(all_metrics), all_predictions, classes


# --------------------------------------------------------------------------- #
def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auroc_ovr_macro"]
    summary = (
        metrics_df.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    return summary


def aggregated_confusion(predictions: list[tuple[np.ndarray, np.ndarray]], classes) -> pd.DataFrame:
    y_true = np.concatenate([p[0] for p in predictions])
    y_pred = np.concatenate([p[1] for p in predictions])
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])


def top_features_random_forest(X: pd.DataFrame, y: pd.Series, top_k: int = 25, seed: int = 42) -> pd.DataFrame:
    """Train a Random Forest on the full dataset to rank discriminative taxa."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=seed
    )
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_k).rename("importance").to_frame()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Multi-class disease prediction from gut microbiome.")
    parser.add_argument("--abundance", type=str, default=None, help="Path to abundance table (TSV/CSV).")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata file (TSV/CSV).")
    parser.add_argument("--label-col", type=str, default="disease")
    parser.add_argument("--sample-id-col", type=str, default="sample_id")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-prevalence", type=float, default=0.10)
    parser.add_argument("--no-clr", action="store_true", help="Skip CLR transform.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    if args.abundance and args.metadata:
        print(f"Loading abundance from {args.abundance}")
        print(f"Loading metadata  from {args.metadata}")
        X, y = load_abundance_table(
            args.abundance, args.metadata,
            label_column=args.label_col, sample_id_column=args.sample_id_col,
        )
    else:
        print("No data files provided — generating synthetic microbiome dataset.")
        X, y = generate_synthetic_microbiome(seed=args.seed)

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} taxa")
    print(f"Class distribution:\n{y.value_counts().to_string()}")

    # ---- Preprocess
    print(f"\nPreprocessing  (min_prevalence={args.min_prevalence}, CLR={not args.no_clr}) ...")
    X_proc = preprocess(X, min_prevalence=args.min_prevalence, use_clr=not args.no_clr)
    print(f"After preprocessing: {X_proc.shape[1]} taxa retained.")

    # ---- Cross-validation
    metrics_df, predictions, classes = run_cross_validation(
        X_proc, y, n_splits=args.n_splits, seed=args.seed
    )
    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)

    # ---- Summary
    summary = summarize(metrics_df)
    summary.to_csv(out_dir / "cv_summary.csv")
    print(f"\n{'='*60}\nCross-validation summary (mean ± std across folds)\n{'='*60}")
    print(summary)

    # ---- Per-model confusion matrices
    for name, preds in predictions.items():
        cm = aggregated_confusion(preds, classes)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        cm.to_csv(out_dir / f"confusion_{safe}.csv")

    # ---- Best model classification report
    best_model = summary[("f1_macro", "mean")].idxmax()
    print(f"\nBest model by macro-F1: {best_model}")
    y_true = np.concatenate([p[0] for p in predictions[best_model]])
    y_pred = np.concatenate([p[1] for p in predictions[best_model]])
    print("\nClassification report (out-of-fold predictions, best model):")
    print(classification_report(y_true, y_pred, digits=3))

    # ---- Feature importance
    print("\nComputing top discriminative taxa (Random Forest, full data)...")
    top_taxa = top_features_random_forest(X_proc, y, top_k=25, seed=args.seed)
    top_taxa.to_csv(out_dir / "top_features.csv")
    print(top_taxa.head(15))

    print(f"\n✓ All results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
