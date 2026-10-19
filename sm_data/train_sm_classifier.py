"""
Train a binary classifier (SM vs Healthy) on the genus-level abundance table
produced by the DADA2 pipeline.

Cohort is small (22 SM + 9 Healthy), so:
  - Logistic Regression with L1 (sparse, interpretable) is the headline model.
  - Random Forest is a sanity check.
  - Stratified leave-one-out CV (LOOCV) — appropriate for n<40.
  - Report AUROC, balanced accuracy, top discriminative genera.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, classification_report,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
SEED = 42


def clr_transform(X: np.ndarray, pseudocount: float | None = None) -> np.ndarray:
    if pseudocount is None:
        nz = X[X > 0]
        pseudocount = nz.min() / 2 if nz.size else 1e-6
    X = X + pseudocount
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    return log_X - log_X.mean(axis=1, keepdims=True)


def main():
    genus = pd.read_csv(HERE / "out" / "genus_table.tsv", sep="\t", index_col=0)
    meta  = pd.read_csv(HERE / "samples.tsv", sep="\t")

    # Map run accession -> disease (samples.tsv has run_accession + disease)
    meta = meta[["run_accession", "disease", "age", "ethnicity"]].drop_duplicates("run_accession")
    meta = meta.set_index("run_accession")

    # Align: genus index is run_accession (e.g., SRR33413328)
    common = genus.index.intersection(meta.index)
    print(f"Aligned: {len(common)} samples (out of {len(genus)} genus rows, {len(meta)} meta rows)")
    X = genus.loc[common]
    y_meta = meta.loc[common]
    y = (y_meta["disease"] == "SM").astype(int).values

    print(f"Class balance: SM={y.sum()}  Healthy={(y==0).sum()}")
    print(f"Genus features: {X.shape[1]}")

    # ---- Light prevalence filter (>=10% of samples)
    keep = (X > 0).mean(axis=0) >= 0.10
    X = X.loc[:, keep]
    print(f"After prevalence filter: {X.shape[1]} genera")

    # ---- CLR transform
    X_clr = clr_transform(X.values.astype(float))

    # ---- Models
    models = {
        "LR-L1":   Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                                       class_weight="balanced", random_state=SEED))]),
        "LR-L2":   Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                                                       class_weight="balanced", random_state=SEED))]),
        "RF":      RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                          class_weight="balanced", n_jobs=-1, random_state=SEED),
    }

    # ---- Leave-one-out CV (appropriate for n=31)
    print("\n=== Leave-one-out CV ===")
    loo = LeaveOneOut()
    rows = []
    for name, model in models.items():
        y_true, y_pred, y_prob = [], [], []
        for tr, te in loo.split(X_clr):
            m = model
            m.fit(X_clr[tr], y[tr])
            y_true.append(y[te][0])
            y_pred.append(m.predict(X_clr[te])[0])
            if hasattr(m, "predict_proba"):
                y_prob.append(m.predict_proba(X_clr[te])[0, 1])
            else:
                y_prob.append(m.decision_function(X_clr[te])[0])
        y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
        rows.append({
            "model": name,
            "accuracy":          accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "auroc":             roc_auc_score(y_true, y_prob),
        })
        print(f"  {name:8s} acc={rows[-1]['accuracy']:.3f}  "
              f"bal_acc={rows[-1]['balanced_accuracy']:.3f}  AUROC={rows[-1]['auroc']:.3f}")

    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(HERE / "out" / "loo_metrics.csv")

    # ---- Top discriminative genera (RF on full data)
    print("\n=== Top 15 discriminative genera (RF importance, full data) ===")
    rf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X_clr, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(15).to_string())
    imp.head(50).to_csv(HERE / "out" / "top_genera.csv", header=["importance"])

    # ---- Final classification report on the LOO predictions of the best model
    best = res["balanced_accuracy"].idxmax()
    print(f"\n=== Best model: {best} ===")
    # rerun LOO for that model to get full y_true/y_pred for the report
    m = models[best]
    y_true, y_pred = [], []
    for tr, te in loo.split(X_clr):
        m.fit(X_clr[tr], y[tr])
        y_true.append(y[te][0]); y_pred.append(m.predict(X_clr[te])[0])
    print(classification_report(y_true, y_pred, target_names=["Healthy", "SM"], digits=3))

    print(f"\n✓ Wrote {HERE/'out'/'loo_metrics.csv'} and {HERE/'out'/'top_genera.csv'}")


if __name__ == "__main__":
    main()
