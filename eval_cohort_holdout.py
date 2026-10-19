"""
Leave-one-cohort-out evaluation for diseases that span multiple studies.

For each (disease) with samples from >=2 cohorts:
  Restrict to (disease + Healthy) samples from those cohorts.
  For each cohort C: train on other cohorts, test on C — binary AUROC.
  Compare against random 5-fold CV on the same restricted data.

The gap between the two is the batch-effect inflation in the headline number.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess

warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
SEED = 42


def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000,
                                  class_weight="balanced", random_state=SEED)),
    ])


def auroc_binary(model, X_tr, y_tr, X_te, y_te) -> float:
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
    else:
        proba = model.decision_function(X_te)
    return roc_auc_score(y_te, proba)


def evaluate_disease(disease: str, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cohorts = meta.loc[meta.disease == disease, "dataset"].unique().tolist()
    print(f"\n===== {disease}  ({len(cohorts)} cohorts: {cohorts}) =====")

    # Restrict to (this disease + Healthy) samples from cohorts that have the disease
    keep = (meta.disease.isin([disease, "Healthy"])) & (meta.dataset.isin(cohorts))
    X_d = X.loc[keep]
    m_d = meta.loc[keep]
    y_d = (m_d.disease == disease).astype(int).values  # 1 = disease, 0 = healthy

    print(f"  samples: {len(X_d)}  (disease={y_d.sum()}, healthy={(y_d==0).sum()})")
    print("  per-cohort counts (disease/healthy):")
    for c in cohorts:
        in_c = m_d.dataset == c
        print(f"    {c}: {(y_d[in_c]==1).sum()}/{(y_d[in_c]==0).sum()}")

    rows = []

    # ---- Leave-one-cohort-out (the honest cross-cohort generalization test)
    for held in cohorts:
        train_mask = (m_d.dataset != held).values
        test_mask  = (m_d.dataset == held).values
        if y_d[test_mask].sum() == 0 or (y_d[test_mask] == 0).sum() == 0:
            print(f"  [LOCO {held}] skipped — only one class present in test fold")
            continue
        auc = auroc_binary(make_lr(),
                           X_d.iloc[train_mask], y_d[train_mask],
                           X_d.iloc[test_mask],  y_d[test_mask])
        rows.append({"split": "LOCO", "held_out": held,
                     "n_test": int(test_mask.sum()), "auroc": auc})
        print(f"  [LOCO held={held}] n_test={test_mask.sum()}  AUROC={auc:.3f}")

    # ---- Random 5-fold stratified CV on the same restricted data (the inflated number)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rand_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X_d, y_d), 1):
        auc = auroc_binary(make_lr(), X_d.iloc[tr], y_d[tr], X_d.iloc[te], y_d[te])
        rand_aucs.append(auc)
        rows.append({"split": "random_5fold", "held_out": f"fold_{fold}",
                     "n_test": len(te), "auroc": auc})
    print(f"  [random 5-fold]  AUROC = {np.mean(rand_aucs):.3f} ± {np.std(rand_aucs):.3f}")

    return pd.DataFrame(rows).assign(disease=disease)


def main():
    print("Loading data ...")
    abundance = pd.read_csv(DATA / "abundance.tsv", sep="\t", index_col=0)
    metadata  = pd.read_csv(DATA / "metadata.tsv",  sep="\t", index_col=0)
    abundance, metadata = abundance.align(metadata, join="inner", axis=0)

    print(f"Preprocessing {abundance.shape} ...")
    X = preprocess(abundance, min_prevalence=0.10, use_clr=True)
    print(f"  retained {X.shape[1]} taxa")

    all_results = []
    for disease in ["T2D", "IBD"]:
        all_results.append(evaluate_disease(disease, X, metadata))

    out = pd.concat(all_results, ignore_index=True)
    out.to_csv("results/cohort_holdout.csv", index=False)
    print("\nSaved results/cohort_holdout.csv")

    print("\n===== Summary =====")
    summary = out.groupby(["disease", "split"])["auroc"].agg(["mean", "std", "count"]).round(3)
    print(summary)


if __name__ == "__main__":
    main()
