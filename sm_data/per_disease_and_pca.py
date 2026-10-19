"""
Two clean follow-ups before any fancier modeling:

(1) Per-disease BINARY classifiers — train FD-vs-Healthy and SM-vs-Healthy
    independently, each on its own cohort. No cross-cohort confound.
    Headline metric: per-disease AUROC.

(2) PCA + simple variance decomposition — visual proof that "cohort" is
    a stronger signal than "disease" in the combined data. This is the
    figure that makes the cohort-confound finding inarguable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT  = HERE / "combined_out"
OUT.mkdir(exist_ok=True)
SEED = 42


def clr(X, pseudocount=None):
    if pseudocount is None:
        nz = X[X > 0]; pseudocount = nz.min() / 2 if nz.size else 1e-6
    X = X + pseudocount
    X = X / X.sum(axis=1, keepdims=True)
    return np.log(X) - np.log(X).mean(axis=1, keepdims=True)


def main():
    X_full = pd.read_csv(OUT / "combined_genus.tsv", sep="\t", index_col=0)
    meta   = pd.read_csv(OUT / "combined_meta.tsv",  sep="\t", index_col=0)

    rel = X_full.div(X_full.sum(axis=1), axis=0)
    keep = (rel > 0).mean(axis=0) >= 0.10
    rel = rel.loc[:, keep]
    X = clr(rel.values.astype(float))
    print(f"Combined CLR matrix: {X.shape}")

    # ------------------------------------------------------------------ #
    # (1) Per-disease binary classifiers
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("(1) Per-disease binary classifiers (within-cohort)")
    print("="*60)

    binary_results = []
    for cohort_name, disease_label in [("FD", "FD"), ("SM", "SM")]:
        mask = (meta["cohort"] == cohort_name).values
        X_c = X[mask]
        y_c = (meta.loc[mask, "disease"] == disease_label).astype(int).values
        n_pos, n_neg = y_c.sum(), (y_c == 0).sum()
        print(f"\n--- {disease_label} vs Healthy  (cohort={cohort_name}, n={mask.sum()}: "
              f"{n_pos} {disease_label}, {n_neg} Healthy) ---")

        models = {
            "LR-L1": Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                                         class_weight="balanced", random_state=SEED))]),
            "LR-L2": Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                                         class_weight="balanced", random_state=SEED))]),
            "RF":    RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=SEED),
        }

        n_splits = min(5, n_pos, n_neg)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for mname, model in models.items():
            from sklearn.base import clone
            aucs, baccs = [], []
            for tr, te in skf.split(X_c, y_c):
                m = clone(model); m.fit(X_c[tr], y_c[tr])
                proba = m.predict_proba(X_c[te])[:, 1]
                pred  = m.predict(X_c[te])
                aucs.append(roc_auc_score(y_c[te], proba))
                baccs.append(balanced_accuracy_score(y_c[te], pred))
            row = {
                "disease": disease_label, "cohort": cohort_name, "model": mname,
                "n_total": int(mask.sum()), "n_pos": int(n_pos), "n_neg": int(n_neg),
                "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
                "bacc_mean": float(np.mean(baccs)), "bacc_std": float(np.std(baccs)),
            }
            binary_results.append(row)
            print(f"  {mname:7s}  AUROC={row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}   "
                  f"bal_acc={row['bacc_mean']:.3f} ± {row['bacc_std']:.3f}")

    pd.DataFrame(binary_results).to_csv(OUT / "per_disease_binary.csv", index=False)

    # ------------------------------------------------------------------ #
    # (2) PCA — visualize cohort vs disease as the dominant axis
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("(2) PCA: which separates samples better — cohort or disease?")
    print("="*60)

    pca = PCA(n_components=4, random_state=SEED)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    var = pca.explained_variance_ratio_
    print(f"PC variance explained: {(var*100).round(1)} %  (cumulative {var.cumsum()[-1]*100:.1f}%)")

    cohort_arr  = meta["cohort"].values
    disease_arr = meta["disease"].values

    # Very simple variance-decomposition: % of variance in PC1+PC2 explained by group label
    def explained_by(labels, Zk):
        labels = np.asarray(labels)
        total = np.sum((Zk - Zk.mean(0))**2)
        within = 0.0
        for g in np.unique(labels):
            sub = Zk[labels == g]
            within += np.sum((sub - sub.mean(0))**2)
        return 1.0 - within/total

    print(f"PC1+PC2 variance explained by cohort label : {explained_by(cohort_arr,  Z[:, :2]):.1%}")
    print(f"PC1+PC2 variance explained by disease label: {explained_by(disease_arr, Z[:, :2]):.1%}")

    # ---- Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cohort_colors  = {"FD": "tab:blue", "SM": "tab:orange"}
    disease_colors = {"Healthy": "tab:green", "FD": "tab:blue", "SM": "tab:red"}

    for label, color in cohort_colors.items():
        mask = (cohort_arr == label)
        axes[0].scatter(Z[mask, 0], Z[mask, 1], s=30, alpha=0.65, label=f"{label} (n={mask.sum()})",
                        edgecolors="white", linewidths=0.5, color=color)
    axes[0].set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); axes[0].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[0].set_title(f"Colored by COHORT  (variance explained: {explained_by(cohort_arr, Z[:, :2]):.0%})")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for label, color in disease_colors.items():
        mask = (disease_arr == label)
        axes[1].scatter(Z[mask, 0], Z[mask, 1], s=30, alpha=0.65, label=f"{label} (n={mask.sum()})",
                        edgecolors="white", linewidths=0.5, color=color)
    axes[1].set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[1].set_title(f"Colored by DISEASE (variance explained: {explained_by(disease_arr, Z[:, :2]):.0%})")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Cohort effect dominates disease effect on combined SM+FD genus profiles",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = OUT / "pca_cohort_vs_disease.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {fig_path}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
