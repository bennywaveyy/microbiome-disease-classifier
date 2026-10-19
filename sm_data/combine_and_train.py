"""
Combine SM and FD genus tables -> multi-class classifier with cohort-aware CV.

Three evaluation modes:
  (1) Random stratified 5-fold CV (the OPTIMISTIC headline number)
  (2) Leave-one-cohort-out (honest cross-study generalization)
  (3) Per-cohort within-CV (sanity baseline)

Models: LR-L1, LR-L2, RF, MLP. Multi-class (Healthy / SM / FD).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
SEED = 42
MIN_PREVALENCE = 0.10


# --------------------------------------------------------------------------- #
def clr_transform(X: np.ndarray, pseudocount: float | None = None) -> np.ndarray:
    if pseudocount is None:
        nz = X[X > 0]
        pseudocount = nz.min() / 2 if nz.size else 1e-6
    X = X + pseudocount
    X = X / X.sum(axis=1, keepdims=True)
    return np.log(X) - np.log(X).mean(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(128, 64), dropout=0.4):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(128, 64), dropout=0.4, lr=1e-3, weight_decay=1e-4,
                 epochs=80, batch_size=32, random_state=SEED):
        self.hidden, self.dropout = hidden, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size = epochs, batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        self._le = LabelEncoder(); ye = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._model = _MLP(Xs.shape[1], len(self.classes_), tuple(self.hidden), self.dropout)
        Xt, yt = torch.from_numpy(Xs), torch.from_numpy(ye).long()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                             batch_size=self.batch_size, shuffle=True)
        cw = np.bincount(ye); weights = torch.tensor(cw.sum()/(len(cw)*cw), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(self._model(xb), yb); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        self._model.eval()
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            return torch.softmax(self._model(torch.from_numpy(Xs)), dim=1).numpy()

    def predict(self, X):
        return self._le.inverse_transform(self.predict_proba(X).argmax(axis=1))


# --------------------------------------------------------------------------- #
def build_models():
    return {
        "LR-L1": Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(penalty="l1", C=1.0, solver="saga",
                                                     max_iter=5000, class_weight="balanced",
                                                     random_state=SEED))]),
        "LR-L2": Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                     max_iter=2000, class_weight="balanced",
                                                     random_state=SEED))]),
        "RF":    RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                        class_weight="balanced", n_jobs=-1, random_state=SEED),
        "MLP":   MLPClassifier(),
    }


def multiclass_auroc(model, X_te, y_te, all_classes):
    """Macro one-vs-rest AUROC."""
    if not hasattr(model, "predict_proba"):
        return np.nan
    proba = model.predict_proba(X_te)
    # Align proba columns to all_classes order
    model_classes = list(model.classes_)
    proba_aligned = np.zeros((len(y_te), len(all_classes)))
    for i, c in enumerate(all_classes):
        if c in model_classes:
            proba_aligned[:, i] = proba[:, model_classes.index(c)]
    lb = LabelBinarizer().fit(all_classes)
    y_bin = lb.transform(y_te)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    # Drop columns where the test fold has no positives or no negatives (AUROC undefined)
    keep = (y_bin.sum(axis=0) > 0) & (y_bin.sum(axis=0) < y_bin.shape[0])
    if keep.sum() < 2:
        return np.nan
    return roc_auc_score(y_bin[:, keep], proba_aligned[:, keep], average="macro")


def evaluate_split(name_models, splits, X, y, classes, label):
    from sklearn.base import clone
    rows = []
    for split_name, splitter in splits.items():
        for fold_idx, (tr, te) in enumerate(splitter, 1):
            for mname, model in name_models.items():
                m = clone(model)
                m.fit(X[tr], y[tr])
                y_pred = m.predict(X[te])
                rows.append({
                    "split": split_name, "fold": fold_idx, "model": mname,
                    "n_train": len(tr), "n_test": len(te),
                    "accuracy":          accuracy_score(y[te], y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y[te], y_pred),
                    "auroc_ovr_macro":   multiclass_auroc(m, X[te], y[te], classes),
                })
    df = pd.DataFrame(rows)
    out_path = HERE / "combined_out" / f"metrics_{label}.csv"
    df.to_csv(out_path, index=False)
    return df


# --------------------------------------------------------------------------- #
def main():
    out_dir = HERE / "combined_out"; out_dir.mkdir(exist_ok=True)

    # ---- Load both genus tables
    sm_g  = pd.read_csv(HERE / "out"    / "genus_table.tsv", sep="\t", index_col=0)
    fd_g  = pd.read_csv(HERE / "fd_out" / "genus_table.tsv", sep="\t", index_col=0)
    print(f"SM: {sm_g.shape}   FD: {fd_g.shape}")

    # ---- Build per-cohort metadata (sample_id -> disease, cohort)
    sm_meta = pd.read_csv(HERE / "samples.tsv", sep="\t").drop_duplicates("run_accession")
    sm_meta = sm_meta.set_index("run_accession")
    sm_meta["cohort"] = "SM"

    fd_runs = pd.read_csv(HERE / "fd_runs.tsv", sep="\t")
    fd_bs   = pd.read_csv(HERE / "fd_biosample_meta.tsv", sep="\t")
    def map_disease(g):
        if pd.isna(g) or g == "NA": return None
        g = str(g).strip()
        return "Healthy" if g.lower() in ("relative", "control", "healthy", "unaffected") else "FD"
    fd_bs["disease"] = fd_bs["host_genotype"].apply(map_disease)
    fd_meta = fd_runs.merge(fd_bs.rename(columns={"biosample":"sample_accession"}),
                            on="sample_accession").set_index("run_accession")
    fd_meta["cohort"] = "FD"

    # ---- Combine: union of genera (fill missing with 0)
    all_genera = sorted(set(sm_g.columns) | set(fd_g.columns))
    print(f"Combined genera (union): {len(all_genera)}")
    sm_a = sm_g.reindex(columns=all_genera, fill_value=0)
    fd_a = fd_g.reindex(columns=all_genera, fill_value=0)
    combined = pd.concat([sm_a, fd_a])
    combined.index.name = "sample_id"
    print(f"Combined table: {combined.shape}")

    # ---- Combined metadata
    meta_combined = pd.concat([
        sm_meta[["disease", "cohort"]],
        fd_meta[["disease", "cohort"]],
    ])
    aligned = combined.index.intersection(meta_combined.index)
    combined = combined.loc[aligned]
    meta_combined = meta_combined.loc[aligned]
    print(f"After alignment: {combined.shape[0]} samples")
    print(meta_combined.groupby(["cohort", "disease"]).size())
    combined.to_csv(out_dir / "combined_genus.tsv", sep="\t")
    meta_combined.to_csv(out_dir / "combined_meta.tsv", sep="\t")

    # ---- Convert counts -> relative abundance, then prevalence filter, then CLR
    rel = combined.div(combined.sum(axis=1), axis=0)
    keep = (rel > 0).mean(axis=0) >= MIN_PREVALENCE
    rel = rel.loc[:, keep]
    print(f"After prevalence filter: {rel.shape[1]} genera")
    X = clr_transform(rel.values.astype(float))
    y = meta_combined["disease"].values
    cohort = meta_combined["cohort"].values
    classes = np.array(sorted(np.unique(y)))
    print(f"Classes: {classes}")

    models = build_models()

    # ---- (1) Random 5-fold stratified CV (optimistic)
    print("\n========== Random 5-fold stratified CV (optimistic) ==========")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits_random = {"random": list(skf.split(X, y))}
    df_random = evaluate_split(models, splits_random, X, y, classes, "random")
    print(df_random.groupby("model")[["accuracy", "balanced_accuracy", "auroc_ovr_macro"]]
          .agg(["mean", "std"]).round(3))

    # ---- (2) Leave-one-cohort-out (honest)
    print("\n========== Leave-one-cohort-out (honest) ==========")
    logo = LeaveOneGroupOut()
    splits_loco = {"LOCO": list(logo.split(X, y, groups=cohort))}
    df_loco = evaluate_split(models, splits_loco, X, y, classes, "loco")
    print(df_loco.assign(held_out_cohort=lambda d: [cohort[te[0]]
            for te in [list(logo.split(X, y, groups=cohort))[i-1][1] for i in d['fold']]]))

    # ---- (3) Within-cohort 5-fold (sanity baseline)
    print("\n========== Within-cohort 5-fold (sanity baseline) ==========")
    for c in np.unique(cohort):
        mask = (cohort == c)
        X_c, y_c = X[mask], y[mask]
        if len(np.unique(y_c)) < 2:
            print(f"  Skipping {c}: only one class present.")
            continue
        # Smaller k if small cohort
        n_splits = min(5, np.bincount(LabelEncoder().fit_transform(y_c)).min())
        if n_splits < 2:
            print(f"  Skipping {c}: too few samples per class for CV.")
            continue
        skf_c = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        df_c = evaluate_split(models, {"within_"+c: list(skf_c.split(X_c, y_c))},
                              X_c, y_c, np.unique(y_c), f"within_{c}")
        print(f"  --- Cohort {c} (n={mask.sum()}) ---")
        print(df_c.groupby("model")[["accuracy", "balanced_accuracy", "auroc_ovr_macro"]]
              .agg(["mean", "std"]).round(3))

    print(f"\n✓ All metrics saved to {out_dir}/")


if __name__ == "__main__":
    main()
