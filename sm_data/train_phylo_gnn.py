"""
Phylogenetic tree neural network on the combined SM + FD genus table.

Architecture (GIN-flavored, but on a fixed tree so we use plain torch):
  - Per-genus embedding (each leaf has 1 input feature: CLR abundance)
  - Sum-aggregate genera -> family embeddings, then MLP
  - Sum-aggregate families -> phylum embeddings, then MLP
  - Concat (genus, family, phylum) embeddings -> classifier MLP
  - Sum is the GIN-correct aggregator (max/mean lose information per Xu+19)

Why no PyTorch Geometric: the phylogenetic tree is the SAME for every sample
(only node features change per patient), so aggregation reduces to a fixed
sparse matmul. PyG would add a 1+GB install for no architectural benefit.

Evaluation matches combine_and_train.py exactly: random 5-fold, LOCO,
within-cohort. Reports AUROC alongside the LR/RF/MLP baseline.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT  = HERE / "combined_out"
SEED = 42
MIN_PREV = 0.10


# --------------------------------------------------------------------------- #
def clr(X, pseudocount=None):
    if pseudocount is None:
        nz = X[X > 0]; pseudocount = nz.min() / 2 if nz.size else 1e-6
    X = X + pseudocount
    X = X / X.sum(axis=1, keepdims=True)
    return np.log(X) - np.log(X).mean(axis=1, keepdims=True)


def build_tree_aggregators(genus_columns: list[str]):
    """
    Parse 'p__Firmicutes|f__Ruminococcaceae|g__Faecalibacterium' style names
    into a tree, returning sparse (binary) aggregation matrices:
       A_g2f : (n_families, n_genera) — sums genera within each family
       A_f2p : (n_phyla,    n_families)
    Plus the family/phylum label vectors for readability.
    """
    parsed = [c.split("|") for c in genus_columns]
    phyla    = [p[0] for p in parsed]
    families = [p[1] for p in parsed]

    fam_unique = sorted(set(families))
    fam_idx = {f: i for i, f in enumerate(fam_unique)}
    A_g2f = np.zeros((len(fam_unique), len(genus_columns)), dtype=np.float32)
    for j, f in enumerate(families):
        A_g2f[fam_idx[f], j] = 1.0

    # Map families -> phyla (each family has a single parent phylum in this dataset)
    fam_to_phy = {}
    for p, f in zip(phyla, families):
        fam_to_phy[f] = p
    phy_unique = sorted(set(phyla))
    phy_idx = {p: i for i, p in enumerate(phy_unique)}
    A_f2p = np.zeros((len(phy_unique), len(fam_unique)), dtype=np.float32)
    for f, i in fam_idx.items():
        A_f2p[phy_idx[fam_to_phy[f]], i] = 1.0

    print(f"  tree built: {len(genus_columns)} genera -> "
          f"{len(fam_unique)} families -> {len(phy_unique)} phyla")
    return torch.from_numpy(A_g2f), torch.from_numpy(A_f2p), fam_unique, phy_unique


class PhyloGIN(nn.Module):
    def __init__(self, n_genera, A_g2f, A_f2p, hidden=32, n_classes=3, dropout=0.4):
        super().__init__()
        self.register_buffer("A_g2f", A_g2f)
        self.register_buffer("A_f2p", A_f2p)
        n_fam = A_g2f.shape[0]; n_phy = A_f2p.shape[0]

        # Per-leaf embedding (each genus has 1 input feature -> hidden)
        self.genus_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # GIN-style MLP after each aggregation step
        self.family_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.phylum_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # Readout: concat per-level pooled embeddings (sum-pool over each level)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (B, n_genera)
        g_emb = self.genus_embed(x.unsqueeze(-1))                 # (B, N_g, h)
        f_emb = self.family_mlp(self.A_g2f @ g_emb)               # (B, N_f, h)
        p_emb = self.phylum_mlp(self.A_f2p @ f_emb)               # (B, N_p, h)

        # Sum-pool each level, concat
        g_pool = g_emb.sum(dim=1)
        f_pool = f_emb.sum(dim=1)
        p_pool = p_emb.sum(dim=1)
        h = torch.cat([g_pool, f_pool, p_pool], dim=1)            # (B, 3h)
        return self.classifier(h)


# --------------------------------------------------------------------------- #
class PhyloGINClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper so we can reuse the existing CV machinery."""
    def __init__(self, A_g2f=None, A_f2p=None, n_genera=None, hidden=32, dropout=0.4,
                 lr=1e-3, weight_decay=1e-4, epochs=120, batch_size=32, random_state=SEED):
        self.A_g2f, self.A_f2p, self.n_genera = A_g2f, A_f2p, n_genera
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

        self._model = PhyloGIN(self.n_genera, self.A_g2f, self.A_f2p,
                               hidden=self.hidden, n_classes=len(self.classes_),
                               dropout=self.dropout)

        Xt, yt = torch.from_numpy(Xs), torch.from_numpy(ye).long()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True,
        )
        cw = np.bincount(ye)
        weights = torch.tensor(cw.sum() / (len(cw) * cw), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self._model(xb), yb)
                loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        self._model.eval()
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            return torch.softmax(self._model(torch.from_numpy(Xs)), dim=1).numpy()

    def predict(self, X):
        return self._le.inverse_transform(self.predict_proba(X).argmax(axis=1))


# --------------------------------------------------------------------------- #
def multiclass_auroc(model, X_te, y_te, all_classes):
    if not hasattr(model, "predict_proba"):
        return np.nan
    proba = model.predict_proba(X_te)
    model_classes = list(model.classes_)
    aligned = np.zeros((len(y_te), len(all_classes)))
    for i, c in enumerate(all_classes):
        if c in model_classes:
            aligned[:, i] = proba[:, model_classes.index(c)]
    lb = LabelBinarizer().fit(all_classes)
    yb = lb.transform(y_te)
    if yb.shape[1] == 1:
        yb = np.hstack([1 - yb, yb])
    keep = (yb.sum(axis=0) > 0) & (yb.sum(axis=0) < yb.shape[0])
    if keep.sum() < 2:
        return np.nan
    return roc_auc_score(yb[:, keep], aligned[:, keep], average="macro")


def cv_eval(model, splits, X, y, classes, label):
    rows = []
    for split_name, splitter in splits.items():
        for fold_idx, (tr, te) in enumerate(splitter, 1):
            m = clone(model)
            m.fit(X[tr], y[tr])
            yp = m.predict(X[te])
            rows.append({
                "split": split_name, "fold": fold_idx, "model": label,
                "n_train": len(tr), "n_test": len(te),
                "balanced_accuracy": balanced_accuracy_score(y[te], yp),
                "auroc_ovr_macro":   multiclass_auroc(m, X[te], y[te], classes),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    print("Loading data ...")
    X_full = pd.read_csv(OUT / "combined_genus.tsv", sep="\t", index_col=0)
    meta   = pd.read_csv(OUT / "combined_meta.tsv",  sep="\t", index_col=0)

    rel = X_full.div(X_full.sum(axis=1), axis=0)
    keep = (rel > 0).mean(axis=0) >= MIN_PREV
    rel = rel.loc[:, keep]
    print(f"After prevalence filter: {rel.shape[1]} genera")
    X = clr(rel.values.astype(float))
    y = meta["disease"].values
    cohort = meta["cohort"].values
    classes = np.array(sorted(np.unique(y)))

    print("Building phylogenetic tree from genus column names ...")
    A_g2f, A_f2p, fam_names, phy_names = build_tree_aggregators(rel.columns.tolist())

    model = PhyloGINClassifier(
        A_g2f=A_g2f, A_f2p=A_f2p, n_genera=rel.shape[1],
        hidden=32, dropout=0.4, epochs=120,
    )

    # ---- Same CV splits as combine_and_train.py
    print("\n========== Random 5-fold stratified CV ==========")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df_random = cv_eval(model, {"random": list(skf.split(X, y))}, X, y, classes, "PhyloGIN")
    print(df_random.groupby("model")[["balanced_accuracy", "auroc_ovr_macro"]]
          .agg(["mean", "std"]).round(3))

    print("\n========== Leave-one-cohort-out ==========")
    logo = LeaveOneGroupOut()
    df_loco = cv_eval(model, {"LOCO": list(logo.split(X, y, groups=cohort))}, X, y, classes, "PhyloGIN")
    splits = list(logo.split(X, y, groups=cohort))
    df_loco["held_out"] = [cohort[splits[i-1][1][0]] for i in df_loco["fold"]]
    print(df_loco[["held_out", "balanced_accuracy", "auroc_ovr_macro"]].to_string(index=False))

    print("\n========== Within-cohort 5-fold ==========")
    within_rows = []
    for c in np.unique(cohort):
        mask = cohort == c
        Xc, yc = X[mask], y[mask]
        n_per_class = np.bincount(LabelEncoder().fit_transform(yc))
        n_splits = min(5, n_per_class.min())
        if n_splits < 2:
            print(f"  Skipping {c}: too few samples per class.")
            continue
        skf_c = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        df_c = cv_eval(model, {f"within_{c}": list(skf_c.split(Xc, yc))},
                       Xc, yc, np.unique(yc), "PhyloGIN")
        df_c["cohort"] = c
        print(f"  --- {c} (n={mask.sum()}) ---")
        print(df_c[["balanced_accuracy", "auroc_ovr_macro"]].agg(["mean", "std"]).round(3))
        within_rows.append(df_c)

    # ---- Save and print head-to-head comparison
    pd.concat([df_random.assign(split_type="random"),
               df_loco.assign(split_type="LOCO"),
               *[d.assign(split_type=f"within_{d['cohort'].iloc[0]}") for d in within_rows]],
              ignore_index=True).to_csv(OUT / "metrics_phylo_gnn.csv", index=False)

    # ---- Comparison vs baselines from previous run
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD: PhyloGIN vs baselines (means)")
    print("=" * 60)
    try:
        baseline_random = pd.read_csv(OUT / "metrics_random.csv")
        b = baseline_random.groupby("model")["auroc_ovr_macro"].mean().round(3)
        print("\nRandom 5-fold AUROC (multi-class, OvR macro):")
        print(b.to_string())
        print(f"  PhyloGIN: {df_random['auroc_ovr_macro'].mean():.3f}")
    except Exception as e:
        print("(baseline comparison skipped):", e)

    print(f"\n✓ Saved {OUT/'metrics_phylo_gnn.csv'}")


if __name__ == "__main__":
    main()
