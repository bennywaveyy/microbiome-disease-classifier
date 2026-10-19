"""
Score new gut-microbiome samples against the trained MLP.

Usage
-----
    python predict.py --abundance new_samples.tsv

Input: TSV/CSV with rows = samples, columns = taxa, first column = sample_id.
Taxa names should follow the MetaPhlAn-style lineage strings used at training
(k__...|p__...|c__...|o__...|f__...|g__...|s__...). The predictor aligns
columns to what the model expects: missing taxa are filled with 0; extra taxa
are dropped. A warning is printed if too many expected taxa are missing.

Output: per-sample class probabilities + the predicted top class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models import _MLP

ART = Path(__file__).parent / "artifacts"
COVERAGE_WARN_THRESHOLD = 0.50  # warn if >50% of expected taxa are missing


def _clr(X: pd.DataFrame, pseudocount: float) -> np.ndarray:
    arr = X.to_numpy(dtype=float) + pseudocount
    arr = arr / arr.sum(axis=1, keepdims=True)
    log = np.log(arr)
    return log - log.mean(axis=1, keepdims=True)


def load_predictor():
    meta = json.loads((ART / "predictor_meta.json").read_text())
    scaler = joblib.load(ART / "scaler.joblib")
    label_encoder = joblib.load(ART / "label_encoder.joblib")
    model = _MLP(
        in_dim=meta["n_features"],
        n_classes=len(meta["classes"]),
        hidden=tuple(meta["hidden"]),
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(ART / "mlp_state.pt", map_location="cpu"))
    model.eval()
    return meta, scaler, label_encoder, model


def align_to_training_taxa(X_new: pd.DataFrame, expected_taxa: list[str]) -> pd.DataFrame:
    present = [t for t in expected_taxa if t in X_new.columns]
    coverage = len(present) / len(expected_taxa)
    if coverage < (1 - COVERAGE_WARN_THRESHOLD):
        print(
            f"⚠  Only {coverage:.0%} of the model's expected taxa ({len(present)}/{len(expected_taxa)}) "
            f"are present in the input. Predictions will be noisy.",
            file=sys.stderr,
        )
    aligned = pd.DataFrame(0.0, index=X_new.index, columns=expected_taxa)
    aligned[present] = X_new[present].astype(float)

    # Re-normalize each row to a relative abundance vector (sum to 1)
    sums = aligned.sum(axis=1).replace(0, 1)
    return aligned.div(sums, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Score new microbiome samples.")
    parser.add_argument("--abundance", required=True,
                        help="Path to TSV/CSV: rows=samples, cols=taxa, first col=sample_id.")
    parser.add_argument("--out", default="predictions.tsv",
                        help="Where to write per-sample probabilities.")
    args = parser.parse_args()

    sep = "\t" if Path(args.abundance).suffix in {".tsv", ".txt"} else ","
    X_new = pd.read_csv(args.abundance, sep=sep, index_col=0)
    print(f"Loaded {X_new.shape[0]} samples × {X_new.shape[1]} taxa")

    meta, scaler, label_encoder, model = load_predictor()
    classes = meta["classes"]

    X_aligned = align_to_training_taxa(X_new, meta["taxa"])
    X_clr = _clr(X_aligned, meta["pseudocount"])
    X_scaled = scaler.transform(X_clr).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled))
        probs = torch.softmax(logits, dim=1).numpy()

    out = pd.DataFrame(probs, index=X_new.index, columns=[f"P({c})" for c in classes])
    out["predicted"] = [classes[i] for i in probs.argmax(axis=1)]
    out["max_prob"] = probs.max(axis=1)
    out.to_csv(args.out, sep="\t")
    print(f"\nWrote {args.out}")
    print(out.head(10).to_string())


if __name__ == "__main__":
    main()
