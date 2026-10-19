"""
Train the MLP on the FULL dataset and persist everything needed for inference:
  - the PyTorch model weights
  - the StandardScaler fit on the CLR-transformed training matrix
  - the list of taxa columns the model expects (in order)
  - the LabelEncoder for class names
  - the CLR pseudocount used at training time

`predict.py` consumes these artifacts and scores new samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models import MLPClassifier, _MLP   # noqa: F401 (kept for unpickling/import parity)
from preprocessing import filter_low_prevalence

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(exist_ok=True)
SEED = 42
MIN_PREVALENCE = 0.10


def _clr_with_pseudocount(X: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    arr = X.to_numpy(dtype=float) + pseudocount
    arr = arr / arr.sum(axis=1, keepdims=True)
    log = np.log(arr)
    clr = log - log.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=X.index, columns=X.columns)


def main():
    print("Loading data ...")
    abundance = pd.read_csv(DATA / "abundance.tsv", sep="\t", index_col=0)
    metadata  = pd.read_csv(DATA / "metadata.tsv",  sep="\t", index_col=0)
    abundance, metadata = abundance.align(metadata, join="inner", axis=0)
    y = metadata["disease"]

    # ---- Preprocess (matches train.py defaults exactly)
    X_filt = filter_low_prevalence(abundance, min_prevalence=MIN_PREVALENCE)
    nonzero = X_filt.to_numpy()[X_filt.to_numpy() > 0]
    pseudocount = float(nonzero.min() / 2)
    print(f"  retained {X_filt.shape[1]} taxa, pseudocount={pseudocount:.2e}")

    X_clr = _clr_with_pseudocount(X_filt, pseudocount)

    # ---- Train MLP on full data
    print("Training MLP on full dataset ...")
    mlp = MLPClassifier(random_state=SEED, verbose=True, epochs=100)
    mlp.fit(X_clr, y)

    # ---- Persist
    torch.save(mlp._model.state_dict(), OUT / "mlp_state.pt")
    joblib.dump(mlp._scaler, OUT / "scaler.joblib")
    joblib.dump(mlp._label_encoder, OUT / "label_encoder.joblib")

    meta = {
        "taxa": list(X_filt.columns),
        "n_features": int(X_filt.shape[1]),
        "pseudocount": pseudocount,
        "min_prevalence": MIN_PREVALENCE,
        "hidden": list(mlp.hidden),
        "dropout": float(mlp.dropout),
        "classes": list(map(str, mlp.classes_)),
    }
    (OUT / "predictor_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Wrote artifacts to {OUT.resolve()}")
    print(f"  classes: {meta['classes']}")
    print(f"  features: {meta['n_features']}")


if __name__ == "__main__":
    main()
