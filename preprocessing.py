"""
Microbiome-specific preprocessing.

Microbiome abundance data is COMPOSITIONAL (constrained to a simplex —
each sample sums to 1). Standard ML preprocessing on raw relative abundances
violates this constraint and can produce spurious correlations.

Standard practice in the field:
  1. Filter low-prevalence taxa (present in too few samples).
  2. Apply a log-ratio transform — most commonly Centered Log-Ratio (CLR).
  3. Then standard scalers / models work on the transformed values.

References:
  - Aitchison, "The Statistical Analysis of Compositional Data" (1986)
  - Gloor et al. "Microbiome Datasets Are Compositional" (2017)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_low_prevalence(
    X: pd.DataFrame,
    min_prevalence: float = 0.10,
    min_abundance: float = 1e-5,
) -> pd.DataFrame:
    """
    Drop taxa that are present (above `min_abundance`) in fewer than
    `min_prevalence` fraction of samples. Reduces noise and dimensionality.
    """
    present = (X > min_abundance).mean(axis=0)
    keep = present >= min_prevalence
    return X.loc[:, keep]


def clr_transform(X: pd.DataFrame, pseudocount: float | None = None) -> pd.DataFrame:
    """
    Centered Log-Ratio transform.

    For a compositional vector x, CLR(x)_i = log(x_i) - mean(log(x)).
    Zeros must be handled before the log; we add a pseudocount.

    A common choice: pseudocount = (smallest nonzero) / 2.
    """
    X_arr = X.to_numpy(dtype=float)

    if pseudocount is None:
        nonzero = X_arr[X_arr > 0]
        pseudocount = nonzero.min() / 2 if nonzero.size else 1e-6

    X_arr = X_arr + pseudocount

    # Re-normalize so rows sum to 1 (still compositional)
    X_arr = X_arr / X_arr.sum(axis=1, keepdims=True)

    log_X = np.log(X_arr)
    clr = log_X - log_X.mean(axis=1, keepdims=True)

    return pd.DataFrame(clr, index=X.index, columns=X.columns)


def preprocess(
    X: pd.DataFrame,
    min_prevalence: float = 0.10,
    use_clr: bool = True,
) -> pd.DataFrame:
    """Full preprocessing pipeline: prevalence filter -> CLR (optional)."""
    X_filtered = filter_low_prevalence(X, min_prevalence=min_prevalence)
    if use_clr:
        return clr_transform(X_filtered)
    return X_filtered
