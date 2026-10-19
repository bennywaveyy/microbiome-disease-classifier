"""
Data loading and synthetic data generation for gut microbiome ML.

Real microbiome data typically comes as a sample-by-taxon abundance matrix:
    - Rows: samples (patients)
    - Columns: microbial taxa (species, genera, OTUs, ASVs)
    - Values: read counts (integers) or relative abundances (floats summing to ~1)

The synthetic generator below mimics realistic properties:
    - Compositional (rows sum to 1)
    - Sparse (most taxa absent in any given sample)
    - Disease-specific signature taxa (a small subset is up/down-regulated per class)
    - Heavy-tailed abundance distribution (lognormal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_microbiome(
    n_samples_per_class: int = 120,
    n_taxa: int = 300,
    classes: tuple[str, ...] = ("Healthy", "IBD", "CRC", "T2D", "Obesity"),
    n_signature_taxa_per_class: int = 15,
    sparsity: float = 0.7,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a synthetic gut microbiome abundance table for multi-class disease prediction.

    Returns
    -------
    X : pd.DataFrame of shape (n_samples, n_taxa)
        Relative abundances. Each row sums to 1.
    y : pd.Series of shape (n_samples,)
        Disease label per sample.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(classes)
    n_samples = n_samples_per_class * n_classes

    # Realistic taxon names (faux but plausibly structured)
    taxa = [f"s__Species_{i:03d}" for i in range(n_taxa)]

    # Baseline lognormal abundances — same draw shape across samples,
    # then we'll perturb per-class signatures.
    base_log_means = rng.normal(loc=-2.0, scale=1.5, size=n_taxa)

    # Pick signature taxa (disjoint, but not strictly — biology overlaps)
    signature_taxa = {}
    for cls in classes:
        idx = rng.choice(n_taxa, size=n_signature_taxa_per_class, replace=False)
        # Each signature taxon gets a directional shift (up or down) for this class
        directions = rng.choice([-1, 1], size=n_signature_taxa_per_class)
        magnitudes = rng.uniform(2.0, 4.0, size=n_signature_taxa_per_class)
        signature_taxa[cls] = (idx, directions * magnitudes)

    X_rows = []
    y_rows = []
    for cls in classes:
        sig_idx, sig_shift = signature_taxa[cls]
        for _ in range(n_samples_per_class):
            log_abund = base_log_means + rng.normal(scale=0.6, size=n_taxa)
            log_abund[sig_idx] += sig_shift  # disease-specific signature
            abund = np.exp(log_abund)

            # Apply sparsity: zero out a random fraction of taxa
            zero_mask = rng.random(n_taxa) < sparsity
            abund[zero_mask] = 0.0

            # Avoid all-zero rows
            if abund.sum() == 0:
                abund[rng.integers(n_taxa)] = 1.0

            # Normalize to relative abundance (compositional)
            abund = abund / abund.sum()
            X_rows.append(abund)
            y_rows.append(cls)

    X = pd.DataFrame(X_rows, columns=taxa)
    y = pd.Series(y_rows, name="disease")

    # Shuffle so classes aren't grouped
    perm = rng.permutation(n_samples)
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)
    return X, y


def load_abundance_table(
    abundance_path: str | Path,
    metadata_path: str | Path,
    label_column: str = "disease",
    sample_id_column: str = "sample_id",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a real microbiome dataset.

    Expected formats:
      abundance_path: TSV/CSV with rows = samples, columns = taxa.
                      First column should be the sample ID (matching `sample_id_column`).
      metadata_path : TSV/CSV with one row per sample, containing at least
                      [sample_id_column, label_column].

    Many public datasets (e.g., curatedMetagenomicData, MetaPhlAn outputs)
    use a transposed layout (taxa as rows). If your file is transposed,
    set `.T` after loading or transpose upstream.
    """
    abundance_path = Path(abundance_path)
    metadata_path = Path(metadata_path)

    sep_a = "\t" if abundance_path.suffix in {".tsv", ".txt"} else ","
    sep_m = "\t" if metadata_path.suffix in {".tsv", ".txt"} else ","

    X = pd.read_csv(abundance_path, sep=sep_a, index_col=sample_id_column)
    meta = pd.read_csv(metadata_path, sep=sep_m).set_index(sample_id_column)

    # Align on shared samples
    common = X.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(
            "No shared sample IDs between abundance and metadata files. "
            f"Check `sample_id_column` (currently '{sample_id_column}')."
        )
    X = X.loc[common]
    y = meta.loc[common, label_column]

    return X, y
