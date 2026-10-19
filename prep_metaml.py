"""
Convert the MetAML (Pasolli 2016) abundance.txt dump into the
abundance.tsv + metadata.tsv pair that train.py expects.

Source layout: rows = features (metadata + taxa), columns = samples.
We transpose, restrict to stool samples, consolidate disease labels into
a clean multi-class scheme, and keep only species-level (|s__) taxa.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW = Path(__file__).parent / "raw_data" / "abundance.txt"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# Map raw disease tokens -> clean class. None means drop the sample.
LABEL_MAP = {
    "n":                       "Healthy",
    "nd":                      "Healthy",
    "ibd_ulcerative_colitis":  "IBD",
    "ibd_crohn_disease":       "IBD",
    "t2d":                     "T2D",
    "cirrhosis":               "Cirrhosis",
    "obesity":                 "Obesity",
    "cancer":                  "CRC",
    "small_adenoma":           "CRC",
    "large_adenoma":           "CRC",
}

print(f"Reading {RAW} ...")
raw = pd.read_csv(RAW, sep="\t", header=None, low_memory=False, dtype=str)
raw = raw.set_index(0)               # row labels (feature names) become the index
raw.columns = raw.iloc[0]            # first row holds dataset_name per sample; reuse as columns header
df = raw.T                           # now rows = samples, cols = features
df.index.name = "sample_index"

print(f"  shape after transpose: {df.shape}")
print(f"  available metadata-ish cols: bodysite={df['bodysite'].value_counts().head(3).to_dict()}")

# ---- Sample filtering --------------------------------------------------
mask_stool = df["bodysite"].str.lower().str.strip() == "stool"
df = df[mask_stool].copy()
print(f"  stool samples: {len(df)}")

df["disease_clean"] = df["disease"].str.strip().map(LABEL_MAP)
df = df.dropna(subset=["disease_clean"]).copy()
print(f"  samples with usable disease label: {len(df)}")
print(df["disease_clean"].value_counts())

# ---- Split metadata vs taxa --------------------------------------------
# Taxa rows in the source begin with k__ (kingdom). After transpose they're columns.
taxa_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("k__")]
print(f"  total taxa columns: {len(taxa_cols)}")

# Restrict to species-level lines: paths that contain |s__ as a segment.
species_cols = [c for c in taxa_cols if "|s__" in c and "|t__" not in c]
print(f"  species-level columns: {len(species_cols)}")

abundance = df[species_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# MetAML stores abundances as percentages (0-100). Convert to relative (0-1).
row_sums = abundance.sum(axis=1)
print(f"  median row sum BEFORE renorm: {row_sums.median():.2f}  (expect ~100)")
abundance = abundance.div(row_sums.replace(0, 1), axis=0)
print(f"  median row sum AFTER renorm:  {abundance.sum(axis=1).median():.4f}  (expect ~1.0)")

# Use stable, unique sample IDs. Original sampleID can collide across studies.
sample_ids = [
    f"{ds}__{sid}" for ds, sid in zip(df["dataset_name"].astype(str), df["sampleID"].astype(str))
]
abundance.index = sample_ids
abundance.index.name = "sample_id"

metadata = pd.DataFrame({
    "sample_id": sample_ids,
    "disease":   df["disease_clean"].values,
    "dataset":   df["dataset_name"].values,
}).set_index("sample_id")

abundance.to_csv(OUT / "abundance.tsv", sep="\t")
metadata.to_csv(OUT / "metadata.tsv", sep="\t")

print(f"\nWrote {OUT/'abundance.tsv'}: {abundance.shape}")
print(f"Wrote {OUT/'metadata.tsv'}:  {metadata.shape}")
print("\nClass × cohort breakdown:")
print(pd.crosstab(metadata["disease"], metadata["dataset"]))
