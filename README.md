# Gut Microbiome → Multi-Class Disease Prediction

A complete, modular machine-learning pipeline for predicting disease class from gut microbiome abundance data. Includes both classic ML (Logistic Regression, Random Forest, XGBoost) and a PyTorch neural network, with proper microbiome-specific preprocessing.

The design follows the standard approach used in published microbiome-ML studies (e.g. Su et al. 2022, Bang et al. 2019, Li et al. 2023): per-sample relative abundances → prevalence filter → CLR transform → stratified cross-validation across multiple models.

> **For the actual results, methodology, and findings on real public data (MetAML 1,709 samples + SM/FD 16S 419 samples, 2,128 total): see [FINDINGS.md](FINDINGS.md).**

---

## Project layout

```
microbiome_ml/
├── data_utils.py      # Synthetic data generator + real-data loader
├── preprocessing.py   # Prevalence filter + Centered Log-Ratio (CLR) transform
├── models.py          # LR, Random Forest, XGBoost, PyTorch MLP
├── train.py           # Stratified K-fold CV + metrics + feature importance
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost torch
```

### 2. Run the synthetic-data demo

```bash
python train.py
```

This generates 600 samples (5 disease classes × 120 samples) over 300 simulated taxa, runs 5-fold stratified CV on every model, and writes results to `results/`.

### 3. Run on your own data

```bash
python train.py \
    --abundance path/to/abundance_table.tsv \
    --metadata  path/to/metadata.tsv \
    --label-col disease \
    --sample-id-col sample_id
```

**Expected file formats:**

`abundance_table.tsv` — rows = samples, columns = taxa. First column is the sample ID.

| sample_id | s__Bacteroides_fragilis | s__Faecalibacterium_prausnitzii | … |
|-----------|------------------------:|-------------------------------:|---|
| S001      | 0.0421                  | 0.1183                         | … |
| S002      | 0.0033                  | 0.0892                         | … |

`metadata.tsv` — one row per sample, must contain `sample_id` and the label column:

| sample_id | disease |
|-----------|---------|
| S001      | IBD     |
| S002      | Healthy |

If your abundance file is **transposed** (taxa as rows, samples as columns — common in MetaPhlAn / Kraken outputs), transpose it before running. Many public datasets like [curatedMetagenomicData](https://waldronlab.io/curatedMetagenomicData/) need this.

---

## What the pipeline does

### Preprocessing (`preprocessing.py`)

Microbiome data is **compositional** — each sample's abundances sum to 1, so taxa are not independent. Standard scalers/correlations applied to raw relative abundances produce spurious signals. Two steps fix this:

1. **Prevalence filter.** Drop taxa present in fewer than 10% of samples (default). This removes noisy, sparse features that hurt generalization.
2. **CLR transform.** For each sample, replace each `x_i` with `log(x_i) − mean(log(x))`. This puts the data in real Euclidean space where standard ML works correctly. A small pseudocount handles zeros.

### Models (`models.py`)

| Model | Why it's here |
|-------|---------------|
| Logistic Regression (L2) | Strong, interpretable linear baseline. Wrapped in a scaling pipeline. |
| Random Forest | Robust nonlinear model, gives feature importances. |
| XGBoost | Gradient boosting; usually the strongest classic-ML model on tabular data. |
| MLP (PyTorch) | Small regularized neural net. Microbiome datasets are usually too small for deep nets, but a 2-layer MLP is a fair point of comparison. |

The PyTorch MLP is wrapped in a scikit-learn-compatible class so it plugs into the same CV loop. It uses class-weighted cross-entropy loss to handle imbalanced classes.

### Cross-validation & metrics (`train.py`)

5-fold **stratified** K-fold (preserves class proportions in every fold). Reports:

- Accuracy
- Balanced accuracy (mean of per-class recall — important when classes are imbalanced)
- Macro and weighted F1
- One-vs-rest macro AUROC

Plus, for the best model: a full classification report and aggregated confusion matrix from out-of-fold predictions.

### Feature importance

After CV, a Random Forest is fit on the full dataset to rank the most discriminative taxa. These are the candidate **microbial biomarkers** worth investigating biologically.

---

## Realistic expectations

Published cross-cohort multi-disease classifiers in the gut microbiome literature typically report **AUROC of 0.69–0.91** depending on the disease. Performance is generally:

- **High** for intestinal diseases (IBD, CRC) — strong, consistent dysbiosis signatures.
- **Moderate** for metabolic diseases (T2D, obesity) — signatures vary across cohorts.
- **Lower** for neurological / psychiatric conditions.

If you're running on real data, expect a noticeable drop when validating on a different cohort than the training one (this is called the **batch effect** problem and is the central open challenge in microbiome ML).

---

## Where to get real data

- **curatedMetagenomicData** (R/Bioconductor) — thousands of harmonized samples across many diseases.
- **MGnify** (EBI) — public metagenomic studies.
- **NCBI SRA / ENA** — raw reads from individual studies; you'd need to run a profiler like MetaPhlAn or Kraken first to get an abundance table.

For a quick start with real data, the curatedMetagenomicData R package can export abundance + metadata as TSVs that drop straight into this pipeline.

---

## Outputs

After running, `results/` contains:

- `cv_metrics.csv` — per-fold metrics for every model
- `cv_summary.csv` — mean ± std summary across folds
- `confusion_<model>.csv` — aggregated confusion matrix
- `top_features.csv` — top 25 discriminative taxa
