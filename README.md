# Gut Microbiome Disease Classification Across Cohorts

**Random cross-validation made this classifier look far better than it was. Pooling two 16S cohorts and running random 5-fold CV gave an MLP AUROC of 0.91; holding out one whole cohort at a time dropped it to 0.56. The model had learned which lab sequenced each sample, not which patients were sick.**

End-to-end machine learning on 2,128 public stool samples from 13 cohorts: shotgun MetaPhlAn profiles plus raw 16S V4 reads processed from FASTQ through DADA2 and SILVA, CLR-transformed, and modeled with logistic regression, random forest, a PyTorch MLP and a phylogeny-aware graph network. Full write-up: [FINDINGS.md](FINDINGS.md).

**About this project:** built with [Claude Code](https://claude.com/claude-code) as a coding collaborator. I directed the work: choosing which conditions and public datasets to study, making the modeling and validation decisions (multi-class vs per-disease models, cross-cohort validation, the phylogenetic GNN experiment), and interpreting the results. Claude wrote most of the implementation.

---

## Key results

| Question | Data | Result |
|---|---|---|
| Does pooling cohorts inflate accuracy? | 16S, SM + FD combined (n=419) | Yes. MLP AUROC 0.91 with random 5-fold CV vs 0.56 leave-one-cohort-out: **0.36 inflation** (0.29–0.36 across all four models) |
| Is there real signal inside one cohort? | FD vs healthy relatives (n=388) | Yes. MLP AUROC **0.886 ± 0.029** (5-fold CV within the cohort) |
| Does the shotgun multi-class model hold up across studies? | MetAML, 6 classes (n=1,709) | Mostly. MLP macro AUROC 0.916 (5-fold CV); leave-one-cohort-out on IBD and T2D cost only 0.03–0.04 |
| Does a phylogeny-aware graph network help? | 16S, SM + FD (n=419) | No. AUROC 0.68 random CV, 0.52 leave-one-cohort-out, 0.47 within FD, below every simpler model |

PCA on the combined 16S matrix shows the same thing directly: cohort label explains 15.8% of PC1+PC2 variance, disease label 12.8%.

![PCA: cohort vs disease](sm_data/combined_out/pca_cohort_vs_disease.png)

---

## Datasets

All public and reproducible from the accessions below.

| Dataset | Condition(s) | Technology | Samples |
|---|---|---|---|
| MetAML / Pasolli 2016 ([SegataLab/metaml](https://github.com/SegataLab/metaml)) | Healthy, IBD, colorectal cancer (CRC), type 2 diabetes (T2D), cirrhosis, obesity | Shotgun, MetaPhlAn species | 1,709 |
| Krausfeldt 2025 ([PRJNA1258241](https://www.ncbi.nlm.nih.gov/bioproject/1258241)) | Systemic mastocytosis (SM) vs healthy | 16S V4, paired-end MiSeq | 31 (22 SM + 9 healthy) |
| Pinkham 2023 ([PRJNA787213](https://www.ncbi.nlm.nih.gov/bioproject/787213)) | Familial dysautonomia (FD) vs healthy relatives | 16S V4, paired-end MiSeq | 388 (198 FD + 190 healthy) |

---

## Repository layout

```
.
├── FINDINGS.md               # full results, methods and limitations
├── prep_metaml.py            # MetAML dump -> data/abundance.tsv + data/metadata.tsv
├── data_utils.py             # data loaders (+ synthetic demo generator)
├── preprocessing.py          # prevalence filter + CLR transform
├── models.py                 # logistic regression, random forest, PyTorch MLP (+ XGBoost if installed)
├── train.py                  # stratified 5-fold CV, metrics, feature importance
├── eval_cohort_holdout.py    # random CV vs leave-one-cohort-out for IBD and T2D
├── train_predictor.py        # trains the shotgun MLP on all data and saves artifacts/
├── predict.py                # scores new samples with the trained MLP
├── artifacts/                # trained MLP weights, scaler, label encoder, taxa list
├── data/                     # prepared MetAML abundance + metadata
├── results/                  # shotgun CV and cohort-holdout outputs
└── sm_data/                  # 16S track (SM + FD)
    ├── dada2_pipeline.R          # SM: FASTQ -> ASVs -> SILVA v138.1 genus table
    ├── dada2_fd.R                # FD: same pipeline, primers already trimmed
    ├── train_sm_classifier.py    # SM vs healthy, leave-one-out CV (n=31)
    ├── combine_and_train.py      # SM + FD combined: random, leave-one-cohort-out, within-cohort CV
    ├── per_disease_and_pca.py    # per-disease binary models + PCA figure
    ├── train_phylo_gnn.py        # phylogeny-aware graph network (PhyloGIN)
    └── combined_out/             # 16S metrics and PCA figure
```

---

## Reproducing the results

```bash
pip install numpy pandas scikit-learn torch matplotlib joblib   # xgboost optional
```

**Shotgun track (MetAML).** Download the MetAML `abundance.txt` into `raw_data/`, then:

```bash
python prep_metaml.py
python train.py --abundance data/abundance.tsv --metadata data/metadata.tsv --label-col disease
python eval_cohort_holdout.py
python train_predictor.py        # optional: build the deployable MLP
```

**16S track (SM + FD).** Needs R with the Bioconductor `dada2` package. Download the FASTQs listed in `sm_data/fastq_urls.txt` and `sm_data/fd_urls.txt`, and a SILVA v138.1 reference into `sm_data/ref/`, then:

```bash
Rscript sm_data/dada2_pipeline.R
Rscript sm_data/dada2_fd.R
python sm_data/combine_and_train.py
python sm_data/per_disease_and_pca.py
python sm_data/train_phylo_gnn.py
```

**Score new samples** with the trained shotgun MLP:

```bash
python predict.py --abundance new_samples.tsv
```

Input is rows = samples, columns = MetaPhlAn-style taxa, first column = sample ID. Missing taxa are filled with 0 and extra taxa are dropped.

Running `python train.py` with no arguments runs a synthetic demo (5 classes × 120 samples × 300 taxa) to check the install.

---

## Method notes

- **Compositional data.** Each sample's abundances sum to 1, so taxa aren't independent. Taxa present in fewer than 10% of samples are dropped, then a centered log-ratio (CLR) transform with a small pseudocount moves the data into Euclidean space.
- **Class imbalance** is handled with class-weighted losses. Oversampling (SMOTE) was avoided because at n=22 it produces unrealistic synthetic samples.
- **Metrics:** accuracy, balanced accuracy, macro and weighted F1, one-vs-rest macro AUROC.
- **Validation:** every headline number states its split. Random stratified 5-fold CV is reported next to leave-one-cohort-out wherever a disease spans more than one study.
- **Not done:** formal batch correction (ConQuR, MMUPHin) was out of scope.
