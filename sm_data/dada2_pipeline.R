#!/usr/bin/env Rscript
# DADA2 pipeline for the SM 16S V4 dataset (PRJNA1258241).
# Inputs:  fastq/SRR*_{1,2}.fastq.gz, ref/silva_*_v138.1_*.fa.gz
# Outputs: out/asv_table.tsv, out/taxonomy.tsv, out/genus_table.tsv, out/track.tsv

suppressPackageStartupMessages(library(dada2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
setwd(dirname(normalizePath(sub("^--file=", "", script_arg))))
dir.create("out", showWarnings = FALSE)
dir.create("filt", showWarnings = FALSE)

fnFs <- sort(list.files("fastq", pattern = "_1\\.fastq\\.gz$", full.names = TRUE))
fnRs <- sort(list.files("fastq", pattern = "_2\\.fastq\\.gz$", full.names = TRUE))
sample.names <- sub("_1\\.fastq\\.gz$", "", basename(fnFs))
cat("Samples:", length(sample.names), "\n")
stopifnot(length(fnFs) == length(fnRs))

filtFs <- file.path("filt", paste0(sample.names, "_F_filt.fastq.gz"))
filtRs <- file.path("filt", paste0(sample.names, "_R_filt.fastq.gz"))
names(filtFs) <- sample.names
names(filtRs) <- sample.names

# 515F (19 nt) and 806R (20 nt) primers for V4. Trim them with trimLeft.
# Truncate at 230/180 — standard for 250x2 MiSeq V4 with overlap.
cat("=== filterAndTrim ===\n")
out <- filterAndTrim(
  fnFs, filtFs, fnRs, filtRs,
  trimLeft = c(19, 20),
  truncLen = c(230, 180),
  maxN = 0, maxEE = c(2, 2), truncQ = 2, rm.phix = TRUE,
  compress = TRUE, multithread = TRUE
)
print(head(out))

cat("=== learnErrors (forward) ===\n")
errF <- learnErrors(filtFs, multithread = TRUE)
cat("=== learnErrors (reverse) ===\n")
errR <- learnErrors(filtRs, multithread = TRUE)

cat("=== dada (forward) ===\n")
dadaFs <- dada(filtFs, err = errF, multithread = TRUE)
cat("=== dada (reverse) ===\n")
dadaRs <- dada(filtRs, err = errR, multithread = TRUE)

cat("=== mergePairs ===\n")
mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose = TRUE)

cat("=== makeSequenceTable ===\n")
seqtab <- makeSequenceTable(mergers)
cat("seqtab dims:", dim(seqtab), "\n")
cat("amplicon length distribution:\n"); print(table(nchar(getSequences(seqtab))))

cat("=== removeBimeraDenovo ===\n")
seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus", multithread = TRUE, verbose = TRUE)
cat("post-chimera dims:", dim(seqtab.nochim), "\n")

# Track reads through the pipeline
getN <- function(x) sum(getUniques(x))
track <- cbind(out, sapply(dadaFs, getN), sapply(dadaRs, getN),
               sapply(mergers, getN), rowSums(seqtab.nochim))
colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
rownames(track) <- sample.names
write.table(track, "out/track.tsv", sep = "\t", quote = FALSE)
cat("=== Read tracking ===\n"); print(track)

cat("=== assignTaxonomy (SILVA v138.1 train_set) ===\n")
taxa <- assignTaxonomy(
  seqtab.nochim, "ref/silva_nr99_v138.1_train_set.fa.gz",
  multithread = TRUE, tryRC = TRUE
)
cat("=== addSpecies ===\n")
taxa <- addSpecies(taxa, "ref/silva_species_assignment_v138.1.fa.gz", tryRC = TRUE)

# ---- Write ASV-level table ----
asv_seqs <- colnames(seqtab.nochim)
asv_ids  <- paste0("ASV", sprintf("%05d", seq_along(asv_seqs)))
colnames(seqtab.nochim) <- asv_ids
write.table(t(seqtab.nochim), "out/asv_table.tsv", sep = "\t", quote = FALSE, col.names = NA)

tax_df <- as.data.frame(taxa)
rownames(tax_df) <- asv_ids
tax_df$ASV_seq <- asv_seqs
write.table(tax_df, "out/taxonomy.tsv", sep = "\t", quote = FALSE, col.names = NA)

# ---- Aggregate to genus level (sample x genus matrix) ----
cat("=== Aggregating to genus level ===\n")
genus_labels <- with(tax_df, paste(
  ifelse(is.na(Phylum), "p__NA", paste0("p__", Phylum)),
  ifelse(is.na(Family), "f__NA", paste0("f__", Family)),
  ifelse(is.na(Genus),  "g__NA", paste0("g__", Genus)),
  sep = "|"
))
counts <- t(seqtab.nochim)                     # ASVs x samples
genus_mat <- rowsum(counts, group = genus_labels, reorder = TRUE)
genus_mat <- t(genus_mat)                      # samples x genera
cat("genus table:", dim(genus_mat), "\n")
write.table(genus_mat, "out/genus_table.tsv", sep = "\t", quote = FALSE, col.names = NA)

cat("\n✓ Pipeline complete. Outputs in out/\n")
