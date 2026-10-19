#!/usr/bin/env Rscript
# DADA2 pipeline for the FD 16S V4 dataset (PRJNA787213).
# FD reads have primers ALREADY trimmed -> trimLeft = c(0, 0).
# Outputs go to fd_out/.

suppressPackageStartupMessages(library(dada2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
setwd(dirname(normalizePath(sub("^--file=", "", script_arg))))
dir.create("fd_out",  showWarnings = FALSE)
dir.create("fd_filt", showWarnings = FALSE)

fnFs <- sort(list.files("fd_fastq", pattern = "_1\\.fastq\\.gz$", full.names = TRUE))
fnRs <- sort(list.files("fd_fastq", pattern = "_2\\.fastq\\.gz$", full.names = TRUE))
sample.names <- sub("_1\\.fastq\\.gz$", "", basename(fnFs))
cat("FD samples:", length(sample.names), "\n")
stopifnot(length(fnFs) == length(fnRs))

filtFs <- file.path("fd_filt", paste0(sample.names, "_F_filt.fastq.gz"))
filtRs <- file.path("fd_filt", paste0(sample.names, "_R_filt.fastq.gz"))
names(filtFs) <- sample.names
names(filtRs) <- sample.names

# FD: primers already trimmed; truncate at 230/180 same as SM
cat("=== filterAndTrim ===\n")
out <- filterAndTrim(
  fnFs, filtFs, fnRs, filtRs,
  trimLeft = c(0, 0),
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

getN <- function(x) sum(getUniques(x))
track <- cbind(out, sapply(dadaFs, getN), sapply(dadaRs, getN),
               sapply(mergers, getN), rowSums(seqtab.nochim))
colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
rownames(track) <- sample.names
write.table(track, "fd_out/track.tsv", sep = "\t", quote = FALSE)

cat("=== assignTaxonomy (SILVA v138.1 train_set) ===\n")
taxa <- assignTaxonomy(seqtab.nochim, "ref/silva_nr99_v138.1_train_set.fa.gz",
                      multithread = TRUE, tryRC = TRUE)
cat("=== addSpecies ===\n")
taxa <- addSpecies(taxa, "ref/silva_species_assignment_v138.1.fa.gz", tryRC = TRUE)

asv_seqs <- colnames(seqtab.nochim)
asv_ids  <- paste0("ASV", sprintf("%05d", seq_along(asv_seqs)))
colnames(seqtab.nochim) <- asv_ids
write.table(t(seqtab.nochim), "fd_out/asv_table.tsv", sep = "\t", quote = FALSE, col.names = NA)

tax_df <- as.data.frame(taxa)
rownames(tax_df) <- asv_ids
tax_df$ASV_seq <- asv_seqs
write.table(tax_df, "fd_out/taxonomy.tsv", sep = "\t", quote = FALSE, col.names = NA)

cat("=== Aggregating to genus level ===\n")
genus_labels <- with(tax_df, paste(
  ifelse(is.na(Phylum), "p__NA", paste0("p__", Phylum)),
  ifelse(is.na(Family), "f__NA", paste0("f__", Family)),
  ifelse(is.na(Genus),  "g__NA", paste0("g__", Genus)),
  sep = "|"
))
counts <- t(seqtab.nochim)
genus_mat <- rowsum(counts, group = genus_labels, reorder = TRUE)
genus_mat <- t(genus_mat)
cat("FD genus table:", dim(genus_mat), "\n")
write.table(genus_mat, "fd_out/genus_table.tsv", sep = "\t", quote = FALSE, col.names = NA)

cat("\n✓ FD pipeline complete. Outputs in fd_out/\n")
