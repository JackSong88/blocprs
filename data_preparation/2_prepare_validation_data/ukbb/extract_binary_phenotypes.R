#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(bigreadr)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(PheWAS)
})

option_list <- list(
  make_option(c("-f", "--ukb-file"), type = "character", help = "UKB wide-format CSV"),
  make_option("--withdrawn-file", type = "character", help = "Headerless withdrawn-ID file"),
  make_option(c("-p", "--phecodes"), type = "character", help = "Comma-separated phecodes"),
  make_option(c("-n", "--phenotype-names"), type = "character", help = "Names corresponding to the phecodes"),
  make_option(c("-o", "--outdir"), type = "character", help = "Phenotype output directory"),
  make_option("--coding609-file", type = "character", help = "UKB coding609 TSV for self-reported codes"),
  make_option("--include-cancer", action = "store_true", default = FALSE,
              help = "Include the cancer ICD fields"),
  make_option("--include-selfreported", action = "store_true", default = FALSE,
              help = "Include field 20002, mapped through coding609"),
  make_option("--apply-phecode-exclusion", action = "store_true", default = FALSE,
              help = "Apply PheWAS phenotype-specific control exclusions")
)
opt <- parse_args(OptionParser(option_list = option_list))
required <- c("ukb-file", "withdrawn-file", "phecodes", "outdir")
if (any(vapply(required, function(name) is.null(opt[[name]]), logical(1)))) {
  stop("Required: --ukb-file, --withdrawn-file, --phecodes and --outdir.")
}
if (opt$`include-selfreported` && is.null(opt$`coding609-file`)) {
  stop("--coding609-file is required with --include-selfreported.")
}
phecodes_requested <- unlist(strsplit(opt$phecodes, ",\\s*"))
phenotype_names <- if (is.null(opt$`phenotype-names`)) phecodes_requested else
  unlist(strsplit(opt$`phenotype-names`, ",\\s*"))
if (length(phenotype_names) != length(phecodes_requested)) {
  stop("--phenotype-names must have the same length as --phecodes.")
}
phenotype_map <- tibble(phecode = phecodes_requested, phenotype = phenotype_names) %>%
  group_by(phenotype) %>% summarise(phecodes = list(unique(phecode)), .groups = "drop")
filenames <- gsub("[[:space:]]+", "_", phenotype_map$phenotype)
filenames <- gsub("[^A-Za-z0-9_\\-\\.]", "", filenames)
if (any(filenames == "") || anyDuplicated(filenames)) stop("Phenotype output names must be nonempty and unique.")
outputs <- file.path(opt$outdir, paste0(filenames, ".txt"))

sex <- fread2(opt$`ukb-file`, select = "22001-0.0")[[1]]
eid <- fread2(opt$`ukb-file`, select = "eid", colClasses = "character")[[1]]
if (anyNA(eid) || anyDuplicated(eid)) stop("UKB participant IDs must be present and unique.")

icd10_columns <- c(
  paste0("40001-", 0:1, ".0"), paste0("40002-0.", 0:13), paste0("40002-1.", 0:13),
  paste0("41201-0.", 0:21), paste0("41202-0.", 0:79),
  paste0("41204-0.", 0:209), paste0("41270-0.", 0:258)
)
if (opt$`include-cancer`) icd10_columns <- unique(c(icd10_columns, paste0("40006-", 0:16, ".0")))
diagnoses <- fread2(opt$`ukb-file`, colClasses = "character", select = icd10_columns)
if (opt$`include-selfreported`) {
  coding609 <- fread2(opt$`coding609-file`)
  columns <- c(paste0("20002-0.", 0:33), paste0("20002-1.", 0:33),
               paste0("20002-2.", 0:33), paste0("20002-3.", 0:33))
  selfreported <- fread2(opt$`ukb-file`, colClasses = "character", select = columns) %>%
    mutate_all(~ as.character(factor(., levels = coding609$coding, labels = coding609$meaning)))
  diagnoses <- bind_cols(diagnoses, selfreported)
}

build_long <- function(frame) {
  frame %>% mutate_all(~ ifelse(. == "", NA, .)) %>% mutate(id = row_number()) %>%
    pivot_longer(-id, values_to = "code", values_drop_na = TRUE) %>%
    group_by(id, code) %>% summarise(count = n(), .groups = "drop") %>% ungroup() %>%
    transmute(id = as.integer(id), vocabulary_id = "ICD10",
              code = trimws(as.character(code)), count = as.integer(count)) %>% as_tibble()
}

# One qualifying code is sufficient; optional exclusions affect control status.
phenotypes <- createPhenotypes(
  build_long(diagnoses),
  id.sex = tibble(id = seq_along(sex), sex = c("F", "M")[sex + 1L]),
  vocabulary.map = mutate_at(phecode_map_icd10, "code", ~ sub("\\.", "", .)),
  min.code.count = 1,
  add.phecode.exclusions = opt$`apply-phecode-exclusion`,
  full.population.ids = seq_along(sex), translate = TRUE
)
phenotypes <- phenotypes[order(phenotypes$id), , drop = FALSE]
if (!identical(as.integer(phenotypes$id), seq_along(eid))) {
  stop("PheWAS output does not retain the expected participant order.")
}
present <- intersect(phecodes_requested, colnames(phenotypes))
missing <- setdiff(phecodes_requested, present)
if (length(missing)) warning("Missing phecodes: ", paste(missing, collapse = ", "))
selected <- phenotypes %>% select(all_of(c("id", present)))
selected$eid <- eid
withdrawn <- fread2(opt$`withdrawn-file`, header = FALSE, colClasses = "character")[[1]]
selected <- selected %>% filter(!eid %in% withdrawn)

# Combine requested codes by OR; all missing remains missing, not a control.
collapse_phecodes_or <- function(frame, columns) {
  if (!length(columns)) return(rep(NA_integer_, nrow(frame)))
  values <- frame %>% select(all_of(columns)) %>%
    mutate(across(everything(), ~ suppressWarnings(as.numeric(.)))) %>% as.matrix()
  any_case <- apply(values == 1, 1, function(row) any(row, na.rm = TRUE))
  all_missing <- apply(is.na(values), 1, all)
  as.integer(ifelse(any_case, 1L, ifelse(all_missing, NA_integer_, 0L)))
}

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
written <- 0L
for (i in seq_len(nrow(phenotype_map))) {
  requested <- phenotype_map$phecodes[[i]]
  present <- intersect(requested, names(selected))
  missing <- setdiff(requested, names(selected))
  if (length(missing)) warning(phenotype_map$phenotype[i], ": missing phecodes ", paste(missing, collapse = ", "))
  if (!length(present)) next
  output <- tibble(FID = selected$eid, IID = selected$eid,
                   PHENO = collapse_phecodes_or(selected, present)) %>% filter(!is.na(PHENO))
  write.table(output, file = outputs[i], sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
  written <- written + 1L
}
message("Wrote ", written, " binary phenotype files to ", opt$outdir)
