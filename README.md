# BLocPRS: locus-aware ensembles of polygenic scores provide accurate linear prediction (2026)

**Under construction. Dedicated documentation coming soon!**

This repository contains code and scripts for reproducing the analyses in
our manuscript. BLocPRS is a locus-aware framework that combines pretrained
polygenic risk scores (PRSs) through genome-wide source weights and regularized
regional adjustments, allowing both the magnitude and relative contributions
of source PRSs to vary across genomic regions. It can integrate scores derived
from different ancestries, sexes, related traits, or construction methods,
with fitted coefficients that can be exported as a single set of additive
variant weights for conventional PRS scoring.

## Workflow and repository structure

1. `data_preparation/1_prepare_pgs_weights/`: download and prepare source PRS weights.
2. `data_preparation/2_prepare_validation_data/`: prepare cohort genotypes, phenotypes and covariates.
3. `data_preparation/3_pgsc_calc_pipeline/`: run pgsc_calc and extract genome-wide scores, ancestry assignments and projected PCs.
4. `data_preparation/4_prepare_window_scores/`: harmonize variants across cohorts, assign genomic windows, score and construct window-specific PRSs.
5. `data_preparation/5_generate_datasets/`: use `create_splits.py` once (or reuse existing IDs), then build genome-wide and windowed datasets with those same IDs using `create_datasets.py`. Transfer datasets use `--all-participants`; helpers create training subsets and structured source controls.
6. `model/`: fit BLocPRS and comparator models, save fitted parameters and export SNP weights. The reusable model implementation is in `model/blocprs/`.
7. `evaluation/`: evaluate prediction performance, paired bootstrap uncertainty and AUROC comparisons.
8. `analyses/`: run the supporting analyses:

   - `simulations/`: global-only, diffuse regional and concentrated regional simulation scenarios.
   - `mtl_prs/`: implementation of the transfer-learning MTL-PRS baseline.
   - `interpretation/`: window contributions, ranked-window ablations, and GWAS associated analyses.

9. `plotting/`: generate main Figures 2-7 from completed results; `supplementary/` contains the supplementary figure scripts.

## Citation
```bibtex
@article {Song2026.10.26.759503,
	author = {Song, Junyi and Zabad, Shadi and Gravel, Simon and Li, Yue},
	title = {BLocPRS: locus-aware ensembles of polygenic scores provide accurate linear prediction},
	elocation-id = {2026.10.26.759503},
	year = {2026},
	doi = {10.64898/2026.10.26.759503},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/10/30/2026.10.26.759503},
	eprint = {https://www.biorxiv.org/content/early/2026/10/30/2026.10.26.759503.full.pdf},
	journal = {bioRxiv}
}

```
