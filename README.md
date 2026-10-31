# BLocPRS

**Under construction. Dedicated documentation coming soon!**

Code for BLocPRS, a locus-aware model for combining polygenic risk scores
through genome-wide source weights and regularized regional adjustments.

## Workflow and repository structure

1. `data_preparation/1_prepare_pgs_weights/`: download and prepare source PRS weights.
2. `data_preparation/2_prepare_validation_data/`: prepare cohort genotypes, phenotypes and covariates.
3. `data_preparation/3_pgsc_calc_pipeline/`: run pgsc_calc and extract genome-wide scores, ancestry assignments and projected PCs.
4. `data_preparation/4_prepare_window_scores/`: harmonize variants across cohorts, assign genomic windows, score each window and check reconstruction of genome-wide scores.
5. `data_preparation/5_generate_datasets/`: use `create_splits.py` once (or reuse existing IDs), then build genome-wide and windowed datasets with those same IDs using `create_datasets.py`. Transfer datasets use `--all-participants`; helpers create training subsets and structured source controls.
6. `model/`: fit BLocPRS and comparator models, save fitted parameters and export SNP weights. The reusable model implementation is in `model/blocprs/`.
7. `evaluation/`: evaluate frozen genetic scores with covariate-adjusted metrics, paired bootstrap uncertainty and AUROC comparisons.
8. `analyses/`: run the supporting analyses:

   - `simulations/`: global-only, diffuse regional and concentrated regional simulation scenarios.
   - `mtl_prs/`: individual-level MTL-PRS fitting, scoring and held-out comparison.
   - `interpretation/`: window contributions, ranked-window ablations, training GWAS and GWAS concentration.

9. `plotting/`: generate main Figures 2-7 from completed results; `supplementary/` contains the supplementary figure scripts.

## Supporting files

- `tables/`: PRS-source panels; generate window definitions from the variants used in your data.
- `setup_environment.sh`: Python environment setup.
- `pyproject.toml`: package metadata and Python dependencies.

Run preprocessing from the repository root with the project environment activated.
Paths are relative to the repository; supply your own cohort data. Batch scripts
have editable CPU, memory and time defaults at the top, with no cluster account
or partition prescribed. External tools such as PLINK 2, R, Java and Apptainer
must be available for the preprocessing steps that use them.

## Citation

```bibtex
@article{Song2026.10.26.759503,
  author = {Song, Junyi and Zabad, Shadi and Gravel, Simon and Li, Yue},
  title = {{BLocPRS}: locus-aware ensembles of polygenic scores provide accurate linear prediction},
  year = {2026},
  doi = {10.64898/2026.10.26.759503},
  journal = {bioRxiv},
  URL = {https://www.biorxiv.org/content/10.64898/2026.10.26.759503v1}
}
```
