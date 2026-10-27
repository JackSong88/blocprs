# BLocPRS

**Under construction. Dedicated documentation is coming soon.**

Code for BLocPRS, a locus-aware model for combining polygenic risk
scores through genome-wide source weights and regularized regional
adjustments.

## Code overview

- `data_preparation/`: source weights, cohort preparation, PRS scoring,
  genomic windows, and model-ready datasets.
- `model/`: BLocPRS, comparator models, fitting, and SNP-weight export.
- `evaluation/`: covariate-adjusted evaluation and bootstrap uncertainty.
- `tables/`: source panels and genomic window definitions.
- `setup_environment.sh`: Python environment setup.

Some preprocessing scripts are adapted from
[MoEPRS](https://github.com/shz9/moe-prs-paper); see `LICENSE`.
