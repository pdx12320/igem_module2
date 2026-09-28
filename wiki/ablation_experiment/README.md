# Reproduce the fixed PUF feature ablations

This directory includes the 81-construct input manifest and observed labels, reference sequence, cached ESM scores, exact inner/outer splits, training code and 66 fold-level results. No ESM checkpoint or GPU is required because the fixed reference score table is cached.

Use Python 3.11 and the environment in ../../../puf_reuse/requirements.txt (from the repository root: puf_reuse/requirements.txt). Run from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r puf_reuse/requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python wiki/ablation_experiment/run_ablation.py --limit 1 --output runs/pilot
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python wiki/ablation_experiment/run_ablation.py --output runs/full_cv
```

Rerunning with the same output directory resumes completed folds. Omitting --output resumes the bundled fold records, so it does not demonstrate a fresh training run. Use a new output directory to reproduce training. This small CPU benchmark took about 50 seconds on the original server; hardware changes runtime.

Results: oof_ablation.csv, ablation_metrics.csv, validation.json and fold_*.json. All supervised methods use identical construct/sequence groups and strategy A. The historical inner tie-break fraction remains 15/66. Reporting uses Top30 for the ablation; subsets smaller than 30 have no Top30 estimate. Missing C labels are not imputed.

This is a fixed-dataset reproduction tool, not an arbitrary new-assay training interface. Input hashes are verified. New data require new audited groups and splits; do not reuse these indices on another dataset. Selection and performance are development-stage, not prospective validation.
