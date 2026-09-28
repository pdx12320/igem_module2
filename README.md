# iGEM Module 2 — PUF specificity modelling

## Wiki pages

- [PUF dry-lab Wiki](wiki/PUF_Dry_Lab_Wiki_EN_ESM.md)
- [Module 2 Design–Build–Test–Learn](wiki/PUF_Module2_DBTL_EN_ESM.md)
- [Reusable Markdown and image addresses](wiki/REUSE_LINKS.md)

The current pages include the Hsu-inspired ESM-2 score + one-hot + Ridge model, matched feature ablations, Top5/15/30 recovery, and 15 retrospective case examples. All model assessments are development-stage cross-validation, not independent prospective validation.

## Reproducibility

- [Ablation code, inputs and fold records](wiki/ablation_experiment/)
- [Ranking and comparison tables](wiki/data/)
- [ORCA plotting scripts](wiki/scripts/)
- [Editable workflow and figure exports](wiki/figures/)

No model was retrained in this repository import. Missing assay labels remain missing; the specificity classifier is untrained because the current complete cohort contains no negative S labels. The pre-existing repository license is retained.

## Reuse package

- [Install and predict new 493-aa FASTA sequences](puf_reuse/README.md)
- [Reproduce grouped feature-ablation training](wiki/ablation_experiment/README.md)

The prediction package contains the full-data model weight and cached reference score table; no GPU or structure files are needed.
