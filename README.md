# iGEM Module 2 — PUF specificity modelling
The main Wiki text and figures are synchronized from the current [WIKI_SUSTech-BIO Model page source](https://github.com/pdx12320/WIKI_SUSTech-BIO/tree/main/wiki/components/model). See [source and figure checksums](wiki/CURRENT_MODEL_SOURCE.json).


## Wiki pages

- [PUF dry-lab Wiki](wiki/PUF_Dry_Lab_Wiki_EN_ESM.md)
- [Module 2 Design–Build–Test–Learn](wiki/PUF_Module2_DBTL_EN_ESM.md)
- [Reusable Markdown and image addresses](wiki/REUSE_LINKS.md)

The main Wiki page follows the current WIKI_SUSTech-BIO content and figures and includes the PUF-ESM Specificity Ranker (ESM-2 score + one-hot + Ridge), matched feature ablations, Top5/15/30 recovery, and the current Wiki figures. All model assessments are development-stage cross-validation, not independent prospective validation.

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
