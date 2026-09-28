# Evidence map

- Architecture, labels, grouping, selection: data/protocol.json; current model.py implementation in the existing experiment.
- Ranking and target claims: data/hsu_A_oof_rankings.csv; comparison_all66.csv; no prediction values changed.
- Hsu approach: https://doi.org/10.1038/s41587-021-01146-5 and author implementation https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data . This project is an adaptation using ESM-2 35M and an additive masked-reference score.
- Wet-lab values: pre-existing construct-level S labels, not a new validation round.
- ORCA skill: supplied ORCA-scientific-illustrator-latest-20260918.zip, byte-identical to the earlier (1) archive. Native PowerPoint editing was interrupted by another active window; the user explicitly approved code-generated editable PPTX.
- Figures are deterministic computational diagrams and real-data plots; no image-generation assets were needed.

- New ablations: data/ablation_protocol.json; data/ablation_metrics.csv; data/oof_ablation.csv; per-fold training records in ablation_experiment/results/. Recomputed old model predictions agree to numerical precision.

Repository import: the retained Non-TRM consensus figure was copied from the existing PUF_alphafold checkout (wiki/figures/fig5_model3_nontrm_consensus.png). SHA256: a2419c9524055f9b8fdd201720c407699480c7440ad1ce2a23eb37ec3019bd8b. The new Wiki hosts this figure locally rather than depending on the previous raw URL.
