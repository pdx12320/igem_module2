# Terminology ledger

- S: specificity score, fixed as C388 − (C295 + C871)/2; not absolute editing efficiency.
- PUF-ESM feature integration: ESM-2 masked-marginal score + reference-centred one-hot encoding + Ridge; not LoRA or exact paper replication.
- Out-of-fold (OOF): excludes the test sequence group from training.
- Top30 overlap: intersection of predicted and measured Top30 sets among the same 66 constructs.
- Random expectation: uniform selection from the measured cohort; not random mutagenesis.
- Wet-lab comparison: retrospective agreement, not prospective validation.
- Best: highest Spearman within this 12-method comparison only.

- Position-only: per-residue binary substitution flags, without amino-acid identity.
- Position × amino-acid one-hot: reference-centred difference encoding, not a bag of amino acids.
- ESM-2 zero-shot: unchanged raw mutation score, without current assay fitting or sign tuning.

- Public model name: PUF-ESM Specificity Ranker; short name: PUF-ESM. Historical Hsu_ESM2_OneHot_Ridge_A identifiers denote the same fitted model and remain unchanged in archived experiment records. Hsu et al. remain cited as methodological inspiration.
