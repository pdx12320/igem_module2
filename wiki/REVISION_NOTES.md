# Revision notes

1. Replaced previous small-K evaluation displays with Top30 across both Wiki pages, plots, captions, target flags and comparison tables.
2. Recomputed overlaps from unchanged out-of-fold predictions: 20/30 versus 13.64 expected by uniform random selection.
3. All five highlighted constructs appear in the predicted Top30; preserved exact predicted and measured ranks and scores.
4. Recorded that Top30 was selected retrospectively for reporting; kept the original training protocol and validation partitions unchanged.
5. Preserved the Non-TRM text and all prior packages. No new training or wet-lab validation was performed.

6. Added matched position-only, amino-acid one-hot, zero-shot ESM, score-only Ridge and combined-feature ablations. Recomputed supervised models on the authorized server with 2 CPU threads; no GPU or new dependencies.
7. Added two ORCA plots, the five-method table and four evidence-based interpretations to both Markdown pages.

8. Latest layout: removed the main hit-summary table and ablation representation table from both pages; displayed three K values in Figure 2 with integer model hit labels; added a short excerpt from the actual feature/Ridge implementation. Numerical source tables are retained as CSV.

Latest case-display revision: replaced the original five-target section with five higher-ranked and five lower-ranked concordant cases, excluding the original five. Selection is explicitly post hoc, based on rank agreement. Original target data and all cohort-level predictions remain available and unchanged.

Latest revision: restored the original five high-measured-S targets before the ten selected rank-agreement cases. The page now shows 15 distinct constructs: original five, five additional higher-ranked cases, then five lower-ranked cases. Predictions and cohort metrics are unchanged.
