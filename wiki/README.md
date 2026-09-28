# PUF Wiki: Hsu-inspired model A, Top30 edition

Two English Wiki pages, a native editable workflow, three ORCA result plots, all 66 rankings and model/subset comparison tables. The main recovery chart displays Top5, Top15 and Top30 using integer model hit labels. The two redundant summary tables are omitted from the Markdown pages; source CSVs remain available.

Hsu A: Spearman 0.404947; Top30 overlap 20/30 (66.7%), random expectation 13.64/30 (45.5%); enrichment 1.47×. Selected mean measured S is 0.291578. All five highlighted constructs enter predicted Top30.

Reporting K was changed after examining predictions. The feature ablations were fitted in this edition; previous model predictions were reproduced, and the historical inner-selection protocol remains unchanged. Measurements provide retrospective agreement only. Subsets smaller than 30 have no Top30 metric. Previous packages retain all earlier results.

Keep data/ and figures/ next to the pages. Plot scripts require NumPy and Matplotlib; use --output with a new directory to regenerate figures.

Feature ablation edition: five representations, matched grouped CV and A selection. Position-only rho 0.155 / Top30 14; one-hot 0.292 / 16; zero-shot −0.265 / 9; score-only Ridge 0.098 / 16; combined 0.405 / 20. All values are development-stage estimates. Reproducible experiment inputs, code and fold records are in ablation_experiment/. Run python run_ablation.py there with an existing NumPy/Pandas/SciPy/scikit-learn environment; completed folds resume automatically.

This layout revision adds a compact core-code excerpt. No training or prediction values changed.

Latest case-display revision: replaced the original five-target section with five higher-ranked and five lower-ranked concordant cases, excluding the original five. Selection is explicitly post hoc, based on rank agreement. Original target data and all cohort-level predictions remain available and unchanged.

Latest revision: restored the original five high-measured-S targets before the ten selected rank-agreement cases. The page now shows 15 distinct constructs: original five, five additional higher-ranked cases, then five lower-ranked cases. Predictions and cohort metrics are unchanged.
