# PUF specificity prediction and non-TRM sequence design

## Background and engineering objective

PUF variants can change RNA recognition, but stronger editing at a target does not necessarily imply greater specificity. We therefore ranked variants by target editing relative to two comparison sites, using the fixed score:

$$S=C_{388}-\frac{C_{295}+C_{871}}{2}.$$

Our two experimental batches contain 81 constructs. Sixty-six have measurements at all three sites; the remaining 15 lack at least one measurement. All 66 complete scores are positive. The current task is therefore **ranking among constructs with known positive S**. The S > 0 specificity classifier remains untrained because no negative class is available.

## Model principles: PUF-ESM Specificity Ranker

We adapted the approach of [Hsu et al.](https://doi.org/10.1038/s41587-021-01146-5), which combines evolutionary information with assay-labelled sequence features. Our implementation joins a frozen ESM-2 score with reference-centred one-hot sequence encoding, then fits Ridge regression to the measured editing outputs.

![Sequence features, training-side selection, three editing predictions and specificity ranking](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/architecture/figure@2x.png)

*Figure 1. PUF-ESM feature integration. A 493-residue sequence supplies reference-centred one-hot features and an ESM-2 mutation score. Three Ridge regressors use the available labels for C295, C388 and C871. Their predictions define S. Grouped inner validation selects parameters within each outer training fold. Held-out predictions evaluate ranking across the 66 complete constructs. The diagram describes this adaptation, not an exact reproduction of Hsu et al.*

### Sequence features and missing measurements

The evolutionary feature sums mutant-versus-reference log-probability differences at substituted residues. Each probability is computed from the reference sequence with that position masked, using frozen ESM-2 35M weights. This additive score captures sequence-model preferences; it does not directly measure editing specificity or model interactions between substitutions.

The second feature block encodes amino-acid substitutions relative to the fixed 493-residue reference. A separate Ridge regressor predicts each C value using only its observed training labels. Available labels from all 81 constructs contribute when eligible for the training fold; missing values are never replaced with zero. This model uses neither LoRA nor a structure input.

### Grouped validation and parameter selection

Outer validation leaves out one sequence group at a time. Repeated measurements and identical sequences remain grouped, and every held-out group is excluded from all training records. Ranking metrics use the same 66 constructs with complete S throughout.

Within each outer training fold, two-fold grouped validation tests Ridge penalties of 0.1, 1, 10, 100 and 1,000, with evolutionary-feature weights of 1 or 10. Feature standardization uses training data only. Strategy A prioritizes inner Spearman correlation, followed by the measured mean S of the proportional predicted-top subset and then regression error. Outer test labels do not select parameters.

### Core implementation

The following excerpt shows feature combination, masked per-target regression and specificity calculation. `selected_spec` is chosen using the inner training-side validation folds; `train_idx` excludes the held-out sequence group. `fit` uses only observed labels for each C output, and standardizes the ESM score within the training records. The full implementation is provided in [model.py](https://github.com/pdx12320/igem_module2/blob/main/wiki/ablation_experiment/model.py).

```python
from model import encode, fit, predict

onehot, esm_score = encode(sequences, reference, reference_logp)
models = fit(onehot, esm_score, Y, train_idx,
             "PUF_ESM", selected_spec)
C_hat = predict(models, onehot, esm_score, test_idx)
S_hat = C_hat[:, 1] - (C_hat[:, 0] + C_hat[:, 2]) / 2
```

Here, `Y` contains C295, C388 and C871 in that order. The excerpt uses the experiment's existing variables and functions; missing labels remain missing.

## Results: overall ranking and Top-K recovery

The PUF-ESM Specificity Ranker reached **Spearman ρ = 0.405** and **Kendall τ = 0.264**. It had the highest Spearman correlation among the original 12 methods in the matched comparison and among the five feature ablations below. This scope does not establish it as the best historical model or the best method for every Top30 objective.

Removing the evolutionary feature reduced Spearman correlation to 0.292 in the matched one-hot Ridge model using the same A selection rule. Frozen local ESM plus Ridge reached 0.289 under its existing B selection rule. The latter comparison changes both features and selection strategy, so it does not isolate the contribution of evolutionary augmentation.

![Observed Top5, Top15 and Top30 overlap versus random expectation](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/topk_random/figure@2x.png)

*Figure 2. Top-K recovery in development-stage cross-validation for K = 5, 15 and 30. A hit appears in both the predicted and measured Top-K sets among the same 66 constructs. Model labels show integer hit counts: 1, 6 and 20. Uniform random selection gives expected overlap K²/66: 0.38, 3.41 and 13.64. Fractional baseline values are expectations, not observed counts.*

The model recovered 1, 6 and 20 measured Top-K members at K = 5, 15 and 30, respectively. The displayed cutoffs exceeded their random expectations. At Top30, the model recovered 20 of the 30 highest-scoring constructs, corresponding to a 66.7% hit rate. Uniform random selection would recover 13.64 on average, or 45.5%. The selected constructs had a measured mean S of 0.292, compared with 0.245 across all 66 constructs.

Top30 was chosen for reporting after inspection of the existing predictions. This retrospective choice leaves model fitting and its original inner-selection rules unchanged. The comparison is descriptive development-stage evidence, not an independently validated improvement. Random selection here is not random generation of mutant sequences.

![Measured versus held-out predicted specificity for all 66 constructs](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/oof_scatter/figure@2x.png)

*Figure 3. Measured and out-of-fold predicted S for all 66 complete constructs. The predicted Top30 is highlighted; the remaining 36 constructs are shown in grey. Every point uses a prediction from a model that excluded its sequence group. The dashed identity line is a visual reference, not a fitted regression. Spearman ρ = 0.405; Kendall τ = 0.264.*

### Feature ablations: position, amino-acid identity and evolutionary score

We tested whether ranking depended on mutation location, amino-acid identity or the evolutionary score. The position-only baseline uses a 493-dimensional binary vector, with one indicating a substitution at that residue. It contains no amino-acid identity. The amino-acid model uses the existing 493 × 20 reference-centred one-hot difference encoding: −1 for the reference residue and +1 for its replacement at each mutated position.

All supervised variants predict the three C values separately, using available labels from the same 81-construct pool. They reuse the identical 66 outer sequence-group folds, inner partitions, Ridge penalty grid and A selection rule. Test groups remain excluded from every training record. The historical inner-selection rule is unchanged; Top30 is the retrospective reporting cutoff.

The zero-shot control ranks the fixed ESM-2 mutation score directly, without using the current assay labels for training, scaling or sign selection. Higher pretrained log-odds remain ranked higher. An additional score-only Ridge control tests whether supervised calibration of that single feature is sufficient.

![Spearman correlation across five feature ablations](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/ablation_spearman/figure@2x.png)

*Figure 4. Feature ablations evaluated on the same 66 complete constructs. Supervised models use identical grouped outer and inner validation partitions and A selection. The zero-shot score uses no assay fitting. Bars show cohort-level Spearman correlation, with zero as a reference. No uncertainty intervals or significance tests are inferred from these point estimates.*

![Top30 overlap across five feature ablations versus random expectation](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/ablation_top30/figure@2x.png)

*Figure 5. Top30 overlap between predicted and measured sets for each ablation. The dashed line marks the uniform random-selection expectation of 30²/66 = 13.64 hits. Position-only Ridge recovers 14, amino-acid one-hot Ridge 16, zero-shot ESM-2 9, score-only Ridge 16 and the combined model 20. All candidates and the reporting cutoff are shared.*

**Mutation position alone provides limited ranking information.** Position-only Ridge reached ρ = 0.155 and recovered 14 of the measured Top30, close to the random expectation of 13.64. Its selected mean S was 0.241, compared with 0.245 across the full cohort. Location alone therefore did not provide strong candidate enrichment in this evaluation.

**Amino-acid identity improves the point estimates.** Adding identity through position-specific one-hot features increased ρ from 0.155 to 0.292 and Top30 recovery from 14 to 16. These results suggest that the substituted residue contributes information beyond mutation location within this library.

**The uncalibrated ESM-2 score does not directly rank specificity well.** Its correlation was negative (ρ = −0.265), and Top30 recovery was 9, below the random expectation. The score was not sign-flipped after inspecting these results. This finding applies to the specified additive reference-masked score, not to every possible ESM representation. Fitting Ridge to the score increased recovery to 16 but yielded only ρ = 0.098.

**Combining the two feature sources performs best within this ablation.** The combined model reached ρ = 0.405 and recovered 20 of the measured Top30. Relative to amino-acid one-hot Ridge, this corresponds to Δρ = 0.113 and four additional hits; relative to score-only Ridge, Δρ = 0.307 and four additional hits. The combined selected mean S was 0.292, versus 0.259 for one-hot Ridge and 0.264 for score-only Ridge. Thus, this comparison supports a benefit from combining the features after assay-based fitting, despite poor direct zero-shot ranking.

These are descriptive differences from development-stage cross-validation, not evidence of statistically established improvement or independent experimental validation. Sequence-group holdout can retain examples at the same mutation positions in training; it does not test generalization to entirely unseen positions. Independent experiments remain necessary.

### Original five constructs with high measured specificity

We retain the original five highlighted constructs alongside the rank-agreement examples below. All five belong to both the predicted and measured Top30 sets. They were highlighted for their measured specificity, rather than selected for the smallest rank errors; their prediction ranks differ from measured ranks by 7–20 places.

![Predicted and measured ranks of the original five high-specificity constructs](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/five_targets/figure@2x.png)

*Figure 6. The original five highlighted constructs, ranked among the same 66 complete constructs. All enter the predicted Top30, although their exact ranks differ from measurement. Each prediction excludes its own sequence group from training. Existing measurements provide retrospective comparison, not independent prospective validation.*

| Construct | Predicted S | Measured S | Predicted rank / 66 | Measured rank / 66 | Predicted Top30 |
|---|---:|---:|---:|---:|:---:|
| P8-GVE | 0.240 | 0.481 | 22 | 2 | Yes |
| P9-NTQ | 0.245 | 0.422 | 19 | 6 | Yes |
| P9-NPS | 0.270 | 0.375 | 15 | 8 | Yes |
| P9-NSG | 0.233 | 0.345 | 24 | 11 | Yes |
| P9-GNS | 0.339 | 0.342 | 4 | 13 | Yes |

### Retrospective examples of agreement at higher and lower ranks

We selected five examples with higher specificity and five with lower specificity, excluding the five previously highlighted constructs. Higher-ranked examples must appear in both predicted and measured Top30 sets. Lower-ranked examples must appear in both bottom-30 sets, corresponding to ranks 37–66.

Within each eligible group, we chose the five smallest absolute differences between predicted and measured ranks, breaking ties by absolute S error and then construct identifier. This selection uses measured outcomes after prediction. The examples illustrate agreement and must not replace the full-cohort performance assessment. Lower specificity here means a lower relative S within this library, not S ≤ 0 or a non-working construct.

![Five concordant higher-ranked examples followed by five concordant lower-ranked examples](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/selected_cases/figure@2x.png)

*Figure 7. Post hoc rank-agreement examples for PUF-ESM among 66 complete constructs. The first five cases belong to both Top30 sets; the final five belong to both bottom-30 sets. Every prediction excludes its own sequence group during training. Cases were selected for small absolute rank errors, so their agreement is not representative of the whole cohort. Existing measurements provide retrospective comparison, not subsequent independent validation.*

#### Five higher-ranked examples

| Construct | Predicted S | Measured S | Predicted rank / 66 | Measured rank / 66 | Absolute rank error |
|---|---:|---:|---:|---:|---:|
| P9-NFP | 0.270 | 0.336 | 14 | 14 | 0 |
| P4-SNE + P7-SNE | 0.406 | 0.541 | 2 | 1 | 1 |
| P1-GVD | 0.243 | 0.257 | 21 | 23 | 2 |
| P9-QKQ | 0.266 | 0.264 | 16 | 19 | 3 |
| P9-NWP | 0.244 | 0.288 | 20 | 17 | 3 |

#### Five lower-ranked examples

| Construct | Predicted S | Measured S | Predicted rank / 66 | Measured rank / 66 | Absolute rank error |
|---|---:|---:|---:|---:|---:|
| P2-CKR | 0.180 | 0.133 | 60 | 60 | 0 |
| P5-AYE | 0.211 | 0.201 | 42 | 44 | 2 |
| P2-CRR | 0.176 | 0.106 | 61 | 63 | 2 |
| P2-TKR | 0.166 | 0.140 | 62 | 59 | 3 |
| P10-SRQ | 0.201 | 0.198 | 49 | 45 | 4 |

The higher-ranked examples differ from their measured ranks by 0–3 places; the lower-ranked examples differ by 0–4 places. Close ranks do not necessarily imply accurate absolute S values. For example, P4-SNE + P7-SNE ranks second by prediction and first by measurement, but its predicted S (0.406) underestimates the measured S (0.541).

The original five-target data remain available in the [original target summary](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/five_target_summary.csv). The [ten selected cases](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/selected_rank_cases.csv) and [explicit selection rule](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/case_selection_protocol.json) make this illustrative selection auditable. Full-cohort Spearman and Top-K results are unchanged.

### Sequence-only inference and next experimental cycle

The deployment model uses the prespecified A selection rule on all available training data. Its fitted weights are separate from the out-of-fold performance evidence. New candidates require only a 493-residue PUF FASTA sequence with standard amino acids; the fixed reference-scoring table is stored with the model. Insertions and deletions are outside the current model scope.

These data have been examined across multiple modelling rounds. All reported metrics therefore represent development-stage cross-validation. The next Design–Build–Test–Learn cycle should prospectively select candidates, measure all three C values, and compare predictions with an independent experimental batch.

## Data, figures and reproducibility

The [ablation comparison](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/ablation_metrics.csv), [per-construct predictions](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/oof_ablation.csv), [locked ablation protocol](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/ablation_protocol.json) and [validation record](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/ablation_validation.json) accompany the new figures. The `ablation_experiment/` directory contains inputs, training code and all 66 fold-level parameter searches.

The package contains [all 66 ranked predictions](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/hsu_A_oof_rankings.csv), the [five-target audit](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/five_target_summary.csv), the [Top-K baseline comparison](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/topk_random_comparison.csv), [all-method metrics](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/comparison_all66.csv) and the [fixed protocol](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/protocol.json). Batch-specific and double-repeat subset metrics are retained in [the complete subset table](https://github.com/pdx12320/igem_module2/blob/main/wiki/data/metrics_all_subsets.csv). Top30 is marked not evaluable for subsets with fewer than 30 constructs; it is not silently replaced with a smaller K.

The workflow is supplied as editable PPTX, with a Wiki PNG. Data figures include PNG, SVG, PDF, captions, alt text and reproducible plotting code. This revision adds a position-only baseline and reruns the supervised feature ablations. The three previously trained feature models reproduce the stored predictions within numerical tolerance; no new assay measurements were added. The Non-TRM section below retains the supplied text and evidence and was not reanalysed.

## Non-TRM design

### Extending design beyond RNA recognition

The measured editing profiles provide experimentally characterized TRM backgrounds for additional design. We next examined substitutions outside the recognition code. Such substitutions offer testable hypotheses about scaffold organization and the protein–RNA interface while preserving the chosen TRMs.

We adapted the AiCE framework, AI-informed constraints for protein engineering, to structure-conditioned sequence sampling (Fei et al., 2025). Pretrained inverse-folding models nominate mutations, while measured editing profiles guide background selection and interpretation.

### The inverse-folding workflow

| Step | Input and operation | Output |
|---|---|---|
| **1. Define the structural reference** | Use the predicted 493-residue PUF12 complex with a 17-nt APOE4 RNA segment; map repeats and recognition positions | A shared protein reference and explicit residue numbering |
| **2. Sample compatible sequences** | Generate 10,000 sequences per model at temperature 0.5 | ProteinMPNN and LigandMPNN sequence ensembles |
| **3. Scan every position** | Count WT and alternative amino-acid frequencies across all 493 positions | Position-specific sequence preferences |
| **4. Protect the recognition code** | Apply the configured recognition-position filter after sampling and retain supported non-TRM changes | 81 nominated positions, including 17 same-substitution dual-model nominations |
| **5. Combine with experimental backgrounds** | Add selected non-TRM substitutions to P8-GVE, P9-NTQ or P7-SYVIRR | Background-specific mutation proposals |
| **6. Return to the wet lab** | Compare candidates with their backgrounds using C295, C388 and C871 measurements | A direct test of the proposed substitutions |

ProteinMPNN conditions sequence generation on the protein backbone (Dauparas et al., 2022). LigandMPNN additionally uses the surrounding atomic context, including RNA atoms (Dauparas et al., 2025). Both use the same structural reference, with different conditioning information.

Sampling covered the protein sequence without fixing TRM positions during generation. Recognition positions were protected during candidate filtering and construction of the final sequences. For this run, the implemented frequency filter used a threshold of 0.8; all exported positions had the flexible-region flag set to false. No task-specific retraining of either MPNN model was performed.

### Seventeen consensus substitutions

The screen identified **17 substitutions supported by both inverse-folding models**. Both models favoured the same alternative amino acid at these positions. The table separates substitutions used in the proposed core set from further consensus candidates.

| Design use | Exact substitutions | Interpretation |
|---|---|---|
| Consensus set used in the proposed core combinations | **T350V, D374E, M458L, L274I, T278I, V366I** | Test a combined change to the surrounding scaffold |
| N-terminal-region candidate | **R45T** | Test an additional change near the fusion-end region |
| Further dual-model nominations | **V294I, L166I, R93K, E351L, R462L, V208E, A136E, V388E, A244E, S100E** | Extend the candidate pool with single-mutant comparisons |
| Additional interface proposals | A438G, H392D | Supported by one model at the stated threshold |
| Additional N-terminal proposals | S30A, L41R; N12S as an exploratory design choice | Included in geometry combinations; N12S has the nomination-label discrepancy described below |

![Consensus non-TRM substitutions](https://raw.githubusercontent.com/pdx12320/igem_module2/main/wiki/figures/fig5_model3_nontrm_consensus.png)


The broader 81-position export requires an additional identity check. At N12, the ProteinMPNN frequency above the threshold supports **N12G**, while the proposed construct contains **N12S**. We retain N12S as an exploratory design choice and document the discrepancy in the [Non-TRM design methods page](https://github.com/pdx12320/PUF_alphafold/blob/b5a7aaa7ef8426d1c98243d1a09c0db9a5db1c1a/research/aice_mpnn_20260922/README.md). Neither substitution has a measured editing benefit in this release.

Residue numbers refer to the full **493-residue reference protein**. N12 identifies its twelfth residue; TRM positions 12, 13 and 16 refer to positions within a repeat. Numbering should be mapped explicitly when using constructs with different terminal sequences.

### Repeat-level preferences and candidate locations

Non-TRM design provides two complementary position-level readouts. First, retention of the WT recognition residues indicates which repeat codes the sampled sequences favour. LigandMPNN retains WT residues at the three TRM positions most strongly in **P9 (90.4%), P10 (71.1%), P6 (64.6%) and P3 (63.0%)**. ProteinMPNN also ranks **P9** highest, at **43.8%**. These values are the mean WT frequency across the three TRM positions in each repeat; they describe sequence preferences under the structural input.

Second, the non-TRM screen identifies positions with a strongly favoured alternative residue. **P10 contains four of the 17 dual-model consensus substitutions**; P4, P7, P8 and P12 each contain two. The complete mapping is:

| Repeat position | Dual-model consensus non-TRM substitutions |
|---|---|
| P1 | R45T |
| P2 | R93K |
| P3 | S100E |
| P4 | L166I, A136E |
| P6 | V208E |
| P7 | L274I, A244E |
| P8 | V294I, T278I |
| P10 | T350V, D374E, V366I, E351L |
| P11 | V388E |
| P12 | M458L, R462L |

P10 combines high WT retention at recognition residues with several nominated non-TRM changes, providing a defined starting point for experiments. P3 also showed recognition-site retention in the inverse-folding analysis. These readouts describe structural sequence preferences and do not constitute residue attribution for the ESM models.

### Exact mutation matches to the wet-lab results

We compared the model outputs with **the same amino-acid substitutions measured experimentally**, using the verified residue alignment and matched controls. None of the 17 consensus non-TRM substitutions has an exact match in the current 81-variant assay collection. Their experimental validation therefore remains a subsequent design step.

The sampled sequences also allow a direct comparison with experimentally tested **TRM substitutions**, before the non-TRM protection filter is applied. The positive examples below retain C388 within 10 pp of its matched control or improve it, reduce C871 by at least 10 pp, and do not increase C295. They are descriptive assay examples selected by this stated rule; the complete comparison includes all 81 variants.

| Experimentally tested variant | Exact substitutions in the 493-residue reference | ΔC388 (pp) | ΔC295 (pp) | ΔC871 (pp) |
|---|---|---:|---:|---:|
| **P1-SHE** | Y37H | -9.22 | -5.65 | -14.49 |
| **P4-SNE** | Y145N | +1.75 | -20.49 | -18.80 |
| **P4-SNE + P7-SNE** | Y145N, Y253N | +7.77 | -13.15 | -34.90 |
| **P9-NTQ** | Y325T | -4.10 | -8.55 | -38.60 |
| **P8-GVE** | S288G, Y289V | -6.70 | -7.73 | -57.27 |

The P4/P7 combination is a favourable three-site example: C388 editing rises by **7.77 pp**, while C295 and C871 fall by **13.15 pp** and **34.90 pp**. P8-GVE produces the largest C871 reduction among these displayed examples (**57.27 pp**) with a **6.70 pp** reduction in C388. P9-NTQ retains C388 within **4.10 pp** of its control while reducing both bystander sites.

To compare these exact mutations with the inverse-folding outputs, we counted generated sequences carrying every listed changed residue and compared that count with WT residues at the same positions. Each model contributes 10,000 sampled sequences; the rest of each sequence is unconstrained.

| Exact experimental substitution set | ProteinMPNN: mutant / WT count | LigandMPNN: mutant / WT count |
|---|---:|---:|
| P1-SHE: Y37H | 22 / 5 | 2 / 0 |
| P4-SNE: Y145N | 569 / 839 | 0 / 1889 |
| P4-SNE + P7-SNE: Y145N, Y253N | 120 / 359 | 0 / 461 |
| P9-NTQ: Y325T | 19 / 1017 | 1 / 7464 |
| P8-GVE: S288G, Y289V | 0 / 305 | 0 / 3006 |

**P1-SHE (Y37H)** provides a limited example of agreement: the mutant residue occurs more often than WT at that position in both sampling models and has a favourable measured bystander profile. Its absolute frequency is low—**0.22%** in ProteinMPNN and **0.02%** in LigandMPNN—well below the 80% non-TRM nomination threshold. Several strong experimental variants receive low or zero exact-set counts, including P8-GVE. The comparison therefore supports their measured utility while showing that sampling frequency alone does not rank editing performance reliably.

### Structural preferences and editing outcomes

Agreement with favourable measured TRM variants is limited. ProteinMPNN and LigandMPNN generate sequences compatible with a supplied structure; they were not trained here on editing outcomes. The PUF–RNA input also omits the complete PUF–APOBEC fusion and cellular context, which can affect catalytic positioning and editing. Moreover, the sampled sequences contain multiple substitutions, so their frequencies do not isolate the effect of a single experimental mutation. These methodological differences plausibly explain the mismatch; they have not been established as its experimental causes. A zero count means no occurrence in 10,000 samples. The 17 non-TRM nominations remain candidates for wet-lab testing.

The [complete mutation-to-assay table](https://github.com/pdx12320/PUF_alphafold/blob/b5a7aaa7ef8426d1c98243d1a09c0db9a5db1c1a/research/model_interpretation_20260923/model3_exact_mutation_wetlab_comparison.csv) retains every construct, matched-control change and exact sampling count. The [analysis methods](https://github.com/pdx12320/PUF_alphafold/blob/b5a7aaa7ef8426d1c98243d1a09c0db9a5db1c1a/research/model_interpretation_20260923/README.md) explain the numbering verification, sample parsing and example-selection rule.

## References

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., Bodenstein, S. W., Evans, D. A., Hung, C.-C., O’Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., Žemgulytė, A., Arvaniti, E., … Jumper, J. M. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature, 630*(8016), 493–500. https://doi.org/10.1038/s41586-024-07487-w

Cheong, C.-G., & Hall, T. M. T. (2006). Engineering RNA sequence specificity of Pumilio repeats. *Proceedings of the National Academy of Sciences, 103*(37), 13635–13639. https://doi.org/10.1073/pnas.0606294103

Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R. J., Milles, L. F., Wicky, B. I. M., Courbet, A., de Haas, R. J., Bethel, N., Leung, P. J. Y., Huddy, T. F., Pellock, S., Tischer, D., Chan, F., Koepnick, B., Nguyen, H., Kang, A., Sankaran, B., … Baker, D. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science, 378*(6615), 49–56. https://doi.org/10.1126/science.add2187

Dauparas, J., Lee, G. R., Pecoraro, R., An, L., Anishchenko, I., Glasscock, C., & Baker, D. (2025). Atomic context-conditioned protein sequence design using LigandMPNN. *Nature Methods, 22*(4), 717–723. https://doi.org/10.1038/s41592-025-02626-1

Fei, H., Li, Y., Liu, Y., Wei, J., Chen, A., & Gao, C. (2025). Advancing protein evolution with inverse folding models integrating structural and evolutionary constraints. *Cell, 188*(17), 4674–4692.e19. https://doi.org/10.1016/j.cell.2025.06.014

Radivojević, T., Costello, Z., Workman, K., & Garcia Martin, H. (2020). A machine learning Automated Recommendation Tool for synthetic biology. *Nature Communications, 11*, Article 4879. https://doi.org/10.1038/s41467-020-18008-4

Wang, X., McLachlan, J., Zamore, P. D., & Hall, T. M. T. (2002). Modular recognition of RNA by a human Pumilio-homology domain. *Cell, 110*(4), 501–512. https://doi.org/10.1016/S0092-8674(02)00873-5

Zhao, Y.-Y., Mao, M.-W., Zhang, W.-J., Wang, J., Li, H.-T., Yang, Y., Wang, Z., & Wu, J.-W. (2018). Expanding RNA binding specificity and affinity of engineered PUF domains. *Nucleic Acids Research, 46*(9), 4771–4782. https://doi.org/10.1093/nar/gky134

Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science, 379*, 1123–1130. https://doi.org/10.1126/science.ade2574

Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685

Hsu, C., Nisonoff, H., Fannjiang, C., & Listgarten, J. (2022). Learning protein fitness models from evolutionary and assay-labeled data. *Nature Biotechnology, 40*, 1114–1122. https://doi.org/10.1038/s41587-021-01146-5
