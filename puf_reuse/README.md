# PUF-ESM Specificity Ranker

## What is included

- `predict.py` and `model.py`: batch prediction and feature/Ridge implementation.
- `weights/PUF_ESM_Specificity_Ranker.joblib`: the existing full-data refit, selected using training-side strategy A. This is not an OOF model or performance estimate.
- `examples/example.fasta`: three previously measured examples for interface checking, not independent test data.
- `requirements.txt`: a pinned Python environment; Python 3.11 is recommended.
- `../wiki/ablation_experiment/`: fixed-data training/CV reproduction, inputs, splits and fold records.

## Install and predict

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r puf_reuse/requirements.txt
python puf_reuse/predict.py \
  --model puf_reuse/weights/PUF_ESM_Specificity_Ranker.joblib \
  --fasta puf_reuse/examples/example.fasta \
  --output example_predictions.csv
```

For new candidates, change --fasta to your input file and use a new output filename. Existing output files are not overwritten.

## Input and output contract

Each FASTA record must have a unique nonempty identifier and exactly 493 uppercase standard amino acids. Sequences must use the same PUF reference alignment; substitutions are supported, insertions/deletions are not. No structure file is needed. The weight file includes the fixed reference masked-log-probability table, so prediction does not rerun ESM or need a GPU.

CSV output contains the candidate ID, predicted C295/C388/C871, S = C388 − (C295 + C871)/2, ESM score, rank within the input batch and classifier status. Ridge predictions are not clipped; a prediction is not an experimentally validated editing measurement. The specificity classifier remains untrained because no negative class was available.

## Reproduce the reported validation

Follow [the training guide](../wiki/ablation_experiment/README.md). A new --output directory runs fresh grouped CV; default bundled fold files are cached results. Do not interpret fitting predictions from the supplied all-data weight file as validation performance.

Only load trusted joblib files: this serialization format can execute code. `SHA256SUMS.json` records package file hashes. No credentials, server access or user-specific paths are needed.

The public model name is **PUF-ESM Specificity Ranker** (short name **PUF-ESM**). The former weight filename remains as a compatibility alias with identical bytes. Historical experiment identifiers are retained for reproducibility; no weights, training rules or prediction values changed in this rename.
