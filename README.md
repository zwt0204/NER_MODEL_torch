# NER_MODEL_torch

PyTorch NER baselines extracted and refactored from `NER_MODEL`.

## Current status

This repo currently includes a fully runnable **BiLSTM-CRF** baseline with:

- PyTorch implementation
- Training entry
- Evaluation entry
- Prediction entry
- Fake-data generator for smoke tests
- Dataset adapters for `jsonl` and `conll/bio`
- End-to-end test script

## Structure

- `BILSTM_CRF/` - torch implementation and docs
- `test_bilstm_crf_e2e.py` - end-to-end validation script

## Quick start

```bash
pip install -r BILSTM_CRF/requirements.txt
python -m BILSTM_CRF.generate_fake_data
python -m BILSTM_CRF.train \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --train-file BILSTM_CRF/demo_data/train.json \
  --train-format jsonl \
  --dev-file BILSTM_CRF/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir BILSTM_CRF/demo_data/model \
  --epochs 5
python -m BILSTM_CRF.evaluate \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --model-dir BILSTM_CRF/demo_data/model \
  --eval-file BILSTM_CRF/demo_data/dev.json \
  --eval-format jsonl
python -m BILSTM_CRF.predict \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --model-path BILSTM_CRF/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```

## Notes

- Current migration scope: `BILSTM_CRF` only.
- Push marker: initial release.
- Fake data is only for pipeline validation, not for real-world benchmark claims.
- Other models from the original repo are not yet migrated in this repo.
