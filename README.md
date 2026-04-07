# NER_MODEL_torch

PyTorch NER baselines extracted and refactored from `NER_MODEL`.

## Current status

This repo currently includes runnable torch baselines with:

- **BiLSTM-CRF**
- **BiLSTM-CNN-CRF**
- **IDCNN-CRF**
- **Transformer-CRF**

Shared capabilities:

- PyTorch implementation
- Training entry
- Evaluation entry
- Prediction entry
- Fake-data generator for smoke tests
- Dataset adapters for `jsonl` and `conll/bio`
- End-to-end test script

## Structure

- `BILSTM_CRF/` - torch implementation and docs
- `BILSTM_CNN_CRF/` - torch implementation and docs
- `transformer_crf/` - torch implementation and docs
- `test_bilstm_crf_e2e.py` - BiLSTM-CRF validation script
- `test_bilstm_cnn_crf_e2e.py` - BiLSTM-CNN-CRF validation script
- `test_idcnn_crf_e2e.py` - IDCNN-CRF validation script
- `test_transformer_crf_e2e.py` - Transformer-CRF validation script

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

## Verified synthetic-data experiment

A reproducible synthetic-data run has been completed in this repo with:

- train size: `500`
- dev size: `100`
- epochs: `5`

Observed results:

- Epoch 1: `train loss=7.263347`, `dev f1=0.9771`
- Epoch 2: `train loss=0.750803`, `dev f1=1.0000`
- Epoch 3: `train loss=0.192112`, `dev f1=1.0000`
- Epoch 4: `train loss=0.075270`, `dev f1=1.0000`
- Epoch 5: `train loss=0.044115`, `dev f1=1.0000`

Standalone evaluation:

- `acc=1.0000`
- `prec=1.0000`
- `recall=1.0000`
- `f1=1.0000`
- `support=963`

Prediction smoke test:

- input: `请问肯德基优惠券在哪里`
- output: `肯德基 , 优惠券`

## Notes

- Current migration scope: `BILSTM_CRF` only.
- Push marker: initial release.
- Fake data is only for pipeline validation, not for real-world benchmark claims.
- On synthetic data, perfect scores mainly indicate that the pipeline is correct and the task pattern is learnable.
- `BILSTM_CNN_CRF` has also been migrated and smoke-tested in this repo.
- `IDCNN_CRF` has also been migrated and smoke-tested in this repo.
- `transformer_crf` has also been migrated and end-to-end validated in this repo.
- `transformer_crf` synthetic e2e result: `acc=0.9946`, `prec=1.0000`, `recall=0.9896`, `f1=0.9948`.
- `transformer_crf` currently aligns its synthetic validation label space to `O/B-BRD/I-BRD/B-KWD/I-KWD`.
- Real-data effect is still unverified for all baselines.
- Other models from the original repo are not yet migrated in this repo.
