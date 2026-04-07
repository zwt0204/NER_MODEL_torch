# NER_MODEL_torch

PyTorch NER baselines extracted and refactored from `NER_MODEL`.

## Current status

This repo currently includes runnable torch baselines with:

- **BiLSTM-CRF**
- **BiLSTM-CNN-CRF**
- **IDCNN-CRF**
- **Transformer-CRF**
- **Lattice-LSTM**
- **Albert-CRF NER** (current environment uses an ALBERT-style stub encoder, not real pretrained `AlbertModel` weights)
- **Albert-BiLSTM-CRF** (current environment uses an ALBERT-style stub encoder, not real pretrained `AlbertModel` weights)

Shared capabilities:

- PyTorch implementation
- Training entry
- Evaluation entry
- Prediction entry
- Fake-data generator for smoke tests
- Dataset adapters for `jsonl` and `conll/bio`
- End-to-end test script

## Migration status matrix

| Model | Torch implementation | Synthetic e2e verified | Real pretrained backbone | Real-data verified |
| --- | --- | --- | --- | --- |
| `BILSTM_CRF` | Yes | Yes | N/A | No |
| `BILSTM_CNN_CRF` | Yes | Yes | N/A | No |
| `IDCNN_CRF` | Yes | Yes | N/A | No |
| `transformer_crf` | Yes | Yes | N/A | No |
| `Lattice_LSTM` | Yes | Yes | N/A | No |
| `albert_crf_ner` | Yes | Yes | No — current version is a stub ALBERT-style encoder baseline | No |
| `albert_bisltm_crf` | Yes | Yes | No — current version is a stub ALBERT-style encoder baseline | No |

Interpretation:

- **Torch implementation** means train / evaluate / predict flow is runnable in this repo.
- **Synthetic e2e verified** means fake-data generation, training, evaluation, and prediction were run end-to-end successfully.
- **Real pretrained backbone** is only relevant for transformer / ALBERT-style families. The two ALBERT directories are currently pipeline-compatible baselines, not true HuggingFace pretrained-weight integrations.
- **Real-data verified** is still `No` across the board.

## Structure

- `BILSTM_CRF/` - torch implementation and docs
- `BILSTM_CNN_CRF/` - torch implementation and docs
- `IDCNN_CRF/` - torch implementation and docs
- `transformer_crf/` - torch implementation and docs
- `Lattice_LSTM/` - torch implementation and docs
- `albert_crf_ner/` - ALBERT-style stub encoder + CRF baseline and docs
- `albert_bisltm_crf/` - ALBERT-style stub encoder + BiLSTM + CRF baseline and docs
- `test_bilstm_crf_e2e.py` - BiLSTM-CRF validation script
- `test_bilstm_cnn_crf_e2e.py` - BiLSTM-CNN-CRF validation script
- `test_idcnn_crf_e2e.py` - IDCNN-CRF validation script
- `test_transformer_crf_e2e.py` - Transformer-CRF validation script
- `test_Lattice_LSTM_e2e.py` - Lattice-LSTM validation script
- `test_albert_crf_ner_e2e.py` - Albert-CRF baseline validation script
- `test_albert_bisltm_crf_e2e.py` - Albert-BiLSTM-CRF baseline validation script

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

- Fake data is only for pipeline validation, not for real-world benchmark claims.
- On synthetic data, very high scores mainly indicate that the pipeline is correct and the task pattern is learnable.
- `transformer_crf` synthetic e2e result: `acc=0.9946`, `prec=1.0000`, `recall=0.9896`, `f1=0.9948`.
- `transformer_crf` currently aligns its synthetic validation label space to `O/B-BRD/I-BRD/B-KWD/I-KWD`.
- `Lattice_LSTM` has been moved closer to the original repo's trie / gazetteer data flow, and now uses a dedicated instance-builder stage for train / evaluate / predict. It is still **not** a faithful reimplementation of the original TensorFlow `LatticeLSTMCell`.
- `albert_crf_ner` and `albert_bisltm_crf` are currently **torch baselines with ALBERT-style stub encoders**. They are useful for migration continuity and synthetic e2e validation, but should **not** be described as real pretrained-ALBERT reproductions yet.
- Real-data effect is still unverified for all baselines.
