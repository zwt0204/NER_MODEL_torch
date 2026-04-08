# NER_MODEL_torch

PyTorch NER baselines extracted and refactored from `NER_MODEL`.

## Current status

This repo currently includes runnable torch baselines with:

- **BiLSTM-CRF**
- **BiLSTM-CNN-CRF**
- **IDCNN-CRF**
- **Transformer-CRF**
- **Lattice-LSTM**
- **Albert-CRF NER** (implemented on top of HuggingFace `AlbertModel`; default path uses a small random-initialized local config, with optional pretrained loading)
- **Albert-BiLSTM-CRF** (implemented on top of HuggingFace `AlbertModel` + BiLSTM; default path uses a small random-initialized local config, with optional pretrained loading)

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
| `albert_crf_ner` | Yes | Yes | Partial — real `AlbertModel` architecture is wired in, but pretrained loading / real-data effect is still unverified | No |
| `albert_bisltm_crf` | Yes | Yes | Partial — real `AlbertModel` + BiLSTM architecture is wired in, but pretrained loading / real-data effect is still unverified | No |

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
- `albert_crf_ner/` - HuggingFace `AlbertModel` + CRF baseline and docs
- `albert_bisltm_crf/` - HuggingFace `AlbertModel` + BiLSTM + CRF baseline and docs
- `test_bilstm_crf_e2e.py` - BiLSTM-CRF validation script
- `test_bilstm_cnn_crf_e2e.py` - BiLSTM-CNN-CRF validation script
- `test_idcnn_crf_e2e.py` - IDCNN-CRF validation script
- `test_transformer_crf_e2e.py` - Transformer-CRF validation script
- `test_Lattice_LSTM_e2e.py` - Lattice-LSTM validation script
- `test_albert_crf_ner_e2e.py` - Albert-CRF baseline validation script
- `test_albert_bisltm_crf_e2e.py` - Albert-BiLSTM-CRF validation script
- `test_albert_e2e_serial.py` - Serial runner for the two ALBERT-family e2e tests (recommended to avoid environment-level instability when running them concurrently)

All e2e scripts now apply a conservative runtime environment (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`) to reduce environment-level flakiness during synthetic validation.

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

Recommended ALBERT-family test execution:

```bash
python test_albert_e2e_serial.py
```

ALBERT-family note:

- `test_albert_crf_ner_e2e.py` and `test_albert_bisltm_crf_e2e.py` are both individually runnable.
- In the current environment, they are recommended to run **serially**, not concurrently, because concurrent execution has shown intermittent low-level runtime instability during `albert_bisltm_crf.evaluate`.

## Notes

- Fake data is only for pipeline validation, not for real-world benchmark claims.
- On synthetic data, very high scores mainly indicate that the pipeline is correct and the task pattern is learnable.
- `transformer_crf` synthetic e2e result: `acc=0.9946`, `prec=1.0000`, `recall=0.9896`, `f1=0.9948`.
- `transformer_crf` currently aligns its synthetic validation label space to `O/B-BRD/I-BRD/B-KWD/I-KWD`.
- `Lattice_LSTM` has been moved closer to the original repo's trie / gazetteer data flow, and now uses a dedicated instance-builder stage for train / evaluate / predict. It is still **not** a faithful reimplementation of the original TensorFlow `LatticeLSTMCell`.
- `albert_crf_ner` and `albert_bisltm_crf` now use real HuggingFace `AlbertModel` code paths. Their default runtime remains self-contained via a small random-initialized local config, and optional pretrained loading is available, but pretrained-weight behavior and real-data effect are still unverified.
- Real-data effect is still unverified for all baselines.
