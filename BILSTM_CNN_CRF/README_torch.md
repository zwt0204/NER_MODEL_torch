# BILSTM_CNN_CRF Torch Version

PyTorch implementation of `BILSTM_CNN_CRF` using:

- Embedding
- 1D CNN
- BiLSTM
- CRF

## Quick start

```bash
pip install -r BILSTM_CNN_CRF/requirements.txt
python -m BILSTM_CNN_CRF.generate_fake_data
python -m BILSTM_CNN_CRF.train \
  --vocab-file BILSTM_CNN_CRF/demo_data/vocab.json \
  --train-file BILSTM_CNN_CRF/demo_data/train.json \
  --train-format jsonl \
  --dev-file BILSTM_CNN_CRF/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir BILSTM_CNN_CRF/demo_data/model \
  --epochs 5
python -m BILSTM_CNN_CRF.evaluate \
  --vocab-file BILSTM_CNN_CRF/demo_data/vocab.json \
  --model-dir BILSTM_CNN_CRF/demo_data/model \
  --eval-file BILSTM_CNN_CRF/demo_data/dev.json \
  --eval-format jsonl
python -m BILSTM_CNN_CRF.predict \
  --vocab-file BILSTM_CNN_CRF/demo_data/vocab.json \
  --model-path BILSTM_CNN_CRF/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```

## Notes

- Data adapters support `jsonl` and `conll/bio`.
- Synthetic data is for pipeline validation only.
