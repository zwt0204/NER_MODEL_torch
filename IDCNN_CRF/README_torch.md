# IDCNN_CRF Torch Version

PyTorch implementation of `IDCNN_CRF` using:

- char embedding
- simple segmentation embedding
- stacked dilated convolutions
- CRF

## Quick start

```bash
pip install -r IDCNN_CRF/requirements.txt
python -m IDCNN_CRF.generate_fake_data
python -m IDCNN_CRF.train \
  --vocab-file IDCNN_CRF/demo_data/vocab.json \
  --train-file IDCNN_CRF/demo_data/train.json \
  --train-format jsonl \
  --dev-file IDCNN_CRF/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir IDCNN_CRF/demo_data/model \
  --epochs 5
python -m IDCNN_CRF.evaluate \
  --vocab-file IDCNN_CRF/demo_data/vocab.json \
  --model-dir IDCNN_CRF/demo_data/model \
  --eval-file IDCNN_CRF/demo_data/dev.json \
  --eval-format jsonl
python -m IDCNN_CRF.predict \
  --vocab-file IDCNN_CRF/demo_data/vocab.json \
  --model-path IDCNN_CRF/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```
