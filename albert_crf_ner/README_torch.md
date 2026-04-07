# albert_crf_ner Torch Version

PyTorch baseline implementation of `albert_crf_ner`.

## Important scope note

This migration now uses a real HuggingFace **`AlbertModel`** implementation as the encoder backbone.

Current scope is intentionally conservative:

- the code path is now based on `transformers` + `AlbertModel`
- the default runtime path uses a **small random-initialized `AlbertConfig`** so the existing train / evaluate / predict pipeline can run without depending on an external model download
- this means the project has moved from a pure local stub encoder to a **real transformer implementation**, but it is still **not yet a verified pretrained-ALBERT reproduction**

So the current version is suitable for:

- PyTorch pipeline migration
- train / evaluate / predict entry alignment
- synthetic-data e2e verification on top of a real transformer architecture

But it should still **not** be described as a validated pretrained-ALBERT result yet.

## Current validation label space

For synthetic validation, the active label set is aligned to the shared fake-data generator:

- `O`
- `B-BRD`
- `I-BRD`
- `B-KWD`
- `I-KWD`

## Quick start

```bash
pip install -r albert_crf_ner/requirements.txt
python -m albert_crf_ner.generate_fake_data
python -m albert_crf_ner.train \
  --vocab-file albert_crf_ner/demo_data/vocab.json \
  --train-file albert_crf_ner/demo_data/train.json \
  --train-format jsonl \
  --dev-file albert_crf_ner/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir albert_crf_ner/demo_data/model \
  --epochs 5
python -m albert_crf_ner.evaluate \
  --vocab-file albert_crf_ner/demo_data/vocab.json \
  --model-dir albert_crf_ner/demo_data/model \
  --eval-file albert_crf_ner/demo_data/dev.json \
  --eval-format jsonl
python -m albert_crf_ner.predict \
  --vocab-file albert_crf_ner/demo_data/vocab.json \
  --model-path albert_crf_ner/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```

## Notes

- `TorchCRF` is used, not `torchcrf`.
- Fake data is only for pipeline validation, not for real business quality claims.
- The encoder implementation now depends on `transformers` and `AlbertModel`.
- The current default path uses a small random-initialized ALBERT config to keep the pipeline self-contained.
- Pretrained-weight loading / real-data effect is still unverified.
