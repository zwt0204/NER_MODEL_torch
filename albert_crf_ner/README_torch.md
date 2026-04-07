# albert_crf_ner Torch Version

PyTorch baseline implementation of `albert_crf_ner`.

## Important scope note

The current runtime environment does **not** provide the `transformers` package, so this migration does **not** load a real HuggingFace `AlbertModel` yet.

To keep the migration moving with minimal scope expansion, this version provides a **stub ALBERT-style encoder baseline** with the same high-level intent:

- encoder
- linear classifier
- CRF (`TorchCRF`)

This means the current implementation is suitable for:

- PyTorch pipeline migration
- train / evaluate / predict entry alignment
- synthetic-data e2e verification

But it is **not yet** a claim of real pretrained-ALBERT equivalence.

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
- Real ALBERT pretrained weight loading is still unimplemented in this environment.
