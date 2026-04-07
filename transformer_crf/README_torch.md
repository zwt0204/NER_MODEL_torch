# transformer_crf Torch Version

PyTorch implementation of `transformer_crf` using:

- char embedding
- positional encoding
- `nn.TransformerEncoder`
- linear classifier
- CRF (`TorchCRF`)

This version keeps the old project intent (`TransformerEncoder -> Linear -> CRF`) while reusing the same torch training / evaluation / prediction skeleton used by the other migrated baselines.

## Current validation label space

For the current fake-data e2e validation, the active label set is aligned to the synthetic generator:

- `O`
- `B-BRD`
- `I-BRD`
- `B-KWD`
- `I-KWD`

## Partially aligned settings from the old implementation

- `sequence_length = 50`
- `num_heads = 8`
- `dropout_rate = 0.1`
- `batch_size = 128`
- `lr = 0.0001`

## Quick start

```bash
pip install -r transformer_crf/requirements.txt
python -m transformer_crf.generate_fake_data
python -m transformer_crf.train \
  --vocab-file transformer_crf/demo_data/vocab.json \
  --train-file transformer_crf/demo_data/train.json \
  --train-format jsonl \
  --dev-file transformer_crf/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir transformer_crf/demo_data/model \
  --epochs 5
python -m transformer_crf.evaluate \
  --vocab-file transformer_crf/demo_data/vocab.json \
  --model-dir transformer_crf/demo_data/model \
  --eval-file transformer_crf/demo_data/dev.json \
  --eval-format jsonl
python -m transformer_crf.predict \
  --vocab-file transformer_crf/demo_data/vocab.json \
  --model-path transformer_crf/demo_data/model/ner.pt \
  --text "高勇，男，中国国籍，无境外居留权"
```

## Notes

- `TorchCRF` is used, not `torchcrf`.
- Fake data is only for pipeline validation, not for business quality claims.
- The old path `models\Transformer` is normalized to portable paths such as `models/Transformer` or custom `--model-dir`.
