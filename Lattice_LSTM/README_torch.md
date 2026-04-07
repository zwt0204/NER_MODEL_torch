# Lattice_LSTM Torch Version

PyTorch simplified lattice-style baseline implementation of `Lattice_LSTM`.

## Important scope note

The original TensorFlow project contains a custom `LatticeLSTMCell` together with
`gazetteer / trie / alphabet` preprocessing modules.

This torch migration does **not** fully re-implement that custom recurrent cell or the
full original preprocessing stack.

Instead, to keep migration scope controlled while preserving the high-level modeling idea,
this version uses a **simplified lattice-style feature fusion baseline**:

- char embedding
- lexicon feature aggregation per position
- BiLSTM
- linear classifier
- CRF (`TorchCRF`)

So this version is suitable for:

- PyTorch pipeline migration
- train / evaluate / predict entry alignment
- synthetic-data e2e verification
- a practical baseline that still includes lexicon-style features

But it is **not** a claim of exact equivalence with the original TensorFlow `LatticeLSTMCell`.

## Current validation label space

For synthetic validation, the active label set is:

- `O`
- `B-BRD`
- `I-BRD`
- `B-KWD`
- `I-KWD`

## Quick start

```bash
pip install -r Lattice_LSTM/requirements.txt
python -m Lattice_LSTM.generate_fake_data
python -m Lattice_LSTM.train \
  --vocab-file Lattice_LSTM/demo_data/vocab.json \
  --train-file Lattice_LSTM/demo_data/train.json \
  --train-format jsonl \
  --dev-file Lattice_LSTM/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir Lattice_LSTM/demo_data/model \
  --epochs 8
python -m Lattice_LSTM.evaluate \
  --vocab-file Lattice_LSTM/demo_data/vocab.json \
  --model-dir Lattice_LSTM/demo_data/model \
  --eval-file Lattice_LSTM/demo_data/dev.json \
  --eval-format jsonl
python -m Lattice_LSTM.predict \
  --vocab-file Lattice_LSTM/demo_data/vocab.json \
  --model-path Lattice_LSTM/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```

## Notes

- `TorchCRF` is used, not `torchcrf`.
- Lexicon features are simplified and generated from local spans in the input text.
- Full original gazetteer/trie/alphabet behavior is not fully reproduced in this baseline.
- Fake data is only for pipeline validation, not for real business quality claims.
