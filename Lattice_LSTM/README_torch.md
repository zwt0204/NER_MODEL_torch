# Lattice_LSTM Torch Version

PyTorch simplified lattice-style baseline implementation of `Lattice_LSTM`.

## Important scope note

The original TensorFlow project contains a custom `LatticeLSTMCell` together with
`gazetteer / trie / alphabet` preprocessing modules.

This torch migration does **not** fully re-implement that custom recurrent cell.

However, compared with the previous baseline, this version now moves one step closer
to the original data flow by introducing a **real trie/gazetteer matching stage**
and a more unified **instance builder** layer:

- fake data generation writes `gazetteer.txt`
- train / evaluate / predict load the gazetteer
- lexicon features are built from trie prefix matching (`enumerate_match_list`)
- matched lexicon ids + lengths are fused with char features before BiLSTM + CRF
- sample construction is now centralized through `LatticeInstanceBuilder`

So this version is now:

- closer to the original `gazetteer/trie` pipeline than plain span heuristics
- closer to the original `data.py` organization than scattered per-script tensor building
- still **not** an exact re-implementation of the original TensorFlow `LatticeLSTMCell`

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
  --gazetteer-file Lattice_LSTM/demo_data/gazetteer.txt \
  --train-file Lattice_LSTM/demo_data/train.json \
  --train-format jsonl \
  --dev-file Lattice_LSTM/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir Lattice_LSTM/demo_data/model \
  --epochs 8
python -m Lattice_LSTM.evaluate \
  --vocab-file Lattice_LSTM/demo_data/vocab.json \
  --gazetteer-file Lattice_LSTM/demo_data/gazetteer.txt \
  --model-dir Lattice_LSTM/demo_data/model \
  --eval-file Lattice_LSTM/demo_data/dev.json \
  --eval-format jsonl
python -m Lattice_LSTM.predict \
  --vocab-file Lattice_LSTM/demo_data/vocab.json \
  --gazetteer-file Lattice_LSTM/demo_data/gazetteer.txt \
  --model-path Lattice_LSTM/demo_data/model/ner.pt \
  --text "请问肯德基优惠券在哪里"
```

## Current synthetic validation

After the instance-builder refactor, the synthetic e2e check still passes:

- eval: `acc=1.0000`
- `prec=1.0000`
- `recall=1.0000`
- `f1=1.0000`
- predict example: `请问肯德基优惠券在哪里 -> 肯德基 , 优惠券`
- final result: `ALL_CHECKS_PASSED`

## Notes

- `TorchCRF` is used, not `torchcrf`.
- This version now uses a lightweight trie/gazetteer matching stage.
- `train / evaluate / predict` now share a centralized `LatticeInstanceBuilder` instead of each script constructing tensors independently.
- Full original `alphabet/data/gazetteer/trie + custom lattice recurrent cell` behavior is still not fully reproduced.
- Fake data is only for pipeline validation, not for real business quality claims.
