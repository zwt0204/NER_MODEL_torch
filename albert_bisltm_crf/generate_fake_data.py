# -*- encoding: utf-8 -*-
from __future__ import annotations

from BILSTM_CRF.generate_fake_data import *  # reuse same synthetic generator


def main(output_dir: str = 'albert_bisltm_crf/demo_data', train_size: int = 200, dev_size: int = 40):
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_records, dev_records = generate_dataset(train_size=train_size, dev_size=dev_size)
    vocab = build_vocab(train_records + dev_records)
    dump_jsonl(train_records, output_path / 'train.json')
    dump_jsonl(dev_records, output_path / 'dev.json')
    dump_vocab(vocab, output_path / 'vocab.json')
    print(f'generated train={len(train_records)}, dev={len(dev_records)} -> {output_path}')


if __name__ == '__main__':
    main()
