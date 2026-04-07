# -*- encoding: utf-8 -*-
from __future__ import annotations

from BILSTM_CRF.generate_fake_data import *  # reuse same synthetic generator


DEFAULT_GAZETTEER = [
    '肯德基', '麦当劳', '星巴克', '小米', '苹果', '华为', '特斯拉',
    '优惠券', '新品', '手机', '电脑', '总部', '地址', '客服', '门店',
    '肯德基优惠券', '麦当劳优惠券', '星巴克新品', '小米手机', '苹果手机', '华为手机',
]


def dump_gazetteer(terms, file_path):
    with open(file_path, 'w', encoding='utf-8') as writer:
        for term in terms:
            writer.write(term + '\n')


def main(output_dir: str = 'Lattice_LSTM/demo_data', train_size: int = 200, dev_size: int = 40):
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_records, dev_records = generate_dataset(train_size=train_size, dev_size=dev_size)
    vocab = build_vocab(train_records + dev_records)
    dump_jsonl(train_records, output_path / 'train.json')
    dump_jsonl(dev_records, output_path / 'dev.json')
    dump_vocab(vocab, output_path / 'vocab.json')
    dump_gazetteer(DEFAULT_GAZETTEER, output_path / 'gazetteer.txt')
    print(f'generated train={len(train_records)}, dev={len(dev_records)} -> {output_path}')


if __name__ == '__main__':
    main()
