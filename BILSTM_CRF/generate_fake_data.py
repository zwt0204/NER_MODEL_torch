# -*- encoding: utf-8 -*-
"""
Synthetic data generator for quick local testing.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from .data_utils import build_vocab, dump_jsonl, dump_vocab


BRANDS = ["肯德基", "麦当劳", "星巴克", "小米", "苹果", "华为", "特斯拉"]
KEYWORDS = ["优惠券", "新品", "手机", "电脑", "总部", "地址", "客服", "门店"]
SUFFIXES = ["在哪里", "怎么样", "好不好", "电话多少", "值得买吗", "地址在哪"]
FILLERS = ["请问", "想知道", "帮我查下", "麻烦问下", "我想了解"]


def label_text(text: str, brand: str, keyword: str) -> str:
    tags = ["O"] * len(text)
    for term, prefix in ((brand, "BRD"), (keyword, "KWD")):
        start = text.find(term)
        if start >= 0:
            tags[start] = f"B-{prefix}"
            for i in range(start + 1, start + len(term)):
                tags[i] = f"I-{prefix}"
    return " ".join(tags)


def build_sample() -> dict:
    brand = random.choice(BRANDS)
    keyword = random.choice(KEYWORDS)
    suffix = random.choice(SUFFIXES)
    filler = random.choice(FILLERS)
    patterns = [
        f"{brand}{keyword}{suffix}",
        f"{filler}{brand}{keyword}{suffix}",
        f"{brand}的{keyword}{suffix}",
        f"{keyword}{brand}{suffix}",
    ]
    text = random.choice(patterns)
    return {"text": text, "label": label_text(text, brand, keyword)}


def generate_dataset(train_size: int = 200, dev_size: int = 40) -> Tuple[List[dict], List[dict]]:
    train_records = [build_sample() for _ in range(train_size)]
    dev_records = [build_sample() for _ in range(dev_size)]
    return train_records, dev_records


def main(output_dir: str = "BILSTM_CRF/demo_data", train_size: int = 200, dev_size: int = 40):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_records, dev_records = generate_dataset(train_size=train_size, dev_size=dev_size)
    vocab = build_vocab(train_records + dev_records)
    dump_jsonl(train_records, output_path / "train.json")
    dump_jsonl(dev_records, output_path / "dev.json")
    dump_vocab(vocab, output_path / "vocab.json")
    print(f"generated train={len(train_records)}, dev={len(dev_records)} -> {output_path}")


if __name__ == "__main__":
    main()
