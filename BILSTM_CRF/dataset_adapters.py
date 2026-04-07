# -*- encoding: utf-8 -*-
"""
Dataset adapters for common NER formats.

Supported input:
1. jsonl: {"text": "肯德基在哪里", "label": "B-BRD I-BRD I-BRD O O O"}
2. conll/bio txt: one token+tag per line, blank line separated sequences
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .data_utils import build_vocab, dump_jsonl, dump_vocab, load_jsonl


def load_conll(path: str | Path) -> List[dict]:
    records = []
    chars = []
    tags = []
    with open(path, "r", encoding="utf-8") as reader:
        for raw in reader:
            line = raw.strip()
            if not line:
                if chars:
                    records.append({"text": "".join(chars), "label": " ".join(tags)})
                    chars, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid conll line: {raw!r}")
            chars.append(parts[0])
            tags.append(parts[-1])
    if chars:
        records.append({"text": "".join(chars), "label": " ".join(tags)})
    return records


def load_dataset(path: str | Path, fmt: str = "auto") -> List[dict]:
    path = Path(path)
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() in {".jsonl", ".json"} else "conll"
    if fmt == "jsonl":
        return load_jsonl(path)
    if fmt == "conll":
        return load_conll(path)
    raise ValueError(f"unsupported dataset format: {fmt}")


def convert_dataset(input_path: str | Path, output_dir: str | Path, fmt: str = "auto"):
    records = load_dataset(input_path, fmt=fmt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(records, output_dir / "train.json")
    dump_vocab(build_vocab(records), output_dir / "vocab.json")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", default="auto", choices=["auto", "jsonl", "conll"])
    args = parser.parse_args()

    out = convert_dataset(args.input, args.output_dir, fmt=args.format)
    print(f"converted dataset -> {out}")
