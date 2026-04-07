# -*- encoding: utf-8 -*-
"""
Utilities for loading/saving NER samples and vocab.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def load_jsonl(path: str | Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_jsonl(records: Iterable[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_vocab(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def dump_vocab(vocab: Iterable[str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(list(vocab), writer, ensure_ascii=False, indent=2)


def build_vocab(records: Iterable[dict]) -> List[str]:
    vocab = []
    seen = set()
    for record in records:
        for ch in record["text"]:
            if ch not in seen:
                seen.add(ch)
                vocab.append(ch)
    return vocab
