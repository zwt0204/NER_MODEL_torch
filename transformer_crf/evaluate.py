# -*- encoding: utf-8 -*-
from __future__ import annotations

import argparse
import os

from .dataset_adapters import load_dataset
from .train import NerTrainner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-file', required=True)
    parser.add_argument('--model-dir', default='model/ner')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--eval-file', required=True)
    parser.add_argument('--eval-format', default='auto', choices=['auto', 'jsonl', 'conll'])
    args = parser.parse_args()

    trainer = NerTrainner(vocab_file=args.vocab_file, model_dir=args.model_dir)
    model_path = args.model_path or os.path.join(args.model_dir, 'ner.pt')
    ok = trainer.load(model_path)
    if not ok:
        raise FileNotFoundError(f'model checkpoint not found: {model_path}')
    records = load_dataset(args.eval_file, fmt=args.eval_format)
    metrics = trainer.evaluate(records)
    print(
        f"Eval metrics: acc={metrics.accuracy:.4f}, prec={metrics.precision:.4f}, "
        f"recall={metrics.recall:.4f}, f1={metrics.f1:.4f}, support={metrics.support}"
    )


if __name__ == '__main__':
    main()
