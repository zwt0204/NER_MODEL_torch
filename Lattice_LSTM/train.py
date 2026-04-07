# -*- encoding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import List

import torch
from torch.optim import Adam

from .data_utils import dump_jsonl
from .dataset_adapters import load_dataset
from .eval_utils import compute_token_metrics
from .gazetteer_utils import Gazetteer, load_gazetteer
from .instance_builder import LatticeInstance, LatticeInstanceBuilder
from .model import NerCore


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor
    lexicon_ids: torch.Tensor
    lexicon_lengths: torch.Tensor
    instances: List[LatticeInstance]


class NerTrainner:
    def __init__(self, vocab_file: str = 'vocab.json', model_dir: str = 'models/LatticeLSTM', gazetteer_file: str | None = None):
        self.model_dir = model_dir
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 50
        self.max_lexicon_words_num = 4
        self.gazetteer = load_gazetteer(gazetteer_file) if gazetteer_file else Gazetteer()
        vocab_size = len(self.char_index) + 1
        self.classnames = {'O': 0, 'B-BRD': 1, 'I-BRD': 2, 'B-KWD': 3, 'I-KWD': 4}
        class_size = len(self.classnames)
        keep_prob = 0.95
        learning_rate = 0.001
        trainable = True
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.builder = LatticeInstanceBuilder(
            char_index=self.char_index,
            label_index=self.classnames,
            gazetteer=self.gazetteer,
            io_sequence_size=self.io_sequence_size,
            max_lexicon_words_num=self.max_lexicon_words_num,
            unk_id=self.unknow_char_id,
        )

        self.model = NerCore(
            self.io_sequence_size,
            vocab_size,
            class_size,
            keep_prob,
            learning_rate,
            trainable,
            lexicon_vocab_size=max(vocab_size, self.gazetteer.size() + 1),
            max_lexicon_words_num=self.max_lexicon_words_num,
        )
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        os.makedirs(self.model_dir, exist_ok=True)

    def load_dict(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as reader:
            items = json.load(reader)
        for i, charvalue in enumerate(items, start=1):
            self.char_index[charvalue.strip()] = i

    def train(self, epochs: int, dstfile: str = 'train.json', devfile: str | None = None):
        records = self.load_samples(dstfile)
        if not records:
            raise ValueError(f'no training records found in {dstfile}')
        dev_records = self.load_samples(devfile) if devfile and os.path.exists(devfile) else None
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            batch_count = 0
            for batch in self.iter_batches(records):
                self.optimizer.zero_grad()
                loss = self.model.loss(
                    batch.inputs,
                    batch.targets,
                    batch.lengths,
                    batch.lexicon_ids,
                    batch.lexicon_lengths,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                batch_count += 1
                if batch_count % 20 == 0:
                    print(f'Progress {batch_count}, loss={loss.item():.6f}')
            avg_loss = total_loss / max(batch_count, 1)
            print(f'Epoch: {epoch + 1}/{epochs}, train loss={avg_loss:.6f}')
            if dev_records:
                metrics = self.evaluate(dev_records)
                print(
                    'Dev metrics: '
                    f'acc={metrics.accuracy:.4f}, prec={metrics.precision:.4f}, '
                    f'recall={metrics.recall:.4f}, f1={metrics.f1:.4f}, support={metrics.support}'
                )
            self.save()

    def iter_batches(self, records: List[dict]):
        for i in range(0, len(records), self.batch_size):
            instances = self.builder.build_instances(records[i:i + self.batch_size])
            yield self.convert_instances_to_batch(instances)

    def convert_instances_to_batch(self, instances: List[LatticeInstance]) -> Batch:
        batch = self.builder.tensorize_instances(instances, self.device)
        return Batch(
            inputs=batch['inputs'],
            targets=batch['targets'],
            lengths=batch['lengths'],
            lexicon_ids=batch['lexicon_ids'],
            lexicon_lengths=batch['lexicon_lengths'],
            instances=batch['instances'],
        )

    def load_samples(self, dstfile: str = 'train.json') -> List[dict]:
        return load_dataset(dstfile, fmt='auto')

    def save(self, path: str | None = None):
        model_path = path or os.path.join(self.model_dir, 'ner.pt')
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'char_index': self.char_index,
                'classnames': self.classnames,
                'io_sequence_size': self.io_sequence_size,
                'max_lexicon_words_num': self.max_lexicon_words_num,
                'gazetteer_size': self.gazetteer.size(),
            },
            model_path,
        )
        return model_path

    def load(self, path: str | None = None) -> bool:
        model_path = path or os.path.join(self.model_dir, 'ner.pt')
        if not os.path.exists(model_path):
            return False
        payload = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device)
        return True

    def evaluate(self, records: List[dict]):
        self.model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for batch in self.iter_batches(records):
                pred_batch = self.model.predict(
                    batch.inputs,
                    batch.lengths,
                    batch.lexicon_ids,
                    batch.lexicon_lengths,
                )
                true_batch = batch.targets.detach().cpu().tolist()
                lengths = batch.lengths.detach().cpu().tolist()
                for seq_true, seq_pred, seq_len in zip(true_batch, pred_batch, lengths):
                    true_labels.append(seq_true[:seq_len])
                    pred_labels.append(seq_pred[:seq_len])
        return compute_token_metrics(true_labels, pred_labels, o_id=self.classnames['O'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-file', default='vocab.json')
    parser.add_argument('--train-file', default='train.json')
    parser.add_argument('--train-format', default='auto', choices=['auto', 'jsonl', 'conll'])
    parser.add_argument('--dev-file', default=None)
    parser.add_argument('--dev-format', default='auto', choices=['auto', 'jsonl', 'conll'])
    parser.add_argument('--model-dir', default='models/LatticeLSTM')
    parser.add_argument('--gazetteer-file', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    trainner = NerTrainner(vocab_file=args.vocab_file, model_dir=args.model_dir, gazetteer_file=args.gazetteer_file)
    train_records = load_dataset(args.train_file, fmt=args.train_format)
    tmp_dir = tempfile.mkdtemp(prefix='ner_lattice_lstm_train_')
    train_file = os.path.join(tmp_dir, 'train.json')
    dump_jsonl(train_records, train_file)
    if args.dev_file:
        dev_records = load_dataset(args.dev_file, fmt=args.dev_format)
        dev_file = os.path.join(tmp_dir, 'dev.json')
        dump_jsonl(dev_records, dev_file)
        trainner.train(args.epochs, dstfile=train_file, devfile=dev_file)
    else:
        trainner.train(args.epochs, dstfile=train_file)


if __name__ == '__main__':
    main()
