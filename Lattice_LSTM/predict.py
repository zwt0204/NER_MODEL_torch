# -*- encoding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from .gazetteer_utils import Gazetteer, load_gazetteer
from .instance_builder import LatticeInstanceBuilder
from .model import NerCore


class NerPredicter:
    def __init__(
        self,
        vocab_file: str = 'vocab.json',
        model_path: str = 'models/LatticeLSTM/ner.pt',
        gazetteer_file: str | None = None,
    ):
        self.model_path = model_path
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 50
        self.max_lexicon_words_num = 4
        self.gazetteer = load_gazetteer(gazetteer_file) if gazetteer_file else Gazetteer()
        vocab_size = len(self.char_index) + 1
        keep_prob = 1.0
        learning_rate = 0.0001
        trainable = False
        self.classnames = {'O': 0, 'B-BRD': 1, 'I-BRD': 2, 'B-KWD': 3, 'I-KWD': 4}
        class_size = len(self.classnames)
        self.classids = {value: key for key, value in self.classnames.items()}
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
        self.load()

    def load_dict(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as reader:
            items = json.load(reader)
        for i, charvalue in enumerate(items, start=1):
            self.char_index[charvalue.strip()] = i

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'model checkpoint not found: {self.model_path}')
        payload = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_text: str) -> str:
        input_text = input_text.strip().lower()
        record = {'text': input_text, 'label': ' '.join(['O'] * min(len(input_text), self.io_sequence_size))}
        instances = self.builder.build_instances([record])
        batch = self.builder.tensorize_instances(instances, self.device)
        seq_len = int(batch['lengths'][0].item())
        with torch.no_grad():
            label_list = self.model.predict(
                batch['inputs'],
                batch['lengths'],
                batch['lexicon_ids'],
                batch['lexicon_lengths'],
            )
        taggs: List[str] = [self.classids[idx] for idx in label_list[0][:seq_len]]
        output_labels = self.model.decode(instances[0].chars[:seq_len], taggs)
        data_items = []
        if output_labels:
            for key in output_labels.keys():
                terms = [record[0] for record in output_labels[key]]
                data_items.append(' '.join(terms))
        return ' , '.join(data_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-file', default='vocab.json')
    parser.add_argument('--model-path', default='models/LatticeLSTM/ner.pt')
    parser.add_argument('--gazetteer-file', default=None)
    parser.add_argument('--text', default='请问肯德基优惠券在哪里')
    args = parser.parse_args()

    predicter = NerPredicter(vocab_file=args.vocab_file, model_path=args.model_path, gazetteer_file=args.gazetteer_file)
    line = predicter.predict(args.text)
    print('-> ' + args.text)
    print('--> ' + line)


if __name__ == '__main__':
    main()
