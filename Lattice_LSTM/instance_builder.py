# -*- encoding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .gazetteer_utils import Gazetteer


@dataclass
class LatticeInstance:
    text: str
    chars: List[str]
    char_ids: List[int]
    labels: List[str]
    label_ids: List[int]
    gaz_matches: List[List[str]]
    gaz_ids: List[List[int]]
    gaz_lengths: List[List[float]]
    seq_len: int


class LatticeInstanceBuilder:
    def __init__(
        self,
        char_index: dict[str, int],
        label_index: dict[str, int],
        gazetteer: Gazetteer,
        io_sequence_size: int = 50,
        max_lexicon_words_num: int = 4,
        unk_id: int | None = None,
    ):
        self.char_index = char_index
        self.label_index = label_index
        self.gazetteer = gazetteer
        self.io_sequence_size = io_sequence_size
        self.max_lexicon_words_num = max_lexicon_words_num
        self.unk_id = unk_id if unk_id is not None else len(char_index)

    def build(self, record: dict) -> LatticeInstance:
        text = record['text']
        chars = list(text[:self.io_sequence_size])
        seq_len = len(chars)
        labels = record.get('label', '').split()[:self.io_sequence_size]
        if len(labels) < seq_len:
            labels = labels + ['O'] * (seq_len - len(labels))
        char_ids = [self.char_index.get(ch, self.unk_id) for ch in chars]
        label_ids = [self.label_index.get(tag, self.label_index['O']) for tag in labels]

        gaz_matches: List[List[str]] = []
        gaz_ids: List[List[int]] = []
        gaz_lengths: List[List[float]] = []
        for start in range(seq_len):
            matched_list = self.gazetteer.enumerate_match_list(chars[start:])[:self.max_lexicon_words_num]
            gaz_matches.append(matched_list)
            gaz_ids.append([self.gazetteer.search_id(list(term)) for term in matched_list])
            gaz_lengths.append([float(len(term)) for term in matched_list])

        while len(gaz_matches) < self.io_sequence_size:
            gaz_matches.append([])
            gaz_ids.append([])
            gaz_lengths.append([])

        return LatticeInstance(
            text=text,
            chars=chars,
            char_ids=char_ids,
            labels=labels,
            label_ids=label_ids,
            gaz_matches=gaz_matches,
            gaz_ids=gaz_ids,
            gaz_lengths=gaz_lengths,
            seq_len=seq_len,
        )

    def build_instances(self, records: Sequence[dict]) -> List[LatticeInstance]:
        return [self.build(record) for record in records]

    def tensorize_instances(self, instances: Sequence[LatticeInstance], device: torch.device) -> dict[str, torch.Tensor]:
        count = len(instances)
        xrows = torch.zeros((count, self.io_sequence_size), dtype=torch.long)
        yrows = torch.zeros((count, self.io_sequence_size), dtype=torch.long)
        xlens = torch.zeros((count,), dtype=torch.long)
        lexicon_rows = torch.zeros((count, self.io_sequence_size, self.max_lexicon_words_num), dtype=torch.long)
        lexicon_length_rows = torch.zeros((count, self.io_sequence_size, self.max_lexicon_words_num), dtype=torch.float32)

        for i, instance in enumerate(instances):
            xlens[i] = instance.seq_len
            if instance.char_ids:
                xrows[i, :instance.seq_len] = torch.tensor(instance.char_ids, dtype=torch.long)
                yrows[i, :instance.seq_len] = torch.tensor(instance.label_ids, dtype=torch.long)
            for pos in range(min(instance.seq_len, self.io_sequence_size)):
                ids = instance.gaz_ids[pos][:self.max_lexicon_words_num]
                lens = instance.gaz_lengths[pos][:self.max_lexicon_words_num]
                if ids:
                    lexicon_rows[i, pos, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    lexicon_length_rows[i, pos, :len(lens)] = torch.tensor(lens, dtype=torch.float32)

        return {
            'inputs': xrows.to(device),
            'targets': yrows.to(device),
            'lengths': xlens.to(device),
            'lexicon_ids': lexicon_rows.to(device),
            'lexicon_lengths': lexicon_length_rows.to(device),
            'instances': list(instances),
        }

    def batchify(self, records: Sequence[dict], device: torch.device) -> dict[str, torch.Tensor]:
        instances = self.build_instances(records)
        return self.tensorize_instances(instances, device)
