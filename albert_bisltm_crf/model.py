# -*- encoding: utf-8 -*-
"""
PyTorch baseline implementation of Albert-BiLSTM-CRF-style NER.
This runtime currently uses a lightweight encoder stub instead of a real pretrained AlbertModel.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from TorchCRF import CRF


class NerCore(nn.Module):
    def __init__(
        self,
        io_sequence_size: int,
        vocab_size: int,
        class_size: int = 6,
        keep_prob: float = 0.9,
        learning_rate: float = 1e-4,
        trainable: bool = False,
    ):
        super().__init__()
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.io_sequence_size = io_sequence_size
        self.learning_rate = learning_rate
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.embedding_size = 128
        self.hidden_size = 128
        self.num_layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(self.hidden_size * 2, self.output_class_size)
        self.crf = CRF(self.output_class_size, pad_idx=None, use_gpu=torch.cuda.is_available())

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)

    def forward(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        x = self.embedding(inputs.long())
        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=sequence_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.encoder(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=inputs.size(1),
        )
        x = self.dropout(x)
        emissions = self.classifier(x)
        emissions = emissions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return emissions

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        emissions = self.forward(inputs, sequence_lengths)
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        log_likelihood = self.crf(emissions, targets.long(), mask=mask)
        return -log_likelihood.mean()

    def predict(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor) -> List[List[int]]:
        emissions = self.forward(inputs, sequence_lengths)
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        return self.crf.viterbi_decode(emissions, mask=mask)

    def decode(self, terms: Sequence[str], taggs: Sequence[str]) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        char_item = []
        tag_item = []
        raw_content = {}
        for i in range(len(terms)):
            if taggs[i][0] == 'B':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    key = tag_item[0][2:]
                    position = (i - len(content), len(content))
                    raw_content.setdefault(key, []).append((content, position))
                    char_item = []
                    tag_item = []
                char_item.append(terms[i])
                tag_item.append(taggs[i])
            elif taggs[i][0] == 'O':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    key = tag_item[0][2:]
                    position = (i - len(content), len(content))
                    raw_content.setdefault(key, []).append((content, position))
                    char_item = []
                    tag_item = []
            else:
                char_item.append(terms[i])
                tag_item.append(taggs[i])
        if len(char_item) > 0 and len(tag_item) > 0:
            content = ''.join(char_item)
            key = tag_item[0][2:]
            position = (len(terms) - len(content), len(content))
            raw_content.setdefault(key, []).append((content, position))
        return raw_content
