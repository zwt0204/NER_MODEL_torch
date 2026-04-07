# -*- encoding: utf-8 -*-
"""
PyTorch implementation of BiLSTM-CRF for NER.
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
        keep_prob: float = 0.5,
        learning_rate: float = 0.001,
        trainable: bool = False,
    ):
        super().__init__()
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.io_sequence_size = io_sequence_size
        self.learning_rate = learning_rate
        self.embedding_size = 256
        self.hidden_size = 128
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.num_layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.classifier = nn.Linear(self.hidden_size * 2, self.output_class_size)
        self.crf = CRF(self.output_class_size, pad_idx=None, use_gpu=torch.cuda.is_available())

    def forward(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        embedded = self.embedding(inputs.long())
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=inputs.size(1),
        )
        output = self.dropout(output)
        emissions = self.classifier(output)
        emissions = emissions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return emissions

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)

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
                    if key in raw_content.keys():
                        raw_content[key].append((content, position))
                    else:
                        raw_content[key] = [(content, position)]
                    char_item = []
                    tag_item = []
                char_item.append(terms[i])
                tag_item.append(taggs[i])
            elif taggs[i][0] == 'O':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    position = (i - len(content), len(content))
                    key = tag_item[0][2:]
                    if key in raw_content.keys():
                        raw_content[key].append((content, position))
                    else:
                        raw_content[key] = [(content, position)]
                    char_item = []
                    tag_item = []
            else:
                char_item.append(terms[i])
                tag_item.append(taggs[i])
        if len(char_item) > 0 and len(tag_item) > 0:
            content = ''.join(char_item)
            key = tag_item[0][2:]
            position = (len(terms) - len(content), len(content))
            if key in raw_content.keys():
                raw_content[key].append((content, position))
            else:
                raw_content[key] = [(content, position)]
        return raw_content
