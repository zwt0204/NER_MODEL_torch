# -*- encoding: utf-8 -*-
"""
PyTorch implementation of Transformer-CRF for NER.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from TorchCRF import CRF


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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
        self.num_heads = 4
        self.ffn_dim = 256
        self.num_layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.position = PositionalEncoding(self.embedding_size, max_len=io_sequence_size + 5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=1 - self.keep_prob,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        if hasattr(self.encoder, 'enable_nested_tensor'):
            self.encoder.enable_nested_tensor = False
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(self.embedding_size, self.output_class_size)
        self.crf = CRF(self.output_class_size, pad_idx=None, use_gpu=torch.cuda.is_available())

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)

    def forward(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        x = self.embedding(inputs.long())
        x = self.position(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
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
