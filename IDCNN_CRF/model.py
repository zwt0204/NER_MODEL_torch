# -*- encoding: utf-8 -*-
"""
PyTorch implementation of IDCNN-CRF for NER.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from TorchCRF import CRF


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


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
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.char_dim = 128
        self.seg_dim = 16
        self.num_segs = 4
        self.num_filters = 100
        self.repeat_times = 4
        self.dilations = [1, 1, 2]

        self.char_embedding = nn.Embedding(self.vocab_size, self.char_dim, padding_idx=0)
        self.seg_embedding = nn.Embedding(self.num_segs, self.seg_dim, padding_idx=0)
        self.input_proj = nn.Conv1d(self.char_dim + self.seg_dim, self.num_filters, kernel_size=1)
        self.idcnn_blocks = nn.ModuleList([
            DilatedConvBlock(self.num_filters, kernel_size=3, dilation=d) for _ in range(self.repeat_times) for d in self.dilations
        ])
        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.classifier = nn.Linear(self.num_filters * self.repeat_times, self.output_class_size)
        self.crf = CRF(self.output_class_size, pad_idx=None, use_gpu=torch.cuda.is_available())

    def forward(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor, seg_inputs: torch.Tensor | None = None) -> torch.Tensor:
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        if seg_inputs is None:
            seg_inputs = self.build_seg_features(inputs, sequence_lengths)
        char_embed = self.char_embedding(inputs.long())
        seg_embed = self.seg_embedding(seg_inputs.long())
        embedded = torch.cat([char_embed, seg_embed], dim=-1).transpose(1, 2)
        conv_input = self.input_proj(embedded)

        outputs = []
        x = conv_input
        block_size = len(self.dilations)
        for i, block in enumerate(self.idcnn_blocks, start=1):
            x = block(x)
            if i % block_size == 0:
                outputs.append(x.transpose(1, 2))
        features = torch.cat(outputs, dim=-1)
        features = self.dropout(features)
        emissions = self.classifier(features)
        emissions = emissions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return emissions

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)

    @staticmethod
    def build_seg_features(inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, seq_len = inputs.size()
        segs = torch.zeros((batch, seq_len), dtype=torch.long, device=inputs.device)
        for i, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            if length == 1:
                segs[i, 0] = 0
            else:
                segs[i, 0] = 1
                if length > 2:
                    segs[i, 1:length - 1] = 2
                segs[i, length - 1] = 3
        return segs

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sequence_lengths: torch.Tensor,
        seg_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emissions = self.forward(inputs, sequence_lengths, seg_inputs=seg_inputs)
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        log_likelihood = self.crf(emissions, targets.long(), mask=mask)
        return -log_likelihood.mean()

    def predict(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor,
        seg_inputs: torch.Tensor | None = None,
    ) -> List[List[int]]:
        emissions = self.forward(inputs, sequence_lengths, seg_inputs=seg_inputs)
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
