# -*- encoding: utf-8 -*-
"""
PyTorch Albert-CRF-style NER using a real HuggingFace `AlbertModel` encoder.
By default this module builds a small random-initialized ALBERT config so the
existing training / evaluation / prediction pipeline can run without requiring
external model downloads.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from TorchCRF import CRF
from transformers import AlbertConfig, AlbertModel


class NerCore(nn.Module):
    def __init__(
        self,
        io_sequence_size: int,
        vocab_size: int,
        class_size: int = 6,
        keep_prob: float = 0.9,
        learning_rate: float = 1e-4,
        trainable: bool = False,
        pretrained_model_name_or_path: str | None = None,
    ):
        super().__init__()
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.io_sequence_size = io_sequence_size
        self.learning_rate = learning_rate
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.hidden_size = 128
        self.embedding_size = self.hidden_size
        self.num_heads = 4
        self.ffn_dim = 256
        self.num_layers = 2
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        config = AlbertConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.ffn_dim,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            max_position_embeddings=max(512, io_sequence_size + 8),
            hidden_dropout_prob=1 - self.keep_prob,
            attention_probs_dropout_prob=1 - self.keep_prob,
            type_vocab_size=1,
            pad_token_id=0,
        )
        self.encoder = self.build_encoder(config, pretrained_model_name_or_path)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(self.hidden_size, self.output_class_size)
        self.crf = CRF(self.output_class_size, pad_idx=None, use_gpu=torch.cuda.is_available())

    @staticmethod
    def build_encoder(config: AlbertConfig, pretrained_model_name_or_path: str | None) -> AlbertModel:
        if pretrained_model_name_or_path:
            try:
                print(f'Loading pretrained AlbertModel from: {pretrained_model_name_or_path}')
                return AlbertModel.from_pretrained(pretrained_model_name_or_path)
            except Exception as exc:
                print(
                    'Falling back to random-initialized AlbertConfig because pretrained load failed: '
                    f'{exc}'
                )
        return AlbertModel(config)

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)

    def forward(self, inputs: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        mask = self.sequence_mask(sequence_lengths, inputs.size(1))
        outputs = self.encoder(input_ids=inputs.long(), attention_mask=mask.long())
        x = self.dropout(outputs.last_hidden_state)
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
