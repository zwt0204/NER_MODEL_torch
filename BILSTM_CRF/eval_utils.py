# -*- encoding: utf-8 -*-
"""
Evaluation helpers for token-level NER metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TokenMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


def flatten_valid_labels(true_labels: Iterable[List[int]], pred_labels: Iterable[List[int]]):
    y_true = []
    y_pred = []
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        size = min(len(true_seq), len(pred_seq))
        y_true.extend(true_seq[:size])
        y_pred.extend(pred_seq[:size])
    return y_true, y_pred


def compute_token_metrics(true_labels: Iterable[List[int]], pred_labels: Iterable[List[int]], o_id: int = 0) -> TokenMetrics:
    y_true, y_pred = flatten_valid_labels(true_labels, pred_labels)
    if not y_true:
        return TokenMetrics(0.0, 0.0, 0.0, 0.0, 0)

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    tp = sum(int(t == p and t != o_id) for t, p in zip(y_true, y_pred))
    fp = sum(int(t != p and p != o_id) for t, p in zip(y_true, y_pred))
    fn = sum(int(t != p and t != o_id) for t, p in zip(y_true, y_pred))

    accuracy = correct / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return TokenMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=len(y_true),
    )
