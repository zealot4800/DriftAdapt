from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def macro_f1(expected: torch.Tensor, predicted: torch.Tensor) -> float:
    if expected.numel() == 0:
        return float("nan")
    return float(f1_score(
        expected.detach().cpu().numpy(), predicted.detach().cpu().numpy(),
        average="macro", zero_division=0,
    ))


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=float)
    return float("nan") if np.isnan(array).all() else float(np.nanmean(array))
