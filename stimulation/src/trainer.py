from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def form_training_set(features: torch.Tensor, labels: torch.Tensor, *, balance_binary: bool = True):
    if labels.numel() == 0:
        return None, None
    if not balance_binary:
        return features, labels
    classes = torch.unique(labels)
    if classes.numel() != 2 or not all(int(x) in (0, 1) for x in classes):
        return None, None
    positive = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
    count = min(positive.numel(), negative.numel())
    if count == 0:
        return None, None
    positive = positive[torch.randperm(positive.numel(), device=positive.device)[:count]]
    negative = negative[torch.randperm(negative.numel(), device=negative.device)[:count]]
    indices = torch.cat((negative, positive))
    return features[indices], labels[indices]


def retrain_model(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                  *, epochs: int, batch_size: int, learning_rate: float,
                  optimizer_name: str = "Adam", optimizer=None) -> tuple[float, torch.optim.Optimizer]:
    if optimizer_name.lower() != "adam":
        raise ValueError("The DRIFTADAPT artifact uses Adam; no other optimizer is supported")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            scores = model(batch_features)
            loss = loss_fn(scores, batch_labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return time.perf_counter() - started, optimizer
