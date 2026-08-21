from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """Deterministic FIFO replay storage with a fixed sample capacity."""

    def __init__(self, capacity: int, feature_count: int, device: torch.device) -> None:
        if capacity < 1:
            raise ValueError("replay.capacity must be positive")
        self.capacity = capacity
        self.features = torch.empty((0, feature_count), dtype=torch.float32, device=device)
        self.labels = torch.empty(0, dtype=torch.long, device=device)

    def add(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.numel() == 0:
            return
        self.features = torch.cat((self.features, features.detach()))[-self.capacity:]
        self.labels = torch.cat((self.labels, labels.detach()))[-self.capacity:]

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = min(max(count, 0), self.labels.numel())
        if count == 0:
            return self.features[:0], self.labels[:0]
        return self.features[-count:], self.labels[-count:]

    def __len__(self) -> int:
        return self.labels.numel()


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


def selective_retrain_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    selected_modules: list[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str = "Adam",
) -> tuple[float, torch.optim.Optimizer]:
    """Retrain selected hidden modules and the final classifier only."""
    if optimizer_name.lower() != "adam":
        raise ValueError("The DRIFTADAPT artifact uses Adam; no other optimizer is supported")

    hidden_modules = model.named_hidden_modules()
    unknown = set(selected_modules) - set(hidden_modules)
    if unknown:
        raise ValueError(f"Unknown hidden modules selected for adaptation: {sorted(unknown)}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in selected_modules:
        for parameter in hidden_modules[name].parameters():
            parameter.requires_grad_(True)
    _, classifier = model.named_classifier_module()
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
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
