from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class InferenceModel(nn.Module):
    def run_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.forward(x)
        return scores, torch.argmax(scores, dim=1)

    def named_hidden_modules(self) -> dict[str, nn.Linear]:
        """Return hidden layers by their checkpoint-compatible module names."""
        linear_modules = [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        ]
        return dict(linear_modules[:-1])

    def named_classifier_module(self) -> tuple[str, nn.Linear]:
        """Return the final classifier and its checkpoint-compatible name."""
        linear_modules = [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_modules:
            raise ValueError("An adaptation model must contain at least one linear layer")
        return linear_modules[-1]


class ID_CIC_IDS2017_small_pforest(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2), nn.Softmax(dim=1),
        )
        self.linear_relu_stack.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


class ID_CIC_IDS2017_large_pforest(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [16, 128, 64, 32, 16, 8, 4, 2]
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            layers.append(nn.Softmax(dim=1) if index == len(sizes) - 2 else nn.ReLU())
        self.linear_relu_stack = nn.Sequential(*layers)
        self.linear_relu_stack.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


class ID_UNSW_NB15_N3IC(InferenceModel):
    def __init__(self, input_shape: int = 20, neurons: list[int] | None = None) -> None:
        super().__init__()
        neurons = neurons or [32, 16, 2]
        self.model = nn.Sequential(
            nn.Linear(input_shape, neurons[0]), nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]), nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]), nn.Softmax(dim=1),
        )
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNSW_IoT_N3IC(InferenceModel):
    def __init__(self, input_shape: int = 16, neurons: list[int] | None = None) -> None:
        super().__init__()
        neurons = neurons or [64, 32, 10]
        self.model = nn.Sequential(
            nn.Linear(input_shape, neurons[0]), nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]), nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]), nn.Softmax(dim=1),
        )
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


MODEL_REGISTRY = {
    cls.__name__: cls for cls in (
        ID_CIC_IDS2017_small_pforest,
        ID_CIC_IDS2017_large_pforest,
        ID_UNSW_NB15_N3IC,
        UNSW_IoT_N3IC,
    )
}


def build_model(name: str, kwargs: dict[str, Any] | None = None) -> InferenceModel:
    try:
        return MODEL_REGISTRY[name](**(kwargs or {}))
    except KeyError as exc:
        raise ValueError(f"Unsupported DRIFTADAPT model: {name}") from exc


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
