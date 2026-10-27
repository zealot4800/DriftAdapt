from __future__ import annotations

import time
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM

def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _two_hidden_stack(input_size: int, first: int, second: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, first), nn.ReLU(),
        nn.Linear(first, second), nn.ReLU(),
        nn.Linear(second, output_size), nn.Softmax(dim=1),
    )


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
        self.linear_relu_stack = _two_hidden_stack(16, 8, 4, 2)
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
        self.model = _two_hidden_stack(input_shape, neurons[0], neurons[1], neurons[2])
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNSW_IoT_N3IC(InferenceModel):
    def __init__(self, input_shape: int = 16, neurons: list[int] | None = None) -> None:
        super().__init__()
        neurons = neurons or [64, 32, 10]
        self.model = _two_hidden_stack(input_shape, neurons[0], neurons[1], neurons[2])
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


MODEL_REGISTRY = {cls.__name__: cls for cls in (ID_CIC_IDS2017_small_pforest, ID_CIC_IDS2017_large_pforest, ID_UNSW_NB15_N3IC, UNSW_IoT_N3IC)}


def build_model(name: str, kwargs: dict[str, Any] | None = None) -> InferenceModel:
    try:
        return MODEL_REGISTRY[name](**(kwargs or {}))
    except KeyError as exc:
        raise ValueError(f"Unsupported classifier model: {name}") from exc


class OneClassDetector:
    """Slack-corrected linear nu-one-class objective, with anomaly-positive sign.

    Use 0.5||u||² + sum(xi)/(nu*n) - r, xi>=0, u.x>=r-xi; w=-u, rho=-r.
    The linear detector uses the slack-corrected one-class objective.
    """
    def __init__(self, config):
        self.config = dict(config)
        self.weights = None
        self.rho = None

    def fit_benign(self, vectors):
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('at least two finite benign source windows are required')
        model = OneClassSVM(kernel='linear', nu=self.config['nu'],
            tol=self.config['tolerance'], max_iter=self.config['max_iterations']).fit(x)
        if model.fit_status_ != 0:
            raise RuntimeError('one-class optimization did not converge')
        # libsvm uses sum(alpha)=nu*n and alpha<=1. Normalize to the stated
        # primal's sum(alpha)=1, alpha<=1/(nu*n); signs alone would conceal
        # this scale difference until the online hinge-loss updates.
        scale = self.config['nu'] * len(x)
        self.weights = -model.coef_.reshape(-1).copy() / scale
        self.rho = float(model.intercept_[0]) / scale
        return self

    def score(self, vectors):
        if self.weights is None:
            raise RuntimeError('classifier has not been trained')
        x = np.asarray(vectors, dtype=np.float64)
        if x.shape[-1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError('invalid classifier vector')
        return x @ self.weights - self.rho

    def predict(self, vectors):
        return np.where(self.score(vectors) >= 0, 1, -1)


class SourceScaler:
    """Source-only min/max scaling; no mean centering of the linear OCSVM."""
    def fit(self, x):
        self.minimum = np.min(x, axis=0)
        self.span = np.ptp(x, axis=0)
        self.span = np.where(self.span > 0, self.span, 1.0)
        return self

    def transform(self, x):
        # Do not clip target values: preserve evidence of out-of-range traffic.
        return (np.asarray(x) - self.minimum) / self.span


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
                  optimizer_name: str = "Adam", optimizer=None,
                  class_weights: torch.Tensor | None = None,
                  criterion: nn.Module | None = None) -> tuple[float, torch.optim.Optimizer]:
    if optimizer_name.lower() != "adam":
        raise ValueError("The DRIFTADAPT artifact uses Adam; no other optimizer is supported")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss(weight=class_weights)
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


def load_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload.get('state_dict', payload))


def prepare_student(section, config, data, *, train=False):
    """Load a causal checkpoint, or explicitly train a new source-only checkpoint."""
    from .config import path_for
    path = path_for(config, section['checkpoint'])
    model = build_model(section['class'], section.get('kwargs')).to(data.device)
    first = next(iter(model.named_hidden_modules().values()))
    if first.in_features != len(data.feature_names) or model.named_classifier_module()[1].out_features != config['dataset']['num_classes']:
        raise ValueError('Configured model dimensions do not match the feature/class schema')
    if path.exists():
        saved = torch.load(path, map_location=data.device, weights_only=True)
        if saved.get('preprocessing_id') != data.preprocessing_id:
            raise ValueError(f'{path}: legacy/incompatible preprocessing; choose a new checkpoint path and run train to regenerate it')
        model.load_state_dict(saved['state_dict'])
    elif train:
        settings = config['training']
        if (data.labels[0] < 0).any():
            raise ValueError('Supervised student/teacher training requires an explicit nonnegative class_mapping for negative ground-truth labels')
        retrain_model(model, data.features[0], data.labels[0], epochs=settings['epochs'],
                      batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
                      optimizer_name=settings['optimizer'])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'preprocessing_id': data.preprocessing_id,
                    'preprocessing': data.preprocessing, 'class': section['class']}, path)
    else:
        raise FileNotFoundError(f'{path}: missing source-trained checkpoint; run train explicitly')
    return model.eval()
