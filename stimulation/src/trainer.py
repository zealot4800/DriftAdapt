from __future__ import annotations

import time
from collections.abc import Iterable, Mapping

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


def capacity_selective_retrain_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    active_neuron_masks: Mapping[str, torch.Tensor],
    affected_neuron_indices: Mapping[str, Iterable[int] | torch.Tensor],
    newly_active_neuron_indices: Mapping[str, Iterable[int] | torch.Tensor],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str = "Adam",
    seed: int = 0,
) -> tuple[float, int]:
    """Retrain only capacity-associated parameter elements.

    A selected hidden neuron owns its incoming weights and bias as well as its
    outgoing weights in the following layer.  The classifier bias and the
    classifier weights connected to active final-layer neurons are trainable.
    Connections to dormant neurons and all unrelated parameter elements remain
    exactly equal to their pre-training values.

    Returns ``(elapsed_seconds, trainable_element_count)``.  A fresh Adam
    optimizer and a locally seeded data-loader generator make separate capacity
    trials independent of optimizer history and global sample-order RNG state.
    """
    if optimizer_name.lower() != "adam":
        raise ValueError("The DRIFTADAPT artifact uses Adam; no other optimizer is supported")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Capacity training features and labels must have the same length")
    if labels.numel() == 0:
        raise ValueError("Capacity training requires at least one labeled sample")
    if epochs < 1:
        raise ValueError("Capacity training epochs must be positive")
    if batch_size < 1:
        raise ValueError("Capacity training batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("Capacity training learning_rate must be positive")

    hidden_modules = model.named_hidden_modules()
    hidden_names = list(hidden_modules)
    if not hidden_names:
        raise ValueError("Capacity-selective training requires at least one hidden layer")

    expected_names = set(hidden_names)
    mask_names = set(active_neuron_masks)
    if mask_names != expected_names:
        missing = sorted(expected_names - mask_names)
        unknown = sorted(mask_names - expected_names)
        raise ValueError(
            "Active-neuron masks must cover the hidden modules exactly "
            f"(missing={missing}, unknown={unknown})"
        )
    unknown_selected = (
        set(affected_neuron_indices) | set(newly_active_neuron_indices)
    ) - expected_names
    if unknown_selected:
        raise ValueError(
            "Unknown hidden modules selected for capacity training: "
            f"{sorted(unknown_selected)}"
        )

    active_masks: dict[str, torch.Tensor] = {}
    selected_masks: dict[str, torch.Tensor] = {}
    previous_width: int | None = None
    for name in hidden_names:
        module = hidden_modules[name]
        if previous_width is not None and module.in_features != previous_width:
            raise ValueError(
                "Capacity-selective training supports feed-forward hidden layers "
                "whose adjacent dimensions match"
            )
        previous_width = module.out_features

        active = torch.as_tensor(
            active_neuron_masks[name], device=module.weight.device, dtype=torch.bool
        ).flatten()
        if active.numel() != module.out_features:
            raise ValueError(
                f"Active-neuron mask for {name!r} has {active.numel()} entries; "
                f"expected {module.out_features}"
            )
        active_masks[name] = active

        selected = torch.zeros(module.out_features, dtype=torch.bool, device=module.weight.device)
        for source in (affected_neuron_indices, newly_active_neuron_indices):
            raw_indices = source.get(name, ())
            if isinstance(raw_indices, torch.Tensor):
                indices = raw_indices.detach().flatten().to(device=module.weight.device)
            else:
                indices = torch.as_tensor(
                    list(raw_indices), device=module.weight.device, dtype=torch.long
                )
            if indices.numel() == 0:
                continue
            if indices.dtype == torch.bool or indices.is_floating_point():
                raise ValueError(f"Capacity neuron indices for {name!r} must be integers")
            indices = indices.to(dtype=torch.long)
            if bool(torch.any(indices < 0)) or bool(torch.any(indices >= module.out_features)):
                raise ValueError(f"Capacity neuron index out of range for {name!r}")
            selected[indices] = True
        if bool(torch.any(selected & ~active)):
            raise ValueError(f"Dormant neurons cannot be selected for capacity training in {name!r}")
        selected_masks[name] = selected

    classifier_name, classifier = model.named_classifier_module()
    final_name = hidden_names[-1]
    if classifier.in_features != hidden_modules[final_name].out_features:
        raise ValueError(
            f"Classifier {classifier_name!r} input width does not match the final hidden layer"
        )

    parameter_masks: dict[int, tuple[nn.Parameter, torch.Tensor]] = {}

    def add_parameter_mask(parameter: nn.Parameter | None, mask: torch.Tensor) -> None:
        if parameter is None:
            return
        if mask.shape != parameter.shape:
            raise ValueError("Internal capacity gradient-mask shape mismatch")
        key = id(parameter)
        if key in parameter_masks:
            old_parameter, old_mask = parameter_masks[key]
            parameter_masks[key] = (old_parameter, old_mask | mask)
        else:
            parameter_masks[key] = (parameter, mask.clone())

    previous_name: str | None = None
    for name in hidden_names:
        module = hidden_modules[name]
        active_output = active_masks[name]
        selected_output = selected_masks[name]
        weight_mask = selected_output[:, None].expand_as(module.weight)
        if previous_name is not None:
            active_input = active_masks[previous_name]
            selected_input = selected_masks[previous_name]
            weight_mask = (
                weight_mask | selected_input[None, :].expand_as(module.weight)
            ) & active_output[:, None] & active_input[None, :]
        else:
            weight_mask = weight_mask & active_output[:, None]
        add_parameter_mask(module.weight, weight_mask)
        if module.bias is not None:
            add_parameter_mask(module.bias, selected_output)
        previous_name = name

    classifier_weight_mask = active_masks[final_name][None, :].expand_as(classifier.weight)
    add_parameter_mask(classifier.weight, classifier_weight_mask)
    if classifier.bias is not None:
        add_parameter_mask(
            classifier.bias,
            torch.ones_like(classifier.bias, dtype=torch.bool),
        )

    for parameter in model.parameters():
        if id(parameter) not in parameter_masks:
            parameter_masks[id(parameter)] = (
                parameter,
                torch.zeros_like(parameter, dtype=torch.bool),
            )

    original_requires_grad = {
        id(parameter): parameter.requires_grad for parameter in model.parameters()
    }
    snapshots = {
        key: parameter.detach().clone()
        for key, (parameter, _) in parameter_masks.items()
    }
    hook_handles = []
    trainable_parameters: list[nn.Parameter] = []
    trainable_element_count = 0
    for parameter, mask in parameter_masks.values():
        is_trainable = bool(torch.any(mask))
        parameter.requires_grad_(is_trainable)
        if not is_trainable:
            continue
        trainable_parameters.append(parameter)
        trainable_element_count += int(torch.count_nonzero(mask).item())
        hook_handles.append(parameter.register_hook(lambda gradient, mask=mask: gradient * mask))

    if not trainable_parameters:
        raise ValueError("Capacity-selective training produced no trainable parameter elements")

    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    started = time.perf_counter()
    try:
        model.train()
        for _ in range(epochs):
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                scores = model(batch_features)
                loss = loss_fn(scores, batch_labels)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for key, (parameter, mask) in parameter_masks.items():
                        parameter[~mask] = snapshots[key][~mask]
        model.eval()

        for key, (parameter, mask) in parameter_masks.items():
            if not torch.equal(parameter.detach()[~mask], snapshots[key][~mask]):
                raise RuntimeError("Capacity training changed a frozen parameter element")
    finally:
        for handle in hook_handles:
            handle.remove()
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])

    if hasattr(model, "accumulate_capacity_training_delta"):
        model.accumulate_capacity_training_delta({
            name: parameter.detach() - snapshots[id(parameter)]
            for name, parameter in model.named_parameters()
        })

    return time.perf_counter() - started, trainable_element_count
