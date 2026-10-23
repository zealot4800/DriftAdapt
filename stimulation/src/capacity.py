from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


CAPACITY_EVENT_TYPES = frozenset({"expand", "contract", "rejected_expand"})


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"capacity.{name} must be boolean")
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"capacity.{name} must be an integer")
    if value < minimum:
        raise ValueError(f"capacity.{name} must be at least {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"capacity.{name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"capacity.{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"capacity.{name} must be finite")
    return converted


def _unit_interval(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"capacity.{name} must be between zero and one")
    return converted


def _strictly_greater(value: float, threshold: float) -> bool:
    """Implement mathematical ``>`` without binary round-off equality leaks."""
    return value > threshold and not math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-12,
    )


def resolve_capacity_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the validated, explicit software-capacity configuration.

    A missing block disables capacity changes while retaining deterministic
    defaults. Model-specific compatibility is intentionally checked by the
    simulation pipeline, not by this configuration-only helper.
    """
    source = dict(config or {})
    enabled = _boolean(source.get("enabled", False), "enabled")
    initial_layer1 = _integer(source.get("initial_layer1", 8), "initial_layer1")
    initial_layer2 = _integer(source.get("initial_layer2", 4), "initial_layer2")
    max_layer1 = _integer(source.get("max_layer1", 16), "max_layer1")
    max_layer2 = _integer(source.get("max_layer2", 8), "max_layer2")
    if initial_layer1 > max_layer1:
        raise ValueError("capacity.initial_layer1 must not exceed capacity.max_layer1")
    if initial_layer2 > max_layer2:
        raise ValueError("capacity.initial_layer2 must not exceed capacity.max_layer2")

    minimum_improvement = _unit_interval(
        source.get("min_improvement", 0.0), "min_improvement",
    )
    utility_threshold = _unit_interval(
        source.get("contraction_utility_threshold", 0.01),
        "contraction_utility_threshold",
    )
    initialization_noise = _finite_float(
        source.get("initialization_noise", 0.001), "initialization_noise",
    )
    if initialization_noise < 0.0:
        raise ValueError("capacity.initialization_noise must be nonnegative")

    return {
        "enabled": enabled,
        "initial_layer1": initial_layer1,
        "initial_layer2": initial_layer2,
        "max_layer1": max_layer1,
        "max_layer2": max_layer2,
        "residual_f1_threshold": _unit_interval(
            source.get("residual_f1_threshold", 0.05), "residual_f1_threshold",
        ),
        "saturation_threshold": _unit_interval(
            source.get("saturation_threshold", 0.75), "saturation_threshold",
        ),
        "layer1_expand_step": _integer(
            source.get("layer1_expand_step", 2), "layer1_expand_step",
        ),
        "layer2_expand_step": _integer(
            source.get("layer2_expand_step", 1), "layer2_expand_step",
        ),
        "min_improvement": minimum_improvement,
        "contraction_enabled": _boolean(
            source.get("contraction_enabled", True), "contraction_enabled",
        ),
        "contraction_utility_threshold": utility_threshold,
        "contraction_patience": _integer(
            source.get("contraction_patience", 10), "contraction_patience",
        ),
        "initialization_noise": initialization_noise,
    }


def neuron_saturation(
    impact_scores: Sequence[float],
    active_mask: Sequence[int | bool],
    *,
    impact_threshold: float,
) -> dict[str, float | int]:
    """Calculate high-impact saturation using active neurons only."""
    if len(impact_scores) != len(active_mask):
        raise ValueError("Neuron impact scores and active mask must have equal length")
    threshold = _unit_interval(impact_threshold, "impact_threshold")
    if any(value != 0 and value != 1 for value in active_mask):
        raise ValueError("Active neuron mask must be binary")
    mask = [int(value) for value in active_mask]
    scores = [_finite_float(value, "impact_score") for value in impact_scores]
    active_count = sum(mask)
    if active_count < 1:
        raise ValueError("Neuron saturation needs at least one active neuron")
    high_impact_count = sum(
        active and score >= threshold for score, active in zip(scores, mask)
    )
    return {
        "active_neurons": active_count,
        "high_impact_neurons": int(high_impact_count),
        "saturation": high_impact_count / active_count,
    }


def capacity_ratios(
    active_layer1: int,
    active_layer2: int,
    max_layer1: int,
    max_layer2: int,
) -> dict[str, float | int]:
    """Return the required per-layer and overall active-capacity metrics."""
    active1 = _integer(active_layer1, "active_layer1")
    active2 = _integer(active_layer2, "active_layer2")
    maximum1 = _integer(max_layer1, "max_layer1")
    maximum2 = _integer(max_layer2, "max_layer2")
    if active1 > maximum1 or active2 > maximum2:
        raise ValueError("Active capacity must not exceed maximum capacity")
    return {
        "active_layer1": active1,
        "active_layer2": active2,
        "layer1_capacity_ratio": active1 / maximum1,
        "layer2_capacity_ratio": active2 / maximum2,
        "overall_capacity_ratio": (active1 + active2) / (maximum1 + maximum2),
    }


def residual_recovery_status(
    *,
    reference_f1: float | None,
    repaired_f1: float,
    threshold: float,
) -> dict[str, float | bool]:
    """Measure repair recovery before any feature or capacity escalation."""
    repaired = _unit_interval(repaired_f1, "repaired_f1")
    reference = (
        None if reference_f1 is None
        else _unit_interval(reference_f1, "reference_f1")
    )
    limit = _unit_interval(threshold, "residual_f1_threshold")
    residual = max(0.0, reference - repaired) if reference is not None else 0.0
    insufficient = bool(
        reference is not None and _strictly_greater(residual, limit)
    )
    return {
        "residual_f1_loss": residual,
        "recovery_reference_available": reference is not None,
        "recovery_insufficient": insufficient,
    }


def capacity_pressure_decision(
    *,
    reference_f1: float | None,
    repaired_f1: float,
    layer1_saturation: float,
    layer2_saturation: float,
    active_layer1: int,
    active_layer2: int,
    config: Mapping[str, Any],
) -> dict[str, float | int | bool]:
    """Apply the deterministic residual-loss AND saturation policy."""
    resolved = resolve_capacity_config(config)
    saturation1 = _unit_interval(layer1_saturation, "layer1_saturation")
    saturation2 = _unit_interval(layer2_saturation, "layer2_saturation")
    active1 = _integer(active_layer1, "active_layer1")
    active2 = _integer(active_layer2, "active_layer2")
    if active1 < resolved["initial_layer1"] or active1 > resolved["max_layer1"]:
        raise ValueError("Layer-1 active capacity is outside configured bounds")
    if active2 < resolved["initial_layer2"] or active2 > resolved["max_layer2"]:
        raise ValueError("Layer-2 active capacity is outside configured bounds")

    recovery = residual_recovery_status(
        reference_f1=reference_f1,
        repaired_f1=repaired_f1,
        threshold=resolved["residual_f1_threshold"],
    )
    residual = float(recovery["residual_f1_loss"])
    residual_pressure = bool(
        resolved["enabled"]
        and recovery["recovery_insufficient"]
    )
    layer1_pressure = bool(
        residual_pressure
        and _strictly_greater(saturation1, resolved["saturation_threshold"])
    )
    layer2_pressure = bool(
        residual_pressure
        and _strictly_greater(saturation2, resolved["saturation_threshold"])
    )
    expand_layer1 = layer1_pressure and active1 < resolved["max_layer1"]
    expand_layer2 = layer2_pressure and active2 < resolved["max_layer2"]
    proposed_layer1 = min(
        resolved["max_layer1"],
        active1 + resolved["layer1_expand_step"] if expand_layer1 else active1,
    )
    proposed_layer2 = min(
        resolved["max_layer2"],
        active2 + resolved["layer2_expand_step"] if expand_layer2 else active2,
    )
    return {
        "residual_f1_loss": residual,
        "residual_pressure": residual_pressure,
        "layer1_pressure": layer1_pressure,
        "layer2_pressure": layer2_pressure,
        "capacity_pressure": layer1_pressure or layer2_pressure,
        "expand_layer1": expand_layer1,
        "expand_layer2": expand_layer2,
        "proposed_layer1": proposed_layer1,
        "proposed_layer2": proposed_layer2,
    }


def make_capacity_event(
    *,
    window_id: int,
    event_type: str,
    old_capacity: tuple[int, int],
    new_capacity: tuple[int, int],
    reference_f1: float | None,
    pre_repair_f1: float | None,
    post_repair_f1: float | None,
    post_capacity_f1: float | None,
    residual_f1_loss: float,
    layer1_saturation: float,
    layer2_saturation: float,
    activated_layer1_neurons: Sequence[int] = (),
    activated_layer2_neurons: Sequence[int] = (),
    deactivated_layer1_neurons: Sequence[int] = (),
    deactivated_layer2_neurons: Sequence[int] = (),
    training_samples: int = 0,
    trainable_parameters: int = 0,
    accepted: bool,
    model_version: int,
) -> dict[str, Any]:
    """Build the fixed-schema JSON-serializable capacity event record."""
    kind = str(event_type)
    if kind not in CAPACITY_EVENT_TYPES:
        raise ValueError(f"Unsupported capacity event type: {kind!r}")
    if not isinstance(accepted, bool):
        raise ValueError("Capacity event accepted must be boolean")
    if (kind == "rejected_expand") == accepted:
        raise ValueError("Capacity event type and accepted status are inconsistent")

    def capacity_pair(value: tuple[int, int], label: str) -> dict[str, int]:
        if len(value) != 2:
            raise ValueError(f"{label} must contain two layer capacities")
        return {
            "layer1": _integer(value[0], f"{label}.layer1"),
            "layer2": _integer(value[1], f"{label}.layer2"),
        }

    def optional_f1(value: float | None, label: str) -> float | None:
        return None if value is None else _unit_interval(value, label)

    def neuron_indices(values: Sequence[int], label: str) -> list[int]:
        converted = [_integer(value, label, minimum=0) for value in values]
        if len(set(converted)) != len(converted):
            raise ValueError(f"{label} must not contain duplicate neuron indices")
        return sorted(converted)

    residual = _unit_interval(residual_f1_loss, "residual_f1_loss")
    return {
        "window_id": _integer(window_id, "window_id"),
        "event_type": kind,
        "old_capacity": capacity_pair(old_capacity, "old_capacity"),
        "new_capacity": capacity_pair(new_capacity, "new_capacity"),
        "reference_f1": optional_f1(reference_f1, "reference_f1"),
        "pre_repair_f1": optional_f1(pre_repair_f1, "pre_repair_f1"),
        "post_repair_f1": optional_f1(post_repair_f1, "post_repair_f1"),
        "post_capacity_f1": optional_f1(post_capacity_f1, "post_capacity_f1"),
        "residual_f1_loss": residual,
        "layer1_saturation": _unit_interval(
            layer1_saturation, "layer1_saturation",
        ),
        "layer2_saturation": _unit_interval(
            layer2_saturation, "layer2_saturation",
        ),
        "activated_layer1_neurons": neuron_indices(
            activated_layer1_neurons, "activated_layer1_neurons",
        ),
        "activated_layer2_neurons": neuron_indices(
            activated_layer2_neurons, "activated_layer2_neurons",
        ),
        "deactivated_layer1_neurons": neuron_indices(
            deactivated_layer1_neurons, "deactivated_layer1_neurons",
        ),
        "deactivated_layer2_neurons": neuron_indices(
            deactivated_layer2_neurons, "deactivated_layer2_neurons",
        ),
        "training_samples": _integer(
            training_samples, "training_samples", minimum=0,
        ),
        "trainable_parameters": _integer(
            trainable_parameters, "trainable_parameters", minimum=0,
        ),
        "accepted": accepted,
        "model_version": _integer(model_version, "model_version", minimum=0),
    }


def select_impacted_modules(scores: dict[str, float], threshold: float) -> list[str]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("adaptation.impact_threshold must be between 0 and 1")
    return [name for name, score in scores.items() if score >= threshold]


def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
    maximum = values.max() if values.numel() else values.new_tensor(0.0)
    return values / maximum if maximum > 0 else torch.zeros_like(values)


@torch.inference_mode()
def map_drift_to_neural_impact(
    model: nn.Module,
    feature_drift_scores: dict[str, float | None],
    threshold: float,
    activation_magnitudes: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Propagate a fixed JSD vector through hidden-layer absolute weights."""
    hidden_modules = model.named_hidden_modules()
    if not hidden_modules:
        return {
            "impacted_neuron_scores": {},
            "module_impact_scores": {},
            "impact_selected_modules": [],
            "affected_parameter_count": 0,
            "affected_parameter_percentage": 0.0,
            "impact_state_elements": len(feature_drift_scores),
            "impact_multiply_accumulate_count": 0,
        }

    first_module = next(iter(hidden_modules.values()))
    drift_vector = torch.tensor(
        [0.0 if score is None else score for score in feature_drift_scores.values()],
        dtype=first_module.weight.dtype,
        device=first_module.weight.device,
    )
    drift_vector = torch.nan_to_num(drift_vector, nan=0.0, posinf=1.0, neginf=0.0)
    if drift_vector.numel() != first_module.in_features:
        raise ValueError("Feature drift vector does not match the model input size")

    neuron_masks = (
        model.hidden_neuron_masks()
        if hasattr(model, "hidden_neuron_masks") else {}
    )
    neuron_scores: dict[str, list[float]] = {}
    raw_module_scores: dict[str, float] = {}
    propagated = drift_vector
    multiply_accumulates = 0
    neuron_state_elements = 0
    for name, module in hidden_modules.items():
        if propagated.numel() != module.in_features:
            raise ValueError(f"Hidden module {name} does not match the preceding impact vector")
        impact = torch.mv(module.weight.detach().abs(), propagated)
        if activation_magnitudes and name in activation_magnitudes:
            activation = activation_magnitudes[name].to(impact).abs().flatten()
            if activation.numel() != impact.numel():
                raise ValueError(f"Activation magnitude for {name} has the wrong size")
            impact = impact * activation
        active_mask = None
        if name in neuron_masks:
            mask = neuron_masks[name].to(device=impact.device, dtype=impact.dtype).flatten()
            if mask.numel() != impact.numel():
                raise ValueError(f"Neuron mask for {name} has the wrong size")
            impact = impact * mask
            active_mask = mask.to(torch.bool)
        normalized = _normalize_vector(impact)
        neuron_scores[name] = normalized.detach().cpu().tolist()
        active_scores = normalized if active_mask is None else normalized[active_mask]
        raw_module_scores[name] = float(active_scores.mean().item())
        propagated = normalized
        multiply_accumulates += module.weight.numel()
        neuron_state_elements += module.out_features

    maximum_module_score = max(raw_module_scores.values(), default=0.0)
    module_scores = {
        name: score / maximum_module_score if maximum_module_score > 0 else 0.0
        for name, score in raw_module_scores.items()
    }
    selected = select_impacted_modules(module_scores, threshold)
    affected_parameters = sum(
        parameter.numel()
        for name in selected
        for parameter in hidden_modules[name].parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "impacted_neuron_scores": neuron_scores,
        "module_impact_scores": module_scores,
        "impact_selected_modules": selected,
        "affected_parameter_count": affected_parameters,
        "affected_parameter_percentage": 100.0 * affected_parameters / total_parameters,
        "impact_state_elements": (
            drift_vector.numel() + neuron_state_elements + len(hidden_modules)
        ),
        "impact_multiply_accumulate_count": multiply_accumulates,
    }


@torch.inference_mode()
def select_drift_associated_samples(
    features: torch.Tensor,
    drift_scores: dict[str, float | None],
    drifted_feature_indices: list[int],
    range_min: torch.Tensor,
    range_max: torch.Tensor,
    *,
    enabled: bool,
    max_samples: int,
) -> torch.Tensor:
    """Select a deterministic, bounded, stream-ordered drift-associated subset."""
    if not drifted_feature_indices:
        return torch.empty(0, dtype=torch.long, device=features.device)
    if not enabled:
        return torch.arange(features.shape[0], device=features.device)
    if max_samples < 1:
        raise ValueError("sample_selection.max_samples must be positive")

    indices = torch.tensor(drifted_feature_indices, dtype=torch.long, device=features.device)
    span = (range_max[indices] - range_min[indices]).clamp_min(1e-6)
    center = 0.5 * (range_max[indices] + range_min[indices])
    drift_values = list(drift_scores.values())
    jsd = torch.tensor(
        [drift_values[index] or 0.0 for index in drifted_feature_indices],
        dtype=features.dtype,
        device=features.device,
    )
    scores = (((features[:, indices] - center).abs() / span) * jsd).sum(dim=1)
    count = min(max_samples, max(features.shape[0] - 1, 0))
    selected = torch.argsort(scores, descending=True, stable=True)[:count]
    return torch.sort(selected).values
