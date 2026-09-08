from __future__ import annotations

import torch
from torch import nn


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
