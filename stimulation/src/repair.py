from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RepairPlan:
    affected_modules: list[str]
    module_magnitudes: dict[str, float]
    confidence: float
    parameter_deltas: dict[str, torch.Tensor]
    update_parameter_count: int = 0
    update_bytes: int = 0
    classifier_name: str | None = None
    classifier_delta: torch.Tensor | None = None


def _module_vector(module: nn.Module) -> torch.Tensor:
    parameters = [parameter.detach().reshape(-1) for parameter in module.parameters()]
    return torch.cat(parameters) if parameters else torch.empty(0)


@torch.inference_mode()
def model_fingerprint(model: nn.Module) -> str:
    """Return a deterministic fingerprint for the exact repair anchor weights."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def extract_oracle_repair(
    original: nn.Module,
    oracle: nn.Module,
    *,
    module_threshold: float = 0.30,
) -> dict:
    """Extract hidden-module deltas and normalized change magnitudes."""
    if not 0.0 <= module_threshold <= 1.0:
        raise ValueError("module_threshold must be between zero and one")
    original_modules = original.named_hidden_modules()
    oracle_modules = oracle.named_hidden_modules()
    if list(original_modules) != list(oracle_modules):
        raise ValueError("Original and oracle models have different hidden modules")

    deltas: dict[str, torch.Tensor] = {}
    relative: dict[str, float] = {}
    delta_norms: dict[str, float] = {}
    for name, original_module in original_modules.items():
        before = _module_vector(original_module)
        after = _module_vector(oracle_modules[name]).to(before)
        delta = after - before
        deltas[name] = delta.detach().cpu()
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        base_norm = float(torch.linalg.vector_norm(before).item())
        delta_norms[name] = delta_norm
        relative[name] = delta_norm / max(base_norm, torch.finfo(torch.float32).eps)

    maximum = max(relative.values(), default=0.0)
    normalized = {
        name: value / maximum if maximum > 0 else 0.0
        for name, value in relative.items()
    }
    selected = [
        name for name, value in normalized.items()
        if value >= module_threshold and delta_norms[name] > 0
    ]
    original_classifier_name, original_classifier = original.named_classifier_module()
    oracle_classifier_name, oracle_classifier = oracle.named_classifier_module()
    if original_classifier_name != oracle_classifier_name:
        raise ValueError("Original and oracle models have different classifiers")
    classifier_before = _module_vector(original_classifier)
    classifier_after = _module_vector(oracle_classifier).to(classifier_before)
    classifier_delta = (classifier_after - classifier_before).detach().cpu()
    return {
        "oracle_modules": selected,
        "oracle_module_change_magnitudes": normalized,
        "oracle_module_relative_changes": relative,
        "oracle_module_delta_norms": delta_norms,
        "oracle_parameter_deltas": deltas,
        "oracle_classifier_name": original_classifier_name,
        "oracle_classifier_parameter_delta": classifier_delta,
        "oracle_classifier_delta_norm": float(
            torch.linalg.vector_norm(classifier_delta).item()
        ),
    }


def apply_repair_plan(
    model: nn.Module,
    plan: RepairPlan,
    *,
    parameter_bits: int = 32,
    anchor_model: nn.Module | None = None,
) -> tuple[nn.Module, dict]:
    """Apply base-relative module deltas to a copied candidate model.

    When an anchor is supplied, the entire candidate starts from that anchor
    before selected parameters are assigned ``anchor + delta``.  This prevents
    both repeated and disjoint drift episodes from retaining stale repairs.
    """
    if parameter_bits < 1:
        raise ValueError("parameter_bits must be positive")
    if anchor_model is not None:
        model_state = model.state_dict()
        anchor_state = anchor_model.state_dict()
        if (
            list(model_state) != list(anchor_state)
            or any(
                model_state[name].shape != anchor_state[name].shape
                for name in model_state
            )
        ):
            raise ValueError("Repair anchor model layout does not match the active model")
    candidate = copy.deepcopy(anchor_model if anchor_model is not None else model)
    modules = candidate.named_hidden_modules()
    anchor_modules = anchor_model.named_hidden_modules() if anchor_model is not None else None
    unknown = set(plan.affected_modules) - set(modules)
    if unknown:
        raise ValueError(f"Repair plan contains unknown modules: {sorted(unknown)}")
    if anchor_modules is not None and list(anchor_modules) != list(modules):
        raise ValueError("Repair anchor hidden-module layout does not match the model")

    for name in plan.affected_modules:
        delta = plan.parameter_deltas[name].to(
            device=next(modules[name].parameters()).device,
            dtype=next(modules[name].parameters()).dtype,
        ).flatten()
        expected = sum(parameter.numel() for parameter in modules[name].parameters())
        if delta.numel() != expected:
            raise ValueError(f"Repair delta for {name} has {delta.numel()} values; expected {expected}")
        offset = 0
        anchor_parameters = (
            list(anchor_modules[name].parameters()) if anchor_modules is not None else []
        )
        for parameter_index, parameter in enumerate(modules[name].parameters()):
            count = parameter.numel()
            update = delta[offset:offset + count].view_as(parameter)
            with torch.no_grad():
                if anchor_modules is None:
                    parameter.add_(update)
                else:
                    parameter.copy_(
                        anchor_parameters[parameter_index].to(parameter) + update
                    )
            offset += count

    if plan.classifier_delta is not None:
        classifier_name, classifier = candidate.named_classifier_module()
        if plan.classifier_name != classifier_name:
            raise ValueError(
                f"Repair classifier {plan.classifier_name!r} does not match {classifier_name!r}"
            )
        delta = plan.classifier_delta.to(
            device=next(classifier.parameters()).device,
            dtype=next(classifier.parameters()).dtype,
        ).flatten()
        expected = sum(parameter.numel() for parameter in classifier.parameters())
        if delta.numel() != expected:
            raise ValueError(
                f"Repair delta for {classifier_name} has {delta.numel()} values; "
                f"expected {expected}"
            )
        anchor_classifier = None
        if anchor_model is not None:
            anchor_classifier_name, anchor_classifier = anchor_model.named_classifier_module()
            if anchor_classifier_name != classifier_name:
                raise ValueError("Repair anchor classifier layout does not match the model")
        anchor_parameters = (
            list(anchor_classifier.parameters()) if anchor_classifier is not None else []
        )
        offset = 0
        for parameter_index, parameter in enumerate(classifier.parameters()):
            count = parameter.numel()
            update = delta[offset:offset + count].view_as(parameter)
            with torch.no_grad():
                if anchor_classifier is None:
                    parameter.add_(update)
                else:
                    parameter.copy_(
                        anchor_parameters[parameter_index].to(parameter) + update
                    )
            offset += count

    active_parameters = dict(model.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if list(active_parameters) != list(candidate_parameters):
        raise ValueError("Repair candidate layout does not match the active model")
    changed = sum(
        int(torch.count_nonzero(
            parameter.detach()
            - active_parameters[name].detach().to(parameter)
        ).item())
        for name, parameter in candidate_parameters.items()
    )
    total = sum(parameter.numel() for parameter in candidate.parameters())
    candidate.eval()
    return candidate, {
        "repair_affected_modules": list(plan.affected_modules),
        "repair_module_magnitudes": dict(plan.module_magnitudes),
        "repair_confidence": float(plan.confidence),
        "repair_changed_parameters": changed,
        "repair_changed_parameter_percentage": 100.0 * changed / total if total else 0.0,
        "repair_update_bytes": (changed * parameter_bits + 7) // 8,
    }


def recovery_ratio(f1_repair: float, f1_drifted: float, f1_oracle: float) -> float | None:
    denominator = f1_oracle - f1_drifted
    if denominator <= 1e-12:
        return None
    return (f1_repair - f1_drifted) / denominator
