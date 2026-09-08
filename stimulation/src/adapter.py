from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .drift_signature import DRIFT_SIGNATURE_FORMAT_VERSION, SIGNATURE_FIELDS
from .metrics import macro_f1
from .repair import (
    RepairPlan,
    apply_feature_mask,
    apply_repair_plan,
    model_fingerprint,
    recovery_ratio,
)


ADAPTER_FORMAT_VERSION = 6


class DriftConditionedAdapter(nn.Module):
    """Small multi-task MLP for repair and feature-configuration prediction."""

    def __init__(
        self,
        input_size: int,
        module_count: int,
        feature_count: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        if input_size < 1 or module_count < 1 or feature_count < 1 or hidden_size < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.input_size = input_size
        self.module_count = module_count
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.module_head = nn.Linear(hidden_size, module_count)
        self.magnitude_head = nn.Linear(hidden_size, module_count)
        self.feature_head = nn.Linear(hidden_size, feature_count)
        self.confidence_head = nn.Linear(hidden_size, 1)

    def forward(self, signature: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(signature)
        return {
            "module_logits": self.module_head(encoded),
            "module_magnitudes": torch.sigmoid(self.magnitude_head(encoded)),
            "feature_logits": self.feature_head(encoded),
            "confidence": torch.sigmoid(self.confidence_head(encoded)).squeeze(-1),
        }


@dataclass
class AdapterBundle:
    adapter: DriftConditionedAdapter
    module_names: list[str]
    repair_prototypes: dict[str, list[dict]]
    signature_size: int
    classifier_name: str | None = None
    classifier_prototypes: list[dict] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    signature_config: dict = field(default_factory=dict)
    base_model_fingerprint: str | None = None
    signature_fields: list[str] = field(default_factory=lambda: list(SIGNATURE_FIELDS))
    signature_format_version: int = DRIFT_SIGNATURE_FORMAT_VERSION
    oracle_module_threshold: float = 0.80
    default_feature_mask: list[int] = field(default_factory=list)
    feature_reference_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    feature_reconfiguration_config: dict = field(default_factory=dict)


def _confidence_target(record: dict) -> float:
    gain = record["oracle_retrained_f1"] - record["f1_after_drift"]
    available = max(1.0 - record["f1_after_drift"], 1e-8)
    return min(max(gain / available, 0.0), 1.0)


def _feature_target(record: dict) -> list[int]:
    return list(
        record["oracle_feature_mask"]
        if record["feature_reconfiguration_required"]
        else record["default_feature_mask"]
    )


def _training_tensors(records: list[dict], module_names: list[str]):
    signatures = torch.stack([record["drift_signature_vector"].float() for record in records])
    selected = torch.tensor([
        [float(name in record["oracle_modules"]) for name in module_names]
        for record in records
    ])
    magnitudes = torch.tensor([
        [record["oracle_module_change_magnitudes"][name] for name in module_names]
        for record in records
    ], dtype=torch.float32)
    feature_masks = torch.tensor([
        _feature_target(record) for record in records
    ], dtype=torch.float32)
    confidence = torch.tensor([_confidence_target(record) for record in records])
    return signatures, selected, magnitudes, feature_masks, confidence


def _repair_variants(record: dict) -> list[dict]:
    return list(record["oracle_repair_candidates"])


def _repair_prototypes(records: list[dict], module_names: list[str]) -> dict[str, list[dict]]:
    prototypes: dict[str, list[dict]] = {name: [] for name in module_names}
    for record in records:
        for variant in _repair_variants(record):
            for name in module_names:
                prototypes[name].append({
                    "signature": record["drift_signature_vector"].float().cpu(),
                    "parameter_delta": variant["oracle_parameter_deltas"][name].float().cpu(),
                    "oracle_magnitude": float(
                        variant["oracle_module_change_magnitudes"][name]
                    ),
                    "scenario_id": int(record["scenario_id"]),
                    "feature_mask": list(variant["feature_mask"]),
                })
    return prototypes


def _classifier_prototypes(records: list[dict]) -> list[dict]:
    return [{
        "signature": record["drift_signature_vector"].float().cpu(),
        "parameter_delta": variant["oracle_classifier_parameter_delta"].float().cpu(),
        "scenario_id": int(record["scenario_id"]),
        "feature_mask": list(variant["feature_mask"]),
    } for record in records for variant in _repair_variants(record)]


def _nearest_prototype(signature: torch.Tensor, candidates: list[dict]) -> dict:
    similarities = [
        float(torch.nn.functional.cosine_similarity(
            signature[None], candidate["signature"][None], dim=1, eps=1e-8,
        ).item())
        for candidate in candidates
    ]
    return candidates[max(range(len(candidates)), key=similarities.__getitem__)]


def _compatible_feature_masks(
    bundle: AdapterBundle,
    selected_modules: list[str],
) -> list[tuple[int, ...]]:
    masks = {
        tuple(candidate["feature_mask"])
        for candidate in bundle.classifier_prototypes
    }
    for name in selected_modules:
        masks &= {
            tuple(candidate["feature_mask"])
            for candidate in bundle.repair_prototypes[name]
        }
    budget = int(bundle.feature_reconfiguration_config["active_feature_budget"])
    masks = {mask for mask in masks if sum(mask) <= budget}
    default = tuple(bundle.default_feature_mask)
    if not bundle.feature_reconfiguration_config["enabled"]:
        return [default]
    if not masks:
        raise ValueError("Drift adapter has no feature-compatible repair prototypes")
    return ([default] if default in masks else []) + sorted(masks - {default})


def _decode_feature_mask(
    probabilities: torch.Tensor,
    candidates: list[tuple[int, ...]],
) -> list[int]:
    clipped = probabilities.clamp(1e-7, 1.0 - 1e-7)
    scores = []
    for candidate in candidates:
        mask = torch.tensor(candidate, dtype=clipped.dtype)
        scores.append(float((
            mask * torch.log(clipped) + (1.0 - mask) * torch.log1p(-clipped)
        ).sum().item()))
    return list(candidates[max(range(len(candidates)), key=scores.__getitem__)])


def train_drift_adapter(
    dataset: dict,
    *,
    hidden_size: int = 32,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> tuple[AdapterBundle, dict]:
    records = [record for record in dataset["records"] if record["split"] == "train"]
    if not records:
        raise ValueError("Drift adapter needs at least one training scenario")
    if hidden_size < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError("Adapter hidden size, epochs, and learning rate must be positive")
    torch.manual_seed(seed)
    module_names = list(dataset["module_names"])
    x, selected, magnitudes, feature_masks, confidence = _training_tensors(
        records, module_names,
    )
    feature_count = len(dataset["feature_names"])
    adapter = DriftConditionedAdapter(
        x.shape[1], len(module_names), feature_count, hidden_size,
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    classification_loss = nn.BCEWithLogitsLoss()
    regression_loss = nn.MSELoss()
    adapter.train()
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        output = adapter(x)
        module_loss = classification_loss(output["module_logits"], selected)
        magnitude_loss = regression_loss(output["module_magnitudes"], magnitudes)
        feature_loss = classification_loss(output["feature_logits"], feature_masks)
        confidence_loss = regression_loss(output["confidence"], confidence)
        loss = module_loss + magnitude_loss + feature_loss + confidence_loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    adapter.eval()
    bundle = AdapterBundle(
        adapter=adapter,
        module_names=module_names,
        repair_prototypes=_repair_prototypes(records, module_names),
        signature_size=x.shape[1],
        classifier_name=records[0]["oracle_classifier_name"],
        classifier_prototypes=_classifier_prototypes(records),
        feature_names=list(dataset["feature_names"]),
        signature_config=dict(dataset["signature_config"]),
        base_model_fingerprint=dataset["model_fingerprint"],
        signature_fields=list(dataset["signature_fields"]),
        signature_format_version=int(dataset["signature_format_version"]),
        oracle_module_threshold=float(dataset["oracle_module_threshold"]),
        default_feature_mask=list(dataset["default_feature_mask"]),
        feature_reference_values=dataset["feature_reference_values"].float().cpu(),
        feature_reconfiguration_config=dict(dataset["feature_reconfiguration_config"]),
    )
    return bundle, {
        "adapter_training_scenarios": len(records),
        "adapter_training_loss": final_loss,
        "adapter_hidden_size": hidden_size,
        "adapter_epochs": epochs,
        "feature_reconfiguration_training_scenarios": sum(
            record["feature_reconfiguration_required"] for record in records
        ),
        "default_feature_training_scenarios": sum(
            not record["feature_reconfiguration_required"] for record in records
        ),
    }


def _prediction_state(
    bundle: AdapterBundle,
    signature_vector: list[float] | torch.Tensor,
    *,
    module_threshold: float,
) -> dict:
    if not 0.0 <= module_threshold <= 1.0:
        raise ValueError("adapter.module_threshold must be between zero and one")
    signature = torch.as_tensor(signature_vector, dtype=torch.float32).flatten()
    if signature.numel() != bundle.signature_size:
        raise ValueError("Drift signature does not match the trained adapter input")
    device = next(bundle.adapter.parameters()).device
    output = bundle.adapter(signature.to(device)[None])
    probabilities = torch.sigmoid(output["module_logits"])[0].cpu()
    magnitudes = output["module_magnitudes"][0].cpu()
    feature_probabilities = torch.sigmoid(output["feature_logits"])[0].cpu()
    confidence = float(output["confidence"][0].cpu().item())
    selected = [
        name for index, name in enumerate(bundle.module_names)
        if probabilities[index] >= module_threshold and bundle.repair_prototypes[name]
    ]
    return {
        "signature": signature,
        "probabilities": probabilities,
        "magnitudes": magnitudes,
        "feature_probabilities": feature_probabilities,
        "confidence": confidence,
        "selected": selected,
    }


def _build_plan_for_mask(
    bundle: AdapterBundle,
    state: dict,
    feature_mask: Sequence[int],
    *,
    parameter_bits: int,
) -> tuple[RepairPlan, dict]:
    if parameter_bits < 1:
        raise ValueError("parameter_bits must be positive")
    normalized_mask = [int(value) for value in feature_mask]
    compatible = _compatible_feature_masks(bundle, state["selected"])
    if tuple(normalized_mask) not in compatible:
        raise ValueError("Requested feature mask has no compatible repair prototypes")

    signature = state["signature"]
    selected = state["selected"]
    magnitudes = state["magnitudes"]
    confidence = state["confidence"]
    feature_mask = normalized_mask
    needs_weight_repair = bool(selected) or feature_mask != bundle.default_feature_mask
    parameter_deltas: dict[str, torch.Tensor] = {}
    selected_prototypes: dict[str, int] = {}
    classifier_candidates = [
        candidate for candidate in bundle.classifier_prototypes
        if tuple(candidate["feature_mask"]) == tuple(feature_mask)
    ]
    coherent_candidate = (
        _nearest_prototype(signature, classifier_candidates)
        if needs_weight_repair and classifier_candidates else None
    )
    coherent_scenario = (
        coherent_candidate["scenario_id"] if coherent_candidate is not None else None
    )
    for name in selected:
        candidates = [
            candidate for candidate in bundle.repair_prototypes[name]
            if tuple(candidate["feature_mask"]) == tuple(feature_mask)
        ]
        if not candidates:
            raise ValueError(
                f"No repair prototype for module {name!r} and predicted feature mask"
            )
        candidate = next(
            (
                value for value in candidates
                if value["scenario_id"] == coherent_scenario
            ),
            None,
        )
        if candidate is None:
            candidate = _nearest_prototype(signature, candidates)
        module_index = bundle.module_names.index(name)
        predicted_magnitude = float(magnitudes[module_index].item())
        scale = predicted_magnitude / max(candidate["oracle_magnitude"], 1e-8)
        parameter_deltas[name] = candidate["parameter_delta"] * scale
        selected_prototypes[name] = candidate["scenario_id"]
    classifier_delta = None
    classifier_prototype = None
    if needs_weight_repair and bundle.classifier_name and coherent_candidate is not None:
        classifier_delta = coherent_candidate["parameter_delta"]
        classifier_prototype = coherent_candidate["scenario_id"]
    magnitude_map = {
        name: float(magnitudes[index].item())
        for index, name in enumerate(bundle.module_names)
    }
    update_parameter_count = sum(
        int(torch.count_nonzero(delta).item()) for delta in parameter_deltas.values()
    )
    if classifier_delta is not None:
        update_parameter_count += int(torch.count_nonzero(classifier_delta).item())
    plan = RepairPlan(
        affected_modules=selected,
        module_magnitudes=magnitude_map,
        confidence=confidence,
        parameter_deltas=parameter_deltas,
        update_parameter_count=update_parameter_count,
        update_bytes=(update_parameter_count * parameter_bits + 7) // 8,
        classifier_name=bundle.classifier_name,
        classifier_delta=classifier_delta,
        feature_mask=feature_mask,
    )
    return plan, {
        "repair_feature_mask": feature_mask,
        "selected_repair_prototypes": selected_prototypes,
        "selected_classifier_repair_prototype": classifier_prototype,
        "predicted_repair_parameter_count": plan.update_parameter_count,
        "predicted_repair_update_bytes": plan.update_bytes,
    }


def _prediction_metrics(bundle: AdapterBundle, state: dict) -> dict:
    return {
        "adapter_confidence": state["confidence"],
        "adapter_module_probabilities": {
            name: float(state["probabilities"][index].item())
            for index, name in enumerate(bundle.module_names)
        },
        "predicted_modules": list(state["selected"]),
        "predicted_module_magnitudes": {
            name: float(state["magnitudes"][index].item())
            for index, name in enumerate(bundle.module_names)
        },
    }


@torch.inference_mode()
def predict_repair(
    bundle: AdapterBundle,
    signature_vector: list[float] | torch.Tensor,
    *,
    module_threshold: float = 0.50,
    parameter_bits: int = 32,
) -> tuple[RepairPlan, dict]:
    """Predict the existing joint mask-compatible repair candidate."""
    state = _prediction_state(
        bundle, signature_vector, module_threshold=module_threshold,
    )
    feature_mask = _decode_feature_mask(
        state["feature_probabilities"],
        _compatible_feature_masks(bundle, state["selected"]),
    )
    plan, plan_metrics = _build_plan_for_mask(
        bundle, state, feature_mask, parameter_bits=parameter_bits,
    )
    return plan, {
        **_prediction_metrics(bundle, state),
        "predicted_feature_mask": feature_mask,
        **plan_metrics,
    }


@torch.inference_mode()
def predict_staged_repair(
    bundle: AdapterBundle,
    signature_vector: list[float] | torch.Tensor,
    current_feature_mask: Sequence[int],
    *,
    module_threshold: float = 0.50,
    parameter_bits: int = 32,
) -> tuple[RepairPlan, list[int] | None, dict]:
    """Build strict weight-first and optional feature-stage repair plans.

    The plan retains the currently deployed mask, so it is a true weight-only
    stage. The optional feature-stage result is only a mask: runtime applies it
    to the already weight-repaired model without applying a second set of
    weight deltas. Runtime decides whether that mask is needed only after
    evaluating the weight stage.
    """
    state = _prediction_state(
        bundle, signature_vector, module_threshold=module_threshold,
    )
    current = [int(value) for value in current_feature_mask]
    compatible = _compatible_feature_masks(bundle, state["selected"])
    if tuple(current) not in compatible:
        raise ValueError("Current feature mask has no compatible repair prototypes")

    weight_plan, weight_metrics = _build_plan_for_mask(
        bundle, state, current, parameter_bits=parameter_bits,
    )
    feature_candidates = [
        mask for mask in compatible
        if tuple(mask) != tuple(current)
    ] if bundle.feature_reconfiguration_config["enabled"] else []
    feature_mask = None
    if feature_candidates:
        feature_mask = _decode_feature_mask(
            state["feature_probabilities"], feature_candidates,
        )

    predicted_feature_mask = (
        list(feature_mask) if feature_mask is not None else current
    )
    return weight_plan, feature_mask, {
        **_prediction_metrics(bundle, state),
        "predicted_feature_mask": predicted_feature_mask,
        "weight_stage_prediction": weight_metrics,
        "feature_stage_prediction": (
            {"feature_mask": list(feature_mask)}
            if feature_mask is not None else None
        ),
    }


def save_adapter_bundle(bundle: AdapterBundle, path: Path, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": ADAPTER_FORMAT_VERSION,
        "signature_size": bundle.signature_size,
        "module_names": bundle.module_names,
        "feature_count": bundle.adapter.feature_count,
        "hidden_size": bundle.adapter.hidden_size,
        "adapter_state_dict": bundle.adapter.state_dict(),
        "repair_prototypes": bundle.repair_prototypes,
        "classifier_name": bundle.classifier_name,
        "classifier_prototypes": bundle.classifier_prototypes,
        "feature_names": bundle.feature_names,
        "signature_config": bundle.signature_config,
        "base_model_fingerprint": bundle.base_model_fingerprint,
        "signature_fields": bundle.signature_fields,
        "signature_format_version": bundle.signature_format_version,
        "oracle_module_threshold": bundle.oracle_module_threshold,
        "default_feature_mask": bundle.default_feature_mask,
        "feature_reference_values": bundle.feature_reference_values.float().cpu(),
        "feature_reconfiguration_config": bundle.feature_reconfiguration_config,
        "metadata": metadata or {},
    }, path)


def load_adapter_bundle(
    path: Path,
    model: nn.Module,
    device: torch.device,
    *,
    feature_names: list[str],
    signature_config: dict,
    feature_reconfiguration_config: dict,
    confidence_threshold: float,
    module_threshold: float,
) -> AdapterBundle:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != ADAPTER_FORMAT_VERSION:
        raise ValueError(
            "Unsupported drift-adapter checkpoint format; regenerate the offline "
            "drift-repair dataset and drift-adapter checkpoint"
        )
    if (
        payload.get("signature_format_version") != DRIFT_SIGNATURE_FORMAT_VERSION
        or payload.get("signature_fields") != list(SIGNATURE_FIELDS)
    ):
        raise ValueError("Drift adapter signature format does not match runtime")
    expected_signature_size = len(feature_names) * len(SIGNATURE_FIELDS)
    if payload.get("signature_size") != expected_signature_size:
        raise ValueError("Drift adapter signature dimensions do not match the dataset")
    if payload.get("feature_count") != len(feature_names):
        raise ValueError("Drift adapter feature-head dimensions do not match the dataset")
    validation = payload.get("metadata", {}).get("scientific_validation", {})
    if not validation.get("scientific_validation_passed", False):
        raise ValueError("Drift adapter checkpoint has not passed scientific validation")
    if float(validation.get("evaluated_confidence_threshold", -1.0)) != float(
        confidence_threshold
    ):
        raise ValueError("Drift adapter confidence threshold was not scientifically validated")
    if float(validation.get("evaluated_module_threshold", -1.0)) != float(module_threshold):
        raise ValueError("Drift adapter module threshold was not scientifically validated")
    if not 0.0 <= float(payload.get("oracle_module_threshold", -1.0)) <= 1.0:
        raise ValueError("Drift adapter oracle module threshold is invalid")
    if payload.get("base_model_fingerprint") != model_fingerprint(model):
        raise ValueError("Drift adapter was trained for different base-model weights")
    if list(feature_names) != payload.get("feature_names"):
        raise ValueError("Drift adapter feature ordering does not match the dataset")
    if dict(signature_config) != payload.get("signature_config"):
        raise ValueError("Drift adapter signature configuration does not match runtime")
    if dict(feature_reconfiguration_config) != payload.get(
        "feature_reconfiguration_config"
    ):
        raise ValueError(
            "Drift adapter feature-reconfiguration configuration does not match runtime"
        )
    feature_config = dict(payload["feature_reconfiguration_config"])
    default_feature_mask = list(payload.get("default_feature_mask", []))
    budget = int(feature_config.get("active_feature_budget", 0))
    if (
        not isinstance(feature_config.get("enabled"), bool)
        or not 0.0 <= float(feature_config.get("minimum_oracle_gain", -1.0)) <= 1.0
        or int(feature_config.get("max_candidate_masks", 0)) < 1
        or not 1 <= budget <= len(feature_names)
    ):
        raise ValueError("Drift adapter feature-reconfiguration metadata is invalid")
    if (
        len(default_feature_mask) != len(feature_names)
        or any(value not in {0, 1} for value in default_feature_mask)
        or sum(default_feature_mask) > budget
        or default_feature_mask != feature_config.get("default_feature_mask")
    ):
        raise ValueError("Drift adapter default feature mask is invalid")
    reference_values = torch.as_tensor(
        payload.get("feature_reference_values", []), dtype=torch.float32,
    ).flatten()
    if reference_values.numel() != len(feature_names) or not torch.isfinite(
        reference_values
    ).all():
        raise ValueError("Drift adapter feature-reference values are invalid")

    def prototype_mask(candidate: dict) -> tuple[int, ...]:
        mask = tuple(candidate.get("feature_mask", []))
        if (
            len(mask) != len(feature_names)
            or any(value not in {0, 1} for value in mask)
            or sum(mask) > budget
        ):
            raise ValueError("Drift adapter repair prototype has an invalid feature mask")
        return mask

    module_names = list(model.named_hidden_modules())
    if module_names != payload["module_names"]:
        raise ValueError("Drift adapter hidden-module layout does not match the model")
    if set(payload["repair_prototypes"]) != set(module_names):
        raise ValueError("Drift adapter repair-module schema does not match the model")
    hidden_modules = model.named_hidden_modules()
    for name, candidates in payload["repair_prototypes"].items():
        if not candidates:
            raise ValueError(f"Drift adapter has no repair prototypes for {name}")
        expected = sum(parameter.numel() for parameter in hidden_modules[name].parameters())
        if any(
            candidate["parameter_delta"].numel() != expected
            or candidate["signature"].numel() != expected_signature_size
            for candidate in candidates
        ):
            raise ValueError(
                "Drift adapter hidden-module prototype dimensions do not match runtime"
            )
        for candidate in candidates:
            prototype_mask(candidate)
    classifier_name, classifier = model.named_classifier_module()
    if classifier_name != payload["classifier_name"]:
        raise ValueError("Drift adapter classifier layout does not match the model")
    classifier_expected = sum(parameter.numel() for parameter in classifier.parameters())
    if not payload["classifier_prototypes"]:
        raise ValueError("Drift adapter has no classifier repair prototypes")
    if any(
        candidate["parameter_delta"].numel() != classifier_expected
        or candidate["signature"].numel() != expected_signature_size
        for candidate in payload["classifier_prototypes"]
    ):
        raise ValueError("Drift adapter classifier prototype dimensions do not match runtime")
    classifier_keys = {
        (int(candidate["scenario_id"]), prototype_mask(candidate))
        for candidate in payload["classifier_prototypes"]
    }
    if not any(mask == tuple(default_feature_mask) for _, mask in classifier_keys):
        raise ValueError("Drift adapter has no default-mask repair prototype")
    for name, candidates in payload["repair_prototypes"].items():
        module_keys = {
            (int(candidate["scenario_id"]), prototype_mask(candidate))
            for candidate in candidates
        }
        if module_keys != classifier_keys:
            raise ValueError(
                f"Drift adapter repair prototypes for {name!r} are not feature-compatible"
            )
    adapter = DriftConditionedAdapter(
        payload["signature_size"], len(module_names), len(feature_names),
        payload["hidden_size"],
    ).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()
    return AdapterBundle(
        adapter=adapter,
        module_names=module_names,
        repair_prototypes=payload["repair_prototypes"],
        signature_size=payload["signature_size"],
        classifier_name=payload["classifier_name"],
        classifier_prototypes=payload["classifier_prototypes"],
        feature_names=payload["feature_names"],
        signature_config=payload["signature_config"],
        base_model_fingerprint=payload["base_model_fingerprint"],
        signature_fields=payload["signature_fields"],
        signature_format_version=payload["signature_format_version"],
        oracle_module_threshold=payload["oracle_module_threshold"],
        default_feature_mask=default_feature_mask,
        feature_reference_values=reference_values,
        feature_reconfiguration_config=feature_config,
    )


@torch.inference_mode()
def evaluate_adapter(
    dataset: dict,
    bundle: AdapterBundle,
    original_model: nn.Module,
    *,
    module_threshold: float = 0.50,
    confidence_threshold: float = 0.50,
    parameter_bits: int = 32,
) -> dict:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    rows = []
    for record in dataset["records"]:
        plan, prediction = predict_repair(
            bundle, record["drift_signature_vector"],
            module_threshold=module_threshold, parameter_bits=parameter_bits,
        )
        repair_applied = bool(
            prediction["adapter_confidence"] >= confidence_threshold
            and (
                plan.affected_modules
                or plan.classifier_delta is not None
            )
        )
        effective_plan = plan if repair_applied else RepairPlan(
            affected_modules=[],
            module_magnitudes=plan.module_magnitudes,
            confidence=plan.confidence,
            parameter_deltas={},
            feature_mask=list(bundle.default_feature_mask),
        )
        effective_feature_mask = list(effective_plan.feature_mask or bundle.default_feature_mask)
        candidate, repair_metrics = apply_repair_plan(
            original_model, effective_plan, parameter_bits=parameter_bits,
            feature_mask=effective_feature_mask,
        )
        device = next(candidate.parameters()).device
        features = apply_feature_mask(
            record["drifted_features"].to(device), effective_feature_mask,
            bundle.feature_reference_values,
        )
        labels = record["labels"].to(device)
        repaired_f1 = macro_f1(labels, candidate.run_inference(features)[1])
        truth = set(record["oracle_modules"])
        predicted = set(prediction["predicted_modules"])
        union = truth | predicted
        module_accuracy = len(truth & predicted) / len(union) if union else 1.0
        rows.append({
            "scenario_id": record["scenario_id"],
            "split": record["split"],
            "predicted_modules": prediction["predicted_modules"],
            "oracle_modules": record["oracle_modules"],
            "module_prediction_accuracy": module_accuracy,
            "adapter_confidence": prediction["adapter_confidence"],
            "predicted_feature_mask": prediction["predicted_feature_mask"],
            "oracle_feature_mask": record["oracle_feature_mask"],
            "selected_feature_mask": _feature_target(record),
            "feature_mask_accuracy": sum(
                int(left == right) for left, right in zip(
                    prediction["predicted_feature_mask"], _feature_target(record)
                )
            ) / len(record["default_feature_mask"]),
            "feature_reconfiguration_required": record[
                "feature_reconfiguration_required"
            ],
            "repair_applied": repair_applied,
            "f1_after_drift": record["f1_after_drift"],
            "f1_after_predicted_repair": repaired_f1,
            "oracle_weight_only_f1": record["oracle_weight_only_f1"],
            "oracle_feature_weight_f1": record["oracle_feature_weight_f1"],
            "feature_gain": record["feature_gain"],
            "oracle_retrained_f1": record["oracle_retrained_f1"],
            "oracle_improved": (
                record["oracle_retrained_f1"] - record["f1_after_drift"] > 1e-12
            ),
            "recovery_ratio": recovery_ratio(
                repaired_f1, record["f1_after_drift"], record["oracle_retrained_f1"],
            ),
            **repair_metrics,
        })

    def average(key: str, selected: list[dict]) -> float | None:
        values = [row[key] for row in selected if row[key] is not None]
        return sum(values) / len(values) if values else None

    split_reports = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        split_reports[split] = {
            "scenario_count": len(selected),
            "module_prediction_accuracy": average("module_prediction_accuracy", selected),
            "feature_mask_accuracy": average("feature_mask_accuracy", selected),
            "recovery_ratio": average("recovery_ratio", selected),
            "eligible_recovery_scenarios": sum(
                row["recovery_ratio"] is not None for row in selected
            ),
            "oracle_improvement_rate": sum(
                row["oracle_improved"] for row in selected
            ) / len(selected),
            "repair_coverage": sum(row["repair_applied"] for row in selected) / len(selected),
        }
    return {
        "module_prediction_accuracy": average("module_prediction_accuracy", rows),
        "feature_mask_accuracy": average("feature_mask_accuracy", rows),
        "recovery_ratio": average("recovery_ratio", rows),
        "eligible_recovery_scenarios": sum(
            row["recovery_ratio"] is not None for row in rows
        ),
        "oracle_improvement_rate": sum(row["oracle_improved"] for row in rows) / len(rows),
        "repair_coverage": sum(row["repair_applied"] for row in rows) / len(rows),
        "splits": split_reports,
        "scenarios": rows,
    }
