from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch

from .drift import FeatureHistogramDriftDetector
from .drift_injector import TrafficDriftInjector
from .drift_signature import (
    DRIFT_SIGNATURE_FORMAT_VERSION,
    SIGNATURE_FIELDS,
    build_drift_signature,
)
from .metrics import macro_f1
from .repair import apply_feature_mask, extract_oracle_repair, model_fingerprint
from .trainer import retrain_model


DATASET_FORMAT_VERSION = 5


def resolve_feature_reconfiguration_config(
    config: dict | None,
    feature_count: int,
) -> dict:
    """Return the validated, explicit feature-reconfiguration configuration."""
    if feature_count < 1:
        raise ValueError("Feature reconfiguration needs at least one configured feature")
    source = dict(config or {})
    enabled = source.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("feature_reconfiguration.enabled must be boolean")
    minimum_gain = float(source.get("minimum_oracle_gain", 0.02))
    max_candidate_masks = int(source.get("max_candidate_masks", 32))
    active_budget = int(source.get("active_feature_budget", feature_count))
    raw_default = source.get("default_feature_mask", [1] * feature_count)
    if not isinstance(raw_default, (list, tuple)):
        raise ValueError(
            "feature_reconfiguration.default_feature_mask must be a list"
        )
    if len(raw_default) != feature_count or any(
        value != 0 and value != 1 for value in raw_default
    ):
        raise ValueError(
            "feature_reconfiguration.default_feature_mask must contain one binary "
            "value per configured feature"
        )
    default_mask = [int(value) for value in raw_default]
    if not 0.0 <= minimum_gain <= 1.0:
        raise ValueError(
            "feature_reconfiguration.minimum_oracle_gain must be between zero and one"
        )
    if max_candidate_masks < 1:
        raise ValueError("feature_reconfiguration.max_candidate_masks must be positive")
    if not 1 <= active_budget <= feature_count:
        raise ValueError(
            "feature_reconfiguration.active_feature_budget must be between one and "
            "the configured feature count"
        )
    if sum(default_mask) > active_budget:
        raise ValueError(
            "feature_reconfiguration.default_feature_mask exceeds the active feature budget"
        )
    return {
        "enabled": enabled,
        "minimum_oracle_gain": minimum_gain,
        "max_candidate_masks": max_candidate_masks,
        "active_feature_budget": active_budget,
        "default_feature_mask": default_mask,
    }


def candidate_feature_masks(
    default_feature_mask: list[int],
    drift_scores: list[float | None],
    *,
    active_feature_budget: int,
    max_candidate_masks: int,
) -> list[list[int]]:
    """Build a deterministic, bounded set of drift-informed feature masks."""
    feature_count = len(default_feature_mask)
    config = resolve_feature_reconfiguration_config({
        "enabled": True,
        "default_feature_mask": default_feature_mask,
        "active_feature_budget": active_feature_budget,
        "max_candidate_masks": max_candidate_masks,
    }, feature_count)
    default = tuple(config["default_feature_mask"])
    scores = [
        float(value) if value is not None else float("-inf")
        for value in drift_scores
    ]
    if len(scores) != feature_count:
        raise ValueError("Drift scores must match the feature mask width")
    active = [index for index, value in enumerate(default) if value]
    inactive = [index for index, value in enumerate(default) if not value]
    drifted_active = sorted(
        (index for index in active if scores[index] != float("-inf")),
        key=lambda index: (-scores[index], index),
    )[:4]
    stable_inactive = sorted(inactive, key=lambda index: (scores[index], index))
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(mask: list[int] | tuple[int, ...]) -> None:
        if len(candidates) >= max_candidate_masks:
            return
        value = tuple(int(bit) for bit in mask)
        if (
            value not in seen
            and len(value) == feature_count
            and all(bit in {0, 1} for bit in value)
            and sum(value) <= active_feature_budget
        ):
            seen.add(value)
            candidates.append(value)

    add(default)

    # Prefer a small set of equal-cardinality reconfigurations when inactive
    # alternatives exist, without letting swaps crowd out simpler candidates.
    single_swap_limit = min(8, max_candidate_masks - 1)
    single_swap_count = 0
    for removed in drifted_active:
        for activated in stable_inactive:
            mask = list(default)
            mask[removed] = 0
            mask[activated] = 1
            add(mask)
            single_swap_count += 1
            if single_swap_count >= single_swap_limit:
                break
        if single_swap_count >= single_swap_limit:
            break

    for removed in drifted_active:
        mask = list(default)
        mask[removed] = 0
        add(mask)
    if sum(default) < active_feature_budget:
        for activated in stable_inactive:
            if len(candidates) >= max_candidate_masks:
                break
            mask = list(default)
            mask[activated] = 1
            add(mask)
    double_swap_count = 0
    for removed_pair in itertools.combinations(drifted_active, 2):
        for activated_pair in itertools.combinations(stable_inactive, 2):
            mask = list(default)
            for index in removed_pair:
                mask[index] = 0
            for index in activated_pair:
                mask[index] = 1
            add(mask)
            double_swap_count += 1
            if double_swap_count >= 4:
                break
        if double_swap_count >= 4:
            break
    for removed_pair in itertools.combinations(drifted_active, 2):
        if len(candidates) >= max_candidate_masks:
            break
        mask = list(default)
        mask[removed_pair[0]] = 0
        mask[removed_pair[1]] = 0
        add(mask)
    return [list(mask) for mask in candidates[:max_candidate_masks]]


def _balanced_feature_pairs(feature_count: int, count: int | None) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations(range(feature_count), 2))
    if count is None or count >= len(pairs):
        return pairs
    if count < 0:
        raise ValueError("max_feature_pairs must be nonnegative")
    degrees = [0] * feature_count
    selected: list[tuple[int, int]] = []
    remaining = set(pairs)
    while len(selected) < count:
        pair = min(
            remaining,
            key=lambda value: (
                degrees[value[0]] + degrees[value[1]],
                max(degrees[value[0]], degrees[value[1]]),
                value,
            ),
        )
        selected.append(pair)
        remaining.remove(pair)
        degrees[pair[0]] += 1
        degrees[pair[1]] += 1
    return selected


def controlled_drift_scenarios(
    feature_count: int,
    severities: list[float],
    *,
    max_feature_pairs: int | None = None,
    drift_type: str = "abrupt",
    directions: tuple[int, ...] = (-1, 1),
) -> list[dict]:
    """Create deterministic single-feature and held-out pair scenarios."""
    if feature_count < 2 or len(set(severities)) < 2 or any(value <= 0 for value in severities):
        raise ValueError("Drift scenarios require features and at least two positive severities")
    ordered_severities = sorted(set(float(value) for value in severities))
    if not directions or any(direction not in {-1, 1} for direction in directions):
        raise ValueError("Drift scenario directions must contain only -1 and 1")
    pairs = _balanced_feature_pairs(feature_count, max_feature_pairs)
    scenarios = []
    for direction in directions:
        for severity in ordered_severities:
            for feature in range(feature_count):
                scenarios.append({
                    "feature_indices": [feature],
                    "severity": severity,
                    "direction": direction,
                    "drift_type": drift_type,
                    "split": "unseen_severity" if severity == ordered_severities[-1] else "train",
                })
            for pair in pairs:
                scenarios.append({
                    "feature_indices": list(pair),
                    "severity": severity,
                    "direction": direction,
                    "drift_type": drift_type,
                    "split": "unseen_combination",
                })
    return scenarios


@torch.inference_mode()
def _predict(model, features: torch.Tensor) -> torch.Tensor:
    return model.run_inference(features)[1]


def generate_drift_repair_dataset(
    model,
    windows: list,
    feature_names: list[str],
    drift_config: dict,
    training_config: dict,
    scenarios: list[dict],
    *,
    seed: int,
    oracle_epochs: int | None = None,
    module_threshold: float = 0.30,
    signature_shift_clip: float = 8.0,
    feature_reconfiguration_config: dict | None = None,
) -> dict:
    """Generate oracle repairs; labels are consumed only inside this offline function."""
    reference_windows = int(drift_config.get("reference_windows", 10))
    if len(windows) <= reference_windows + 1:
        raise ValueError(
            "Offline drift dataset needs separate oracle-training and evaluation windows"
        )
    torch.manual_seed(seed)
    detector = FeatureHistogramDriftDetector(
        feature_names,
        bins=int(drift_config.get("bins", 16)),
        threshold=float(drift_config.get("threshold", 0.10)),
        consecutive_windows=1,
        reference_windows=reference_windows,
        device=windows[0].features.device,
    )
    for window in windows[:reference_windows]:
        detector.update(window.features)
    frozen_detector = copy.deepcopy(detector)
    feature_config = resolve_feature_reconfiguration_config(
        feature_reconfiguration_config, len(feature_names),
    )
    reference_values = torch.cat([
        window.features for window in windows[:reference_windows]
    ]).mean(dim=0)
    if not torch.isfinite(reference_values).all():
        raise ValueError("Frozen feature-reference means must be finite")
    default_feature_mask = list(feature_config["default_feature_mask"])
    oracle_training_window = windows[reference_windows]
    evaluation_window = windows[reference_windows + 1]
    epochs = int(training_config["epochs"] if oracle_epochs is None else oracle_epochs)
    if epochs < 1:
        raise ValueError("Oracle epochs must be positive")
    records = []

    for scenario_index, scenario in enumerate(scenarios):
        scenario_seed = seed + scenario_index
        torch.manual_seed(scenario_seed)
        kind = scenario.get("drift_type", "abrupt")
        injector_config = {
            "enabled": True,
            "start_window": 1,
            "end_window": 1 if kind in {"gradual", "recurrent"} else None,
            "type": kind,
            "feature_indices": scenario["feature_indices"],
            "magnitude": float(scenario["severity"]),
            "direction": int(scenario.get("direction", 1)),
        }
        injector = TrafficDriftInjector(injector_config, feature_names)
        oracle_training_features, injection = injector.apply(
            oracle_training_window.features, 1,
        )
        evaluation_features, _ = injector.apply(evaluation_window.features, 1)
        scenario_detector = copy.deepcopy(frozen_detector)
        drift_result = scenario_detector.update(oracle_training_features)
        signature = build_drift_signature(
            feature_names,
            scenario_detector.reference_counts,
            oracle_training_features,
            scenario_detector.range_min,
            scenario_detector.range_max,
            drift_result["feature_drift_scores"],
            normalized_shift_clip=signature_shift_clip,
        )

        original = copy.deepcopy(model).eval()
        default_clean_features = apply_feature_mask(
            evaluation_window.features, default_feature_mask, reference_values,
        )
        default_drifted_features = apply_feature_mask(
            evaluation_features, default_feature_mask, reference_values,
        )
        f1_clean = macro_f1(
            evaluation_window.labels,
            _predict(original, default_clean_features),
        )
        f1_drifted = macro_f1(
            evaluation_window.labels,
            _predict(original, default_drifted_features),
        )

        def train_oracle(feature_mask: list[int]) -> dict:
            # The same per-scenario seed makes mask comparisons independent of
            # DataLoader shuffle order.
            torch.manual_seed(scenario_seed)
            oracle = copy.deepcopy(model)
            for parameter in oracle.parameters():
                parameter.requires_grad_(True)
            masked_training = apply_feature_mask(
                oracle_training_features, feature_mask, reference_values,
            )
            masked_evaluation = apply_feature_mask(
                evaluation_features, feature_mask, reference_values,
            )
            training_time, _ = retrain_model(
                oracle, masked_training, oracle_training_window.labels,
                epochs=epochs,
                batch_size=int(training_config["batch_size"]),
                learning_rate=float(training_config["learning_rate"]),
                optimizer_name=training_config.get("optimizer", "Adam"),
                optimizer=None,
            )
            f1 = macro_f1(
                evaluation_window.labels, _predict(oracle, masked_evaluation),
            )
            return {
                "feature_mask": list(feature_mask),
                "f1": f1,
                "training_time": training_time,
                **extract_oracle_repair(
                    original, oracle, module_threshold=module_threshold,
                ),
            }

        scores = [
            drift_result["feature_drift_scores"][name]
            for name in feature_names
        ]
        masks = [default_feature_mask]
        if feature_config["enabled"]:
            masks = candidate_feature_masks(
                default_feature_mask, scores,
                active_feature_budget=feature_config["active_feature_budget"],
                max_candidate_masks=feature_config["max_candidate_masks"],
            )
        oracle_candidates = [train_oracle(mask) for mask in masks]
        weight_only = oracle_candidates[0]
        best_candidate = max(oracle_candidates, key=lambda candidate: candidate["f1"])
        feature_gain = best_candidate["f1"] - weight_only["f1"]
        reconfiguration_required = bool(
            feature_config["enabled"]
            and best_candidate["feature_mask"] != default_feature_mask
            and feature_gain > 0.0
            and feature_gain >= feature_config["minimum_oracle_gain"]
        )
        effective = best_candidate if reconfiguration_required else weight_only
        repair_candidates = [{
            key: value for key, value in candidate.items()
            if key not in {"training_time"}
        } for candidate in oracle_candidates]
        repair = {
            key: value for key, value in effective.items()
            if key not in {"feature_mask", "f1", "training_time"}
        }
        records.append({
            "scenario_id": scenario_index,
            "split": scenario["split"],
            "drift_type": kind,
            "severity": float(scenario["severity"]),
            "direction": int(scenario.get("direction", 1)),
            "affected_feature_indices": list(scenario["feature_indices"]),
            "affected_feature_names": [feature_names[i] for i in scenario["feature_indices"]],
            "drift_signature_vector": torch.tensor(
                signature["drift_signature_vector"], dtype=torch.float32,
            ).cpu(),
            "drift_signature_per_feature": signature["drift_signature_per_feature"],
            "drifted_features": evaluation_features.detach().cpu(),
            "labels": evaluation_window.labels.detach().cpu(),
            "oracle_training_sample_count": int(oracle_training_window.labels.numel()),
            "oracle_evaluation_sample_count": int(evaluation_window.labels.numel()),
            "oracle_training_window": reference_windows + 1,
            "oracle_evaluation_window": reference_windows + 2,
            "model_f1_before_drift": f1_clean,
            "f1_after_drift": f1_drifted,
            "default_feature_mask": list(default_feature_mask),
            "oracle_feature_mask": list(best_candidate["feature_mask"]),
            "selected_feature_mask": list(effective["feature_mask"]),
            "feature_reconfiguration_required": reconfiguration_required,
            "oracle_weight_only_f1": weight_only["f1"],
            "oracle_feature_weight_f1": best_candidate["f1"],
            "feature_gain": feature_gain,
            "oracle_candidate_mask_count": len(oracle_candidates),
            "oracle_repair_candidates": repair_candidates,
            "oracle_retrained_f1": effective["f1"],
            "oracle_f1_before_adaptation": f1_drifted,
            "oracle_f1_after_adaptation": effective["f1"],
            "oracle_training_time": effective["training_time"],
            "oracle_search_training_time": sum(
                candidate["training_time"] for candidate in oracle_candidates
            ),
            "injection": injection,
            **repair,
        })

    return {
        "format_version": DATASET_FORMAT_VERSION,
        "seed": seed,
        "feature_names": feature_names,
        "feature_reconfiguration_config": feature_config,
        "default_feature_mask": default_feature_mask,
        "feature_reference_values": reference_values.detach().cpu(),
        "signature_fields_per_feature": len(SIGNATURE_FIELDS),
        "signature_fields": list(SIGNATURE_FIELDS),
        "signature_format_version": DRIFT_SIGNATURE_FORMAT_VERSION,
        "module_names": list(model.named_hidden_modules()),
        "oracle_module_threshold": module_threshold,
        "model_fingerprint": model_fingerprint(model),
        "signature_config": {
            "bins": int(drift_config.get("bins", 16)),
            "threshold": float(drift_config.get("threshold", 0.10)),
            "consecutive_windows": int(drift_config.get("consecutive_windows", 3)),
            "reference_windows": reference_windows,
            "normalized_shift_clip": float(signature_shift_clip),
        },
        "records": records,
    }


def save_drift_repair_dataset(dataset: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def load_drift_repair_dataset(path: Path) -> dict:
    try:
        dataset = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        dataset = torch.load(path, map_location="cpu")
    if dataset.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            "Unsupported drift-repair dataset format; regenerate the offline "
            "drift-repair dataset and drift-adapter checkpoint"
        )
    return dataset


def analyze_drift_repair_learnability(dataset: dict) -> dict:
    """Compare pairwise drift signatures with signed oracle parameter repairs."""
    records = dataset["records"]
    module_names = dataset["module_names"]

    def similarity_report(selected: list[dict], include_values: bool = False) -> dict:
        signature_similarities: list[float] = []
        repair_similarities: list[float] = []
        for left, right in itertools.combinations(selected, 2):
            left_signature = left["drift_signature_vector"].float()
            right_signature = right["drift_signature_vector"].float()
            left_repair = torch.cat([
                *(left["oracle_parameter_deltas"][name].float() for name in module_names),
                left["oracle_classifier_parameter_delta"].float(),
            ])
            right_repair = torch.cat([
                *(right["oracle_parameter_deltas"][name].float() for name in module_names),
                right["oracle_classifier_parameter_delta"].float(),
            ])
            signature_similarities.append(float(torch.nn.functional.cosine_similarity(
                left_signature[None], right_signature[None], dim=1, eps=1e-8,
            ).item()))
            repair_similarities.append(float(torch.nn.functional.cosine_similarity(
                left_repair[None], right_repair[None], dim=1, eps=1e-8,
            ).item()))
        correlation = float("nan")
        if len(signature_similarities) >= 2:
            correlation = float(torch.corrcoef(torch.tensor([
                signature_similarities, repair_similarities,
            ], dtype=torch.float64))[0, 1].item())
        report = {
            "scenario_count": len(selected),
            "pair_count": len(signature_similarities),
            "signature_repair_similarity_correlation": correlation,
            "feature_reconfiguration_rate": (
                sum(record["feature_reconfiguration_required"] for record in selected)
                / len(selected) if selected else 0.0
            ),
            "average_feature_gain": (
                sum(record["feature_gain"] for record in selected) / len(selected)
                if selected else 0.0
            ),
        }
        if include_values:
            report.update({
                "drift_signature_similarity": signature_similarities,
                "oracle_repair_similarity": repair_similarities,
            })
        return report

    report = similarity_report(records, include_values=True)
    report.update({
        "train_scenarios": sum(record["split"] == "train" for record in records),
        "unseen_combination_scenarios": sum(
            record["split"] == "unseen_combination" for record in records
        ),
        "unseen_severity_scenarios": sum(
            record["split"] == "unseen_severity" for record in records
        ),
        "splits": {
            split: similarity_report([
                record for record in records if record["split"] == split
            ])
            for split in sorted({record["split"] for record in records})
        },
        "held_out": similarity_report([
            record for record in records if record["split"] != "train"
        ]),
        "feature_oracle_scenarios": [{
            "scenario_id": record["scenario_id"],
            "split": record["split"],
            "oracle_weight_only_f1": record["oracle_weight_only_f1"],
            "oracle_feature_weight_f1": record["oracle_feature_weight_f1"],
            "feature_gain": record["feature_gain"],
            "feature_reconfiguration_required": record[
                "feature_reconfiguration_required"
            ],
        } for record in records],
    })
    return report
