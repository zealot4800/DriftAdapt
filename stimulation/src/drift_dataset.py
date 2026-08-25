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
from .repair import extract_oracle_repair, model_fingerprint
from .trainer import retrain_model


DATASET_FORMAT_VERSION = 4


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
        oracle = copy.deepcopy(model)
        for parameter in oracle.parameters():
            parameter.requires_grad_(True)
        f1_clean = macro_f1(
            evaluation_window.labels,
            _predict(original, evaluation_window.features),
        )
        f1_drifted = macro_f1(
            evaluation_window.labels,
            _predict(original, evaluation_features),
        )
        training_time, _ = retrain_model(
            oracle, oracle_training_features, oracle_training_window.labels,
            epochs=epochs,
            batch_size=int(training_config["batch_size"]),
            learning_rate=float(training_config["learning_rate"]),
            optimizer_name=training_config.get("optimizer", "Adam"),
            optimizer=None,
        )
        f1_oracle = macro_f1(
            evaluation_window.labels,
            _predict(oracle, evaluation_features),
        )
        repair = extract_oracle_repair(
            original, oracle, module_threshold=module_threshold,
        )
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
            "oracle_retrained_f1": f1_oracle,
            "oracle_f1_before_adaptation": f1_drifted,
            "oracle_f1_after_adaptation": f1_oracle,
            "oracle_training_time": training_time,
            "injection": injection,
            **repair,
        })

    return {
        "format_version": DATASET_FORMAT_VERSION,
        "seed": seed,
        "feature_names": feature_names,
        "signature_fields_per_feature": len(SIGNATURE_FIELDS),
        "signature_fields": list(SIGNATURE_FIELDS),
        "signature_format_version": DRIFT_SIGNATURE_FORMAT_VERSION,
        "module_names": list(model.named_hidden_modules()),
        "oracle_module_threshold": module_threshold,
        "model_fingerprint": model_fingerprint(model),
        "signature_config": {
            "bins": int(drift_config.get("bins", 16)),
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
        raise ValueError("Unsupported drift-repair dataset format")
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
    })
    return report
