from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .adapter import load_adapter_bundle, predict_repair
from .data import StreamingDataset, resolve_path
from .drift import COUNTER_BITS, FeatureHistogramDriftDetector
from .drift_injector import TrafficDriftInjector
from .drift_signature import build_drift_signature
from .labelers import ABSTAIN, build_labeler
from .metrics import macro_f1, mean
from .models import build_model, load_checkpoint
from .repair import apply_repair_plan
from .trainer import form_training_set, retrain_model
from .trigger import RetrainingTrigger


SUPPORTED_POLICIES = {"caravan", "drift_conditioned"}
DEPRECATED_POLICY_ALIASES = {"baseline_caravan": "caravan"}


class DriftAdaptPipeline:
    def __init__(self, config: dict, config_path: Path) -> None:
        self.config = config
        runtime = config.get("runtime", {})
        requested = runtime.get("device", "cuda")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        torch.manual_seed(int(runtime.get("seed", 42)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(runtime.get("seed", 42)))

        base = config_path.parent
        dataset_config = copy.deepcopy(config["dataset"])
        dataset_config["files"] = [str(resolve_path(path, base)) for path in dataset_config["files"]]
        self.dataset = StreamingDataset(dataset_config, base, self.device)
        drift_config = config.get("drift", {})
        self.drift_detector = FeatureHistogramDriftDetector(
            list(dataset_config["feature_names"]),
            bins=int(drift_config.get("bins", 16)),
            threshold=float(drift_config.get("threshold", 0.10)),
            consecutive_windows=int(drift_config.get("consecutive_windows", 3)),
            reference_windows=int(drift_config.get("reference_windows", 10)),
            device=self.device,
        )
        self.drift_injector = TrafficDriftInjector(
            config.get("drift_injection", {}), list(dataset_config["feature_names"]),
        )

        model_config = config["model"]
        self.model = build_model(model_config["class"], model_config.get("kwargs")).to(self.device)
        checkpoint = resolve_path(model_config["checkpoint"], base)
        load_checkpoint(self.model, str(checkpoint), self.device)
        self.model.eval()
        self.static_model = copy.deepcopy(self.model).eval()
        self.repair_base_model = copy.deepcopy(self.model).eval()

        policy = config.get("driftadapt", {})
        configured_mode = str(policy.get("mode", "caravan")).lower()
        self.policy_mode = DEPRECATED_POLICY_ALIASES.get(configured_mode, configured_mode)
        if self.policy_mode not in SUPPORTED_POLICIES:
            if configured_mode in {"driftadapt_full", "driftadapt_selective"}:
                raise ValueError(
                    f"Runtime mode {configured_mode!r} was removed; use 'caravan' "
                    "or 'drift_conditioned'"
                )
            raise ValueError("driftadapt.mode must be caravan or drift_conditioned")

        self.parameter_bits = int(policy.get("parameter_bits", 32))
        if self.parameter_bits < 1:
            raise ValueError("driftadapt.parameter_bits must be positive")
        self.labeler = None
        self.labeler_type = None
        self.trigger = None
        self.optimizer = None
        if self.policy_mode == "caravan":
            labeler_config = copy.deepcopy(config["labeler"])
            labeler_config.setdefault("feature_names", dataset_config["feature_names"])
            if labeler_config.get("checkpoint"):
                labeler_config["checkpoint"] = str(
                    resolve_path(labeler_config["checkpoint"], base)
                )
            self.labeler = build_labeler(labeler_config, self.device)
            self.labeler_type = str(labeler_config["type"]).lower()
            trigger_config = config["trigger"]
            if trigger_config.get("type") != "accuracy_proxy":
                raise ValueError("CARAVAN requires trigger.type: accuracy_proxy")
            self.trigger = RetrainingTrigger(
                threshold=float(trigger_config["threshold"]),
                drop_threshold=float(trigger_config.get("drop_threshold", 0.15)),
                consecutive_windows=int(trigger_config.get("consecutive_windows", 2)),
            )

        conditioned = config.get("drift_conditioned", {})
        self.adapter_bundle = None
        self.adapter_confidence_threshold = float(conditioned.get("confidence_threshold", 0.50))
        self.adapter_module_threshold = float(conditioned.get("module_threshold", 0.50))
        self.signature_shift_clip = float(conditioned.get("signature_shift_clip", 8.0))
        if not 0.0 <= self.adapter_confidence_threshold <= 1.0:
            raise ValueError("drift_conditioned.confidence_threshold must be between zero and one")
        if not 0.0 <= self.adapter_module_threshold <= 1.0:
            raise ValueError("drift_conditioned.module_threshold must be between zero and one")
        if self.signature_shift_clip <= 0:
            raise ValueError("drift_conditioned.signature_shift_clip must be positive")
        if self.policy_mode == "drift_conditioned":
            checkpoint_value = conditioned.get("adapter_checkpoint")
            if not checkpoint_value:
                raise ValueError(
                    "drift_conditioned.adapter_checkpoint is required for drift_conditioned mode"
                )
            self.adapter_bundle = load_adapter_bundle(
                resolve_path(checkpoint_value, base), self.model, self.device,
                feature_names=list(dataset_config["feature_names"]),
                signature_config={
                    "bins": int(drift_config.get("bins", 16)),
                    "reference_windows": int(drift_config.get("reference_windows", 10)),
                    "normalized_shift_clip": self.signature_shift_clip,
                },
                confidence_threshold=self.adapter_confidence_threshold,
                module_threshold=self.adapter_module_threshold,
            )
        self.model_version = 0
        self.drift_event_handled = False
        self.last_non_drift_f1: float | None = None
        self.base_dir = base

    @torch.inference_mode()
    def _predict(self, model, features):
        return model.run_inference(features)[1]

    def _empty_labeling_metrics(self) -> dict:
        return {
            "samples_sent_to_teacher": 0,
            "teacher_known_samples": 0,
            "teacher_unknown_samples": 0,
            "fallback_labeled_samples": 0,
            "abstained_samples": 0,
            "teacher_coverage": 0.0,
            "accepted_label_count": 0,
            "labeling_time": 0.0,
            "teacher_predicted_labels": [],
            "teacher_confidences": [],
            "teacher_ood_scores": [],
            "teacher_known_decisions": [],
        }

    def run(self) -> tuple[list[dict], dict]:
        rows: list[dict] = []
        stream = self.config["stream"]
        training = self.config["training"]
        previous_feature_drift = False

        for index, window in enumerate(self.dataset.windows(int(stream["window_size"])), start=1):
            stream_features, injected_drift = self.drift_injector.apply(window.features, index)
            feature_drift = self.drift_detector.update(stream_features)
            drift_signature = build_drift_signature(
                list(self.config["dataset"]["feature_names"]),
                self.drift_detector.reference_counts,
                stream_features,
                self.drift_detector.range_min,
                self.drift_detector.range_max,
                feature_drift["feature_drift_scores"],
                normalized_shift_clip=self.signature_shift_clip,
            )
            student_labels = self._predict(self.model, stream_features)
            static_labels = self._predict(self.static_model, stream_features)
            student_f1 = macro_f1(window.labels, student_labels)
            static_f1 = macro_f1(window.labels, static_labels)
            balanced = bool(training.get("balance_binary", self.config["dataset"]["num_classes"] == 2))
            labeling_metrics = self._empty_labeling_metrics()
            selected_indices = torch.empty(0, dtype=torch.long, device=self.device)
            proxy_f1 = float("nan")
            static_proxy_f1 = float("nan")
            labeler_f1 = float("nan")
            trusted_y = window.labels[:0]
            trusted_indices = selected_indices
            adaptation = {"adaptation_event": False}
            validation_modules: list[str] = []
            conditioned_metrics = {
                "adapter_prediction_attempted": False,
                "adapter_prediction_latency": 0.0,
                "adapter_confidence": None,
                "adapter_confidence_threshold": self.adapter_confidence_threshold,
                "adapter_confidence_passed": None,
                "adapter_module_probabilities": {},
                "predicted_modules": [],
                "predicted_module_magnitudes": {},
                "selected_repair_prototypes": {},
                "selected_classifier_repair_prototype": None,
                "predicted_repair_parameter_count": 0,
                "predicted_repair_update_bytes": 0,
                "model_f1_before_drift": self.last_non_drift_f1,
                "f1_after_drift": None,
                "f1_after_predicted_repair": None,
                "adaptation_latency": 0.0,
                "repair_affected_modules": [],
                "repair_module_magnitudes": {},
                "repair_confidence": None,
                "repair_changed_parameters": 0,
                "repair_changed_parameter_percentage": 0.0,
                "repair_update_bytes": 0,
            }

            if self.policy_mode == "caravan":
                labeling_started = time.perf_counter()
                produced = self.labeler.label(
                    stream_features, window.raw_features, window.generated_labels,
                )
                labeling_time = time.perf_counter() - labeling_started
                valid = (
                    produced != ABSTAIN
                    if self.labeler_type == "device_list"
                    else torch.ones_like(produced, dtype=torch.bool)
                )
                labeler_labels = (
                    window.labels[valid]
                    if self.labeler_type == "device_list"
                    else produced
                )
                training_features = stream_features[valid]
                proxy_student = student_labels[valid]
                proxy_static = static_labels[valid]
                labeler_f1 = macro_f1(window.labels[valid], produced[valid])
                proxy_f1 = macro_f1(labeler_labels, proxy_student)
                static_proxy_f1 = macro_f1(labeler_labels, proxy_static)
                requested = self.trigger.update(proxy_f1)
                train_x, train_y = form_training_set(
                    training_features, labeler_labels, balance_binary=balanced,
                )
                selected_indices = torch.arange(stream_features.shape[0], device=self.device)
                labeling_metrics.update({
                    "samples_sent_to_teacher": int(stream_features.shape[0]),
                    "teacher_known_samples": int(valid.sum().item()),
                    "teacher_unknown_samples": int((~valid).sum().item()),
                    "teacher_coverage": float(valid.float().mean().item()),
                    "accepted_label_count": int(valid.sum().item()),
                    "labeling_time": labeling_time,
                })
                trusted_y = labeler_labels
                trusted_indices = (
                    selected_indices[valid]
                    if self.labeler_type == "device_list"
                    else selected_indices
                )
                if requested and train_x is not None:
                    for parameter in self.model.parameters():
                        parameter.requires_grad_(True)
                    training_time, self.optimizer = retrain_model(
                        self.model, train_x, train_y,
                        epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                        learning_rate=float(training["learning_rate"]),
                        optimizer_name=training.get("optimizer", "Adam"), optimizer=self.optimizer,
                    )
                    self.model_version += 1
                    total = sum(parameter.numel() for parameter in self.model.parameters())
                    adaptation = {
                        "adaptation_event": True,
                        "training_sample_count": int(train_y.numel()),
                        "adaptation_mode": "full",
                        "selected_modules": list(self.model.named_hidden_modules()),
                        "total_parameters": total,
                        "trainable_parameters": total,
                        "parameter_update_percentage": 100.0,
                        "updated_parameters": total,
                        "updated_parameter_percentage": 100.0,
                        "estimated_parameter_update_bytes": (total * self.parameter_bits + 7) // 8,
                        "training_time": training_time,
                        "candidate_validation_time": 0.0,
                        "active_validation_f1": None,
                        "candidate_validation_f1": None,
                        "candidate_accepted": True,
                    }
            elif self.policy_mode == "drift_conditioned":
                drift_active = bool(feature_drift["feature_drift_detected"])
                conditioned_metrics["f1_after_drift"] = student_f1 if drift_active else None
                raw_feature_shift = any(
                    score is not None and score > self.drift_detector.threshold
                    for score in feature_drift["feature_drift_scores"].values()
                )
                if not raw_feature_shift:
                    self.last_non_drift_f1 = student_f1
                if not drift_active:
                    self.drift_event_handled = False
                elif not previous_feature_drift:
                    self.drift_event_handled = False

                if drift_active and not self.drift_event_handled:
                    adaptation_started = time.perf_counter()
                    plan, prediction = predict_repair(
                        self.adapter_bundle,
                        drift_signature["drift_signature_vector"],
                        module_threshold=self.adapter_module_threshold,
                        parameter_bits=self.parameter_bits,
                    )
                    conditioned_metrics.update(prediction)
                    conditioned_metrics["adapter_prediction_attempted"] = True
                    validation_modules = list(prediction["predicted_modules"])
                    confidence_ok = (
                        prediction["adapter_confidence"]
                        >= self.adapter_confidence_threshold
                    )
                    conditioned_metrics["adapter_confidence_passed"] = confidence_ok
                    repaired_f1 = student_f1
                    repair_satisfied = False
                    if confidence_ok and plan.affected_modules:
                        candidate, repair_metrics = apply_repair_plan(
                            self.model, plan, parameter_bits=self.parameter_bits,
                            anchor_model=self.repair_base_model,
                        )
                        if all(torch.isfinite(parameter).all() for parameter in candidate.parameters()):
                            repair_satisfied = True
                            conditioned_metrics.update(repair_metrics)
                            if repair_metrics["repair_changed_parameters"] > 0:
                                self.model = candidate.eval()
                                self.model_version += 1
                                total = sum(
                                    parameter.numel() for parameter in self.model.parameters()
                                )
                                adaptation = {
                                    "adaptation_event": True,
                                    "training_sample_count": 0,
                                    "adaptation_mode": "drift_conditioned",
                                    "selected_modules": plan.affected_modules,
                                    "total_parameters": total,
                                    "trainable_parameters": repair_metrics[
                                        "repair_changed_parameters"
                                    ],
                                    "parameter_update_percentage": repair_metrics[
                                        "repair_changed_parameter_percentage"
                                    ],
                                    "updated_parameters": repair_metrics[
                                        "repair_changed_parameters"
                                    ],
                                    "updated_parameter_percentage": repair_metrics[
                                        "repair_changed_parameter_percentage"
                                    ],
                                    "estimated_parameter_update_bytes": repair_metrics[
                                        "repair_update_bytes"
                                    ],
                                    "training_time": 0.0,
                                    "candidate_validation_time": 0.0,
                                    "active_validation_f1": None,
                                    "candidate_validation_f1": None,
                                    "candidate_accepted": True,
                                }
                    prediction_latency = time.perf_counter() - adaptation_started
                    conditioned_metrics["adapter_prediction_latency"] = prediction_latency
                    if adaptation["adaptation_event"]:
                        conditioned_metrics["adaptation_latency"] = prediction_latency
                        repaired_f1 = macro_f1(
                            window.labels, self._predict(self.model, stream_features),
                        )
                    conditioned_metrics["f1_after_predicted_repair"] = repaired_f1
                    self.drift_event_handled = repair_satisfied

            injection_validation = self.drift_injector.validate(
                index,
                injected_drift,
                feature_drift["drifted_feature_indices"],
                validation_modules,
            )

            event = bool(adaptation["adaptation_event"])
            defaults = {
                "training_sample_count": 0,
                "adaptation_mode": None,
                "selected_modules": [],
                "total_parameters": None,
                "trainable_parameters": None,
                "parameter_update_percentage": None,
                "updated_parameters": 0,
                "updated_parameter_percentage": 0.0,
                "estimated_parameter_update_bytes": 0,
                "training_time": 0.0,
                "candidate_validation_time": 0.0,
                "active_validation_f1": None,
                "candidate_validation_f1": None,
                "candidate_accepted": None,
            }
            defaults.update(adaptation)
            adaptation = defaults
            after_f1 = (
                macro_f1(window.labels, self._predict(self.model, stream_features))
                if event else None
            )
            adaptation_labeling_time = (
                labeling_metrics["labeling_time"] if self.policy_mode == "caravan" else 0.0
            ) if event else 0.0
            total_recovery_time = (
                conditioned_metrics["adaptation_latency"]
                if self.policy_mode == "drift_conditioned" and event else
                adaptation_labeling_time + adaptation["training_time"]
                + adaptation["candidate_validation_time"]
                if event else 0.0
            )
            conditioned_metrics["adaptation_latency"] = total_recovery_time

            row = {
                "window": index,
                "window_samples": int(stream_features.shape[0]),
                "method": self.policy_mode,
                "student_f1": student_f1,
                "static_f1": static_f1,
                "labeler_f1": labeler_f1,
                "proxy_f1": proxy_f1,
                "static_proxy_f1": static_proxy_f1,
                "adaptation_policy": self.policy_mode,
                "adaptation_event": event,
                "retrained": event and self.policy_mode == "caravan",
                "training_samples": adaptation["training_sample_count"],
                "student_f1_before_adaptation": student_f1 if event else None,
                "student_f1_after_adaptation": after_f1,
                "f1_before_adaptation": student_f1 if event else None,
                "f1_after_adaptation": after_f1,
                "samples_selected": int(selected_indices.numel()),
                "selected_sample_indices": selected_indices.detach().cpu().tolist(),
                "trusted_sample_indices": trusted_indices.detach().cpu().tolist(),
                "trusted_labels": trusted_y.detach().cpu().tolist(),
                "adaptation_labeling_time": adaptation_labeling_time,
                "total_adaptation_recovery_time": total_recovery_time,
                "model_version": self.model_version,
                **adaptation,
                **labeling_metrics,
                **feature_drift,
                **drift_signature,
                **conditioned_metrics,
                **injected_drift,
                **injection_validation,
            }
            row.update({
                "teacher_samples": row["samples_sent_to_teacher"],
                "retraining_time": row["training_time"],
                "parameters_updated": row["updated_parameters"],
                "parameter_update_percent": row["updated_parameter_percentage"],
                "model_update_bytes": row["estimated_parameter_update_bytes"],
            })
            rows.append(row)
            previous_feature_drift = bool(feature_drift["feature_drift_detected"])

        summary = self._save(rows)
        return rows, summary

    def _save(self, rows: list[dict]) -> dict:
        dataset_name = self.config["dataset"]["name"]
        output = Path(self.config.get("output", {}).get("directory", "results"))
        if not output.is_absolute():
            output = (self.base_dir.parent / output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f"{dataset_name}.csv", index=False)
        events = [row for row in rows if row["adaptation_event"]]
        retraining_events = [row for row in events if row["retrained"]]
        drift_events = sum(
            bool(row["feature_drift_detected"])
            and (index == 0 or not rows[index - 1]["feature_drift_detected"])
            for index, row in enumerate(rows)
        )
        drift_start = next(
            (index for index, row in enumerate(rows) if row["injected_drift_active"]),
            next((index for index, row in enumerate(rows) if row["feature_drift_detected"]), len(rows)),
        )
        average_f1 = mean([row["student_f1"] for row in rows])
        post_drift_f1 = mean([row["student_f1"] for row in rows[drift_start:]])
        total_teacher_samples = sum(row["teacher_samples"] for row in rows)
        total_labeling_time = sum(float(row["labeling_time"]) for row in rows)
        total_retraining_time = sum(float(row["retraining_time"]) for row in rows)
        total_recovery_time = sum(
            float(row["total_adaptation_recovery_time"]) for row in rows
        )
        average_adaptation_latency = (
            mean([float(row["adaptation_latency"]) for row in events])
            if events else 0.0
        )
        average_update_percentage = (
            mean([float(row["parameter_update_percent"]) for row in events])
            if events else 0.0
        )
        summary = {
            "dataset": dataset_name,
            "method": self.policy_mode,
            "adaptation_policy": self.policy_mode,
            "number_of_windows": len(rows),
            "number_of_retraining_events": len(retraining_events),
            "number_of_drift_events": drift_events,
            "number_of_adaptation_events": len(events),
            "average_f1": average_f1,
            "post_drift_f1": post_drift_f1,
            "total_teacher_samples": total_teacher_samples,
            "total_labeling_time": total_labeling_time,
            "total_retraining_time": total_retraining_time,
            "total_adaptation_recovery_time": total_recovery_time,
            "average_adaptation_latency": average_adaptation_latency,
            "total_updated_parameters": sum(row["parameters_updated"] for row in events),
            "average_parameter_update_percentage": average_update_percentage,
            "total_model_update_bytes": sum(row["model_update_bytes"] for row in events),
            # Compatibility aliases retained for existing result consumers.
            "average_student_f1": average_f1,
            "post_drift_student_f1": post_drift_f1,
            "average_static_f1": mean([row["static_f1"] for row in rows]),
            "average_labeler_f1": mean([row["labeler_f1"] for row in rows]),
            "average_retraining_time": (
                mean([float(row["retraining_time"]) for row in retraining_events])
                if retraining_events else 0.0
            ),
            "number_of_feature_drift_windows": sum(
                bool(row["feature_drift_detected"]) for row in rows
            ),
            "histogram_counter_count": self.drift_detector.histogram_counter_count,
            "histogram_counter_bits": COUNTER_BITS,
            "histogram_state_bytes": self.drift_detector.state_bytes,
            "injected_drift_events": self.drift_injector.event_summaries(),
            "average_adapter_confidence": mean([
                row["adapter_confidence"] for row in rows
                if row["adapter_confidence"] is not None
            ]),
            "total_drift_conditioned_adaptation_latency": (
                total_recovery_time if self.policy_mode == "drift_conditioned" else 0.0
            ),
        }
        with (output / f"{dataset_name}.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "windows": rows}, handle, indent=2, allow_nan=True)
        return summary
