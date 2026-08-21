from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .data import StreamingDataset, resolve_path
from .drift import COUNTER_BITS, FeatureHistogramDriftDetector
from .drift_injector import TrafficDriftInjector
from .impact import map_drift_to_neural_impact
from .labelers import ABSTAIN, HierarchicalLabeler, build_labeler
from .metrics import macro_f1, mean
from .models import build_model, load_checkpoint
from .selector import select_drift_associated_samples
from .trainer import ReplayBuffer, form_training_set, retrain_model, selective_retrain_model
from .trigger import RetrainingTrigger


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

        labeler_config = copy.deepcopy(config["labeler"])
        labeler_config.setdefault("feature_names", dataset_config["feature_names"])
        if labeler_config.get("checkpoint"):
            labeler_config["checkpoint"] = str(resolve_path(labeler_config["checkpoint"], base))
        self.labeler = build_labeler(labeler_config, self.device)
        self.trusted_labeler = HierarchicalLabeler(
            self.labeler, config.get("labeling", {}), int(dataset_config["num_classes"]),
        )
        self.optimizer = None
        adaptation = config.get("adaptation", {})
        self.impact_threshold = float(adaptation.get("impact_threshold", 0.30))
        if not 0.0 <= self.impact_threshold <= 1.0:
            raise ValueError("adaptation.impact_threshold must be between 0 and 1")
        policy = config.get("driftadapt", {})
        self.policy_mode = str(policy.get("mode", "baseline_caravan")).lower()
        if self.policy_mode not in {
            "baseline_caravan", "driftadapt_full", "driftadapt_selective",
        }:
            raise ValueError("driftadapt.mode must be baseline_caravan, driftadapt_full, or driftadapt_selective")
        self.min_labeled_samples = int(policy.get("min_labeled_samples", 64))
        self.full_retrain_threshold = float(policy.get("full_retrain_threshold", 0.70))
        self.parameter_bits = int(policy.get("parameter_bits", 32))
        if self.min_labeled_samples < 1 or self.parameter_bits < 1:
            raise ValueError("DriftAdapt sample and parameter precision settings must be positive")
        if not 0.0 <= self.full_retrain_threshold <= 1.0:
            raise ValueError("driftadapt.full_retrain_threshold must be between 0 and 1")

        selection = config.get("sample_selection", {})
        self.sample_selection_enabled = bool(selection.get("enabled", True))
        self.max_selected_samples = int(selection.get("max_samples", self.min_labeled_samples))
        replay = config.get("replay", {})
        self.replay_enabled = bool(replay.get("enabled", False))
        self.replay_samples_per_update = int(replay.get("samples_per_update", 0))
        replay_capacity = int(replay.get("capacity", 1000))
        if self.replay_enabled and replay_capacity < 1:
            raise ValueError("replay.capacity must be positive when replay is enabled")
        self.replay = ReplayBuffer(
            max(replay_capacity, 1), len(dataset_config["feature_names"]), self.device,
        )
        validation = config.get("validation", {})
        self.validation_enabled = bool(validation.get("enabled", True))
        self.max_validation_f1_drop = float(validation.get("max_f1_drop", 0.02))
        if self.max_selected_samples < 1 or self.replay_samples_per_update < 0:
            raise ValueError("Sample-selection and replay sample counts must be valid")
        if not 0.0 <= self.max_validation_f1_drop <= 1.0:
            raise ValueError("validation.max_f1_drop must be between 0 and 1")
        self.model_version = 0
        self.previous_model = copy.deepcopy(self.model).eval()
        self.pending_features = self.dataset.features[:0]
        self.pending_labels = self.dataset.labels[:0]
        self.pending_capacity = max(self.min_labeled_samples * 4, self.max_selected_samples)
        self.pending_labeling_time = 0.0
        self.drift_event_handled = False
        trigger_config = config["trigger"]
        if trigger_config.get("type") != "accuracy_proxy":
            raise ValueError("This drift-only implementation requires trigger.type: accuracy_proxy")
        self.trigger = RetrainingTrigger(
            threshold=float(trigger_config["threshold"]),
            drop_threshold=float(trigger_config.get("drop_threshold", 0.15)),
            consecutive_windows=int(trigger_config.get("consecutive_windows", 2)),
        )
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

    def _append_pending(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.numel() == 0:
            return
        self.pending_features = torch.cat((self.pending_features, features))[-self.pending_capacity:]
        self.pending_labels = torch.cat((self.pending_labels, labels))[-self.pending_capacity:]

    def _candidate_adaptation(
        self,
        selected_modules: list[str],
        training: dict,
        balance_binary: bool,
    ) -> dict:
        replay_x, replay_y = self.replay.sample(
            self.replay_samples_per_update if self.replay_enabled else 0
        )
        combined_x = torch.cat((self.pending_features, replay_x))
        combined_y = torch.cat((self.pending_labels, replay_y))
        validation_count = (
            min(max(1, combined_y.numel() // 5), 64)
            if self.validation_enabled and combined_y.numel() >= 5 else 0
        )
        if validation_count:
            train_x, validation_x = combined_x[:-validation_count], combined_x[-validation_count:]
            train_y, validation_y = combined_y[:-validation_count], combined_y[-validation_count:]
        else:
            train_x, train_y = combined_x, combined_y
            validation_x, validation_y = combined_x[:0], combined_y[:0]
        train_x, train_y = form_training_set(train_x, train_y, balance_binary=balance_binary)
        if train_x is None:
            return {"adaptation_event": False}

        candidate = copy.deepcopy(self.model)
        hidden_count = max(len(candidate.named_hidden_modules()), 1)
        impacted_parameters = sum(
            parameter.numel()
            for name in selected_modules
            for parameter in candidate.named_hidden_modules()[name].parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in candidate.parameters())
        impacted_fraction = max(
            impacted_parameters / total_parameters,
            len(selected_modules) / hidden_count,
        )
        actual_mode = "full" if (
            self.policy_mode == "driftadapt_full"
            or impacted_fraction >= self.full_retrain_threshold
        ) else "selective"

        if actual_mode == "full":
            for parameter in candidate.parameters():
                parameter.requires_grad_(True)
            training_time, _ = retrain_model(
                candidate, train_x, train_y,
                epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                optimizer_name=training.get("optimizer", "Adam"), optimizer=None,
            )
            adapted_modules = list(candidate.named_hidden_modules())
        else:
            training_time, _ = selective_retrain_model(
                candidate, train_x, train_y, selected_modules,
                epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                optimizer_name=training.get("optimizer", "Adam"),
            )
            adapted_modules = selected_modules

        updated_parameters = sum(
            parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
        )
        validation_started = time.perf_counter()
        active_validation_f1 = None
        candidate_validation_f1 = None
        accepted = True
        if self.validation_enabled and validation_y.numel() > 0:
            active_validation_f1 = macro_f1(
                validation_y, self._predict(self.model, validation_x),
            )
            candidate_validation_f1 = macro_f1(
                validation_y, self._predict(candidate, validation_x),
            )
            accepted = (
                candidate_validation_f1 + self.max_validation_f1_drop
                >= active_validation_f1
            )
        validation_time = time.perf_counter() - validation_started
        if accepted:
            self.previous_model = copy.deepcopy(self.model).eval()
            self.model = candidate.eval()
            self.model_version += 1

        return {
            "adaptation_event": True,
            "training_sample_count": int(train_y.numel()),
            "adaptation_mode": actual_mode,
            "selected_modules": adapted_modules,
            "total_parameters": total_parameters,
            "trainable_parameters": updated_parameters,
            "parameter_update_percentage": 100.0 * updated_parameters / total_parameters,
            "updated_parameters": updated_parameters,
            "updated_parameter_percentage": 100.0 * updated_parameters / total_parameters,
            "estimated_parameter_update_bytes": (
                updated_parameters * self.parameter_bits + 7
            ) // 8,
            "training_time": training_time,
            "candidate_validation_time": validation_time,
            "active_validation_f1": active_validation_f1,
            "candidate_validation_f1": candidate_validation_f1,
            "candidate_accepted": accepted,
        }

    def run(self) -> tuple[list[dict], dict]:
        rows: list[dict] = []
        stream = self.config["stream"]
        training = self.config["training"]
        labeler_type = self.config["labeler"]["type"].lower()
        previous_feature_drift = False

        for index, window in enumerate(self.dataset.windows(int(stream["window_size"])), start=1):
            stream_features, injected_drift = self.drift_injector.apply(window.features, index)
            feature_drift = self.drift_detector.update(stream_features)
            neural_impact = map_drift_to_neural_impact(
                self.model,
                feature_drift["feature_drift_scores"],
                self.impact_threshold,
            )
            injection_validation = self.drift_injector.validate(
                index,
                injected_drift,
                feature_drift["drifted_feature_indices"],
                neural_impact["impact_selected_modules"],
            )
            student_labels = self._predict(self.model, stream_features)
            static_labels = self._predict(self.static_model, stream_features)
            balanced = bool(training.get("balance_binary", self.config["dataset"]["num_classes"] == 2))
            labeling_metrics = self._empty_labeling_metrics()
            selected_indices = torch.empty(0, dtype=torch.long, device=self.device)
            proxy_f1 = float("nan")
            static_proxy_f1 = float("nan")
            labeler_f1 = float("nan")
            trusted_x = stream_features[:0]
            trusted_y = window.labels[:0]
            trusted_indices = selected_indices
            adaptation = {"adaptation_event": False}

            if self.policy_mode == "baseline_caravan":
                labeling_started = time.perf_counter()
                produced = self.labeler.label(
                    stream_features, window.raw_features, window.generated_labels,
                )
                labeling_time = time.perf_counter() - labeling_started
                valid = produced != ABSTAIN if labeler_type == "device_list" else torch.ones_like(
                    produced, dtype=torch.bool,
                )
                labeler_labels = window.labels[valid] if labeler_type == "device_list" else produced
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
                trusted_x = training_features
                trusted_y = labeler_labels
                trusted_indices = selected_indices[valid] if labeler_type == "device_list" else selected_indices
                if requested and train_x is not None:
                    self.previous_model = copy.deepcopy(self.model).eval()
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
            else:
                drift_active = bool(feature_drift["feature_drift_detected"])
                if drift_active and not previous_feature_drift:
                    self.pending_features = stream_features[:0]
                    self.pending_labels = window.labels[:0]
                    self.pending_labeling_time = 0.0
                    self.drift_event_handled = False
                if not drift_active:
                    if previous_feature_drift and self.replay_enabled:
                        self.replay.add(self.pending_features, self.pending_labels)
                    self.pending_features = stream_features[:0]
                    self.pending_labels = window.labels[:0]
                    self.pending_labeling_time = 0.0
                    self.drift_event_handled = False
                if drift_active and not self.drift_event_handled:
                    selected_indices = select_drift_associated_samples(
                        stream_features,
                        feature_drift["feature_drift_scores"],
                        feature_drift["drifted_feature_indices"],
                        self.drift_detector.range_min,
                        self.drift_detector.range_max,
                        enabled=self.sample_selection_enabled,
                        max_samples=self.max_selected_samples,
                    )
                selected_x = stream_features[selected_indices]
                selected_raw = window.raw_features[selected_indices]
                selected_generated = window.generated_labels[selected_indices]
                selected_oracle = window.labels[selected_indices]
                trusted_labels, labeling_metrics = self.trusted_labeler.label(
                    selected_x, selected_raw, selected_generated,
                )
                trusted_labels, labeling_metrics = self.trusted_labeler.apply_fallback(
                    trusted_labels, selected_oracle, labeling_metrics,
                )
                accepted = trusted_labels != ABSTAIN
                trusted_x = selected_x[accepted]
                trusted_y = trusted_labels[accepted]
                accepted_indices = selected_indices[accepted]
                trusted_indices = accepted_indices
                if trusted_y.numel():
                    labeler_f1 = macro_f1(window.labels[accepted_indices], trusted_y)
                    proxy_f1 = macro_f1(trusted_y, student_labels[accepted_indices])
                    static_proxy_f1 = macro_f1(trusted_y, static_labels[accepted_indices])
                self._append_pending(trusted_x, trusted_y)
                self.pending_labeling_time += labeling_metrics["labeling_time"]
                impacted_modules = neural_impact["impact_selected_modules"]
                if (
                    drift_active
                    and not self.drift_event_handled
                    and impacted_modules
                    and self.pending_labels.numel() >= self.min_labeled_samples
                ):
                    adaptation = self._candidate_adaptation(
                        impacted_modules, training, balanced,
                    )
                    if adaptation["adaptation_event"]:
                        self.drift_event_handled = True
                        if self.replay_enabled:
                            self.replay.add(self.pending_features, self.pending_labels)

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
                labeling_metrics["labeling_time"]
                if self.policy_mode == "baseline_caravan"
                else self.pending_labeling_time
            ) if event else 0.0
            total_recovery_time = (
                adaptation_labeling_time
                + adaptation["training_time"]
                + adaptation["candidate_validation_time"]
                if event else 0.0
            )

            row = {
                "window": index,
                "window_samples": int(stream_features.shape[0]),
                "student_f1": macro_f1(window.labels, student_labels),
                "static_f1": macro_f1(window.labels, static_labels),
                "labeler_f1": labeler_f1,
                "proxy_f1": proxy_f1,
                "static_proxy_f1": static_proxy_f1,
                "adaptation_policy": self.policy_mode,
                "adaptation_event": event,
                "retrained": event,
                "training_samples": adaptation["training_sample_count"],
                "student_f1_before_adaptation": macro_f1(window.labels, student_labels) if event else None,
                "student_f1_after_adaptation": after_f1,
                "samples_selected": int(selected_indices.numel()),
                "selected_sample_indices": selected_indices.detach().cpu().tolist(),
                "trusted_sample_indices": trusted_indices.detach().cpu().tolist(),
                "trusted_labels": trusted_y.detach().cpu().tolist(),
                "adaptation_labeling_time": adaptation_labeling_time,
                "total_adaptation_recovery_time": total_recovery_time,
                "model_version": self.model_version,
                "replay_buffer_size": len(self.replay) if self.replay_enabled else 0,
                **adaptation,
                **labeling_metrics,
                **feature_drift,
                **neural_impact,
                **injected_drift,
                **injection_validation,
            }
            rows.append(row)
            if event and self.policy_mode != "baseline_caravan":
                self.pending_labeling_time = 0.0
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
        accepted = [row for row in events if row["candidate_accepted"]]
        times = [float(row["training_time"]) for row in events]
        drift_events = sum(
            bool(row["feature_drift_detected"])
            and (index == 0 or not rows[index - 1]["feature_drift_detected"])
            for index, row in enumerate(rows)
        )
        drift_start = next(
            (index for index, row in enumerate(rows) if row["injected_drift_active"]),
            next((index for index, row in enumerate(rows) if row["feature_drift_detected"]), len(rows)),
        )
        drift_window_samples = sum(
            row["window_samples"] for row in rows if row["feature_drift_detected"]
        )
        selected_drift_samples = sum(
            row["samples_sent_to_teacher"] for row in rows if row["feature_drift_detected"]
        )
        summary = {
            "dataset": dataset_name,
            "adaptation_policy": self.policy_mode,
            "number_of_windows": len(rows),
            "number_of_retraining_events": len(events),
            "number_of_drift_events": drift_events,
            "number_of_adaptation_events": len(accepted),
            "average_student_f1": mean([row["student_f1"] for row in rows]),
            "post_drift_student_f1": mean([row["student_f1"] for row in rows[drift_start:]]),
            "average_static_f1": mean([row["static_f1"] for row in rows]),
            "average_labeler_f1": mean([row["labeler_f1"] for row in rows]),
            "total_retraining_time": sum(times),
            "average_retraining_time": mean(times),
            "total_updated_parameters": sum(row["updated_parameters"] for row in accepted),
            "total_model_update_bytes": sum(
                row["estimated_parameter_update_bytes"] for row in accepted
            ),
            "average_parameter_update_percentage": mean([
                row["updated_parameter_percentage"] for row in accepted
            ]),
            "labeling_reduction": (
                1.0 - selected_drift_samples / drift_window_samples
                if drift_window_samples else 0.0
            ),
            "candidate_rejection_count": sum(
                row["candidate_accepted"] is False for row in events
            ),
            "replay_buffer_size": len(self.replay) if self.replay_enabled else 0,
            "number_of_feature_drift_windows": sum(
                bool(row["feature_drift_detected"]) for row in rows
            ),
            "histogram_counter_count": self.drift_detector.histogram_counter_count,
            "histogram_counter_bits": COUNTER_BITS,
            "histogram_state_bytes": self.drift_detector.state_bytes,
            "impact_state_elements": rows[0]["impact_state_elements"] if rows else 0,
            "impact_multiply_accumulate_count": (
                rows[0]["impact_multiply_accumulate_count"] if rows else 0
            ),
            "injected_drift_events": self.drift_injector.event_summaries(),
        }
        with (output / f"{dataset_name}.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "windows": rows}, handle, indent=2, allow_nan=True)
        return summary
