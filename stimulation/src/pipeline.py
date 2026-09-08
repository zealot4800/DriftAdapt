from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .adapter import load_adapter_bundle, predict_repair, predict_staged_repair
from .capacity import (
    capacity_pressure_decision,
    capacity_ratios,
    make_capacity_event,
    neuron_saturation,
    residual_recovery_status,
    resolve_capacity_config,
)
from .data import StreamingDataset, resolve_path
from .drift import COUNTER_BITS, FeatureHistogramDriftDetector
from .drift_dataset import resolve_feature_reconfiguration_config
from .drift_injector import TrafficDriftInjector
from .drift_signature import build_drift_signature
from .labelers import ABSTAIN, build_labeler
from .impact import map_drift_to_neural_impact
from .metrics import macro_f1, mean
from .models import ElasticTwoHiddenLayerModel, build_model, load_checkpoint
from .repair import apply_feature_mask, apply_repair_plan
from .selector import select_drift_associated_samples
from .trainer import (
    capacity_selective_retrain_model,
    form_training_set,
    retrain_model,
)
from .trigger import RetrainingTrigger


SUPPORTED_POLICIES = {"caravan", "drift_conditioned"}
DEPRECATED_POLICY_ALIASES = {"baseline_caravan": "caravan"}
CAPACITY_MAX_SELECTED_SAMPLES = 64


class DriftAdaptPipeline:
    def __init__(self, config: dict, config_path: Path) -> None:
        self.config = config
        runtime = config.get("runtime", {})
        requested = runtime.get("device", "cuda")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.seed = int(runtime.get("seed", 42))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False

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

        self.capacity_config = resolve_capacity_config(config.get("capacity"))
        model_config = config["model"]
        use_elastic_capacity_model = bool(
            self.policy_mode == "drift_conditioned"
            and self.capacity_config["enabled"]
        )
        runtime_model = (
            model_config.get("simulation_class", model_config["class"])
            if use_elastic_capacity_model
            else model_config["class"]
        )
        self.model = build_model(runtime_model, model_config.get("kwargs")).to(self.device)
        checkpoint = resolve_path(model_config["checkpoint"], base)
        load_checkpoint(self.model, str(checkpoint), self.device)
        self.model.eval()
        self.static_model = copy.deepcopy(self.model).eval()
        self.repair_base_model = copy.deepcopy(self.model).eval()

        self.parameter_bits = int(policy.get("parameter_bits", 32))
        if self.parameter_bits < 1:
            raise ValueError("driftadapt.parameter_bits must be positive")
        if (
            self.capacity_config["enabled"]
            and self.policy_mode == "drift_conditioned"
        ):
            if not isinstance(self.model, ElasticTwoHiddenLayerModel):
                raise ValueError(
                    "Software capacity reconfiguration requires an elastic "
                    "two-hidden-layer simulation model"
                )
            expected = {
                "initial_layer1": self.model.min_layer1_neurons,
                "initial_layer2": self.model.min_layer2_neurons,
                "max_layer1": self.model.max_layer1_neurons,
                "max_layer2": self.model.max_layer2_neurons,
            }
            mismatched = [
                name for name, value in expected.items()
                if self.capacity_config[name] != value
            ]
            if mismatched:
                raise ValueError(
                    "capacity initial/max widths must match the elastic model: "
                    + ", ".join(mismatched)
                )
            if self.model.capacity_state() != (
                self.capacity_config["initial_layer1"],
                self.capacity_config["initial_layer2"],
            ):
                raise ValueError("Elastic checkpoint does not start at configured capacity")
        self.capacity_enabled = bool(
            self.capacity_config["enabled"]
            and self.policy_mode == "drift_conditioned"
        )
        self.capacity_label_source = (
            "ground_truth_simulation_oracle" if self.capacity_enabled else None
        )
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
        self.default_feature_mask: list[int] | None = None
        self.active_feature_mask: list[int] | None = None
        self.feature_reference_values: torch.Tensor | None = None
        self.feature_reconfiguration_config: dict | None = None
        self.adapter_confidence_threshold = float(conditioned.get("confidence_threshold", 0.50))
        self.adapter_module_threshold = float(conditioned.get("module_threshold", 0.50))
        self.signature_shift_clip = float(conditioned.get("signature_shift_clip", 8.0))
        self.capacity_impact_threshold = float(
            config.get("adaptation", {}).get("impact_threshold", 0.30)
        )
        if not 0.0 <= self.adapter_confidence_threshold <= 1.0:
            raise ValueError("drift_conditioned.confidence_threshold must be between zero and one")
        if not 0.0 <= self.adapter_module_threshold <= 1.0:
            raise ValueError("drift_conditioned.module_threshold must be between zero and one")
        if self.signature_shift_clip <= 0:
            raise ValueError("drift_conditioned.signature_shift_clip must be positive")
        if not 0.0 <= self.capacity_impact_threshold <= 1.0:
            raise ValueError("adaptation.impact_threshold must be between zero and one")
        if self.policy_mode == "drift_conditioned":
            self.feature_reconfiguration_config = resolve_feature_reconfiguration_config(
                config.get("feature_reconfiguration"),
                len(dataset_config["feature_names"]),
            )
            checkpoint_value = (
                conditioned.get(
                    "simulation_adapter_checkpoint",
                    conditioned.get("adapter_checkpoint"),
                )
                if self.capacity_enabled else conditioned.get("adapter_checkpoint")
            )
            if not checkpoint_value:
                raise ValueError(
                    "drift_conditioned.adapter_checkpoint is required for drift_conditioned mode"
                )
            self.adapter_bundle = load_adapter_bundle(
                resolve_path(checkpoint_value, base), self.model, self.device,
                feature_names=list(dataset_config["feature_names"]),
                signature_config={
                    "bins": int(drift_config.get("bins", 16)),
                    "threshold": float(drift_config.get("threshold", 0.10)),
                    "consecutive_windows": int(
                        drift_config.get("consecutive_windows", 3)
                    ),
                    "reference_windows": int(drift_config.get("reference_windows", 10)),
                    "normalized_shift_clip": self.signature_shift_clip,
                },
                feature_reconfiguration_config=self.feature_reconfiguration_config,
                confidence_threshold=self.adapter_confidence_threshold,
                module_threshold=self.adapter_module_threshold,
            )
            self.default_feature_mask = list(self.adapter_bundle.default_feature_mask)
            self.active_feature_mask = list(self.default_feature_mask)
            self.feature_reference_values = self.adapter_bundle.feature_reference_values.to(
                self.device
            )
        self.model_version = 0
        self.drift_event_handled = False
        self.drift_event_repair_attempted = False
        self.drift_event_pre_repair_model = None
        self.drift_event_pre_repair_feature_mask: list[int] | None = None
        self.drift_event_pending_feature_mask: list[int] | None = None
        self.drift_event_feature_stage_status = "not_started"
        self.drift_event_stage_order: list[str] = []
        self.drift_event_weight_repair_f1: float | None = None
        self.drift_event_feature_reconfiguration_f1: float | None = None
        self.last_non_drift_f1: float | None = None
        self.capacity_low_utility_streaks: dict[str, torch.Tensor] = {}
        if self.capacity_enabled:
            self.capacity_low_utility_streaks = {
                name: torch.zeros(
                    module.out_features, dtype=torch.long, device=self.device,
                )
                for name, module in self.model.named_hidden_modules().items()
            }
        self.base_dir = base

    @torch.inference_mode()
    def _predict(self, model, features):
        return model.run_inference(features)[1]

    def _capacity_metrics(self) -> dict:
        if hasattr(self.model, "capacity_metrics"):
            return self.model.capacity_metrics()
        hidden = list(self.model.named_hidden_modules().values())
        if len(hidden) < 2:
            return {
                "active_layer1": None,
                "active_layer2": None,
                "layer1_capacity_ratio": None,
                "layer2_capacity_ratio": None,
                "overall_capacity_ratio": None,
            }
        return capacity_ratios(
            hidden[0].out_features, hidden[1].out_features,
            hidden[0].out_features, hidden[1].out_features,
        )

    def _reset_capacity_streaks(self) -> None:
        for streaks in self.capacity_low_utility_streaks.values():
            streaks.zero_()

    def _reset_drift_event_state(self) -> None:
        self.drift_event_handled = False
        self.drift_event_repair_attempted = False
        self.drift_event_pre_repair_model = None
        self.drift_event_pre_repair_feature_mask = None
        self.drift_event_pending_feature_mask = None
        self.drift_event_feature_stage_status = "not_started"
        self.drift_event_stage_order = []
        self.drift_event_weight_repair_f1 = None
        self.drift_event_feature_reconfiguration_f1 = None

    def _repair_recovery_status(
        self, reference_f1: float | None, repaired_f1: float,
    ) -> dict[str, float | bool]:
        return residual_recovery_status(
            reference_f1=reference_f1,
            repaired_f1=repaired_f1,
            threshold=self.capacity_config["residual_f1_threshold"],
        )

    def _feature_stage_resolved(self) -> bool:
        return self.drift_event_feature_stage_status in {
            "not_applicable", "applied", "rejected",
        }

    def _apply_pending_feature_reconfiguration(
        self,
        *,
        stream_features: torch.Tensor,
        labels: torch.Tensor,
        reference_f1: float | None,
    ) -> dict:
        """Apply only the pending feature mask after weight recovery failed."""
        if not self.drift_event_repair_attempted:
            raise AssertionError("Feature reconfiguration requires weight repair first")
        if self.drift_event_feature_stage_status != "pending":
            raise AssertionError("Feature reconfiguration is not pending")
        if self.drift_event_pending_feature_mask is None:
            raise AssertionError("Pending feature reconfiguration has no candidate mask")

        old_mask = list(self.active_feature_mask)
        new_mask = list(self.drift_event_pending_feature_mask)
        if new_mask == old_mask:
            raise AssertionError("Feature reconfiguration must change the active mask")
        old_features = apply_feature_mask(
            stream_features, old_mask, self.feature_reference_values,
        )
        new_features = apply_feature_mask(
            stream_features, new_mask, self.feature_reference_values,
        )
        weight_repair_f1 = macro_f1(
            labels, self._predict(self.model, old_features),
        )
        feature_reconfigured_f1 = macro_f1(
            labels, self._predict(self.model, new_features),
        )

        # The feature stage changes inputs only. The already repaired model is
        # deliberately retained so the runtime order cannot collapse back into
        # a joint feature-and-weight repair. A non-improving mask is rolled back
        # before capacity is allowed to run.
        accepted = bool(feature_reconfigured_f1 > weight_repair_f1 + 1e-12)
        final_f1 = feature_reconfigured_f1 if accepted else weight_repair_f1
        final_features = new_features if accepted else old_features
        if accepted:
            self.active_feature_mask = new_mask
            self.model_version += 1
        self.drift_event_pending_feature_mask = None
        self.drift_event_feature_stage_status = (
            "applied" if accepted else "rejected"
        )
        self.drift_event_feature_reconfiguration_f1 = feature_reconfigured_f1
        self.drift_event_stage_order.append("feature_reconfiguration")
        self._reset_capacity_streaks()
        recovery = self._repair_recovery_status(
            reference_f1, final_f1,
        )
        return {
            "feature_reconfiguration_attempted": True,
            "feature_reconfiguration_event": accepted,
            "feature_reconfiguration_accepted": accepted,
            "feature_reconfiguration_status": self.drift_event_feature_stage_status,
            "feature_reconfiguration_weight_repair_f1": weight_repair_f1,
            "feature_reconfiguration_f1": feature_reconfigured_f1,
            "feature_reconfiguration_final_f1": final_f1,
            "feature_reconfiguration_residual_f1_loss": recovery[
                "residual_f1_loss"
            ],
            "feature_reconfiguration_recovery_insufficient": recovery[
                "recovery_insufficient"
            ],
            "features_enabled": [
                self.config["dataset"]["feature_names"][feature_index]
                for feature_index, (before, after) in enumerate(zip(old_mask, new_mask))
                if accepted and not before and after
            ],
            "features_disabled": [
                self.config["dataset"]["feature_names"][feature_index]
                for feature_index, (before, after) in enumerate(zip(old_mask, new_mask))
                if accepted and before and not after
            ],
            "student_features": final_features,
        }

    def _capacity_sample_split(
        self,
        stream_features: torch.Tensor,
        feature_drift: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select drift-associated samples and reserve a deterministic tail."""
        selected = select_drift_associated_samples(
            stream_features,
            feature_drift["feature_drift_scores"],
            feature_drift["drifted_feature_indices"],
            self.drift_detector.range_min,
            self.drift_detector.range_max,
            enabled=True,
            max_samples=CAPACITY_MAX_SELECTED_SAMPLES,
        )
        if selected.numel() < 2:
            return selected, selected[:0], selected[:0]
        validation_count = min(
            max(1, int(selected.numel()) // 5), int(selected.numel()) - 1,
        )
        return (
            selected,
            selected[:-validation_count],
            selected[-validation_count:],
        )

    def _capacity_neural_impact(
        self,
        student_features: torch.Tensor,
        feature_drift: dict,
    ) -> dict:
        feature_scores = dict(feature_drift["feature_drift_scores"])
        if self.active_feature_mask is not None:
            feature_scores = {
                name: score if self.active_feature_mask[index] else 0.0
                for index, (name, score) in enumerate(feature_scores.items())
            }
        activation_magnitudes = self.model.hidden_activation_magnitudes(
            student_features,
        )
        return map_drift_to_neural_impact(
            self.model,
            feature_scores,
            self.capacity_impact_threshold,
            activation_magnitudes=activation_magnitudes,
        )

    def _attempt_capacity_expansion(
        self,
        *,
        window_id: int,
        stream_features: torch.Tensor,
        student_features: torch.Tensor,
        labels: torch.Tensor,
        pre_repair_predictions: torch.Tensor,
        feature_drift: dict,
        reference_f1: float | None,
        repair_attempted: bool,
        feature_stage_completed: bool,
        training: dict,
    ) -> dict:
        """Evaluate pressure, train an elastic candidate, and accept or discard it."""
        if not repair_attempted:
            raise AssertionError("Capacity expansion requires a preceding repair attempt")
        if not feature_stage_completed:
            raise AssertionError(
                "Capacity expansion requires the feature stage to be resolved"
            )
        selected, train_indices, validation_indices = self._capacity_sample_split(
            stream_features, feature_drift,
        )
        current_predictions = self._predict(self.model, student_features)
        evaluation_indices = (
            validation_indices if validation_indices.numel() else
            torch.arange(labels.numel(), device=labels.device)
        )
        pre_repair_f1 = macro_f1(
            labels, pre_repair_predictions,
        )
        repaired_f1 = macro_f1(
            labels, current_predictions,
        )
        validation_repaired_f1 = macro_f1(
            labels[evaluation_indices], current_predictions[evaluation_indices],
        )

        impact = self._capacity_neural_impact(student_features, feature_drift)
        hidden_names = list(self.model.named_hidden_modules())
        if len(hidden_names) != 2:
            raise AssertionError("Elastic CIC capacity requires exactly two hidden layers")
        masks = self.model.hidden_neuron_masks()
        layer1 = neuron_saturation(
            impact["impacted_neuron_scores"][hidden_names[0]],
            masks[hidden_names[0]].detach().cpu().tolist(),
            impact_threshold=self.capacity_impact_threshold,
        )
        layer2 = neuron_saturation(
            impact["impacted_neuron_scores"][hidden_names[1]],
            masks[hidden_names[1]].detach().cpu().tolist(),
            impact_threshold=self.capacity_impact_threshold,
        )
        old_capacity = self.model.capacity_state()
        pressure = capacity_pressure_decision(
            reference_f1=reference_f1,
            repaired_f1=repaired_f1,
            layer1_saturation=float(layer1["saturation"]),
            layer2_saturation=float(layer2["saturation"]),
            active_layer1=old_capacity[0],
            active_layer2=old_capacity[1],
            config=self.capacity_config,
        )
        result = {
            "selected_indices": selected,
            "training_indices": train_indices,
            "validation_indices": validation_indices,
            "pre_repair_f1": pre_repair_f1,
            "post_repair_f1": repaired_f1,
            "post_capacity_f1": None,
            "validation_post_repair_f1": validation_repaired_f1,
            "validation_post_capacity_f1": None,
            "capacity_event": None,
            "capacity_pressure": bool(pressure["capacity_pressure"]),
            "residual_f1_loss": float(pressure["residual_f1_loss"]),
            "layer1_saturation": float(layer1["saturation"]),
            "layer2_saturation": float(layer2["saturation"]),
            "layer1_high_impact_neurons": int(layer1["high_impact_neurons"]),
            "layer2_high_impact_neurons": int(layer2["high_impact_neurons"]),
            "neural_impact": impact,
            "training_time": 0.0,
            "validation_time": 0.0,
            "trainable_parameters": 0,
            "accepted": False,
        }
        if (
            not pressure["capacity_pressure"]
            or not (pressure["expand_layer1"] or pressure["expand_layer2"])
            or train_indices.numel() == 0
            or validation_indices.numel() == 0
        ):
            return result

        candidate = copy.deepcopy(self.model).eval()
        scores1 = impact["impacted_neuron_scores"][hidden_names[0]]
        scores2 = impact["impacted_neuron_scores"][hidden_names[1]]
        activated_layer1 = candidate.activate_neurons(
            1,
            int(pressure["proposed_layer1"]) - old_capacity[0],
            scores1,
            self.capacity_config["initialization_noise"],
        )
        activated_layer2 = candidate.activate_neurons(
            2,
            int(pressure["proposed_layer2"]) - old_capacity[1],
            scores2,
            self.capacity_config["initialization_noise"],
        )
        if not activated_layer1 and not activated_layer2:
            return result

        affected = {
            name: [
                neuron_index
                for neuron_index, (score, active) in enumerate(zip(
                    impact["impacted_neuron_scores"][name],
                    masks[name].detach().cpu().tolist(),
                ))
                if active and score >= self.capacity_impact_threshold
            ]
            for name in hidden_names
        }
        newly_active = {
            hidden_names[0]: activated_layer1,
            hidden_names[1]: activated_layer2,
        }
        training_time, trainable_parameters = capacity_selective_retrain_model(
            candidate,
            student_features[train_indices],
            labels[train_indices],
            candidate.hidden_neuron_masks(),
            affected,
            newly_active,
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            optimizer_name=training.get("optimizer", "Adam"),
            seed=self.seed + window_id,
        )
        validation_started = time.perf_counter()
        validation_capacity_f1 = macro_f1(
            labels[validation_indices],
            self._predict(candidate, student_features[validation_indices]),
        )
        capacity_f1 = macro_f1(
            labels, self._predict(candidate, student_features),
        )
        validation_time = time.perf_counter() - validation_started
        candidate_is_finite = all(
            torch.isfinite(parameter).all() for parameter in candidate.parameters()
        )
        accepted = bool(
            candidate_is_finite
            and validation_capacity_f1 - validation_repaired_f1
            > self.capacity_config["min_improvement"]
        )
        attempted_capacity = candidate.capacity_state()
        if accepted:
            self.model = candidate.eval()
            self.model_version += 1
            self._reset_capacity_streaks()
        final_capacity = self.model.capacity_state()
        event = make_capacity_event(
            window_id=window_id,
            event_type="expand" if accepted else "rejected_expand",
            old_capacity=old_capacity,
            new_capacity=final_capacity,
            reference_f1=reference_f1,
            pre_repair_f1=pre_repair_f1,
            post_repair_f1=repaired_f1,
            post_capacity_f1=capacity_f1,
            residual_f1_loss=float(pressure["residual_f1_loss"]),
            layer1_saturation=float(layer1["saturation"]),
            layer2_saturation=float(layer2["saturation"]),
            activated_layer1_neurons=activated_layer1,
            activated_layer2_neurons=activated_layer2,
            training_samples=int(train_indices.numel()),
            trainable_parameters=trainable_parameters,
            accepted=accepted,
            model_version=self.model_version,
        )
        result.update({
            "post_capacity_f1": capacity_f1,
            "validation_post_capacity_f1": validation_capacity_f1,
            "capacity_event": event,
            "training_time": training_time,
            "validation_time": validation_time,
            "trainable_parameters": trainable_parameters,
            "accepted": accepted,
            "attempted_capacity": attempted_capacity,
        })
        return result

    @staticmethod
    def _capacity_outcome_metrics(outcome: dict) -> dict:
        attempted = outcome["capacity_event"] is not None
        training_indices = outcome["training_indices"]
        return {
            "capacity_pressure": outcome["capacity_pressure"],
            "capacity_event": outcome["capacity_event"],
            "capacity_pre_repair_f1": outcome["pre_repair_f1"],
            "capacity_post_repair_f1": outcome["post_repair_f1"],
            "capacity_reconfigured_f1": outcome["post_capacity_f1"],
            "capacity_validation_post_repair_f1": outcome[
                "validation_post_repair_f1"
            ],
            "capacity_validation_reconfigured_f1": outcome[
                "validation_post_capacity_f1"
            ],
            "residual_f1_loss": outcome["residual_f1_loss"],
            "layer1_saturation": outcome["layer1_saturation"],
            "layer2_saturation": outcome["layer2_saturation"],
            "layer1_high_impact_neurons": outcome[
                "layer1_high_impact_neurons"
            ],
            "layer2_high_impact_neurons": outcome[
                "layer2_high_impact_neurons"
            ],
            "capacity_impacted_neuron_scores": outcome["neural_impact"][
                "impacted_neuron_scores"
            ],
            "capacity_module_impact_scores": outcome["neural_impact"][
                "module_impact_scores"
            ],
            "capacity_training_sample_indices": (
                training_indices.detach().cpu().tolist() if attempted else []
            ),
            "capacity_validation_sample_indices": outcome[
                "validation_indices"
            ].detach().cpu().tolist(),
            "capacity_training_samples": (
                int(training_indices.numel()) if attempted else 0
            ),
            "capacity_trainable_parameters": outcome["trainable_parameters"],
            "capacity_training_time": outcome["training_time"],
            "capacity_validation_time": outcome["validation_time"],
        }

    def _capacity_adaptation_metrics(
        self, outcome: dict, prior_adaptation: dict | None = None,
    ) -> dict:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = outcome["trainable_parameters"]
        result = {
            "adaptation_event": True,
            "training_sample_count": int(outcome["training_indices"].numel()),
            "adaptation_mode": "drift_conditioned_capacity",
            "selected_modules": list(
                outcome["neural_impact"]["impact_selected_modules"]
            ),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "parameter_update_percentage": (
                100.0 * trainable / total if total else 0.0
            ),
            "updated_parameters": trainable,
            "updated_parameter_percentage": (
                100.0 * trainable / total if total else 0.0
            ),
            "estimated_parameter_update_bytes": (
                trainable * self.parameter_bits + 7
            ) // 8,
            "training_time": outcome["training_time"],
            "candidate_validation_time": outcome["validation_time"],
            "active_validation_f1": outcome["validation_post_repair_f1"],
            "candidate_validation_f1": outcome["validation_post_capacity_f1"],
            "candidate_accepted": True,
        }
        if prior_adaptation and prior_adaptation.get("adaptation_event"):
            prior_updates = int(prior_adaptation.get("updated_parameters", 0))
            combined_updates = prior_updates + trainable
            result.update({
                "adaptation_mode": "drift_conditioned_repair_capacity",
                "selected_modules": sorted(set(
                    prior_adaptation.get("selected_modules", [])
                    + result["selected_modules"]
                )),
                "trainable_parameters": combined_updates,
                "parameter_update_percentage": (
                    100.0 * combined_updates / total if total else 0.0
                ),
                "updated_parameters": combined_updates,
                "updated_parameter_percentage": (
                    100.0 * combined_updates / total if total else 0.0
                ),
                "estimated_parameter_update_bytes": int(
                    prior_adaptation.get("estimated_parameter_update_bytes", 0)
                ) + result["estimated_parameter_update_bytes"],
            })
        return result

    def _feature_adaptation_metrics(
        self, outcome: dict, prior_adaptation: dict | None = None,
    ) -> dict:
        """Record a feature-only deployment without claiming weight updates."""
        if prior_adaptation and prior_adaptation.get("adaptation_event"):
            result = dict(prior_adaptation)
            result.update({
                "adaptation_mode": "drift_conditioned_weight_feature",
                "active_validation_f1": outcome[
                    "feature_reconfiguration_weight_repair_f1"
                ],
                "candidate_validation_f1": outcome[
                    "feature_reconfiguration_f1"
                ],
                "candidate_accepted": True,
            })
            return result
        total = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "adaptation_event": True,
            "training_sample_count": 0,
            "adaptation_mode": "drift_conditioned_feature",
            "selected_modules": [],
            "total_parameters": total,
            "trainable_parameters": 0,
            "parameter_update_percentage": 0.0,
            "updated_parameters": 0,
            "updated_parameter_percentage": 0.0,
            "estimated_parameter_update_bytes": 0,
            "training_time": 0.0,
            "candidate_validation_time": 0.0,
            "active_validation_f1": outcome[
                "feature_reconfiguration_weight_repair_f1"
            ],
            "candidate_validation_f1": outcome["feature_reconfiguration_f1"],
            "candidate_accepted": True,
        }

    def _maybe_contract_capacity(
        self,
        *,
        window_id: int,
        student_features: torch.Tensor,
        labels: torch.Tensor,
        reference_f1: float,
    ) -> dict | None:
        if not (
            self.capacity_enabled
            and self.capacity_config["contraction_enabled"]
        ):
            return None
        magnitudes = self.model.hidden_activation_magnitudes(student_features)
        hidden_names = list(self.model.named_hidden_modules())
        masks = self.model.hidden_neuron_masks()
        initial = (
            self.capacity_config["initial_layer1"],
            self.capacity_config["initial_layer2"],
        )
        eligible: list[list[int]] = [[], []]
        for layer_index, name in enumerate(hidden_names):
            active = masks[name]
            streaks = self.capacity_low_utility_streaks[name]
            utility = magnitudes[name]
            streaks[~active] = 0
            for neuron_index in range(initial[layer_index], active.numel()):
                if not bool(active[neuron_index].item()):
                    continue
                if float(utility[neuron_index].item()) < self.capacity_config[
                    "contraction_utility_threshold"
                ]:
                    streaks[neuron_index] += 1
                else:
                    streaks[neuron_index] = 0
                if int(streaks[neuron_index].item()) >= self.capacity_config[
                    "contraction_patience"
                ]:
                    eligible[layer_index].append(neuron_index)

        old_capacity = self.model.capacity_state()
        chosen: list[list[int]] = [[], []]
        for layer_index, name in enumerate(hidden_names):
            if eligible[layer_index] and old_capacity[layer_index] > initial[layer_index]:
                chosen[layer_index] = [min(
                    eligible[layer_index],
                    key=lambda neuron_index: (
                        float(magnitudes[name][neuron_index].item()), neuron_index,
                    ),
                )]
        if not chosen[0] and not chosen[1]:
            return None

        current_f1 = macro_f1(
            labels, self._predict(self.model, student_features),
        )
        candidate = copy.deepcopy(self.model).eval()
        deactivated_layer1 = candidate.deactivate_neurons(1, chosen[0])
        deactivated_layer2 = candidate.deactivate_neurons(2, chosen[1])
        post_capacity_f1 = macro_f1(
            labels, self._predict(candidate, student_features),
        )
        if post_capacity_f1 + 1e-12 < current_f1:
            for layer_index, name in enumerate(hidden_names):
                for neuron_index in chosen[layer_index]:
                    self.capacity_low_utility_streaks[name][neuron_index] = 0
            return None
        self.model = candidate
        self.model_version += 1
        self._reset_capacity_streaks()
        return make_capacity_event(
            window_id=window_id,
            event_type="contract",
            old_capacity=old_capacity,
            new_capacity=self.model.capacity_state(),
            reference_f1=reference_f1,
            pre_repair_f1=None,
            post_repair_f1=None,
            post_capacity_f1=post_capacity_f1,
            residual_f1_loss=0.0,
            layer1_saturation=0.0,
            layer2_saturation=0.0,
            deactivated_layer1_neurons=deactivated_layer1,
            deactivated_layer2_neurons=deactivated_layer2,
            training_samples=0,
            trainable_parameters=0,
            accepted=True,
            model_version=self.model_version,
        )

    def _empty_labeling_metrics(self) -> dict:
        return {
            "samples_sent_to_teacher": 0,
            "teacher_known_samples": 0,
            "teacher_unknown_samples": 0,
            "teacher_coverage": 0.0,
            "accepted_label_count": 0,
            "labeling_time": 0.0,
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
            if self.policy_mode == "drift_conditioned":
                student_features = apply_feature_mask(
                    stream_features, self.active_feature_mask,
                    self.feature_reference_values,
                )
                static_features = apply_feature_mask(
                    stream_features, self.default_feature_mask,
                    self.feature_reference_values,
                )
            else:
                student_features = stream_features
                static_features = stream_features
            student_labels = self._predict(self.model, student_features)
            static_labels = self._predict(self.static_model, static_features)
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
                "predicted_feature_mask": [],
                "active_feature_names": (
                    list(self.config["dataset"]["feature_names"])
                    if self.policy_mode == "caravan" else []
                ),
                "number_of_active_features": (
                    len(self.config["dataset"]["feature_names"])
                    if self.policy_mode == "caravan" else 0
                ),
                "feature_reconfiguration_event": False,
                "feature_reconfiguration_attempted": False,
                "feature_reconfiguration_accepted": None,
                "feature_reconfiguration_status": (
                    self.drift_event_feature_stage_status
                ),
                "feature_reconfiguration_weight_repair_f1": None,
                "feature_reconfiguration_f1": (
                    self.drift_event_feature_reconfiguration_f1
                ),
                "feature_reconfiguration_final_f1": None,
                "feature_reconfiguration_residual_f1_loss": None,
                "feature_reconfiguration_recovery_insufficient": None,
                "features_enabled": [],
                "features_disabled": [],
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
                "repair_stage_order": list(self.drift_event_stage_order),
                "weight_repair_attempted": self.drift_event_repair_attempted,
                "weight_repair_f1": self.drift_event_weight_repair_f1,
                "weight_repair_residual_f1_loss": None,
                "weight_repair_recovery_insufficient": None,
                "final_repair_stage": (
                    "feature_reconfiguration"
                    if self.drift_event_feature_stage_status == "applied"
                    else "weight_repair"
                    if self.drift_event_repair_attempted else None
                ),
                "recovery_evaluation_label_source": self.capacity_label_source,
            }
            capacity_metrics = {
                "capacity_enabled": self.capacity_enabled,
                "capacity_label_source": self.capacity_label_source,
                "capacity_impact_threshold": self.capacity_impact_threshold,
                "capacity_repair_attempted_for_drift_event": (
                    self.drift_event_repair_attempted
                ),
                "capacity_feature_stage_completed_for_drift_event": (
                    self._feature_stage_resolved()
                    if self.capacity_enabled else False
                ),
                "capacity_pressure": False,
                "capacity_event": None,
                "capacity_reference_f1": self.last_non_drift_f1,
                "capacity_pre_repair_f1": None,
                "capacity_post_repair_f1": None,
                "capacity_reconfigured_f1": None,
                "capacity_validation_post_repair_f1": None,
                "capacity_validation_reconfigured_f1": None,
                "residual_f1_loss": 0.0,
                "layer1_saturation": None,
                "layer2_saturation": None,
                "layer1_high_impact_neurons": 0,
                "layer2_high_impact_neurons": 0,
                "capacity_impacted_neuron_scores": {},
                "capacity_module_impact_scores": {},
                "capacity_training_sample_indices": [],
                "capacity_validation_sample_indices": [],
                "capacity_training_samples": 0,
                "capacity_trainable_parameters": 0,
                "capacity_training_time": 0.0,
                "capacity_validation_time": 0.0,
            }
            capacity_evaluated_this_window = False

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
                reference_f1 = self.last_non_drift_f1
                capacity_metrics["capacity_reference_f1"] = reference_f1
                if self.capacity_enabled and (drift_active or raw_feature_shift):
                    self._reset_capacity_streaks()
                if not drift_active:
                    self._reset_drift_event_state()
                elif not previous_feature_drift:
                    self._reset_drift_event_state()

                if drift_active and not self.drift_event_handled:
                    adaptation_started = time.perf_counter()
                    if not self.drift_event_repair_attempted:
                        self.drift_event_pre_repair_model = copy.deepcopy(
                            self.model
                        ).eval()
                        self.drift_event_pre_repair_feature_mask = list(
                            self.active_feature_mask
                        )
                    prediction_started = time.perf_counter()
                    pending_feature_mask = None
                    if self.capacity_enabled:
                        plan, pending_feature_mask, prediction = (
                            predict_staged_repair(
                                self.adapter_bundle,
                                drift_signature["drift_signature_vector"],
                                self.active_feature_mask,
                                module_threshold=self.adapter_module_threshold,
                                parameter_bits=self.parameter_bits,
                            )
                        )
                        prediction = {
                            **prediction,
                            **prediction["weight_stage_prediction"],
                        }
                    else:
                        plan, prediction = predict_repair(
                            self.adapter_bundle,
                            drift_signature["drift_signature_vector"],
                            module_threshold=self.adapter_module_threshold,
                            parameter_bits=self.parameter_bits,
                        )
                    conditioned_metrics["adapter_prediction_latency"] = (
                        time.perf_counter() - prediction_started
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
                    if confidence_ok:
                        previous_mask = list(self.active_feature_mask)
                        candidate_mask = list(
                            plan.feature_mask or self.default_feature_mask
                        )
                        if self.capacity_enabled and candidate_mask != previous_mask:
                            raise AssertionError(
                                "Weight-first repair must retain the active feature mask"
                            )
                        candidate, repair_metrics = apply_repair_plan(
                            self.model, plan, parameter_bits=self.parameter_bits,
                            anchor_model=self.repair_base_model,
                            feature_mask=candidate_mask,
                        )
                        if all(torch.isfinite(parameter).all() for parameter in candidate.parameters()):
                            repair_satisfied = True
                            self.drift_event_repair_attempted = True
                            conditioned_metrics.update(repair_metrics)
                            mask_changed = candidate_mask != previous_mask
                            weights_changed = repair_metrics["repair_changed_parameters"] > 0
                            self.model = candidate.eval()
                            self.active_feature_mask = candidate_mask
                            student_features = apply_feature_mask(
                                stream_features, self.active_feature_mask,
                                self.feature_reference_values,
                            )
                            if self.capacity_enabled:
                                self.drift_event_pending_feature_mask = (
                                    list(pending_feature_mask)
                                    if pending_feature_mask is not None else None
                                )
                                self.drift_event_feature_stage_status = (
                                    "pending"
                                    if pending_feature_mask is not None
                                    else "not_applicable"
                                )
                                self.drift_event_stage_order.append("weight_repair")
                                conditioned_metrics.update({
                                    "feature_reconfiguration_status": (
                                        self.drift_event_feature_stage_status
                                    ),
                                    "weight_repair_attempted": True,
                                })
                            else:
                                conditioned_metrics.update({
                                    "feature_reconfiguration_event": mask_changed,
                                    "feature_reconfiguration_status": (
                                        "joint_applied" if mask_changed else "not_applicable"
                                    ),
                                    "features_enabled": [
                                        self.config["dataset"]["feature_names"][feature_index]
                                        for feature_index, (before, after) in enumerate(zip(
                                            previous_mask, candidate_mask,
                                        ))
                                        if not before and after
                                    ],
                                    "features_disabled": [
                                        self.config["dataset"]["feature_names"][feature_index]
                                        for feature_index, (before, after) in enumerate(zip(
                                            previous_mask, candidate_mask,
                                        ))
                                        if before and not after
                                    ],
                                })
                            if weights_changed or mask_changed:
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
                    repaired_f1 = macro_f1(
                        window.labels, self._predict(self.model, student_features),
                    )
                    if self.capacity_enabled and repair_satisfied:
                        self.drift_event_weight_repair_f1 = repaired_f1
                    recovery = self._repair_recovery_status(
                        reference_f1, repaired_f1,
                    )
                    conditioned_metrics.update({
                        "weight_repair_f1": (
                            repaired_f1 if repair_satisfied else None
                        ),
                        "weight_repair_residual_f1_loss": (
                            recovery["residual_f1_loss"]
                            if repair_satisfied else None
                        ),
                        "weight_repair_recovery_insufficient": (
                            recovery["recovery_insufficient"]
                            if repair_satisfied else None
                        ),
                    })

                    if (
                        self.capacity_enabled
                        and repair_satisfied
                        and recovery["recovery_insufficient"]
                        and self.drift_event_feature_stage_status == "pending"
                    ):
                        feature_outcome = self._apply_pending_feature_reconfiguration(
                            stream_features=stream_features,
                            labels=window.labels,
                            reference_f1=reference_f1,
                        )
                        student_features = feature_outcome.pop("student_features")
                        conditioned_metrics.update(feature_outcome)
                        repaired_f1 = feature_outcome[
                            "feature_reconfiguration_final_f1"
                        ]
                        recovery = self._repair_recovery_status(
                            reference_f1, repaired_f1,
                        )
                        if feature_outcome["feature_reconfiguration_accepted"]:
                            adaptation = self._feature_adaptation_metrics(
                                feature_outcome, adaptation,
                            )
                    conditioned_metrics["f1_after_predicted_repair"] = repaired_f1
                    conditioned_metrics["final_repair_stage"] = (
                        "feature_reconfiguration"
                        if self.drift_event_feature_stage_status == "applied"
                        else "weight_repair" if repair_satisfied else None
                    )
                    conditioned_metrics["repair_stage_order"] = list(
                        self.drift_event_stage_order
                    )
                    capacity_metrics.update({
                        "capacity_post_repair_f1": (
                            repaired_f1 if repair_satisfied else None
                        ),
                        "residual_f1_loss": (
                            recovery["residual_f1_loss"]
                            if repair_satisfied else 0.0
                        ),
                        "capacity_feature_stage_completed_for_drift_event": (
                            self._feature_stage_resolved()
                        ),
                    })

                    capacity_outcome = None
                    if (
                        self.capacity_enabled
                        and repair_satisfied
                        and recovery["recovery_insufficient"]
                        and self._feature_stage_resolved()
                    ):
                        capacity_outcome = self._attempt_capacity_expansion(
                            window_id=index,
                            stream_features=stream_features,
                            student_features=student_features,
                            labels=window.labels,
                            pre_repair_predictions=student_labels,
                            feature_drift=feature_drift,
                            reference_f1=reference_f1,
                            repair_attempted=repair_satisfied,
                            feature_stage_completed=self._feature_stage_resolved(),
                            training=training,
                        )
                        capacity_evaluated_this_window = True
                        capacity_metrics[
                            "capacity_repair_attempted_for_drift_event"
                        ] = True
                        selected_indices = capacity_outcome["selected_indices"]
                        if capacity_outcome["capacity_event"] is not None:
                            trusted_indices = capacity_outcome["training_indices"]
                            trusted_y = window.labels[trusted_indices]
                        capacity_metrics.update(
                            self._capacity_outcome_metrics(capacity_outcome)
                        )
                        if capacity_outcome["accepted"]:
                            adaptation = self._capacity_adaptation_metrics(
                                capacity_outcome, adaptation,
                            )
                        if capacity_outcome["capacity_event"] is not None:
                            self.drift_event_stage_order.append(
                                "capacity_reconfiguration"
                            )

                    capacity_attempted = bool(
                        capacity_outcome is not None
                        and capacity_outcome["capacity_event"] is not None
                    )
                    self.drift_event_handled = repair_satisfied or capacity_attempted
                    conditioned_metrics["repair_stage_order"] = list(
                        self.drift_event_stage_order
                    )
                    if (
                        adaptation["adaptation_event"]
                        or conditioned_metrics["feature_reconfiguration_attempted"]
                        or capacity_attempted
                    ):
                        conditioned_metrics["adaptation_latency"] = (
                            time.perf_counter() - adaptation_started
                        )
                    if self.capacity_enabled and adaptation["adaptation_event"]:
                        self._reset_capacity_streaks()

                # A weight repair is applied once per persistent-drift episode.
                # If it recovers initially but later deteriorates, the pending
                # feature-only stage is applied before capacity is considered.
                if (
                    drift_active
                    and self.capacity_enabled
                    and self.drift_event_repair_attempted
                    and not capacity_evaluated_this_window
                ):
                    escalation_started = time.perf_counter()
                    repaired_f1 = macro_f1(
                        window.labels, self._predict(self.model, student_features),
                    )
                    recovery = self._repair_recovery_status(
                        reference_f1, repaired_f1,
                    )
                    conditioned_metrics.update({
                        "weight_repair_attempted": True,
                        "weight_repair_f1": repaired_f1,
                        "weight_repair_residual_f1_loss": recovery[
                            "residual_f1_loss"
                        ],
                        "weight_repair_recovery_insufficient": recovery[
                            "recovery_insufficient"
                        ],
                    })
                    if (
                        recovery["recovery_insufficient"]
                        and self.drift_event_feature_stage_status == "pending"
                    ):
                        feature_outcome = self._apply_pending_feature_reconfiguration(
                            stream_features=stream_features,
                            labels=window.labels,
                            reference_f1=reference_f1,
                        )
                        student_features = feature_outcome.pop("student_features")
                        conditioned_metrics.update(feature_outcome)
                        repaired_f1 = feature_outcome[
                            "feature_reconfiguration_final_f1"
                        ]
                        recovery = self._repair_recovery_status(
                            reference_f1, repaired_f1,
                        )
                        if feature_outcome["feature_reconfiguration_accepted"]:
                            adaptation = self._feature_adaptation_metrics(
                                feature_outcome, adaptation,
                            )

                    conditioned_metrics.update({
                        "f1_after_predicted_repair": repaired_f1,
                        "feature_reconfiguration_status": (
                            self.drift_event_feature_stage_status
                        ),
                        "final_repair_stage": (
                            "feature_reconfiguration"
                            if self.drift_event_feature_stage_status == "applied"
                            else "weight_repair"
                        ),
                        "repair_stage_order": list(self.drift_event_stage_order),
                    })
                    capacity_metrics.update({
                        "capacity_post_repair_f1": repaired_f1,
                        "residual_f1_loss": recovery["residual_f1_loss"],
                        "capacity_feature_stage_completed_for_drift_event": (
                            self._feature_stage_resolved()
                        ),
                    })

                    capacity_outcome = None
                    if (
                        recovery["recovery_insufficient"]
                        and self._feature_stage_resolved()
                    ):
                        if (
                            self.drift_event_pre_repair_model is None
                            or self.drift_event_pre_repair_feature_mask is None
                        ):
                            raise AssertionError(
                                "Persistent drift lost its pre-repair state"
                            )
                        pre_repair_features = apply_feature_mask(
                            stream_features,
                            self.drift_event_pre_repair_feature_mask,
                            self.feature_reference_values,
                        )
                        pre_repair_predictions = self._predict(
                            self.drift_event_pre_repair_model,
                            pre_repair_features,
                        )
                        capacity_outcome = self._attempt_capacity_expansion(
                            window_id=index,
                            stream_features=stream_features,
                            student_features=student_features,
                            labels=window.labels,
                            pre_repair_predictions=pre_repair_predictions,
                            feature_drift=feature_drift,
                            reference_f1=reference_f1,
                            repair_attempted=self.drift_event_repair_attempted,
                            feature_stage_completed=self._feature_stage_resolved(),
                            training=training,
                        )
                        capacity_evaluated_this_window = True
                        capacity_metrics[
                            "capacity_repair_attempted_for_drift_event"
                        ] = True
                        capacity_metrics.update(
                            self._capacity_outcome_metrics(capacity_outcome)
                        )
                        selected_indices = capacity_outcome["selected_indices"]
                        if capacity_outcome["capacity_event"] is not None:
                            trusted_indices = capacity_outcome["training_indices"]
                            trusted_y = window.labels[trusted_indices]
                            self.drift_event_stage_order.append(
                                "capacity_reconfiguration"
                            )
                        if capacity_outcome["accepted"]:
                            adaptation = self._capacity_adaptation_metrics(
                                capacity_outcome, adaptation,
                            )
                    conditioned_metrics["repair_stage_order"] = list(
                        self.drift_event_stage_order
                    )
                    if (
                        adaptation["adaptation_event"]
                        or conditioned_metrics["feature_reconfiguration_attempted"]
                        or capacity_outcome is not None
                        and capacity_outcome["capacity_event"] is not None
                    ):
                        conditioned_metrics["adaptation_latency"] = (
                            time.perf_counter() - escalation_started
                        )

                if not drift_active and not raw_feature_shift:
                    contraction_event = self._maybe_contract_capacity(
                        window_id=index,
                        student_features=student_features,
                        labels=window.labels,
                        reference_f1=student_f1,
                    )
                    if contraction_event is not None:
                        capacity_metrics.update({
                            "capacity_event": contraction_event,
                            "capacity_reconfigured_f1": contraction_event[
                                "post_capacity_f1"
                            ],
                        })
                        adaptation = {
                            "adaptation_event": True,
                            "training_sample_count": 0,
                            "adaptation_mode": "capacity_contract",
                            "selected_modules": [],
                            "total_parameters": sum(
                                parameter.numel() for parameter in self.model.parameters()
                            ),
                            "trainable_parameters": 0,
                            "parameter_update_percentage": 0.0,
                            "updated_parameters": 0,
                            "updated_parameter_percentage": 0.0,
                            "estimated_parameter_update_bytes": 0,
                            "training_time": 0.0,
                            "candidate_validation_time": 0.0,
                            "active_validation_f1": student_f1,
                            "candidate_validation_f1": contraction_event[
                                "post_capacity_f1"
                            ],
                            "candidate_accepted": True,
                        }
                        self.last_non_drift_f1 = contraction_event[
                            "post_capacity_f1"
                        ]
                    else:
                        self.last_non_drift_f1 = student_f1

                conditioned_metrics.update({
                    "active_feature_names": [
                        name for name, active in zip(
                            self.config["dataset"]["feature_names"],
                            self.active_feature_mask,
                        )
                        if active
                    ],
                    "number_of_active_features": sum(self.active_feature_mask),
                })

            capacity_metrics.update(self._capacity_metrics())

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
                macro_f1(window.labels, self._predict(self.model, student_features))
                if event else None
            )
            adaptation_labeling_time = (
                labeling_metrics["labeling_time"] if self.policy_mode == "caravan" else 0.0
            ) if event else 0.0
            recovery_attempted = bool(
                event
                or conditioned_metrics["feature_reconfiguration_attempted"]
                or capacity_metrics["capacity_event"] is not None
            )
            total_recovery_time = (
                conditioned_metrics["adaptation_latency"]
                if self.policy_mode == "drift_conditioned" and recovery_attempted else
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
                **capacity_metrics,
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
        capacity_events = [
            row["capacity_event"] for row in rows
            if row["capacity_event"] is not None
        ]
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
            "number_of_capacity_events": len(capacity_events),
            "number_of_capacity_expansions": sum(
                event["event_type"] == "expand" for event in capacity_events
            ),
            "number_of_rejected_capacity_expansions": sum(
                event["event_type"] == "rejected_expand"
                for event in capacity_events
            ),
            "number_of_capacity_contractions": sum(
                event["event_type"] == "contract" for event in capacity_events
            ),
            "capacity_label_source": self.capacity_label_source,
            "capacity_events": capacity_events,
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
