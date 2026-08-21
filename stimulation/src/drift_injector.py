from __future__ import annotations

import torch


class TrafficDriftInjector:
    """Simulation-only feature drift injection and localization validation."""

    def __init__(self, config: dict, feature_names: list[str]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.kind = str(config.get("type", "abrupt")).lower()
        self.start_window = int(config.get("start_window", 1))
        configured_end = config.get("end_window")
        self.end_window = int(configured_end) if configured_end is not None else None
        self.magnitude = float(config.get("magnitude", 0.0))
        self.feature_names = feature_names
        if self.kind not in {"abrupt", "gradual", "recurrent"}:
            raise ValueError("drift_injection.type must be abrupt, gradual, or recurrent")
        if self.start_window < 1 or self.magnitude < 0:
            raise ValueError("drift injection start_window must be positive and magnitude nonnegative")
        if self.kind in {"gradual", "recurrent"} and self.end_window is None:
            raise ValueError(f"drift_injection.end_window is required for {self.kind} drift")
        if self.end_window is not None and self.end_window < self.start_window:
            raise ValueError("drift_injection.end_window must not precede start_window")

        indices = [int(index) for index in config.get("feature_indices", [])]
        name_lookup = {name.strip().lower(): index for index, name in enumerate(feature_names)}
        for name in config.get("feature_names", []):
            try:
                indices.append(name_lookup[str(name).strip().lower()])
            except KeyError as exc:
                raise ValueError(f"Unknown injected-drift feature name: {name}") from exc
        if any(index < 0 or index >= len(feature_names) for index in indices):
            raise ValueError("drift_injection.feature_indices contains an invalid feature")
        self.feature_indices = sorted(set(indices))
        if self.enabled and not self.feature_indices:
            raise ValueError("Enabled drift injection requires affected feature indices or names")

        self.detected_windows: dict[int, int] = {}
        self.events: dict[int, dict] = {}

    def _event(self, window_index: int) -> tuple[bool, int | None, int, float]:
        if not self.enabled or window_index < self.start_window:
            return False, None, self.start_window, 0.0
        if self.kind == "abrupt":
            active = self.end_window is None or window_index <= self.end_window
            return active, 1 if active else None, self.start_window, self.magnitude if active else 0.0
        if self.kind == "gradual":
            duration = self.end_window - self.start_window + 1
            progress = min(1.0, (window_index - self.start_window + 1) / duration)
            return True, 1, self.start_window, self.magnitude * progress

        duration = self.end_window - self.start_window + 1
        period_index = (window_index - self.start_window) // duration
        active = period_index % 2 == 0
        event_id = period_index // 2 + 1
        event_start = self.start_window + (event_id - 1) * 2 * duration
        return active, event_id if active else None, event_start, self.magnitude if active else 0.0

    @torch.inference_mode()
    def apply(self, features: torch.Tensor, window_index: int) -> tuple[torch.Tensor, dict]:
        active, event_id, event_start, applied_magnitude = self._event(window_index)
        injected = features.clone()
        if active and applied_magnitude != 0.0:
            injected[:, self.feature_indices] += applied_magnitude
        return injected, {
            "drift_injection_enabled": self.enabled,
            "injected_drift_active": active,
            "injected_drift_event_id": event_id,
            "true_drift_start_window": event_start,
            "true_drifted_feature_indices": self.feature_indices,
            "true_drifted_feature_names": [self.feature_names[index] for index in self.feature_indices],
            "injected_drift_type": self.kind if self.enabled else None,
            "true_drift_magnitude": self.magnitude if self.enabled else 0.0,
            "applied_drift_magnitude": applied_magnitude,
        }

    def validate(
        self,
        window_index: int,
        metadata: dict,
        detected_indices: list[int],
        selected_modules: list[str],
    ) -> dict:
        predicted = set(detected_indices)
        truth = set(self.feature_indices) if metadata["injected_drift_active"] else set()
        true_positive = predicted & truth
        false_positive = sorted(predicted - truth)
        missed = sorted(truth - predicted)
        precision = len(true_positive) / len(predicted) if predicted else 0.0
        recall = len(true_positive) / len(truth) if truth else None
        event_id = metadata["injected_drift_event_id"]

        detection_delay = None
        if event_id is not None:
            event = self.events.setdefault(event_id, {
                "event_id": event_id,
                "true_drift_start_window": metadata["true_drift_start_window"],
                "true_drifted_feature_indices": self.feature_indices,
                "true_drifted_feature_names": metadata["true_drifted_feature_names"],
                "drift_type": self.kind,
                "magnitude": self.magnitude,
                "detection_delay": None,
                "feature_localization_precision": 0.0,
                "feature_localization_recall": 0.0,
                "false_positive_features": [],
                "missed_features": metadata["true_drifted_feature_names"],
                "selected_neural_modules": [],
            })
            event["selected_neural_modules"] = sorted(set(
                event["selected_neural_modules"] + selected_modules
            ))
            if true_positive and event_id not in self.detected_windows:
                self.detected_windows[event_id] = window_index
                event["detection_delay"] = window_index - metadata["true_drift_start_window"]
                event["feature_localization_precision"] = precision
                event["feature_localization_recall"] = recall
                event["false_positive_features"] = [
                    self.feature_names[index] for index in false_positive
                ]
                event["missed_features"] = [self.feature_names[index] for index in missed]
            if event_id in self.detected_windows:
                detection_delay = self.detected_windows[event_id] - metadata["true_drift_start_window"]

        return {
            "injected_drift_detection_delay": detection_delay,
            "feature_localization_precision": precision if truth else None,
            "feature_localization_recall": recall,
            "false_positive_features": [self.feature_names[index] for index in false_positive],
            "missed_features": [self.feature_names[index] for index in missed],
            "injected_drift_selected_modules": selected_modules if truth else [],
        }

    def event_summaries(self) -> list[dict]:
        return [self.events[event_id] for event_id in sorted(self.events)]
