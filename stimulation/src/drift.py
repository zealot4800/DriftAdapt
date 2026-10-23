from __future__ import annotations

import math

import torch


COUNTER_BITS = 32
COUNTER_MAX = (1 << COUNTER_BITS) - 1


def fixed_histogram_counts(
    features: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    """Count all feature histograms with one integer bincount operation."""
    span = maximum - minimum
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    indices = torch.floor((features - minimum) * bins / safe_span).long()
    indices.clamp_(0, bins - 1)
    offsets = torch.arange(features.shape[1], device=features.device) * bins
    return torch.bincount(
        (indices + offsets).reshape(-1), minlength=features.shape[1] * bins,
    ).reshape(features.shape[1], bins)


def jensen_shannon_divergence(
    reference_counts: torch.Tensor,
    current_counts: torch.Tensor,
) -> float:
    """Return base-2 JSD for two histograms; isolated for hardware replacement."""
    reference = reference_counts.to(torch.float64)
    current = current_counts.to(torch.float64)
    reference /= reference.sum().clamp_min(1.0)
    current /= current.sum().clamp_min(1.0)
    midpoint = 0.5 * (reference + current)

    def divergence(distribution: torch.Tensor) -> torch.Tensor:
        valid = distribution > 0
        return torch.sum(distribution[valid] * torch.log2(distribution[valid] / midpoint[valid]))

    return float((0.5 * divergence(reference) + 0.5 * divergence(current)).item())


class FeatureHistogramDriftDetector:
    """Fixed-state, label-free per-feature histogram drift detector."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        bins: int,
        threshold: float,
        consecutive_windows: int,
        reference_windows: int,
        device: torch.device,
    ) -> None:
        if bins < 2:
            raise ValueError("drift.bins must be at least 2")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("drift.threshold must be between 0 and 1")
        if consecutive_windows < 1 or reference_windows < 1:
            raise ValueError("drift consecutive_windows and reference_windows must be positive")

        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.bins = bins
        self.threshold = threshold
        self.consecutive_windows = consecutive_windows
        self.reference_windows = reference_windows
        self.windows_seen = 0
        self.range_min = torch.full((self.num_features,), math.inf, device=device)
        self.range_max = torch.full((self.num_features,), -math.inf, device=device)
        self.bin_edges = torch.zeros((self.num_features, bins + 1), device=device)
        # int64 keeps PyTorch operations portable; every write saturates to the
        # unsigned 32-bit range intended for the hardware counter implementation.
        self.reference_counts = torch.zeros(
            (self.num_features, bins), dtype=torch.int64, device=device,
        )
        self.current_counts = torch.zeros_like(self.reference_counts)
        self.consecutive_counts = torch.zeros(
            self.num_features, dtype=torch.int64, device=device,
        )

    @property
    def histogram_counter_count(self) -> int:
        return 2 * self.num_features * self.bins

    @property
    def state_bytes(self) -> int:
        counter_bytes = self.histogram_counter_count * (COUNTER_BITS // 8)
        edge_bytes = self.num_features * (self.bins + 1) * 4
        streak_bytes = self.num_features * 4
        return counter_bytes + edge_bytes + streak_bytes

    def _histogram(
        self,
        features: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
    ) -> torch.Tensor:
        return fixed_histogram_counts(features, minimum, maximum, self.bins).clamp_max(COUNTER_MAX)

    def _rebin_reference(
        self,
        old_minimum: torch.Tensor,
        old_maximum: torch.Tensor,
        new_minimum: torch.Tensor,
        new_maximum: torch.Tensor,
    ) -> None:
        positions = torch.arange(
            self.bins, device=self.reference_counts.device, dtype=torch.float32,
        ) + 0.5
        rebinned = torch.zeros_like(self.reference_counts)
        for feature_index in range(self.num_features):
            old_span = old_maximum[feature_index] - old_minimum[feature_index]
            centers = old_minimum[feature_index] + positions * old_span / self.bins
            new_span = new_maximum[feature_index] - new_minimum[feature_index]
            if new_span > 0:
                indices = torch.floor(
                    (centers - new_minimum[feature_index]) * self.bins / new_span
                ).long()
            else:
                indices = torch.zeros(self.bins, dtype=torch.long, device=centers.device)
            indices.clamp_(0, self.bins - 1)
            rebinned[feature_index].scatter_add_(
                0, indices, self.reference_counts[feature_index],
            )
        self.reference_counts = rebinned.clamp_max(COUNTER_MAX)

    def _freeze_edges(self) -> None:
        positions = torch.linspace(
            0.0, 1.0, self.bins + 1, device=self.range_min.device,
        )
        self.bin_edges = self.range_min[:, None] + (
            self.range_max - self.range_min
        )[:, None] * positions

    def update(self, features: torch.Tensor) -> dict:
        """Consume one feature-only window and return fixed-schema drift output."""
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError("Drift detector input does not match configured feature count")
        self.windows_seen += 1

        if self.windows_seen <= self.reference_windows:
            window_minimum = features.amin(dim=0)
            window_maximum = features.amax(dim=0)
            old_minimum = self.range_min.clone()
            old_maximum = self.range_max.clone()
            new_minimum = torch.minimum(self.range_min, window_minimum)
            new_maximum = torch.maximum(self.range_max, window_maximum)
            if self.windows_seen > 1:
                self._rebin_reference(old_minimum, old_maximum, new_minimum, new_maximum)
            self.range_min = new_minimum
            self.range_max = new_maximum
            self.current_counts = self._histogram(features, self.range_min, self.range_max)
            self.reference_counts = (
                self.reference_counts + self.current_counts
            ).clamp_max(COUNTER_MAX)
            if self.windows_seen == self.reference_windows:
                self._freeze_edges()
            return self._result([None] * self.num_features, [])

        self.current_counts = self._histogram(features, self.range_min, self.range_max)
        scores = [
            jensen_shannon_divergence(self.reference_counts[index], self.current_counts[index])
            for index in range(self.num_features)
        ]
        for index, score in enumerate(scores):
            if score > self.threshold:
                self.consecutive_counts[index] = min(
                    int(self.consecutive_counts[index]) + 1,
                    self.consecutive_windows,
                )
            else:
                self.consecutive_counts[index] = 0
        drifted = torch.nonzero(
            self.consecutive_counts >= self.consecutive_windows, as_tuple=False,
        ).squeeze(1).tolist()
        return self._result(scores, drifted)

    def _result(self, scores: list[float | None], drifted: list[int]) -> dict:
        return {
            "feature_drift_detected": bool(drifted),
            "feature_drift_reference_ready": self.windows_seen >= self.reference_windows,
            "feature_drift_scores": dict(zip(self.feature_names, scores)),
            "drifted_feature_indices": drifted,
            "drifted_feature_names": [self.feature_names[index] for index in drifted],
            "number_of_drifted_features": len(drifted),
            "histogram_counter_count": self.histogram_counter_count,
            "histogram_counter_bits": COUNTER_BITS,
            "histogram_state_bytes": self.state_bytes,
        }


class TrafficDriftInjector:
    """Simulation-only feature drift injection and localization validation."""

    def __init__(self, config: dict, feature_names: list[str]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.kind = str(config.get("type", "abrupt")).lower()
        self.start_window = int(config.get("start_window", 1))
        configured_end = config.get("end_window")
        self.end_window = int(configured_end) if configured_end is not None else None
        self.magnitude = float(config.get("magnitude", 0.0))
        self.direction = int(config.get("direction", 1))
        self.feature_names = feature_names
        if self.kind not in {"abrupt", "gradual", "recurrent"}:
            raise ValueError("drift_injection.type must be abrupt, gradual, or recurrent")
        if self.start_window < 1 or self.magnitude < 0:
            raise ValueError("drift injection start_window must be positive and magnitude nonnegative")
        if self.direction not in {-1, 1}:
            raise ValueError("drift_injection.direction must be -1 or 1")
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
            return active, 1 if active else None, self.start_window, (
                self.direction * self.magnitude if active else 0.0
            )
        if self.kind == "gradual":
            duration = self.end_window - self.start_window + 1
            progress = min(1.0, (window_index - self.start_window + 1) / duration)
            return True, 1, self.start_window, self.direction * self.magnitude * progress

        duration = self.end_window - self.start_window + 1
        period_index = (window_index - self.start_window) // duration
        active = period_index % 2 == 0
        event_id = period_index // 2 + 1
        event_start = self.start_window + (event_id - 1) * 2 * duration
        return active, event_id if active else None, event_start, (
            self.direction * self.magnitude if active else 0.0
        )

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
            "true_drift_direction": self.direction if self.enabled else 0,
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
                "direction": self.direction,
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
