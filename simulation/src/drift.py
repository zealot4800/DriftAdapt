from __future__ import annotations

import torch


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

class RetrainingTrigger:
    """Detect temporal degradation in the labeler-based accuracy proxy."""

    def __init__(self, threshold: float, *, drop_threshold: float = 0.15,
                 consecutive_windows: int = 2) -> None:
        if not 0.0 <= threshold <= 1.0 or not 0.0 <= drop_threshold <= 1.0:
            raise ValueError("drift thresholds must be between 0 and 1")
        if consecutive_windows < 1:
            raise ValueError("consecutive_windows must be at least one")
        self.threshold = threshold
        self.drop_threshold = drop_threshold
        self.consecutive_windows = consecutive_windows
        self.previous_proxy: float | None = None
        self.low_proxy_windows = 0

    def update(self, model_proxy_f1: float) -> bool:
        abrupt_drop = (
            self.previous_proxy is not None
            and self.previous_proxy - model_proxy_f1 >= self.drop_threshold
        )
        self.low_proxy_windows = (
            self.low_proxy_windows + 1 if model_proxy_f1 < self.threshold else 0
        )
        self.previous_proxy = model_proxy_f1
        return abrupt_drop or self.low_proxy_windows >= self.consecutive_windows
