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

SIGNATURE_FIELDS = (
    "jsd",
    "normalized_mean_shift",
    "normalized_std_shift",
    "normalized_median_shift",
    "direction",
)


DRIFT_SIGNATURE_FORMAT_VERSION = 1


def _histogram_statistics(
    counts: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate per-feature mean/std/median from fixed-bin histograms."""
    feature_count, bins = counts.shape
    positions = (
        torch.arange(bins, device=counts.device, dtype=torch.float64) + 0.5
    ) / bins
    minimum = minimum.to(torch.float64)
    maximum = maximum.to(torch.float64)
    centers = minimum[:, None] + (maximum - minimum)[:, None] * positions
    weights = counts.to(torch.float64)
    totals = weights.sum(dim=1).clamp_min(1.0)
    means = (weights * centers).sum(dim=1) / totals
    variances = (weights * (centers - means[:, None]).square()).sum(dim=1) / totals
    cumulative = weights.cumsum(dim=1)
    median_bins = (cumulative < (totals[:, None] * 0.5)).sum(dim=1)
    median_bins.clamp_(0, bins - 1)
    medians = centers.gather(1, median_bins[:, None]).squeeze(1)
    if feature_count == 0:
        return means, variances.sqrt(), medians
    return means, variances.clamp_min(0.0).sqrt(), medians


@torch.inference_mode()
def build_drift_signature(
    feature_names: list[str],
    reference_counts: torch.Tensor,
    current_features: torch.Tensor,
    range_min: torch.Tensor,
    range_max: torch.Tensor,
    feature_drift_scores: dict[str, float | None],
    *,
    normalized_shift_clip: float = 8.0,
) -> dict:
    """Build a feature-only, feature-major drift-signature vector.

    Each feature contributes JSD, normalized mean/std/median shifts, and the
    sign of its mean shift. Reference moments are reconstructed from the same
    bounded histograms used by the detector; no label argument is accepted.
    """
    feature_count = len(feature_names)
    if current_features.ndim != 2 or current_features.shape[1] != feature_count:
        raise ValueError("Drift-signature features do not match the configured feature count")
    if reference_counts.ndim != 2 or reference_counts.shape[0] != feature_count:
        raise ValueError("Drift-signature reference histogram has the wrong shape")
    if normalized_shift_clip <= 0:
        raise ValueError("normalized_shift_clip must be positive")

    reference_mean, reference_std, reference_median = _histogram_statistics(
        reference_counts, range_min, range_max,
    )
    current_counts = fixed_histogram_counts(
        current_features, range_min, range_max, reference_counts.shape[1],
    )
    current_mean, current_std, current_median = _histogram_statistics(
        current_counts, range_min, range_max,
    )
    span = (range_max - range_min).to(torch.float64).abs()
    normalizer = torch.maximum(reference_std, span / reference_counts.shape[1])
    normalizer = normalizer.clamp_min(torch.finfo(torch.float64).eps)

    mean_shift = ((current_mean - reference_mean) / normalizer).clamp(
        -normalized_shift_clip, normalized_shift_clip,
    )
    std_shift = ((current_std - reference_std) / normalizer).clamp(
        -normalized_shift_clip, normalized_shift_clip,
    )
    median_shift = ((current_median - reference_median) / normalizer).clamp(
        -normalized_shift_clip, normalized_shift_clip,
    )
    direction = torch.sign(mean_shift)

    per_feature: dict[str, dict[str, float]] = {}
    vector: list[float] = []
    for index, name in enumerate(feature_names):
        values = {
            "jsd": float(feature_drift_scores.get(name) or 0.0),
            "normalized_mean_shift": float(mean_shift[index].item()),
            "normalized_std_shift": float(std_shift[index].item()),
            "normalized_median_shift": float(median_shift[index].item()),
            "direction": float(direction[index].item()),
        }
        per_feature[name] = values
        vector.extend(values[field] for field in SIGNATURE_FIELDS)

    return {
        "drift_signature_vector": vector,
        "drift_signature_per_feature": per_feature,
        "drift_signature_fields": list(SIGNATURE_FIELDS),
        "drift_signature_size": len(vector),
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


# Existing FPGA runtime import.
_fixed_histogram_counts = fixed_histogram_counts
