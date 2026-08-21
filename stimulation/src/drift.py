from __future__ import annotations

import math

import torch


COUNTER_BITS = 32
COUNTER_MAX = (1 << COUNTER_BITS) - 1


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
        span = maximum - minimum
        safe_span = torch.where(span > 0, span, torch.ones_like(span))
        indices = torch.floor((features - minimum) * self.bins / safe_span).long()
        indices = indices.clamp(0, self.bins - 1)
        counts = torch.zeros_like(self.reference_counts)
        for feature_index in range(self.num_features):
            counts[feature_index] = torch.bincount(
                indices[:, feature_index], minlength=self.bins,
            ).clamp_max(COUNTER_MAX)
        return counts

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
