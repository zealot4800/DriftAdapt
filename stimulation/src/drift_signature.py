from __future__ import annotations

import torch

from .drift import fixed_histogram_counts


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
