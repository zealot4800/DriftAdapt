from __future__ import annotations

import torch


@torch.inference_mode()
def select_drift_associated_samples(
    features: torch.Tensor,
    drift_scores: dict[str, float | None],
    drifted_feature_indices: list[int],
    range_min: torch.Tensor,
    range_max: torch.Tensor,
    *,
    enabled: bool,
    max_samples: int,
) -> torch.Tensor:
    """Select a deterministic, bounded, stream-ordered drift-associated subset."""
    if not drifted_feature_indices:
        return torch.empty(0, dtype=torch.long, device=features.device)
    if not enabled:
        return torch.arange(features.shape[0], device=features.device)
    if max_samples < 1:
        raise ValueError("sample_selection.max_samples must be positive")

    indices = torch.tensor(drifted_feature_indices, dtype=torch.long, device=features.device)
    span = (range_max[indices] - range_min[indices]).clamp_min(1e-6)
    center = 0.5 * (range_max[indices] + range_min[indices])
    jsd = torch.tensor(
        [list(drift_scores.values())[index] or 0.0 for index in drifted_feature_indices],
        dtype=features.dtype,
        device=features.device,
    )
    scores = (((features[:, indices] - center).abs() / span) * jsd).sum(dim=1)
    count = min(max_samples, max(features.shape[0] - 1, 0))
    selected = torch.argsort(scores, descending=True, stable=True)[:count]
    return torch.sort(selected).values
