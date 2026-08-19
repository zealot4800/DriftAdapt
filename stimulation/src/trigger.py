from __future__ import annotations


class RetrainingTrigger:
    """Retrain only when DRIFTADAPT's labeler-based accuracy proxy indicates drift."""

    def __init__(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("drift threshold must be between 0 and 1")
        self.threshold = threshold

    def update(self, *, window_index: int, model_proxy_f1: float, static_proxy_f1: float) -> bool:
        del window_index, static_proxy_f1
        # DRIFTADAPT's accuracy proxy estimates current student quality using the
        # labeling agent instead of unavailable online ground truth.
        return model_proxy_f1 < self.threshold
