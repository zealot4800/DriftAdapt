from __future__ import annotations


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
