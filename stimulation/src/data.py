from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch


@dataclass
class Window:
    features: torch.Tensor
    labels: torch.Tensor
    raw_features: torch.Tensor
    generated_labels: torch.Tensor | None


class StreamingDataset:
    """CARAVAN streaming data with optional per-trace preprocessing segments."""

    def __init__(self, config: dict, base_dir: Path, device: torch.device) -> None:
        self.device = device
        feature_names = [name.strip().lower() for name in config["feature_names"]]
        label_column = config.get("label_column", "label").strip().lower()
        generated_column = config.get("generated_label_column")
        frames: list[pd.DataFrame] = []
        for configured_path in config["files"]:
            path = resolve_path(configured_path, base_dir)
            frame = pd.read_csv(path)
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            frames.append(frame)

        frame = pd.concat(frames, ignore_index=True)
        if "total packet length" in feature_names and "total packet length" not in frame:
            frame["total packet length"] = (
                frame["total length of fwd packets"] + frame["total length of bwd packets"]
            )
        if "number of packets" in feature_names and "number of packets" not in frame:
            frame["number of packets"] = frame["total fwd packets"] + frame["total backward packets"]

        raw = torch.tensor(frame[feature_names].to_numpy(dtype="float32"))
        segments = config.get("preprocessing_segments", [len(current) for current in frames])
        if sum(segments) != len(frame) or any(length <= 0 for length in segments):
            raise ValueError("dataset.preprocessing_segments must be positive and cover every row")
        processed: list[torch.Tensor] = []
        start = 0
        for length in segments:
            features = raw[start:start + length].clone()
            if config.get("standardize", True):
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True)
                features = torch.nan_to_num((features - mean) / std)
            if config.get("normalize", False):
                minimum = features.amin(dim=0, keepdim=True)
                span = features.amax(dim=0, keepdim=True) - minimum
                features = torch.nan_to_num((features - minimum) / span)
            processed.append(features)
            start += length

        self.features = torch.cat(processed).to(device)
        self.raw_features = raw.to(device)
        self.labels = torch.tensor(frame[label_column].to_numpy(), dtype=torch.long, device=device)
        self.generated_labels = (
            torch.tensor(frame[generated_column.strip().lower()].to_numpy(), dtype=torch.long, device=device)
            if generated_column else torch.full_like(self.labels, -1)
        )

    def windows(self, window_size: int) -> Iterator[Window]:
        for start in range(0, len(self.labels), window_size):
            end = min(start + window_size, len(self.labels))
            if end - start < window_size:
                break  # artifact only evaluates complete windows
            yield Window(
                self.features[start:end], self.labels[start:end],
                self.raw_features[start:end], self.generated_labels[start:end],
            )


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured CARAVAN asset does not exist: {path}")
    return path
