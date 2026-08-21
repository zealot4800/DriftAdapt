#!/usr/bin/env python3
"""Build reproducible U55C model and traffic-source memory images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


PARAMETER_KEYS = (
    "linear_relu_stack.0.weight",
    "linear_relu_stack.0.bias",
    "linear_relu_stack.2.weight",
    "linear_relu_stack.2.bias",
    "linear_relu_stack.4.weight",
    "linear_relu_stack.4.bias",
)
FEATURE_COUNT = 16
PARAMETER_COUNT = 182


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_fixed(values: np.ndarray, fraction_bits: int, bits: int) -> np.ndarray:
    scaled = np.rint(values.astype(np.float64) * (1 << fraction_bits)).astype(np.int64)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if np.any(scaled < minimum) or np.any(scaled > maximum):
        bad = scaled[(scaled < minimum) | (scaled > maximum)][0]
        raise ValueError(f"fixed-point value {bad} does not fit signed {bits}-bit storage")
    return scaled


def saturate_s32(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def fixed_prediction(features: np.ndarray, parameters: np.ndarray) -> int:
    """Mirror driftadapt_dnn_axis.sv exactly with Python integers."""
    layer1_weights = parameters[0:128].reshape(8, 16)
    layer1_bias = parameters[128:136]
    layer2_weights = parameters[136:168].reshape(4, 8)
    layer2_bias = parameters[168:172]
    layer3_weights = parameters[172:180].reshape(2, 4)
    layer3_bias = parameters[180:182]

    activations = [int(value) << 8 for value in features]
    next_activations: list[int] = []
    for neuron in range(8):
        accumulator = int(layer1_bias[neuron])
        for feature in range(16):
            accumulator += (int(layer1_weights[neuron, feature]) * activations[feature]) >> 16
        next_activations.append(max(0, saturate_s32(accumulator)))

    activations = next_activations
    next_activations = []
    for neuron in range(4):
        accumulator = int(layer2_bias[neuron])
        for feature in range(8):
            accumulator += (int(layer2_weights[neuron, feature]) * activations[feature]) >> 16
        next_activations.append(max(0, saturate_s32(accumulator)))

    logits: list[int] = []
    for neuron in range(2):
        accumulator = int(layer3_bias[neuron])
        for feature in range(4):
            accumulator += (int(layer3_weights[neuron, feature]) * next_activations[feature]) >> 16
        logits.append(accumulator)
    return int(logits[1] > logits[0])


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    missing = [key for key in PARAMETER_KEYS if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing parameters: {', '.join(missing)}")
    return state


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "stimulation/checkpoint/cic-ids2017-in-network-dnn.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "stimulation/configs/cic-ids2017.yaml",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repo_root / "testbed/hardware/u55c/assets",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint is missing: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"stimulation configuration is missing: {config_path}")

    stimulation_root = repo_root / "stimulation"
    sys.path.insert(0, str(stimulation_root))
    from src.data import StreamingDataset

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = StreamingDataset(config["dataset"], config_path.parent, torch.device("cpu"))
    if dataset.features.ndim != 2 or dataset.features.shape[1] != FEATURE_COUNT:
        parser.error(f"dataset must contain exactly {FEATURE_COUNT} features")
    rows = dataset.features.cpu().numpy().astype(np.float64)
    labels = dataset.labels.cpu().numpy().astype(np.int64)
    if not np.all((labels == 0) | (labels == 1)):
        parser.error("dataset labels must be binary")
    feature_fixed = rounded_fixed(rows, fraction_bits=8, bits=16)

    state = load_checkpoint(checkpoint)
    parameter_float = np.concatenate(
        [state[key].detach().cpu().numpy().reshape(-1) for key in PARAMETER_KEYS]
    )
    if len(parameter_float) != PARAMETER_COUNT:
        parser.error(f"expected {PARAMETER_COUNT} model parameters, found {len(parameter_float)}")
    parameter_fixed = rounded_fixed(parameter_float, fraction_bits=16, bits=32)

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    weight_path = output / "driftadapt_weights.mem"
    sample_path = output / "driftadapt_samples.mem"
    manifest_path = output / "manifest.json"

    with weight_path.open("w", encoding="ascii", newline="\n") as stream:
        for value in parameter_fixed:
            stream.write(f"{int(value) & 0xffffffff:08x}\n")

    with sample_path.open("w", encoding="ascii", newline="\n") as stream:
        for features in feature_fixed:
            packed = 0
            for index, value in enumerate(features):
                packed |= (int(value) & 0xffff) << (16 * index)
            stream.write(f"{packed:064x}\n")

    predictions = np.fromiter(
        (fixed_prediction(features, parameter_fixed) for features in feature_fixed),
        dtype=np.int64,
        count=len(labels),
    )
    true_positive = int(np.count_nonzero((predictions == 1) & (labels == 1)))
    true_negative = int(np.count_nonzero((predictions == 0) & (labels == 0)))
    false_positive = int(np.count_nonzero((predictions == 1) & (labels == 0)))
    false_negative = int(np.count_nonzero((predictions == 0) & (labels == 1)))
    manifest = {
        "format_version": 2,
        "sample_count": int(len(labels)),
        "feature_count": FEATURE_COUNT,
        "parameter_count": PARAMETER_COUNT,
        "feature_format": "signed Q8.8",
        "sample_record": "sixteen features; no label",
        "parameter_format": "signed Q16.16",
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "weights_sha256": sha256(weight_path),
        "samples_sha256": sha256(sample_path),
        "expected": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "correct": true_positive + true_negative,
            "accuracy": (true_positive + true_negative) / len(labels),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(labels)} samples to {sample_path}")
    print(f"Wrote {len(parameter_fixed)} parameters to {weight_path}")
    print(json.dumps(manifest["expected"], indent=2))


if __name__ == "__main__":
    main()
